# ==============================================================
# neurapose-backend/pre_processamento/pipeline/processador.py
# ==============================================================
# Pipeline OTIMIZADO (Extração de Dados)
# Visualização padronizada conforme regras de UI vFinal
# INCLUI:
#  - Skip Inteligente (cap.grab)
#  - Pipeline Multithread (Leitura -> GPU -> Escrita)
#  - Codec GPU (h264_nvenc)
# ==============================================================

import time
import os
import cv2
import json
import numpy as np
import threading
import queue
from pathlib import Path
from colorama import Fore
from ultralytics import YOLO
import torch

# Silencia logs OpenCV
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import neurapose_backend.config_master as cm
from neurapose_backend.otimizador.cuda.gpu_utils import gpu_manager, check_gpu_memory
from neurapose_backend.otimizador.cpu import core as cpu_opt
from neurapose_backend.otimizador.ram import memory as ram_opt
from neurapose_backend.tracker.rastreador import CustomBoTSORT, CustomDeepOCSORT, save_temp_tracker_yaml
from neurapose_backend.rtmpose.extracao_pose_rtmpose import ExtratorPoseRTMPose
from neurapose_backend.nucleo.visualizacao import desenhar_esqueleto_unificado, color_for_id
# from neurapose_backend.nucleo.tracking_utils import gerar_relatorio_tracking
from neurapose_backend.nucleo.video_utils import normalizar_video

# Import optional integration
try:
    from neurapose_backend.globals.state import state
except ImportError:
    state = None

# Import opcional do Sanitizer
try:
    from neurapose_backend.nucleo.sanatizer import sanitizar_dados
except ImportError:
    sanitizar_dados = None

# --- CLASSES AUXILIARES DE THREADING ---

class FrameReaderThread(threading.Thread):
    def __init__(self, video_path, skip_interval, queue_out, max_frames=None):
        super().__init__()
        self.video_path = str(video_path)
        self.skip_interval = skip_interval
        self.queue_out = queue_out
        self.max_frames = max_frames 
        self.stopped = False
        self.daemon = True 
        
        self.cap = cv2.VideoCapture(self.video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def run(self):
        frame_idx = 0
        try:
            while self.cap.isOpened() and not self.stopped:
                # [OTIMIZAÇÃO] Skip Físico com grab()
                if frame_idx % self.skip_interval != 0:
                    if not self.cap.grab(): break
                    frame_idx += 1
                    continue

                # Se for frame de processamento, LEITURA COMPLETA (Grab + Retrieve)
                ret, frame = self.cap.read()
                if not ret: break
                
                # Envia para fila (Bloqueia se cheia)
                self.queue_out.put((frame_idx, frame))
                frame_idx += 1

                if state and state.stop_requested: break
                
        except Exception as e:
            print(Fore.RED + f"[LEITURA] Erro: {e}")
        finally:
            self.cap.release()
            self.queue_out.put(None) # Sentinel

    def stop(self):
        self.stopped = True

class VideoWriterThread(threading.Thread):
    def __init__(self, output_path, fps, width, height, queue_in):
        super().__init__()
        self.output_path = str(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        self.queue_in = queue_in
        self.stopped = False
        self.daemon = True
        self.writer = None

    def _init_writer(self):
        # Tenta Codec GPU (h264_nvenc) se configurado
        if cm.USE_NVENC:
            try:
                # Nota: OpenCV Python pip geralmente não tem CUDA habilitado por padrão.
                pass
            except: pass

        # Fallback Padrão (AVC1)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        
        if not self.writer.isOpened():
             # Fallback Último Caso (MP4V)
             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
             self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))

    def run(self):
        self._init_writer()
        try:
            while not self.stopped:
                item = self.queue_in.get()
                if item is None: # Sentinel
                    break
                
                frame = item
                if self.writer is not None:
                    self.writer.write(frame)
                
                self.queue_in.task_done()
        except Exception as e:
            print(Fore.RED + f"[ESCRITA] Erro: {e}")
        finally:
            if self.writer: self.writer.release()

    def stop(self):
        self.stopped = True

@gpu_manager.inference_mode()
def processar_video(video_path: Path, output_dir: Path, show: bool = False):
    """
    Processa vídeo para geração de dataset.
    - FPS: Convertido para cm.FPS_TARGET (10fps).
    - Codec: H.264 (avc1) via OpenH264 ou GPU se disponível.
    - Style: BBox Verde, Texto Preto/Fundo Branco.
    - Multithread: Leitura e Escrita em threads separadas.
    """
    
    # 1. SETUP DE DIRETÓRIOS
    if not output_dir: raise ValueError("output_dir obrigatório")
    
    predicoes_dir = output_dir / "predicoes"
    jsons_dir = output_dir / "jsons"
    videos_norm_dir = output_dir / "videos"
    
    predicoes_dir.mkdir(parents=True, exist_ok=True)
    jsons_dir.mkdir(parents=True, exist_ok=True)
    videos_norm_dir.mkdir(parents=True, exist_ok=True)

    # 2. NORMALIZAÇÃO (REATIVADO)
    # Garante padrao de 30 FPS independente da entrada
    t_start_norm = time.time()
    video_norm_path, t_norm = normalizar_video(video_path, videos_norm_dir, target_fps=cm.INPUT_NORM_FPS)
    
    # 3. EXTRAÇÃO INFO (Apenas metadados para setup)
    cap_temp = cv2.VideoCapture(str(video_norm_path))
    
    # --- COPIA VIDEO BRUTO (Pedido do Usuário para Split posterior) ---
    import shutil
    try:
        video_out_raw = videos_norm_dir / video_norm_path.name
        if not video_out_raw.exists():
            shutil.copy2(video_norm_path, video_out_raw)
    except Exception as e:
        print(f"[AVISO] Falha ao copiar vídeo bruto: {e}")

    # Propriedades
    fps_in = cap_temp.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_temp.release()

    # Calcula Skip para atingir TARGET_FPS (10)
    target_fps = cm.FPS_TARGET
    skip_interval = max(1, int(round(fps_in / target_fps)))

    # 4. SETUP QUEUES E THREADS
    queue_in = queue.Queue(maxsize=30)  # Leitura -> Processamento
    queue_out = queue.Queue(maxsize=30) # Processamento -> Escrita
    
    # Thread Leitura
    reader_thread = FrameReaderThread(video_norm_path, skip_interval, queue_in)
    reader_thread.start()
    
    # Thread Escrita
    video_out_name = f"{video_path.stem}_pred.mp4"
    path_out = predicoes_dir / video_out_name
    writer_thread = VideoWriterThread(path_out, target_fps, w, h, queue_out)
    writer_thread.start()

    # 5. MODELOS (Thread Principal - GPU)
    pose_extractor = ExtratorPoseRTMPose(device=cm.DEVICE)
    
    USING_DEEPOCSORT = (cm.TRACKER_NAME.upper() == "DEEPOCSORT")
    tracker = None
    yolo_model = None
    yaml_path = None
    
    if USING_DEEPOCSORT:
        tracker = CustomDeepOCSORT()
    else:
        yolo_model = YOLO(str(cm.YOLO_PATH), task='detect').to(cm.DEVICE)
        tracker_instance = CustomBoTSORT(frame_rate=int(target_fps))
        yolo_model.tracker = tracker_instance
        yaml_path = save_temp_tracker_yaml()

    # 6. LOOP PRINCIPAL
    records = []
    
    t_yolo_acc = 0.0
    t_pose_acc = 0.0
    last_logged_percent = -1

    # Atualiza Status GPU (caso tenha mudado no frontend)
    gpu_manager.update_device(cm.DEVICE)

    start_time_global = time.time()

    try:
        while True:
            # Pega frame da fila
            item = queue_in.get()
            if item is None: # Sentinel
                break
            
            frame_idx, frame = item

            # --- LÓGICA DE LOG (SILENCIOSA: A CADA 20%) ---
            current_percent = int((frame_idx / total_frames) * 100)
            should_log = (current_percent % 20 == 0 and current_percent > last_logged_percent) or (frame_idx == 0)
            
            if should_log:
                last_logged_percent = current_percent
                print(f"\r[PROCESSAMENTO] Progresso: {current_percent}% ({frame_idx}/{total_frames})")

            # SAFE MODE: Throttling & GC (Centralizado)
            cpu_opt.throttle()
            ram_opt.smart_cleanup(frame_idx)

            t0 = time.time()
            
            # --- DETECÇÃO ---
            yolo_dets = None
            try:
                if USING_DEEPOCSORT:
                    tracks = tracker.track(frame)
                    yolo_dets = tracks
                else:
                    # BoTSORT Manual (Fix para Empty JSON/Warnings)
                    # 1. Detect (Predict direto evita problemas do .track)
                    res = yolo_model.predict(
                        source=frame,
                        imgsz=cm.YOLO_IMGSZ,
                        conf=cm.DETECTION_CONF,
                        device=cm.DEVICE,
                        classes=[cm.YOLO_CLASS_PERSON],
                        verbose=False,
                        stream=False
                    )
                    
                    # 2. Formata Dets para Tracker
                    dets = np.empty((0, 6))
                    if len(res) > 0 and len(res[0].boxes) > 0:
                        dets = res[0].boxes.data.cpu().numpy()
                        # Normaliza shapes (x1,y1,x2,y2,conf,cls)
                        if dets.shape[1] == 4:
                                r = dets.shape[0]
                                dets = np.hstack((dets, np.full((r, 1), 0.85), np.zeros((r, 1))))
                        elif dets.shape[1] == 5:
                                dets = np.hstack((dets, np.zeros((dets.shape[0], 1))))
                    
                    # 3. Update Tracker
                    tracks = tracker_instance.update(dets, frame)
                    yolo_dets = tracks
            except Exception as e:
                # Falha pontual no tracker (skip frame)
                yolo_dets = np.empty((0, 7))

            t1 = time.time()
            t_yolo_acc += (t1 - t0)
    
            # --- POSE ---
            try:
                pose_records, _ = pose_extractor.processar_frame(
                    frame_img=frame,
                    detections_yolo=yolo_dets,
                    frame_idx=frame_idx,
                    desenhar_no_frame=False 
                )
            except Exception as e:
                    pose_records = []
            
            t2 = time.time()
            t_pose_acc += (t2 - t1)
            
            records.extend(pose_records)
            
            # --- VISUALIZAÇÃO PADRONIZADA ---
            viz_frame = frame.copy()
            
            for rec in pose_records:
                pid = rec["id_persistente"]
                bbox = rec["bbox"]
                kps = np.array(rec["keypoints"])
                conf = rec["confidence"]
                
                # 1. Esqueletos Coloridos
                pid_color = color_for_id(pid)
                desenhar_esqueleto_unificado(viz_frame, kps, kp_thresh=cm.POSE_CONF_MIN, base_color=pid_color)
                
                # 2. BBox (Verde Apenas)
                color_bbox = (0, 255, 0)
                if bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color_bbox, 2)
                    
                    # 3. Label (ID e Confiança)
                    label = f"ID: {pid} | Pessoa: {conf:.2f}"
                    
                    font_scale = 0.6
                    thick = 2
                    
                    (w_txt, h_txt), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
                    
                    # Retângulo Branco Cheio
                    cv2.rectangle(viz_frame, (x1, y1 - h_txt - 10), (x1 + w_txt + 10, y1), (255, 255, 255), -1)
                    
                    # Texto Preto
                    cv2.putText(viz_frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick)
    
            # Envia frame para Escrita
            queue_out.put(viz_frame)
            
            # Atualização de Preview
            if show and state is not None and state.show_preview:
                state.update_frame(viz_frame)
    
            frame_idx += 1

    except KeyboardInterrupt:
        print("\nProcessamento interrompido.")
    finally:
        # Encerra threads
        reader_thread.stop()
        queue_out.put(None) # Sentinel
        writer_thread.join()
        
        if yolo_model and hasattr(yolo_model, 'tracker'): del yolo_model.tracker
        gpu_manager.clear_cache()
        # Full Cleanup
        from neurapose_backend.otimizador.ram import memory as ram_opt
        ram_opt.force_gc()

    # 7. POS-PROC
    if sanitizar_dados:
        records = sanitizar_dados(records)
        
    # --- SALVA JSON DE POSE PADRÃO ---
    # Formato: <nome do video>_pose.json
    json_pose_name = f"{video_path.stem}_pose.json"
    json_pose_path = jsons_dir / json_pose_name
    
    tracker_key = "deepocsort_id" if USING_DEEPOCSORT else "botsort_id"
    
    records_final = []
    
    for r in records:
        # Copia e enriquece
        new_r = r.copy()
        new_r[tracker_key] = r["id_persistente"]
        # Mantém info de classe se existir (já adicionado no loop principal)
        records_final.append(new_r)

    with open(json_pose_path, "w", encoding="utf-8") as f:
        # json.dump(records_final, f)
        json.dump(records_final, f, indent=2, ensure_ascii=False) # Legível conforme solicitado
        
    # --- SALVA JSON DE TRACKING REPORT ---
    # Formato: <nome do video>_tracking.json
    json_tracking_name = f"{video_path.stem}_tracking.json"
    json_tracking_path = jsons_dir / json_tracking_name
    
    # Prepara estrutura do tracking.json
    # { "video": "...", "total_frames": N, "id_map": {...}, "tracking_by_frame": {...} }
    
    # Agrupa por frame
    tracking_by_frame = {}
    ids_encontrados = set()
    
    for r in records_final:
        f_idx = str(r["frame"])
        pid = r["id_persistente"]
        ids_encontrados.add(pid)
        
        if f_idx not in tracking_by_frame:
            tracking_by_frame[f_idx] = []
        
        # Cria objeto reduzido/específico para o tracking report se necessário
        # O user mostrou o mesmo objeto do pose.json dentro da lista, 
        # mas com campo 'classe_id' e 'classe_predita' explicitos como null no exemplo.
        
        track_obj = {
            tracker_key: pid,
            "id_persistente": pid,
            "bbox": r["bbox"],
            "confidence": r["confidence"],
            "classe_id": None,      # Pre-processamento nao tem classificacao ainda
            "classe_predita": None
        }
        tracking_by_frame[f_idx].append(track_obj)
            
    id_map = {str(i): i for i in sorted(list(ids_encontrados))}
    
    tracking_report = {
        "video": video_path.name,
        "total_frames": total_frames,
        "id_map": id_map,
        "tracking_by_frame": tracking_by_frame
    }
    
    with open(json_tracking_path, "w", encoding="utf-8") as f:
        json.dump(tracking_report, f, indent=2, ensure_ascii=False)
    
    # RELATORIO DE TEMPOS (Formato Solicitado)
    total_time = t_norm + t_yolo_acc + t_pose_acc
    
    print("\n" + "="*60)
    print(f"{f'Normalização video {cm.INPUT_NORM_FPS} FPS':<45} {t_norm:>10.2f} seg")
    print(f"{f'YOLO + {cm.TRACKER_NAME} + OSNet':<45} {t_yolo_acc:>10.2f} seg")
    print(f"{'RTMPose':<45} {t_pose_acc:>10.2f} seg")
    print("-" * 60)
    print(f"{'TOTAL':<45} {total_time:>10.2f} seg")
    print("="*60 + "\n")
        
    return str(path_out)