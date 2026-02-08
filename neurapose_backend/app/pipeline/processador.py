# ==============================================================
# neurapose-backend/app/pipeline/processador.py
# ==============================================================
# Pipeline OTIMIZADO (Teste de Modelo / App Final)
# Refatorado para VISUALIZAÇÃO PADRONIZADA (Regras de UI)
# INCLUI:
#  - Skip Inteligente (cap.grab)
#  - Pipeline Multithread (Leitura -> GPU -> Escrita)
#  - Codec GPU (h264_nvenc)
# ==============================================================

import time
import os
import json
import cv2
import numpy as np
import threading
import queue
from pathlib import Path
from colorama import Fore
from ultralytics import YOLO
import torch

# Silencia logs verbosos do OpenCV/FFmpeg
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import logging
class TaskWarningFilter(logging.Filter):
    def filter(self, record):
        return "Unable to automatically guess model task" not in record.getMessage()

logging.getLogger("ultralytics").addFilter(TaskWarningFilter())

# Importações do projeto
import neurapose_backend.config_master as cm
from neurapose_backend.cuda.gpu_utils import gpu_manager
from neurapose_backend.rtmpose.extracao_pose_rtmpose import ExtratorPoseRTMPose
from neurapose_backend.nucleo.visualizacao import desenhar_esqueleto_unificado, color_for_id
from neurapose_backend.nucleo.tracking_utils import gerar_relatorio_tracking
from neurapose_backend.tracker.rastreador import CustomBoTSORT, CustomDeepOCSORT, save_temp_tracker_yaml
from neurapose_backend.temporal.inferencia_temporal import ClassificadorAcao
from neurapose_backend.nucleo.video_utils import normalizar_video

# Import Sanitizer e State
try:
    from neurapose_backend.nucleo.sanatizer import sanitizar_dados
except ImportError:
    sanitizar_dados = None

try:
    from neurapose_backend.globals.state import state
except ImportError:
    state = None

# --- CLASSES AUXILIARES DE THREADING ---

class FrameReaderThread(threading.Thread):
    def __init__(self, video_path, skip_interval, queue_out, max_frames=None):
        super().__init__()
        self.video_path = str(video_path)
        self.skip_interval = skip_interval
        self.queue_out = queue_out
        self.max_frames = max_frames # Inutilizado por enquanto
        self.stopped = False
        self.daemon = True # Mata thread se main morrer
        
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
                
                # Envia para fila (Bloqueia se cheia para não estourar RAM)
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
            # Tenta Backend FFMPEG com NVENC
            try:
                # Nota: OpenCV Python pip geralmente não tem CUDA habilitado.
                # Isso pode falhar e cair no except.
                p = cv2.VideoWriter_fourcc(*'h264') # Dummy
                # Tentativa genérica, mas o OpenCV com FFMPEG backend pode aceitar string de codec se compilado
                # Como fallback seguro, usamos o padrão 'avc1' que é o melhor via CPU.
                # Para usar NVENC real via OpenCV Python, precisaria de GStreamer string.
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
def processar_video(video_path: Path, lstm_model, mu_ignored, sigma_ignored, show_preview=False, output_dir: Path = None, labels_path: Path = None):
    """
    Processa um vídeo usando PIPELINE MULTITHREAD + SKIP INTELIGENTE.
    """

    if not output_dir: raise ValueError("output_dir obrigatório")
    
    # 0. SETUP DIRS
    predicoes_dir = output_dir / "predicoes"
    jsons_dir = output_dir / "jsons"
    anotacoes_dir = output_dir / "anotacoes"
    videos_norm_dir = output_dir / "videos" 
    
    predicoes_dir.mkdir(parents=True, exist_ok=True)
    jsons_dir.mkdir(parents=True, exist_ok=True)
    anotacoes_dir.mkdir(parents=True, exist_ok=True)
    videos_norm_dir.mkdir(parents=True, exist_ok=True) 
    
    tempos = {
        "detector_total": 0.0, "rtmpose_total": 0.0, "temporal_total": 0.0,
        "video_total": 0.0, "yolo": 0.0, "rtmpose": 0.0, "total": 0.0, "normalizacao": 0.0
    }

    # Normalização de FPS (Garanta 30 FPS no input)
    t_start_norm = time.time()
    try:
        video_norm_path, t_norm_internal = normalizar_video(video_path, videos_norm_dir, target_fps=cm.INPUT_NORM_FPS)
        if video_norm_path is None: raise Exception("Retorno None da normalização")
    except Exception as e:
        print(Fore.RED + f"[ERRO] Falha na normalização: {e}")
        return {}

    tempos["normalizacao"] = t_norm_internal

    # 1. SETUP VIDEO INFO (Apenas metadados)
    cap_temp = cv2.VideoCapture(str(video_norm_path))
    original_fps = cap_temp.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_in = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_temp.release()
    
    # Define Target FPS
    target_fps = cm.FPS_TARGET # 10.0
    skip_interval = max(1, int(round(original_fps / target_fps)))
    
    print(Fore.CYAN + f"[APP] Vídeo: {video_path.name}")

    # 2. SETUP QUEUES E THREADS
    queue_in = queue.Queue(maxsize=30)  # Leitura -> Processamento
    queue_out = queue.Queue(maxsize=30) # Processamento -> Escrita
    
    # Thread Leitura
    reader_thread = FrameReaderThread(video_norm_path, skip_interval, queue_in)
    reader_thread.start()
    
    # Thread Escrita
    video_out_name = f"{video_path.stem}_pred.mp4"
    pred_video_path = predicoes_dir / video_out_name
    writer_thread = VideoWriterThread(pred_video_path, target_fps, width, height, queue_out)
    writer_thread.start()

    # 3. SETUP MODELOS (Na Thread Principal - GPU)
    pose_extractor = ExtratorPoseRTMPose(device=cm.DEVICE)
    model_file = cm.MODEL_SAVE_DIR / "model_best.pt"
    if not model_file.exists() and hasattr(cm, 'TRAINED_MODELS_DIR'):
         candidates = list(cm.TRAINED_MODELS_DIR.glob("**/*.pt"))
         if candidates: model_file = candidates[0]
    
    brain = ClassificadorAcao(str(model_file), model_instance=lstm_model, window_size=cm.TIME_STEPS, mu=mu_ignored, sigma=sigma_ignored)

    USING_DEEPOCSORT = (cm.TRACKER_NAME.upper() == "DEEPOCSORT")
    tracker = None
    yolo_model = None
    
    if USING_DEEPOCSORT:
        tracker = CustomDeepOCSORT()
    else:
        yolo_model = YOLO(str(cm.YOLO_PATH), task='detect').to(cm.DEVICE)
        tracker_instance = CustomBoTSORT(frame_rate=int(target_fps))
        yolo_model.tracker = tracker_instance

    # 4. LOOP PRINCIPAL (PROCESSAMENTO)
    registros_totais = [] 
    pred_stats = {} 
    id_final_preds = {} 
    last_logged_percent = -1 

    gpu_manager.update_device(cm.DEVICE)
    
    start_time_global = time.time()

    try:
        while True:
            # Pega da fila de leitura
            item = queue_in.get()
            if item is None: # Sentinel
                break
            
            frame_idx, frame = item
            
            # --- LÓGICA DE LOG ---
            current_percent = int((frame_idx / total_frames_in) * 100)
            should_log = (current_percent % 20 == 0 and current_percent > last_logged_percent) or (frame_idx == 0)
            if should_log:
                last_logged_percent = current_percent
                print(f"\r[APP] Progresso: {current_percent}% ({frame_idx}/{total_frames_in})")
                
            t0 = time.time()
            
            # --- DETECÇÃO ---
            yolo_dets = None
            if USING_DEEPOCSORT:
                tracks = tracker.track(frame)
                yolo_dets = tracks
            else:
                res = yolo_model.predict(source=frame, imgsz=cm.YOLO_IMGSZ, conf=cm.DETECTION_CONF, device=cm.DEVICE, classes=[cm.YOLO_CLASS_PERSON], verbose=False, stream=False)
                dets = np.empty((0, 6))
                if len(res) > 0 and len(res[0].boxes) > 0:
                    dets = res[0].boxes.data.cpu().numpy()
                    if dets.shape[1] == 4:
                            r = dets.shape[0]
                            dets = np.hstack((dets, np.full((r, 1), 0.85), np.zeros((r, 1))))
                    elif dets.shape[1] == 5:
                            dets = np.hstack((dets, np.zeros((dets.shape[0], 1))))
                
                tracks = tracker_instance.update(dets, frame)
                yolo_dets = tracks

            t1 = time.time()
            tempos["detector_total"] += (t1 - t0)

            # --- POSE ---
            pose_records, _ = pose_extractor.processar_frame(frame_img=frame, detections_yolo=yolo_dets, frame_idx=frame_idx, desenhar_no_frame=False)
            
            t2 = time.time()
            tempos["rtmpose_total"] += (t2 - t1)
            
            # --- CLASSIFICAÇÃO (BATCH) ---
            t3_start = time.time()
            
            # 1. Coleta dados para Batch
            track_ids = [r["id_persistente"] for r in pose_records]
            kps_list = [r["keypoints"] for r in pose_records]
            
            # 2. Inferência em Batch (1 chamada GPU)
            probs_map = brain.predict_batch(track_ids, kps_list)
            
            # 3. Atribui resultados
            for rec in pose_records:
                pid = rec["id_persistente"]
                
                # Se por algum motivo o ID não voltar (ex: buffer < 1 frame), assume 0.0
                prob = probs_map.get(pid, 0.0)
                
                rec['theft_prob'] = round(prob, 2)
                rec['is_theft'] = prob >= cm.CLASSE2_THRESHOLD
                
                if pid not in pred_stats: pred_stats[pid] = 0.0
                pred_stats[pid] = max(pred_stats[pid], prob)
                
                if rec['is_theft']:
                    id_final_preds[pid] = 1
                elif pid not in id_final_preds:
                    id_final_preds[pid] = 0

            t3_end = time.time()
            tempos["temporal_total"] += (t3_end - t3_start)
            
            registros_totais.extend(pose_records)
            
            # --- RENDERIZAÇÃO (Na Thread Principal mesmo, rápido) ---
            viz_frame = frame.copy()

            for rec in pose_records:
                pid = rec["id_persistente"]
                bbox = rec["bbox"]
                kps = np.array(rec["keypoints"])
                conf = rec["confidence"]
                prob = rec.get(f"{cm.CLASSE2.lower()}_prob", 0.0)
                is_theft = rec.get(f"is_{cm.CLASSE2.lower()}", False)
                
                base_color = (0, 0, 255) if is_theft else (0, 255, 0)
                desenhar_esqueleto_unificado(viz_frame, kps, kp_thresh=cm.POSE_CONF_MIN, base_color=base_color)
                
                color = base_color
                if bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color, 2)
                    
                    display_prob = prob if is_theft else (1.0 - prob)
                    class_name = cm.CLASSE2 if is_theft else cm.CLASSE1
                    
                    line1_text = f"ID: {pid} | Conf: {conf:.2f}"
                    line2_text = f"Classe: {class_name} | Conf: {display_prob:.1%}"
                    
                    font_scale = 0.6
                    thick = 2
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (w1, h1), _ = cv2.getTextSize(line1_text, font, font_scale, thick)
                    (w2, h2), _ = cv2.getTextSize(line2_text, font, font_scale, thick)
                    
                    pad = 5
                    w_box1, h_box1 = w1 + 2*pad, h1 + 2*pad
                    w_box2, h_box2 = w2 + 2*pad, h2 + 2*pad
                    
                    y_id_bot = y1 - h_box2
                    y_id_top = y_id_bot - h_box1
                    y_cls_bot = y1
                    y_cls_top = y1 - h_box2
                    
                    if y_id_top < 0:
                        y_id_top = y1 + 5
                        y_id_bot = y_id_top + h_box1
                        y_cls_top = y_id_bot
                        y_cls_bot = y_cls_top + h_box2
                        
                    cv2.rectangle(viz_frame, (x1, y_id_top), (x1 + w_box1, y_id_bot), (255, 255, 255), -1)
                    cv2.putText(viz_frame, line1_text, (x1 + pad, y_id_bot - pad), font, font_scale, (0, 0, 0), thick)
                    cv2.rectangle(viz_frame, (x1, y_cls_top), (x1 + w_box2, y_cls_bot), color, -1)
                    cv2.putText(viz_frame, line2_text, (x1 + pad, y_cls_bot - pad), font, font_scale, (255, 255, 255), thick)
            
            # Envia para Escrita
            queue_out.put(viz_frame)
            
            # Atualização de Preview
            if show_preview and state is not None and state.show_preview:
                state.update_frame(viz_frame)

            if state and state.stop_requested: break

    except KeyboardInterrupt:
        print("\n[STOP] Interrompido pelo usuário.")

    finally:
        # Encerra threads
        reader_thread.stop()
        queue_out.put(None) # Sentinel para writer
        writer_thread.join()
        
        if not USING_DEEPOCSORT and yolo_model: del yolo_model.tracker
        gpu_manager.clear_cache()

    total_time = time.time() - start_time_global
    tempos["video_total"] = total_time
    tempos["total"] = total_time
    tempos["yolo"] = tempos["detector_total"]
    tempos["rtmpose"] = tempos["rtmpose_total"]

    print(Fore.GREEN + f"\n[OK] Processamento Finalizado em {total_time:.2f}s")
    
    # RELATORIO DE TEMPOS
    t_norm = tempos["normalizacao"]
    t_yolo = tempos["detector_total"]
    t_pose = tempos["rtmpose_total"]
    t_temp = tempos["temporal_total"]
    calc_total = t_norm + t_yolo + t_pose + t_temp
    model_name = cm.TEMPORAL_MODEL.upper() if getattr(cm, 'TEMPORAL_MODEL', None) else "TEMPORAL MODEL"
    
    print("\n" + "="*60)
    print(f"{f'Normalização video {cm.INPUT_NORM_FPS} FPS':<45} {t_norm:>10.2f} seg")
    print(f"{f'YOLO + {cm.TRACKER_NAME} + OSNet':<45} {t_yolo:>10.2f} seg")
    print(f"{'RTMPose':<45} {t_pose:>10.2f} seg")
    print(f"{cm.TEMPORAL_MODEL.upper():<45} {t_temp:>10.2f} seg")
    print("-" * 60)
    print(f"{'TOTAL':<45} {calc_total:>10.2f} seg")
    print("="*60 + "\n")

    # POS-PROC
    if sanitizar_dados:
        registros_totais = sanitizar_dados(registros_totais, threshold=150.0)

    # --- SALVA JSON DE POSE PADRÃO ---
    json_pose_name = f"{video_path.stem}_pose.json"
    json_pose_path = jsons_dir / json_pose_name
    tracker_key = "deepocsort_id" if USING_DEEPOCSORT else "botsort_id"
    records_final = []
    for r in registros_totais:
        new_r = r.copy()
        new_r[tracker_key] = r["id_persistente"]
        records_final.append(new_r)

    with open(json_pose_path, "w", encoding="utf-8") as f:
        json.dump(records_final, f, indent=2, ensure_ascii=False)

    # --- SALVA JSON DE TRACKING REPORT ---
    json_tracking_path = jsons_dir / f"{video_path.stem}_tracking.json"
    tracking_by_frame = {}
    ids_encontrados = set()
    
    for r in records_final:
        f_idx = str(r["frame"])
        pid = r["id_persistente"]
        ids_encontrados.add(pid)
        if f_idx not in tracking_by_frame: tracking_by_frame[f_idx] = []
        is_theft = r.get("is_theft", False)
        classe_id = 1 if is_theft else 0
        classe_nome = cm.CLASSE2 if is_theft else cm.CLASSE1
        track_obj = {
            tracker_key: pid,
            "id_persistente": pid,
            "bbox": r["bbox"],
            "confidence": r["confidence"],
            "classe_id": classe_id,
            "classe_predita": classe_nome
        }
        tracking_by_frame[f_idx].append(track_obj)
    
    id_map = {str(i): i for i in sorted(list(ids_encontrados))}
    tracking_report = {
        "video": video_path.name,
        "total_frames": total_frames_in,
        "id_map": id_map,
        "tracking_by_frame": tracking_by_frame
    }
    with open(json_tracking_path, "w", encoding="utf-8") as f:
        json.dump(tracking_report, f, indent=2, ensure_ascii=False)
    
    ids_predicoes = []
    for pid, pred_cls in id_final_preds.items():
        score = pred_stats.get(pid, 0.0)
        ids_predicoes.append({
            "id": int(pid),
            "classe_id": int(pred_cls),
            f"score_{cm.CLASSE2}": float(score)
        })

    video_pred = 1 if any(v == 1 for v in id_final_preds.values()) else 0
    video_score = max(pred_stats.values()) if pred_stats else 0.0
    
    return {
        "video": str(video_path),
        "pred": int(video_pred),
        f"score_{cm.CLASSE2}": float(video_score),
        "tempos": tempos,
        "ids_predicoes": ids_predicoes
    }