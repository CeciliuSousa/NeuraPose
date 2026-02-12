# ./neurapose_backend/pre_processamento/pipeline/processador.py

import time
import os
import cv2
import json
import numpy as np
import queue
from pathlib import Path
from colorama import Fore

os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import logging
class TaskWarningFilter(logging.Filter):
    def filter(self, record):
        return "Unable to automatically guess model task" not in record.getMessage()

logging.getLogger("ultralytics").addFilter(TaskWarningFilter())

import neurapose_backend.config_master as cm
from neurapose_backend.otimizador.cuda.gpu_utils import gpu_manager
from neurapose_backend.otimizador.cpu import core as cpu_opt
from neurapose_backend.otimizador.ram import memory as ram_opt
from neurapose_backend.detector.yolo_stream import YoloDetectorPerson
from neurapose_backend.rtmpose.extracao_pose_rtmpose import ExtratorPoseRTMPose
from neurapose_backend.thread.frame_reader import FrameReaderThread
from neurapose_backend.thread.video_writer import VideoWriterThread
from neurapose_backend.nucleo.visualizacao import desenhar_esqueleto, color_for_id
from neurapose_backend.nucleo.video_utils import normalizar_video

try:
    from neurapose_backend.globals.state import state
except ImportError:
    state = None

try:
    from neurapose_backend.nucleo.sanatizer import sanitizar_dados
except ImportError:
    sanitizar_dados = None

@gpu_manager.inference_mode()
def processar_video(video_path: Path, output_dir: Path, show: bool = False):
    """
    Processa um vídeo usando PIPELINE MULTITHREAD + SKIP INTELIGENTE.
    """
    
    if not output_dir: raise ValueError("Diretório de saída obrigatório")
    
    predicoes_dir = output_dir / "predicoes"
    jsons_dir = output_dir / "jsons"
    videos_norm_dir = output_dir / "videos"
    
    predicoes_dir.mkdir(parents=True, exist_ok=True)
    jsons_dir.mkdir(parents=True, exist_ok=True)
    videos_norm_dir.mkdir(parents=True, exist_ok=True)

    try:
        video_norm_path, t_norm = normalizar_video(video_path, videos_norm_dir, target_fps=cm.INPUT_NORM_FPS)
        if video_norm_path is None: raise Exception("Retorno None da normalização")
    except Exception as e:
        print(Fore.RED + f"[ERRO] Falha na normalização: {e}")
        return {}

    import shutil
    try:
        video_out_raw = videos_norm_dir / video_norm_path.name
        if not video_out_raw.exists():
            shutil.copy2(video_norm_path, video_out_raw)
    except Exception as e:
        print(f"[AVISO] Falha ao copiar vídeo bruto: {e}")

    cap_temp = cv2.VideoCapture(str(video_norm_path))
    original_fps = cap_temp.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_temp.release()

    target_fps = cm.FPS_TARGET
    skip_interval = max(1, int(round(original_fps / target_fps)))

    queue_in = queue.Queue(maxsize=30)
    queue_out = queue.Queue(maxsize=30)
    
    reader_thread = FrameReaderThread(video_norm_path, skip_interval, queue_in)
    reader_thread.start()
    
    video_out_name = f"{video_path.stem}_pred.mp4"
    pred_video_path = predicoes_dir / video_out_name
    writer_thread = VideoWriterThread(pred_video_path, target_fps, width, height, queue_out)
    writer_thread.start()

    pose_extractor = ExtratorPoseRTMPose(device=cm.DEVICE)
    detector = YoloDetectorPerson(target_fps=target_fps)
    tracker_key = "deepocsort_id" if detector.using_tracker else "botsort_id"
    records = []
    
    t_yolo_acc = 0.0
    t_pose_acc = 0.0
    last_logged_percent = -1

    # Atualiza Status GPU (caso tenha mudado no frontend)
    gpu_manager.update_device(cm.DEVICE)

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
            # [REFACTOR] Wrapper Unificado
            yolo_dets = detector.process_frame(frame, frame_idx=frame_idx)

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
                desenhar_esqueleto(viz_frame, kps, kp_thresh=cm.POSE_CONF_MIN, base_color=pid_color)
                
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
        
        if detector: detector.cleanup()
        gpu_manager.clear_cache()
        # Full Cleanup
        ram_opt.force_gc()

    # 7. POS-PROC
    if sanitizar_dados:
        records = sanitizar_dados(records)
        
    # --- SALVA JSON DE POSE PADRÃO ---
    # Formato: <nome do video>_pose.json
    json_pose_name = f"{video_path.stem}_pose.json"
    json_pose_path = jsons_dir / json_pose_name
    
    # [REFACTOR] tracker_key já definido acima
    # tracker_key = "deepocsort_id" if USING_DEEPOCSORT else "botsort_id"
    
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
        
    return str(pred_video_path)