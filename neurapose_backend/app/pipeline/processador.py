# ./neurapose-backend/app/pipeline/processador.py

import time
import os
import json
import cv2
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
from neurapose_backend.rtmpose.extracao_pose_rtmpose import ExtratorPoseRTMPose
from neurapose_backend.detector.yolo_stream import YoloDetectorPerson
from neurapose_backend.thread.frame_reader import FrameReaderThread
from neurapose_backend.thread.video_writer import VideoWriterThread
from neurapose_backend.nucleo.visualizacao import desenhar_esqueleto
from neurapose_backend.nucleo.video_utils import normalizar_video

from neurapose_backend.temporal.inferencia_temporal import ClassificadorAcao

try:
    from neurapose_backend.globals.state import state
except ImportError:
    state = None

try:
    from neurapose_backend.nucleo.sanatizer import sanitizar_dados
except ImportError:
    sanitizar_dados = None

@gpu_manager.inference_mode()
def processar_video(video_path: Path, lstm_model, mu_ignored, sigma_ignored, show_preview=False, output_dir: Path = None, labels_path: Path = None):
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
    
    tempos = {
        "detector_total": 0.0, "rtmpose_total": 0.0, "temporal_total": 0.0,
        "video_total": 0.0, "yolo": 0.0, "rtmpose": 0.0, "total": 0.0, "normalizacao": 0.0
    }

    try:
        video_norm_path, t_norm = normalizar_video(video_path, videos_norm_dir, target_fps=cm.INPUT_NORM_FPS)
        if video_norm_path is None: raise Exception("Retorno None da normalização")
    except Exception as e:
        print(Fore.RED + f"[ERRO] Falha na normalização: {e}")
        return {}

    tempos["normalizacao"] = t_norm

    cap_temp = cv2.VideoCapture(str(video_norm_path))
    original_fps = cap_temp.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_temp.release()
    
    target_fps = cm.FPS_TARGET
    skip_interval = max(1, int(round(original_fps / target_fps)))
    
    # print(Fore.CYAN + f"[APP] Vídeo: {video_path.name}")

    queue_in = queue.Queue(maxsize=30)  
    queue_out = queue.Queue(maxsize=30) 
    
    reader_thread = FrameReaderThread(video_norm_path, skip_interval, queue_in)
    reader_thread.start()
    
    video_out_name = f"{video_path.stem}_pred.mp4"
    pred_video_path = predicoes_dir / video_out_name
    writer_thread = VideoWriterThread(pred_video_path, target_fps, width, height, queue_out)
    writer_thread.start()

    pose_extractor = ExtratorPoseRTMPose(device=cm.DEVICE)

    model_file = cm.MODEL_SAVE_DIR / "model_best.pt"
    if not model_file.exists() and hasattr(cm, 'TRAINED_MODELS_DIR'):
         candidates = list(cm.TRAINED_MODELS_DIR.glob("**/*.pt"))
         if candidates: model_file = candidates[0]
    
    brain = ClassificadorAcao(str(model_file), model_instance=lstm_model, window_size=cm.TIME_STEPS, mu=mu_ignored, sigma=sigma_ignored)

    detector = YoloDetectorPerson(target_fps=target_fps)

    # 4. LOOP PRINCIPAL (PROCESSAMENTO)
    registros_totais = [] 
    pred_stats = {} 
    id_final_preds = {} 
    last_logged_percent = -1 

    gpu_manager.update_device(cm.DEVICE)
    
    try:
        while True:
            # Pega da fila de leitura
            item = queue_in.get()
            if item is None: # Sentinel
                break
            
            frame_idx, frame = item
            
            # --- LÓGICA DE LOG ---
            current_percent = int((frame_idx / total_frames) * 100)
            should_log = (current_percent % 20 == 0 and current_percent > last_logged_percent) or (frame_idx == 0)
            if should_log:
                last_logged_percent = current_percent
                print(f"\r[APP] Progresso: {current_percent}% ({frame_idx}/{total_frames})")
                
            # SAFE MODE: Throttling & GC
            cpu_opt.throttle()
            ram_opt.smart_cleanup(frame_idx)

            t0 = time.time()
            
            # --- DETECÇÃO ---
            # [REFACTOR] Wrapper Unificado
            yolo_dets = detector.process_frame(frame, frame_idx=frame_idx)

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
                
                rec['anomalia_prob'] = round(prob, 2)
                rec['anomalia'] = prob >= cm.CLASSE2_THRESHOLD
                
                if pid not in pred_stats: pred_stats[pid] = 0.0
                pred_stats[pid] = max(pred_stats[pid], prob)
                
                if rec['anomalia']:
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
                prob = rec.get("anomalia_prob", 0.0)
                anomalia = rec.get("anomalia", False)
                
                base_color = (0, 0, 255) if anomalia else (0, 255, 0)
                desenhar_esqueleto(viz_frame, kps, kp_thresh=cm.POSE_CONF_MIN, base_color=base_color)
                
                color = base_color
                if bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color, 2)
                    
                    display_prob = prob if anomalia else (1.0 - prob)
                    class_name = cm.CLASSE2 if anomalia else cm.CLASSE1
                    
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
        
        
        if detector: detector.cleanup()
        gpu_manager.clear_cache()
        
        # Otimização: Limpeza Final
        ram_opt.force_gc()

    start_time_global = time.time()

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
    tracker_key = "deepocsort_id" if detector.using_tracker else "botsort_id"
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
        anomalia = r.get("anomalia", False)
        classe_id = 1 if anomalia else 0
        classe_nome = cm.CLASSE2 if anomalia else cm.CLASSE1
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
        "total_frames": total_frames,
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