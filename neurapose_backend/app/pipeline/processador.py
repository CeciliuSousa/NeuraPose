# ==============================================================
# neurapose-backend/app/pipeline/processador.py
# ==============================================================
# Pipeline OTIMIZADO (Teste de Modelo)
# Refatorado para PARIDADE com Main Processor (Logical Skip)
# ==============================================================

import time
import os
# Silencia logs verbosos do OpenCV/FFmpeg
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import json
import cv2
import numpy as np
from pathlib import Path
from colorama import Fore
from ultralytics import YOLO

# Importações do projeto
import neurapose_backend.config_master as cm

# --- Módulos Unificados ---
from neurapose_backend.rtmpose.extracao_pose_rtmpose import ExtratorPoseRTMPose
from neurapose_backend.nucleo.visualizacao import desenhar_esqueleto_unificado, color_for_id
from neurapose_backend.nucleo.tracking_utils import gerar_relatorio_tracking
from neurapose_backend.tracker.rastreador import CustomBoTSORT, CustomReID, CustomDeepOCSORT, save_temp_tracker_yaml

# Módulo de Inferência LSTM (Específico do APP)
from neurapose_backend.app.modulos.inferencia_lstm import rodar_lstm_batch
from neurapose_backend.nucleo.sequencia import montar_sequencia_lote

# Import Sanitizer (Task Response)
try:
    from neurapose_backend.nucleo.sanatizer import sanitizar_dados
except ImportError:
    sanitizar_dados = None

def processar_video(video_path: Path, model, mu, sigma, show_preview=False, output_dir: Path = None, labels_path: Path = None):
    """
    Processa um vídeo para TESTE DE MODELO usando LOGICAL SKIP.
    - Input: 30fps (Original)
    - IA: 10fps (Logical Skip)
    - Output: 30fps (Fluid)
    - Extra: Classificação LSTM e Relatório Comparativo
    """

    if not output_dir: raise ValueError("output_dir obrigatório")
    
    # Preparação de Pastas
    predicoes_dir = output_dir / "predicoes"
    jsons_dir = output_dir / "jsons"
    predicoes_dir.mkdir(parents=True, exist_ok=True)
    jsons_dir.mkdir(parents=True, exist_ok=True)

    tempos = {
        "normalizacao": 0.0, # Deprecated (0.0)
        "detector_total": 0.0,
        "rtmpose_total": 0.0,
        "temporal_total": 0.0,
        "video_total": 0.0,
        "yolo": 0.0, "rtmpose": 0.0, "total": 0.0
    }

    # ------------------ Setup Video Input -----------------------
    cap = cv2.VideoCapture(str(video_path))
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    target_fps = cm.FPS_TARGET # 10.0
    skip_interval = max(1, int(round(original_fps / target_fps)))
    
    print(Fore.CYAN + f"[INFO] PROCESSANDO (TESTE): {video_path.name}")
    print(Fore.WHITE + f"       FPS: {original_fps:.2f} -> IA: {target_fps:.2f} (Skip {skip_interval})")

    # ------------------ Setup Writer (Fluid 30fps) -----------------------
    video_out_name = f"{video_path.stem}_pred.mp4"
    pred_video_path = predicoes_dir / video_out_name
    writer = cv2.VideoWriter(str(pred_video_path), cv2.VideoWriter_fourcc(*'mp4v'), original_fps, (width, height))

    # ------------------ Setup Models -----------------------
    pose_extractor = ExtratorPoseRTMPose(device=cm.DEVICE)
    
    USING_DEEPOCSORT = (cm.TRACKER_NAME.upper() == "DEEPOCSORT")
    tracker = None
    yolo_model = None
    yaml_path = None
    
    if USING_DEEPOCSORT:
        print(Fore.MAGENTA + "[TRACKER] Mode: DeepOCSORT (Frame-by-Frame)")
        tracker = CustomDeepOCSORT()
    else:
        print(Fore.CYAN + "[TRACKER] Mode: BoTSORT (Ultralytics)")
        yolo_model = YOLO(str(cm.YOLO_PATH)).to(cm.DEVICE)
        tracker_instance = CustomBoTSORT(frame_rate=int(target_fps)) # Tuned to IA rate
        yolo_model.tracker = tracker_instance
        yaml_path = save_temp_tracker_yaml()

    # ------------------ Processing Loop -----------------------
    frame_idx = 0
    start_time_global = time.time()
    
    registros_totais = [] # Para LSTM
    last_pose_records = [] # Cache para desenho
    
    t_yolo_acc = 0.0
    t_pose_acc = 0.0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            t0 = time.time()
            
            # IA Step
            if frame_idx % skip_interval == 0:
                yolo_dets = None
                
                # 1. Detection
                if USING_DEEPOCSORT:
                    # DeepOCSORT handles infer + track
                    tracks = tracker.track(frame)
                    yolo_dets = tracks
                else:
                    res = yolo_model.track(
                        source=frame, persist=True, tracker=str(yaml_path),
                        verbose=False, classes=[cm.YOLO_CLASS_PERSON]
                    )
                    if len(res) > 0:
                        yolo_dets = res[0].boxes

                t1 = time.time()
                t_yolo_acc += (t1 - t0)

                # 2. Pose
                pose_records, _ = pose_extractor.processar_frame(
                    frame_img=frame,
                    detections_yolo=yolo_dets,
                    frame_idx=frame_idx,
                    desenhar_no_frame=False
                )
                
                t2 = time.time()
                t_pose_acc += (t2 - t1)
                
                last_pose_records = pose_records
                registros_totais.extend(pose_records)
            
            # Render Step (Always run)
            viz_frame = frame.copy()
            for rec in last_pose_records:
                pid = rec["id_persistente"]
                kps = np.array(rec["keypoints"])
                bbox = rec["bbox"]
                conf = rec["confidence"]
                
                if conf < cm.MIN_POSDETECTION_CONF: continue
                
                # Base rendering (Sem classificação ainda, desenha neutro ou amarelo?)
                # Como rodamos LSTM depois, ainda nao temos a classe furto/normal frame a frame
                # Desenha Amarelo/Neutro por enquanto
                color = (0, 255, 255) 
                
                desenhar_esqueleto_unificado(viz_frame, kps, kp_thresh=cm.POSE_CONF_MIN, base_color=color)
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(viz_frame, f"ID: {pid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            writer.write(viz_frame)
            frame_idx += 1
            
            if frame_idx % 30 == 0:
                print(f"\r[TEST] Frame {frame_idx}/{total_frames}", end="")

    finally:
        cap.release()
        writer.release()
        if not USING_DEEPOCSORT and yolo_model:
            del yolo_model.tracker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[INFO] Extração finalizada.")
    
    # ------------------ Post-Processing & LSTM -----------------------
    tempos["detector_total"] = t_yolo_acc
    tempos["rtmpose_total"] = t_pose_acc
    
    # [SANITIZAÇÃO]
    if sanitizar_dados:
        # print(Fore.CYAN + "[INFO] Aplicando Sanitização (Spatial Lock + Velocity Gating)...")
        registros_totais = sanitizar_dados(registros_totais, threshold=150.0)

    # [LSTM CLASSIFICATION]
    print(Fore.CYAN + f"[INFO] Executando Classificador {cm.TEMPORAL_MODEL.upper()}...")
    t0_lstm = time.time()
    
    ids_validos = list(set(r["id_persistente"] for r in registros_totais))
    
    id_preds = {gid: 0 for gid in ids_validos}
    id_scores = {gid: 0.0 for gid in ids_validos}
    
    seqs_dict = montar_sequencia_lote(registros_totais, ids_validos)
    
    if seqs_dict:
        batch_preds, batch_scores = rodar_lstm_batch(seqs_dict, model, mu, sigma)
        for gid, raw_pred in batch_preds.items():
            score = batch_scores.get(gid, 0.0)
            classe_id = 1 if score >= cm.CLASSE2_THRESHOLD else 0
            id_preds[gid] = classe_id
            id_scores[gid] = score

    tempos["temporal_total"] = time.time() - t0_lstm

    # ------------------ Metrics & Reporting -----------------------
    # Setup Real Labels
    effective_labels_path = labels_path if labels_path else (output_dir / "anotacoes" / "labels.json" if output_dir else None)
    real_labels = {}
    video_stem = video_path.stem
    
    if effective_labels_path and effective_labels_path.exists():
        try:
            with open(effective_labels_path, "r", encoding="utf-8") as f:
                all_labels = json.load(f)
            for key in all_labels:
                if key in video_stem or video_stem in key:
                    video_labels = all_labels[key]
                    for pid, cls in video_labels.items():
                        if isinstance(cls, dict):
                            real_labels[int(pid)] = cls.get("classe", cm.CLASSE1).upper()
                        else:
                            real_labels[int(pid)] = str(cls).upper()
                    break
        except Exception: pass

    # Table Print
    print("")
    print("| ID  | Real   | Predito | Conf.  | OK? |")
    print("|-----|--------|---------|--------|-----|")
    
    correct = 0
    total = 0
    
    ids_predicoes = [] # Format for return
    
    for gid in sorted(ids_validos):
        classe_id = id_preds.get(gid, 0)
        score = id_scores.get(gid, 0.0)
        pred_class = cm.CLASSE2 if classe_id == 1 else cm.CLASSE1
        real_class = real_labels.get(gid, "?")
        
        is_correct = False
        if real_class != "?":
            is_correct = (real_class.upper() == pred_class.upper())
            if is_correct: correct += 1
            total += 1
            status = "✓"
        else:
            status = "-"
            
        print(f"| {gid:>3} | {real_class[:6]:<6} | {pred_class[:7]:<7} | {score*100:>5.1f}% | {status:^3} |")
        
        ids_predicoes.append({
            "id": int(gid),
            "classe_id": int(classe_id),
            f"score_{cm.CLASSE2}": float(score)
        })

    if total > 0:
        print(f"\nResultado: {correct}/{total} ({ (correct/total)*100:.1f}%)")
    
    # ------------------ Finalize -----------------------
    # Save JSON with classification info
    for r in registros_totais:
        gid = r["id_persistente"]
        r["classe_id"] = id_preds.get(gid, 0)
        r["classe_predita"] = cm.CLASSE2 if r["classe_id"] == 1 else cm.CLASSE1
        r[f"score_{cm.CLASSE2}_id"] = id_scores.get(gid, 0.0)
        
    json_path = jsons_dir / f"{video_path.stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(registros_totais, f, indent=2, ensure_ascii=False)

    # Tracking Report
    tracking_json_path = jsons_dir / f"{video_path.stem}_tracking.json"
    id_map_dummy = {id: id for id in ids_validos}
    gerar_relatorio_tracking(registros_totais, id_map_dummy, ids_validos, frame_idx, video_path.name, tracking_json_path)

    total_time = time.time() - start_time_global
    tempos["video_total"] = total_time
    tempos["total"] = total_time
    tempos["yolo"] = tempos["detector_total"]
    tempos["rtmpose"] = tempos["rtmpose_total"]

    print(Fore.GREEN + f"[OK] Teste Finalizado em {total_time:.2f}s")

    # Return structure matching expected interface
    video_pred = 1 if any(v == 1 for v in id_preds.values()) else 0
    video_score = max(id_scores.values()) if id_scores else 0.0
    
    return {
        "video": str(video_path),
        "pred": int(video_pred),
        f"score_{cm.CLASSE2}": float(video_score),
        "tempos": tempos,
        "ids_predicoes": ids_predicoes
    }