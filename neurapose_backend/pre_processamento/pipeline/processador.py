# ==============================================================
# neurapose-backend/pre_processamento/pipeline/processador.py
# ==============================================================
# Pipeline OTIMIZADO (Extração de Dados)
# Visualização padronizada conforme regras de UI vFinal
# ==============================================================

import time
import os
import cv2
import json
import numpy as np
from pathlib import Path
from colorama import Fore
from ultralytics import YOLO
import torch

# Silencia logs OpenCV
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import neurapose_backend.config_master as cm
from neurapose_backend.tracker.rastreador import CustomBoTSORT, CustomDeepOCSORT, save_temp_tracker_yaml
from neurapose_backend.rtmpose.extracao_pose_rtmpose import ExtratorPoseRTMPose
from neurapose_backend.nucleo.visualizacao import desenhar_esqueleto_unificado, color_for_id
from neurapose_backend.nucleo.tracking_utils import gerar_relatorio_tracking
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

def processar_video(video_path: Path, output_dir: Path, show: bool = False):
    """
    Processa vídeo para geração de dataset.
    - FPS: Convertido para cm.FPS_TARGET (10fps).
    - Codec: H.264 (avc1) via OpenH264.
    - Style: BBox Verde, Texto Preto/Fundo Branco.
    """
    
    # 1. SETUP DE DIRETÓRIOS
    if not output_dir: raise ValueError("output_dir obrigatório")
    
    predicoes_dir = output_dir / "predicoes"
    jsons_dir = output_dir / "jsons"
    videos_norm_dir = output_dir / "videos"
    
    predicoes_dir.mkdir(parents=True, exist_ok=True)
    jsons_dir.mkdir(parents=True, exist_ok=True)
    videos_norm_dir.mkdir(parents=True, exist_ok=True)

    # print(Fore.CYAN + f"[PROC] Iniciando: {video_path.name}")

    # 2. NORMALIZAÇÃO (REMOVIDO POR PERFORMANCE)
    # OBS: Otimização solicitada para reduzir overhead de ffmpeg.
    # Usaremos LOGICAL SKIP para processar apenas os frames alvo (10fps) do vídeo original.
    
    # video_norm_path, t_norm = normalizar_video(video_path, videos_norm_dir)
    t_norm = 0.0
    video_norm_path = video_path # Usa original

    # 3. EXTRAÇÃO (Lê o vídeo ORIGINAL 30fps e aplica SKIP)
    cap = cv2.VideoCapture(str(video_norm_path))
    
    # Propriedades
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calcula Skip para atingir TARGET_FPS (10)
    target_fps = cm.FPS_TARGET
    skip_interval = max(1, int(round(fps_in / target_fps)))

    # 4. SETUP WRITER (Predições a 10fps)
    video_out_name = f"{video_path.stem}_pred.mp4"
    path_out = predicoes_dir / video_out_name
    
    # Tenta usar codec AVC1 (H.264)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(str(path_out), fourcc, target_fps, (w, h))

    # Fallback se falhar
    if not writer.isOpened():
        print(Fore.YELLOW + "[AVISO] 'avc1' falhou. Tentando 'mp4v'.")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(path_out), fourcc, target_fps, (w, h))

    # 5. MODELOS
    pose_extractor = ExtratorPoseRTMPose(device=cm.DEVICE)
    
    USING_DEEPOCSORT = (cm.TRACKER_NAME.upper() == "DEEPOCSORT")
    tracker = None
    yolo_model = None
    yaml_path = None
    
    if USING_DEEPOCSORT:
        tracker = CustomDeepOCSORT()
    else:
        yolo_model = YOLO(str(cm.YOLO_PATH)).to(cm.DEVICE)
        # Tracker configurado para o FPS do target (10fps) já que vamos pular frames
        tracker_instance = CustomBoTSORT(frame_rate=int(target_fps))
        yolo_model.tracker = tracker_instance
        yaml_path = save_temp_tracker_yaml()

    # 6. LOOP
    records = []
    frame_idx = 0
    
    t_yolo_acc = 0.0
    t_pose_acc = 0.0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # LOGICAL SKIP: Apenas processa se for o frame do intervalo
            if frame_idx % skip_interval == 0:
                t0 = time.time()
                
                # --- DETECÇÃO ---
                yolo_dets = None
                if USING_DEEPOCSORT:
                    tracks = tracker.track(frame)
                    yolo_dets = tracks
                else:
                    res = yolo_model.track(
                        source=frame, persist=True, tracker=str(yaml_path),
                        verbose=False, classes=[cm.YOLO_CLASS_PERSON]
                    )
                    if len(res) > 0: yolo_dets = res[0].boxes
                
                t1 = time.time()
                t_yolo_acc += (t1 - t0)
    
                # --- POSE ---
                pose_records, _ = pose_extractor.processar_frame(
                    frame_img=frame,
                    detections_yolo=yolo_dets,
                    frame_idx=frame_idx,
                    desenhar_no_frame=False 
                )
                
                t2 = time.time()
                t_pose_acc += (t2 - t1)
                
                records.extend(pose_records)
                
                # --- VISUALIZAÇÃO PADRONIZADA ---
                # Regra: BBox Verde, Texto Preto/Branco, Esqueletos Coloridos
                viz_frame = frame.copy()
                
                for rec in pose_records:
                    pid = rec["id_persistente"]
                    bbox = rec["bbox"]
                    kps = np.array(rec["keypoints"])
                    conf = rec["confidence"]
                    
                    # 1. Esqueletos Coloridos (Padrão RTMPose / Random ID color)
                    pid_color = color_for_id(pid)
                    desenhar_esqueleto_unificado(viz_frame, kps, kp_thresh=cm.POSE_CONF_MIN, base_color=pid_color)
                    
                    # 2. BBox Verde Apenas
                    color_bbox = (0, 255, 0) # Verde BGR
                    if bbox is not None:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color_bbox, 2)
                        
                        # 3. Texto Preto com Fundo Branco
                        # "ID: <id> | Pessoa: <conf>"
                        label = f"ID: {pid} | Pessoa: {conf:.2f}"
                        
                        font_scale = 0.6
                        thick = 2
                        (w_txt, h_txt), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
                        
                        # Retângulo Branco Cheio
                        cv2.rectangle(viz_frame, (x1, y1 - h_txt - 10), (x1 + w_txt + 10, y1), (255, 255, 255), -1)
                        
                        # Texto Preto
                        cv2.putText(viz_frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick)
    
                # Grava frame processado (10fps output)
                writer.write(viz_frame)
                
                # Atualização de Preview para o App Web (Throttle implícito pelo skip_interval = 10fps)
                if show and state is not None:
                    state.update_frame(viz_frame)
    
            frame_idx += 1
            
            # print(f"\r[INFO] INICIANDO FASE PROCESSAMENTO...")
            
            if frame_idx % 10 == 0:
                print(f"\r[PROCESSAMENTO] Frame {frame_idx}/{total_frames}", end="")

    except KeyboardInterrupt:
        print("\nProcessamento interrompido.")
    finally:
        cap.release()
        writer.release()
        if yolo_model and hasattr(yolo_model, 'tracker'): del yolo_model.tracker
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # 7. POS-PROC
    if sanitizar_dados:
        records = sanitizar_dados(records)
        
    # --- SALVA JSON DE POSE PADRÃO ---
    # Formato: <nome do video>_pose.json (como solicitado)
    # Contém lista flat de detecções
    json_pose_name = f"{video_path.stem}_pose.json"
    json_pose_path = jsons_dir / json_pose_name
    
    # Adicionar chave dinâmica do tracker para conformidade com o request
    # O request pede "<tracker configurado>: 1"
    tracker_key = "deepocsort_id" if USING_DEEPOCSORT else "botsort_id"
    
    records_final = []
    for r in records:
        # Copia para não alterar o original se for usado depois
        new_r = r.copy()
        # Adiciona a chave especifica do tracker alem do id_persistente
        new_r[tracker_key] = r["id_persistente"]
        records_final.append(new_r)

    with open(json_pose_path, "w", encoding="utf-8") as f:
        json.dump(records_final, f, indent=2, ensure_ascii=False)
        
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
    print(f"{'Normalização video 10 FPS':<45} {t_norm:>10.2f} seg")
    print(f"{'YOLO + BoTSORT-Custom + OSNet':<45} {t_yolo_acc:>10.2f} seg")
    print(f"{'RTMPose':<45} {t_pose_acc:>10.2f} seg")
    print("-" * 60)
    print(f"{'TOTAL':<45} {total_time:>10.2f} seg")
    print("="*60 + "\n")
        
    return str(path_out)