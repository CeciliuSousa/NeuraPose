# ==============================================================
# neurapose_backend/pre_processamento/pipeline/processador.py
# ==============================================================
# Pipeline UNIFICADO E MODULARIZADO (Pré-processamento)
# REFATORED: Logical Skip (30fps Input -> 10fps Inference -> 30fps Output)
# ==============================================================

import sys
import cv2
import json
import time
import numpy as np
import torch
from pathlib import Path
from colorama import Fore
from ultralytics import YOLO

# Configuração e Detector
import neurapose_backend.config_master as cm

# Módulos Modulares Unificados
from neurapose_backend.rtmpose.extracao_pose_rtmpose import ExtratorPoseRTMPose
from neurapose_backend.nucleo.visualizacao import desenhar_esqueleto_unificado, color_for_id
from neurapose_backend.nucleo.tracking_utils import gerar_relatorio_tracking
from neurapose_backend.tracker.rastreador import CustomBoTSORT, CustomReID, CustomDeepOCSORT, save_temp_tracker_yaml

# [TASK 3: Sanitizer Import]
try:
    from neurapose_backend.nucleo.sanatizer import sanitizar_dados
except ImportError:
    sanitizar_dados = None
    print(Fore.YELLOW + "[WARN] Módulo Sanitizer não encontrado. Pulando etapa.")

try:
    from neurapose_backend.globals.state import state as state_notifier
except:
    state_notifier = None

# Carregar config do usuário se disponível
try:
    from neurapose_backend.nucleo.user_config_manager import UserConfigManager
    user_config = UserConfigManager.load_config()
    for k, v in user_config.items():
        if hasattr(cm, k):
            setattr(cm, k, v)
except Exception as e:
    print(Fore.YELLOW + f"[CONFIG] Falha ao carregar configurações do usuário: {e}")


def processar_video(video_path: Path, out_root: Path, show=False):
    """
    Processa um vídeo usando estratégia de LOGICAL SKIP.
    - Lê vídeo original (ex: 30fps).
    - Executa IA a cada N frames (target 10fps).
    - Grava saída no FPS original (fluido).
    """

    # ------------------ Diretorios -----------------------
    videos_dir = out_root / "videos"
    preds_dir = out_root / "predicoes"
    json_dir = out_root / "jsons"

    videos_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    # ------------------ Setup Video Input -----------------------
    cap = cv2.VideoCapture(str(video_path))
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    target_fps = cm.FPS_TARGET # 10.0
    
    # Calcula salto: 30 / 10 = 3 => Processa frames 0, 3, 6...
    skip_interval = max(1, int(round(original_fps / target_fps)))
    
    print(Fore.CYAN + f"[INFO] PROCESSANDO: {video_path.name}")
    print(Fore.WHITE + f"       Input FPS: {original_fps:.2f} | Target IA: {target_fps:.2f} | Skip Interval: {skip_interval}")

    # ------------------ Setup Writer (30fps Fluido) -----------------------
    video_out_name = f"{video_path.stem}_{int(target_fps)}fps_pose.mp4"
    video_out_path = preds_dir / video_out_name
    
    # Codec mp4v
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_out_path), fourcc, original_fps, (width, height))

    # ------------------ Setup Models -----------------------
    pose_extractor = ExtratorPoseRTMPose(device=cm.DEVICE)
    
    USING_DEEPOCSORT = (cm.TRACKER_NAME.upper() == "DEEPOCSORT")
    tracker = None
    yolo_model = None
    
    if USING_DEEPOCSORT:
        print(Fore.MAGENTA + "[TRACKER] Inicializando DeepOCSORT (Frame-a-Frame)...")
        tracker = CustomDeepOCSORT()
    else:
        print(Fore.CYAN + "[TRACKER] Inicializando BoTSORT (Ultralytics)...")
        yolo_model = YOLO(str(cm.YOLO_PATH)).to(cm.DEVICE)
        # Tracker update rate matches target_fps since we skip frames
        tracker_instance = CustomBoTSORT(frame_rate=int(target_fps))
        yolo_model.tracker = tracker_instance
        yaml_path = save_temp_tracker_yaml()

    # ------------------ Processing Loop -----------------------
    frame_idx = 0
    start_time = time.time()
    
    # Accumulators
    registros_totais = []
    
    # Cache para frames pulados (Visualização Fluida)
    # Mantemos os ultimos registros validos para desenhar nos frames intermediarios
    last_pose_records = []
    
    # Estatisticas (Estimativa simples pois agora é tudo junto)
    t_yolo_acc = 0.0
    t_pose_acc = 0.0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Stop Check
            if state_notifier and state_notifier.stop_requested:
                print(Fore.YELLOW + "[STOP] Interrompido pelo usuário.")
                break

            t0 = time.time()
            
            # Lógica de Skip: Roda IA apenas nos frames chave
            should_run_ia = (frame_idx % skip_interval == 0)
            
            if should_run_ia:
                yolo_dets = None
                
                # 1. Detection & Tracking
                if USING_DEEPOCSORT:
                    # DeepOCSORT roda YOLO internamente + Tracker
                    # Retorna Ndarray: [x1, y1, x2, y2, id, conf, cls]
                    tracks = tracker.track(frame)
                    yolo_dets = tracks 
                
                else:
                    # BoTSORT via Ultralytics
                    res = yolo_model.track(
                        source=frame,
                        persist=True,
                        tracker=str(yaml_path),
                        verbose=False,
                        classes=[cm.YOLO_CLASS_PERSON]
                    )
                    # res[0].boxes é objeto Boxes
                    # Extrator de pose lida com isso
                    if len(res) > 0:
                        yolo_dets = res[0].boxes
                
                t1 = time.time()
                t_yolo_acc += (t1 - t0)

                # 2. Pose Extraction
                # O Extrator também normaliza coordenadas, faz o crop, roda RTMPose e suaviza
                pose_records, _ = pose_extractor.processar_frame(
                    frame_img=frame,
                    detections_yolo=yolo_dets,
                    frame_idx=frame_idx,
                    desenhar_no_frame=False
                )
                
                t2 = time.time()
                t_pose_acc += (t2 - t1)
                
                # Atualiza cache
                last_pose_records = pose_records
                
                # Salva para JSON (apenas frames processados ou todos?)
                # Normalmente JSON de dataset deve ter apenas frames amostrados (10fps).
                # Se salvarmos todos interpolados, o dataset fica enorme e redundante.
                # Mantemos logs apenas dos frames de IA.
                registros_totais.extend(pose_records)
                
            else:
                # Frame Pulado: (Custo Zero de IA)
                # Não atualizamos 'last_pose_records', apenas reutilizamos para desenho (Repetição/Freeze)
                pass

            # ------------------ Renderização 30FPS -----------------------
            # Desenha os esqueletos (seja do frame atual ou do cache) no frame atual
            # Isso garante fluidez visual (caixas acompanham, mesmo que "travadas" por 2 frames)
            viz_frame = frame.copy()
            
            for rec in last_pose_records:
                # Extrai dados do registro cached
                pid = rec["id_persistente"]
                kps = np.array(rec["keypoints"]) # (K, 3)
                bbox = rec["bbox"] # [x1, y1, x2, y2]
                conf = rec["confidence"]
                
                # Filtra confiança baixa
                if conf < cm.MIN_POSDETECTION_CONF: continue

                # Desenha Esqueleto
                base_color = color_for_id(pid)
                viz_frame = desenhar_esqueleto_unificado(viz_frame, kps, kp_thresh=cm.POSE_CONF_MIN, base_color=base_color)

                # Desenha Box e Label (Opcional, mas user gosta)
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(viz_frame, (x1, y1), (x2, y2), base_color, 2)
                
                label = f"ID: {pid} | {conf:.2f}"
                cv2.putText(viz_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            writer.write(viz_frame)
            
            # Log de Progresso
            if frame_idx % 30 == 0:
                prog = (frame_idx / total_frames) * 100
                sys.stdout.write(f"\r{Fore.GREEN}[PROCESSAMENTO]{Fore.WHITE} Frame {frame_idx}/{total_frames} ({prog:.1f}%)")
                sys.stdout.flush()
                
            frame_idx += 1

    finally:
        cap.release()
        writer.release()
        if not USING_DEEPOCSORT and yolo_model:
            del yolo_model.tracker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    print(Fore.GREEN + f"\n\n[CONCLUIDO] Processamento Finalizado em {total_time:.2f}s")

    # ------------------ Post-Processing -----------------------
    
    # [TASK 3: Sanitizer Logic]
    if sanitizar_dados:
        print(Fore.CYAN + "[INFO] Aplicando Sanitização (Anti-Teleporte)...")
        # Sanitizer filtra saltos impossíveis de velocidade
        registros_totais = sanitizar_dados(registros_totais, threshold=150.0)
    
    # Save JSON
    json_path = json_dir / f"{video_path.stem}_{int(target_fps)}fps.json"
    with open(json_path, "w") as f:
        json.dump(registros_totais, f, indent=2)
    print(Fore.GREEN + f"[OK] JSON Salvo: {json_path.name}")
    
    # Relatório Tracking (Assume ids continuos)
    tracking_path = json_dir / f"{video_path.stem}_{int(target_fps)}fps_tracking.json"
    # Precisamos reconstruir id_map e valid ids a partir dos dados finais se quisermos relatorio preciso
    # Simplifcação: Geramos com os dados brutos
    ids_encontrados = list(set(r["id_persistente"] for r in registros_totais))
    
    # Preenche dummy helper data pra função de relatorio funcionar
    dummy_map = {i: i for i in ids_encontrados}
    
    gerar_relatorio_tracking(
        registros=registros_totais,
        id_map=dummy_map, 
        ids_validos=ids_encontrados,
        total_frames=frame_idx,
        video_name=video_path.name,
        output_path=tracking_path
    )

    return {"yolo": t_yolo_acc, "rtmpose": t_pose_acc, "total": total_time}