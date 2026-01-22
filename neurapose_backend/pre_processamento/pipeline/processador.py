# ==============================================================
# neurapose_backend/pre_processamento/pipeline/processador.py
# ==============================================================
# Pipeline UNIFICADO E MODULARIZADO (Pré-processamento)
# ==============================================================

import sys
import cv2
import json
import time
import numpy as np
import torch
from pathlib import Path
from colorama import Fore

# Configuração e Detector
import neurapose_backend.config_master as cm
from neurapose_backend.detector.yolo_detector import yolo_detector_botsort

# Módulos Modulares Unificados
from neurapose_backend.rtmpose.extracao_pose_rtmpose import ExtratorPoseRTMPose
from neurapose_backend.nucleo.filtros import filtrar_ids_validos_v6, filtrar_ghosting_v5
from neurapose_backend.nucleo.visualizacao import gerar_video_predicao
from neurapose_backend.nucleo.video_utils import normalizar_video

try:
    from neurapose_backend.globals.state import state as state_notifier
except:
    state_notifier = None

# Carregar config do usuário se disponível
try:
    from neurapose_backend.app.user_config_manager import UserConfigManager
    user_config = UserConfigManager.load_config()
    for k, v in user_config.items():
        if hasattr(cm, k):
            setattr(cm, k, v)
except Exception as e:
    print(Fore.YELLOW + f"[CONFIG] Falha ao carregar configurações do usuário: {e}")


def processar_video(video_path: Path, out_root: Path, show=False):
    """
    Processa um vídeo para gerar dataset de treinamento.
    Usa a mesma arquitetura modular do App para garantir consistência (RTMPose + Filtros V6).
    
    Args:
        video_path: Path do vídeo original.
        sess, input_name: (Legados/Opcionais) - O Extrator carrega sua própria sessão.
        out_root: Diretório raiz de saída.
        show: Boolean para mostrar preview (não usado em batch mode geralmente).
    """

    # ------------------ Diretorios -----------------------
    videos_dir = out_root / "videos"
    preds_dir = out_root / "predicoes"
    json_dir = out_root / "jsons"

    videos_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)



    # 1. NORMALIZAÇÃO DE VÍDEO (FPS) - MODULARIZADO
    # --------------------------------------------------------
    print(Fore.CYAN + f"[INFO] Normalizando video para {cm.FPS_TARGET} FPS...")
    
    # Chama o módulo central
    norm_path, time_norm = normalizar_video(video_path, videos_dir, cm.FPS_TARGET)
    
    if not norm_path:
        print(Fore.RED + f"[SKIP] Falha na normalização de {video_path.name}")
        return {"yolo": 0, "rtmpose": 0, "total": 0}




    # 2. DETECÇÃO (YOLO + BoTSORT)
    # --------------------------------------------------------
    print(Fore.CYAN + f"[INFO] Iniciando deteccao YOLO + BoTSORT...")
    sys.stdout.flush()
    
    time_yolo_start = time.time()
    res_list = yolo_detector_botsort(videos_dir=norm_path, batch_size=cm.YOLO_BATCH_SIZE)
    time_yolo = time.time() - time_yolo_start

    print(Fore.GREEN + f"[OK] Deteccao concluida. Processando resultados.")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not res_list:
        print(Fore.RED + "[ERRO] Nenhuma detecção retornada pelo YOLO.")
        return {"yolo": time_yolo, "rtmpose": 0, "total": 0}

    res = res_list[0]
    results = res["results"]
    id_map = res.get("id_map", {})


    # 3. EXTRAÇÃO DE POSE (RTMPose Modular)
    # --------------------------------------------------------
    print(Fore.CYAN + f"[INFO] Iniciando inferencia RTMPos...")
    sys.stdout.flush()
    time_rtmpose_start = time.time()

    # Instancia Extrator Unificado
    pose_extractor = ExtratorPoseRTMPose(device=cm.DEVICE)
    
    registros = []
    
    cap = cv2.VideoCapture(str(norm_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 1
    last_progress = 0

    while True:
        if state_notifier is not None and state_notifier.stop_requested:
            print(Fore.YELLOW + "[STOP] Interrompido pelo usuário.")
            break
            
        ok, frame = cap.read()
        if not ok: break

        # Detecções do frame atual
        dets = None
        if frame_idx <= len(results):
            dets_data = results[frame_idx-1]["boxes"]
            # Envolve em objeto mock simples se necessário, ou adapta Extrator
            # O Extrator espera algo com .xyxy, .conf, .id OU lista crua.
            # O yolo_detector retorna numpy array [N, 6+] (xyxy, conf, cls, id...)
            dets = dets_data # Passa o array direto, o Extrator deve lidar (já tem fallback)
            
        # Processa Frame (Inferencia + Suavização)
        recs_frame, _ = pose_extractor.processar_frame(
            frame_img=frame,
            detections_yolo=dets,
            frame_idx=frame_idx,
            id_map=id_map,
            desenhar_no_frame=False
        )
        
        registros.extend(recs_frame)
        
        # Progresso
        progress = int((frame_idx / total_frames) * 100)
        if progress >= last_progress + 10:
            print(Fore.CYAN + f"[PROGRESSO] {progress}% ({frame_idx}/{total_frames})")
            last_progress = progress
            
        frame_idx += 1

    cap.release()
    time_rtmpose = time.time() - time_rtmpose_start
    print(Fore.GREEN + f"[OK] Inferencia RTMPose concluida. {len(registros)} poses extraídas.")


    # 4. FILTRAGEM (Ghosting + V6)
    # --------------------------------------------------------
    
    # A) Anti-Ghosting (V5)
    registros = filtrar_ghosting_v5(registros, iou_thresh=0.8)
    
    # B) Limpeza Inteligente (V6)
    # Filtra e retorna lista de IDs válidos
    ids_validos = filtrar_ids_validos_v6(
        registros=registros,
        min_frames=cm.MIN_FRAMES_PER_ID, # 30
        min_dist=50.0,
        verbose=True
    )
    
    # Remove registros inválidos
    registros = [r for r in registros if r["id_persistente"] in ids_validos]
    
    if not registros:
        print(Fore.RED + "[AVISO] Todos os IDs foram filtrados!")
    

    # 5. EXPORTAÇÃO (JSON + VIDEO)
    # --------------------------------------------------------
    json_path = json_dir / f"{video_path.stem}_{int(cm.FPS_TARGET)}fps.json"
    
    # Salva JSON Limpo
    with open(json_path, "w") as f:
        json.dump(registros, f, indent=2)
    print(Fore.GREEN + f"[OK] JSON Final salvo: {json_path.name}")

    # Gera Vídeo de Predição (Visualização Universal)
    out_video_path = preds_dir / f"{video_path.stem}_{int(cm.FPS_TARGET)}fps_pose.mp4"
    
    if registros:
        print(Fore.CYAN + f"[INFO] Gerando vídeo visualização...")
        gerar_video_predicao(
            video_path=norm_path,
            registros=registros,
            video_out_path=out_video_path,
            show_preview=show,
            modelo_nome="PRE-PROC"
        )
    
    
    # 6. TRACKING JSON (Relatório Final)
    # --------------------------------------------------------
    id_map_limpo = {str(k): int(v) for k, v in id_map.items() if v in ids_validos}
    
    tracking_analysis = {
        "video": video_path.name,
        "total_frames": frame_idx - 1,
        "id_map": id_map_limpo,
        "tracking_by_frame": {}
    }
    
    for reg in registros:
        f_id = reg["frame"]
        if f_id not in tracking_analysis["tracking_by_frame"]:
            tracking_analysis["tracking_by_frame"][f_id] = []
        
        tracking_analysis["tracking_by_frame"][f_id].append({
            "botsort_id": reg["botsort_id"],
            "id_persistente": reg["id_persistente"],
            "bbox": reg["bbox"],
            "confidence": reg["confidence"]
        })
        
    tracking_path = json_dir / f"{video_path.stem}_{int(cm.FPS_TARGET)}fps_tracking.json"
    with open(tracking_path, "w", encoding="utf-8") as f:
        json.dump(tracking_analysis, f, indent=2, ensure_ascii=False)


    # 7. RELATÓRIO FINAL
    # --------------------------------------------------------
    time_total = time_yolo + time_rtmpose + time_norm
    
    print(Fore.CYAN + "\n" + "="*60)
    print(Fore.CYAN + f"  TEMPOS DE PROCESSAMENTO - {video_path.name}")
    print(Fore.CYAN + "="*60)
    print(Fore.YELLOW + f"  {f'Normalização video {int(cm.FPS_TARGET)} FPS':<40} {time_norm:>10.2f} seg")
    print(Fore.YELLOW + f"  {'YOLO + BoTSORT + OSNet':<40} {time_yolo:>10.2f} seg")
    print(Fore.YELLOW + f"  {'RTMPose':<40} {time_rtmpose:>10.2f} seg")
    print(Fore.WHITE + "-"*60)
    print(Fore.GREEN + f"  {'TOTAL':<40} {time_total:>10.2f} seg")
    print(Fore.CYAN + "="*60 + "\n")
    
    return {"yolo": time_yolo, "rtmpose": time_rtmpose, "total": time_total}
