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
# Configuração e Detector
import neurapose_backend.config_master as cm

# Módulos Modulares Unificados
from neurapose_backend.rtmpose.extracao_pose_rtmpose import ExtratorPoseRTMPose
from neurapose_backend.nucleo.visualizacao import gerar_video_predicao
from neurapose_backend.nucleo.video_utils import normalizar_video
from neurapose_backend.nucleo.tracking_utils import gerar_relatorio_tracking
from neurapose_backend.nucleo.pipeline import executar_pipeline_extracao

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
    Processa um vídeo para gerar dataset de treinamento.
    Usa a mesma arquitetura modular do App para garantir consistência (RTMPose + Filtros V6).
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
    print(Fore.CYAN + f"[INFO] NORMALIZANDO VIDEOS PARA {cm.FPS_TARGET} FPS...")
    
    # Chama o módulo central
    norm_path, time_norm = normalizar_video(video_path, videos_dir, cm.FPS_TARGET)
    
    if not norm_path:
        print(Fore.RED + f"[SKIP] Falha na normalização de {video_path.name}")
        return {"yolo": 0, "rtmpose": 0, "total": 0}

    # 2. PIPELINE UNIFICADO (Detecção + Pose + Filtros)
    # --------------------------------------------------------
    # Inicializa Extrator
    pose_extractor = ExtratorPoseRTMPose(device=cm.DEVICE)

    registros, id_map, ids_validos, total_frames, tempos = executar_pipeline_extracao(
        video_path_norm=norm_path,
        pose_extractor=pose_extractor,
        batch_size=cm.YOLO_BATCH_SIZE,
        verbose=True
    )
    
    time_yolo = tempos["yolo"]
    time_rtmpose = tempos["rtmpose"]
    frame_idx = total_frames + 1

    if not registros:
        return {"yolo": time_yolo, "rtmpose": time_rtmpose, "total": 0}
    

    # 7. RELATÓRIO FINAL (MOVIDO PARA ANTES DA RENDERIZAÇÃO)
    # --------------------------------------------------------
    time_total = time_yolo + time_rtmpose + time_norm
    
    print(Fore.WHITE + "="*60)
    print(Fore.WHITE + f"TEMPO DE PROCESSAMENTO - {video_path.name}")
    print(Fore.WHITE + "="*60)
    print(Fore.WHITE + f"{f'Normalização video {int(cm.FPS_TARGET)} FPS':<45} {time_norm:>10.2f} seg")
    print(Fore.WHITE + f"{'YOLO + BoTSORT-Custom + OSNet':<45} {time_yolo:>10.2f} seg")
    print(Fore.WHITE + f"{'RTMPose':<45} {time_rtmpose:>10.2f} seg")
    print(Fore.WHITE + "-"*60)
    print(Fore.WHITE + f"{'TOTAL':<45} {time_total:>10.2f} seg")
    print(Fore.WHITE + "="*60 + "\n")

    # 5. EXPORTAÇÃO (JSON + VIDEO)
    # --------------------------------------------------------
    json_path = json_dir / f"{video_path.stem}_{int(cm.FPS_TARGET)}fps.json"
    
    # Salva JSON Limpo
    with open(json_path, "w") as f:
        json.dump(registros, f, indent=2)
    print(Fore.GREEN + "[OK]" + Fore.WHITE + f" JSON FINAL SALVO: {json_path.name}")

    # Gera Vídeo de Predição (Visualização Universal)
    out_video_path = preds_dir / f"{video_path.stem}_{int(cm.FPS_TARGET)}fps_pose.mp4"
    
    if registros:
        # print(Fore.CYAN + f"[INFO] Gerando vídeo visualização...")
        print(Fore.CYAN + f"[INFO] RENDERIZANDO VÍDEO: {out_video_path.name}...")
        gerar_video_predicao(
            video_path=norm_path,
            registros=registros,
            video_out_path=out_video_path,
            show_preview=show,
            modelo_nome="PRE-PROC"
        )
    
    tracking_path = json_dir / f"{video_path.stem}_{int(cm.FPS_TARGET)}fps_tracking.json"
    
    gerar_relatorio_tracking(
        registros=registros,
        id_map=id_map,
        ids_validos=ids_validos,
        total_frames=total_frames,
        video_name=video_path.name,
        output_path=tracking_path
    )
    
    return {"yolo": time_yolo, "rtmpose": time_rtmpose, "total": time_total}
