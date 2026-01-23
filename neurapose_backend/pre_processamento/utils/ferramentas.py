# ==============================================================
# pre_processamento/utils/ferramentas.py
# ==============================================================
# Funções utilitárias gerais: download de vídeos, carregamento
# de modelos ONNX e formatação de logs.

import torch
import onnxruntime as ort
from pathlib import Path
from yt_dlp import YoutubeDL
from colorama import Fore

# Importa do config_master (absoluto)
import neurapose_backend.config_master as cm



def status_str(ok: bool):
    """Retorna string colorida [OK] ou [ERRO]."""
    return Fore.GREEN + "[OK]" if ok else Fore.RED + "[ERRO]"


def imprimir_banner(onnx_path: Path):
    """Imprime banner informativo sobre o ambiente e modelos."""
    print("\n===============================================================")
    print("PRÉ-PROCESSAMENTO — NEURAPOSE AI")
    print("===============================================================")

    # YOLO (usa config_master)
    yolopath = cm.ROOT / "detector" / "modelos" / cm.YOLO_MODEL
    yolo_name = cm.YOLO_MODEL.replace('.pt', '')
    print(f"YOLO                  : {status_str(yolopath.exists())} {yolo_name}")

    # Tracker
    print(f"TRACKER               : {status_str(True)} {cm.TRACKER_NAME}")

    # OSNet (usa config_master)
    print(f"OSNet ReID            : {status_str(cm.OSNET_PATH.exists())} {cm.OSNET_PATH.name}")

    # RTMPose
    print(
        f"RTMPose-l             : {status_str(onnx_path.exists())} "
        f"{onnx_path.parent.name}/{onnx_path.name}"
    )

    print("---------------------------------------------------------------")
    # Dataset
    # print(f"Dataset               : {DATASET_NAME}")

    # GPU Info
    if torch.cuda.is_available():
        print(Fore.GREEN + f"GPU detectada         : {torch.cuda.get_device_name(0)}")
    else:
        print(Fore.YELLOW + "Dispositivo           : CPU (sem GPU)")
    
    print("===============================================================\n")