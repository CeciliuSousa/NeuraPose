# neurapose_backend/pre_processamento/utils/ferramentas.py
# Utils gerais: banner, status, download de videos.

import torch
import platform
import psutil
from pathlib import Path
from colorama import Fore
import neurapose_backend.config_master as cm

def status_str(ok: bool):
    return Fore.GREEN + "[OK]" if ok else Fore.RED + "[ERRO]"

def format_seconds_to_hms(seconds):
    """Formata segundos para HH:MM:SS"""
    import math
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def get_system_info():
    # RAM
    mem = psutil.virtual_memory()
    ram_total = f"{mem.total / (1024**3):.1f}GB"
    ram_avail = f"{mem.available / (1024**3):.1f}GB"
    
    # GPU / VRAM
    gpu_name = "Não detectada"
    vram_status = "N/A"
    gpu_ok = False
    
    if torch.cuda.is_available():
        gpu_ok = True
        gpu_name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        vram_total = f"{total / (1024**3):.1f}GB"
        vram_free = f"{free / (1024**3):.1f}GB"
        vram_status = f"{vram_free} / {vram_total}"

    return {
        "ram": f"{ram_avail} / {ram_total}",
        "gpu": gpu_name,
        "vram": vram_status,
        "gpu_ok": gpu_ok
    }


def imprimir_banner(onnx_path: Path):
    sys_info = get_system_info()
    
    print(Fore.WHITE + "\n" + "="*62)
    print(Fore.WHITE + "PRÉ-PROCESSAMENTO — NEURAPOSE")
    print(Fore.WHITE + "="*62)
    
    # Ferramentas
    yolopath = cm.ROOT / "detector" / "modelos" / cm.YOLO_MODEL
    print(Fore.WHITE + f"YOLO              : {status_str(yolopath.exists())} {Fore.WHITE}{cm.YOLO_MODEL}")
    print(Fore.WHITE + f"TRACKER           : {status_str(True)} {Fore.WHITE}{cm.TRACKER_NAME}-Custom")
    print(Fore.WHITE + f"OSNet ReID        : {status_str(cm.OSNET_PATH.exists())} {Fore.WHITE}{cm.OSNET_PATH.name[:25]}...")
    print(Fore.WHITE + f"RTMPose-m         : {status_str(onnx_path.exists())} {Fore.WHITE}rtmpose.../{onnx_path.name}")
    
    print(Fore.WHITE + "-"*62)
    
    # Hardware
    gpu_color = status_str(sys_info['gpu_ok'])
    print(Fore.WHITE + f"GPU detectada     : {gpu_color} {Fore.WHITE}{sys_info['gpu']}")
    print(Fore.WHITE + f"VRAM              : {gpu_color} {Fore.WHITE}{sys_info['vram']}")
    print(Fore.WHITE + f"RAM               : {status_str(True)} {Fore.WHITE}{sys_info['ram']}")
    
    print(Fore.WHITE + "="*62 + "\n")