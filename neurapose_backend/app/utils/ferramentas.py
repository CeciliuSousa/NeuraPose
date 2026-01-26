# neurapose-backend/app/utils/ferramentas.py
# Utilitarios para o App: checagem de recursos e banners.

import torch
import platform
import psutil
from pathlib import Path
from yt_dlp import YoutubeDL
from colorama import Fore
import neurapose_backend.config_master as cm

from neurapose_backend.app.configuracao.config import (
    BEST_MODEL_PATH, LABELS_TEST_PATH, DATASET_DIR, MODEL_NAME, DATASET_NAME
)

def status_str(ok: bool):
    return Fore.GREEN + "[OK]" if ok else Fore.RED + "[ERRO]"

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

def verificar_recursos():
    modelo_default = f"{MODEL_NAME}-{DATASET_NAME}"
    return {
        "yolo": cm.YOLO_PATH.exists(),
        "osnet": cm.OSNET_PATH.exists(),
        "rtmpose": cm.RTMPOSE_PATH.exists(),
        "modelo_temporal": BEST_MODEL_PATH.exists(),
        "labels": LABELS_TEST_PATH.exists(),
        "dataset": DATASET_DIR.exists(),
        "modelo_temporal_nome": modelo_default, 
        "dataset_path": str(DATASET_DIR),
    }


def imprimir_banner(checks):
    sys_info = get_system_info()
    
    print(Fore.WHITE + "\n" + "="*62)
    print(Fore.WHITE + "TESTE DE MODELO — NEURAPOSE")
    print(Fore.WHITE + "="*62)
    
    # Ferramentas
    print(Fore.WHITE + f"YOLO              : {status_str(checks['yolo'])} {Fore.WHITE}{cm.YOLO_PATH.name}")
    print(Fore.WHITE + f"TRACKER           : {status_str(True)} {Fore.WHITE}{cm.TRACKER_NAME}-Custom")
    print(Fore.WHITE + f"OSNet ReID        : {status_str(checks['osnet'])} {Fore.WHITE}{cm.OSNET_PATH.name[:25]}...")
    print(Fore.WHITE + f"RTMPose-l         : {status_str(checks['rtmpose'])} {Fore.WHITE}rtmpose.../{cm.RTMPOSE_PATH.name}")
    
    mod_temp = "Temporal Fusion Transformer" if cm.TEMPORAL_MODEL == "tft" else "LSTM / BiLSTM"
    print(Fore.WHITE + f"Modelo Temporal   : {status_str(checks['modelo_temporal'])} {Fore.WHITE}{mod_temp}")
    
    print(Fore.WHITE + "-"*62)
    
    # Hardware
    gpu_color = status_str(sys_info['gpu_ok'])
    print(Fore.WHITE + f"GPU detectada     : {gpu_color} {Fore.WHITE}{sys_info['gpu']}")
    print(Fore.WHITE + f"VRAM              : {gpu_color} {Fore.WHITE}{sys_info['vram']}")
    print(Fore.WHITE + f"RAM               : {status_str(True)} {Fore.WHITE}{sys_info['ram']}")

    print(Fore.WHITE + "="*62 + "\n")


def baixar_video_ytdlp(url: str, pasta_saida: Path) -> Path:
    """
    Baixa um vídeo do YouTube usando yt-dlp.
    Retorna o caminho do arquivo baixado.
    """
    pasta_saida.mkdir(parents=True, exist_ok=True)
    print(Fore.CYAN + f"Baixando vídeo do YouTube...\n {url}")
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "outtmpl": str(pasta_saida / "%(title)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": False
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        output_file = pasta_saida / f"{info['title']}.mp4"
    print(Fore.GREEN + f"Download concluído: {output_file}")
    return output_file