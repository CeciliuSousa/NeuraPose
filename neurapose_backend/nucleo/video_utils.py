# ==============================================================
# neurapose_backend/nucleo/video_utils.py
# ==============================================================
# Módulo de utilitários rápidos para manipulação de vídeo
# ==============================================================

import cv2
import sys
import time
from pathlib import Path
from colorama import Fore
import neurapose_backend.config_master as cm

def normalizar_video(input_path: Path, output_dir: Path, target_fps: float = None) -> (Path, float):
    """
    Normaliza o frame rate de um vídeo para o FPS alvo.
    
    Args:
        input_path: Caminho do vídeo original.
        output_dir: Diretório onde salvar o vídeo normalizado.
        target_fps: FPS desejado (default: cm.FPS_TARGET).
        
    Returns:
        (caminho_video_normalizado, tempo_gasto_segundos)
    """
    if target_fps is None:
        target_fps = cm.FPS_TARGET
        
    start_time = time.time()
    
    # Define nome de saída: video_30fps.mp4
    out_name = f"{input_path.stem}_{int(target_fps)}fps.mp4"
    output_path = output_dir / out_name
    
    # Se já existe (cache simples), podemos pular? 
    # Por segurança, vamos processar sempre ou confiar no usuario.
    # Aqui vamos reprocessar para garantir.
    
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(Fore.RED + f"[ERRO] Não foi possível abrir o vídeo: {input_path}")
        return None, 0.0

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(Fore.CYAN + f"[NORMALIZACAO] Convertendo {input_path.name} para {target_fps} FPS...")
    sys.stdout.flush()

    # Tenta codec H.264 (avc1), fallback para mp4v
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (W, H))
    
    if not writer.isOpened():
        print(Fore.YELLOW + f"[AVISO] Codec 'avc1' falhou. Tentando fallback 'mp4v'...")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (W, H))
        
    if not writer.isOpened():
        print(Fore.RED + f"[FATAL] Falha ai criar arquivo de vídeo normalizado.")
        cap.release()
        return None, 0.0

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        writer.write(frame)
        count += 1

    cap.release()
    writer.release()
    
    duration = time.time() - start_time
    print(Fore.GREEN + f"[OK] Vídeo normalizado salvo em {duration:.2f}s: {output_path.name}")
    
    return output_path, duration
