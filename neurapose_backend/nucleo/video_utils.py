# ==============================================================
# neurapose_backend/nucleo/video_utils.py
# ==============================================================
# Módulo de utilitários rápidos para manipulação de vídeo
# ==============================================================

import cv2
import sys
import time
import shutil
from pathlib import Path
from colorama import Fore
import neurapose_backend.config_master as cm


def normalizar_video(input_path: Path, output_dir: Path, target_fps: float = None, tolerancia: float = 0.5) -> (Path, float):
    """
    Normaliza o frame rate de um vídeo para o FPS alvo.
    
    INTELIGÊNCIA: Se o vídeo JÁ ESTÁ no FPS alvo (dentro da tolerância),
    pula a normalização e apenas copia o arquivo para evitar re-encoding
    e alterações de pixels.
    
    Args:
        input_path: Caminho do vídeo original.
        output_dir: Diretório onde salvar o vídeo normalizado.
        target_fps: FPS desejado (default: cm.FPS_TARGET).
        tolerancia: Margem de FPS aceita sem re-encoding (default: 0.5).
        
    Returns:
        (caminho_video_normalizado, tempo_gasto_segundos)
    """
    if target_fps is None:
        target_fps = cm.FPS_TARGET
        
    start_time = time.time()
    
    # Define nome de saída: video_30fps.mp4
    out_name = f"{input_path.stem}_{int(target_fps)}fps.mp4"
    output_path = output_dir / out_name
    
    # Abre o vídeo para verificar FPS atual
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(Fore.RED + f"[ERRO] Não foi possível abrir o vídeo: {input_path}")
        return None, 0.0

    fps_atual = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # ============================================================
    # SKIP INTELIGENTE: Se já está no FPS correto, apenas copia
    # ============================================================
    if abs(fps_atual - target_fps) <= tolerancia:
        cap.release()
        
        # Evita sobrescrever o mesmo arquivo
        if input_path.resolve() != output_path.resolve():
            print(Fore.GREEN + f"[SKIP] Vídeo já está em {fps_atual:.1f} FPS (alvo: {target_fps}). Copiando...")
            shutil.copy(str(input_path), str(output_path))
        else:
            print(Fore.GREEN + f"[SKIP] Vídeo já está em {fps_atual:.1f} FPS. Usando arquivo existente.")
            output_path = input_path  # Usa o próprio arquivo
            
        duration = time.time() - start_time
        print(Fore.GREEN + f"[OK] Normalização pulada em {duration:.2f}s: {output_path.name}")
        return output_path, duration
    
    # ============================================================
    # NORMALIZAÇÃO: Re-encoda o vídeo para o FPS alvo
    # ============================================================
    print(Fore.CYAN + f"[NORMALIZACAO] Convertendo {input_path.name} de {fps_atual:.1f} para {target_fps} FPS...")
    sys.stdout.flush()

    # Tenta codec H.264 (avc1), fallback para mp4v
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (W, H))
    
    if not writer.isOpened():
        print(Fore.YELLOW + f"[AVISO] Codec 'avc1' falhou. Tentando fallback 'mp4v'...")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (W, H))
        
    if not writer.isOpened():
        print(Fore.RED + f"[FATAL] Falha ao criar arquivo de vídeo normalizado.")
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

