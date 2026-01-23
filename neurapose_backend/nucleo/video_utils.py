# ==============================================================
# neurapose_backend/nucleo/video_utils.py
# ==============================================================
# Módulo de utilitários rápidos para manipulação de vídeo
# Otimizado com suporte a NVENC (GPU encoder)
# ==============================================================

import cv2
import sys
import time
import shutil
import subprocess
from pathlib import Path
from colorama import Fore
import neurapose_backend.config_master as cm


def _normalizar_com_nvenc(input_path: Path, output_path: Path, target_fps: float) -> bool:
    """
    Normaliza vídeo usando FFmpeg com NVENC (GPU encoder).
    
    Returns:
        True se sucesso, False se falhou.
    """
    try:
        cmd = [
            "ffmpeg",
            "-y",                           # Sobrescreve sem perguntar
            "-hwaccel", "cuda",             # Acelera decodificação com CUDA
            "-hwaccel_output_format", "cuda",
            "-i", str(input_path),
            "-c:v", "h264_nvenc",           # Encoder GPU NVIDIA
            "-preset", cm.NVENC_PRESET,     # p1 (rápido) a p7 (qualidade)
            "-r", str(int(target_fps)),     # FPS alvo
            "-c:a", "copy",                 # Copia áudio sem re-encodar
            "-loglevel", "error",           # Silencia output
            str(output_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutos timeout
        )
        
        return result.returncode == 0
        
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(Fore.YELLOW + f"[NVENC] Falha: {str(e)[:50]}...")
        return False


def _normalizar_com_opencv(input_path: Path, output_path: Path, target_fps: float) -> bool:
    """
    Normaliza vídeo usando OpenCV (fallback CPU).
    
    Returns:
        True se sucesso, False se falhou.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return False

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Tenta codec H.264 (avc1), fallback para mp4v
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (W, H))
    
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (W, H))
        
    if not writer.isOpened():
        cap.release()
        return False

    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        writer.write(frame)

    cap.release()
    writer.release()
    return True


def normalizar_video(input_path: Path, output_dir: Path, target_fps: float = None, tolerancia: float = 0.5) -> (Path, float):
    """
    Normaliza o frame rate de um vídeo para o FPS alvo.
    
    INTELIGÊNCIA: 
    - Se o vídeo JÁ ESTÁ no FPS alvo, pula a normalização
    - Se USE_NVENC=True, usa FFmpeg com GPU NVIDIA (muito mais rápido)
    - Fallback para OpenCV se NVENC falhar
    
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
    cap.release()
    
    # ============================================================
    # SKIP INTELIGENTE: Se já está no FPS correto, apenas copia
    # ============================================================
    if abs(fps_atual - target_fps) <= tolerancia:
        # Evita sobrescrever o mesmo arquivo
        if input_path.resolve() != output_path.resolve():
            print(Fore.GREEN + f"[SKIP] Vídeo já está em {fps_atual:.1f} FPS. Copiando...")
            shutil.copy(str(input_path), str(output_path))
        else:
            print(Fore.GREEN + f"[SKIP] Vídeo já está em {fps_atual:.1f} FPS. Usando arquivo existente.")
            output_path = input_path
            
        duration = time.time() - start_time
        return output_path, duration
    
    # ============================================================
    # NORMALIZAÇÃO COM NVENC (GPU) - Muito mais rápido
    # ============================================================
    sucesso = False
    
    if cm.USE_NVENC:
        print(Fore.CYAN + f"[NVENC] Convertendo {input_path.name} de {fps_atual:.1f} para {target_fps} FPS (GPU)...")
        sys.stdout.flush()
        sucesso = _normalizar_com_nvenc(input_path, output_path, target_fps)
        
        if sucesso:
            duration = time.time() - start_time
            print(Fore.GREEN + f"[OK] Normalizado com NVENC em {duration:.2f}s: {output_path.name}")
            return output_path, duration
        else:
            print(Fore.YELLOW + f"[FALLBACK] NVENC falhou, usando OpenCV...")
    
    # ============================================================
    # FALLBACK: OpenCV (CPU)
    # ============================================================
    print(Fore.CYAN + f"[OPENCV] Convertendo {input_path.name} de {fps_atual:.1f} para {target_fps} FPS (CPU)...")
    sys.stdout.flush()
    
    sucesso = _normalizar_com_opencv(input_path, output_path, target_fps)
    
    if not sucesso:
        print(Fore.RED + f"[FATAL] Falha ao normalizar vídeo.")
        return None, 0.0
    
    duration = time.time() - start_time
    print(Fore.GREEN + f"[OK] Normalizado com OpenCV em {duration:.2f}s: {output_path.name}")
    
    return output_path, duration


