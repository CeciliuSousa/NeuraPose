# neurapose_backend/nucleo/video_utils.py

import cv2
import sys
import time
import shutil
import subprocess
from pathlib import Path
from colorama import Fore
import neurapose_backend.config_master as cm

# Importação do estado global
try:
    from neurapose_backend.globals.state import state
except ImportError:
    state = None

def _normalizar_com_nvenc(input_path: Path, output_path: Path, target_fps: float) -> bool:
    if state and state.stop_requested:
        return False
        
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-i", str(input_path),
            "-c:v", "h264_nvenc",
            "-preset", cm.NVENC_PRESET,
            "-r", str(int(target_fps)),
            "-c:a", "copy",
            "-loglevel", "error",
            str(output_path)
        ]
        
        # Use Popen instead of run to allow interruption
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        while True:
            # Check if process finished
            ret = process.poll()
            if ret is not None:
                return ret == 0
                
            # Check stop request
            if state and state.stop_requested:
                print(Fore.YELLOW + "[STOP] Normalização (NVENC) interrompida pelo usuário.")
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                # Clean up partial file
                if output_path.exists():
                    try:
                        os.remove(output_path)
                    except: pass
                return False
                
            time.sleep(0.5)
        
    except Exception as e:
        print(Fore.RED + f"[ERRO] Falha NVENC: {e}")
        return False

def _normalizar_com_opencv(input_path: Path, output_path: Path, target_fps: float) -> bool:
    if state and state.stop_requested:
        return False

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened(): return False

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (W, H))
    
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (W, H))
        
    if not writer.isOpened():
        cap.release()
        return False

    frame_idx = 0
    t0 = time.time()
    last_p = -1
    
    while True:
        if state and state.stop_requested:
            print(Fore.YELLOW + "[STOP] Normalização (OpenCV) interrompida pelo usuário.")
            break
            
        ret, frame = cap.read()
        if not ret: break
        writer.write(frame)
        
        frame_idx += 1
        p = int((frame_idx / (total_frames or 1)) * 100)
        if p != last_p and p % 10 == 0:
            sys.stdout.write(f"\r{Fore.YELLOW}[NORMALIZAÇÃO] {p}%")
            sys.stdout.flush()
            last_p = p

    sys.stdout.write("\n")
    cap.release()
    writer.release()
    
    # Se interrompido, apaga arquivo parcial
    if state and state.stop_requested:
        if output_path.exists():
            try:
                os.remove(output_path)
            except: pass
        return False
        
    return True

def normalizar_video(input_path: Path, output_dir: Path, target_fps: float = None, tolerancia: float = 0.5) -> (Path, float):
    if target_fps is None: 
        target_fps = cm.INPUT_NORM_FPS
        
    start_time = time.time()
    # out_name = f"{input_path.stem}_{int(target_fps)}fps.mp4"
    out_name = f"{input_path.stem}.mp4"
    output_path = output_dir / out_name
    
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(Fore.RED + f"[ERRO] Falha ao abrir video: {input_path}")
        return None, 0.0

    fps_atual = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # ============================================================
    # SKIP INTELIGENTE
    # ============================================================
    if abs(fps_atual - target_fps) <= tolerancia:
        if input_path.resolve() != output_path.resolve():
            shutil.copy(str(input_path), str(output_path))
        else:
            output_path = input_path
            
        print(Fore.YELLOW + f"[NORMALIZAÇÃO] VIDEO NORMALIZADO EM {int(fps_atual)} FPS.")
        return output_path, (time.time() - start_time)
    
    # Inicia Processo
    print(Fore.CYAN + f"[INFO] NORMALIZANDO VIDEOS PARA {int(target_fps)} FPS...")
    
    # 2. NVENC
    if cm.USE_NVENC:
        if _normalizar_com_nvenc(input_path, output_path, target_fps):
            print(Fore.GREEN + "[OK] NORMALIZAÇÃO CONCLUÍDA!")
            return output_path, (time.time() - start_time)
    
    # 3. OpenCV
    if _normalizar_com_opencv(input_path, output_path, target_fps):
        print(Fore.GREEN + "[OK] NORMALIZAÇÃO CONCLUÍDA!")
        return output_path, (time.time() - start_time)
    
    print(Fore.RED + "[ERRO] Falha geral na normalização.")
    return None, 0.0
