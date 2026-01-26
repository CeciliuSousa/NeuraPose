# ==============================================================
# neurapose_backend/nucleo/pipeline.py
# ==============================================================
# Módulo que unifica a lógica de Detecção (YOLO+BoTSORT) + 
# Extração de Pose (RTMPose) + Filtragem (Ghosting/V6).
# Garante consistência total entre App e Pré-processamento.
# ==============================================================

import time
import cv2
import torch
import sys
from pathlib import Path
from colorama import Fore

import neurapose_backend.config_master as cm
from neurapose_backend.detector.yolo_detector import yolo_detector_botsort as yolo_detector
from neurapose_backend.nucleo.filtros import filtrar_ids_validos_v6, filtrar_ghosting_v5

# Import opcional para feedback de estado no App
try:
    from neurapose_backend.globals.state import state
except ImportError:
    state = None
import random
import numpy as np

def executar_pipeline_extracao(
    video_path_norm: Path,
    pose_extractor,
    batch_size: int = None,
    verbose: bool = True
):
    """
    Executa o núcleo do pipeline:
    1. Detecção e Tracking (YOLO + BoTSORT)
    2. Extração de Pose (RTMPose)
    3. Filtragem de Ghosting
    4. Filtragem de IDs Válidos

    Args:
        video_path_norm: Caminho do vídeo JÁ NORMALIZADO.
        pose_extractor: Instância de ExtratorPoseRTMPose (já inicializada).
        batch_size: Tamanho do batch para YOLO (default: cm.YOLO_BATCH_SIZE).
        verbose: Se True, imprime logs coloridos.

    Returns:
        Um tupla contendo:
        - registros_finais (List[dict]): Lista de registros de pose filtrados.
        - id_map_full (dict): Mapa completo de IDs original do tracker.
        - ids_validos (List[int]): Lista de IDs que passaram no filtro.
        - total_frames (int): Total de frames processados.
        - tempos (dict): Dicionário com tempos de execução ('yolo', 'rtmpose').
    """
    
    # Fixa seeds para determinismo (Garante paridade App x Processamento)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    
    if batch_size is None:
        batch_size = cm.YOLO_BATCH_SIZE

    tempos = {"yolo": 0.0, "rtmpose": 0.0}

    # 1. DETECTOR (YOLO + BoTSORT)
    # --------------------------------------------------------
    if verbose:
        # print(Fore.CYAN + f"[INFO] Iniciando deteccao YOLO + BoTSORT + OSNet")
        # print(Fore.CYAN + f"[INFO] Iniciando deteccao YOLO + BoTSORT (Batch {batch_size})...")
        pass

    
    t0 = time.time()
    # Executa YOLO
    resultados_list = yolo_detector(videos_dir=video_path_norm, batch_size=batch_size)
    t1 = time.time()
    tempos["yolo"] = t1 - t0

    if not resultados_list:
        if verbose: print(Fore.RED + "[ERRO] Nenhuma detecção retornada pelo YOLO.")
        return [], {}, [], 0, tempos

    res = resultados_list[0]
    results_yolo = res["results"] # Lista de dicts por frame
    id_map_full = res.get("id_map", {})

    # 2. EXTRAÇÃO DE POSE (RTMPose Modular)
    # --------------------------------------------------------
    if verbose:
        print(Fore.YELLOW + f"[RTMPOSE] PROCESSANDO VIDEO...")
    
    t0 = time.time()
    records = []
    
    # Usa VideoReaderAsync para pre-fetch de frames (quando habilitado)
    if cm.USE_PREFETCH:
        from neurapose_backend.nucleo.video_reader import VideoReaderAsync
        
        with VideoReaderAsync(video_path_norm) as reader:
            total_frames = reader.total_frames
            last_progress = 0
            
            for frame_idx, frame in reader:
                # Verifica parada solicitada (App)
                if state and state.stop_requested:
                    if verbose: print(Fore.YELLOW + "[STOP] Interrompido pelo usuário.")
                    break
                
                # Recupera boxes do frame atual
                dets = None
                if frame_idx <= len(results_yolo):
                    dets = results_yolo[frame_idx-1].get("boxes")
                
                # Processa frame com o Extrator Unificado
                frame_regs, _ = pose_extractor.processar_frame(
                    frame_img=frame,
                    detections_yolo=dets,
                    frame_idx=frame_idx,
                    id_map=id_map_full,
                    desenhar_no_frame=False
                )
                
                records.extend(frame_regs)
                
                # Log de progresso (a cada 10%)
                if verbose:
                    progress = int((frame_idx / (total_frames or 1)) * 100)
                    if progress >= last_progress + 10:
                        print(Fore.CYAN + f"[RTMPose] Progresso: {progress}% ({frame_idx}/{total_frames})")
                        sys.stdout.flush()
                        last_progress = progress
                        
            final_frame_idx = frame_idx if 'frame_idx' in dir() else 0
    else:
        # Fallback: leitura síncrona com OpenCV
        cap = cv2.VideoCapture(str(video_path_norm))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 1
        last_progress = 0

        while True:
            if state and state.stop_requested:
                if verbose: print(Fore.YELLOW + "[STOP] Interrompido pelo usuário.")
                break
                
            ok, frame = cap.read()
            if not ok: break
            
            dets = None
            if frame_idx <= len(results_yolo):
                dets = results_yolo[frame_idx-1].get("boxes") 
            
            frame_regs, _ = pose_extractor.processar_frame(
                frame_img=frame,
                detections_yolo=dets,
                frame_idx=frame_idx,
                id_map=id_map_full,
                desenhar_no_frame=False
            )
            
            records.extend(frame_regs)
            
            if verbose:
                progress = int((frame_idx / (total_frames or 1)) * 100)
                if progress >= last_progress + 10:
                    print(Fore.YELLOW + f"[RTMPOSE] {progress} %")
                    sys.stdout.flush()
                    last_progress = progress

            frame_idx += 1
            
        cap.release()
        final_frame_idx = frame_idx - 1
        
    t1 = time.time()
    tempos["rtmpose"] = t1 - t0

    if not records:
        if verbose: print(Fore.RED + "[AVISO] Nenhuma pose detectada.")
        return [], id_map_full, [], final_frame_idx, tempos

    # 3. FILTRAGEM (Ghosting + V6)
    # --------------------------------------------------------
    if verbose:
        print(Fore.CYAN + f"[INFO] Filtrando Ghosting e IDs (V6)...")

    # A) Anti-Ghosting (CRÍTICO: Garante que IDs duplicados/fantasmas sejam removidos antes da validação)
    records = filtrar_ghosting_v5(records, iou_thresh=0.8)
    
    # B) Filtros de Validade (Duração, Movimento)
    ids_validos = filtrar_ids_validos_v6(
        registros=records,
        min_frames=cm.MIN_FRAMES_PER_ID, # 30
        min_dist=50.0,                   # Padrão V6
        verbose=verbose
    )
    
    # C) Filtra registros finais
    registros_finais = [r for r in records if r["id_persistente"] in ids_validos]

    if not registros_finais:
        if verbose: print(Fore.RED + "[AVISO] Todos os IDs foram removidos pelo filtro.")
    
    return registros_finais, id_map_full, ids_validos, final_frame_idx, tempos

