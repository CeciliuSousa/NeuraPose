# neurapose_backend/detector/yolo_detector.py

import os
import sys
import cv2
import torch
import logging
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Importacoes do tracker
from neurapose_backend.tracker.rastreador import CustomBoTSORT, CustomReID, save_temp_tracker_yaml

from neurapose_backend.globals.state import state
from colorama import Fore

import neurapose_backend.config_master as cm

logging.getLogger("ultralytics").setLevel(logging.ERROR)
os.environ["YOLO_VERBOSE"] = "False"

from ultralytics.trackers import bot_sort

bot_sort.BOTSORT = CustomBoTSORT
bot_sort.ReID = CustomReID


# 1. Fusao de IDs
# ================================================================
def merge_tracks(track_data, gap_thresh=1.5):
    """
    track_data: dict[track_id] -> {start, end, frames, features}
    Retorna:
      merged_tracks: dict[id_original] -> {start, end, aliases}
      id_map: dict[id_qualquer] -> id_original

    Regra:
      - Nao usa embedding para decidir fusao.
      - So usa tempo (start/end):
          overlap = not (end_a < start_b or end_b < start_a)
          gap = start_b - end_a
          if not overlap and 0 < gap < gap_thresh -> funde id_b em id_a
    """
    merged_tracks = {}
    used = set()

    # Ordena por instante de inicio
    ids_sorted = sorted(track_data.keys(), key=lambda tid: track_data[tid]["start"])

    for i, id_a in enumerate(ids_sorted):
        if id_a in used:
            continue

        data_a = track_data[id_a]
        merged_tracks[id_a] = {
            "start": data_a["start"],
            "end": data_a["end"],
            "aliases": []
        }
        used.add(id_a)

        # Compara com os IDs seguintes na lista (id_b comeca depois de id_a)
        for id_b in ids_sorted[i + 1:]:
            if id_b in used:
                continue

            data_b = track_data[id_b]

            start_a = merged_tracks[id_a]["start"]
            end_a = merged_tracks[id_a]["end"]
            start_b = data_b["start"]
            end_b = data_b["end"]

            # Mesmo criterio de overlap do codigo original
            overlap = not (end_a < start_b or end_b < start_a)
            gap = start_b - end_a  # importante: b depois de a

            if not overlap and 0 < gap < gap_thresh:
                merged_tracks[id_a]["aliases"].append(id_b)
                merged_tracks[id_a]["end"] = max(end_a, end_b)
                used.add(id_b)

    # Cria o mapa id_atual -> id_persistente
    id_map = {}
    for orig, data in merged_tracks.items():
        id_map[int(orig)] = int(orig)
        for alias in data["aliases"]:
            id_map[int(alias)] = int(orig)

    return merged_tracks, id_map


# ================================================================
# 2. YOLO + BoTSORT + coleta de IDs + fusao de IDs
# ================================================================
def yolo_detector_botsort(videos_dir=None, batch_size=None):
    """
    Roda YOLOv8x + CustomBoTSORT em varios videos com Processamento em LOTE.
    Retorna lista com:
        - video
        - fps
        - track_data
        - merged_tracks
        - id_map
        - results (frame a frame)
    """

    videos_path = Path(videos_dir or (cm.ROOT / "videos"))

    if batch_size is None:
        batch_size = cm.YOLO_BATCH_SIZE

    # ================================================================
    # MODELO YOLO - Usar caminho centralizado (config_master) e baixar se nao existir
    # ================================================================
    model_path = cm.YOLO_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        model_base = cm.YOLO_MODEL.replace('.pt', '')
        try:
            temp_model = YOLO(model_base)
            temp_model.save(str(model_path))
        except Exception as e:
            if model_path.exists():
                os.remove(model_path)
            raise FileNotFoundError(f"Erro ao baixar {cm.YOLO_PATH}: {e}")
    
    # Carrega o modelo
    model = YOLO(str(model_path)).to(cm.DEVICE)

    # Lista de videos
    if videos_path.is_file():
        videos = [videos_path]
    else:
        videos = [
            v for v in videos_path.glob("*")
            if v.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]
        ]

    if not videos:
        print("[WARN] Nenhum video encontrado.")
        return []

    resultados_finais = []

    for video in videos:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Reset Tracker State for new video
        tracker = CustomBoTSORT(frame_rate=int(fps))
        model.tracker = tracker
        
        # Configure Tracker YAML
        tracker_yaml_path = save_temp_tracker_yaml()
        
        print(Fore.CYAN + f"[INFO] PROCESSANDO VIDEO COM YOLO + BOTSORT-CUSTOM + OSNET...")
        sys.stdout.flush()

        results = []
        
        # ... (Loop code not shown here, assumed handled by caller or context) ...
        # But we need to update the PRINT inside the loop if we can see it in previous views (we can).
        # Wait, I need to use 'view_file' first to be safe or rely on recent edits.
        # I saw lines 252 in previous outputs.
        # I'll replacing the start block first.

        # Note: The `replace_file_content` below targets the progress print which is inside the loop.
        # I'll try to target the progress block if I can match context.

        track_data = {}
        
        frame_idx_global = 0
        last_progress = 0
        
        # Spacer para a barra de progresso não apagar o header
        print("")

        while True:
            # Check Stop
            if state.stop_requested:
                print(Fore.YELLOW + "[STOP] Detecção interrompida.")
                break

            # 1. Carregar Batch de Frames
            frames_batch = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                frames_batch.append(frame)

            if not frames_batch:
                break # Fim do vídeo

            # 2. Inferência em Batch (Bloqueante mas muito rápida na GPU)
            # persist=True mantém o tracking entre os batches sequenciais
            batch_results = model.track(
                source=frames_batch,
                imgsz=cm.YOLO_IMGSZ,
                conf=cm.DETECTION_CONF,
                device=cm.DEVICE,
                persist=True,
                tracker=str(tracker_yaml_path),
                classes=[cm.YOLO_CLASS_PERSON],
                verbose=False,
                half=True,
                stream=False # Batch retorna lista completa imediatamente
            )

            # 3. Processar Resultados do Batch
            for i, r in enumerate(batch_results):
                current_frame_idx = frame_idx_global + i
                
                # Feedback visual para o frontend (opcional, pega o último do batch)
                if i == len(batch_results) - 1:
                     # Apenas marcamos progresso, não enviamos imagem para não travar
                     pass

                # Extrai dados leves
                frame_res = {"boxes": None}
                
                if r.boxes is not None and len(r.boxes) > 0:
                    boxes_data = r.boxes.data.cpu().numpy()
                    frame_res["boxes"] = boxes_data
                    
                    ids = boxes_data[:, 4] if boxes_data.shape[1] > 4 else None
                    current_time = current_frame_idx / fps

                    if ids is not None:
                        for tid_raw in ids:
                            if tid_raw is None or tid_raw < 0: continue
                            tid = int(tid_raw)
                            
                            # Feature extraction (se disponível)
                            try:
                                # Acesso direto interno ao tracker do Ultralytics
                                # Pode variar dependendo da versão, protegemos com try
                                trk = tracker.tracks[tid]
                                f = trk.feat.copy() if hasattr(trk, 'feat') else np.zeros(512)
                            except:
                                f = np.zeros(512, dtype=np.float32)

                            if tid not in track_data:
                                track_data[tid] = {
                                    "start": current_time,
                                    "end": current_time,
                                    "features": [f],
                                    "frames": {current_frame_idx},
                                }
                            else:
                                track_data[tid]["end"] = current_time
                                track_data[tid]["frames"].add(current_frame_idx)
                                if len(track_data[tid]["features"]) < 20:
                                    track_data[tid]["features"].append(f)
                
                results.append(frame_res)

            # Atualiza Indices
            frame_idx_global += len(frames_batch)
            
            prog = int((frame_idx_global / total_frames) * 100)
            if prog >= last_progress + 10:
                sys.stdout.write(f"\r{Fore.YELLOW}[YOLO]{Fore.WHITE} Progresso: {prog} %")
                sys.stdout.flush()
                last_progress = prog

        cap.release()
        sys.stdout.write('\n')
        sys.stdout.flush()
        
        # Merge Tracks
        if state.stop_requested:
            break

        if not track_data:
            print("[WARN] Nenhum track valido identificado.")
            continue

        merged_tracks, id_map = merge_tracks(track_data)

        resultados_finais.append({
            "video": str(video),
            "fps": fps,
            "track_data": track_data,
            "merged_tracks": merged_tracks,
            "id_map": id_map,
            "results": results,
        })
        
        # Clean Memory
        del model.tracker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(Fore.GREEN + "[OK] DETECÇÃO E IDENTIFICAÇÃO CONCLUIDA!")
    return resultados_finais
