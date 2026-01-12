# ================================================================
# neurapose_backend/detector/yolo_detector.py
# ================================================================

import os
import cv2
import torch
import logging
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Importacoes do tracker (usando interface modular via rastreador.py)
from neurapose_backend.tracker.rastreador import CustomBoTSORT, CustomReID, save_temp_tracker_yaml

# Importa nome do modelo YOLO e ROOT do config centralizado
from neurapose_backend.config_master import YOLO_PATH, YOLO_MODEL, ROOT, DETECTION_CONF, YOLO_CLASS_PERSON, DEVICE

logging.getLogger("ultralytics").setLevel(logging.ERROR)
os.environ["YOLO_VERBOSE"] = "False"

from ultralytics.trackers import bot_sort

# SUBSTITUI O BOT-SORT INTERNO
bot_sort.BOTSORT = CustomBoTSORT

# SUBSTITUI O ReID INTERNO
bot_sort.ReID = CustomReID


# ================================================================
# 1. Funcao de fusao de IDs - Gera IDs persistentes
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
def yolo_detector_botsort(videos_dir=None):
    """
    Roda YOLOv8x + CustomBoTSORT em varios videos.
    Retorna lista com:
        - video
        - fps
        - track_data
        - merged_tracks
        - id_map
        - results (frame a frame)
    """

    videos_path = Path(videos_dir or (ROOT / "videos"))

    # ================================================================
    # MODELO YOLO - Usar caminho centralizado (config_master) e baixar se nao existir
    # ================================================================
    model_path = YOLO_PATH

    # Criar diretorio de modelos se nao existir
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        # Extrai o nome base do modelo (ex: "yolov8l" de "yolov8l.pt")
        model_base = YOLO_MODEL.replace('.pt', '')

        try:
            # Ultralytics baixa automaticamente quando voce instancia com nome
            temp_model = YOLO(model_base)

            # Salva o modelo baixado no local correto
            temp_model.save(str(model_path))

        except Exception as e:
            # Se o download falhar, remove o arquivo parcial/corrompido
            if model_path.exists():
                os.remove(model_path)
            raise FileNotFoundError(
                f"Erro ao baixar modelo {YOLO_PATH}. "
                f"Verifique sua conexao com a internet ou se o modelo nao esta corrompido.\nErro: {e}"
            )
    
    # Carrega o modelo (local ou recem-baixado)
    model = YOLO(str(model_path)).to(DEVICE)

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

    # ============================================================
    # Loop para todos os videos
    # ============================================================
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        tracker = CustomBoTSORT(frame_rate=int(fps))
        model.tracker = tracker

        tracker_yaml_path = save_temp_tracker_yaml()
        logging.getLogger("neurapose.tracker").info(f"Usando BoTSORT YAML: {tracker_yaml_path}")

        # Execucao do YOLO + tracking
        results = model.track(
            source=str(video),
            imgsz=1280,
            conf=DETECTION_CONF,
            device=DEVICE,
            persist=True,
            tracker=str(tracker_yaml_path),
            classes=[YOLO_CLASS_PERSON],
            verbose=False
        )

        if not results or not hasattr(results[0], "boxes"):
            print("[ERRO] Sem resultados validos.")
            cap.release()
            continue

        frame_idx = 0
        track_data = {}

        # ============================================================
        # COLETA DE TRACKS POR FRAME (start, end, frames)
        # ============================================================
        while True:
            ret, _ = cap.read()
            if not ret or frame_idx >= len(results):
                break

            r = results[frame_idx]

            if r.boxes is not None and len(r.boxes) > 0:
                ids_tensor = r.boxes.id
                if ids_tensor is None:
                    frame_idx += 1
                    continue

                ids = ids_tensor.cpu().numpy()
                current_time = frame_idx / fps

                for track_id in ids:
                    if track_id is None or track_id < 0:
                        continue

                    tid = int(track_id)

                    # OBS: aqui nao precisamos mais de embeddings para fusao;
                    # mantemos lista de features so para futuro, mas nao usamos.
                    try:
                        f = tracker.tracks[tid].feat.copy()
                    except Exception:
                        f = np.zeros(512, dtype=np.float32)

                    if tid not in track_data:
                        track_data[tid] = {
                            "start": current_time,
                            "end": current_time,
                            "features": [f],
                            "frames": {frame_idx},
                        }
                    else:
                        track_data[tid]["end"] = current_time
                        track_data[tid]["frames"].add(frame_idx)

                        if len(track_data[tid]["features"]) < 20:
                            track_data[tid]["features"].append(f)

            frame_idx += 1

        cap.release()

        if not track_data:
            print("[WARN] Nenhum track valido.")
            continue

        # ============================================================
        # FUSAO FINAL DE IDs
        # ============================================================
        merged_tracks, id_map = merge_tracks(track_data)

        # Salva tudo organizado
        resultados_finais.append({
            "video": str(video),
            "fps": fps,
            "track_data": track_data,
            "merged_tracks": merged_tracks,
            "id_map": id_map,
            "results": results,
        })

    return resultados_finais
