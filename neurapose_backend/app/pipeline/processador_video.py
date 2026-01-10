# ==============================================================
# neurapose-backend/app/pipeline/processador_video.py
# ==============================================================

import time
import json
import cv2
from pathlib import Path
from colorama import Fore

# Importações do projeto
from neurapose_backend.detector.yolo_detector import yolo_detector_botsort as yolo_detector
from neurapose_backend.app.configuracao.config import (
    CLASSE1,
    CLASSE2,
    PREDICOES_DIR,
    JSONS_DIR,
    RELATORIOS_ROOT,
    CLASSE2_THRESHOLD,
)
from neurapose_backend.app.modulos.extracao_pose import extrair_keypoints_rtmpose_padronizado
from neurapose_backend.app.modulos.processamento_sequencia import montar_sequencia_individual
from neurapose_backend.app.modulos.inferencia_lstm import rodar_lstm_uma_sequencia
from neurapose_backend.app.modulos.tracking import TrackHistory
from neurapose_backend.app.utils.visualizacao import gerar_video_predicao


def processar_video(video_path: Path, model, mu, sigma, sess, input_name, show_preview=False):
    """
    Processa um único vídeo do início ao fim.
    Retorna um dicionário com estatísticas e resultados.
    """
    tempos = {
        "detector_total": 0.0,   # YOLO + BoTSORT + OSNet
        "rtmpose_total": 0.0,
        "temporal_total": 0.0,
        "video_total": 0.0,
    }
    t0_video = time.time()

    # ---------------- DETECTOR (YOLO + BoTSORT + OSNet) ----------------
    d0 = time.time()

    resultados_list = yolo_detector(videos_dir=video_path)
    d1 = time.time()
    tempos["detector_total"] = d1 - d0

    if not resultados_list:
        return None

    # yolo_detector_botsort retorna lista de dicts
    res = resultados_list[0]
    video_original = Path(res["video"])
    results = res["results"]
    id_map = res.get("id_map", {})

    pred_video_path = PREDICOES_DIR / f"{video_path.stem}_pred.mp4"
    json_path = JSONS_DIR / f"{video_path.stem}.json"

    # ---------------- POSE (RTMPose) ----------------
    p0 = time.time()

    records = extrair_keypoints_rtmpose_padronizado(
        video_path=video_original,
        results=results,
        sess=sess,
        input_name=input_name,
        id_map=id_map,
        show_preview=show_preview,
        model=None,
        mu=None,
        sigma=None
    )
    p1 = time.time()
    tempos["rtmpose_total"] = p1 - p0

    if not records:
        return None

    # ---------------- LSTM / BATCH ----------------
    # Descobre todos os IDs presentes no vídeo
    ids_presentes = sorted({int(r["id"]) for r in records})

    id_preds = {}   # id -> classe_id (0 CLASSE1, 1 CLASSE2)
    id_scores = {}  # id -> score para CLASSE2 (probabilidade)

    t0_temp = time.time()

    for gid in ids_presentes:
        # Inicialmente, todos são CLASSE1 com score 0.0
        id_preds[gid] = 0
        id_scores[gid] = 0.0

        # Monta sequência para este ID
        seq_np = montar_sequencia_individual(records, target_id=gid, min_frames=5)
        if seq_np is None:
            continue  # sem frames suficientes, mantém CLASSE1

        # Roda o modelo temporal para este ID
        score, pred_raw = rodar_lstm_uma_sequencia(seq_np, model, mu, sigma)

        # Aplica threshold B: ID só vira a CLASSE2 se score >= CLASSE2_THRESHOLD
        if score >= CLASSE2_THRESHOLD:
            classe_id = 1
        else:
            classe_id = 0

        id_preds[gid] = classe_id
        id_scores[gid] = score

    t1_temp = time.time()
    tempos["temporal_total"] = t1_temp - t0_temp

    # ---------------- ATRIBUIR CLASSE AOS REGISTROS ----------------
    for r in records:
        gid = int(r["id"])
        classe_id = int(id_preds.get(gid, 0))
        score_id = float(id_scores.get(gid, 0.0))

        r["classe_id"] = classe_id
        r["classe_predita"] = CLASSE2 if classe_id == 1 else CLASSE1
        r[F"score_{CLASSE2}_id"] = score_id

    # Salvar JSON já com keypoints + classe por ID
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # print(Fore.GREEN + f"[OK] JSON de keypoints + classe salvo em: {json_path}")

    # ---------------- GERAR VÍDEO FINAL COM OVERLAY DE CLASSE ----------------
    gerar_video_predicao(
        video_path=video_original,
        registros=records,
        video_out_path=pred_video_path,
        show_preview=False,  # Evita duplicar preview se já foi mostrado antes
    )

    # ---------------- TRACKING REPORT ----------------
    # Gera relatório de tracking (duração de cada ID)
    try:
        cap_fps = cv2.VideoCapture(str(video_original))
        fps = cap_fps.get(cv2.CAP_PROP_FPS) or 30.0
        cap_fps.release()

        track_history = TrackHistory()
        for r in records:
            # r["frame"] é 1-based, mas para tempo podemos usar frame/fps
            t_sec = float(r["frame"]) / fps
            # Usando botsort_id (ID original do tracker)
            track_history.update(r["botsort_id"], t_sec)

        TRACKINGS_DIR = RELATORIOS_ROOT / "trackings"
        TRACKINGS_DIR.mkdir(parents=True, exist_ok=True)
        
        tracking_txt_path = TRACKINGS_DIR /f"{video_path.stem}_trackings.txt"
        track_history.save_txt(tracking_txt_path)
        # print(Fore.GREEN + f"[OK] Relatório de tracking salvo em: {tracking_txt_path}")
    except Exception as e:
        print(Fore.RED + f"[ERRO] Falha ao gerar relatório de tracking: {e}")

    tempos["video_total"] = time.time() - t0_video

    # Resumo a nível de vídeo para métricas:
    # vídeo é CLASSE2 se pelo menos um ID foi classificado como classe 2
    video_pred = 1 if any(v == 1 for v in id_preds.values()) else 0
    video_score = max(id_scores.values()) if len(id_scores) > 0 else 0.0

    # Detalhamento por ID
    ids_predicoes = []
    for gid in ids_presentes:
        ids_predicoes.append(
            {
                "id": int(gid),
                "classe_id": int(id_preds.get(gid, 0)),
                f"score_{CLASSE2}": float(id_scores.get(gid, 0.0)),
            }
        )

    return {
        "video": str(video_path),
        "pred": int(video_pred),
        f"score_{CLASSE2}": float(video_score),
        "tempos": tempos,
        "ids_predicoes": ids_predicoes,
    }