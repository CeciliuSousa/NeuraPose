# ==============================================================
# neurapose-backend/app/modulos/extracao_pose.py
# ==============================================================

import cv2
import numpy as np
from tqdm import tqdm
from neurapose_backend.app.configuracao.config import (
    CLASSE1,
    CLASSE2,
    SIMCC_W,
    SIMCC_H,
    CLAMP_MARGIN,
    POSE_CONF_MIN,
    CLASSE2_THRESHOLD,
    MODEL_NAME
)
from colorama import Fore, init

init(autoreset=True)

from neurapose_backend.app.modulos.processamento_sequencia import (
    EmaSmoother, 
    _expand_bbox, 
    montar_sequencia_individual
)

from neurapose_backend.app.utils.geometria import (
    get_affine_transform,
    transform_preds,
    _calc_center_scale
)

from neurapose_backend.app.modulos.rtmpose import (
    decode_simcc_output,
    preprocess_rtmpose_input
)

from neurapose_backend.app.modulos.inferencia_lstm import rodar_lstm_uma_sequencia

from neurapose_backend.app.utils.visualizacao import desenhar_esqueleto, desenhar_info_predicao

# Estado global para controle de parada
from neurapose_backend.globals.state import state



def extrair_keypoints_rtmpose_padronizado(
    video_path,
    results,
    sess,
    input_name,
    id_map=None,
    show_preview: bool = False,
    model=None,
    mu=None,
    sigma=None,
    modelo_nome: str = MODEL_NAME
):
    """
    Usa as detecções (bbox + ids) vindas do YOLO+BoTSORT e aplica RTMPose.
    Retorna uma lista de registros com keypoints e metadados.
    """
    if id_map is None:
        id_map = {}

    cap = cv2.VideoCapture(str(video_path))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    registros = []
    smoother = EmaSmoother()

    # Reset de IDs a cada vídeo (ID usado na LSTM)
    id_reset_map = {}
    next_id = 1
    
    # Histórico para inferência online (apenas para preview)
    id_history = {}
    id_score_ema = {}
    ALPHA_EMA = 0.2

    pbar = tqdm(
        total=total_frames,
        ncols=110,
        desc=f"RTMPose {video_path.stem}",
        leave=False,
    )

    frame_idx = 0

    while True:
        # Verifica se foi solicitada parada
        if state.stop_requested:
            print(Fore.YELLOW + "[STOP] Processamento interrompido pelo usuário.")
            break

        ok, frame = cap.read()
        if not ok:
            break

        # Se não há mais resultados, apenas avança
        if frame_idx >= len(results):
            if show_preview:
                state.set_frame(frame)  # Stream para browser
            pbar.update(1)
            continue

        # Pegamos resultados do YOLO+BoTSORT
        result = results[frame_idx]
        frame_idx += 1

        # Agora result é um dict: {"boxes": numpy_array [N, 7]} ou None
        # data contém: [x1, y1, x2, y2, id, conf, cls]
        if result.get("boxes") is None or len(result["boxes"]) == 0:
            if show_preview:
                state.set_frame(frame)  # Stream para browser
            pbar.update(1)
            continue

        boxes_data = result["boxes"]
        boxes = boxes_data[:, :4]
        ids = boxes_data[:, 4]
        confs = boxes_data[:, 5]

        # frame_id humano (1-based) para salvar nos registros
        frame_id = frame_idx

        # Vamos desenhar no preview somente se o usuário quiser
        if show_preview:
            frame_preview = frame.copy()
        else:
            frame_preview = None

        for box, conf, track_id in zip(boxes, confs, ids):
            if track_id is None or track_id < 0:
                continue

            raw_tid = int(track_id)

            # ID persistente (BoTSORT + fusão)
            pid = int(id_map.get(raw_tid, raw_tid))

            # ID reindexado para LSTM (por vídeo)
            if raw_tid not in id_reset_map:
                id_reset_map[raw_tid] = next_id
                next_id += 1
            gid = int(id_reset_map[raw_tid])

            x1, y1, x2, y2 = map(int, box)
            center, scale = _calc_center_scale(x1, y1, x2, y2)
            trans = get_affine_transform(center, scale, 0, (SIMCC_W, SIMCC_H))
            crop = cv2.warpAffine(frame, trans, (SIMCC_W, SIMCC_H), flags=cv2.INTER_LINEAR)

            inp = preprocess_rtmpose_input(crop)
            simx, simy = sess.run(None, {input_name: inp})
            coords_in, conf_arr = decode_simcc_output(simx, simy)
            coords_fr = transform_preds(coords_in[0], center, scale, (SIMCC_W, SIMCC_H))

            # (K,3)
            kps = np.concatenate([coords_fr, conf_arr[0][:, None]], axis=1).astype(np.float32)

            # EMA por ID pessoa
            kps = smoother.step(gid, kps)

            # Clamping
            if CLAMP_MARGIN > 0:
                ex1, ey1, ex2, ey2 = _expand_bbox(x1, y1, x2, y2, CLAMP_MARGIN, W, H)
                kps[:, 0] = np.clip(kps[:, 0], ex1, ex2)
                kps[:, 1] = np.clip(kps[:, 1], ey1, ey2)

            # Prepara registro
            registro_atual = {
                "frame": int(frame_id),
                "id": int(gid),
                "botsort_id": int(raw_tid),
                "id_persistente": int(pid),
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "keypoints": kps.tolist(),
            }
            registros.append(registro_atual)

            # --- INFERÊNCIA ---
            classe_id = 0
            score = 0.0
            classe_nome = CLASSE1

            if model is not None:
                # Atualiza histórico
                if gid not in id_history:
                    id_history[gid] = []
                id_history[gid].append(registro_atual)
                
                # Mantém apenas últimos 60 frames
                if len(id_history[gid]) > 60:
                    id_history[gid].pop(0)
                
                # Tenta rodar inferência
                seq_np = montar_sequencia_individual(id_history[gid], target_id=gid, min_frames=15)
                
                if seq_np is not None:
                    raw_score, _ = rodar_lstm_uma_sequencia(seq_np, model, mu, sigma)
                    
                    # Aplica suavização EMA no score
                    last_ema = id_score_ema.get(gid, 0.0)
                    if gid not in id_score_ema:
                        new_ema = raw_score
                    else:
                        new_ema = last_ema * (1 - ALPHA_EMA) + raw_score * ALPHA_EMA
                    
                    id_score_ema[gid] = new_ema
                    score = new_ema

                    if score >= CLASSE2_THRESHOLD:
                        classe_id = 1
                        classe_nome = CLASSE2
            
            # Atualiza o registro com a classificação do momento
            registro_atual["classe_id"] = classe_id
            registro_atual["classe_predita"] = classe_nome
            registro_atual[f"score_{CLASSE2}_id"] = float(score)

            if show_preview and frame_preview is not None:
                # desenha o esqueleto
                frame_preview = desenhar_esqueleto(frame_preview, kps, kp_thresh=POSE_CONF_MIN)
                
                # Desenha bbox e labels usando função centralizada
                frame_preview = desenhar_info_predicao(
                    frame_preview,
                    [x1, y1, x2, y2],
                    raw_tid,
                    pid,
                    gid,
                    classe_id,
                    conf,
                    classe_nome,
                    modelo_nome
                )

        if show_preview:
            # Stream para browser via state.set_frame (MJPEG)
            state.set_frame(frame_preview if frame_preview is not None else frame)

        pbar.update(1)

    pbar.close()
    cap.release()
    cv2.destroyAllWindows()

    return registros