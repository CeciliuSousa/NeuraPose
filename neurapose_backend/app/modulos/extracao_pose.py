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

    # Batch Processing Variables
    from neurapose_backend.app.configuracao.config import RTMPOSE_BATCH_SIZE
    batch_crops = []
    batch_meta = [] # (frame_img_copy, x1,y1,x2,y2, center, scale, pid, conf, raw_tid, gid, frame_id)
    
    frame_idx = 0

    def process_batch(crop_list, meta_list):
        if not crop_list:
            return

        # Stack e Inferência
        full_inp = np.concatenate(crop_list, axis=0)
        
        # Sessão ONNX Run
        simx, simy = sess.run(None, {input_name: full_inp})
        
        # Post-Process Batch
        coords_batch, conf_batch = decode_simcc_output(simx, simy)

        # Itera pelo batch
        for i, (fr_img, x1, y1, x2, y2, c, s, pid, conf, r_tid, gid, fid) in enumerate(meta_list):
            
            # Transform Back
            coords_fr = transform_preds(coords_batch[i], c, s, (SIMCC_W, SIMCC_H))
            kps = np.concatenate([coords_fr, conf_batch[i][:, None]], axis=1).astype(np.float32)
            kps = smoother.step(gid, kps)

            if CLAMP_MARGIN > 0:
                ex1, ey1, ex2, ey2 = _expand_bbox(x1, y1, x2, y2, CLAMP_MARGIN, W, H)
                kps[:, 0] = np.clip(kps[:, 0], ex1, ex2)
                kps[:, 1] = np.clip(kps[:, 1], ey1, ey2)

            # Registro
            registro_atual = {
                "frame": int(fid),
                "id": int(gid),
                "botsort_id": int(r_tid),
                "id_persistente": int(pid),
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "keypoints": kps.tolist(),
            }
            registros.append(registro_atual)

            # Lógica LSTM e Classificação (Mantida original)
            classe_id = 0
            score = 0.0
            classe_nome = CLASSE1

            if model is not None:
                if gid not in id_history:
                    id_history[gid] = []
                id_history[gid].append(registro_atual)
                if len(id_history[gid]) > 60:
                    id_history[gid].pop(0)
                
                seq_np = montar_sequencia_individual(id_history[gid], target_id=gid, min_frames=15)
                if seq_np is not None:
                    raw_score, _ = rodar_lstm_uma_sequencia(seq_np, model, mu, sigma)
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

            registro_atual["classe_id"] = classe_id
            registro_atual["classe_predita"] = classe_nome
            registro_atual[f"score_{CLASSE2}_id"] = float(score)

            # Visualização (Preview)
            # Como estamos em batch, o 'fr_img' é o frame correspondente a essa detecção
            # Nota: Se tiver 2 pessoas no mesmo frame, desenharemos 2 vezes no mesmo array base?
            # Precisamos coordenar o desenho se tiver multi-pessoas por frame, mas simples é desenhar direto
            if show_preview and fr_img is not None:
                # CUIDADO: fr_img aqui é uma cópia segura ou o frame original?
                # Se for batch across frames, precisamos renderizar e enviar o frame SOMENTE qdo todas deteccoes daquele frame estiverem prontas.
                # Simplificação: Desenhamos na copia local e enviamos (pode causar flicker se tiver 2 pessoas e enviar 2 vezes)
                # Melhor: enviar apenas a última versão do frame processado.
                
                # Desenha esqueleto
                desenhar_esqueleto(fr_img, kps, kp_thresh=POSE_CONF_MIN)
                desenhar_info_predicao(
                    fr_img, [x1, y1, x2, y2],
                    r_tid, pid, gid, classe_id, conf, classe_nome, modelo_nome
                )

    # Dicionário para controlar envio de preview por frame (evitar flicker/envio parcial)
    # frame_id -> {image: img, pending_count: N}
    pending_frames_preview = {}

    while True:
        if state.stop_requested:
            print(Fore.YELLOW + "[STOP] Processamento interrompido pelo usuário.")
            break

        ok, frame = cap.read()
        if not ok:
            break

        # Limite de segurança YOLO results
        if frame_idx >= len(results):
            pbar.update(1)
            continue
        
        result = results[frame_idx]
        frame_id = frame_idx + 1 # 1-based
        frame_idx += 1

        # Frame Preview Management
        if show_preview:
            # Mantém cópia para desenho
            # Se houver detecções, vamos desenhar nela e enviar depois
            # Se não houver, enviamos direto
            frame_disp = frame.copy()
        else:
            frame_disp = None

        if result.get("boxes") is None or len(result["boxes"]) == 0:
            if show_preview:
                state.set_frame(frame)
            pbar.update(1)
            continue

        boxes_data = result["boxes"]
        boxes = boxes_data[:, :4]
        ids = boxes_data[:, 4]
        confs = boxes_data[:, 5]
        
        valid_indices = [i for i, tid in enumerate(ids) if tid is not None and tid >= 0]
        
        if show_preview and len(valid_indices) > 0:
            pending_frames_preview[frame_id] = {
                "img": frame_disp,
                "count": len(valid_indices)
            }

        for i in valid_indices:
            box = boxes[i]
            conf = confs[i]
            raw_tid = int(ids[i])
            
            # ID Maps
            pid = int(id_map.get(raw_tid, raw_tid))
            if raw_tid not in id_reset_map:
                id_reset_map[raw_tid] = next_id
                next_id += 1
            gid = int(id_reset_map[raw_tid])

            x1, y1, x2, y2 = map(int, box)
            center, scale = _calc_center_scale(x1, y1, x2, y2)
            trans = get_affine_transform(center, scale, 0, (SIMCC_W, SIMCC_H))
            crop = cv2.warpAffine(frame, trans, (SIMCC_W, SIMCC_H), flags=cv2.INTER_LINEAR)
            
            inp = preprocess_rtmpose_input(crop)
            
            # Adiciona ao batch
            batch_crops.append(inp)
            # Meta guardamos o frame_disp (referência) para desenhar direto nele
            batch_meta.append((frame_disp, x1, y1, x2, y2, center, scale, pid, conf, raw_tid, gid, frame_id))
            
            # Flush se encher
            if len(batch_crops) >= RTMPOSE_BATCH_SIZE:
                process_batch(batch_crops, batch_meta)
                
                # Check Preview Flush
                if show_preview:
                    # Verifica quais frames já terminaram de ser processados neste lote
                    # frames_processed = set([m[-1] for m in batch_meta])
                    # Para cada frame tocado, decrementamos o contador
                    for m in batch_meta:
                        fid = m[-1]
                        if fid in pending_frames_preview:
                            pending_frames_preview[fid]["count"] -= 1
                            # Se zerou, envia
                            if pending_frames_preview[fid]["count"] <= 0:
                                state.set_frame(pending_frames_preview[fid]["img"])
                                del pending_frames_preview[fid]
                                
                batch_crops = []
                batch_meta = []

        # Se não usamos preview ou batch não encheu, o loop continua
        # O preview só é enviado quando batch processa OU se não tinha boxes
        pbar.update(1)

    # Final Flush
    if batch_crops:
        process_batch(batch_crops, batch_meta)
        if show_preview:
            for m in batch_meta:
                fid = m[-1]
                if fid in pending_frames_preview:
                    state.set_frame(pending_frames_preview[fid]["img"])
                    del pending_frames_preview[fid]

    pbar.close()
    cap.release()
    cv2.destroyAllWindows()

    return registros