# neurapose_backend/nucleo/visualizacao.py
# Geração de vídeos e visualização (esqueletos, boxes, overlays).

import cv2
import os
import numpy as np
import hashlib
from pathlib import Path
from colorama import Fore


import neurapose_backend.config_master as cm

# Estado global para streaming de vídeo
from neurapose_backend.globals.state import state


def _hash_to_color(i: int):
    """Gera uma cor consistente baseada no hash do ID."""
    h = int(hashlib.md5(str(i).encode()).hexdigest(), 16)
    hue = h % 180
    sat = 200 + (h // 180) % 55
    val = 200 + (h // (180 * 55)) % 55
    hsv = np.uint8([[[hue, sat, val]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    return tuple(map(int, bgr))


def color_for_id(global_id: int):
    """Retorna cor (B,G,R) para um dado ID global."""
    return _hash_to_color(global_id)


def desenhar_esqueleto_unificado(frame, keypoints, kp_thresh=cm.POSE_CONF_MIN, base_color=(0, 255, 0), edge_color=None):
    """
    Desenha o esqueleto (keypoints e conexões) no frame.
    keypoints: array (K, 3) -> [x, y, conf]
    (Renomeado de desenhar_esqueleto para compatibilidade com backend modular)
    """
    if edge_color is None:
        edge_color = tuple(int(c * 0.6) for c in base_color)

    # Desenha conexões (limbs)
    for a, b in cm.PAIRS:
        # Verifica bounds
        if a < len(keypoints) and b < len(keypoints):
            if keypoints[a][2] >= kp_thresh and keypoints[b][2] >= kp_thresh:
                pt1 = (int(keypoints[a][0]), int(keypoints[a][1]))
                pt2 = (int(keypoints[b][0]), int(keypoints[b][1]))
                cv2.line(frame, pt1, pt2, edge_color, 1, lineType=cv2.LINE_AA)

    # Desenha pontos (joints)
    for i, (x, y, conf) in enumerate(keypoints):
        if conf >= kp_thresh:
            cv2.circle(frame, (int(x), int(y)), 2, base_color, -1, lineType=cv2.LINE_AA)

    return frame


def desenhar_info_predicao_padrao(frame, bbox, pid, conf, pred_name=None, classe_id=0):
    """
    Desenha bounding box e labels informativos no frame.
    (Renomeado de desenhar_info_predicao para compatibilidade com backend modular)
    Adptado para assinatura nova: (frame, bbox, pid, conf, pred_name=None, classe_id=0)
    Ignora botsort_id e gid extras que a versao antiga pedia, usa pid como ID visual principal.
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # ============================================================
    # CORES DINÂMICAS: Verde para NORMAL, Vermelho para CLASSE2
    # ============================================================
    if classe_id == 1:    # CLASSE2 (ex: FURTO)
        cor_bbox = (0, 0, 255)       # vermelho
        cor_label = (0, 0, 255)      # vermelho
    else:                # CLASSE1 (ex: NORMAL)
        cor_bbox = (0, 255, 0)       # verde
        cor_label = (0, 255, 0)      # verde

    cv2.rectangle(frame, (x1, y1), (x2, y2), cor_bbox, 2)

    # ============================================================
    # LINHA 1: ID_P | Pessoa: conf
    # ============================================================
    label_linha1 = f"ID: {pid} | Pessoa: {conf:.2f}"

    # ============================================================
    # LINHA 2: Classe: NORMAL/FURTO (apenas se existir pred_name)
    # ============================================================
    
    # Desenha fundo da linha 1 (branco)
    (tw1, th1), _ = cv2.getTextSize(label_linha1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(
        frame,
        (x1, max(0, y1 - th1 - 10)),
        (x1 + tw1 + 6, y1),
        (255, 255, 255),  # fundo branco
        -1,
    )
    cv2.putText(
        frame,
        label_linha1,
        (x1 + 3, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),  # texto preto
        1,
        cv2.LINE_AA,
    )

    if pred_name:
        label_linha2 = f"Classe: {pred_name}"
        # Desenha fundo da linha 2 (cor dinâmica: verde ou vermelho)
        (tw2, th2), _ = cv2.getTextSize(label_linha2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(
            frame,
            (x1, y1),
            (x1 + tw2 + 6, y1 + th2 + 10),
            cor_label,  # fundo verde ou vermelho
            -1,
        )
        cv2.putText(
            frame,
            label_linha2,
            (x1 + 3, y1 + th2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # texto branco para contraste
            1,
            cv2.LINE_AA,
        )
    
    return frame


def gerar_video_predicao(
    video_path: Path,
    registros,
    video_out_path: Path,
    show_preview: bool = False,
    preview_callback=None, # Mantido para compatibilidade, mas ignorado em favor de state
    modelo_nome: str = "CLASSE"
):
    """
    Lê o vídeo original e desenha:
      - esqueleto
      - bounding box
      - overlays de BoTSORT + classe (CLASSE1/CLASSE2)
      usando os registros já enriquecidos com "classe_id" e "classe_predita".
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Tenta carregar DLL do OpenH264 explicitamente (Windows Python 3.8+)
    if hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(str(cm.ROOT))
        except Exception:
            pass

    # Tenta usar avc1 primeiro (OpenH264)
    writer = cv2.VideoWriter(
        str(video_out_path),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (W, H),
    )
    
    # Fallback codec se avc1 falhar
    if not writer.isOpened():
        # print(f"[AVISO] Codec 'avc1' falhou. Tentando 'mp4v'.")
        writer = cv2.VideoWriter(
            str(video_out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H),
        )

    if not writer.isOpened():
        print(f"[ERRO] Falha ao iniciar VideoWriter (Visualizacao).")
        return

    # Agrupa registros por frame
    registros_por_frame = {}
    for r in registros:
        f_id = int(r["frame"])
        if f_id not in registros_por_frame:
            registros_por_frame[f_id] = []
        registros_por_frame[f_id].append(r)

    print(Fore.CYAN + f"[INFO] RENDERIZANDO VÍDEO: {video_path.name}...")

    frame_idx = 1 # frames humanos (1-based)

    while True:
        # Verifica se foi solicitada parada
        if state.stop_requested:
            break
            
        ok, frame = cap.read()
        if not ok:
            break

        # Garante 3 canais (BGR) para evitar erro no writer
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        regs = registros_por_frame.get(frame_idx, [])

        for r in regs:
            # Extrai dados robustamente
            bbox = r.get("bbox")
            if bbox is None: continue
            
            kps = np.array(r["keypoints"], dtype=np.float32)
            pid = int(r.get("id_persistente", r.get("botsort_id", 0)))
            conf = float(r.get("confidence", 0.0))
            
            classe_id = int(r.get("classe_id", 0))
            pred_name = r.get("classe_predita", None)

            # 1. Desenha Esqueleto
            frame = desenhar_esqueleto_unificado(frame, kps, kp_thresh=cm.POSE_CONF_MIN, base_color=color_for_id(pid))

            # 2. Desenha Info (Box + Texto)
            frame = desenhar_info_predicao_padrao(
                frame, 
                bbox, 
                pid, 
                conf, 
                pred_name=pred_name, 
                classe_id=classe_id
            )

        writer.write(frame)

        if show_preview:
            # Stream para browser via MJPEG (Usa state global)
            state.set_frame(frame)

        frame_idx += 1

    cap.release()
    writer.release()
    # cv2.destroyAllWindows()
