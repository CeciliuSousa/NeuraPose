# ==============================================================
# neurapose-backend/app/utils/visualizacao.py (COMPLETO E CORRIGIDO)
# ==============================================================
# Este módulo é responsável pela geração de vídeos e visualização
# dos resultados, desenhando esqueletos, bounding boxes e overlays.

import cv2
import numpy as np
import hashlib
from pathlib import Path
from tqdm import tqdm

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


def desenhar_esqueleto(frame, keypoints, kp_thresh=0.3, base_color=(0, 255, 0), edge_color=None):
    """
    Desenha o esqueleto (keypoints e conexões) no frame.
    keypoints: array (K, 3) -> [x, y, conf]
    """
    if edge_color is None:
        edge_color = tuple(int(c * 0.6) for c in base_color)

    # Desenha conexões (limbs)
    for a, b in cm.PAIRS:
        if keypoints[a][2] >= kp_thresh and keypoints[b][2] >= kp_thresh:
            pt1 = (int(keypoints[a][0]), int(keypoints[a][1]))
            pt2 = (int(keypoints[b][0]), int(keypoints[b][1]))
            cv2.line(frame, pt1, pt2, edge_color, 1, lineType=cv2.LINE_AA)

    # Desenha pontos (joints)
    for (x, y, conf) in keypoints:
        if conf >= kp_thresh:
            cv2.circle(frame, (int(x), int(y)), 2, base_color, -1, lineType=cv2.LINE_AA)

    return frame


def desenhar_info_predicao(frame, bbox, botsort_id, pid, gid, classe_id, conf, pred_name, modelo_nome="CLASSE"):
    """
    Desenha bounding box e labels informativos no frame.
    Estilo: 
      - Linha 1: ID_P e Pessoa (confiança)
      - Linha 2: Classe predita
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
    # LINHA 1: ID_P | Pessoa: conf (estilo igual ao processamento)
    # ============================================================
    label_linha1 = f"ID_P: {pid} | Pessoa: {conf:.2f}"

    # ============================================================
    # LINHA 2: Classe: NORMAL/FURTO (adicional para classificação)
    # ============================================================
    label_linha2 = f"Classe: {pred_name}"

    # Desenha fundo da linha 1 (branco)
    (tw1, th1), _ = cv2.getTextSize(label_linha1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(
        frame,
        (x1, max(0, y1 - th1 - 12)),
        (x1 + tw1 + 10, y1),
        (255, 255, 255),  # fundo branco
        -1,
    )
    cv2.putText(
        frame,
        label_linha1,
        (x1 + 5, y1 - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),  # texto preto
        2,
        cv2.LINE_AA,
    )

    # Desenha fundo da linha 2 (cor dinâmica: verde ou vermelho)
    (tw2, th2), _ = cv2.getTextSize(label_linha2, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(
        frame,
        (x1, y1),
        (x1 + tw2 + 10, y1 + th2 + 12),
        cor_label,  # fundo verde ou vermelho
        -1,
    )
    cv2.putText(
        frame,
        label_linha2,
        (x1 + 5, y1 + th2 + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),  # texto branco para contraste
        2,
        cv2.LINE_AA,
    )
    
    return frame


def gerar_video_predicao(
    video_path: Path,
    registros,
    video_out_path: Path,
    show_preview: bool = False,
    modelo_nome: str = "CLASSE" # <--- NOVO ARGUMENTO
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

    writer = cv2.VideoWriter(
        str(video_out_path),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (W, H),
    )
    if not writer.isOpened():
        print(f"[ERRO] Falha ao iniciar VideoWriter (Visualizacao) com avc1.")
        return

    # Agrupa registros por frame
    registros_por_frame = {}
    for r in registros:
        f_id = int(r["frame"])
        if f_id not in registros_por_frame:
            registros_por_frame[f_id] = []
        registros_por_frame[f_id].append(r)

    pbar = tqdm(
        total=total_frames,
        ncols=110,
        desc=f"Geração vídeo {video_path.stem}",
        leave=False,
    )

    frame_idx = 1 # frames humanos (1-based)

    while True:
        # Verifica se foi solicitada parada
        if state.stop_requested:
            break
            
        ok, frame = cap.read()
        if not ok:
            break

        regs = registros_por_frame.get(frame_idx, [])

        for r in regs:
            x1, y1, x2, y2 = map(int, r["bbox"])
            conf = float(r["confidence"])
            kps = np.array(r["keypoints"], dtype=np.float32)

            botsort_id = int(r.get("botsort_id", -1))
            pid = int(r.get("id_persistente", botsort_id))
            gid = int(r.get("id", 0))

            classe_id = int(r.get("classe_id", 0))
            classe_id = int(r.get("classe_id", 0))
            pred_name = r.get("classe_predita", cm.CLASSE1 if classe_id == 0 else cm.CLASSE2)

            # Desenha esqueleto
            frame = desenhar_esqueleto(frame, kps, kp_thresh=cm.POSE_CONF_MIN)

            # Desenha bbox e labels usando função centralizada
            frame = desenhar_info_predicao(
                frame, 
                r["bbox"], 
                botsort_id, 
                pid, 
                gid, 
                classe_id, 
                conf, 
                pred_name,
                modelo_nome # <--- NOVO: PASSANDO O NOME CORRETO
            )

        writer.write(frame)

        if show_preview:
            # Stream para browser via MJPEG
            state.set_frame(frame)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # print(Fore.GREEN + f"[OK] Vídeo com esqueleto + classe salvo em: {video_out_path}")