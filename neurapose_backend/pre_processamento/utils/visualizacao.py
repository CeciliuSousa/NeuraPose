# ==============================================================
# pre_processamento/utils/visualizacao.py
# ==============================================================
# Funções para desenho de esqueletos, cores por ID e visualização.

import cv2
import numpy as np
import hashlib
from neurapose_backend.pre_processamento.configuracao.config import PAIRS

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
    for a, b in PAIRS:
        if keypoints[a][2] >= kp_thresh and keypoints[b][2] >= kp_thresh:
            pt1 = (int(keypoints[a][0]), int(keypoints[a][1]))
            pt2 = (int(keypoints[b][0]), int(keypoints[b][1]))
            cv2.line(frame, pt1, pt2, edge_color, 1, lineType=cv2.LINE_AA)

    # Desenha pontos (joints)
    for (x, y, conf) in keypoints:
        if conf >= kp_thresh:
            cv2.circle(frame, (int(x), int(y)), 2, base_color, -1, lineType=cv2.LINE_AA)

    return frame
