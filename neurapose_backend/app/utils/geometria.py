# ==============================================================
# neurapose-backend/app/utils/geometria.py
# ==============================================================

import cv2
import numpy as np
from neurapose_backend.app.configuracao.config import SIMCC_W, SIMCC_H


def get_dir(src_point, rot_rad):
    """Calcula vetor de direção rotacionado."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return [src_point[0] * cs - src_point[1] * sn,
            src_point[0] * sn + src_point[1] * cs]


def get_3rd_point(a, b):
    """Calcula o terceiro ponto para definir transformação afim."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=False):
    """
    Gera a matriz de transformação afim para crop/resize mantendo aspect ratio.
    Baseado na implementação do MMPose.
    """
    # scale é [w, h] / 200
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)
    scale_tmp = scale * 200.0

    src_w = scale_tmp[0]
    dst_w, dst_h = output_size
    rot_rad = np.pi * rot / 180
    src_dir = np.array(get_dir([0, src_w * -0.5], rot_rad), dtype=np.float32)
    dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = get_3rd_point(src[0, :], src[1, :])

    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(dst, src)
    else:
        trans = cv2.getAffineTransform(src, dst)
    return trans


def affine_transform(pt, t):
    """Aplica transformação afim a um ponto."""
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_preds(coords, center, scale, output_size):
    """
    Mapeia predições do espaço do modelo (output_size) de volta para a imagem original.
    """
    target_w, target_h = output_size
    trans = get_affine_transform(center, scale, 0, (target_w, target_h), inv=True)
    preds = np.zeros_like(coords)
    for i in range(coords.shape[0]):
        preds[i, 0:2] = affine_transform(coords[i, 0:2], trans)
    return preds


def _calc_center_scale(x1, y1, x2, y2):
    """
    Calcula centro e escala do bbox para input do RTMPose.
    """
    bbox_scale_factor = 1.25
    w = max(2, x2 - x1)
    h = max(2, y2 - y1)

    center = np.array([(x1 + x2)/2.0, (y1 + y2)/2.0], dtype=np.float32)
    aspect = SIMCC_W / SIMCC_H

    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect

    w *= bbox_scale_factor
    h *= bbox_scale_factor

    scale = np.array([w/200.0, h/200.0], dtype=np.float32)

    return center, scale


def _expand_bbox(x1, y1, x2, y2, margin, W, H):
    """
    Expande bounding box com margem, respeitando limites da imagem.
    """
    w, h = x2 - x1, y2 - y1
    mx, my = margin * w, margin * h
    return (
        max(0, int(x1 - mx)),
        max(0, int(y1 - my)),
        min(W - 1, int(x2 + mx)),
        min(H - 1, int(y2 + my)),
    )
