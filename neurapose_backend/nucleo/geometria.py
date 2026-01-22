# ==============================================================
# neurapose_backend/nucleo/geometria.py
# ==============================================================
# Módulo centralizado para funções geométricas e transformações espaciais.
# Usado tanto no APP (Inferência) quanto no PRÉ-PROCESSAMENTO.
# ==============================================================

import numpy as np
import cv2

def calcular_deslocamento(p_inicial, p_final):
    """
    Calcula a distância euclidiana em pixels entre o ponto inicial e final.
    Usado para filtrar objetos estáticos.
    
    Args:
        p_inicial (list/array): Coordenadas [x, y] do início da trajetória.
        p_final (list/array): Coordenadas [x, y] do fim da trajetória.
        
    Returns:
        float: Distância em pixels.
    """
    p1 = np.array(p_inicial)
    p2 = np.array(p_final)
    return np.linalg.norm(p2 - p1)


def calcular_iou(boxA, boxB):
    """
    Calcula Intersection over Union (IoU) entre duas caixas bbox.
    
    Args:
        boxA (list/array): [x1, y1, x2, y2]
        boxB (list/array): [x1, y1, x2, y2]
        
    Returns:
        float: Valor do IoU entre 0.0 e 1.0.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def calcular_centro_escala(x1, y1, x2, y2, aspect_ratio=(192, 256), scale_factor=1.25):
    """
    Calcula o centro e a escala de uma bbox para crop do RTMPose.
    
    Args:
        x1, y1, x2, y2: Coordenadas da bbox.
        aspect_ratio: Tupla (width, height) alvo do modelo (ex: 192, 256).
        scale_factor: Fator de expansão da bbox (padrão 1.25 para incluir contexto).
        
    Returns:
        tuple: (center, scale) onde center é [cx, cy] e scale é [sx, sy].
    """
    center = np.array([x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2], dtype=np.float32)

    w = x2 - x1
    h = y2 - y1

    # Lógica padrão do mmpose/rtmpose para aspect ratio
    if w > aspect_ratio[0] / aspect_ratio[1] * h:
        h = w * 1.0 / (aspect_ratio[0] / aspect_ratio[1])
    elif w < aspect_ratio[0] / aspect_ratio[1] * h:
        w = h * aspect_ratio[0] / aspect_ratio[1]
    
    scale = np.array([w * 1.0 / 200.0, h * 1.0 / 200.0], dtype=np.float32)
    
    # Aplica fator de escala extra (1.25x)
    scale = scale * scale_factor

    return center, scale


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    """
    Gera matriz de transformação afim para crop e resize.
    Baseado na implementação do mmpose.
    """
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(dst, src)
    else:
        trans = cv2.getAffineTransform(src, dst)

    return trans


def transform_preds(coords, center, scale, output_size):
    """
    Transforma coordenadas preditas (no espaço do crop 192x256) de volta para a imagem original.
    """
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    target_coords = np.zeros(coords.shape)
    
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
        
    return target_coords


# --- Utils Internos de Geometria ---

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]
