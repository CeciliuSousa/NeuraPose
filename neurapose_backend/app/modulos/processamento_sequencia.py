# ==============================================================
# neurapose-backend/app/modulos/processamento_sequencia.py
# ==============================================================

import numpy as np
from neurapose_backend.app.configuracao.config import EMA_ALPHA, EMA_MIN_CONF


def _expand_bbox(x1, y1, x2, y2, margin, W, H):
    """
    Expande o bounding box com uma margem, respeitando os limites da imagem.
    """
    w, h = x2 - x1, y2 - y1
    mx, my = margin * w, margin * h
    ex1 = max(0, int(x1 - mx))
    ey1 = max(0, int(y1 - my))
    ex2 = min(W - 1, int(x2 + mx))
    ey2 = min(H - 1, int(y2 + my))
    return ex1, ey1, ex2, ey2


class EmaSmoother:
    """
    Suavizador de keypoints usando Média Móvel Exponencial (EMA).
    """
    def __init__(self, alpha=EMA_ALPHA, min_conf=EMA_MIN_CONF):
        self.alpha = alpha
        self.min_conf = min_conf
        self.buf = {}

    def step(self, gid, kps):
        if gid not in self.buf:
            self.buf[gid] = kps.copy()
            return kps
        prev = self.buf[gid]
        out = kps.copy()
        conf_mask = (kps[:, 2] >= self.min_conf).astype(np.float32)[:, None]
        out[:, :2] = (1 - self.alpha) * kps[:, :2] + self.alpha * prev[:, :2]
        out[:, :2] = conf_mask * out[:, :2] + (1 - conf_mask) * kps[:, :2]
        self.buf[gid] = out
        return out


def montar_sequencia_individual(records, target_id, max_frames=60, min_frames=5):
    """
    Monta a sequência temporal para UM ID específico (target_id).
    Retorna:
      seq: np.ndarray (2, T, 17) no formato (C,T,V)
      ou None, se não houver frames suficientes.
    """
    frames = []

    for r in sorted(records, key=lambda x: int(x["frame"])):
        if int(r["id"]) != int(target_id):
            continue
        coords = np.array(
            [[kp[0], kp[1]] for kp in r["keypoints"][:17]], dtype=np.float32
        )
        frames.append(coords)

    if len(frames) < min_frames:
        return None

    seq = np.zeros((max_frames, 17, 2), dtype=np.float32)
    num = min(len(frames), max_frames)
    seq[:num] = frames[:num]

    # (T,V,2) -> (2,T,V)
    seq = np.transpose(seq, (2, 0, 1))
    return seq
