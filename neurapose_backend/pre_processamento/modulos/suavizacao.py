# ==============================================================
# neurapose_backend/pre_processamento/modulos/suavizacao.py
# ==============================================================

import numpy as np
from neurapose_backend.config_master import EMA_ALPHA, EMA_MIN_CONF


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

        # Máscara para aplicar EMA apenas onde a confiança é alta o suficiente
        conf_mask = (kps[:, 2] >= self.min_conf).astype(np.float32)[:, None]

        out[:, :2] = (1 - self.alpha) * kps[:, :2] + self.alpha * prev[:, :2]
        out[:, :2] = conf_mask * out[:, :2] + (1 - conf_mask) * kps[:, :2]

        self.buf[gid] = out
        return out
