# ==============================================================
# neurapose-backend/app/modulos/rtmpose.py
# ==============================================================

import cv2
import numpy as np
from neurapose_backend.app.configuracao.config import (
    SIMCC_W, SIMCC_H, SIMCC_SPLIT_RATIO, MEAN, STD
)


def decode_simcc_output(simcc_x, simcc_y, split_ratio=SIMCC_SPLIT_RATIO):
    """
    Decodifica a saída SimCC do RTMPose para coordenadas (x, y).
    simcc_x/y: (B, K, Lx/Ly)
    Retorna:
      coords: (B, K, 2) em [0..W), [0..H)
      conf : (B, K)
    """
    # Argmax ao longo dos eixos discretizados
    x_idx = np.argmax(simcc_x, axis=2)
    y_idx = np.argmax(simcc_y, axis=2)

    # Confiabilidade simples baseada no pico
    conf = np.sqrt(np.max(simcc_x, axis=2) * np.max(simcc_y, axis=2)).astype(np.float32)

    # Reescala dos índices pelo split_ratio para voltar ao grid original
    x = (x_idx.astype(np.float32) / split_ratio)
    y = (y_idx.astype(np.float32) / split_ratio)

    coords = np.stack([x, y], axis=2).astype(np.float32)
    return coords, conf


def preprocess_rtmpose_input(bgr_crop):
    """
    Converte BGR->RGB, normaliza por mean/std, redimensiona e reordena para NCHW float32.
    """
    # 1️ Redimensiona para o tamanho padrão do modelo RTMPose
    resized = cv2.resize(bgr_crop, (SIMCC_W, SIMCC_H))  # (W, H) = (192, 256)

    # 2️ Converte BGR → RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)

    # 3️ Normaliza com médias e desvios padrão (ImageNet)
    rgb = (rgb - MEAN) / STD

    # 4️ Reordena para formato NCHW
    chw = rgb.transpose(2, 0, 1)[None]  # NCHW

    return chw.astype(np.float32)
