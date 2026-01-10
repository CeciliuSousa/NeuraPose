# ==============================================================
# src/modulos/tracking.py
# ==============================================================

import cv2
import numpy as np
from pathlib import Path

class TrackHistory:
    """Mantém histórico de duração de tracks para análise."""
    def __init__(self):
        self.data = {}

    def update(self, raw_tid, t_seconds):
        if raw_tid not in self.data:
            self.data[raw_tid] = {"start": t_seconds, "end": t_seconds}
        else:
            self.data[raw_tid]["end"] = t_seconds

    def save_txt(self, path: Path):
        lines = [
            "ID Original | IDs Associados | Início (s) | Fim (s) | Duração (s)",
            "-" * 70,
        ]
        for tid, rec in self.data.items():
            dur = rec["end"] - rec["start"]
            lines.append(
                f"{tid:<12} | -               | {rec['start']:>6.2f}     | "
                f"{rec['end']:>6.2f}     | {dur:>6.2f}"
            )

        with open(path, "w") as f:
            f.write("\n".join(lines))


def bbox_clip(x1, y1, x2, y2, w, h):
    """Garante que bbox esteja dentro dos limites da imagem."""
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    return x1, y1, x2, y2


def compute_hist_descriptor(frame, bbox):
    """Calcula histograma HSV para ReID simples (fallback)."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox_clip(*bbox, W, H)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    hsv = cv2.merge([h, s, v])
    hist = cv2.calcHist([hsv], [0, 1, 2], None,
                    [16, 16, 16], [0, 180, 0, 256, 0, 256])

    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def cosine_sim(a, b):
    """Calcula similaridade de cosseno entre dois vetores."""
    if a is None or b is None:
        return -1.0
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))
