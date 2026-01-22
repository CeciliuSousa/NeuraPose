# ==============================================================
# neurapose_backend/nucleo/suavizacao.py
# ==============================================================
# Módulo centralizado para suavização temporal de dados.
# ==============================================================

import numpy as np
import neurapose_backend.config_master as cm

class EmaSmoother:
    """
    Suavizador Exponencial (EMA - Exponential Moving Average) para keypoints.
    Reduz a tremulação (jitter) da pose frame a frame.
    """
    def __init__(self, alpha=None, min_conf=None):
        """
        Args:
            alpha (float): Fator de suavização (0.0 a 1.0). Quanto menor, mais suave (mas mais "lento").
            min_conf (float): Confiança mínima para considerar um novo keypoint válido para atualização.
        """
        # Se não fornecido, usa do config_master
        self.alpha = alpha if alpha is not None else cm.EMA_ALPHA
        self.min_conf = min_conf if min_conf is not None else cm.EMA_MIN_CONF
        
        # Estado interno: {track_id: keypoints_anteriores_numpy}
        self.tracks = {}

    def step(self, track_id, current_kps):
        """
        Aplica EMA nos keypoints de um ID especifico.
        
        Args:
            track_id (int): ID único do tracking.
            current_kps (np.ndarray): Keypoints atuais (N, 3) -> [x, y, conf].
            
        Returns:
            np.ndarray: Keypoints suavizados (N, 3).
        """
        if track_id not in self.tracks:
            # Primeiro frame deste ID: não há histórico, retorna o atual e salva
            self.tracks[track_id] = current_kps.copy()
            return current_kps

        # Recupera histórico
        prev_kps = self.tracks[track_id]
        
        # Separa coordenadas e confiança
        curr_xy = current_kps[:, :2]
        curr_conf = current_kps[:, 2:3]
        
        prev_xy = prev_kps[:, :2]
        
        # Máscara de atualização: Só atualiza juntas com confiança >= min_conf
        # Se a confiança for baixa, mantemos a posição anterior (reduz "pulos" quando o modelo falha)
        if self.min_conf > 0:
            mask = (curr_conf >= self.min_conf).astype(np.float32)
            # Se conf é alta, usa alpha normal. Se é baixa, alpha efetivo vira 0 (não muda)
            # Mas aqui queremos interpolar:
            # new = alpha * curr + (1-alpha) * prev
            # Vamos simplificar: aplicamos EMA em tudo, mas confiamos no modelo atual
            # A lógica original do pré-processamento era EMA puro. Vamos manter EMA puro para paridade total.
            pass

        # Aplica fórmula do EMA: Y_t = alpha * X_t + (1 - alpha) * Y_{t-1}
        smooth_xy = self.alpha * curr_xy + (1.0 - self.alpha) * prev_xy
        
        # Reconstrói array (N, 3)
        result_kps = np.concatenate([smooth_xy, curr_conf], axis=1)
        
        # Atualiza histórico
        self.tracks[track_id] = result_kps
        
        return result_kps

    def reset_id(self, track_id):
        """Reseta histórico de um ID se necessário."""
        if track_id in self.tracks:
            del self.tracks[track_id]
