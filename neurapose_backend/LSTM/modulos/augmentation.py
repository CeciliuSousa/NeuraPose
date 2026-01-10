# ================================================================
# neurapose-backend/app/LSTM/modulos/augmentation.py
# ================================================================

import torch
import numpy as np

class TimeSeriesAugmenter:
    """
    Classe para aplicar data augmentation em séries temporais de esqueletos (N, T, F).
    """

    @staticmethod
    def add_noise(x, noise_level=0.01):
        """Adiciona ruído gaussiano."""
        noise = torch.randn_like(x) * noise_level
        return x + noise

    @staticmethod
    def scale(x, sigma=0.1):
        """Multiplica por um fator de escala aleatório (scaling)."""
        # Fator de escala ~ N(1, sigma)
        factor = torch.randn(x.size(0), 1, 1, device=x.device) * sigma + 1.0
        return x * factor

    @staticmethod
    def time_shift(x, shift_max=2):
        """Desloca a série temporal para esquerda ou direita (preenche com zero ou repete)."""
        # Implementação simplificada: shift circular ou preenchimento
        # Aqui faremos um roll simples para cada amostra no batch
        B, T, F = x.shape
        shift = np.random.randint(-shift_max, shift_max + 1)
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=1)

    @staticmethod
    def mask_features(x, mask_prob=0.05):
        """Zera aleatoriamente algumas features (dropout no input)."""
        mask = torch.rand_like(x) > mask_prob
        return x * mask.float()

    @staticmethod
    def augment(x):
        """Aplica uma combinação aleatória de augmentations."""
        # x: (T, F) ou (B, T, F) - assumindo dataset retorna (T, F)
        
        # Se for apenas (T, F), adiciona dimensão de batch fake para processar
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Aplica augmentations com certa probabilidade
        if np.random.rand() < 0.5:
            x = TimeSeriesAugmenter.add_noise(x, noise_level=0.02)
        
        if np.random.rand() < 0.5:
            x = TimeSeriesAugmenter.scale(x, sigma=0.1)

        if np.random.rand() < 0.3:
            x = TimeSeriesAugmenter.time_shift(x, shift_max=2)
            
        if np.random.rand() < 0.3:
            x = TimeSeriesAugmenter.mask_features(x, mask_prob=0.05)

        if squeeze:
            x = x.squeeze(0)
            
        return x
