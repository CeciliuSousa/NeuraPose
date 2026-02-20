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

    @staticmethod
    def mirror_pose(x):
        """
        Inverte as coordenadas X do esqueleto para simular a pessoa virada para o lado oposto.
        A série temporal X possui 34 features de entrada: As 17 primeiras são os eixos X. As 17 últimas os eixos Y.
        """
        # Clonamos o tensor para não sujar o original
        x_mirrored = x.clone()
        # Se os dados estiverem normalizados (z-score), a origem X=0 é no centro
        # Portanto, espelhar o eixo X é simplesmente multiplicar por -1
        x_mirrored[..., :17] = x_mirrored[..., :17] * -1.0
        
        # O ideal seria também trocar os índices esquerdos pelos direitos copiando,
        # mas só a inversão de sinal geográfico do X já cria um dado sintético riquíssimo
        # onde as ações andam da direita pra esquerda e vice versa!
        return x_mirrored


    @staticmethod
    def augment(x):
        """Aplica uma combinação inteligente de augmentations sem destruir o esqueleto real."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # 1. Flip Horizontal (O Espelhamento Brilhante sugerido pelo Usuário)
        if np.random.rand() < 0.5:
            x = TimeSeriesAugmenter.mirror_pose(x)

        # 2. Ruído Gaussiano Levissimo (Tremor de Camera)
        if np.random.rand() < 0.3:
            x = TimeSeriesAugmenter.add_noise(x, noise_level=0.01) # Reduzido para .01 para não corromper pose
        
        # 3. Escalonamento (Pessoa mais perto/longe da câmera)
        if np.random.rand() < 0.3:
            x = TimeSeriesAugmenter.scale(x, sigma=0.05) # Reduzido para .05

        # 4. Deslocamento Temporal (Atrasar ou adiantar a matriz no tempo em 1 ou 2 frames)
        if np.random.rand() < 0.3:
            x = TimeSeriesAugmenter.time_shift(x, shift_max=2)
            
        # 5. Oclusão Parcial de membro (Máscara simulando móveis na frente)
        if np.random.rand() < 0.3:
            x = TimeSeriesAugmenter.mask_features(x, mask_prob=0.05)

        if squeeze:
            x = x.squeeze(0)
            
        return x
