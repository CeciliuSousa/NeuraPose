# ===========================================
# neurapose_backend/LSTM/modulos/custom_dataset.py
# ===========================================

import torch
from torch.utils.data import Dataset
from neurapose_backend.LSTM.modulos.augmentation import TimeSeriesAugmenter


class AugmentedDataset(Dataset):
    def __init__(self, X, y, meta=None, augment=False):
        """
        Args:
            X (Tensor): Dados de entrada (N, T, F).
            y (Tensor): Rótulos (N,).
            meta (Tensor, optional): Metadados (N, 4).
            augment (bool): Se True, aplica data augmentation on-the-fly.
        """
        self.X = X
        self.y = y
        self.meta = meta
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_sample = self.X[idx]
        y_sample = self.y[idx]

        if self.augment:
            x_sample = TimeSeriesAugmenter.augment(x_sample)

        # Retorna tupla com 3 ou 2 elementos
        if self.meta is not None:
            meta_sample = self.meta[idx]
            return x_sample, y_sample, meta_sample
            
        return x_sample, y_sample
