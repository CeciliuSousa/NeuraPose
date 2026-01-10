import torch
from torch.utils.data import Dataset
from ..modulos.augmentation import TimeSeriesAugmenter

class AugmentedDataset(Dataset):
    def __init__(self, X, y, augment=False):
        """
        Args:
            X (Tensor): Dados de entrada (N, T, F).
            y (Tensor): Rótulos (N,).
            augment (bool): Se True, aplica data augmentation on-the-fly.
        """
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_sample = self.X[idx]
        y_sample = self.y[idx]

        if self.augment:
            # Aplica augmentation. O método augment espera tensor.
            # Como estamos pegando uma amostra, x_sample é (T, F).
            # O augment lida com isso.
            x_sample = TimeSeriesAugmenter.augment(x_sample)

        return x_sample, y_sample
