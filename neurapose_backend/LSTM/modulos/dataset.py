# ================================================================
# LSTM/modulos/dataset.py
# ================================================================
# Carregamento e preparação de dados para o LSTM.

import pickle # Mantido para compatibilidade, mas não usado para carregar .pt
import numpy as np
import torch
from torch.utils.data import TensorDataset

def load_data_pt(path):
    """
    Carrega o dataset salvo no formato PyTorch (.pt) contendo o dicionário:
    {'data': Tensor(N, C, T, V), 'labels': Tensor(N)}
    """
    # ⚠️ CORREÇÃO: Usar torch.load para carregar o arquivo .pt
    data_dict = torch.load(path, map_location='cpu')

    # O script de conversão já salvou 'data' e 'labels' como tensores PyTorch.
    X = data_dict["data"]  # Shape: (N, C=2, T=30, V=17)
    y = data_dict["labels"] # Shape: (N)

    # Verifica o shape para garantir que o T=30 está correto
    if X.dim() != 4 or X.shape[2] != 30:
        raise ValueError(
            f"O tensor X_data carregado tem shape {X.shape}, mas o formato esperado é (N, C, T=30, V). "
            "Verifique se o script de conversão usou max_frames=30 e salvou corretamente."
        )

    # --------------------------------------------------------------------------
    # Transformação para a entrada LSTM: (N, C, T, V) -> (N, T, C*V)
    # A LSTM espera Batch x Seq_Len x Features
    # Features por time step: C*V = 2*17 = 34
    # Seq_Len (T) = 30
    # --------------------------------------------------------------------------
    
    # 1. Permutar: (N, C, T, V) -> (N, T, C, V)
    X = X.permute(0, 2, 1, 3) 
    
    # 2. Achatar Features: (N, T, C, V) -> (N, T, C*V)
    X = X.reshape(X.shape[0], X.shape[1], -1) 
    
    # X final é (N, 30, 34)
    
    # Retorna o TensorDataset e o tensor de labels separado (para o StratifiedShuffleSplit)
    return TensorDataset(X, y), y