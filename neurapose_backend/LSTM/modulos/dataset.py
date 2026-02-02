# ================================================================
# neurapose-backend/app/LSTM/modulos/dataset.py
# ================================================================

import torch
from torch.utils.data import TensorDataset

def load_data_pt(path):
    """
    Carrega o dataset salvo no formato PyTorch (.pt) contendo o dicionário:
    {'data': Tensor(N, C, T, V), 'labels': Tensor(N)}
    """
    # CORREÇÃO: Usar torch.load para carregar o arquivo .pt
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
    
    # 1. Permutar: (N, C, T, V) -> (N, T, C, V)
    X = X.permute(0, 2, 1, 3) 
    
    # 2. Achatar Features: (N, T, C, V) -> (N, T, C*V)
    X = X.reshape(X.shape[0], X.shape[1], -1) 
    
    # X final é (N, 30, 34)
    
    # X final é (N, 30, 34)
    
    # [NOVO] Suporte a Metadados (ID, Cena, Clip)
    # Se o arquivo .pt tiver a chave 'metadata', nós a carregamos.
    # Metadata shape esperado: (N, 4) -> [scene, clip, pid, sample_idx]
    metadata = None
    if isinstance(data_dict, dict) and "metadata" in data_dict:
        # Carrega e converte para tensor long se não for
        meta_raw = data_dict["metadata"]
        if isinstance(meta_raw, list):
            metadata = torch.tensor(meta_raw, dtype=torch.long)
        elif isinstance(meta_raw, torch.Tensor):
            metadata = meta_raw.long()
    
    # Retorna o TensorDataset e os componentes separados
    # Se metadata existir, incluimos no TensorDataset para ser batched pelo DataLoader
    if metadata is not None:
        # Garante que metadata tenha mesmo N
        if len(metadata) == len(y):
             return TensorDataset(X, y, metadata), y, metadata
        else:
             print(f"[AVISO] Metadata len ({len(metadata)}) != Data len ({len(y)}). Ignorando metadata.")
             return TensorDataset(X, y), y, None
    
    return TensorDataset(X, y), y, None