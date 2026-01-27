# ==============================================================
# neurapose-backend/app/modulos/inferencia_lstm.py
# ==============================================================

import torch
import numpy as np
import neurapose_backend.config_master as cm


def rodar_lstm_uma_sequencia(seq_np, model, mu, sigma):
    """
    Roda o modelo temporal (LSTM) em UMA sequência (2,T,17).
    Retorna:
      score_classe2: probabilidade da classe 1 (CLASSE2)
      pred_raw: rótulo argmax (0 ou 1), sem threshold extra
    """
    x = torch.tensor(seq_np, dtype=torch.float32).unsqueeze(0).to(cm.DEVICE)

    # Normalização se mu/sigma existirem
    if mu is not None and sigma is not None:
        B, C, T, V = x.shape
        # Permuta para (B, T, C, V) -> (B, T, C*V) para igualar ao treino
        xf = x.permute(0, 2, 1, 3).reshape(B, T, C * V)

        mu_f = mu.to(cm.DEVICE).reshape(1, 1, 34)
        sigma_f = sigma.to(cm.DEVICE).reshape(1, 1, 34)

        # Aplica normalização
        xf = (xf - mu_f) / sigma_f.clamp_min(1e-6)
        
        # Retorna ao formato original (B, C, T, V) para o modelo (que vai chamar ensure_BTF depois)
        # (B, T, C*V) -> (B, T, C, V) -> (B, C, T, V)
        x = xf.reshape(B, T, C, V).permute(0, 2, 1, 3)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        score_classe2 = float(probs[:, 1].item())
        pred_raw = int(torch.argmax(probs, dim=1).item())

    return score_classe2, pred_raw


def rodar_lstm_batch(sequences_dict, model, mu, sigma, batch_size=64):
    """
    Roda inferência LSTM para múltiplos IDs simultaneamente (Batch Processing).
    
    Args:
        sequences_dict (dict): {id: np.ndarray (C, T, V)}
        model: Modelo PyTorch carregado.
        mu, sigma: Estatísticas para normalização.
        batch_size: Tamanho do mini-batch para inferência.
        
    Returns:
        tuple: (dict_preds, dict_scores) onde:
            dict_preds: {id: classe_id (0 ou 1)}
            dict_scores: {id: score_float}
    """
    if not sequences_dict:
        return {}, {}
        
    ids = list(sequences_dict.keys())
    arrays = list(sequences_dict.values())
    
    # Empilha todos em um tensor (Total, C, T, V)
    # Ex: (100, 2, 30, 17)
    batch_tensor = torch.tensor(np.array(arrays), dtype=torch.float32).to(cm.DEVICE)
    
    # Normalização em Batch
    if mu is not None and sigma is not None:
        B, C, T, V = batch_tensor.shape
        # (B, C, T, V) -> (B, T, C, V) -> (B, T, C*V)
        xf = batch_tensor.permute(0, 2, 1, 3).reshape(B, T, C * V)
        
        mu_f = mu.to(cm.DEVICE).reshape(1, 1, 34)
        sigma_f = sigma.to(cm.DEVICE).reshape(1, 1, 34)
        
        xf = (xf - mu_f) / sigma_f.clamp_min(1e-6)
        
        # Volta para (B, C, T, V)
        batch_tensor = xf.reshape(B, T, C, V).permute(0, 2, 1, 3)
        
    results_probs = []
    
    # Processa em mini-batches para não estourar VRAM se N for muito grande
    with torch.no_grad():
        total = batch_tensor.size(0)
        for i in range(0, total, batch_size):
            batch_x = batch_tensor[i : i + batch_size]
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)
            results_probs.append(probs)
            
    # Concatena resultados
    if results_probs:
        all_probs = torch.cat(results_probs, dim=0)
    else:
        return {}, {}
        
    # Mapeia de volta para os IDs
    dict_preds = {}
    dict_scores = {}
    
    for idx, gid in enumerate(ids):
        prob_classe2 = float(all_probs[idx, 1].item())
        pred_raw = int(torch.argmax(all_probs[idx]).item())
        
        dict_scores[gid] = prob_classe2
        # Aplica threshold padrão aqui para consistência ou retorna raw
        # Aqui retornamos raw score, o threshold pode ser aplicado fora se desejar,
        # mas geralmente o 'pred_raw' é o argmax.
        # Para consistência com código antigo que aplicava CLASSE2_THRESHOLD fora:
        # Vamos retornar o que o modelo "vê"
        
        dict_preds[gid] = pred_raw # Nota: Caller pode re-aplicar threshold no score
        
    return dict_preds, dict_scores