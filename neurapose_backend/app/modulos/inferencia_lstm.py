# ==============================================================
# neurapose-backend/app/modulos/inferencia_lstm.py
# ==============================================================

import torch
from neurapose_backend.app.configuracao.config import DEVICE


def rodar_lstm_uma_sequencia(seq_np, model, mu, sigma):
    """
    Roda o modelo temporal (LSTM) em UMA sequência (2,T,17).
    Retorna:
      score_classe2: probabilidade da classe 1 (CLASSE2)
      pred_raw: rótulo argmax (0 ou 1), sem threshold extra
    """
    x = torch.tensor(seq_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Normalização se mu/sigma existirem
    if mu is not None and sigma is not None:
        B, C, T, V = x.shape
        # Permuta para (B, T, C, V) -> (B, T, C*V) para igualar ao treino
        xf = x.permute(0, 2, 1, 3).reshape(B, T, C * V)

        mu_f = mu.to(DEVICE).reshape(1, 1, 34)
        sigma_f = sigma.to(DEVICE).reshape(1, 1, 34)

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