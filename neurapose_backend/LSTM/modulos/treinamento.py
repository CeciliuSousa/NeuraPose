# ================================================================
# neurapose-backend/LSTM/modulos/treinamento.py
# ================================================================

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, y_true, y_pred = 0.0, [], []
    for xb, yb in loader:
        from neurapose_backend.globals.state import state
        if state.stop_requested:
            break
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
        y_true.extend(yb.cpu().numpy())
        y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())

    avg_loss = running_loss / len(loader)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    f1w = f1_score(y_true, y_pred, average="weighted")
    return avg_loss, acc, f1m, f1w


class EarlyStopper:
    def __init__(self, patience=10, min_delta_f1=0.002, min_delta_loss=0.001):
        self.patience = patience
        self.min_delta_f1 = min_delta_f1
        self.min_delta_loss = min_delta_loss
        self.best_f1 = -np.inf
        self.best_loss = np.inf
        self.counter = 0

    def should_stop(self, f1, loss):
        improved_f1 = f1 > self.best_f1 + self.min_delta_f1
        improved_loss = loss < self.best_loss - self.min_delta_loss
        if improved_f1 or improved_loss:
            self.best_f1 = max(self.best_f1, f1)
            self.best_loss = min(self.best_loss, loss)
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
