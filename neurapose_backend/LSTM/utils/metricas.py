# ================================================================
# neurapose-backend/app/LSTM/utils/metricas.py
# ================================================================

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from neurapose_backend.config_master import CLASSE1, CLASSE2


def counts_from_labels(y_tensor):
    c = Counter(y_tensor.tolist())
    return c.get(0, 0), c.get(1, 0)

def evaluate(model, dataloader, device, criterion=None, class_names=[CLASSE1, CLASSE2]):
    model.eval()
    preds, labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            preds.extend(pred)
            labels.extend(yb.cpu().numpy())
            if criterion:
                total_loss += criterion(out, yb).item()

    acc = accuracy_score(labels, preds)
    f1m = f1_score(labels, preds, average="macro")
    f1w = f1_score(labels, preds, average="weighted")
    report = classification_report(labels, preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(labels, preds)
    avg_loss = total_loss / max(1, len(dataloader))
    return avg_loss, acc, f1m, f1w, report, preds, labels, cm

def plot_curves(history, out_path_png):
    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(11, 8))
    plt.subplot(3,1,1)
    plt.plot(epochs, [h["train_loss"] for h in history], label="Train Loss")
    plt.plot(epochs, [h["val_loss"] for h in history], label="Val Loss")
    plt.legend(); plt.title("Loss"); plt.grid(alpha=0.3)

    plt.subplot(3,1,2)
    plt.plot(epochs, [h["train_acc"] for h in history], label="Train Acc")
    plt.plot(epochs, [h["val_acc"] for h in history], label="Val Acc")
    plt.legend(); plt.title("Acurácia"); plt.grid(alpha=0.3)

    plt.subplot(3,1,3)
    plt.plot(epochs, [h["train_f1m"] for h in history], label="Train F1 Macro")
    plt.plot(epochs, [h["val_f1m"] for h in history], label="Val F1 Macro")
    plt.legend(); plt.title("F1 Macro"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path_png, dpi=150); plt.close()

def plot_confusion_matrix(cm, class_names, out_path_png, title="Matriz de Confusão (Validação)"):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Real",
        xlabel="Predito",
        title=title
    )
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout(); plt.savefig(out_path_png, dpi=150); plt.close()
