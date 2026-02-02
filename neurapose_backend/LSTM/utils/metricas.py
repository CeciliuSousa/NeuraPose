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

import neurapose_backend.config_master as cm


def counts_from_labels(y_tensor):
    c = Counter(y_tensor.tolist())
    return c.get(0, 0), c.get(1, 0)

def evaluate(model, dataloader, device, criterion=None, class_names=[cm.CLASSE1, cm.CLASSE2]):
    model.eval()
    
    # Listas para métricas baseada em FRAMES (padrão)
    preds_frame, labels_frame = [], []
    
    # Dicionario para agrupar por ID: pid -> {'true': label, 'scores': [], 'preds': []}
    id_results = {}
    
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            # Handle variable unpacking (compatibilidade com ou sem metadata)
            if len(batch) == 3:
                xb, yb, meta = batch
                meta = meta.to(device)
            else:
                xb, yb = batch
                meta = None
                
            xb, yb = xb.to(device), yb.to(device)
            
            out = model(xb)
            
            # Predições Frame-Level
            pred_classes = torch.argmax(out, dim=1).cpu().numpy()
            true_classes = yb.cpu().numpy()
            
            # Scores da classe positiva (1) para acumular
            # Softmax para ter probabilidade real se model nao tiver
            probs = torch.softmax(out, dim=1)
            pos_scores = probs[:, 1].cpu().numpy()
            
            preds_frame.extend(pred_classes)
            labels_frame.extend(true_classes)
            
            if criterion:
                total_loss += criterion(out, yb).item()

            # Agrupamento por ID (se metadata existir)
            if meta is not None:
                meta_np = meta.cpu().numpy() # [scene, clip, pid, sample_idx]
                for i in range(len(true_classes)):
                    # [CORRECTION] Unique ID = combination of scene, clip, and pid
                    scene = int(meta_np[i, 0])
                    clip = int(meta_np[i, 1])
                    pid = int(meta_np[i, 2])
                    
                    # Composite key (string for dict)
                    unique_id = f"{scene}_{clip}_{pid}"
                    
                    true_cls = int(true_classes[i])
                    score = float(pos_scores[i])
                    pred_cls = int(pred_classes[i])
                    
                    if unique_id not in id_results:
                        id_results[unique_id] = {'pid': pid, 'true': true_cls, 'scores': [], 'votes': []}
                    
                    # Consistência check (opcional, assume-se que ID tem mesma label sempre)
                    id_results[unique_id]['scores'].append(score)
                    id_results[unique_id]['votes'].append(pred_cls)

    # 1. Métricas Frame-Level (Clássicas)
    acc = accuracy_score(labels_frame, preds_frame)
    f1m = f1_score(labels_frame, preds_frame, average="macro")
    f1w = f1_score(labels_frame, preds_frame, average="weighted")
    report_frame = classification_report(labels_frame, preds_frame, target_names=class_names, output_dict=True)
    avg_loss = total_loss / max(1, len(dataloader))

    # 2. Métricas ID-Level (Granulares)
    # Se não tiver metadata, retorna padrao zerado ou igual frame
    report_id_level = []
    
    y_true_id = []
    y_pred_id = []
    
    if id_results:
        for uid, data in id_results.items():
            # Decisão por Voto Majoritário (mais robusto que média de score as vezes)
            votes = data['votes']
            final_pred = Counter(votes).most_common(1)[0][0]
            
            # Decisão por Média de Score (alternativa)
            # avg_score = np.mean(data['scores'])
            # final_pred = 1 if avg_score >= 0.5 else 0
            
            true_cls = data['true']
            pid_val = data['pid'] # Original PID for display
            
            y_true_id.append(true_cls)
            y_pred_id.append(final_pred)
            
            # Dados para tabela detalhada
            truth_str = class_names[true_cls]
            pred_str = class_names[final_pred]
            
            # Score médio para exibição de confiança
            avg_conf = float(np.mean(data['scores']))
            if final_pred == 0: avg_conf = 1.0 - avg_conf
            
            is_ok = (final_pred == true_cls)
            
            report_id_level.append({
                "id": pid_val, # Use original PID for display
                "uid": uid,    # Keep internal unique ID
                "real": truth_str.upper(),
                "predito": pred_str.upper(),
                "conf": round(avg_conf * 100, 2),
                "ok": is_ok,
                "status_symbol": "✓" if is_ok else "✗"
            })
        
        # Ordena por UID para consistência
        report_id_level.sort(key=lambda x: x['uid'])
        
        # Recalcula métricas globais baseadas em ID para o relatório de validação final
        # Isso atende ao pedido do usuario de que "metricas devem ser salvas a partir desse detalhe"
        # Vamos retornar essas métricas "reais" (ID-based) substituindo ou complementando as frame-based?
        # O user disse: "é dai que vamos tirar as metricas de treinamento e teste!"
        # Então vamos calcular Acc/F1 de ID também.
        acc_id = accuracy_score(y_true_id, y_pred_id)
        f1m_id = f1_score(y_true_id, y_pred_id, average="macro")
        f1w_id = f1_score(y_true_id, y_pred_id, average="weighted")
        cm_id = confusion_matrix(y_true_id, y_pred_id)
        
        # Retorna metricas de ID como principais se houver dados de ID, 
        # mas mantemos avg_loss (frame-level) pois loss é calculada por sample.
        return avg_loss, acc_id, f1m_id, f1w_id, report_frame, preds_frame, labels_frame, cm_id, report_id_level

    # Fallback se sem metadata (usa frame-level e lista detalhada vazia)
    cm_frame = confusion_matrix(labels_frame, preds_frame)
    return avg_loss, acc, f1m, f1w, report_frame, preds_frame, labels_frame, cm_frame, []

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
