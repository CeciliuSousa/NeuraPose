# ================================================================
# neurapose-backend/app/LSTM/pipeline/treinador.py
# ================================================================

"""
Script principal para treinamento do modelo temporal (TFT/LSTM).
Configuracoes vem do config_master.py
"""

import json
import torch
import numpy as np
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from colorama import Fore, init
import os
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit

# Imports do projeto
from ..configuracao.config import get_config
from ..modulos.dataset import load_data_pt
from ..utils.metricas import (
    counts_from_labels, evaluate, plot_curves, plot_confusion_matrix
)
from ..modulos.treinamento import train_one_epoch, EarlyStopper
from ..modulos.loss import FocalLoss
from ..modulos.custom_dataset import AugmentedDataset

# Configuracoes centralizadas
from neurapose_backend.config_master import (
    MODEL_BEST_FILENAME,
    MODEL_FINAL_FILENAME,
    NORM_STATS_FILENAME,
    CLASS_NAMES,
    DEVICE,
)

init(autoreset=True)


def get_gpu_info():
    """Retorna nome e memoria da GPU."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return f"{gpu_name} ({gpu_memory:.1f} GB)"
    return "CPU (sem GPU)"


def main():
    """Pipeline principal de treinamento."""
    args, ModelClass = get_config()
    device = torch.device(DEVICE)

    # micro-otimizacao: ativa benchmark para cuDNN quando em GPU (melhora throughput)
    if device.type == 'cuda':
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
    
    # Banner
    print("\n" + "="*70)
    print("NEURAPOSE - TREINAMENTO")
    print("="*70)
    print(f"[DISPOSITIVO] {get_gpu_info()}")
    print(f"[MODELO] {args.model.upper()}")
    print(f"[DATASET] {args.name}")
    print("="*70 + "\n")

    # Nomes das classes
    primeiraClasse = CLASS_NAMES[0].lower()
    segundaClasse = CLASS_NAMES[1].lower()
    
    # Carrega dataset
    DATA_PATH = args.dataset
    full_ds, y_all = load_data_pt(DATA_PATH)
    X_all, y_all = full_ds.tensors
    
    # Verifica dimensao temporal
    time_dimension = X_all.shape[1]
    if time_dimension != 30:
        print(Fore.RED + f"[ERRO] Dimensao T={time_dimension}, esperado T=30")
        return

    # Split treino/validacao estratificado 80/20
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(y_all)), y_all.cpu().numpy()))
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    # Normalizacao z-score
    mu = X_train.mean(dim=(0,1), keepdim=True)
    sigma = X_train.std(dim=(0,1), keepdim=True).clamp_min(1e-6)
    X_train = (X_train - mu) / sigma
    X_val = (X_val - mu) / sigma

    # Analise de balanceamento
    n0, n1 = counts_from_labels(y_train)
    total = n0 + n1
    ratio = max(n0, n1) / max(1, min(n0, n1))
    
    print(Fore.CYAN + "\n[BALANCEAMENTO]")
    print(f"  Total treino: {total}")
    print(f"  {primeiraClasse}: {n0} | {segundaClasse}: {n1}")
    print(f"  Razao: {ratio:.2f}x")

    # Sampler ponderado se desbalanceado
    if ratio > 1.1:
        print(Fore.YELLOW + f"  Aplicando sampler ponderado")
        weights_per_class = torch.tensor([1.0 / n0, 1.0 / n1])
        sample_weights = weights_per_class[y_train]
        train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        balanced_mode = True
    else:
        print(Fore.GREEN + "  Dataset balanceado")
        train_sampler = None
        balanced_mode = False

    # DataLoaders
    train_ds = AugmentedDataset(X_train, y_train, augment=True)
    val_ds = AugmentedDataset(X_val, y_val, augment=False)

    # Configuracao de DataLoader otimizada
    num_workers = min(4, os.cpu_count() or 1)
    pin_memory = True if device.type == 'cuda' else False

    if balanced_mode:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # Pesos para loss
    inv_freq = torch.tensor([1.0 / n0, 1.0 / n1], device=device)
    weights = inv_freq / inv_freq.sum() * 2
    
    print(Fore.YELLOW + f"\n[LOSS] Focal Loss (gamma=2.0)")
    criterion = FocalLoss(alpha=weights, gamma=2.0).to(device)

    # Modelo e otimizador
    model = ModelClass(input_size=X_train.shape[2]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=min(args.lr, 3e-4), weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=12, min_lr=1e-6)

    # Diretorio de saida
    model_name = args.name
    model_dir = Path(args.output_dir).parent / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    best_model_path = model_dir / MODEL_BEST_FILENAME
    final_model_path = model_dir / MODEL_FINAL_FILENAME
    curves_path = model_dir / "curvas_treino_validacao.png"
    cm_val_path = model_dir / "matriz_confusao_validacao.png"

    # Salva estatisticas de normalizacao
    torch.save({"mu": mu.cpu(), "sigma": sigma.cpu()}, model_dir / NORM_STATS_FILENAME)

    print(Fore.CYAN + f"\n[PATHS]")
    print(f"  Modelo: {model_dir}")

    # Early stopping
    early = EarlyStopper(patience=12)
    best_val_f1, best_epoch = -1, -1
    history = []

    # Loop de treinamento
    print(Fore.CYAN + "\n[TREINO]")
    for epoch in range(1, args.epochs + 1):
        with tqdm(total=len(train_loader), desc=f"Epoca {epoch}", ncols=120) as pbar:
            tr_loss, tr_acc, tr_f1m, tr_f1w = train_one_epoch(model, train_loader, optimizer, criterion, device)
            va_loss, va_acc, va_f1m, va_f1w, _, _, _, _ = evaluate(model, val_loader, device, criterion)

            scheduler.step(va_f1m)

            history.append({
                "epoch": epoch,
                "train_loss": tr_loss, "val_loss": va_loss,
                "train_acc": tr_acc, "val_acc": va_acc,
                "train_f1m": tr_f1m, "val_f1m": va_f1m
            })

            if va_f1m > best_val_f1:
                best_val_f1, best_epoch = va_f1m, epoch
                torch.save(model.state_dict(), best_model_path)

            pbar.set_postfix({"TrLoss": f"{tr_loss:.4f}", "ValF1": f"{va_f1m:.4f}"})
            pbar.update(len(train_loader))
            pbar.write(f"[{epoch:03d}] Loss: {tr_loss:.4f}/{va_loss:.4f} | F1: {tr_f1m:.4f}/{va_f1m:.4f}")

        if early.should_stop(va_f1m, va_loss):
            print(Fore.MAGENTA + f"[STOP] Early stopping na epoca {epoch}")
            break

    # Avaliacao final
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    tr_loss, tr_acc, tr_f1m, tr_f1w, _, _, _, _ = evaluate(model, train_loader, device, criterion)
    va_loss, va_acc, va_f1m, va_f1w, va_report, _, _, va_cm = evaluate(model, val_loader, device, criterion)

    # Salva graficos e modelo final
    plot_curves(history, curves_path)
    plot_confusion_matrix(va_cm, [primeiraClasse, segundaClasse], cm_val_path)
    torch.save(model.state_dict(), final_model_path)

    # Relatorio
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = model_dir / f"relatorio_{timestamp}.txt"
    hist_json_path = model_dir / f"historico_{timestamp}.json"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("RELATORIO DE TREINAMENTO\n")
        f.write(f"Data: {timestamp}\n")
        f.write(f"Dispositivo: {get_gpu_info()}\n")
        f.write(f"Modelo: {args.model}\n")
        f.write(f"Epocas: {args.epochs}\n")
        f.write(f"Melhor epoca: {best_epoch}\n")
        f.write(f"Val F1 (best): {best_val_f1:.4f}\n")

    with open(hist_json_path, "w", encoding="utf-8") as jf:
        json.dump(history, jf, indent=2)

    print(Fore.GREEN + "\n[SUCESSO] Treinamento concluido!")
    print(f"Modelo: {best_model_path}")
    print(f"F1 (val): {va_f1m:.4f}")


if __name__ == "__main__":
    main()