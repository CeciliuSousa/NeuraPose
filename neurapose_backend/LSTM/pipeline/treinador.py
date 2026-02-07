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
import psutil
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from colorama import Fore, init
import os
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit
import shutil

# Imports do projeto
from neurapose_backend.LSTM.configuracao.config import get_config
from neurapose_backend.LSTM.modulos.dataset import load_data_pt
from neurapose_backend.LSTM.utils.metricas import (
    counts_from_labels, evaluate, plot_curves, plot_confusion_matrix
)
from neurapose_backend.LSTM.modulos.treinamento import train_one_epoch, EarlyStopper
from neurapose_backend.LSTM.modulos.loss import FocalLoss
from neurapose_backend.LSTM.modulos.custom_dataset import AugmentedDataset


# Configuracoes centralizadas
# Configuracoes centralizadas
import neurapose_backend.config_master as cm


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
    DEVICE = cm.DEVICE

    # micro-otimizacao: ativa benchmark para cuDNN quando em GPU (melhora throughput)
    if DEVICE == 'cuda':
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
    
    # -------------------------------------------------------------------------
    # 1. BANNER E INFORMAÇÕES DO SISTEMA (PADRONIZADO)
    # -------------------------------------------------------------------------
    
    # Coleta info do sistema
    mem = psutil.virtual_memory()
    ram_used = mem.used / (1024**3)
    ram_total = mem.total / (1024**3)
    ram_info = f"{ram_used:.1f}GB / {ram_total:.1f}GB"
    
    gpu_name = "Não detectada"
    vram_info = "N/A"
    gpu_ok = False
    if torch.cuda.is_available():
        gpu_ok = True
        gpu_name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        vram_used = (total - free) / (1024**3)
        vram_total = total / (1024**3)
        vram_info = f"{vram_used:.1f}GB / {vram_total:.1f}GB"

    def status_ok(val=True):
        return Fore.GREEN + "[OK]" if val else Fore.RED + "[ERRO]"
    
    def status_usage(is_used):
        return Fore.GREEN + "[OK]" if is_used else Fore.YELLOW + "[ALERT]"

    # Lógica de auditoria de parâmetros por modelo
    m = args.model.lower()
    
    # Definição de quais modelos usam quais parâmetros
    # (Baseado na análise do models.py)
    usage = {
        "dataset": True,
        "epochs": True,
        "batch_size": True,
        "lr": True,
        "dropout": True,       # Maioria usa, exceto talvez alguns simples se não configurado
        "hidden_size": True,   # TFT, LSTM, Robust, BiLSTM...
        "num_layers": True,    # LSTM variants
        "num_heads": False,    # Apenas TFT e Transformer
        "kernel_size": False   # Apenas CNN/TCN/WaveNet
    }

    if "tft" in m or "transformer" in m:
        usage["num_heads"] = True
    
    if "tcn" in m or "wavenet" in m or "cnn" in m:
        usage["kernel_size"] = True
        # TCN não usa hidden_size/num_layers da mesma forma que RNNs, 
        # mas usa channels (derivado). Para simplificar a UI, marcamos como ALERT se não for óbvio.
        if "tcn" in m:
            usage["hidden_size"] = False # TCN usa channels
            usage["num_layers"] = False  # TCN usa dilations (fixo no code ou derivado)

    print(Fore.WHITE + "\n" + "="*62)
    print(Fore.WHITE + "TREINAMENTO DE MODELO — NEURAPOSE")
    print(Fore.WHITE + "="*62)
    
    # Nome amigável
    model_friendly_name = args.model.upper()
    if args.model == 'tft': model_friendly_name = "Temporal Fusion Transformer"
    elif args.model == 'lstm': model_friendly_name = "LSTM"
    
    # Impressão auditada
    # Parâmetros Universais
    print(Fore.WHITE + f"MODELO            : {status_ok(True)} {Fore.WHITE}{model_friendly_name}")
    print(Fore.WHITE + f"DATASET           : {status_ok(True)} {Fore.WHITE}{args.name}")
    print(Fore.WHITE + f"EPOCAS            : {status_usage(usage['epochs'])} {Fore.WHITE}{args.epochs}")
    print(Fore.WHITE + f"BATCH SIZE        : {status_usage(usage['batch_size'])} {Fore.WHITE}{args.batch_size}")
    print(Fore.WHITE + f"LEARNING RATE     : {status_usage(usage['lr'])} {Fore.WHITE}{args.lr}")
    
    # Parâmetros Específicos
    val_dropout = f"{usage['dropout']}" if not usage['dropout'] else f"{args.dropout}"
    print(Fore.WHITE + f"DROPOUT           : {status_usage(usage['dropout'])} {Fore.WHITE}{val_dropout if usage['dropout'] else 'Parametro não utilizado!'}")
    
    val_hidden = f"{args.hidden_size}"
    print(Fore.WHITE + f"HIDDEN SIZE       : {status_usage(usage['hidden_size'])} {Fore.WHITE}{val_hidden if usage['hidden_size'] else 'Parametro não utilizado!'}")
    
    val_layers = f"{args.num_layers}"
    print(Fore.WHITE + f"NUM LAYERS        : {status_usage(usage['num_layers'])} {Fore.WHITE}{val_layers if usage['num_layers'] else 'Parametro não utilizado!'}")

    val_heads = f"{args.num_heads}"
    print(Fore.WHITE + f"NUM HEADS         : {status_usage(usage['num_heads'])} {Fore.WHITE}{val_heads if usage['num_heads'] else 'Parametro não utilizado!'}")
    
    val_kernel = f"{args.kernel_size}"
    print(Fore.WHITE + f"KERNEL SIZE       : {status_usage(usage['kernel_size'])} {Fore.WHITE}{val_kernel if usage['kernel_size'] else 'Parametro não utilizado!'}")

    print(Fore.WHITE + "-"*62)
    
    # Hardware
    print(Fore.WHITE + f"GPU detectada     : {status_ok(gpu_ok)} {Fore.WHITE}{gpu_name}")
    print(Fore.WHITE + f"VRAM              : {status_ok(gpu_ok)} {Fore.WHITE}{vram_info}")
    print(Fore.WHITE + f"RAM               : {status_ok(True)} {Fore.WHITE}{ram_info}")
    
    print(Fore.WHITE + "="*62 + "\n")

    print(Fore.GREEN + "[OK] BANCO DE TREINAMENTO ENCONTRADO!\n")

    # -------------------------------------------------------------------------
    # 2. CARREGAMENTO DE DADOS
    # -------------------------------------------------------------------------

    # Nomes das classes
    primeiraClasse = cm.CLASS_NAMES[0].lower()
    segundaClasse = cm.CLASS_NAMES[1].lower()
    
    # Carrega dataset
    # Carrega dataset
    DATA_PATH = args.dataset
    if not Path(DATA_PATH).exists():
        print(Fore.RED + f"[ERRO] Dataset não encontrado: {DATA_PATH}")
        return

    # [UPDATED] Carrega metadata se disponivel
    load_res = load_data_pt(DATA_PATH)
    if len(load_res) == 3:
        full_ds, y_all, meta_all = load_res
    else:
        full_ds, y_all = load_res
        meta_all = None
    
    # Se full_ds ja tem metadata, nao precisamos fazer nada especial se usarmos subset
    # MAS TensorDataset não suporta StratifiedShuffleSplit direto se quisermos preservar metadata alinhado
    # Então vamos acessar o tensor de dados X_all do dataset
    # O dataset pode ter 2 ou 3 tensores
    tensors = full_ds.tensors
    X_all = tensors[0]
    # y_all ja temos
    
    # Verifica dimensao temporal
    time_dimension = X_all.shape[1]
    if time_dimension != cm.TIME_STEPS:
        print(Fore.RED + f"[ERRO] Dimensao T={time_dimension}, esperado T={cm.TIME_STEPS}")
        return

    # [UPDATED] Split treino/validacao por GRUPO (ID) para evitar vazamento
    # Se metadata existir, usamos o PID (coluna 2 do metadata) como grupo
    # Metadata shape: [scene, clip, pid, sample_idx]
    
    if meta_all is not None:
        print(Fore.BLUE + "[INFO] Utilizando Split por GRUPO (IDs Únicos) para validação rigorosa.")
        # [CORRECTION] IDs are local to video (e.g., ID 1 exists in many videos).
        # We need a global unique ID: Scene + Clip + PID
        # Metadata: [scene, clip, pid, sample_idx]
        
        # Create a unique hash/ID for each person-video instance
        # Assuming reasonable limits: scene < 1000, clip < 1000, pid < 100000
        # global_id = scene * 10^9 + clip * 10^5 + pid
        m = meta_all.clone()
        groups = (m[:, 0] * 1_000_000_000 + m[:, 1] * 100_000 + m[:, 2]).cpu().numpy()
        
        from sklearn.model_selection import GroupShuffleSplit
        # GroupShuffleSplit garante que o mesmo grupo não apareça em train e test
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(X_all, y_all, groups=groups))
        
        # Loga quantidade de IDs únicos
        unique_train = np.unique(groups[train_idx])
        unique_val = np.unique(groups[val_idx])
        print(Fore.YELLOW + f"[SPLIT] IDs ÚNICOS Treino: {len(unique_train)} | IDs ÚNICOS Validação: {len(unique_val)}")
    else:
        print(Fore.YELLOW + "[AVISO] Metadata não encontrado. Usando Split Aleatório (pode haver vazamento de ID).")
        # Fallback para StratifiedShuffleSplit se não houver metadata
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(sss.split(np.zeros(len(y_all)), y_all.cpu().numpy()))
    
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    
    # Split metadata se existir
    meta_train, meta_val = None, None
    if meta_all is not None:
        meta_train = meta_all[train_idx]
        meta_val = meta_all[val_idx]

    # Normalizacao z-score
    mu = X_train.mean(dim=(0,1), keepdim=True)
    sigma = X_train.std(dim=(0,1), keepdim=True).clamp_min(1e-6)
    X_train = (X_train - mu) / sigma
    X_val = (X_val - mu) / sigma

    # Analise de balanceamento (Treino)
    n0, n1 = counts_from_labels(y_train)
    total_train = n0 + n1
    
    # Analise de balanceamento (Validacao)
    n0_val, n1_val = counts_from_labels(y_val)
    total_val = n0_val + n1_val
    
    ratio = max(n0, n1) / max(1, min(n0, n1))
    
    print(Fore.BLUE + "[INFO] ESTATÍSTICAS DO DATASET\n")

    print(Fore.WHITE + f"   TOTAL GERAL    : {total_train + total_val}")
    print(Fore.WHITE + f"   TREINAMENTO    : {total_train} (Aprox. {total_train/(total_train+total_val or 1)*100:.0f}%)")
    print(Fore.WHITE + f"   VALIDAÇÃO      : {total_val} (Aprox. {total_val/(total_train+total_val or 1)*100:.0f}%)")
    print(Fore.WHITE + "-"*40)
    print(Fore.YELLOW + f"[TREINO] NORMAL   : {n0}")
    print(Fore.YELLOW + f"[TREINO] FURTO    : {n1}")
    print(Fore.YELLOW + f"[TREINO] RAZÃO    : {ratio:.2f}x")
    print(Fore.WHITE + "-"*40)
    print(Fore.CYAN + f"[VALID]  NORMAL   : {n0_val}")
    print(Fore.CYAN + f"[VALID]  FURTO    : {n1_val}")
    print(Fore.CYAN + f"[VALID]  TOTAL    : {total_val}\n")

    # Sampler ponderado se desbalanceado
    if ratio > 1.1:
        # print(Fore.YELLOW + f"[INFO] Aplicando sampler ponderado para corrigir desbalanceamento.")
        weights_per_class = torch.tensor([1.0 / n0, 1.0 / n1])
        sample_weights = weights_per_class[y_train]
        train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        balanced_mode = True
    else:
        print(Fore.GREEN + "[OK] DATASET BALANCEADO!\n")
        train_sampler = None
        balanced_mode = False
    
    # DataLoaders - [UPDATED] Passa metadata
    train_ds = AugmentedDataset(X_train, y_train, meta=meta_train, augment=True)
    val_ds = AugmentedDataset(X_val, y_val, meta=meta_val, augment=False)

    num_workers = min(4, os.cpu_count() or 1)
    pin_memory = True if DEVICE == 'cuda' else False

    if balanced_mode:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # Pesos para loss
    inv_freq = torch.tensor([1.0 / n0, 1.0 / n1], device=DEVICE)
    weights = inv_freq / inv_freq.sum() * 2
    
    # print(Fore.YELLOW + f"[LOSS] Focal Loss (gamma=2.0)") 
    criterion = FocalLoss(alpha=weights, gamma=2.0).to(DEVICE)

    # Modelo e otimizador
    model = ModelClass(input_size=X_train.shape[2]).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=min(args.lr, 3e-4), weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=12, min_lr=1e-6)

    # Diretorio de saida
    model_name = args.name
    model_dir = Path(args.output_dir).parent / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    best_model_path = model_dir / cm.MODEL_BEST_FILENAME

    # Salva estatisticas de normalizacao
    torch.save({"mu": mu.cpu(), "sigma": sigma.cpu()}, model_dir / cm.NORM_STATS_FILENAME)

    # Early stopping
    early = EarlyStopper(patience=12)
    best_val_f1, best_epoch = -1, -1
    history = []

    # Loop de treinamento
    from neurapose_backend.globals.state import state
    print(Fore.BLUE + "[INFO] INICIANDO TREINAMENTO...\n")
    
    digits = len(str(args.epochs))

    final_val_report_id = []

    for epoch in range(1, args.epochs + 1):
        # Verifica interrupção
        if state.stop_requested:
            print(Fore.RED + f"\n[STOP] Treinamento interrompido pelo usuário na época {epoch}")
            break

        tr_loss, tr_acc, tr_f1m, tr_f1w = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        # [UPDATED] Unpack 9 values
        va_loss, va_acc, va_f1m, va_f1w, _, _, _, _, val_report_id = evaluate(model, val_loader, DEVICE, criterion)

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
            # Salva o report detalhado da melhor época
            final_val_report_id = val_report_id

        # Log formatado: [TREINANDO] Epoca X/Y ...
        print(f"{Fore.YELLOW}[TREINANDO] Epoca {epoch:0{digits}d}/{args.epochs} | Train loss: {tr_loss:.4f} | Val loss: {va_loss:.4f} | Acc: {va_acc*100:.1f}% | F1: {va_f1m:.4f}")

        if early.should_stop(va_f1m, va_loss):
            print(Fore.MAGENTA + f"\n[STOP] Early stopping na epoca {epoch}")
            break

    if state.stop_requested:
        return

    # Avaliacao final
    print(Fore.GREEN + "\n[OK] TREINAMENTO CONCLUÍDO!\n")
    print(Fore.BLUE + f"[INFO] SALVANDO MODELO {model_name}\n")
    print(Fore.GREEN + "[OK] FINALIZANDO O PROGRAMA DE TREINAMENTO...")

    # Recarrega melhor modelo para avaliação final
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    # Avaliações finais
    # [UPDATED] Unpack
    va_loss, va_acc, va_f1m, va_f1w, va_report, _, _, va_cm, va_report_id = evaluate(model, val_loader, DEVICE, criterion)
    
    # Se a ultima época não for a melhor (quase certo), usamos o report da melhor época ou da atual?
    # Geralmente reporta-se do melhor modelo carregado.
    final_val_report_id = va_report_id

    # Nomenclatura final do diretório
    def get_model_abbr(m):
        m = m.lower()
        if "temporal fusion" in m or m == "tft": return "tft"
        if "robust" in m: return "RobustLSTM"
        if "pooled" in m: return "PooledLSTM"
        if "bilstm" in m: return "BiLSTM"
        if "attention" in m: return "AttentionLSTM"
        if "tcn" in m: return "tcn"
        if "transformer" in m: return "trans"
        if "wavenet" in m: return "wave"
        if "lstm" in m: return "lstm"
        return m

    abbr = get_model_abbr(args.model)
    accuracy_percent = va_acc * 100
    final_dir_name = f"{args.name}-modelo_{abbr}-acc_{accuracy_percent:.1f}"
    
    # Define o novo diretório final e move o conteúdo
    final_model_dir = model_dir.parent / final_dir_name
    final_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Salva graficos e modelo final no novo dir
    plot_curves(history, final_model_dir / "curvas_treino_validacao.png")
    plot_confusion_matrix(va_cm, [primeiraClasse, segundaClasse], final_model_dir / "matriz_confusao_validacao.png")
    torch.save(model.state_dict(), final_model_dir / cm.MODEL_FINAL_FILENAME)
    
    # Move o model_best.pt e o norm_stats.pt
    if best_model_path.exists():
        shutil.move(str(best_model_path), str(final_model_dir / cm.MODEL_BEST_FILENAME))
    if (model_dir / cm.NORM_STATS_FILENAME).exists():
        shutil.move(str(model_dir / cm.NORM_STATS_FILENAME), str(final_model_dir / cm.NORM_STATS_FILENAME))

    # Relatorio
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = final_model_dir / f"relatorio_{timestamp}.txt"
    hist_json_path = final_model_dir / f"historico_{timestamp}.json"
    detalhes_json_path = final_model_dir / f"detalhes_validacao.json"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("RELATORIO DE TREINAMENTO\n")
        f.write(f"Data: {timestamp}\n")
        f.write(f"Dispositivo: {get_gpu_info()}\n")
        f.write(f"Modelo: {args.model} ({abbr})\n")
        f.write(f"Epocas: {args.epochs}\n")
        f.write(f"Melhor epoca: {best_epoch}\n")
        f.write(f"Val Acc (ID-level): {accuracy_percent:.1f}%\n")
        f.write(f"Val F1 (ID-level): {best_val_f1:.4f}\n")
        # Podemos adicionar contagem de IDs corretos
        if final_val_report_id:
            ok_count = sum(1 for x in final_val_report_id if x['ok'])
            total_ids = len(final_val_report_id)
            f.write(f"\nIDs Corretos: {ok_count}/{total_ids} ({(ok_count/total_ids)*100:.1f}%)\n")

    with open(hist_json_path, "w", encoding="utf-8") as jf:
        json.dump(history, jf, indent=2)
        
    # Salva os detalhes granulares (User requested to remove this for training)
    # if final_val_report_id:
    #     with open(detalhes_json_path, "w", encoding="utf-8") as df:
    #         json.dump(final_val_report_id, df, indent=2, ensure_ascii=False)

    # Limpa diretório temporário
    try:
        if model_dir != final_model_dir:
            for item in model_dir.iterdir():
                if item.is_file(): item.unlink()
            model_dir.rmdir()
    except: pass

    # Caminho final relativo para exibição
    # try:
    #     rel_path = final_model_dir.relative_to(cm.ROOT_DIR)
    # except:
    #     rel_path = final_model_dir
    
    # print(Fore.WHITE + f"\n[INFO] Arquivos salvos em: {rel_path}")