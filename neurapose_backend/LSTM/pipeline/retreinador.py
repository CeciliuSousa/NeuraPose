# ================================================================
# neurapose-backend/app/pipeline/retreinador.py
# ================================================================

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from datetime import datetime
from colorama import Fore, init

# Importa configuracoes do config_master (via LSTM/configuracao/config.py)
from LSTM.configuracao.config import get_config
from LSTM.modulos.dataset import load_data_pkl
from LSTM.modulos.treinamento import train_one_epoch


# Importa paths e constantes do config_master
from config_master import (
    RETRAIN_MODELS_DIR,
    CLASS_NAMES,
    DEVICE,
)


init(autoreset=True)


def get_gpu_info():
    """Retorna informacoes sobre a GPU disponivel."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return f"{gpu_name} ({gpu_memory:.1f} GB)"
    return "CPU (sem GPU disponivel)"


def calcular_pesos(labels_json_path):
    if not os.path.exists(labels_json_path):
        print(Fore.YELLOW + f"[AVISO] Arquivo de labels nao encontrado: {labels_json_path}")
        return 0, 0, [1.0, 1.0]

    with open(labels_json_path, encoding='utf-8') as f:
        data = json.load(f)
    
    # Usa nomes das classes do config_master
    primeira_classe = CLASS_NAMES[0].lower()
    segunda_classe = CLASS_NAMES[1].lower()
    
    normal_count = sum(1 for v in data.values() for label in v.values() if label == primeira_classe)
    furto_count = sum(1 for v in data.values() for label in v.values() if label == segunda_classe)
    
    if normal_count == 0 or furto_count == 0:
        return normal_count, furto_count, [1.0, 1.0]

    total = normal_count + furto_count
    weight_furto = total / (2 * furto_count)
    weight_normal = total / (2 * normal_count)
    return normal_count, furto_count, [weight_furto, weight_normal]


def main():
    args, ModelClass = get_config()

    device = torch.device(DEVICE)

    # micro-otimizacao: ativa benchmark para cuDNN quando em GPU (melhora throughput)
    if device.type == 'cuda':
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
    
    # Banner de inicio com info da GPU
    print("\n" + "="*70)
    print("NEURAPOSE - RETREINAMENTO DE MODELO")
    print("="*70)
    print(f"[DISPOSITIVO] {get_gpu_info()}")
    print(f"[MODELO] {args.model.upper()}")
    print("="*70 + "\n")

    # Carregar dataset - usa args.dataset que vem do config_master como default
    try:
        dataset, y = load_data_pkl(args.dataset)
    except FileNotFoundError:
        print(Fore.RED + f"[ERRO] Dataset nao encontrado: {args.dataset}")
        return

    # DataLoader otimizado
    num_workers = min(4, os.cpu_count() or 1)
    pin_memory = True if device.type == 'cuda' else False
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    # Pesos - usa args.annotations que vem do config_master como default
    normal_count, furto_count, pesos = calcular_pesos(args.annotations)
    weights = torch.tensor(pesos, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Inicializar modelo
    try:
        model = ModelClass(input_size=args.input_size,
                           hidden_size=args.hidden_size,
                           num_layers=args.num_layers,
                           dropout=args.dropout).to(device)
    except TypeError:
         # Fallback para modelos com assinaturas diferentes
         model = ModelClass(input_size=args.input_size).to(device)

    if args.pretrained and os.path.exists(args.pretrained):
        print(Fore.GREEN + f"[INFO] Carregando modelo salvo de: {args.pretrained}")
        try:
            model.load_state_dict(torch.load(args.pretrained, map_location=device))
        except Exception as e:
             print(Fore.RED + f"[ERRO] Falha ao carregar pesos: {e}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_time = time.time()
    history = []

    print(Fore.CYAN + f"Iniciando {args.epochs} epocas...")

    for epoch in range(1, args.epochs + 1):
        with tqdm(total=1, desc=f"Epoca {epoch}", ncols=130) as pbar:
            loss, acc, f1m, f1w = train_one_epoch(model, dataloader, optimizer, criterion, device)
            history.append((epoch, loss, acc))
            pbar.set_postfix({"loss": f"{loss:.4f}", "acc": f"{acc:.4f}"})
            pbar.update(1)
        print(f"[Epoca {epoch:3d}] Loss: {loss:.6f} | Acuracia: {acc:.4f}")

    elapsed = time.time() - start_time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Diretorio e salvamento do modelo - Usa RETRAIN_MODELS_DIR do config_master
    save_dir = RETRAIN_MODELS_DIR / (args.name or "modelo")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "model.pt"
    torch.save(model.state_dict(), save_path)

    # Geracao do relatorio
    report_path = save_dir / f"relatorio_{timestamp.replace(':', '').replace(' ', '_')}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("===== RELATORIO DE TREINAMENTO =====\n")
        f.write(f"Data/Hora         : {timestamp}\n")
        f.write(f"Dispositivo       : {get_gpu_info()}\n")
        f.write(f"Modelo Usado      : {args.model}\n")
        f.write(f"Total de Epocas   : {args.epochs}\n")
        f.write(f"Loss Final        : {history[-1][1]:.6f}\n")
        f.write(f"Acuracia Final    : {history[-1][2]:.4f}\n")
        f.write(f"Tempo Total (s)   : {elapsed:.2f}\n")
        f.write("\n--- Distribuicao de Classes ---\n")
        f.write(f"{CLASS_NAMES[0]}: {normal_count}\n")
        f.write(f"{CLASS_NAMES[1]}: {furto_count}\n")
        f.write(f"Pesos usados [{CLASS_NAMES[1].lower()}, {CLASS_NAMES[0].lower()}]: [{pesos[0]:.4f}, {pesos[1]:.4f}]\n")

    print(Fore.GREEN + "\n[SUCESSO] Treinamento concluido.")
    print(f"Modelo usado          : {args.model.upper()}")
    print(f"Modelo salvo em       : {save_path}")
    print(f"Relatorio salvo em    : {report_path}")

if __name__ == "__main__":
    main()
