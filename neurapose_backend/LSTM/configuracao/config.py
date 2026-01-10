# ================================================================
# LSTM/configuracao/config.py
# ================================================================
# ATENCAO: Valores default vem do config_master.py!
# Use args para sobrescrever apenas quando necessario.
# ================================================================

import sys
import argparse
from pathlib import Path

# Importa valores do config_master para usar como defaults
from neurapose_backend.config_master import (
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_CLASSES,
    NUM_JOINTS,
    NUM_CHANNELS,
    TEMPORAL_MODEL,
    TRAINING_DATA_PATH,
    TRAINING_LABELS_PATH,
    MODEL_SAVE_DIR,
    TRAINED_MODEL_NAME,
)

# Importa classes de modelos
from ..models.models import (
    LSTM, RobustLSTM, PooledLSTM, BILSTM, AttentionLSTM,
    TCN, TransformerModel, TemporalFusionTransformer, WaveNet
) 

def get_config():
    """
    Retorna configuracao para treinamento.
    Os valores default vem do config_master.py, mas podem ser sobrescritos via args.
    """
    parser = argparse.ArgumentParser(description="Treinamento de Classificador Temporal (LSTM/Transformer)")

    # Parametros de Treino (defaults do config_master)
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Numero de epocas (default: {EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help=f"Tamanho do batch (default: {BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--model", type=str, default=TEMPORAL_MODEL, help=f"Tipo de modelo (default: {TEMPORAL_MODEL})")
    parser.add_argument("--name", type=str, default=TRAINED_MODEL_NAME, help=f"Nome para salvar o modelo (default: {TRAINED_MODEL_NAME})")

    # Hiperparametros do Modelo
    parser.add_argument('--dropout', type=float, default=0.3, help='Taxa de dropout')
    parser.add_argument('--hidden_size', type=int, default=128, help='Tamanho da camada oculta')
    parser.add_argument('--num_layers', type=int, default=2, help='Numero de camadas RNN')
    parser.add_argument('--num_heads', type=int, default=8, help='Numero de cabecas (Transformer/TFT)')
    parser.add_argument('--num_channels', type=int, default=NUM_CHANNELS, help='Canais (TCN/WaveNet)')
    parser.add_argument('--kernel_size', type=int, default=5, help='Tamanho do kernel (TCN/WaveNet)')
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES, help=f'Numero de classes (default: {NUM_CLASSES})')
    parser.add_argument('--input_size', type=int, default=NUM_JOINTS * NUM_CHANNELS, help='Numero de features por timestep')

    # Paths (defaults do config_master)
    parser.add_argument('--dataset', type=str, default=str(TRAINING_DATA_PATH), help='Caminho do dataset .pt/.pkl')
    parser.add_argument('--annotations', type=str, default=str(TRAINING_LABELS_PATH), help='Caminho para labels.json')
    parser.add_argument('--pretrained', type=str, required=False, help='Caminho do modelo pre-treinado (.pt)')
    parser.add_argument('--output_dir', type=str, default=str(MODEL_SAVE_DIR), help='Diretorio de saida do modelo')

    args, _ = parser.parse_known_args()

    # Mapeamento de string -> Classe
    model_map = {
        "lstm": LSTM,
        "robust": RobustLSTM,
        "robustlstm": RobustLSTM,
        "pooled": PooledLSTM,
        "bilstm": BILSTM,
        "attention": AttentionLSTM,
        "tcn": TCN,
        "transformer": TransformerModel,
        "tft": TemporalFusionTransformer,
        "wavenet": WaveNet
    }

    model_key = args.model.lower()
    if model_key not in model_map:
        print(f"[AVISO] Modelo '{args.model}' nao reconhecido. Usando RobustLSTM padrao.")
        ModelClass = RobustLSTM
    else:
        ModelClass = model_map[model_key]

    return args, ModelClass
