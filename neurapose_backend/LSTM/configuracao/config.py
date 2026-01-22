# ================================================================
# neurapose-backend/app/LSTM/configuracao/config.py
# ================================================================

import argparse

# Importa valores do config_master para usar como defaults
# Importa valores do config_master para usar como defaults
import neurapose_backend.config_master as cm


# Importa classes de modelos
from neurapose_backend.LSTM.models.models import (
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
    parser.add_argument("--epochs", type=int, default=cm.EPOCHS, help=f"Numero de epocas (default: {cm.EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=cm.BATCH_SIZE, help=f"Tamanho do batch (default: {cm.BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=cm.LEARNING_RATE, help=f"Learning rate (default: {cm.LEARNING_RATE})")
    parser.add_argument("--model", type=str, default=cm.TEMPORAL_MODEL, help=f"Tipo de modelo (default: {cm.TEMPORAL_MODEL})")
    parser.add_argument("--name", type=str, default=cm.TRAINED_MODEL_NAME, help=f"Nome para salvar o modelo (default: {cm.TRAINED_MODEL_NAME})")

    # Hiperparametros do Modelo
    parser.add_argument('--dropout', type=float, default=cm.LSTM_DROPOUT, help=f'Taxa de dropout (default: {cm.LSTM_DROPOUT})')
    parser.add_argument('--hidden_size', type=int, default=cm.LSTM_HIDDEN_SIZE, help=f'Tamanho da camada oculta (default: {cm.LSTM_HIDDEN_SIZE})')
    parser.add_argument('--num_layers', type=int, default=cm.LSTM_NUM_LAYERS, help=f'Numero de camadas RNN (default: {cm.LSTM_NUM_LAYERS})')
    parser.add_argument('--num_heads', type=int, default=cm.LSTM_NUM_HEADS, help=f'Numero de cabecas (Transformer/TFT) (default: {cm.LSTM_NUM_HEADS})')
    parser.add_argument('--num_channels', type=int, default=cm.NUM_CHANNELS, help='Canais (TCN/WaveNet)')
    parser.add_argument('--kernel_size', type=int, default=cm.LSTM_KERNEL_SIZE, help=f'Tamanho do kernel (TCN/WaveNet) (default: {cm.LSTM_KERNEL_SIZE})')
    parser.add_argument('--num_classes', type=int, default=cm.NUM_CLASSES, help=f'Numero de classes (default: {cm.NUM_CLASSES})')
    # O input size é fixo baseado nos keypoints e canais definidos no config master
    input_sz = cm.NUM_JOINTS * cm.NUM_CHANNELS
    parser.add_argument('--input_size', type=int, default=input_sz, help=f'Numero de features por timestep (default: {input_sz})')

    # Paths (defaults do config_master)
    parser.add_argument('--dataset', type=str, default=str(cm.TRAINING_DATA_PATH), help='Caminho do dataset .pt/.pkl')
    parser.add_argument('--annotations', type=str, default=str(cm.TRAINING_LABELS_PATH), help='Caminho para labels.json')
    parser.add_argument('--pretrained', type=str, required=False, help='Caminho do modelo pre-treinado (.pt)')
    parser.add_argument('--output_dir', type=str, default=str(cm.MODEL_SAVE_DIR), help='Diretorio de saida do modelo')

    args, _ = parser.parse_known_args()

    # Se o usuário não passou argumentos CLI, garantimos que args reflita o GLOBAL STATE do config_master
    # que já foi atualizado pelo UserConfigManager no main.py
    if args.epochs == cm.EPOCHS: args.epochs = cm.EPOCHS
    if args.batch_size == cm.BATCH_SIZE: args.batch_size = cm.BATCH_SIZE
    if args.lr == cm.LEARNING_RATE: args.lr = cm.LEARNING_RATE
    if args.model == cm.TEMPORAL_MODEL: args.model = cm.TEMPORAL_MODEL
    
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
        print(f"[AVISO] Modelo '{args.model}' nao reconhecido. Usando TemporalFusionTransformer padrao.")
        ModelClass = TemporalFusionTransformer
    else:
        ModelClass = model_map[model_key]

    return args, ModelClass
