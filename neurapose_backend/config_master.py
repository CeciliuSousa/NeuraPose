# ==============================================================
# config_master.py - CONFIGURACAO MESTRA DO PROJETO
# ==============================================================
# CENTRO DE CONTROLE UNICO
#
# Este arquivo centraliza TODAS as configuracoes do projeto:
# - Pre-processamento de videos
# - Treinamento de modelos
# - Teste e validacao
#
# COMO USAR:
# 1. Configure tudo aqui (parametros, modelos, paths, etc.)
# 2. Execute pre-processamento, treino ou teste
# 3. Os codigos importam automaticamente daqui
#
# IMPORTANTE: NUNCA edite config.py dos modulos individuais!
# IMPORTANTE: SEMPRE edite apenas este arquivo!
#
# ==============================================================

import numpy as np
from pathlib import Path
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================
# SECAO 1: MODELOS E FERRAMENTAS
# ==============================================================

# ------------------------------------------------------------------
# 1.1 MODELO YOLO (Deteccao de Pessoas)
# ------------------------------------------------------------------
# Opcoes: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
# 
# yolov8n: Mais rapido, menor acuracia (~3M params)
# yolov8s: Rapido (~11M params)
# yolov8m: Balanceado (~26M params)
# yolov8l: Recomendado! (~44M params)
# yolov8x: Mais preciso, mais lento (~68M params)

YOLO_MODEL = "yolov8l.pt"           # [ALTERE AQUI] modelo YOLO
YOLO_IMGSZ = 1280                   # Resolucao de entrada (640, 1280, 1920)
YOLO_CONF_THRESHOLD = 0.35          # Confianca minima deteccao (0.0-1.0)
YOLO_CLASS_PERSON = 0              # Classe 'person' no COCO

# ------------------------------------------------------------------
# 1.2 MODELO RTMPose (Extracao de Keypoints)
# ------------------------------------------------------------------
# NAO ALTERE (modelo fixo usado no projeto)
RTMPOSE_MODEL = "rtmpose-l_simcc-body7_pt-body7_420e-256x192/end2end.onnx"
RTMPOSE_INPUT_SIZE = (192, 256)     # (Width, Height)

# ------------------------------------------------------------------
# 1.3 MODELO OSNet (Re-Identificacao)
# ------------------------------------------------------------------
# NAO ALTERE (modelo fixo usado no projeto)
OSNET_MODEL = "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"

# ------------------------------------------------------------------
# 1.4 TRACKER (Rastreamento de IDs)
# ------------------------------------------------------------------
TRACKER_NAME = "BoTSORT"            # Tracker usado (BoTSORT ou ByteTrack)

# ------------------------------------------------------------------
# 1.5 MODELO TEMPORAL
# ------------------------------------------------------------------
# Opcoes: "tft" ou "lstm"
# 
# TFT (Temporal Fusion Transformer): Arquitetura moderna, melhor para longas sequencias
# LSTM (Long Short-Term Memory): Classico, mais rapido

TEMPORAL_MODEL = "tft"              # [ALTERE AQUI] "tft" ou "lstm"

# ==============================================================
# SECAO 2: PARAMETROS DE PRE-PROCESSAMENTO
# ==============================================================

#################################################################
# 2.1 DETECCAO (YOLO)
#################################################################
# Confianca minima para aceitar uma deteccao de pessoa
# Menor = mais pessoas (+ falsos positivos)
# Maior = menos pessoas (+ falsos negativos)
# Recomendado: 0.30-0.40

DETECTION_CONF = 0.55               # [ALTERE] Confianca YOLO (padrao: 0.35)

#################################################################
# 2.2 EXTRACAO DE POSE (RTMPose)
#################################################################
# Confianca minima para aceitar um keypoint
# Menor = mais keypoints (+ ruido)
# Maior = menos keypoints (+ dados faltantes)
# Recomendado: 0.25-0.35

POSE_CONF_MIN = 0.30                # [ALTERE] Confianca keypoint (padrao: 0.30)

#################################################################
# 2.3 CLAMPING (Limitacao de Keypoints)
#################################################################
# Margem para expandir bbox antes de limitar keypoints
# 0.0 = sem expansao (keypoints podem sair da bbox)
# 0.3 = expande 30% (keypoints limitados a bbox expandida)
#
# CRITICO: DEVE ser 0.0 para matching treino-teste!

CLAMP_MARGIN = 0.0                  # NAO ALTERE! (deve ser 0.0)

#################################################################
# 2.4 SUAVIZACAO TEMPORAL (EMA)
#################################################################
# Fator de suavizacao dos keypoints ao longo do tempo
# 0.0 = sem suavizacao (mais ruido, mais reativo)
# 1.0 = maxima suavizacao (menos ruido, mais lag)
# Recomendado: 0.30-0.40

EMA_ALPHA = 0.35                    # [ALTERE] Fator EMA (padrao: 0.35)

# Confianca minima para aplicar suavizacao
# 0.0 = suaviza todos os keypoints
# 0.1 = so suaviza keypoints com conf > 0.1
# Recomendado: 0.0

EMA_MIN_CONF = 0.0                  # [ALTERE] Conf minima EMA (padrao: 0.0)

# CONFIGURACAO ATUAL DE CLASSES
CLASSE1 = "NORMAL"
CLASSE2 = "FURTO"
CLASS_NAMES = [CLASSE1, CLASSE2]   # [ALTERE] aqui para seu dataset
NUM_CLASSES = len(CLASS_NAMES)      # Numero de classes (automatico)

# Mapeamento classe -> ID (automatico)
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# ID da classe positiva (a classe que queremos detectar)
POSITIVE_CLASS_ID = 1               # [ALTERE] ID da classe positiva (padrao: 1)
POSITIVE_CLASS_NAME = CLASS_NAMES[POSITIVE_CLASS_ID]

# ==============================================================
# SECAO 3: PARAMETROS DE TREINAMENTO
# ==============================================================

#################################################################
# 3.1 DATASET
#################################################################
# CONFIGURACAO DE DATASETS POR PIPELINE
#n
# Este projeto usa DIFERENTES datasets em cada etapa:
#
# [1] PRE-PROCESSAMENTO (processar.py):
#    - Input: videos/data-labex-completo
#    - Output: processamentos/data-labex-completo/
#
# [2] TREINAMENTO (treinar_modelo.py):
#    - Input: processamentos/data-labex-completo/
#    - Output: meus-modelos-treinados/tft-data-labex/
#
# [3] TESTE (testar_modelo.py):
#    - Input: datasets/data-labex/teste/videos/
#    - Labels: datasets/data-labex/teste/anotacoes/labels.json
#    - Modelo: meus-modelos-treinados/tft-data-labex/
#

# Dataset usado para PROCESSAMENTO (input do processar.py)
PROCESSING_DATASET = "data-labex-completo"  # [ALTERE] Dataset para processar videos

# Dataset usado para TESTE (input do testar_modelo.py)
TEST_DATASET = "data-labex"                 # [ALTERE] Dataset de teste com labels

# Dataset usado no NOME do modelo treinado (output do treinar_modelo.py)
TRAINED_MODEL_NAME = "data-labex-teste"           # [ALTERE] Nome do modelo treinado

# Nome do dataset (usado em logs e banners)
DATASET_NAME = PROCESSING_DATASET           # [ALTERE] Nome do dataset ativo

# Divisao do dataset de TESTE
TRAIN_SPLIT = "treino"              # Pasta de treino
VAL_SPLIT = "validacao"             # Pasta de validacao
TEST_SPLIT = "teste"                # Pasta de teste

#################################################################
# 3.2 HIPERPARAMETROS DE TREINO
#################################################################
# Janela temporal (numero de frames por sequencia)
# Menor = menos contexto, mais rapido
# Maior = mais contexto, mais lento
# Recomendado: 30-60

TIME_STEPS = 30                     # [ALTERE] Janela temporal (padrao: 30)

# Numero de juntas (keypoints) por pessoa
NUM_JOINTS = 17                     # Padrao COCO (NAO ALTERE)

# Numero de canais (x, y coordenadas)
NUM_CHANNELS = 2                    # Padrao (NAO ALTERE)

# Batch size (numero de sequencias por batch)
# Maior = mais rapido, mais memoria GPU
# Menor = mais lento, menos memoria
# Recomendado: 16-64

BATCH_SIZE = 32                     # [ALTERE] Batch size (padrao: 32)

# Learning rate (taxa de aprendizado)
# Maior = convergencia rapida, instavel
# Menor = convergencia lenta, estavel
# Recomendado: 0.0001-0.001

LEARNING_RATE = 0.0003             # [ALTERE] Learning rate (padrao: 0.0003)

# Numero de epocas de treinamento
EPOCHS = 100                        # [ALTERE] Epocas (padrao: 100)

# ==============================================================
# SECAO 4: PARAMETROS DE TESTE/INFERENCIA
# ==============================================================

#################################################################
# 4.1 THRESHOLD DE CLASSIFICACAO
#################################################################
# Threshold para classificar como CLASSE2
# Recomendado: 0.45-0.55

CLASSE2_THRESHOLD = 0.50              # [ALTERE] Threshold para CLASSE2 (padrao: 0.50)

# ==============================================================
# SECAO 5: PATHS E ESTRUTURA DE DIRETORIOS
# ==============================================================
# Geralmente NAO precisa alterar esta secao!

# Root do projeto
ROOT = Path(__file__).resolve().parent

# Modelos
MODELS_DIR = ROOT / "detector" / "modelos"
YOLO_PATH = MODELS_DIR / YOLO_MODEL

RTMPOSE_DIR = ROOT / "app" /"modelos"
RTMPOSE_PATH = RTMPOSE_DIR / RTMPOSE_MODEL

OSNET_DIR = ROOT / "tracker" / "weights"
OSNET_PATH = OSNET_DIR / OSNET_MODEL

# ==============================================================
# PATHS DE PRE-PROCESSAMENTO
# ==============================================================
# Input: pasta com videos RAW para processar
# Exemplo: videos/
PROCESSING_INPUT_DIR = ROOT / "videos"

# Output: pasta onde serao salvos os dados processados
# Exemplo: resultados-processamentos/
PROCESSING_OUTPUT_DIR = ROOT / "resultados-processamentos"

# Subpastas de saida do processamento
PROCESSING_VIDEOS_DIR = PROCESSING_OUTPUT_DIR / "videos"
PROCESSING_PREDS_DIR = PROCESSING_OUTPUT_DIR / "predicoes"
PROCESSING_JSONS_DIR = PROCESSING_OUTPUT_DIR / "jsons"
PROCESSING_ANNOTATIONS_DIR = PROCESSING_OUTPUT_DIR / "anotacoes"

# -------------------------------------------------------------
# REID MANUAL (Ferramenta de correção manual de IDs persistentes)
# -------------------------------------------------------------
# Pasta padrao de saida para ReID
REID_OUTPUT_DIR = ROOT / "resultados-reidentificacao"

# Sufixo a ser aplicado no nome do diretório de saída (ex: "teste" -> "teste-reid")
REID_MANUAL_SUFFIX = "-reid"
# Nome do arquivo de labels gerado pelo reid-manual (mapping por vídeo)
REID_MANUAL_LABELS_FILENAME = "labels_reid.json"

# Modelo RTMPose para pre-processamento (copia local em pre_processamento/modelos)
RTMPOSE_PREPROCESSING_DIR = ROOT / "pre_processamento" / "modelos"
RTMPOSE_PREPROCESSING_PATH = RTMPOSE_PREPROCESSING_DIR / RTMPOSE_MODEL

# ==============================================================
# PARAMETROS DE SEQUENCIA (Pre-proc -> Treino -> Teste)
# ==============================================================
# CRITICO: Devem ser IGUAIS em processamento, treino e teste!
# 
# Estes valores definem a janela temporal das sequencias:
# - MAX_FRAMES_PER_SEQUENCE: Numero maximo de frames por sequencia
# - MIN_FRAMES_PER_ID: Minimo de frames para considerar um ID valido
# - FPS_TARGET: FPS para classificação de videos

MAX_FRAMES_PER_SEQUENCE = 30           # [ALTERE] Frames por sequencia (= TIME_STEPS)
MIN_FRAMES_PER_ID = 30                 # [ALTERE] Minimo de frames para ID valido
FPS_TARGET = 30.0                      # [ALTERE] FPS alvo para CLASSE2izacao

# Display para preview de video
FRAME_DISPLAY_W = 1280
FRAME_DISPLAY_H = 720

# ==============================================================
# PATHS DE TREINAMENTO
# ==============================================================
# Input: dados processados (output do pre-processamento)
# Exemplo: processamentos/data-labex-completo/
TRAINING_INPUT_DIR = PROCESSING_OUTPUT_DIR

# Output: pasta onde sera salvo o modelo treinado
# Exemplo: meus-modelos-treinados/tft-data-labex/
TRAINED_MODELS_DIR = ROOT / "meus-modelos-treinados"
MODEL_SAVE_DIR = TRAINED_MODELS_DIR / f"{TEMPORAL_MODEL}-{TRAINED_MODEL_NAME}"

# ==============================================================
# PATHS DE TESTE
# ==============================================================
# Dataset de teste (com labels para validacao)
TEST_DATASETS_ROOT = ROOT / "datasets" / TEST_DATASET
TEST_DIR = TEST_DATASETS_ROOT / TEST_SPLIT / "videos"
TEST_LABELS_JSON = TEST_DATASETS_ROOT / TEST_SPLIT / "anotacoes" / "labels.json"

# Modelo treinado a ser testado
TEST_MODEL_DIR = MODEL_SAVE_DIR

# ==============================================================
# PATHS DE TREINAMENTO (LSTM/Transformer)
# ==============================================================
# Arquivo de dados de treino (.pt ou .pkl)
TRAINING_DATA_PATH = ROOT / "datasets" / PROCESSING_DATASET / "treino" / "data" / "data.pt"
TRAINING_LABELS_PATH = ROOT / "datasets" / PROCESSING_DATASET / "treino" / "labels.json"

# Diretorio para retreinamentos
RETRAIN_MODELS_DIR = ROOT / "retreinamentos"

# Nomes de arquivos de saida do treinamento (para consistencia)
MODEL_BEST_FILENAME = "model_best.pt"
MODEL_FINAL_FILENAME = "model_final.pt"
NORM_STATS_FILENAME = "norm_stats.pt"

# ==============================================================
# PATHS DO TRACKER (BoTSORT)
# ==============================================================
# Arquivo YAML temporario para configuracao do BoTSORT
BOTSORT_YAML_PATH = ROOT / "tracker" / "configuracao" / "botsort_custom.yaml"

# Configuracao completa do BoTSORT (usada em tracker/configuracao/config.py)
# BOT_SORT_CONFIG = {
#     "tracker_type": "botsort",
#     # Thresholds
#     "track_high_thresh": 0.6,
#     "track_low_thresh": 0.1,
#     "new_track_thresh": 0.3,
#     # Buffer e matching
#     "track_buffer": 50,
#     "match_thresh": 0.75,
#     # Aparencia
#     "appearance_thresh": 0.25,
#     "proximity_thresh": 0.6,
#     # Motion compensation
#     "gmc_method": "orb",
#     "fuse_score": True,
#     # ReID
#     "with_reid": True,
#     "model": str(OSNET_PATH),  # Usa o OSNET_PATH centralizado
# }


BOT_SORT_CONFIG = {
    "tracker_type": "botsort",

    # Thresholds
    "track_high_thresh": 0.5,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.5,

    # Buffer e matching
    "track_buffer": 300,
    "match_thresh": 0.30,

    # Aparencia
    "appearance_thresh": 0.20,
    "proximity_thresh": 0.6,

    # Motion compensation
    "gmc_method": "orb",

    "fuse_score": True,

    # ReID
    "with_reid": True,
    "model": str(OSNET_PATH),
}


# ==============================================================
# PATHS LEGADOS (mantidos para compatibilidade)
# ==============================================================
DATASETS_ROOT = TEST_DATASETS_ROOT  # Aponta para dataset de teste
TRAIN_DIR = TEST_DATASETS_ROOT / TRAIN_SPLIT / "videos"
VAL_DIR = TEST_DATASETS_ROOT / VAL_SPLIT / "videos"
PREPROCESSING_OUTPUT = ROOT / "resultado_processamento"  # Raiz de processamentos

# ==============================================================
# SECAO 6: CONSTANTES FIXAS (NAO ALTERE!)
# ==============================================================

# Dimensoes do modelo SIMCC (RTMPose)
SIMCC_W = RTMPOSE_INPUT_SIZE[0]     # 192
SIMCC_H = RTMPOSE_INPUT_SIZE[1]     # 256
SIMCC_SPLIT_RATIO = 2.0

# ImageNet (para RTMPose)
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# Pares de esqueleto COCO-17 (para visualizacao)
PAIRS = [
    (5, 7), (7, 9), (6, 8), (8, 10),    # Bracos
    (5, 6),                              # Ombros
    (11, 12),                            # Quadris
    (11, 13), (13, 15),                  # Perna esquerda
    (12, 14), (14, 16),                  # Perna direita
    (5, 11), (6, 12)                     # Tronco
]

# ==============================================================
# SECAO 7: VALIDACAO DE CONFIGURACOES
# ==============================================================

def validar_configuracoes():
    """
    Valida se as configuracoes estao corretas antes de executar.
    """
    erros = []
    
    # Validar CLAMP_MARGIN
    if CLAMP_MARGIN != 0.0:
        erros.append(f"[AVISO] CLAMP_MARGIN deve ser 0.0 (atual: {CLAMP_MARGIN})")
    
    # Validar thresholds
    if not (0.0 <= DETECTION_CONF <= 1.0):
        erros.append(f"[ERRO] DETECTION_CONF deve estar entre 0.0-1.0 (atual: {DETECTION_CONF})")
    
    if not (0.0 <= POSE_CONF_MIN <= 1.0):
        erros.append(f"[ERRO] POSE_CONF_MIN deve estar entre 0.0-1.0 (atual: {POSE_CONF_MIN})")
    
    if not (0.0 <= CLASSE2_THRESHOLD <= 1.0):
        erros.append(f"[ERRO] CLASSE2_THRESHOLD deve estar entre 0.0-1.0 (atual: {CLASSE2_THRESHOLD})")
    
    # Validar modelo temporal
    if TEMPORAL_MODEL not in ["tft", "lstm"]:
        erros.append(f"[ERRO] TEMPORAL_MODEL deve ser 'tft' ou 'lstm' (atual: {TEMPORAL_MODEL})")
    
    # Validar paths (apenas avisos/erros, sem mensagens de sucesso)
    if not YOLO_PATH.exists():
        if not YOLO_PATH.parent.exists():
            erros.append(f"[AVISO] Diretorio de modelos YOLO nao existe: {MODELS_DIR}")
        else:
            erros.append(f"[AVISO] Modelo YOLO nao encontrado: {YOLO_PATH.name}")
    
    return erros

# Executar validacao ao importar (apenas se houver erros/avisos reais)
_erros = validar_configuracoes()
if _erros:
    print("\n[AVISOS DE CONFIGURACAO]")
    for erro in _erros:
        print(f"  {erro}")
    print()

# ==============================================================
# RESUMO DAS CONFIGURACOES (para debug)
# ==============================================================

def imprimir_configuracoes():
    """
    Imprime um resumo das configuracoes atuais.
    """
    print("\n" + "="*70)
    print("CONFIGURACOES ATIVAS")
    print("="*70)
    print(f"\n[MODELOS]")
    print(f"  - YOLO: {YOLO_MODEL}")
    print(f"  - Temporal: {TEMPORAL_MODEL.upper()}")
    print(f"  - Tracker: {TRACKER_NAME}")
    
    print(f"\n[CLASSES DO DATASET]")
    print(f"  - Classes: {', '.join(CLASS_NAMES)} ({NUM_CLASSES} classes)")
    print(f"  - Classe Positiva: {POSITIVE_CLASS_NAME} (ID: {POSITIVE_CLASS_ID})")
    
    print(f"\n[DATASETS POR PIPELINE]")
    print(f"  - Processamento: {PROCESSING_DATASET}")
    print(f"  - Teste: {TEST_DATASET}")
    print(f"  - Modelo Treinado: {TEMPORAL_MODEL}-{TRAINED_MODEL_NAME}")
    
    print(f"\n[PATHS PRINCIPAIS]")
    print(f"  - Dados de Treino: {TRAINING_DATA_PATH}")
    print(f"  - Modelos Treinados: {TRAINED_MODELS_DIR}")
    print(f"  - Retreinamentos: {RETRAIN_MODELS_DIR}")
    print(f"  - BoTSORT YAML: {BOTSORT_YAML_PATH}")
    
    print(f"\n[PRE-PROCESSAMENTO]")
    print(f"  - Detection Conf: {DETECTION_CONF}")
    print(f"  - Pose Conf Min: {POSE_CONF_MIN}")
    print(f"  - Clamp Margin: {CLAMP_MARGIN}")
    print(f"  - EMA Alpha: {EMA_ALPHA}")
    
    print(f"\n[TREINAMENTO]")
    print(f"  - Time Steps: {TIME_STEPS}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"  - Epochs: {EPOCHS}")
    
    print(f"\n[TESTE]")
    print(f"  - {POSITIVE_CLASS_NAME.capitalize()} Threshold: {CLASSE2_THRESHOLD}")
    
    print("="*70 + "\n")

def imprimir_configs_yolo_botsort():
    """
    Imprime as configurações detalhadas do YOLO e BoTSORT.
    Lê diretamente do dicionário BOT_SORT_CONFIG para garantir precisão.
    """
    print("\n" + "="*50)
    print(" DETALHES DO RASTREADOR (YOLO + BoTSORT)")
    print("="*50)
    print(f"  [YOLO] Detection Conf:  {DETECTION_CONF}")
    print("-" * 50)
    print(f"  [BoTSORT] Track High:   {BOT_SORT_CONFIG['track_high_thresh']}")
    print(f"  [BoTSORT] Track Low:    {BOT_SORT_CONFIG['track_low_thresh']}")
    print(f"  [BoTSORT] New Track:    {BOT_SORT_CONFIG['new_track_thresh']}")
    print(f"  [BoTSORT] Match (ReID): {BOT_SORT_CONFIG['match_thresh']}")
    print(f"  [BoTSORT] Buffer:       {BOT_SORT_CONFIG['track_buffer']}")
    print(f"  [BoTSORT] Appearance:   {BOT_SORT_CONFIG['appearance_thresh']}")
    print(f"  [BoTSORT] ReID Ativo:   {BOT_SORT_CONFIG['with_reid']}")
    print("="*50 + "\n")

# Para testar: python config_master.py
if __name__ == "__main__":
    imprimir_configuracoes()
