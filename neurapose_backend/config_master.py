# neurapose_backend/config_master.py
# Configuração Central do Projeto - NUNCA edite configs de módulos individuais.

import numpy as np
from pathlib import Path
import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -- OTIMIZAÇÕES --
USE_NVENC = True                    # Usar encoder GPU NVIDIA (h264_nvenc)
NVENC_PRESET = "p4"                 # Preset NVENC (p1=fast, p7=quality)
USE_TENSORRT = False                # Habilitar aceleração TensorRT (.engine)

USE_FP16 = True                     # Half Precision (2x speed em RTX)
USE_ASYNC_LOADER = True             # Pre-fetch frames (Threaded)
ASYNC_BUFFER_SIZE = 64              # Tamanho do buffer de leitura
USE_PREFETCH = False                # Legacy flag (substituido por USE_ASYNC_LOADER)
PREFETCH_BUFFER_SIZE = 32           # Legacy param
YOLO_SKIP_FRAME_INTERVAL = 1        # Intervalo de frames para inferencia YOLO (1=sem skip)
YOLO_BATCH_SIZE = 256
RTMPOSE_BATCH_SIZE = 256

try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3) # 3=ERROR
except ImportError: pass

# -- MODELOS --
# YOLO (Detecção)
YOLO_MODEL = "yolo11s.pt"
YOLO_IMGSZ = 640
YOLO_CONF_THRESHOLD = 0.55
YOLO_CLASS_PERSON = 0

# RTMPose (Keypoints)
RTMPOSE_MODEL = "rtmpose-m_simcc-body7_pt-body7_420e-256x192/end2end.onnx"
RTMPOSE_INPUT_SIZE = (192, 256)
RTMPOSE_MAX_BATCH_SIZE = 10         # Batch size para inferência Pose (GPU)

# OSNet (ReID)
OSNET_MODEL = "osnet_x0_5_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"

TRACKER_NAME = "BoTSORT" # "BoTSORT" ou "DeepOCSORT"
TEMPORAL_MODEL = "tft"
# -- PARÂMETROS PRE-PROC --
DETECTION_CONF = 0.55
POSE_CONF_MIN = 0.30
CLAMP_MARGIN = 0.0                  # NÃO ALTERE
EMA_ALPHA = 1.0                     # SEM SUAVIZAÇÃO (1.0 = Raw Data, Zero Lag)
EMA_MIN_CONF = 0.0

# Filtros Pós-Processamento
MIN_POSDETECTION_CONF = 0.6
YOLO_CLASS_PERSON = 0
MIN_POSE_ACTIVITY = 0.8

# -- CONFIG TRACKING (BoTSORT / ReID) --
CLASSE1 = "NORMAL"
CLASSE2 = "FURTO"
CLASS_NAMES = [CLASSE1, CLASSE2]
NUM_CLASSES = len(CLASS_NAMES)

MIN_MEMBER_ACTIVITY = 5.0 # Pixels de variancia para considerar vivo mesmo se parado geograficamente

CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

POSITIVE_CLASS_ID = 1
POSITIVE_CLASS_NAME = CLASS_NAMES[POSITIVE_CLASS_ID]
MIN_FRAMES_PER_ID = 10

# -- DATASETS --
PROCESSING_DATASET = "data-labex-completo"
TEST_DATASET = "data-labex"
TRAINED_MODEL_NAME = "data-labex-teste"
DATASET_NAME = PROCESSING_DATASET

TRAIN_SPLIT = "treino"
VAL_SPLIT = "validacao"
TEST_SPLIT = "teste"

# -- HIPERPARÂMETROS TREINO --
TIME_STEPS = 30
NUM_JOINTS = 17
NUM_CHANNELS = 2
BATCH_SIZE = 64
LEARNING_RATE = 0.0003
EPOCHS = 5000

# LSTM/Transformer
LSTM_DROPOUT = 0.3
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_NUM_HEADS = 8
LSTM_KERNEL_SIZE = 5

# -- TESTE --
CLASSE2_THRESHOLD = 0.50

# -- PATHS --

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in os.environ["PATH"]:
    os.environ["PATH"] = str(ROOT) + os.pathsep + os.environ["PATH"]

# Vídeo & Codecs
OPENH264_DLL = ROOT / "openh264-1.8.0-win64.dll"

# Modelos
MODELS_DIR = ROOT / "detector" / "modelos"
YOLO_PATH = MODELS_DIR / YOLO_MODEL
RTMPOSE_DIR = ROOT / "rtmpose" / "modelos"
RTMPOSE_PATH = RTMPOSE_DIR / RTMPOSE_MODEL
OSNET_DIR = ROOT / "tracker" / "weights"
OSNET_PATH = OSNET_DIR / OSNET_MODEL

# Pré-processamento
PROCESSING_INPUT_DIR = ROOT / "videos"
PROCESSING_OUTPUT_DIR = ROOT / "resultados-processamentos"
PROCESSING_VIDEOS_DIR = PROCESSING_OUTPUT_DIR / "videos"
PROCESSING_PREDS_DIR = PROCESSING_OUTPUT_DIR / "predicoes"
PROCESSING_JSONS_DIR = PROCESSING_OUTPUT_DIR / "jsons"
PROCESSING_ANNOTATIONS_DIR = PROCESSING_OUTPUT_DIR / "anotacoes"
ANNOTATIONS_OUTPUT_DIR = ROOT / "resultados-reidentificacoes" # User: Anotações input/output is same as Reid

# ReID Manual
REID_OUTPUT_DIR = ROOT / "resultados-reidentificacoes"
REID_MANUAL_SUFFIX = "-reid"
REID_MANUAL_LABELS_FILENAME = "labels_reid.json"
RTMPOSE_PREPROCESSING_PATH = RTMPOSE_PATH

# Parâmetros sequência
MAX_FRAMES_PER_SEQUENCE = 9000  # Para streaming de vídeo (inferência)
TRAIN_SEQUENCE_LENGTH = 10      # Para treinamento/conversão de dataset (modelo espera T=30)
FPS_TARGET = 10.0
FRAME_DISPLAY_W = 1280
FRAME_DISPLAY_H = 720

# Treinamento
TRAINING_INPUT_DIR = PROCESSING_OUTPUT_DIR
TRAINED_MODELS_DIR = ROOT / "modelos-temporais" # User: Modelos temporais
TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_DIR = TRAINED_MODELS_DIR / f"{TEMPORAL_MODEL}-{TRAINED_MODEL_NAME}"

# Teste
DATASETS_ROOT = ROOT / "datasets" # User: Root of all datasets
TEST_DATASETS_ROOT = DATASETS_ROOT / TEST_DATASET
TEST_DIR = TEST_DATASETS_ROOT / TEST_SPLIT / "videos"
TEST_LABELS_JSON = TEST_DATASETS_ROOT / TEST_SPLIT / "anotacoes" / "labels.json"
TEST_MODEL_DIR = MODEL_SAVE_DIR

# Relatórios
TEST_REPORTS_DIR = ROOT / "relatorios-testes"
TEST_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Dados de treino
TRAINING_DATA_PATH = DATASETS_ROOT / PROCESSING_DATASET / "treino" / "data" / "data.pt"
TRAINING_LABELS_PATH = DATASETS_ROOT / PROCESSING_DATASET / "treino" / "labels.json"
RETRAIN_MODELS_DIR = ROOT / "retreinamentos"

# Saída
MODEL_BEST_FILENAME = "model_best.pt"
MODEL_FINAL_FILENAME = "model_final.pt"
NORM_STATS_FILENAME = "norm_stats.pt"

# BoTSORT
BOTSORT_YAML_PATH = ROOT / "tracker" / "configuracao" / "botsort_custom.yaml"

BOT_SORT_CONFIG = {
    "tracker_type": "botsort",
    "track_high_thresh": 0.6,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.60,
    "track_buffer": 300,
    "match_thresh": 0.90,
    "appearance_thresh": 0.25,
    "proximity_thresh": 0.75,
    "gmc_method": "none",
    "fuse_score": True,
    "with_reid": True,
    "model": str(OSNET_PATH),
    "device": DEVICE,
}

DEEPOCSORT_YAML_PATH = ROOT / "tracker" / "configuracao" / "deepocsort_custom.yaml"

DEEP_OC_SORT_CONFIG = {
    "tracker_type": "deepocsort",
    "det_thresh": 0.55,
    "max_age": 120,
    "min_hits": 3,
    "iou_thresh": 0.3,
    "delta_t": 3,
    "asso_func": "giou",
    "inertia": 0.2,
    "w_association_emb": 0.75,
    "reid_weights": str(OSNET_PATH), 
    "half": USE_FP16, 
    "device": DEVICE,
}

TRAIN_DIR = TEST_DATASETS_ROOT / TRAIN_SPLIT / "videos"
VAL_DIR = TEST_DATASETS_ROOT / VAL_SPLIT / "videos"
PREPROCESSING_OUTPUT = ROOT / "resultado_processamento"

# -- CONSTANTES FIXAS --

SIMCC_W = RTMPOSE_INPUT_SIZE[0]     # 192
SIMCC_H = RTMPOSE_INPUT_SIZE[1]     # 256
SIMCC_SPLIT_RATIO = 2.0

# ImageNet
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# Pares de esqueleto COCO-17
PAIRS = [
    (5, 7), (7, 9), (6, 8), (8, 10),     # Braços
    (5, 6),                              # Ombros
    (11, 12),                            # Quadris
    (11, 13), (13, 15),                  # Perna esquerda
    (12, 14), (14, 16),                  # Perna direita
    (5, 11), (6, 12)                     # Tronco
]

# ==============================================================
# VALIDAÇÃO
# ==============================================================

def validar_configuracoes():
    """Valida configurações antes de executar."""
    erros = []
    
    if CLAMP_MARGIN != 0.0:
        erros.append(f"[AVISO] CLAMP_MARGIN deve ser 0.0 (atual: {CLAMP_MARGIN})")
    
    if not (0.0 <= DETECTION_CONF <= 1.0):
        erros.append(f"[ERRO] DETECTION_CONF inválido: {DETECTION_CONF}")
    
    if not (0.0 <= POSE_CONF_MIN <= 1.0):
        erros.append(f"[ERRO] POSE_CONF_MIN inválido: {POSE_CONF_MIN}")
    
    if not (0.0 <= CLASSE2_THRESHOLD <= 1.0):
        erros.append(f"[ERRO] CLASSE2_THRESHOLD inválido: {CLASSE2_THRESHOLD}")
    
    if TEMPORAL_MODEL not in ["tft", "lstm"]:
        erros.append(f"[ERRO] TEMPORAL_MODEL deve ser 'tft' ou 'lstm'")
    
    if not YOLO_PATH.exists():
        erros.append(f"[AVISO] Modelo YOLO não encontrado: {YOLO_PATH.name}")
    
    # Valida DLL OpenH264
    dll_path = ROOT / OPENH264_DLL
    if not dll_path.exists():
        erros.append(f"[AVISO] DLL OpenH264 não encontrada: {OPENH264_DLL}. O codec 'avc1' falhará (usará fallback 'mp4v').")
    
    return erros

# Executa validação ao importar
# _erros = validar_configuracoes()
# if _erros:
#     print("\n[AVISOS DE CONFIGURAÇÃO]")
#     for erro in _erros:
#         print(f"  {erro}")
#     print()

# ==============================================================
# DEBUG
# ==============================================================

def imprimir_configuracoes():
    """Imprime resumo das configurações."""
    print("\n" + "="*60)
    print("CONFIGURAÇÕES ATIVAS")
    print("="*60)
    print(f"[MODELOS] YOLO: {YOLO_MODEL} | Temporal: {TEMPORAL_MODEL.upper()}")
    print(f"[CLASSES] {', '.join(CLASS_NAMES)} | Positiva: {POSITIVE_CLASS_NAME}")
    print(f"[DATASETS] Proc: {PROCESSING_DATASET} | Teste: {TEST_DATASET}")
    print(f"[TREINO] Steps: {TIME_STEPS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print("="*60 + "\n")

def imprimir_configs_yolo_botsort():
    """Imprime configurações do YOLO e BoTSORT."""
    print("\n" + "="*50)
    print(" DETALHES DO RASTREADOR")
    print("="*50)
    print(f"  [YOLO] Detection Conf:  {DETECTION_CONF}")
    print("-" * 50)
    print(f"  [BoTSORT] Track High:   {BOT_SORT_CONFIG['track_high_thresh']}")
    print(f"  [BoTSORT] Track Low:    {BOT_SORT_CONFIG['track_low_thresh']}")
    print(f"  [BoTSORT] Match (ReID): {BOT_SORT_CONFIG['match_thresh']}")
    print(f"  [BoTSORT] Buffer:       {BOT_SORT_CONFIG['track_buffer']}")
    print(f"  [BoTSORT] ReID Ativo:   {BOT_SORT_CONFIG['with_reid']}")
    print("="*50 + "\n")

if __name__ == "__main__":
    imprimir_configuracoes()
