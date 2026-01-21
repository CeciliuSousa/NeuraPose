# ==============================================================
# config_master.py - CONFIGURAÇÃO CENTRAL DO PROJETO
# ==============================================================
# 
# Centralize TODAS as configurações aqui.
# NUNCA edite config.py dos módulos individuais!
#
# ==============================================================

import numpy as np
from pathlib import Path
import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================
# SEÇÃO 1: MODELOS
# ==============================================================

# YOLO (Detecção)
YOLO_MODEL = "yolov8l.pt"           # yolov8n/s/m/l/x
YOLO_IMGSZ = 640                    # 640, 1280, 1920
YOLO_CONF_THRESHOLD = 0.35
YOLO_CLASS_PERSON = 0

# RTMPose (Keypoints) - NÃO ALTERE
RTMPOSE_MODEL = "rtmpose-l_simcc-body7_pt-body7_420e-256x192/end2end.onnx"
RTMPOSE_INPUT_SIZE = (192, 256)     # (Width, Height)

# OSNet (Re-Identificação) - NÃO ALTERE
OSNET_MODEL = "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"

# Tracker
TRACKER_NAME = "BoTSORT"

# Modelo Temporal: "tft" ou "lstm"
TEMPORAL_MODEL = "tft"

# ==============================================================
# SEÇÃO 2: PARÂMETROS DE PRÉ-PROCESSAMENTO
# ==============================================================

DETECTION_CONF = 0.55               # Confiança YOLO
POSE_CONF_MIN = 0.30                # Confiança keypoint
CLAMP_MARGIN = 0.0                  # NÃO ALTERE
EMA_ALPHA = 0.35                    # Suavização temporal
EMA_MIN_CONF = 0.0                  # Conf mínima para EMA

# Filtros de Pós-Processamento
MIN_POSDETECTION_CONF = 0.6  # Confiança mínima para detecção YOLO
YOLO_CLASS_PERSON = 0 # Classe 'pessoa' no COCO dataset
YOLO_BATCH_SIZE = 64  # Tamanho do batch para inferência YOLO (Otimização de Performance)
RTMPOSE_BATCH_SIZE = 64 # Tamanho do batch para inferência Pose (Novo)
MIN_POSE_ACTIVITY = 0.8             # StdDev médio mínimo (pixels) para considerar ID ativo

# ================================================================
# 3. CONFIGURAÇÕES DE TRACKING (BoTSORT / ReID / OSNet)
# ==============================================================

CLASSE1 = "NORMAL"                  # Classe padrão
CLASSE2 = "FURTO"                   # Classe anômala
CLASS_NAMES = [CLASSE1, CLASSE2]
NUM_CLASSES = len(CLASS_NAMES)

# Mapeamento classe <-> ID
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Classe positiva (a que queremos detectar)
POSITIVE_CLASS_ID = 1
POSITIVE_CLASS_NAME = CLASS_NAMES[POSITIVE_CLASS_ID]

# Mínimo de frames para ID aparecer na anotação
MIN_FRAMES_PER_ID = 30

# ==============================================================
# SEÇÃO 4: DATASETS
# ==============================================================

PROCESSING_DATASET = "data-labex-completo"  # Dataset para processar
TEST_DATASET = "data-labex"                 # Dataset de teste
TRAINED_MODEL_NAME = "data-labex-teste"     # Nome do modelo treinado
DATASET_NAME = PROCESSING_DATASET

# Divisão do dataset
TRAIN_SPLIT = "treino"
VAL_SPLIT = "validacao"
TEST_SPLIT = "teste"

# ==============================================================
# SEÇÃO 5: HIPERPARÂMETROS DE TREINO
# ==============================================================

TIME_STEPS = 30                     # Janela temporal
NUM_JOINTS = 17                     # Keypoints COCO
NUM_CHANNELS = 2                    # x, y
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
EPOCHS = 100

# ==============================================================
# SEÇÃO 6: PARÂMETROS DE TESTE
# ==============================================================

CLASSE2_THRESHOLD = 0.50            # Threshold para classificar como anômalo

# Vídeo & Codecs
OPENH264_DLL = "openh264-1.8.0-win64.dll"

# ==============================================================
# SEÇÃO 7: PATHS
# ==============================================================

ROOT = Path(__file__).resolve().parent

# Adiciona ROOT ao PATH do sistema para encontrar DLLs (como OpenH264)
if str(ROOT) not in os.environ["PATH"]:
    os.environ["PATH"] = str(ROOT) + os.pathsep + os.environ["PATH"]

# Modelos
MODELS_DIR = ROOT / "detector" / "modelos"
YOLO_PATH = MODELS_DIR / YOLO_MODEL
RTMPOSE_DIR = ROOT / "app" / "modelos"
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
ANNOTATIONS_OUTPUT_DIR = ROOT / "resultados-anotacoes"

# ReID Manual
REID_OUTPUT_DIR = ROOT / "resultados-reidentificacoes"
REID_MANUAL_SUFFIX = "-reid"
REID_MANUAL_LABELS_FILENAME = "labels_reid.json"

# RTMPose para pré-processamento
RTMPOSE_PREPROCESSING_PATH = RTMPOSE_PATH

# Parâmetros de sequência
MAX_FRAMES_PER_SEQUENCE = 30
FPS_TARGET = 30.0
FRAME_DISPLAY_W = 1280
FRAME_DISPLAY_H = 720

# Treinamento
TRAINING_INPUT_DIR = PROCESSING_OUTPUT_DIR
TRAINED_MODELS_DIR = ROOT / "modelos-lstm-treinados"
TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_DIR = TRAINED_MODELS_DIR / f"{TEMPORAL_MODEL}-{TRAINED_MODEL_NAME}"

# Teste
TEST_DATASETS_ROOT = ROOT / "datasets" / TEST_DATASET
TEST_DIR = TEST_DATASETS_ROOT / TEST_SPLIT / "videos"
TEST_LABELS_JSON = TEST_DATASETS_ROOT / TEST_SPLIT / "anotacoes" / "labels.json"
TEST_MODEL_DIR = MODEL_SAVE_DIR

# Relatórios
TEST_REPORTS_DIR = ROOT / "relatorios-testes"
TEST_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Dados de treino
TRAINING_DATA_PATH = ROOT / "datasets" / PROCESSING_DATASET / "treino" / "data" / "data.pt"
TRAINING_LABELS_PATH = ROOT / "datasets" / PROCESSING_DATASET / "treino" / "labels.json"
RETRAIN_MODELS_DIR = ROOT / "retreinamentos"

# Nomes de arquivos de saída
MODEL_BEST_FILENAME = "model_best.pt"
MODEL_FINAL_FILENAME = "model_final.pt"
NORM_STATS_FILENAME = "norm_stats.pt"

# BoTSORT
BOTSORT_YAML_PATH = ROOT / "tracker" / "configuracao" / "botsort_custom.yaml"

BOT_SORT_CONFIG = {
    "tracker_type": "botsort",
    "track_high_thresh": 0.5,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.5,
    "track_buffer": 300,
    "match_thresh": 0.50,       # Aumentado de 0.30 para 0.50 (mais tolerante para fusão)
    "appearance_thresh": 0.25,  # Aumentado de 0.20 (reduz exigência visual estrita)
    "proximity_thresh": 0.6,
    "gmc_method": "orb",
    "fuse_score": True,
    "with_reid": True,
    "model": str(OSNET_PATH),
}
# Paths legados (compatibilidade)
DATASETS_ROOT = TEST_DATASETS_ROOT
TRAIN_DIR = TEST_DATASETS_ROOT / TRAIN_SPLIT / "videos"
VAL_DIR = TEST_DATASETS_ROOT / VAL_SPLIT / "videos"
PREPROCESSING_OUTPUT = ROOT / "resultado_processamento"

# ==============================================================
# SEÇÃO 8: CONSTANTES FIXAS (NÃO ALTERE)
# ==============================================================

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
    print(" DETALHES DO RASTREADOR (YOLO + BoTSORT)")
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
