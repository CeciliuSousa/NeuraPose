# neurapose-backend/app/configuracao/config.py

import argparse
from pathlib import Path


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--model", type=str, default=None, help="Sobrescreve TEMPORAL_MODEL")
parser.add_argument("--relatorio", type=str, default="teste_v1", help="Nome do relatorio")
parser.add_argument("--show", action="store_true", help="Mostra video durante processamento")
parser.add_argument("--input-dir", type=str, default="", help="Pasta de videos")
parser.add_argument("--input-video", type=str, default="", help="Video unico")
parser.add_argument("--model-dir", type=str, default="", help="Diretorio do modelo treinado")
args, _ = parser.parse_known_args()


import neurapose_backend.config_master as cm
from neurapose_backend.nucleo.user_config_manager import UserConfigManager


user_config = UserConfigManager.load_config()

for k, v in user_config.items():
    if hasattr(cm, k):
        setattr(cm, k, v)
if hasattr(cm, "RTMPOSE_INPUT_SIZE"):
    if isinstance(cm.RTMPOSE_INPUT_SIZE, (tuple, list)) and len(cm.RTMPOSE_INPUT_SIZE) == 2:
        cm.SIMCC_W = cm.RTMPOSE_INPUT_SIZE[0]
        cm.SIMCC_H = cm.RTMPOSE_INPUT_SIZE[1]

cm_vars = {k: v for k, v in vars(cm).items() if not k.startswith("__")}
globals().update(cm_vars)

for k, v in user_config.items():
    globals()[k] = v

YOLO_MODEL_NAME = YOLO_MODEL

if args.model:
    MODEL_NAME = args.model.lower()
else:
    MODEL_NAME = TEMPORAL_MODEL

TEST = TEST_SPLIT
RELATORIO = args.relatorio

if args.input_video:
    DATASET_DIR = Path(args.input_video)
    DATASET_NAME = DATASET_DIR.parent.parent.name
elif args.input_dir:
    DATASET_DIR = Path(args.input_dir)
    p = Path(args.input_dir)
    if p.name == "videos" and p.parent.name == "teste":
        DATASET_NAME = p.parent.parent.name
    elif p.name == "videos" or p.name == "teste":
        DATASET_NAME = p.parent.name
    else:
        DATASET_NAME = p.name
else:
    DATASET_DIR = TEST_DIR
    DATASET_NAME = TEST_DATASET

if args.model_dir:
    MODEL_DIR = Path(args.model_dir)
else:
    MODEL_DIR_NAME = f"{MODEL_NAME}-{TRAINED_MODEL_NAME}"
    MODEL_DIR = TRAINED_MODELS_DIR / MODEL_DIR_NAME

BEST_MODEL_PATH = MODEL_DIR / "model_best.pt"
if not BEST_MODEL_PATH.exists() and (MODEL_DIR / "model_final.pt").exists():
     BEST_MODEL_PATH = MODEL_DIR / "model_final.pt"

LABELS_TEST_PATH = Path(ROOT) / "datasets" / DATASET_NAME / "teste" / "anotacoes" / "labels.json"

if not LABELS_TEST_PATH.exists():
    LABELS_TEST_PATH = Path(ROOT) / "datasets" / DATASET_NAME / "anotacoes" / "labels.json"