# ==============================================================
# neurapose-backend/app/configuracao/config.py
# ==============================================================

import argparse
from pathlib import Path

# ------------------------------------------------------------------
# Argumentos de linha de comando
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--model", type=str, default=None, help="Sobrescreve TEMPORAL_MODEL")
parser.add_argument("--relatorio", type=str, default="teste_v1", help="Nome do relatorio")
parser.add_argument("--show", action="store_true", help="Mostra video durante processamento")
parser.add_argument("--input-dir", type=str, default="", help="Pasta de videos")
parser.add_argument("--input-video", type=str, default="", help="Video unico")
args, _ = parser.parse_known_args()

# ------------------------------------------------------------------
# Importa TUDO do config_master.py
# ------------------------------------------------------------------
from config_master import *


# ------------------------------------------------------------------
# Alias para compatibilidade com codigo antigo
# ------------------------------------------------------------------
YOLO_MODEL_NAME = YOLO_MODEL

if args.model:
    MODEL_NAME = args.model.lower()
else:
    MODEL_NAME = TEMPORAL_MODEL

# Paths especificos de teste
TEST = TEST_SPLIT
RALATORIO = args.relatorio

# DATASET_DIR baseado em args
if args.input_video:
    DATASET_DIR = Path(args.input_video)
elif args.input_dir:
    DATASET_DIR = Path(args.input_dir)
else:
    DATASET_DIR = TEST_DIR

# Caminho dinamico do modelo treinado
# USA TRAINED_MODEL_NAME do config_master.py
MODEL_DIR_NAME = f"{MODEL_NAME}-{TRAINED_MODEL_NAME}"
MODEL_DIR = TRAINED_MODELS_DIR / MODEL_DIR_NAME
BEST_MODEL_PATH = MODEL_DIR / "model_best.pt"

# Labels de teste (Ground Truth)
LABELS_TEST_PATH = DATASETS_ROOT / TEST / "anotacoes" / "labels.json"

# Diretorios de saida para relatorios
RELATORIOS_ROOT = ROOT / "relatorios-teste" / TEST_DATASET / RALATORIO
PREDICOES_DIR = RELATORIOS_ROOT / "predicoes"
JSONS_DIR = RELATORIOS_ROOT / "jsons"
METRICAS_DIR = RELATORIOS_ROOT / "metricas"

# Cria diretorios
for p in [PREDICOES_DIR, JSONS_DIR, METRICAS_DIR]:
    p.mkdir(parents=True, exist_ok=True)