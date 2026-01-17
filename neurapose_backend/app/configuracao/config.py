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
parser.add_argument("--model-dir", type=str, default="", help="Diretorio do modelo treinado")
args, _ = parser.parse_known_args()

# ------------------------------------------------------------------
# Importa TUDO do config_master.py
# ------------------------------------------------------------------
from neurapose_backend.config_master import *


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
RELATORIO = args.relatorio

# DATASET_DIR e DATASET_NAME baseado em args
if args.input_video:
    DATASET_DIR = Path(args.input_video)
    DATASET_NAME = DATASET_DIR.parent.parent.name # assume estrutura datasets/DS_NAME/teste/videos/file.mp4
elif args.input_dir:
    DATASET_DIR = Path(args.input_dir)
    # Se for o path completo de um dataset, pegamos o nome dele
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

# Caminho dinamico do modelo treinado
if args.model_dir:
    MODEL_DIR = Path(args.model_dir)
else:
    # USA TRAINED_MODEL_NAME do config_master.py
    MODEL_DIR_NAME = f"{MODEL_NAME}-{TRAINED_MODEL_NAME}"
    MODEL_DIR = TRAINED_MODELS_DIR / MODEL_DIR_NAME

BEST_MODEL_PATH = MODEL_DIR / "model_best.pt"
if not BEST_MODEL_PATH.exists() and (MODEL_DIR / "model_final.pt").exists():
     BEST_MODEL_PATH = MODEL_DIR / "model_final.pt"

# Labels de teste (Ground Truth)
LABELS_TEST_PATH = ROOT / "datasets" / DATASET_NAME / "teste" / "anotacoes" / "labels.json"
# Fallback caso não encontre na estrutura padrão /teste/anotacoes/
if not LABELS_TEST_PATH.exists():
    LABELS_TEST_PATH = ROOT / "datasets" / DATASET_NAME / "anotacoes" / "labels.json"

# Diretorios de saida para relatorios
RELATORIOS_ROOT = ROOT / "relatorios-testes" / DATASET_NAME / RELATORIO
PREDICOES_DIR = RELATORIOS_ROOT / "predicoes"
JSONS_DIR = RELATORIOS_ROOT / "jsons"
METRICAS_DIR = RELATORIOS_ROOT / "metricas"

# Cria diretorios
for p in [PREDICOES_DIR, JSONS_DIR, METRICAS_DIR]:
    p.mkdir(parents=True, exist_ok=True)