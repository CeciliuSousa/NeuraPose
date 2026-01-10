# neurapose/pre_processamento/split-dataset-label.py
"""
Divide dataset em treino (80%) e teste (20%).
Balanceia classes 1:1 no conjunto de treino.
"""

import sys
import json
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Adiciona root ao path


# Configuracoes centralizadas
from neurapose.config_master import (
    PROCESSING_DATASET,
    PROCESSING_OUTPUT_DIR,
    PROCESSING_VIDEOS_DIR,
    PROCESSING_JSONS_DIR,
    PROCESSING_ANNOTATIONS_DIR,
    TEST_DATASETS_ROOT,
    TRAIN_SPLIT,
    TEST_SPLIT,
    CLASSE1,
    CLASSE2,
)  

# ------------------------------------------------------------
# PATHS DERIVADOS DO CONFIG
# ------------------------------------------------------------
data_name = PROCESSING_DATASET
base = PROCESSING_OUTPUT_DIR
videos_dir = PROCESSING_VIDEOS_DIR
jsons_dir = PROCESSING_JSONS_DIR
labels_path = PROCESSING_ANNOTATIONS_DIR / "labels.json"

# Saida
output_base = TEST_DATASETS_ROOT.parent / data_name
train_dados = output_base / TRAIN_SPLIT / "dados"
train_anotacoes = output_base / TRAIN_SPLIT / "anotacoes"
test_videos = output_base / TEST_SPLIT / "videos"
test_anotacoes = output_base / TEST_SPLIT / "anotacoes"

# Nomes das classes
primeiraClasse = CLASSE1.lower()
segundaClasse = CLASSE2.lower()

# Cria diretorios
for d in [train_dados, train_anotacoes, test_videos, test_anotacoes]:
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# FUNCOES
# ------------------------------------------------------------
def get_major_class(video_id, labels):
    """Retorna classe majoritaria do video."""
    pessoas = labels[video_id]
    counts = {primeiraClasse: 0, segundaClasse: 0}
    for v in pessoas.values():
        if v.lower() in counts:
            counts[v.lower()] += 1
    return max(counts, key=counts.get)


def find_file(base_dir: Path, video_name: str, exts):
    """Procura arquivo com variantes do nome."""
    patterns = {video_name, video_name.replace("_", ""),
                video_name.replace("normal", "normal_").replace("furto", "furto_")}
    for pat in patterns:
        for ext in exts:
            found = list(base_dir.glob(f"{pat}{ext}"))
            if found:
                return found[0]
    matches = list(base_dir.glob(f"*{video_name.replace('_', '')}*"))
    return matches[0] if matches else None


def copy_train_files(videos, dados_dir, anotacoes_dir, label_dict, labels):
    """Copia JSONs para pasta de treino."""
    subset_labels = {}
    for video in videos:
        json_path = find_file(jsons_dir, video, [".json"])
        if json_path and json_path.exists():
            shutil.copy(json_path, dados_dir / f"{video}.json")
        subset_labels[video] = label_dict[video]
    
    with open(anotacoes_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(subset_labels, f, indent=4, ensure_ascii=False)

    print(f"[OK] {len(subset_labels)} JSONs copiados para treino")
    return subset_labels


def copy_test_files(videos, videos_dir_out, anotacoes_dir, label_dict, labels):
    """Copia videos para pasta de teste."""
    subset_labels = {}
    for video in videos:
        vid_path = find_file(videos_dir, video, [".mp4", ".m4v"])
        if vid_path and vid_path.exists():
            shutil.copy(vid_path, videos_dir_out / f"{video}{vid_path.suffix}")
        subset_labels[video] = label_dict[video]
    
    with open(anotacoes_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(subset_labels, f, indent=4, ensure_ascii=False)

    print(f"[OK] {len(subset_labels)} videos copiados para teste")
    return subset_labels


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if not labels_path.exists():
    raise FileNotFoundError(f"labels.json nao encontrado: {labels_path}")

with open(labels_path, "r", encoding="utf-8") as f:
    labels = json.load(f)

print(f"[INFO] {len(labels)} videos no labels.json")
print(f"[INFO] Dataset: {data_name}")
print(f"[INFO] Classes: {primeiraClasse} / {segundaClasse}")

video_names = list(labels.keys())

# Separa por classe
labels_majoritarias = [get_major_class(v, labels) for v in video_names]
counts = {c: labels_majoritarias.count(c) for c in set(labels_majoritarias)}
print(f"[INFO] Distribuicao: {counts}")

normal_videos = [v for v in video_names if get_major_class(v, labels) == primeiraClasse]
furto_videos = [v for v in video_names if get_major_class(v, labels) == segundaClasse]

# Embaralha
np.random.seed(42)
np.random.shuffle(normal_videos)
np.random.shuffle(furto_videos)

# Split 85/15 balanceado
min_count = min(len(normal_videos), len(furto_videos))
train_size_per_class = int(min_count * 0.85)

train_normal = normal_videos[:train_size_per_class]
train_furto = furto_videos[:train_size_per_class]
train_video_list = train_normal + train_furto
np.random.shuffle(train_video_list)

test_normal = normal_videos[train_size_per_class:]
test_furto = furto_videos[train_size_per_class:]
test_video_list = test_normal + test_furto
np.random.shuffle(test_video_list)

print(f"\n[SPLIT] Treino: {len(train_normal)} {primeiraClasse} + {len(train_furto)} {segundaClasse}")
print(f"[SPLIT] Teste: {len(test_normal)} {primeiraClasse} + {len(test_furto)} {segundaClasse}")

# Copia arquivos
print("\n[COPY] Copiando treino...")
train_subset = copy_train_files(train_video_list, train_dados, train_anotacoes, labels, labels)

print("\n[COPY] Copiando teste...")
test_subset = copy_test_files(test_video_list, test_videos, test_anotacoes, labels, labels)

# Salva referencia
balance_ref_path = output_base / TRAIN_SPLIT / "anotacoes" / "balance_reference.json"
with open(balance_ref_path, "w", encoding="utf-8") as f:
    json.dump({"balanced_list": train_video_list}, f, indent=4, ensure_ascii=False)

# Resumo
summary_path = output_base / "summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(f"Dataset: {data_name}\n")
    f.write(f"Total: {len(labels)} videos\n")
    f.write(f"Treino: {len(train_subset)} ({len(train_normal)} {primeiraClasse}, {len(train_furto)} {segundaClasse})\n")
    f.write(f"Teste: {len(test_subset)} ({len(test_normal)} {primeiraClasse}, {len(test_furto)} {segundaClasse})\n")

print(f"\n[OK] Resumo salvo: {summary_path}")
print(f"[OK] Split concluido!")
print(f"[OK] Saida: {output_base}")
