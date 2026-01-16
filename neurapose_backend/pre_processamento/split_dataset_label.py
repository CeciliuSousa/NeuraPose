# ==============================================================================
# neurapose_backend/pre_processamento/split_dataset_label.py
# ==============================================================================

"""
Split dataset CLI (versão com nome de módulo válido).
Uso via: python -m neurapose.pre_processamento.split_dataset_label --input-dir-process <path> --output-root <path> --dataset-name <name>

Mantém comportamento idêntico ao script original `split-dataset-label.py` quando nenhum argumento é passado.
"""

import json
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Adiciona root ao path


# Configuracoes centralizadas (valores padrao)
from config_master import (
    PROCESSING_DATASET,
    PROCESSING_OUTPUT_DIR,
    PROCESSING_VIDEOS_DIR,

    PROCESSING_JSONS_DIR,
    TEST_DATASETS_ROOT,
    TRAIN_SPLIT,
    TEST_SPLIT,
    CLASSE1,
    CLASSE2,
)


def get_major_class(video_id, labels):
    primeiraClasse = CLASSE1.lower()
    segundaClasse = CLASSE2.lower()
    pessoas = labels[video_id]
    counts = {primeiraClasse: 0, segundaClasse: 0}
    for v in pessoas.values():
        if v.lower() in counts:
            counts[v.lower()] += 1
    return max(counts, key=counts.get)


def find_file(base_dir: Path, video_name: str, exts):
    patterns = {video_name, video_name.replace("_", ""),
                video_name.replace("normal", "normal_").replace("furto", "furto_")}
    for pat in patterns:
        for ext in exts:
            found = list(base_dir.glob(f"{pat}{ext}"))
            if found:
                return found[0]
    matches = list(base_dir.glob(f"*{video_name.replace('_', '')}*"))
    return matches[0] if matches else None


def copy_train_files(videos, dados_dir, anotacoes_dir, label_dict):
    subset_labels = {}
    for video in videos:
        json_path = find_file(dados_dir.parent / "jsons", video, [".json"]) if (dados_dir.parent / "jsons").exists() else find_file(PROCESSING_JSONS_DIR, video, [".json"])
        if json_path and json_path.exists():
            shutil.copy(json_path, dados_dir / f"{video}.json")
        subset_labels[video] = label_dict[video]

    with open(anotacoes_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(subset_labels, f, indent=4, ensure_ascii=False)

    print(f"[OK] {len(subset_labels)} JSONs copiados para treino")
    return subset_labels


def copy_test_files(videos, videos_dir_out, anotacoes_dir, label_dict):
    subset_labels = {}
    for video in videos:
        vid_path = find_file(videos_dir_out.parent / "videos", video, [".mp4", ".m4v"]) if (videos_dir_out.parent / "videos").exists() else find_file(PROCESSING_VIDEOS_DIR, video, [".mp4", ".m4v"])
        if vid_path and vid_path.exists():
            shutil.copy(vid_path, videos_dir_out / f"{video}{vid_path.suffix}")
        subset_labels[video] = label_dict[video]

    with open(anotacoes_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(subset_labels, f, indent=4, ensure_ascii=False)

    print(f"[OK] {len(subset_labels)} videos copiados para teste")
    return subset_labels


def run_split(root_path: Path, dataset_name: str, output_root: Path, train_split: str, test_split: str, train_ratio: float = 0.85):
    # Paths base
    base = root_path
    videos_dir = base / "videos"
    jsons_dir = base / "jsons"
    labels_path = base / "anotacoes" / "labels.json"

    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json nao encontrado: {labels_path}")

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    # Saida
    output_base = Path(output_root) / dataset_name
    train_dados = output_base / train_split / "dados"
    train_anotacoes = output_base / train_split / "anotacoes"
    test_videos = output_base / test_split / "videos"
    test_anotacoes = output_base / test_split / "anotacoes"

    for d in [train_dados, train_anotacoes, test_videos, test_anotacoes]:
        d.mkdir(parents=True, exist_ok=True)

    primeiraClasse = CLASSE1.lower()
    segundaClasse = CLASSE2.lower()

    video_names = list(labels.keys())
    labels_majoritarias = [get_major_class(v, labels) for v in video_names]

    normal_videos = [v for v in video_names if get_major_class(v, labels) == primeiraClasse]
    furto_videos = [v for v in video_names if get_major_class(v, labels) == segundaClasse]

    np.random.seed(42)
    np.random.shuffle(normal_videos)
    np.random.shuffle(furto_videos)

    min_count = min(len(normal_videos), len(furto_videos))
    # Usa train_ratio recebido, com fallback seguro para datasets pequenos
    train_size_per_class = max(1, int(min_count * train_ratio)) if min_count > 0 else 0
    # Se após cálculo sobrar 0 para teste, ajusta para deixar pelo menos 1 para teste
    if min_count > 1 and train_size_per_class >= min_count:
        train_size_per_class = min_count - 1

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

    print("\n[COPY] Copiando treino...")
    train_subset = copy_train_files(train_video_list, train_dados, train_anotacoes, labels)

    print("\n[COPY] Copiando teste...")
    test_subset = copy_test_files(test_video_list, test_videos, test_anotacoes, labels)

    balance_ref_path = output_base / train_split / "anotacoes" / "balance_reference.json"
    with open(balance_ref_path, "w", encoding="utf-8") as f:
        json.dump({"balanced_list": train_video_list}, f, indent=4, ensure_ascii=False)

    summary_path = output_base / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Total: {len(labels)} videos\n")
        f.write(f"Treino: {len(train_subset)} ({len(train_normal)} {primeiraClasse}, {len(train_furto)} {segundaClasse})\n")
        f.write(f"Teste: {len(test_subset)} ({len(test_normal)} {primeiraClasse}, {len(test_furto)} {segundaClasse})\n")

    print(f"\n[OK] Resumo salvo: {summary_path}")
    print(f"[OK] Split concluido! Saida: {output_base}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Split dataset (treino/teste) a partir de pastas processadas")

    parser.add_argument("--input-dir-process", "--input-root", dest="root",
                        default=str(PROCESSING_OUTPUT_DIR),
                        help=f"Pasta processada de entrada (default: {PROCESSING_OUTPUT_DIR})")
    parser.add_argument("--dataset-name", dest="dataset", default=PROCESSING_DATASET,
                        help=f"Nome do dataset (default: {PROCESSING_DATASET})")
    parser.add_argument("--output-root", dest="output_root", default=str(TEST_DATASETS_ROOT.parent),
                        help=f"Root onde os datasets serao escritos (default: {TEST_DATASETS_ROOT.parent})")
    parser.add_argument("--train-split", dest="train_split", default=TRAIN_SPLIT,
                        help=f"Nome da pasta de treino (default: {TRAIN_SPLIT})")
    parser.add_argument("--test-split", dest="test_split", default=TEST_SPLIT,
                        help=f"Nome da pasta de teste (default: {TEST_SPLIT})")

    args = parser.parse_args()

    run_split(Path(args.root), args.dataset, Path(args.output_root), args.train_split, args.test_split)


if __name__ == "__main__":
    main()
