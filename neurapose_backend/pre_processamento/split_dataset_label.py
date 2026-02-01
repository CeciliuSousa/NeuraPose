# ==============================================================================
# neurapose_backend/pre_processamento/split_dataset_label.py
# ==============================================================================

"""
Split dataset por IDs (não por vídeos).

Garante:
1. Treino balanceado 1:1 por ID (mesma quantidade NORMAL e FURTO)
2. Sem duplicação - cada ID aparece em exatamente 1 destino
3. Sem deleção - dados originais intactos
4. Labels consistentes com JSONs
"""

import json
import shutil
import numpy as np
from pathlib import Path

import neurapose_backend.config_master as cm


def extract_class_from_label(label_info):
    """Extrai classe de label simples ('FURTO') ou complexo ({'classe': 'FURTO', ...})."""
    if isinstance(label_info, dict):
        return label_info.get("classe", "NORMAL").lower()
    return str(label_info).lower()


def flatten_labels(labels: dict, class1: str, class2: str):
    """
    Transforma labels por vídeo em lista de tuples (video, id, classe).
    Retorna também contagem por classe.
    """
    all_ids = []  # [(video_stem, pid, classe_lower), ...]
    
    for video_stem, id_map in labels.items():
        for pid, label_info in id_map.items():
            classe = extract_class_from_label(label_info)
            all_ids.append((video_stem, str(pid), classe))
    
    # Separa por classe
    class1_ids = [x for x in all_ids if x[2] == class1]
    class2_ids = [x for x in all_ids if x[2] == class2]
    
    return all_ids, class1_ids, class2_ids


def find_file(base_dir: Path, video_name: str, exts):
    """Procura arquivo por nome com várias extensões."""
    patterns = {video_name, video_name.replace("_", ""),
                video_name.replace("normal", "normal_").replace("furto", "furto_")}
    for pat in patterns:
        for ext in exts:
            found = list(base_dir.glob(f"{pat}{ext}"))
            if found:
                return found[0]
    matches = list(base_dir.glob(f"*{video_name.replace('_', '')}*"))
    return matches[0] if matches else None


def run_split(root_path: Path, dataset_name: str, output_root: Path, 
              train_split: str, test_split: str, train_ratio: float = 0.85, logger=None):
    """
    Split por IDs com balanceamento 1:1.
    """
    def _log(msg):
        if logger:
            logger.info(msg)
        print(msg)
    
    # Paths de entrada
    base = root_path
    videos_dir = base / "videos"
    jsons_dir = base / "jsons"
    labels_path = base / "anotacoes" / "labels.json"

    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json nao encontrado: {labels_path}")

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    # Paths de saída
    output_base = Path(output_root) / dataset_name
    train_dados = output_base / train_split / "dados"
    train_anotacoes = output_base / train_split / "anotacoes"
    test_videos_dir = output_base / test_split / "videos"
    test_anotacoes = output_base / test_split / "anotacoes"

    for d in [train_dados, train_anotacoes, test_videos_dir, test_anotacoes]:
        d.mkdir(parents=True, exist_ok=True)

    class1 = cm.CLASSE1.lower()  # normal
    class2 = cm.CLASSE2.lower()  # furto

    # ============================================================
    # PASSO 1: Flatten - Lista de todos os IDs
    # ============================================================
    _log("[1/6] Extraindo IDs de todos os vídeos...")
    all_ids, class1_ids, class2_ids = flatten_labels(labels, class1, class2)
    
    total_ids = len(all_ids)
    total_class1 = len(class1_ids)
    total_class2 = len(class2_ids)
    
    _log(f"[INFO] Total IDs: {total_ids}")
    _log(f"[INFO] {class1.upper()}: {total_class1}")
    _log(f"[INFO] {class2.upper()}: {total_class2}")

    # ============================================================
    # PASSO 2: Balanceamento 1:1
    # ============================================================
    _log("[2/6] Calculando split balanceado...")
    
    min_class = min(total_class1, total_class2)
    train_per_class = max(1, int(min_class * train_ratio))
    
    # Garante que sobra pelo menos 1 para teste
    if min_class > 1 and train_per_class >= min_class:
        train_per_class = min_class - 1
    
    _log(f"[SPLIT] Treino por classe: {train_per_class}")

    # ============================================================
    # PASSO 3: Seleção aleatória (seed fixo)
    # ============================================================
    _log("[3/6] Selecionando IDs para treino/teste...")
    
    np.random.seed(42)
    np.random.shuffle(class1_ids)
    np.random.shuffle(class2_ids)
    
    train_class1 = class1_ids[:train_per_class]
    train_class2 = class2_ids[:train_per_class]
    test_class1 = class1_ids[train_per_class:]
    test_class2 = class2_ids[train_per_class:]
    
    train_ids = train_class1 + train_class2
    test_ids = test_class1 + test_class2
    
    _log(f"[SPLIT] Treino: {len(train_class1)} {class1.upper()} + {len(train_class2)} {class2.upper()} = {len(train_ids)}")
    _log(f"[SPLIT] Teste: {len(test_class1)} {class1.upper()} + {len(test_class2)} {class2.upper()} = {len(test_ids)}")

    # ============================================================
    # PASSO 4: Gerar labels filtrados por ID
    # ============================================================
    def build_filtered_labels(id_list, orig_labels):
        """Cria labels.json apenas com IDs selecionados."""
        filtered = {}
        for video_stem, pid, classe in id_list:
            if video_stem not in filtered:
                filtered[video_stem] = {}
            # Pega label original para manter formato complexo se existir
            filtered[video_stem][pid] = orig_labels[video_stem][pid]
        return filtered
    
    train_labels = build_filtered_labels(train_ids, labels)
    test_labels = build_filtered_labels(test_ids, labels)

    # ============================================================
    # PASSO 5: Copiar arquivos
    # ============================================================
    _log("[4/6] Copiando JSONs para treino...")
    train_videos_set = set(x[0] for x in train_ids)
    copied_train = 0
    for video_stem in train_videos_set:
        json_path = find_file(jsons_dir, video_stem, [".json"])
        if not json_path:
            json_path = find_file(cm.PROCESSING_JSONS_DIR, video_stem, [".json"])
        if json_path and json_path.exists():
            shutil.copy(json_path, train_dados / f"{video_stem}.json")
            copied_train += 1
    _log(f"[OK] {copied_train}/{len(train_videos_set)} JSONs copiados para treino")

    _log("[5/6] Copiando vídeos para teste...")
    test_videos_set = set(x[0] for x in test_ids)
    copied_test = 0
    for video_stem in test_videos_set:
        vid_path = find_file(videos_dir, video_stem, [".mp4", ".m4v"])
        if not vid_path:
            vid_path = find_file(cm.PROCESSING_VIDEOS_DIR, video_stem, [".mp4", ".m4v"])
        if vid_path and vid_path.exists():
            shutil.copy(vid_path, test_videos_dir / f"{video_stem}{vid_path.suffix}")
            copied_test += 1
    _log(f"[OK] {copied_test}/{len(test_videos_set)} vídeos copiados para teste")

    # Salvar labels filtrados
    with open(train_anotacoes / "labels.json", "w", encoding="utf-8") as f:
        json.dump(train_labels, f, indent=4, ensure_ascii=False)
    
    with open(test_anotacoes / "labels.json", "w", encoding="utf-8") as f:
        json.dump(test_labels, f, indent=4, ensure_ascii=False)

    # Referência de balanceamento
    balance_ref = {
        "train_ids": [(v, p, c) for v, p, c in train_ids],
        "total_train": len(train_ids),
        "class1_train": len(train_class1),
        "class2_train": len(train_class2),
    }
    with open(train_anotacoes / "balance_reference.json", "w", encoding="utf-8") as f:
        json.dump(balance_ref, f, indent=4, ensure_ascii=False)

    # ============================================================
    # PASSO 6: Validação de consistência
    # ============================================================
    _log("[6/6] Validando consistência...")
    
    # Verifica sem duplicação
    train_set = set((v, p) for v, p, c in train_ids)
    test_set = set((v, p) for v, p, c in test_ids)
    overlap = train_set & test_set
    
    if overlap:
        _log(f"[ERRO] IDs duplicados entre treino e teste: {len(overlap)}")
    else:
        _log("[CHECK] Sem duplicação entre treino/teste ✓")
    
    # Verifica sem perda
    total_split = len(train_ids) + len(test_ids)
    if total_split == total_ids:
        _log(f"[CHECK] Todos IDs preservados: {len(train_ids)} + {len(test_ids)} = {total_ids} ✓")
    else:
        _log(f"[ERRO] Perda de IDs: {total_ids} originais, {total_split} após split")
    
    # Verifica balanceamento
    if len(train_class1) == len(train_class2):
        _log(f"[CHECK] Treino balanceado 1:1 ({len(train_class1)} cada) ✓")
    else:
        _log(f"[WARN] Treino desbalanceado: {len(train_class1)} vs {len(train_class2)}")

    # Summary
    summary_path = output_base / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Split por IDs (balanceado 1:1)\n")
        f.write(f"Total IDs originais: {total_ids}\n")
        f.write(f"Treino: {len(train_ids)} IDs ({len(train_class1)} {class1}, {len(train_class2)} {class2})\n")
        f.write(f"Teste: {len(test_ids)} IDs ({len(test_class1)} {class1}, {len(test_class2)} {class2})\n")
        f.write(f"Vídeos treino: {len(train_videos_set)}\n")
        f.write(f"Vídeos teste: {len(test_videos_set)}\n")

    # _log(f"[OK] Resumo salvo: {summary_path}")
    # _log(f"[OK] Split concluído! Saída: {output_base}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Split dataset por IDs (balanceado 1:1)")

    parser.add_argument("--input-dir-process", "--input-root", dest="root",
                        default=str(cm.PROCESSING_OUTPUT_DIR),
                        help=f"Pasta processada de entrada (default: {cm.PROCESSING_OUTPUT_DIR})")
    parser.add_argument("--dataset-name", dest="dataset", default=cm.PROCESSING_DATASET,
                        help=f"Nome do dataset (default: {cm.PROCESSING_DATASET})")
    parser.add_argument("--output-root", dest="output_root", default=str(cm.TEST_DATASETS_ROOT.parent),
                        help=f"Root onde os datasets serão escritos (default: {cm.TEST_DATASETS_ROOT.parent})")
    parser.add_argument("--train-split", dest="train_split", default=cm.TRAIN_SPLIT,
                        help=f"Nome da pasta de treino (default: {cm.TRAIN_SPLIT})")
    parser.add_argument("--test-split", dest="test_split", default=cm.TEST_SPLIT,
                        help=f"Nome da pasta de teste (default: {cm.TEST_SPLIT})")

    args = parser.parse_args()

    run_split(Path(args.root), args.dataset, Path(args.output_root), args.train_split, args.test_split)


if __name__ == "__main__":
    main()
