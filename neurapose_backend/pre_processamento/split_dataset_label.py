# ==============================================================================
# neurapose_backend/pre_processamento/split_dataset_label.py
# ==============================================================================

"""
Split dataset por IDs (não por vídeos).

Garante:
1. Treino balanceado 1:1 por ID (mesma quantidade CLASSE1: "NORMAL" e CLASSE2: "ANOMALIA")
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
    """Extrai classe de label simples CLASSE2 ou complexo ({'classe': 'ANOMALIA', ...})."""
    if isinstance(label_info, dict):
        return label_info.get("classe", cm.CLASSE1).lower()
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
                video_name.replace(cm.CLASSE1.lower(), f"{cm.CLASSE1.lower()}_").replace(cm.CLASSE2.lower(), f"{cm.CLASSE2.lower()}_"),
                video_name.replace("_pose", ""),     # Handle Dirty Key -> Clean File
                video_name + "_pose"                 # Handle Clean Key -> Dirty File
               }
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

    # Detectar se é temporal (baseado na estrutura dos labels)
    # Temporal = Contém intervalos de tempo definidos (frames de inicio/fim)
    # Estrutura esperada: Video -> ID -> { "classe": "...", "intervals": [[start, end]] }
    is_temporal = False
    
    try:
        if labels and isinstance(labels, dict):
            # Procura profundidade para achar chaves de intervalo
            for vid, id_map in labels.items():
                if isinstance(id_map, dict):
                    for pid, data in id_map.items():
                        if isinstance(data, dict) and "intervals" in data:
                            is_temporal = True
                            break
                    if is_temporal: break
    except Exception as e:
         _log(f"[WARN] Erro ao verificar temporalidade: {e}")

    # Ajusta nome do dataset de saída
    final_dataset_name = dataset_name
    if is_temporal:
        if not final_dataset_name.endswith("-temporal"):
            final_dataset_name = f"{final_dataset_name}-temporal"
            _log(f"[INFO] Anotações com INTERVALOS detectadas. Dataset marcado como TEMPORAL: {final_dataset_name}")
    else:
        _log(f"[INFO] Anotações SIMPLES (sem intervalos) detectadas. Dataset mantido como: {final_dataset_name}")

    # Paths de saída
    output_base = Path(output_root) / final_dataset_name
    train_dados = output_base / train_split / "dados"
    train_anotacoes = output_base / train_split / "anotacoes"
    test_videos_dir = output_base / test_split / "videos"
    test_anotacoes = output_base / test_split / "anotacoes"

    for d in [train_dados, train_anotacoes, test_videos_dir, test_anotacoes]:
        d.mkdir(parents=True, exist_ok=True)

    class1 = cm.CLASSE1.lower()
    class2 = cm.CLASSE2.lower()

    # ============================================================
    # PASSO 1: Agrupar IDs por Vídeo
    # ============================================================
    _log("[1/6] Agrupando IDs por vídeo...")
    
    # Estrutura: {video_stem: {"class1": [id...], "class2": [id...]}}
    videos_map = {}
    
    for video_stem, id_map in labels.items():
        if video_stem not in videos_map:
            videos_map[video_stem] = {"class1": [], "class2": []}
            
        for pid, label_info in id_map.items():
            classe = extract_class_from_label(label_info)
            if classe == class1:
                videos_map[video_stem]["class1"].append(str(pid))
            elif classe == class2:
                videos_map[video_stem]["class2"].append(str(pid))

    # Métricas globais
    total_class1_global = sum(len(v["class1"]) for v in videos_map.values())
    total_class2_global = sum(len(v["class2"]) for v in videos_map.values())
    
    _log(f"[INFO] Total Global: {total_class1_global} {class1.upper()} | {total_class2_global} {class2.upper()}")

    # ============================================================
    # PASSO 2: Seleção de VÍDEOS para Treino
    # ============================================================
    _log("[2/6] Selecionando VÍDEOS para treino (Alvo: ~85% ANOMALIA)...")
    
    # Alvo de IDs de ANOMALIA no treino
    target_train_class2 = int(total_class2_global * train_ratio)
    
    # Separa vídeos que tem "ANOMALIA" dos que só tem "NORMAL"
    videos_with_class2 = [v for v, dados in videos_map.items() if len(dados["class2"]) > 0]
    videos_only_class1 = [v for v, dados in videos_map.items() if len(dados["class2"]) == 0]
    
    # Embaralha para aleatoriedade
    np.random.seed(42)
    np.random.shuffle(videos_with_class2)
    np.random.shuffle(videos_only_class1)
    
    train_videos = []
    current_train_class2 = 0
    
    # Greedy: Adiciona vídeos com ANOMALIA até atingir meta
    for v in videos_with_class2:
        if current_train_class2 < target_train_class2:
            train_videos.append(v)
            current_train_class2 += len(videos_map[v]["class2"])
        else:
            # Sim, vamos garantir separação por vídeo.
            pass
            
    # [FIX 353 vs 261] Supplementation Strategy
    # Se os vídeos de ANOMALIA não tiverem normais suficientes para 1:1,
    # precisamos pegar vídeos EXCLUSIVAMENTE normais para completar.
    
    # Recalcula quantas classes temos nos vídeos selecionados
    current_train_class1 = sum(len(videos_map[v]["class1"]) for v in train_videos)
    
    # Se faltar CLASSE1 para atingir a quantidade de ANOMALIAs (1:1)
    if current_train_class1 < current_train_class2:
        needed = current_train_class2 - current_train_class1
        _log(f"[BALANCE] Faltam {needed} IDs Normais para parear. Buscando em vídeos sem ANOMALIA...")
        
        for v in videos_only_class1:
            if current_train_class1 >= current_train_class2:
                break
            
            # Adiciona vídeo
            train_videos.append(v)
            current_train_class1 += len(videos_map[v]["class1"])
    
    # Agora sim temos videos suficientes.
    train_videos_set = set(train_videos)
    test_videos = [v for v in videos_map.keys() if v not in train_videos_set]
    
    _log(f"[SPLIT] Vídeos Treino: {len(train_videos)}")
    _log(f"[SPLIT] Vídeos Teste: {len(test_videos)}")
    _log(f"[SPLIT] IDs {cm.POSITIVE_CLASS_ID} no Treino: {current_train_class2}/{total_class2_global} ({current_train_class2/total_class2_global*100:.1f}%)")

    # ============================================================
    # PASSO 3: Balanceamento 1:1 INTERNO no Treino
    # ============================================================
    _log("[3/6] Aplicando balanceamento 1:1 nos IDs de Treino...")
    
    # Coleta todos os IDs disponíveis nos vídeos de treino
    train_pool_class1 = [] # [(video, id), ...]
    train_pool_class2 = [] # [(video, id), ...]
    
    for v in train_videos:
        for pid in videos_map[v]["class1"]:
            train_pool_class1.append((v, pid))
        for pid in videos_map[v]["class2"]:
            train_pool_class2.append((v, pid))
            
    # O numero de ANOMALIA no treino é o limitante
    limit = len(train_pool_class2)
    
    # Seleciona CLASSE1 randomicamente para igualar ANOMALIA
    if len(train_pool_class1) > limit:
        np.random.shuffle(train_pool_class1)
        selected_train_class1 = train_pool_class1[:limit]
    else:
        selected_train_class1 = train_pool_class1
        
    # IDs finais de treino (Tuple: video, id, classe)
    final_train_ids = []
    for v, pid in train_pool_class2:
        final_train_ids.append((v, pid, class2))
    for v, pid in selected_train_class1:
        final_train_ids.append((v, pid, class1))
        
    _log(f"[BALANCE] Treino Final: {len(train_pool_class2)} {class2.upper()} + {len(selected_train_class1)} {class1.upper()}")

    # ============================================================
    # PASSO 4: Gerar Labels
    # ============================================================
    def build_filtered_labels(id_list, orig_labels):
        """Cria labels.json filtrado para treino."""
        filtered = {}
        for video_stem, pid, _ in id_list:
            if video_stem not in filtered: filtered[video_stem] = {}
            filtered[video_stem][pid] = orig_labels[video_stem][pid]
        return filtered
        
    # Treino: Apenas IDs selecionados (balanceados)
    train_labels = build_filtered_labels(final_train_ids, labels)
    
    # Teste: TODOS os labels dos vídeos de teste (Consistência total)
    test_labels = {k: v for k, v in labels.items() if k in test_videos}

    # ============================================================
    # PASSO 5: Copiar Arquivos
    # ============================================================
    
    # A) TREINO (JSONs filtrados)
    _log("[4/6] Gerando dados de Treino...")
    copied_train = 0
    
    # Agrupa IDs de treino por vídeo para facilitar filtro
    train_ids_by_video = {}
    for v, pid, _ in final_train_ids:
        if v not in train_ids_by_video: train_ids_by_video[v] = set()
        train_ids_by_video[v].add(pid)
        
    for video_stem, allowed_ids in train_ids_by_video.items():
        # Busca JSON original
        json_path = find_file(jsons_dir, video_stem, [".json"])
        if not json_path:
            json_path = find_file(cm.PROCESSING_JSONS_DIR, video_stem, [".json"])
            
        if json_path and json_path.exists():
            # Carrega, filtra e salva
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Filtra registros (apenas IDs balanceados)
            filtered_data = [r for r in data if str(r.get("id_persistente", r.get("id"))) in allowed_ids]
            
            if filtered_data:
                with open(train_dados / f"{video_stem}.json", "w", encoding="utf-8") as f:
                    json.dump(filtered_data, f, indent=2)
                copied_train += 1
                
    _log(f"[OK] {copied_train} JSONs de treino gerados (filtrados).")

    # B) TESTE (Vídeos Brutos)
    _log("[5/6] Copiando vídeos de Teste...")
    copied_test = 0
    for video_stem in test_videos:
        vid_path = find_file(videos_dir, video_stem, [".mp4", ".m4v"])
        if not vid_path:
            vid_path = find_file(cm.PROCESSING_VIDEOS_DIR, video_stem, [".mp4", ".m4v"])
        if vid_path and vid_path.exists():
            shutil.copy(vid_path, test_videos_dir / f"{video_stem}{vid_path.suffix}")
            copied_test += 1
    _log(f"[OK] {copied_test} vídeos de teste copiados.")

    # Salvar Labels
    with open(train_anotacoes / "labels.json", "w", encoding="utf-8") as f:
        json.dump(train_labels, f, indent=4, ensure_ascii=False)
    with open(test_anotacoes / "labels.json", "w", encoding="utf-8") as f:
        json.dump(test_labels, f, indent=4, ensure_ascii=False)

    # Reference
    balance_ref = {
        "strategy": "video_split_balanced_ids",
        "train_videos": train_videos,
        "test_videos": test_videos,
        "train_stats": {
            "class1": len(selected_train_class1),
            "class2": len(train_pool_class2)
        }
    }
    with open(train_anotacoes / "balance_reference.json", "w", encoding="utf-8") as f:
        json.dump(balance_ref, f, indent=4, ensure_ascii=False)

    # ============================================================
    # PASSO 6: Validação
    # ============================================================
    # Verifica Interseção de Vídeos
    overlap_videos = set(train_videos) & set(test_videos)
    if overlap_videos:
        _log(f"[ERRO] CRÍTICO: Vídeos duplicados entre treino e teste: {overlap_videos}")
    else:
        _log("[CHECK] Separação de Vídeos: OK (Disjuntos) ✓")
        
    # Verifica Balanceamento Treino
    if len(selected_train_class1) == len(train_pool_class2):
         _log("[CHECK] Balanceamento Treino ID 1:1: OK ✓")
    else:
         _log(f"[WARN] Balanceamento imperfeito: {len(selected_train_class1)} vs {len(train_pool_class2)}")

    # Summary
    summary_path = output_base / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Split Strategy: Video-Level Split + ID Balancing\n")
        f.write(f"Treino: {len(train_videos)} Videos -> {len(final_train_ids)} IDs Balanceados (1:1)\n")
        f.write(f"Teste: {len(test_videos)} Videos -> Full Labels preserved\n")

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
