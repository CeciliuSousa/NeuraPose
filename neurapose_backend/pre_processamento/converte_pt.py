# ==============================================================================
# neurapose_backend/pre_processamento/converte_pt.py
# ==============================================================================

"""
Converte JSONs de keypoints para formato PyTorch (.pt).
Entrada: JSONs do pre-processamento + labels.json
Saida: data.pt com tensores de sequencias normalizadas

[NOVO] Suporta Anotação Temporal:
- Se label for String ("FURTO"): Pega sequencia inteira.
- Se label for Dict ({"classe": "FURTO", "intervals": [[10, 50], [90, 120]]}): 
  Gera MÚLTIPLAS sequencias (samples), uma para cada intervalo.
"""

import sys
import re
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import torch

# Adiciona root ao path para imports


# Configuracoes centralizadas
import neurapose_backend.config_master as cm
  

# ------------------------------------------------------------
# PATHS DINÂMICOS (PODEM SER SOBRESCRITOS PELO MAIN.PY)
# ------------------------------------------------------------
JSONS_DIR = cm.PROCESSING_JSONS_DIR
LABELS_PATH = cm.PROCESSING_ANNOTATIONS_DIR / "labels.json"

# Saída: sempre dentro de <pasta_do_labels>/../data/
OUTPUT_BASE = LABELS_PATH.parent.parent
OUT_PT = OUTPUT_BASE / "data" / "data.pt"
LOG_FILE = OUTPUT_BASE / "data" / "frames_invalidos.txt"
DEBUG_LOG = OUTPUT_BASE / "data" / "debug_log.txt"

OUT_PT.parent.mkdir(parents=True, exist_ok=True)

# Dimensoes da pose
C, V = cm.NUM_CHANNELS, cm.NUM_JOINTS  # C=2 (x,y), V=17 keypoints
max_frames = cm.MAX_FRAMES_PER_SEQUENCE
min_frames_validos = cm.MIN_FRAMES_PER_ID
np.random.seed(42)


def log(msg, console=True):
    """Escreve mensagem no log e console."""
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    line = f"{timestamp} {msg}"
    # Garante que diretorio de log existe
    Path(DEBUG_LOG).parent.mkdir(parents=True, exist_ok=True)
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if console:
        print(line)


def safe_load_json(path: Path):
    """Le JSON ignorando bytes invalidos e ordena por frame."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").replace("\x00", "")
        data = json.loads(text)
    except Exception as e:
        log(f"[ERRO] Ao ler {path.name}: {e}")
        return []

    if isinstance(data, dict):
        data = list(data.values())
    if not isinstance(data, list):
        return []

    valid = [r for r in data if isinstance(r, dict) and "keypoints" in r and "id_persistente" in r]
    return sorted(valid, key=lambda r: (r.get("id_persistente", 0), r.get("frame", 0)))


def find_json(base_dir: Path, stem: str):
    """Procura JSON correspondente ao video."""
    variants = {stem, stem.replace("_", ""), stem.replace("furto", "furto_"),
                stem.replace("normal", "normal_"), stem.replace("__", "_")}
    for v in variants:
        p = base_dir / f"{v}.json"
        if p.exists():
            return p
    matches = list(base_dir.glob(f"*{stem.replace('_','')}*.json"))
    return matches[0] if matches else None


def parse_scene_clip(stem: str):
    """Extrai numeros de cena e clipe do nome do arquivo."""
    nums = re.findall(r"\d+", stem)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    if len(nums) == 1:
        return 0, int(nums[0])
    return 0, 0


def extract_sequence(records, target_id, interval=None, max_seq_len=60, min_seq_len=5):
    """
    Extrai sequencia de keypoints (C,T,V) para um ID.
    Args:
        interval: (start_frame, end_frame) ou None para video todo.
    """
    frames = []
    
    try:
        id_target = int(target_id)
    except:
        return None

    # Filtra records relevantes
    for r in records:
        rid = r.get("id_persistente", None)
        try:
            rid = int(rid)
        except:
            continue
            
        if rid != id_target:
            continue

        # [NOVO] Filtro Temporal
        frame_idx = r.get("frame", -1)
        if interval:
            start_f, end_f = interval
            if not (start_f <= frame_idx <= end_f):
                continue

        kps = r.get("keypoints", [])
        if not isinstance(kps, list) or len(kps) < 2:
            continue

        # Extrai coordenadas x,y de cada keypoint
        coords = np.zeros((V, C), dtype=np.float32)
        for i, kp in enumerate(kps[:V]):
            if len(kp) >= 2:
                coords[i, 0], coords[i, 1] = float(kp[0]), float(kp[1])
        frames.append(coords)

    total_frames = len(frames)
    if total_frames < min_seq_len:
        return None

    # Normaliza para max_seq_len (padding com ultimo frame ou corte)
    seq = np.zeros((max_seq_len, V, C), dtype=np.float32)
    num_frames = min(total_frames, max_seq_len)

    for t in range(num_frames):
        seq[t, :, :] = frames[t]

    if num_frames < max_seq_len:
        last = frames[-1]
        for t in range(num_frames, max_seq_len):
            seq[t, :, :] = last

    # Transpoe para formato (C, T, V) esperada pelo modelo
    return np.transpose(seq, (2, 0, 1))


def main():
    """Converte JSONs de keypoints para tensor PyTorch."""
    log("[INICIO] Conversao JSON -> PT")
    log(f"[PATH] JSONs: {JSONS_DIR}")
    log(f"[PATH] Labels: {LABELS_PATH}")
    log(f"[PATH] Saida: {OUT_PT}")
    
    if not LABELS_PATH.exists():
        log(f"[ERRO] labels.json nao encontrado: {LABELS_PATH}")
        return
        
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    total_videos = len(labels)

    data, y_labels, metadata, invalid = [], [], [], []
    processed_videos = 0
    total_samples = 0
    
    positive_class = cm.CLASS_NAMES[1].lower()  # ex: "furto"

    for video_stem, id_map in sorted(labels.items()):
        processed_videos += 1
        log(f"\n[{processed_videos}/{total_videos}] {video_stem}")

        json_path = find_json(JSONS_DIR, video_stem)
        if not json_path:
            invalid.append(f"JSON nao encontrado: {video_stem}")
            continue

        records = safe_load_json(json_path)
        if not records:
            invalid.append(f"JSON vazio: {video_stem}")
            continue

        # Itera sobre cada ID anotado
        for pid, label_info in sorted(id_map.items(), key=lambda x: int(x[0])):
            
            # Normaliza Label (String vs Dict)
            if isinstance(label_info, dict):
                # Anotação Complexa (Temporal)
                classe = label_info.get("classe", "NORMAL")
                intervals = label_info.get("intervals", [])
            else:
                # Anotação Simples (Legado)
                classe = str(label_info)
                intervals = []

            # Se for normal ou furto sem intervalo, processa como video inteiro (interval=None)
            # MAS: Se for FURTO e tiver intervalos, gera 1 sample por intervalo.
            # Se for NORMAL, geralmente é o video todo, exceto se eu quiser normalizar trechos? 
            # R: Para normal, pega tudo (None). Para furto complexo, pega intervalos.
            
            is_positive = (classe.lower() == positive_class)
            
            # Lista de tarefas para extração: [(intervalo, sufixo_meta)]
            extraction_tasks = []
            
            if is_positive and intervals:
                # Gera multiplas samples
                for i, inter in enumerate(intervals):
                    extraction_tasks.append((inter, i)) # i é indice do clip
            else:
                # Gera sample única (None = todo video)
                extraction_tasks.append((None, 0))

            # Executa extrações
            for interval, clip_idx in extraction_tasks:
                seq = extract_sequence(records, pid, interval, max_frames, min_frames_validos)
                
                if seq is None:
                    # Se falhou extrair (muito curto), loga aviso apenas se era intervalo explícito
                    if interval:
                        log(f"  [WARN] Intervalo curto ignorado: ID {pid} frames {interval}", console=False)
                    else:
                        invalid.append(f"Sem frames suficientes: {video_stem} (id={pid})")
                    continue

                y = 1 if is_positive else 0
                scene, video_clip_num = parse_scene_clip(video_stem)
                
                data.append(seq)
                y_labels.append(y)
                
                # Metadata: (scene, video_clip, pid, sample_idx)
                # sample_idx diferencia múltiplos trechos do mesmo ID no mesmo video
                metadata.append((scene, video_clip_num, int(pid), clip_idx))
                
                int_str = str(interval) if interval else "FULL"
                log(f"  [OK] ID {pid} ({classe}) [{int_str}] - {seq.shape[1]} frames", console=False)
                total_samples += 1

    if len(data) == 0:
        log("[ERRO] Nenhuma amostra válida gerada!")
        return

    # Salva tensor final
    data_array = np.stack(data, axis=0)
    labels_array = np.array(y_labels)

    data_tensor = torch.from_numpy(data_array).float()
    labels_tensor = torch.from_numpy(labels_array).long()

    # Metadata agora tem 4 colunas: scene, video_clip, pid, sample_idx
    final_data = {"data": data_tensor, "labels": labels_tensor, "metadata": metadata}
    torch.save(final_data, OUT_PT)

    if invalid:
        Path(LOG_FILE).write_text("\n".join(invalid), encoding="utf-8")
        log(f"\n[AVISO] {len(invalid)} inconsistencias em {LOG_FILE}")

    log(f"[OK] Dataset salvo: {OUT_PT}")
    log(f"[OK] Total amostras: {len(data_tensor)}")
    log(f"Conversao concluida ({total_videos} videos originais -> {total_samples} amostras)")


if __name__ == "__main__":
    main()