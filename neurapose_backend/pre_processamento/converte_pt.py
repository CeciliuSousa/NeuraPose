# ==============================================================================
# neurapose_backend/pre_processamento/converte_pt.py
# ==============================================================================

"""
Converte JSONs de keypoints para formato PyTorch (.pt).
Entrada: JSONs do pre-processamento + labels.json
Saida: data.pt com tensores de sequencias normalizadas
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


def extract_sequence(records, target_id, max_frames=60, min_frames=5):
    """Extrai sequencia de keypoints (C,T,V) para um ID."""
    frames = []
    
    try:
        id_target = int(target_id)
    except:
        return None

    for r in records:
        rid = r.get("id_persistente", None)
        try:
            rid = int(rid)
        except:
            continue
            
        if rid != id_target:
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
    if total_frames < min_frames:
        return None

    # Normaliza para max_frames (padding com ultimo frame)
    seq = np.zeros((max_frames, V, C), dtype=np.float32)
    num_frames = min(total_frames, max_frames)

    for t in range(num_frames):
        seq[t, :, :] = frames[t]

    if num_frames < max_frames:
        last = frames[-1]
        for t in range(num_frames, max_frames):
            seq[t, :, :] = last

    # Transpoe para formato (C, T, V)
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
    processed = 0
    positive_class = cm.CLASS_NAMES[1].lower()  # ex: "furto"

    for video_stem, id_map in sorted(labels.items()):
        processed += 1
        log(f"\n[{processed}/{total_videos}] {video_stem}")

        json_path = find_json(JSONS_DIR, video_stem)
        if not json_path:
            invalid.append(f"JSON nao encontrado: {video_stem}")
            continue

        records = safe_load_json(json_path)
        if not records:
            invalid.append(f"JSON vazio: {video_stem}")
            continue

        for pid, label in sorted(id_map.items(), key=lambda x: int(x[0])):
            seq = extract_sequence(records, pid, max_frames, min_frames_validos)
            
            if seq is None:
                invalid.append(f"Sem frames: {video_stem} (id={pid})")
                continue

            y = 1 if label.lower() == positive_class else 0
            scene, clip = parse_scene_clip(video_stem)
            
            data.append(seq)
            y_labels.append(y)
            metadata.append((scene, clip, int(pid), 0))
            log(f"  [OK] ID {pid} ({label}) - {seq.shape[1]} frames", console=False)

    # Salva tensor final
    data_array = np.stack(data, axis=0)
    labels_array = np.array(y_labels)

    data_tensor = torch.from_numpy(data_array).float()
    labels_tensor = torch.from_numpy(labels_array).long()

    final_data = {"data": data_tensor, "labels": labels_tensor, "metadata": metadata}
    torch.save(final_data, OUT_PT)

    if invalid:
        Path(LOG_FILE).write_text("\n".join(invalid), encoding="utf-8")
        log(f"\n[AVISO] {len(invalid)} inconsistencias em {LOG_FILE}")

    log(f"[OK] Dataset salvo: {OUT_PT}")
    log(f"[OK] Total amostras: {len(data_tensor)}")
    log(f"[FIM] Conversao concluida ({total_videos} videos)")


if __name__ == "__main__":
    main()