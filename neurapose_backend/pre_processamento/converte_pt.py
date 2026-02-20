# ==============================================================================
# neurapose_backend/pre_processamento/converte_pt.py
# ==============================================================================

"""
Converte JSONs de keypoints para formato PyTorch (.pt).
Entrada: JSONs do pre-processamento + labels.json
Saida: data.pt com tensores de sequencias normalizadas

[NOVO] Suporta Anotação Temporal:
- Se label for String ("{cm.CLASSE2}"): Pega sequencia inteira.
- Se label for Dict ({{"classe": "{cm.CLASSE2}", "intervals": [[10, 50], [90, 120]]}}): 
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

# Saída: Usar configuração central de Dataset
OUT_PT = cm.TRAINING_DATA_PATH
OUTPUT_BASE = OUT_PT.parent # .../dataset/treino/data

LOG_FILE = OUTPUT_BASE / "frames_invalidos.txt"
DEBUG_LOG = OUTPUT_BASE / "debug_log.txt"

OUT_PT.parent.mkdir(parents=True, exist_ok=True)

# Dimensoes da pose
C, V = cm.NUM_CHANNELS, cm.NUM_JOINTS  # C=2 (x,y), V=17 keypoints
max_frames = cm.TRAIN_SEQUENCE_LENGTH  # T=30 para modelo LSTM/TFT
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
    variants = {stem, stem.replace("_", ""), stem.replace(cm.CLASSE2.lower(), f"{cm.CLASSE2.lower()}_"),
                stem.replace(cm.CLASSE1.lower(), f"{cm.CLASSE1.lower()}_"), stem.replace("__", "_")}
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


def extract_sliding_windows(records, target_id, is_positive, intervals=None, max_seq_len=60, min_seq_len=5):
    """
    Extrai MÚLTIPLAS sequências de keypoints (C,T,V) usando Janela Deslizante.
    Permite capturar o Contexto Normal (Hard Negatives) de um criminoso, e múltiplas 
    amostras de Pessoas Normais inteiras.
    """
    try:
        id_target = int(target_id)
    except:
        return [], []

    # 1. Filtra records apenas deste ID
    id_records = []
    for r in records:
        try:
            rid = int(r.get("id_persistente", -1))
            if rid == id_target:
                kps = r.get("keypoints", [])
                if isinstance(kps, list) and len(kps) >= 2:
                    id_records.append(r)
        except:
            continue

    total_frames = len(id_records)
    if total_frames < min_seq_len:
        return [], []

    # 2. Configura a Janela Temporal e o Stride Físico (Pulo de frames capturados)
    target_duration = getattr(cm, "TEMPORAL_CONTEXT_SECONDS", 5.0)
    fps_target = getattr(cm, "FPS_TARGET", 30.0)
    frames_needed = int(target_duration * fps_target)
    stride = max(1, int(frames_needed / max_seq_len))

    # 3. Slide Step (Avanço da Janela): 50% de sobreposição
    slide_step = max(1, frames_needed // 2)

    sequences = []
    y_labels = []

    for start_idx in range(0, total_frames, slide_step):
        end_idx = min(start_idx + frames_needed, total_frames)
        window_records = id_records[start_idx:end_idx]
        
        # Ignora janelas muito curtas no final do vídeo
        if len(window_records) < (frames_needed * 0.5):
            continue

        # 4. Avalia o Label desta Janela (Y)
        y = 0
        if is_positive and intervals:
            overlap = 0
            for r in window_records:
                f_idx = r.get("frame", -1)
                for inter in intervals:
                    if inter[0] <= f_idx <= inter[1]:
                        overlap += 1
                        break
            
            # Se mais de 50% da janela ocupou a janela de furto real = Furto. 
            # Caso o bandido andou normal (sem overlap), y=0 (Hard Negative!)
            if overlap > (len(window_records) * 0.5):
                y = 1
        elif is_positive and not intervals:
            y = 1
            
        # 5. Extração e Padding
        seq = np.zeros((max_seq_len, V, C), dtype=np.float32)
        window_len = len(window_records)

        for t in range(max_seq_len):
            src_idx = t * stride
            if src_idx < window_len:
                kps = window_records[src_idx].get("keypoints", [])
                for i, kp in enumerate(kps[:V]):
                    if len(kp) >= 2:
                        seq[t, i, 0], seq[t, i, 1] = float(kp[0]), float(kp[1])
            else:
                kps = window_records[-1].get("keypoints", [])
                for i, kp in enumerate(kps[:V]):
                    if len(kp) >= 2:
                        seq[t, i, 0], seq[t, i, 1] = float(kp[0]), float(kp[1])

        # Transpoe para formato (C, T, V) esperado pelo modelo
        seq_trans = np.transpose(seq, (2, 0, 1))
        sequences.append(seq_trans)
        y_labels.append(y)

    return sequences, y_labels


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
    
    positive_class = cm.CLASS_NAMES[1].lower()  # ex: "anomalia"

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
                classe = label_info.get("classe", cm.CLASSE1)
                intervals = label_info.get("intervals", [])
            else:
                # Anotação Simples (Legado)
                classe = str(label_info)
                intervals = []

            is_positive = (classe.lower() == positive_class)
            
            # Executa extração das janelas deslizantes (Hard Negatives embutido)
            seqs, ys = extract_sliding_windows(records, pid, is_positive, intervals, max_frames, min_frames_validos)
            
            if not seqs:
                invalid.append(f"Sem frames suficientes: {video_stem} (id={pid})")
                continue

            scene, video_clip_num = parse_scene_clip(video_stem)
            
            for clip_idx, (seq, y) in enumerate(zip(seqs, ys)):
                data.append(seq)
                y_labels.append(y)
                
                # Metadata: (scene, video_clip, pid, sample_idx)
                metadata.append((scene, video_clip_num, int(pid), clip_idx))
                
                win_class = cm.CLASSE2 if y == 1 else cm.CLASSE1
                log(f"  [OK] ID {pid} ({win_class} window) - {seq.shape[1]} frames", console=False)
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