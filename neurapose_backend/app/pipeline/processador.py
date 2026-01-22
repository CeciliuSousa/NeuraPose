# ==============================================================
# neurapose-backend/app/pipeline/processador.py
# ==============================================================
# Pipeline OTIMIZADO e MODULARIZADO (V7)
# Usa módulos centrais 'nucleo' e 'rtmpose' para evitar duplicação.
# ==============================================================

import time
import os
# Silencia logs verbosos do OpenCV/FFmpeg
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
import json
import cv2
import numpy as np
from pathlib import Path
from colorama import Fore

# Importações do projeto
import neurapose_backend.config_master as cm

# --- Módulos Modulares Unificados ---
from neurapose_backend.rtmpose.extracao_pose_rtmpose import ExtratorPoseRTMPose
from neurapose_backend.nucleo.sequencia import montar_sequencia_individual
from neurapose_backend.nucleo.visualizacao import gerar_video_predicao
from neurapose_backend.nucleo.tracking_utils import gerar_relatorio_tracking
from neurapose_backend.nucleo.pipeline_unificado import executar_pipeline_extracao

# Módulo de Inferência LSTM (Específico do APP)
from neurapose_backend.app.modulos.inferencia_lstm import rodar_lstm_uma_sequencia


from neurapose_backend.nucleo.video_utils import normalizar_video

def processar_video(video_path: Path, model, mu, sigma, show_preview=False, output_dir: Path = None):
    """
    Processa um único vídeo do início ao fim usando a arquitetura modularizada.
    
    Args:
        video_path: Caminho do vídeo.
        model: Modelo LSTM carregado.
        mu, sigma: Estatísticas de normalização do LSTM.
    """
    
    # Inicializa Extrator RTMPose
    pose_extractor = ExtratorPoseRTMPose(device=cm.DEVICE)

    tempos = {
        "normalizacao": 0.0,
        "detector_total": 0.0,
        "rtmpose_total": 0.0,
        "temporal_total": 0.0,
        "video_total": 0.0,
        "yolo": 0.0, "rtmpose": 0.0, "total": 0.0 # Compatibilidade
    }
    t0_video = time.time()

    # Preparação de Pastas (Antes da normalização para ter output_dir)
    if not output_dir: raise ValueError("output_dir obrigatório")
    predicoes_dir = output_dir / "predicoes"
    jsons_dir = output_dir / "jsons"
    videos_norm_dir = output_dir / "videos" # Separado para nao poluir
    
    predicoes_dir.mkdir(parents=True, exist_ok=True)
    jsons_dir.mkdir(parents=True, exist_ok=True)
    videos_norm_dir.mkdir(parents=True, exist_ok=True)

    # 1. NORMALIZAÇÃO DE VÍDEO
    # ============================================================
    print(Fore.CYAN + f"[0/4] Normalizando Vídeo...")
    norm_path, t_norm = normalizar_video(video_path, videos_norm_dir)
    tempos["normalizacao"] = t_norm

    if not norm_path:
        return None

    # 2. PIPELINE UNIFICADO (Detecção + Pose + Filtros)
    # ============================================================
    # Substitui toda a lógica manual anterior pela chamada modular
    records, id_map, ids_validos, total_frames, t_extracao = executar_pipeline_extracao(
        video_path_norm=norm_path,
        pose_extractor=pose_extractor,
        batch_size=cm.YOLO_BATCH_SIZE,
        verbose=True
    )
    
    # Atualiza tempos
    tempos["detector_total"] = t_extracao["yolo"]
    tempos["rtmpose_total"] = t_extracao["rtmpose"]

    if not records:
        return None

    pred_video_path = predicoes_dir / f"{video_path.stem}_pred.mp4"
    json_path = jsons_dir / f"{video_path.stem}.json"


    # 4. CLASSIFICAÇÃO SEQUENCIAL (LSTM)
    # ============================================================
    print(Fore.CYAN + f"[4/4] Classificando Comportamentos (LSTM)...")
    t0_temp = time.time()
    
    id_preds = {}   # id -> classe (0 ou 1)
    id_scores = {}  # id -> score
    
    for gid in ids_validos:
        # Padrão: Classe 0 (Normal)
        id_preds[gid] = 0
        id_scores[gid] = 0.0
        
        # Monta sequência usando módulo central (Gararte T=30)
        seq_np = montar_sequencia_individual(records, target_id=gid)
        
        if seq_np is None:
            continue
            
        # Inferência LSTM (Específica do App)
        score, pred_raw = rodar_lstm_uma_sequencia(seq_np, model, mu, sigma)
        
        # Aplica Threshold
        classe_id = 1 if score >= cm.CLASSE2_THRESHOLD else 0
        
        id_preds[gid] = classe_id
        id_scores[gid] = score

    t1_temp = time.time()
    tempos["temporal_total"] = t1_temp - t0_temp

    # Enriquece registros com classificação
    for r in records:
        gid = r["id_persistente"]
        classe_id = id_preds.get(gid, 0)
        score = id_scores.get(gid, 0.0)
        
        r["classe_id"] = classe_id
        r["classe_predita"] = cm.CLASSE2 if classe_id == 1 else cm.CLASSE1
        r[f"score_{cm.CLASSE2}_id"] = score

    # Salva JSON Final
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # 5. GERAÇÃO DE RELATÓRIOS (Tracking JSON)
    # ============================================================
    print(Fore.CYAN + f"[5/5] Gerando Relatórios de Tracking...")
    
    
    trackings_dir = output_dir / "jsons"
    trackings_dir.mkdir(parents=True, exist_ok=True)
    
    tracking_json_path = trackings_dir / f"{video_path.stem}_tracking.json"
    
    gerar_relatorio_tracking(
        registros=records,
        id_map=id_map,
        ids_validos=ids_validos,
        total_frames=total_frames,
        video_name=video_path.name,
        output_path=tracking_json_path
    )
    
    # 6. GERAÇÃO DE VÍDEO FINAL
    # ============================================================
    
    # Vídeo Final (Usa módulo nucleo/visualizacao)
    # IMPORTANTE: Usar o vídeo normalizado para garantir sincronia de frames
    gerar_video_predicao(
        video_path=norm_path,
        registros=records,
        video_out_path=pred_video_path,
        show_preview=show_preview,
        modelo_nome=cm.TEMPORAL_MODEL.upper()
    )
    


    tempos["video_total"] = time.time() - t0_video
    
    # Compatibilidade de chaves
    tempos["yolo"] = tempos["detector_total"]
    tempos["rtmpose"] = tempos["rtmpose_total"]
    tempos["total"] = tempos["video_total"]

    # Imprime Tabela
    print(Fore.CYAN + "\n" + "="*60)
    print(Fore.CYAN + f"  TEMPOS DE PROCESSAMENTO (APP MODULAR) - {video_path.name}")
    print(Fore.CYAN + "="*60)
    print(Fore.YELLOW + f"  {'Normalização':<30} {tempos['normalizacao']:>12.2f} seg")
    print(Fore.YELLOW + f"  {'YOLO + BoTSORT + OSNet':<30} {tempos['detector_total']:>12.2f} seg")
    print(Fore.YELLOW + f"  {'RTMPose':<30} {tempos['rtmpose_total']:>12.2f} seg")
    print(Fore.YELLOW + f"  {str(cm.TEMPORAL_MODEL).upper():<30} {tempos['temporal_total']:>12.2f} seg")
    print(Fore.WHITE + "-"*60)
    print(Fore.GREEN + f"  {'TOTAL VIDEO':<30} {tempos['video_total']:>12.2f} seg")
    print(Fore.CYAN + "="*60 + "\n")

    # Retorno final para main.py
    video_pred = 1 if any(v == 1 for v in id_preds.values()) else 0
    video_score = max(id_scores.values()) if id_scores else 0.0
    
    ids_predicoes = []
    for gid in ids_validos:
        ids_predicoes.append({
            "id": int(gid),
            "classe_id": int(id_preds.get(gid, 0)),
            f"score_{cm.CLASSE2}": float(id_scores.get(gid, 0.0))
        })

    return {
        "video": str(video_path),
        "pred": int(video_pred),
        f"score_{cm.CLASSE2}": float(video_score),
        "tempos": tempos,
        "ids_predicoes": ids_predicoes
    }