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
from neurapose_backend.nucleo.sequencia import montar_sequencia_individual, montar_sequencia_lote
from neurapose_backend.nucleo.visualizacao import gerar_video_predicao
from neurapose_backend.nucleo.tracking_utils import gerar_relatorio_tracking
from neurapose_backend.nucleo.pipeline import executar_pipeline_extracao

# Módulo de Inferência LSTM (Específico do APP)
from neurapose_backend.app.modulos.inferencia_lstm import rodar_lstm_uma_sequencia, rodar_lstm_batch


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
    # print(Fore.CYAN + f"[0/4] Normalizando Vídeo...")
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


    # ============================================================
    # 4. CLASSIFICAÇÃO SEQUENCIAL (BATCH OPTIMIZED)
    # ============================================================
    print(Fore.CYAN + f"[INFO] CLASSIFICANDO VÍDEO {video_path.name} (BATCH MODE)...")
    t0_temp = time.time()
    
    id_preds = {}   # id -> classe (0 ou 1)
    id_scores = {}  # id -> score

    # Inicializa todos com 0 por padrão
    for gid in ids_validos:
        id_preds[gid] = 0
        id_scores[gid] = 0.0

    # 1. Monta sequências em Batch (O(N))
    # Retorna {id: seq_np}
    seqs_dict = montar_sequencia_lote(records, ids_validos)
    
    if seqs_dict:
        # 2. Roda Inferência em Batch (GPU Paralela)
        # Retorna {id: class_id}, {id: score}
        batch_preds, batch_scores = rodar_lstm_batch(seqs_dict, model, mu, sigma)
        
        # 3. Processa Resultados
        for gid, raw_pred in batch_preds.items():
            score = batch_scores.get(gid, 0.0)
            
            # Aplica Threshold (se necessário revalidar lógica do modelo)
            # O rodar_lstm_batch retorna o score raw da classe 1.
            classe_id = 1 if score >= cm.CLASSE2_THRESHOLD else 0
            
            id_preds[gid] = classe_id
            id_scores[gid] = score
            
            print(Fore.YELLOW + "[PREDIÇÃO]" + Fore.WHITE + f" ID: {gid}: {cm.CLASSE2 if classe_id == 1 else cm.CLASSE1} ({score:.2f})")
    
    # Logs para IDs sem sequencia válida (Ex: poucos frames) ficam como 0


    t1_temp = time.time()
    tempos["temporal_total"] = t1_temp - t0_temp
    
    print(Fore.GREEN + "[OK]" + Fore.WHITE + " CLASSIFICAÇÃO CONCLUIDA!")
    
    # 4.1 RELATÓRIO DE TEMPOS (Teste de Modelo)
    # ============================================================
    # Tempo Temporal Model Name
    model_disp_name = "Temporal Fusion Transformer" if cm.TEMPORAL_MODEL == "tft" else "LSTM / BiLSTM"
    
    # Calculo Total Simples (Soma) - Exclui renderização
    calculated_total = tempos["normalizacao"] + tempos["detector_total"] + tempos["rtmpose_total"] + tempos["temporal_total"]
    
    print(Fore.WHITE + "="*60)
    print(Fore.WHITE + f"TEMPO DE TESTE DE MODELO - {video_path.name}")
    print(Fore.WHITE + "="*60)
    print(Fore.WHITE + f"{f'Normalização video {int(cm.FPS_TARGET)} FPS':<45} {tempos['normalizacao']:>10.2f} seg")
    print(Fore.WHITE + f"{'YOLO + BoTSORT-Custom + OSNet':<45} {tempos['detector_total']:>10.2f} seg")
    print(Fore.WHITE + f"{'RTMPose':<45} {tempos['rtmpose_total']:>10.2f} seg")
    print(Fore.WHITE + f"{model_disp_name:<45} {tempos['temporal_total']:>10.2f} seg")
    print(Fore.WHITE + "-"*60)
    print(Fore.WHITE + f"{'TOTAL':<45} {calculated_total:>10.2f} seg")
    print(Fore.WHITE + "="*60 + "\n")
    
    # Atualiza 'total' no dicionario de retorno para ser a soma correta
    tempos["total"] = calculated_total

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
    # print(Fore.BLUE + f"[INFO] Gerando Relatórios de Tracking...")
    
    
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
    print(Fore.CYAN + f"[INFO] RENDERIZANDO VÍDEO: {pred_video_path.name}...")
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

    # Imprime Tabela (REMOVIDO - JÁ IMPRESSO ACIMA)
    # Mantendo apenas calculo final de video_total real para debug interno se necessario
    pass

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