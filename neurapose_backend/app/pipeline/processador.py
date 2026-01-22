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
from neurapose_backend.detector.yolo_detector import yolo_detector_botsort as yolo_detector

# --- Módulos Modulares Unificados ---
from neurapose_backend.rtmpose.extracao_pose_rtmpose import ExtratorPoseRTMPose
from neurapose_backend.nucleo.filtros import filtrar_ids_validos_v6
from neurapose_backend.nucleo.sequencia import montar_sequencia_individual
from neurapose_backend.nucleo.visualizacao import gerar_video_predicao

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
        sess, input_name: (Obsoletos, mantidos por compatibilidade, mas o Extrator usa o seu próprio).
                          IDEALMENTE: ExtratorPoseRTMPose deveria ser injetado ou instanciado uma vez fora.
                          Para refatoração segura, instanciaremos aqui ou usaremos um singleton no futuro.
                          Como 'sess' vem de fora (main.py), vamos ignorá-lo e deixar o Extrator gerenciar sua sessão,
                          ou passar 'sess' se o construtor permitir. O Extrator carrega sua própria sessão hoje.
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
    videos_norm_dir = output_dir / "videos_norm" # Separado para nao poluir
    
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

    # 2. DETECTOR (YOLO + BoTSORT + OSNet)
    # ============================================================
    print(Fore.CYAN + f"[1/4] Executando Detecção e Tracking (YOLO+BoTSORT)...")
    d0 = time.time()
    
    # Executa YOLO no vídeo normalizado
    resultados_list = yolo_detector(videos_dir=norm_path)
    
    d1 = time.time()
    tempos["detector_total"] = d1 - d0

    if not resultados_list:
        return None

    res = resultados_list[0]
    results_yolo = res["results"] # Lista de objetos Results do Ultralytics
    id_map = res.get("id_map", {})

    # As pastas já foram criadas no início

    pred_video_path = predicoes_dir / f"{video_path.stem}_pred.mp4"
    json_path = jsons_dir / f"{video_path.stem}.json"
    
    # Carrega vídeo NORMALIZADO para leitura de frames no passo de Pose
    cap = cv2.VideoCapture(str(norm_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 2. EXTRAÇÃO DE POSE (RTMPose Modular)
    # ============================================================
    print(Fore.CYAN + f"[2/4] Executando Extração de Pose (RTMPose Modular)...")
    p0 = time.time()
    
    records = []
    frame_idx = 1
    
    # Loop frame a frame para extração de pose
    while True:
        ok, frame = cap.read()
        if not ok: break
        
        # Pega as detecções correspondentes a este frame
        if frame_idx <= len(results_yolo):
            dets = results_yolo[frame_idx-1].boxes
        else:
            dets = None
            
        # Processa frame com o Extrator Unificado
        # Ele já faz crop, inferência, transforma coords e suaviza (EMA)
        frame_regs, _ = pose_extractor.processar_frame(
            frame_img=frame,
            detections_yolo=dets,
            frame_idx=frame_idx,
            id_map=id_map,
            desenhar_no_frame=False
        )
        
        records.extend(frame_regs)
        frame_idx += 1
        
    cap.release()
    p1 = time.time()
    tempos["rtmpose_total"] = p1 - p0

    if not records:
        print(Fore.RED + "[AVISO] Nenhuma pose detectada.")
        return None

    # 3. FILTRAGEM E LIMPEZA (Núcleo Modular V6)
    # ============================================================
    print(Fore.CYAN + f"[3/4] Filtrando IDs (Lógica Unificada V6)...")
    
    ids_validos = filtrar_ids_validos_v6(
        registros=records,
        min_frames=cm.MIN_FRAMES_PER_ID, # 30
        min_dist=50.0,                   # Padrão V6
        verbose=False
    )
    
    # Filtra a lista de registros mantendo apenas os válidos
    records = [r for r in records if r["id_persistente"] in ids_validos]

    if not records:
        print(Fore.RED + "[AVISO] Todos os IDs foram removidos pelo filtro.")
        return None

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

    # 5. GERAÇÃO DE VÍDEO E RELATÓRIOS
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