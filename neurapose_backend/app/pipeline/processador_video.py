# ==============================================================
# neurapose-backend/app/pipeline/processador_video.py
# ==============================================================

import time
import json
import cv2
from pathlib import Path
from colorama import Fore
import numpy as np

# Importações do projeto
from neurapose_backend.detector.yolo_detector import yolo_detector_botsort as yolo_detector

import neurapose_backend.config_master as cm

from neurapose_backend.app.modulos.extracao_pose import extrair_keypoints_rtmpose_padronizado
from neurapose_backend.app.modulos.processamento_sequencia import montar_sequencia_individual
from neurapose_backend.app.modulos.inferencia_lstm import rodar_lstm_uma_sequencia
from neurapose_backend.app.modulos.tracking import TrackHistory
from neurapose_backend.app.utils.visualizacao import gerar_video_predicao

# Funções auxiliares portadas do pré-processamento
def calcular_deslocamento(p_inicial, p_final):
    """Calcula a distancia em pixels entre o ponto inicial e final."""
    p1 = np.array(p_inicial)
    p2 = np.array(p_final)
    return np.linalg.norm(p2 - p1)


def calcular_iou(boxA, boxB):
    """Calcula Intersection over Union (IoU) entre duas caixas [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def calcular_pose_activity(kps_historico):
    """
    Calcula a variância média das juntas em relação ao centro da pose.
    Isso ajuda a identificar se o 'esqueleto' está vivo (movendo-se internamente)
    ou se é um objeto estático (cadeira/manequim).
    
    Args:
        kps_historico: Lista de keypoints de N frames. Shape esperado aprox: (N, 17, 3) ou similar via lista.
    Returns:
        float: Média do desvio padrão das juntas (pixels).
    """
    if not kps_historico or len(kps_historico) < 5:
        return 0.0

    # Converter para numpy: (Frames, Juntas, 3) -> pegamos so x,y
    # Remove confianca (idx 2) e foca em x,y
    data = np.array(kps_historico)[:, :, :2] 
    
    # 1. Calcular centro de gravidade da pose em cada frame (média das juntas visíveis)
    centers = np.mean(data, axis=1, keepdims=True) # (Frames, 1, 2)
    
    # 2. Centralizar pose (remove movimento global/câmera)
    data_centered = data - centers
    
    # 3. Calcular StdDev de cada junta ao longo do tempo (Frames)
    stds = np.std(data_centered, axis=0) 
    
    # 4. Média dos desvios (magnitude x+y)
    avg_activity = np.mean(np.linalg.norm(stds, axis=1))
    
    return avg_activity


def filtrar_ghosting(records, iou_thresh=0.8):
    """
    Remove IDs duplicados (Ghosting) que ocupam o mesmo espaço físico no mesmo frame.
    Mantém o ID com maior confiança ou, em caso de empate, o menor ID.
    """
    if not records:
        return []

    # Agrupa por frame
    frames_dict = {}
    for r in records:
        fid = r["frame"]
        if fid not in frames_dict:
            frames_dict[fid] = []
        frames_dict[fid].append(r)

    records_filtrados = []
    ids_removidos_count = 0

    for fid, dets in frames_dict.items():
        if len(dets) < 2:
            records_filtrados.extend(dets)
            continue
        
        # Marca para remoção
        removidos_indices = set()
        
        # Compara par a par
        for i in range(len(dets)):
            if i in removidos_indices: continue
            
            for j in range(i + 1, len(dets)):
                if j in removidos_indices: continue
                
                iou = calcular_iou(dets[i]["bbox"], dets[j]["bbox"])
                if iou > iou_thresh:
                    # Sobreposição detectada! Remove o de menor confiança
                    conf_i = dets[i].get("confidence", 0)
                    conf_j = dets[j].get("confidence", 0)
                    
                    if conf_i < conf_j:
                        removidos_indices.add(i)
                    else:
                        removidos_indices.add(j)
                        
        for k, r in enumerate(dets):
            if k not in removidos_indices:
                records_filtrados.append(r)
            else:
                ids_removidos_count += 1
                
    if ids_removidos_count > 0:
        print(Fore.YELLOW + f"[V5] Filtro Anti-Ghosting: {ids_removidos_count} detecções sobrepostas removidas.")
        
    return records_filtrados



def processar_video(video_path: Path, model, mu, sigma, sess, input_name, show_preview=False, output_dir: Path = None):
    """
    Processa um único vídeo do início ao fim.
    Retorna um dicionário com estatísticas e resultados.
    """
    tempos = {
        "detector_total": 0.0,   # YOLO + BoTSORT + OSNet
        "rtmpose_total": 0.0,
        "temporal_total": 0.0,
        "video_total": 0.0,
        "yolo": 0.0,      # Compatibilidade com relatório antigo
        "rtmpose": 0.0,   # Compatibilidade com relatório antigo
        "total": 0.0      # Compatibilidade com relatório antigo
    }
    t0_video = time.time()

    # ---------------- DETECTOR (YOLO + BoTSORT + OSNet) ----------------
    d0 = time.time()

    resultados_list = yolo_detector(videos_dir=video_path)
    d1 = time.time()
    tempos["detector_total"] = d1 - d0

    if not resultados_list:
        return None

    # yolo_detector_botsort retorna lista de dicts
    res = resultados_list[0]
    video_original = Path(res["video"])
    results = res["results"]
    id_map = res.get("id_map", {})

    # output_dir é obrigatório para evitar pastas extras
    if not output_dir:
        raise ValueError("output_dir é obrigatório para processar_video")
    
    predicoes_dir = output_dir / "predicoes"
    jsons_dir = output_dir / "jsons"
    trackings_dir = output_dir / "trackings"
    predicoes_dir.mkdir(parents=True, exist_ok=True)
    jsons_dir.mkdir(parents=True, exist_ok=True)
    trackings_dir.mkdir(parents=True, exist_ok=True)

    pred_video_path = predicoes_dir / f"{video_path.stem}_pred.mp4"
    json_path = jsons_dir / f"{video_path.stem}.json"

    # ---------------- POSE (RTMPose) ----------------
    p0 = time.time()

    records = extrair_keypoints_rtmpose_padronizado(
        video_path=video_original,
        results=results,
        sess=sess,
        input_name=input_name,
        id_map=id_map,
        show_preview=False,  # Preview desabilitado aqui - será feito em gerar_video_predicao após LSTM
        model=None,
        mu=None,
        sigma=None
    )
    p1 = time.time()
    tempos["rtmpose_total"] = p1 - p0

    if not records:
        return None

    # ============================================================
    # 2. FILTRAGEM INTELIGENTE (LIMPEZA V6 - SYNC COM PRE-PROCESSAMENTO)
    # ============================================================
    
    stats_id = {} 
    for reg in records:
        pid = reg["id_persistente"]
        bbox = reg["bbox"]
        
        # Calcula o centro da caixa (x, y)
        centro = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
        
        if pid not in stats_id:
            stats_id[pid] = {
                "frames": 0,
                "inicio": centro,
                "fim": centro
            }
        
        stats_id[pid]["frames"] += 1
        stats_id[pid]["fim"] = centro # Atualiza última posição conhecida

    # Define quem fica e quem sai
    ids_validos = []
    
    for pid, dados in stats_id.items():
        # REGRA A: Duracao (Ignora "fantasmas" rapidos)
        if dados["frames"] < 30:
            # print(Fore.YELLOW + f"  - ID {pid} removido (Curta duracao: {dados['frames']} frames)")
            continue
            
        # REGRA B: Imobilidade (Ignora objetos estaticos)
        # Se moveu menos de 50 pixels no video todo, remove.
        distancia = calcular_deslocamento(dados["inicio"], dados["fim"])
        
        if distancia < 50.0:
            # print(Fore.YELLOW + f"  - ID {pid} removido (Estatico: Moveu {distancia:.1f}px)")
            continue
            
        ids_validos.append(pid)

    # FILTRAR REGISTROS (MANTEM APENAS IDs VALIDOS)
    records = [r for r in records if r["id_persistente"] in ids_validos]

    if not records:
        print(Fore.RED + "[AVISO] Todos os IDs foram filtrados pela limpeza V6 (App).")
        # Podemos retornar None ou deixar gerar arquivos vazios?
        # Para consistência, retornamos None
        return None

    # ---------------- LSTM / BATCH ----------------
    # Descobre todos os IDs presentes no vídeo (JA FILTRADOS)
    ids_presentes = sorted({int(r["id"]) for r in records})

    id_preds = {}   # id -> classe_id (0 CLASSE1, 1 CLASSE2)
    id_scores = {}  # id -> score para CLASSE2 (probabilidade)

    t0_temp = time.time()

    for gid in ids_presentes:
        # Inicialmente, todos são CLASSE1 com score 0.0
        id_preds[gid] = 0
        id_scores[gid] = 0.0

        # Monta sequência para este ID
        seq_np = montar_sequencia_individual(records, target_id=gid, min_frames=5)
        if seq_np is None:
            continue  # sem frames suficientes, mantém CLASSE1

        # Roda o modelo temporal para este ID
        score, pred_raw = rodar_lstm_uma_sequencia(seq_np, model, mu, sigma)

        # Aplica threshold B: ID só vira a CLASSE2 se score >= cm.CLASSE2_THRESHOLD
        if score >= cm.CLASSE2_THRESHOLD:
            classe_id = 1
        else:
            classe_id = 0

        id_preds[gid] = classe_id
        id_scores[gid] = score

    t1_temp = time.time()
    tempos["temporal_total"] = t1_temp - t0_temp

    # ---------------- ATRIBUIR CLASSE AOS REGISTROS ----------------
    for r in records:
        gid = int(r["id"])
        classe_id = int(id_preds.get(gid, 0))
        score_id = float(id_scores.get(gid, 0.0))

        r["classe_id"] = classe_id
        r["classe_predita"] = cm.CLASSE2 if classe_id == 1 else cm.CLASSE1
        r[F"score_{cm.CLASSE2}_id"] = score_id

    # Salvar JSON já com keypoints + classe por ID (FILTRADO)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # ---------------- GERAR VÍDEO FINAL COM OVERLAY DE CLASSE ----------------
    # (Video gerado apenas com IDs validos)
    gerar_video_predicao(
        video_path=video_original,
        registros=records,
        video_out_path=pred_video_path,
        show_preview=show_preview,
    )

    # ---------------- TRACKING REPORT ----------------
    # (Logica movida para usar ids_validos já implicitos em records)
    # ============================================================
    # 3. SALVAR TRACKING FINAL (Apenas IDs Validos)
    # ============================================================
    # Filtra o mapa de IDs para remover os excluidos
    id_map_limpo = {str(k): int(v) for k, v in id_map.items() if v in ids_validos}

    tracking_analysis = {
        "video": video_path.name,
        "total_frames": len(results), # Aproximado pelo numero de dets YOLO
        "id_map": id_map_limpo,
        "tracking_by_frame": {}
    }
    
    # Filtra os registros frame a frame
    for reg in records:
        # records JÁ ESTÁ FILTRADO
        f_id = reg["frame"]
        if f_id not in tracking_analysis["tracking_by_frame"]:
            tracking_analysis["tracking_by_frame"][f_id] = []
        
        tracking_analysis["tracking_by_frame"][f_id].append({
            "botsort_id": reg["botsort_id"],
            "id_persistente": reg["id_persistente"],
            "bbox": reg["bbox"],
            "confidence": reg["confidence"]
        })

    
    # Salva o arquivo final limpo de tracking JSON (Igual ao pre-processamento)
    # Tenta obter FPS do vídeo original, fallback para 30
    try:
        cap_fps = cv2.VideoCapture(str(video_original))
        fps_out = cap_fps.get(cv2.CAP_PROP_FPS) or 30.0
        cap_fps.release()
    except:
        fps_out = 30.0

    tracking_json_path = jsons_dir / f"{video_path.stem}_{int(fps_out)}fps_tracking.json"
    with open(tracking_json_path, "w", encoding="utf-8") as f:
        json.dump(tracking_analysis, f, indent=2, ensure_ascii=False)
    # print(Fore.GREEN + f"[OK] JSON Tracking v6 salvo: {tracking_json_path.name}")

    # ---------------- TRACKING REPORT (LEGADO - TXT) ----------------
    # Gera relatório de tracking (duração de cada ID)
    try:
        cap_fps = cv2.VideoCapture(str(video_original))
        fps = cap_fps.get(cv2.CAP_PROP_FPS) or 30.0
        cap_fps.release()

        track_history = TrackHistory()
        for r in records:
            # r["frame"] é 1-based, mas para tempo podemos usar frame/fps
            t_sec = float(r["frame"]) / fps
            # Usando botsort_id (ID original do tracker)
            track_history.update(r["botsort_id"], t_sec)

        tracking_txt_path = trackings_dir / f"{video_path.stem}_trackings.txt"
        track_history.save_txt(tracking_txt_path)
        # print(Fore.GREEN + f"[OK] Relatório de tracking salvo em: {tracking_txt_path}")
    except Exception as e:
        print(Fore.RED + f"[ERRO] Falha ao gerar relatório de tracking: {e}")

    tempos["video_total"] = time.time() - t0_video
    
    # Preenche chaves de compatibilidade para o relatório
    tempos["yolo"] = tempos["detector_total"]
    tempos["rtmpose"] = tempos["rtmpose_total"]
    tempos["total"] = tempos["video_total"]

    # ================== TABELA DE TEMPOS (Igual pre-processamento) ==================
    print(Fore.CYAN + "\n" + "="*60)
    print(Fore.CYAN + f"  TEMPOS DE PROCESSAMENTO (APP) - {video_path.name}")
    print(Fore.CYAN + "="*60)
    print(Fore.WHITE + f"  {'Etapa':<30} {'Tempo':>15}")
    print(Fore.WHITE + "-"*60)
    print(Fore.YELLOW + f"  {'YOLO + BoTSORT + OSNet':<30} {tempos['detector_total']:>12.2f} seg")
    print(Fore.YELLOW + f"  {'RTMPose (Pose)':<30} {tempos['rtmpose_total']:>12.2f} seg")
    print(Fore.YELLOW + f"  {'Classificação (LSTM/TFT)':<30} {tempos['temporal_total']:>12.2f} seg")
    print(Fore.WHITE + "-"*60)
    print(Fore.GREEN + f"  {'TOTAL VIDEO':<30} {tempos['video_total']:>12.2f} seg")
    print(Fore.CYAN + "="*60 + "\n")

    # Resumo a nível de vídeo para métricas:
    # vídeo é CLASSE2 se pelo menos um ID foi classificado como classe 2
    video_pred = 1 if any(v == 1 for v in id_preds.values()) else 0
    video_score = max(id_scores.values()) if len(id_scores) > 0 else 0.0

    # Detalhamento por ID
    ids_predicoes = []
    for gid in ids_presentes:
        ids_predicoes.append(
            {
                "id": int(gid),
                "classe_id": int(id_preds.get(gid, 0)),
                f"score_{cm.CLASSE2}": float(id_scores.get(gid, 0.0)),
            }
        )

    return {
        "video": str(video_path),
        "pred": int(video_pred),
        f"score_{cm.CLASSE2}": float(video_score),
        "tempos": tempos,
        "ids_predicoes": ids_predicoes,
    }