# ==============================================================
# neurapose_backend/pre_processamento/pipeline/processador.py
# ==============================================================

import sys
import cv2
import json
import time
import numpy as np
from pathlib import Path
from colorama import Fore

from neurapose_backend.detector.yolo_detector import yolo_detector_botsort
import neurapose_backend.config_master as cm

from neurapose_backend.pre_processamento.utils.geometria import (
    _calc_center_scale,
    get_affine_transform,
    transform_preds
)
from neurapose_backend.pre_processamento.utils.visualizacao import desenhar_esqueleto, color_for_id
from neurapose_backend.pre_processamento.modulos.rtmpose import preprocess_rtmpose_input, decode_simcc_output
from neurapose_backend.pre_processamento.modulos.suavizacao import EmaSmoother
from neurapose_backend.pre_processamento.modulos.suavizacao import EmaSmoother
# FPS_TARGET e RTMPOSE_BATCH_SIZE sao acessados via cm.* agora


# Para integracao com o preview do site
try:
    from neurapose_backend.globals.state import state as state_notifier
except:
    state_notifier = None

# ==============================================================
# CARREGAR CONFIGURAÇÕES DO USUÁRIO (Override config_master)
# ==============================================================
# Isso garante que o pipeline offline respeite as configurações da UI.
try:
    from neurapose_backend.app.user_config_manager import UserConfigManager
    user_config = UserConfigManager.load_config()
    for k, v in user_config.items():
        if hasattr(cm, k):
            setattr(cm, k, v)
    print(Fore.BLUE + f"[CONFIG] Configurações do usuário carregadas e aplicadas.")
except Exception as e:
    print(Fore.YELLOW + f"[CONFIG] Falha ao carregar configurações do usuário: {e}")



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
    # Ignora juntas (0,0) se possível, mas aqui vamos simplificar usando todas
    centers = np.mean(data, axis=1, keepdims=True) # (Frames, 1, 2)
    
    # 2. Centralizar pose (remove movimento global/câmera)
    # Agora temos a posicao de cada junta RELATIVA ao centro do corpo naquele frame
    data_centered = data - centers
    
    # 3. Calcular StdDev de cada junta ao longo do tempo (Frames)
    # stds vai ter shape (Juntas, 2)
    stds = np.std(data_centered, axis=0) 
    
    # 4. Média dos desvios (magnitude x+y)
    # stds.sum(axis=1) soma o stdX e stdY de cada junta
    avg_activity = np.mean(np.linalg.norm(stds, axis=1))
    
    return avg_activity


def processar_video(video_path: Path, sess, input_name, out_root: Path, show=False):
    # ------------------ Diretorios -----------------------
    videos_dir = out_root / "videos"
    preds_dir = out_root / "predicoes"
    json_dir = out_root / "jsons"

    videos_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    # ------------------ Normalizar FPS (usa FPS_TARGET do config) ----------------
    cap_in = cv2.VideoCapture(str(video_path))
    W = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_out = cm.FPS_TARGET

    norm_path = videos_dir / f"{video_path.stem}_{int(fps_out)}fps.mp4"

    print(Fore.CYAN + f"[INFO] Normalizando video para {fps_out} FPS...")
    sys.stdout.flush()

    writer_norm = cv2.VideoWriter(
        str(norm_path),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps_out,
        (W, H)
    )
    if not writer_norm.isOpened():
        print(Fore.RED + f"[ERRO] Falha ao iniciar VideoWriter com codec avc1. Tentando fallback para mp4v...")
        writer_norm = cv2.VideoWriter(
            str(norm_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_out,
            (W, H)
        )
        if not writer_norm.isOpened():
             print(Fore.RED + f"[FATAL] Não foi possível criar o arquivo de vídeo: {norm_path}")
             sys.exit(1)

    while True:
        ok, frame = cap_in.read()
        if not ok:
            break
        writer_norm.write(frame)

    cap_in.release()
    writer_norm.release()

    print(Fore.GREEN + f"[OK] Video normalizado salvo: {norm_path.name}")
    sys.stdout.flush()

    # ------------------ Inferencia -----------------------
    cap = cv2.VideoCapture(str(norm_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    
    # Video Writer (Pred) SETUP MOVED TO END OF PIPELINE (After Filtering)
    out_video = preds_dir / f"{video_path.stem}_{int(fps_out)}fps_pose.mp4"

    json_path = json_dir / f"{video_path.stem}_{int(fps_out)}fps.json"

    # ================== TIMING: YOLO + BoTSORT ==================
    print(Fore.CYAN + f"[INFO] Iniciando deteccao YOLO + BoTSORT...")
    sys.stdout.flush()
    
    time_yolo_start = time.time()
    # Passamos batch_size explicitamente para usar o valor do cm atualizado (config user)
    res_list = yolo_detector_botsort(videos_dir=norm_path, batch_size=cm.YOLO_BATCH_SIZE)
    time_yolo = time.time() - time_yolo_start

    print(Fore.GREEN + f"[OK] Deteccao concluida. Processando resultados...")
    sys.stdout.flush()
    
    # Libera memória GPU após YOLO (evita fragmentação para RTMPose)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

    res = res_list[0]
    results = res["results"]
    id_map = res.get("id_map", {})

    registros = []
    smoother = EmaSmoother()

    frame_idx = 1
    last_progress = 0  # Para prints de progresso a cada 10%
    
    # ================== TIMING: RTMPose ==================
    print(Fore.CYAN + f"[INFO] Iniciando inferencia RTMPose em {total_frames} frames...")
    sys.stdout.flush()
    time_rtmpose_start = time.time()

    batch_crops = []
    # (x1, y1, x2, y2, center, scale, pid, conf, raw_tid, frame_idx)
    batch_meta = [] 

    def process_batch_rtmpose(crop_list, meta_list):
        if not crop_list:
            return

        # Stack e Inferencia
        full_inp = np.concatenate(crop_list, axis=0)
        
        # Sessao ONNX
        simx, simy = sess.run(None, {input_name: full_inp})
        
        # Post-Process
        coords_batch, conf_batch = decode_simcc_output(simx, simy)

        # Itera pelo batch para salvar registros
        for i, (x1, y1, x2, y2, c, s, pid, conf, r_tid, fid) in enumerate(meta_list):
            
            # Transform e Suavizacao
            coords_fr = transform_preds(coords_batch[i], c, s, (cm.SIMCC_W, cm.SIMCC_H))
            kps = np.concatenate([coords_fr, conf_batch[i][:, None]], axis=1)
            kps = smoother.step(pid, kps)

            # Salva registro
            registros.append({
                "frame": int(fid),
                "botsort_id": int(r_tid),
                "id_persistente": int(pid),
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "keypoints": kps.tolist()
            })

            # Desenha no frame de saída (recuperar frame original?)
            # O problema do batch no pipeline é que precisamos desenhar no vídeo LINEARMENTE (writer_pred.write).
            # Se processamos fora de ordem ou em blocos, precisamos ter o frame disponível na hora da escrita.
            # SOLUÇÃO: O pipeline original escreve Frame a Frame seguindo o `cap`.
            # Se fizermos batch de inferência, precisamos "segurar" a escrita até ter o resultado.
            # Isso complica pois teríamos que bufferizar FRAMES também.
            pass

    # Para manter a escrita de vídeo síncrona e correta, vamos bufferizar FRAMES também.
    # Buffer de frames para escrita posterior
    video_write_buffer = [] # [(frame_img, frame_idx, detections_in_this_frame)]

    while True:
        if state_notifier is not None and state_notifier.stop_requested:
            print(Fore.YELLOW + "[STOP] Processamento interrompido pelo usuário.")
            break
            
        ok, frame = cap.read()
        if not ok:
            break

        # Dados do YOLO para este frame
        regs = results[frame_idx - 1].get("boxes") if frame_idx - 1 < len(results) else None
        
        # Frame para preview
        frame_preview = frame.copy() if show else None
        frame_output = frame.copy()

        # Se não tem detecção, escrevemos direto (se o buffer estiver vazio) OU adicionamos ao buffer
        # para manter a ordem se já tivermos frames pendentes.
        # Mas para simplificar a lógica de batch: 
        # Vamos processar Frames em blocos de X, coletar todos crops, rodar, e depois escrever os X frames.
        
        # A lógica atual do loop "While True" lê 1 frame.
        # Vamos mudar a estratégia para ler e processar linearmente, mas segurar RTMPose.
        
        # Coleta detecções
        detections_active = False

        if regs is not None and len(regs) > 0:
            boxes = regs[:, :4]
            confs = regs[:, 5]
            ids = regs[:, 4]
            
            detections_active = True
            
            for box, conf, raw_tid in zip(boxes, confs, ids):
                pid = int(id_map.get(int(raw_tid), int(raw_tid)))
                x1, y1, x2, y2 = map(int, box)
                
                # Crop logic
                center, scale = _calc_center_scale(x1, y1, x2, y2)
                trans = get_affine_transform(center, scale, 0, (cm.SIMCC_W, cm.SIMCC_H))
                crop = cv2.warpAffine(frame, trans, (cm.SIMCC_W, cm.SIMCC_H))
                inp = preprocess_rtmpose_input(crop)
                
                batch_crops.append(inp)
                # Guardamos índice do frame para associar depois
                batch_meta.append((x1, y1, x2, y2, center, scale, pid, conf, raw_tid, frame_idx))

        # Adiciona frame ao buffer de escrita (aguardando inferencia)
        # Precisamos saber quais IDs/Boxes pertencem a este frame para desenhar depois.
        video_write_buffer.append({
            "frame": frame_output,
            "preview": frame_preview,
            "frame_idx": frame_idx,
            "regs": regs
        })
        
        frame_idx += 1

        # Check Flush
        if len(batch_crops) >= cm.RTMPOSE_BATCH_SIZE or len(video_write_buffer) > 100:
            # Roda inferência
            process_batch_rtmpose(batch_crops, batch_meta)
                        
            min_f = video_write_buffer[0]["frame_idx"]
            max_f = video_write_buffer[-1]["frame_idx"]
            
            # Map: frame_id -> list of records
            current_batch_records = {} 

            added_count = len(batch_meta)
            recent_recs = registros[-added_count:] if added_count > 0 else []
            
            for r in recent_recs:
                 fid = r["frame"]
                 if fid not in current_batch_records:
                     current_batch_records[fid] = []
                 current_batch_records[fid].append(r)



            # Limpa Buffers
            batch_crops = []
            batch_meta = []
            video_write_buffer = []

            # Progress
            progress = int((frame_idx / total_frames) * 100)
            if progress >= last_progress + 10:
                print(Fore.CYAN + f"[PROGRESSO] {progress}% ({frame_idx}/{total_frames} frames)")
                sys.stdout.flush()
                last_progress = progress

    # Final Flush (Resto do buffer)
    if video_write_buffer:
        process_batch_rtmpose(batch_crops, batch_meta)
        
        added_count = len(batch_meta)
        recent_recs = registros[-added_count:] if added_count > 0 else []
        current_batch_records = {} 
        for r in recent_recs:
             fid = r["frame"]
             if fid not in current_batch_records:
                 current_batch_records[fid] = []
             current_batch_records[fid].append(r)

        # VIDEO WRITING REMOVED FROM HERE
        pass

    time_rtmpose = time.time() - time_rtmpose_start
    
    print(Fore.GREEN + f"[OK] Inferencia concluida: {total_frames} frames processados.")
    sys.stdout.flush()
    cap.release()
    
    # ============================================================
    # 0. FILTRO GHOSTING (V5) - Antes de tudo
    # ============================================================
    registros = filtrar_ghosting(registros, iou_thresh=0.8)

    # ============================================================
    # 2. FILTRAGEM INTELIGENTE (LIMPEZA V6)
    # ============================================================
    print(Fore.CYAN + "\n[INFO] Iniciando limpeza de IDs...")
    sys.stdout.flush()
    
    # Coleta estatisticas de cada ID Persistente
    stats_id = {} 
    for reg in registros:
        pid = reg["id_persistente"]
        bbox = reg["bbox"]
        kps = reg.get("keypoints", [])
        
        # Calcula o centro da caixa (x, y)
        centro = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
        
        if pid not in stats_id:
            stats_id[pid] = {
                "frames": 0,
                "path": [centro], # Historico de posicoes para calculo acumulado
                "keypoints": []   # Historico de poses para analise de atividade
            }
        
        stats_id[pid]["frames"] += 1
        stats_id[pid]["path"].append(centro)
        stats_id[pid]["keypoints"].append(kps)

    # Define quem fica e quem sai
    ids_validos = []
    
    for pid, dados in stats_id.items():
        # REGRA A: Duracao
        if dados["frames"] < 30:
            # print(Fore.YELLOW + f"  - ID {pid} removido (Curta duracao: {dados['frames']} frames)")
            continue
            
        # REGRA B: Deslocamento ACUMULADO
        caminho = np.array(dados["path"])
        steps = np.diff(caminho, axis=0)
        dist_steps = np.linalg.norm(steps, axis=1)
        distancia_total = np.sum(dist_steps)
        
        if distancia_total < 80.0:
            # print(Fore.YELLOW + f"  - ID {pid} removido (Estatico: Moveu {distancia_total:.1f}px total)")
            continue

        # REGRA C: Atividade de Pose
        activity_score = calcular_pose_activity(dados["keypoints"])
        
        if activity_score < cm.MIN_POSE_ACTIVITY:
            # print(Fore.YELLOW + f"  - ID {pid} removido (Inanimado: Activity {activity_score:.2f})")
            continue
            
        ids_validos.append(pid)
        print(Fore.BLUE + f"  + ID {pid} MANTIDO (Frames: {dados['frames']} | Dist: {distancia_total:.0f}px | Activity: {activity_score:.2f})")

    print(Fore.GREEN + f"[OK] IDs Finais Mantidos: {ids_validos}")
    sys.stdout.flush()

    # FILTRAR REGISTROS (Sobrescrevendo lista original com a filtrada)
    registros = [r for r in registros if r["id_persistente"] in ids_validos]

    if not registros:
        print(Fore.RED + "[AVISO] Todos os IDs foram filtrados!")
        # Cria video vazio ou sai? Vamos salvar JSON vazio.
    
    # ============================================================
    # 3. SALVAR JSON (AGORA LIMPO)
    # ============================================================
    with open(json_path, "w") as f:
        json.dump(registros, f, indent=2)
    print(Fore.GREEN + f"[OK] JSON Limpo salvo em: {json_path.name}") 

    # ============================================================
    # 4. GERAR VÍDEO FINAL (2ª Passada - Apenas IDs Limpos)
    # ============================================================
    print(Fore.CYAN + f"[INFO] Gerando video de predicao LIMPO (2a passada)...")
    
    writer_pred = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps_out,
        (W, H)
    )
    if not writer_pred.isOpened():
        writer_pred = cv2.VideoWriter(
            str(out_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_out,
            (W, H)
        )
    
    if writer_pred.isOpened():
        # Reabre video normalizado
        cap_draw = cv2.VideoCapture(str(norm_path))
        
        # Otimizacao: Agrupar registros por frame no dict
        recs_by_frame = {}
        for r in registros:
            fid = r["frame"]
            if fid not in recs_by_frame: recs_by_frame[fid] = []
            recs_by_frame[fid].append(r)
            
        current_frame_idx = 1
        while True:
            ok, frame = cap_draw.read()
            if not ok: break
            
            frame_recs = recs_by_frame.get(current_frame_idx, [])
            
            # Desenha
            for rec in frame_recs:
                kps = np.array(rec["keypoints"])
                x1,y1,x2,y2 = rec["bbox"]
                pid = rec["id_persistente"]
                conf = rec["confidence"]
                base_color = color_for_id(pid)
                
                frame = desenhar_esqueleto(frame, kps, kp_thresh=cm.POSE_CONF_MIN, base_color=base_color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                
                label = f"ID_P: {pid} | Pessoa: {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 10, y1), (255,255,255), -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            
            writer_pred.write(frame)
            current_frame_idx += 1
            
        cap_draw.release()
        writer_pred.release()
        print(Fore.GREEN + f"[OK] Video Limpo salvo: {out_video.name}")
    else:
        print(Fore.RED + f"[ERRO] Falha ao criar video limpo.")

    # ============================================================
    # 5. SALVAR TRACKING FINAL
    # ============================================================
    # Filtra o mapa de IDs para remover os excluidos
    id_map_limpo = {str(k): int(v) for k, v in id_map.items() if v in ids_validos}

    tracking_analysis = {
        "video": video_path.name,
        "total_frames": current_frame_idx - 1,
        "id_map": id_map_limpo,
        "tracking_by_frame": {}
    }
    
    # Filtra os registros frame a frame
    for reg in registros:
        # Aqui, registros JA ESTA FILTRADO, entao nao precisa checar 'in ids_validos'
        f_id = reg["frame"]
        if f_id not in tracking_analysis["tracking_by_frame"]:
            tracking_analysis["tracking_by_frame"][f_id] = []
        
        tracking_analysis["tracking_by_frame"][f_id].append({
            "botsort_id": reg["botsort_id"],
            "id_persistente": reg["id_persistente"],
            "bbox": reg["bbox"],
            "confidence": reg["confidence"]
        })
    
    # Salva o arquivo final limpo
    tracking_path = json_dir / f"{video_path.stem}_{int(fps_out)}fps_tracking.json"
    with open(tracking_path, "w", encoding="utf-8") as f:
        json.dump(tracking_analysis, f, indent=2, ensure_ascii=False)

    # ================== TABELA DE TEMPOS ==================
    time_total = time_yolo + time_rtmpose
    
    print(Fore.CYAN + "\n" + "="*60)
    print(Fore.CYAN + f"  TEMPOS DE PROCESSAMENTO - {video_path.name}")
    print(Fore.CYAN + "="*60)
    print(Fore.WHITE + f"  {'Etapa':<30} {'Tempo':>15}")
    print(Fore.WHITE + "-"*60)
    print(Fore.YELLOW + f"  {'YOLO + BoTSORT + OSNet':<30} {time_yolo:>12.2f} seg")
    print(Fore.YELLOW + f"  {'RTMPose (Inferencia)':<30} {time_rtmpose:>12.2f} seg")
    print(Fore.WHITE + "-"*60)
    print(Fore.GREEN + f"  {'TOTAL':<30} {time_total:>12.2f} seg")
    print(Fore.CYAN + "="*60 + "\n")
    sys.stdout.flush()

    print(Fore.GREEN + f"[OK] Processamento concluido: {video_path.name}")
    sys.stdout.flush()
    
    # Retorna tempos para soma total
    return {"yolo": time_yolo, "rtmpose": time_rtmpose, "total": time_total}
