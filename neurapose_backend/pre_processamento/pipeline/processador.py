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
from neurapose_backend.config_master import FPS_TARGET


# Para integracao com o preview do site
try:
    from neurapose_backend.globals.state import state as state_notifier
except:
    state_notifier = None



def calcular_deslocamento(p_inicial, p_final):
    """Calcula a distancia em pixels entre o ponto inicial e final."""
    p1 = np.array(p_inicial)
    p2 = np.array(p_final)
    return np.linalg.norm(p2 - p1)

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
    fps_out = FPS_TARGET

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

    out_video = preds_dir / f"{video_path.stem}_{int(fps_out)}fps_pose.mp4"
    writer_pred = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps_out,
        (W, H)
    )
    if not writer_pred.isOpened():
        print(Fore.RED + f"[ERRO] Falha ao iniciar VideoWriter (Pred) com codec avc1. Tentando fallback para mp4v...")
        writer_pred = cv2.VideoWriter(
            str(out_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_out,
            (W, H)
        )
        if not writer_pred.isOpened():
             print(Fore.RED + f"[FATAL] Não foi possível criar o arquivo de vídeo de predições: {out_video}")
             sys.exit(1)

    json_path = json_dir / f"{video_path.stem}_{int(fps_out)}fps.json"

    # ================== TIMING: YOLO + BoTSORT ==================
    print(Fore.CYAN + f"[INFO] Iniciando deteccao YOLO + BoTSORT...")
    sys.stdout.flush()
    
    time_yolo_start = time.time()
    res_list = yolo_detector_botsort(videos_dir=norm_path)
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

    # Escreve frames diretamente no VideoWriter (otimizado - sem buffer)

    while True:
        # Verifica se foi solicitada parada
        if state_notifier is not None and state_notifier.stop_requested:
            print(Fore.YELLOW + "[STOP] Processamento interrompido pelo usuário.")
            break
            
        ok, frame = cap.read()
        if not ok:
            break

        regs = results[frame_idx - 1].get("boxes") if frame_idx - 1 < len(results) else None
        
        # Frame para preview (cópia para não modificar original se preview OFF)
        frame_preview = frame.copy() if show else None

        # Checa se ha deteccoes e IDs validos
        if regs is None or len(regs) == 0:
            # Escreve frame direto no vídeo (sem detecções)
            writer_pred.write(frame)
            
            # Stream para preview mesmo sem detecções
            if show and state_notifier is not None:
                state_notifier.set_frame(frame)

            frame_idx += 1
            # Print de progresso a cada 10%
            progress = int((frame_idx / total_frames) * 100)
            if progress >= last_progress + 10:
                print(Fore.CYAN + f"[PROGRESSO] {progress}% ({frame_idx}/{total_frames} frames)")
                sys.stdout.flush()
                last_progress = progress
            continue

        boxes = regs[:, :4]
        confs = regs[:, 5]
        ids = regs[:, 4]
        
        # Frame para desenho final (sempre desenhamos para o vídeo de saída)
        frame_output = frame.copy()

        for box, conf, raw_tid in zip(boxes, confs, ids):

            pid = int(id_map.get(int(raw_tid), int(raw_tid)))

            x1, y1, x2, y2 = map(int, box)

            # Pose
            center, scale = _calc_center_scale(x1, y1, x2, y2)
            trans = get_affine_transform(center, scale, 0, (cm.SIMCC_W, cm.SIMCC_H))
            crop = cv2.warpAffine(frame, trans, (cm.SIMCC_W, cm.SIMCC_H))

            inp = preprocess_rtmpose_input(crop)
            simx, simy = sess.run(None, {input_name: inp})
            coords_in, conf_arr = decode_simcc_output(simx, simy)
            coords_fr = transform_preds(coords_in[0], center, scale, (cm.SIMCC_W, cm.SIMCC_H))

            kps = np.concatenate([coords_fr, conf_arr[0][:, None]], axis=1)
            kps = smoother.step(pid, kps)

            # Salva registro (dados puros)
            registros.append({
                "frame": frame_idx,
                "botsort_id": int(raw_tid),
                "id_persistente": pid,
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "keypoints": kps.tolist()
            })

            # Desenha no frame de saída (sempre para o vídeo final)
            base_color = color_for_id(pid)
            frame_output = desenhar_esqueleto(frame_output, kps, kp_thresh=cm.POSE_CONF_MIN, base_color=base_color)

            cv2.rectangle(frame_output, (x1, y1), (x2, y2), (0,255,0), 2)

            # Texto: ID_P e Confianca (estilo reid-manual)
            label = f"ID_P: {pid} | Pessoa: {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

            cv2.rectangle(frame_output, (x1, y1 - th - 12), (x1 + tw + 10, y1), (255,255,255), -1)
            cv2.putText(frame_output, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            
            # Se preview ativo, desenha também no frame de preview
            # Se preview ativo, desenha também no frame de preview
            if show:
                frame_preview = desenhar_esqueleto(frame_preview, kps, kp_thresh=cm.POSE_CONF_MIN, base_color=base_color)
                cv2.rectangle(frame_preview, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.rectangle(frame_preview, (x1, y1 - th - 12), (x1 + tw + 10, y1), (255,255,255), -1)
                cv2.putText(frame_preview, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        # Escreve frame processado diretamente no vídeo
        writer_pred.write(frame_output)
        
        # Stream para preview em tempo real (se ativado)
        if show and state_notifier is not None:
            state_notifier.set_frame(frame_preview)

        frame_idx += 1
        # Print de progresso a cada 10%
        progress = int((frame_idx / total_frames) * 100)
        if progress >= last_progress + 10:
            print(Fore.CYAN + f"[PROGRESSO] {progress}% ({frame_idx}/{total_frames} frames)")
            sys.stdout.flush()
            last_progress = progress

    time_rtmpose = time.time() - time_rtmpose_start
    
    print(Fore.GREEN + f"[OK] Inferencia concluida: {total_frames} frames processados.")
    sys.stdout.flush()
    cap.release()
    
    # Finaliza o vídeo
    writer_pred.release()
    print(Fore.GREEN + f"[OK] Video salvo: {out_video.name}")

    # ============================================================
    # 1. SALVAR JSON BRUTO (Opcional, mas bom para debug)
    # ============================================================
    with open(json_path, "w") as f:
        json.dump(registros, f, indent=2)

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
        # REGRA A: Duracao (Ignora "fantasmas" rapidos)
        if dados["frames"] < 30:
            print(Fore.YELLOW + f"  - ID {pid} removido (Curta duracao: {dados['frames']} frames)")
            continue
            
        # REGRA B: Deslocamento ACUMULADO (Soma dos passos)
        # Corrige erro onde pessoa vai e volta para mesmo lugar
        caminho = np.array(dados["path"])
        # Calcula distancias entre pontos consecutivos: (P2-P1, P3-P2, ...)
        steps = np.diff(caminho, axis=0) # shape (N-1, 2)
        dist_steps = np.linalg.norm(steps, axis=1) # shape (N-1,)
        distancia_total = np.sum(dist_steps)
        
        # Deslocamento absoluto (Inicio -> Fim) - Apenas informativo ou regra secundaria
        dist_abs = calcular_deslocamento(caminho[0], caminho[-1])
        
        # Se moveu muito pouco no total (somando ida e volta), é estático.
        if distancia_total < 80.0:  # Aumentamos um pouco pois é soma total
            print(Fore.YELLOW + f"  - ID {pid} removido (Estatico: Moveu {distancia_total:.1f}px total / {dist_abs:.1f}px abs)")
            continue

        # REGRA C: Atividade de Pose (Respiração/Ajustes)
        # Usa RTMPose para ver se o esqueleto "vibra" (vivo) ou é rígido (cadeira)
        activity_score = calcular_pose_activity(dados["keypoints"])
        
        if activity_score < cm.MIN_POSE_ACTIVITY:
            print(Fore.YELLOW + f"  - ID {pid} removido (Objeto Inanimado: Activity {activity_score:.2f} < {cm.MIN_POSE_ACTIVITY})")
            continue
            
        ids_validos.append(pid)
        # Debug info dos que ficaram
        print(Fore.BLUE + f"  + ID {pid} MANTIDO (Frames: {dados['frames']} | Dist: {distancia_total:.0f}px | Activity: {activity_score:.2f})")

    print(Fore.GREEN + f"[OK] IDs Finais Mantidos: {ids_validos}")
    sys.stdout.flush()

    # ============================================================
    # 3. SALVAR TRACKING FINAL (Apenas IDs Validos)
    # ============================================================
    # Filtra o mapa de IDs para remover os excluidos
    id_map_limpo = {str(k): int(v) for k, v in id_map.items() if v in ids_validos}

    tracking_analysis = {
        "video": video_path.name,
        "total_frames": frame_idx - 1,
        "id_map": id_map_limpo,
        "tracking_by_frame": {}
    }
    
    # Filtra os registros frame a frame
    for reg in registros:
        # So adiciona se o ID estiver na lista de validos
        if reg["id_persistente"] in ids_validos:
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
