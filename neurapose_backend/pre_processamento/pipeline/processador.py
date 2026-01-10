# ==============================================================
# pre_processamento/pipeline/processador.py
# ==============================================================
# Pipeline completo de pre-processamento:
# 1. Normalizacao de FPS
# 2. Deteccao (YOLO + BoTSORT)
# 3. Extracao de Pose (RTMPose)
# 4. Geracao de JSON e Video de saida
#
# Paths e configuracoes vem do config_master.py!

import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from colorama import Fore

from ...detector.yolo_detector import yolo_detector_botsort
from ..configuracao.config import (
    SIMCC_W, 
    SIMCC_H, 
    POSE_CONF_MIN,
    FPS_TARGET,
    FRAME_DISPLAY_W,
    FRAME_DISPLAY_H,
)
from ..utils.geometria import (
    _calc_center_scale,
    get_affine_transform,
    transform_preds
)
from ..utils.visualizacao import desenhar_esqueleto, color_for_id
from ..modulos.rtmpose import preprocess_rtmpose_input, decode_simcc_output
from ..modulos.suavizacao import EmaSmoother

def calcular_deslocamento(p_inicial, p_final):
    """Calcula a distância em pixels entre o ponto inicial e final."""
    p1 = np.array(p_inicial)
    p2 = np.array(p_final)
    return np.linalg.norm(p2 - p1)

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

    writer_norm = cv2.VideoWriter(
        str(norm_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_out,
        (W, H)
    )

    while True:
        ok, frame = cap_in.read()
        if not ok:
            break
        writer_norm.write(frame)

    cap_in.release()
    writer_norm.release()

    # ------------------ Inferencia -----------------------
    cap = cv2.VideoCapture(str(norm_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video = preds_dir / f"{video_path.stem}_{int(fps_out)}fps_pose.mp4"
    writer_pred = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_out,
        (W, H)
    )

    json_path = json_dir / f"{video_path.stem}_{int(fps_out)}fps.json"

    # Executa detector
    res_list = yolo_detector_botsort(videos_dir=norm_path)
    res = res_list[0]
    results = res["results"]
    id_map = res.get("id_map", {})

    registros = []
    smoother = EmaSmoother()

    frame_idx = 1
    pbar = tqdm(total=total_frames, desc=f"Inferencia {video_path.stem}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        regs = results[frame_idx - 1].boxes if frame_idx - 1 < len(results) else None

        # Checa se ha deteccoes e IDs validos
        if regs is None or len(regs) == 0 or regs.id is None:
            writer_pred.write(frame)
            if show:
                frame_resized = cv2.resize(frame, (FRAME_DISPLAY_W, FRAME_DISPLAY_H))
                cv2.imshow("Inferencia", frame_resized)
                if cv2.waitKey(1) == ord("q"):
                    break

            frame_idx += 1
            pbar.update(1)
            continue

        boxes = regs.xyxy.cpu().numpy()
        confs = regs.conf.cpu().numpy()
        ids = regs.id.cpu().numpy()

        for box, conf, raw_tid in zip(boxes, confs, ids):

            pid = int(id_map.get(int(raw_tid), int(raw_tid)))

            x1, y1, x2, y2 = map(int, box)

            # Pose
            center, scale = _calc_center_scale(x1, y1, x2, y2)
            trans = get_affine_transform(center, scale, 0, (SIMCC_W, SIMCC_H))
            crop = cv2.warpAffine(frame, trans, (SIMCC_W, SIMCC_H))

            inp = preprocess_rtmpose_input(crop)
            simx, simy = sess.run(None, {input_name: inp})
            coords_in, conf_arr = decode_simcc_output(simx, simy)
            coords_fr = transform_preds(coords_in[0], center, scale, (SIMCC_W, SIMCC_H))

            kps = np.concatenate([coords_fr, conf_arr[0][:, None]], axis=1)
            kps = smoother.step(pid, kps)

            base_color = color_for_id(pid)
            frame = desenhar_esqueleto(frame, kps, kp_thresh=POSE_CONF_MIN, base_color=base_color)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # Texto: ID_P e Confiança (estilo reid-manual)
            label = f"ID_P: {pid} | Pessoa: {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

            cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 10, y1), (255,255,255), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2) 

            registros.append({
                "frame": frame_idx,
                "botsort_id": int(raw_tid),
                "id_persistente": pid,
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "keypoints": kps.tolist()
            })

        writer_pred.write(frame)

        if show:
            frame_resized = cv2.resize(frame, (FRAME_DISPLAY_W, FRAME_DISPLAY_H))
            cv2.imshow("Inferencia", frame_resized)
            if cv2.waitKey(1) == ord("q"):
                break

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer_pred.release()

    # ============================================================
    # 1. SALVAR JSON BRUTO (Opcional, mas bom para debug)
    # ============================================================
    with open(json_path, "w") as f:
        json.dump(registros, f, indent=2)

    # ============================================================
    # 2. FILTRAGEM INTELIGENTE (LIMPEZA V6)
    # ============================================================
    print(Fore.CYAN + "\n[INFO] Iniciando limpeza de IDs...")
    
    # Coleta estatísticas de cada ID Persistente
    stats_id = {} 
    for reg in registros:
        pid = reg["id_persistente"]
        bbox = reg["bbox"]
        # Calcula o centro da caixa (x, y)
        centro = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
        
        if pid not in stats_id:
            stats_id[pid] = {"frames": 0, "inicio": centro, "fim": centro}
        
        stats_id[pid]["frames"] += 1
        stats_id[pid]["fim"] = centro # Atualiza última posição conhecida

    # Define quem fica e quem sai
    ids_validos = []
    
    for pid, dados in stats_id.items():
        # REGRA A: Duração (Ignora "fantasmas" rápidos como o ID 55)
        # Se durou menos de 1 segundo (30 frames), é lixo.
        if dados["frames"] < 30:
            print(Fore.YELLOW + f"  - ID {pid} removido (Curta duração: {dados['frames']} frames)")
            continue
            
        # REGRA B: Imobilidade (Ignora cadeiras fixas como o ID 12)
        # Se moveu menos de 50 pixels no vídeo todo, é lixo.
        distancia = calcular_deslocamento(dados["inicio"], dados["fim"])
        if distancia < 50.0:
            print(Fore.YELLOW + f"  - ID {pid} removido (Estático: moveu apenas {distancia:.1f} px)")
            continue
            
        ids_validos.append(pid)

    print(Fore.GREEN + f"[OK] IDs Mantidos: {ids_validos}")

    # ============================================================
    # 3. SALVAR TRACKING FINAL (Apenas IDs Válidos)
    # ============================================================
    # Filtra o mapa de IDs para remover os excluídos
    id_map_limpo = {str(k): int(v) for k, v in id_map.items() if v in ids_validos}

    tracking_analysis = {
        "video": video_path.name,
        "total_frames": frame_idx - 1,
        "id_map": id_map_limpo,
        "tracking_by_frame": {}
    }
    
    # Filtra os registros frame a frame
    for reg in registros:
        # Só adiciona se o ID estiver na lista de válidos
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

    if show:
        cv2.destroyAllWindows()
