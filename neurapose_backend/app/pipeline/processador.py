# ==============================================================
# neurapose-backend/app/pipeline/processador.py
# ==============================================================
# Pipeline OTIMIZADO (Teste de Modelo / App Final)
# Refatorado para VISUALIZAÇÃO PADRONIZADA (Regras de UI)
# ==============================================================

import time
import os
import json
import cv2
import numpy as np
from pathlib import Path
from colorama import Fore
from colorama import Fore
from ultralytics import YOLO
import torch

# Silencia logs verbosos do OpenCV/FFmpeg
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import logging
class TaskWarningFilter(logging.Filter):
    def filter(self, record):
        return "Unable to automatically guess model task" not in record.getMessage()

logging.getLogger("ultralytics").addFilter(TaskWarningFilter())

# Importações do projeto
import neurapose_backend.config_master as cm
from neurapose_backend.cuda.gpu_utils import gpu_manager
from neurapose_backend.rtmpose.extracao_pose_rtmpose import ExtratorPoseRTMPose
from neurapose_backend.nucleo.visualizacao import desenhar_esqueleto_unificado, color_for_id
from neurapose_backend.nucleo.tracking_utils import gerar_relatorio_tracking
from neurapose_backend.tracker.rastreador import CustomBoTSORT, CustomDeepOCSORT, save_temp_tracker_yaml
from neurapose_backend.temporal.inferencia_temporal import ClassificadorAcao

from neurapose_backend.temporal.inferencia_temporal import ClassificadorAcao
from neurapose_backend.nucleo.video_utils import normalizar_video

# Import Sanitizer e State
try:
    from neurapose_backend.nucleo.sanatizer import sanitizar_dados
except ImportError:
    sanitizar_dados = None

try:
    from neurapose_backend.globals.state import state
except ImportError:
    state = None

@gpu_manager.inference_mode()
def processar_video(video_path: Path, lstm_model, mu_ignored, sigma_ignored, show_preview=False, output_dir: Path = None, labels_path: Path = None):
    """
    Processa um vídeo usando LOGICAL SKIP + INFERÊNCIA EM TEMPO REAL.
    Renderiza vídeo de saída na mesma taxa de quadros do processamento (10 FPS).
    """

    if not output_dir: raise ValueError("output_dir obrigatório")
    
    # 0. SETUP DIRS
    predicoes_dir = output_dir / "predicoes"
    jsons_dir = output_dir / "jsons"
    anotacoes_dir = output_dir / "anotacoes"
    videos_norm_dir = output_dir / "videos" # [FIX] Re-adicionado
    
    predicoes_dir.mkdir(parents=True, exist_ok=True)
    jsons_dir.mkdir(parents=True, exist_ok=True)
    anotacoes_dir.mkdir(parents=True, exist_ok=True)
    videos_norm_dir.mkdir(parents=True, exist_ok=True) # [FIX] Cria dir
    
    tempos = {"detector_total": 0, "rtmpose_total": 0, "normalizacao": 0}

    # Normalização de FPS (Garanta 30 FPS no input)
    t_start_norm = time.time()
    try:
        # [FIX] Passa output_dir e faz unpacking
        video_norm_path, t_norm_internal = normalizar_video(video_path, videos_norm_dir, target_fps=cm.INPUT_NORM_FPS)
        if video_norm_path is None: raise Exception("Retorno None da normalização")
    except Exception as e:
        print(Fore.RED + f"[ERRO] Falha na normalização: {e}")
        return {}

    # Usamos o tempo retornado pela função ou o medido aqui? 
    # A função normalizar_video retorna o tempo de processamento real (excluindo check de skip)
    tempos["normalizacao"] = t_norm_internal

    # 1. SETUP VIDEO (Lê do Normalizado)
    cap = cv2.VideoCapture(str(video_norm_path))
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define Target FPS
    target_fps = cm.FPS_TARGET # 10.0
    skip_interval = max(1, int(round(original_fps / target_fps)))
    
    print(Fore.CYAN + f"[APP] Vídeo: {video_path.name}")
    print(Fore.WHITE + f"      Input: {original_fps:.2f}fps | Process & Render: {target_fps:.2f}fps")

    # 2. SETUP WRITER (Output 10 FPS)
    # Regra: renderizar na qualidade de frames escolhido (10 frames)
    video_out_name = f"{video_path.stem}_pred.mp4"
    pred_video_path = predicoes_dir / video_out_name
    
    # Tenta Codec AVC1 (OpenH264)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(str(pred_video_path), fourcc, target_fps, (width, height))
    
    if not writer.isOpened():
        print(Fore.YELLOW + "[AVISO] 'avc1' falhou. Fallback para 'mp4v'.")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(pred_video_path), fourcc, target_fps, (width, height))

    # 3. SETUP MODELOS
    pose_extractor = ExtratorPoseRTMPose(device=cm.DEVICE)
    
    # Cérebro
    model_file = cm.MODEL_SAVE_DIR / "model_best.pt"
    if not model_file.exists() and hasattr(cm, 'TRAINED_MODELS_DIR'):
         candidates = list(cm.TRAINED_MODELS_DIR.glob("**/*.pt"))
         if candidates: model_file = candidates[0]
    
    # [FIX] Classificador usa o modelo carregado se disponivel
    brain = ClassificadorAcao(str(model_file), model_instance=lstm_model, window_size=cm.TIME_STEPS)

    # Tracker
    USING_DEEPOCSORT = (cm.TRACKER_NAME.upper() == "DEEPOCSORT")
    tracker = None
    yolo_model = None
    yaml_path = None
    
    if USING_DEEPOCSORT:
        tracker = CustomDeepOCSORT()
    else:
        yolo_model = YOLO(str(cm.YOLO_PATH), task='detect').to(cm.DEVICE)
        tracker_instance = CustomBoTSORT(frame_rate=int(target_fps))
        yolo_model.tracker = tracker_instance
        yaml_path = save_temp_tracker_yaml()

    # 4. LOOP
    frame_idx = 0
    start_time_global = time.time()
    
    registros_totais = [] 
    pred_stats = {} 
    id_final_preds = {} 
    last_logged_percent = -1 # [FIX] Inicialização correta 

    # Atualiza Device
    gpu_manager.update_device(cm.DEVICE)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # LOGICAL SKIP: Só processa e grava os frames do intervalo
            # Ex: Frame 0, 3, 6...
            if frame_idx % skip_interval == 0:
                # --- LÓGICA DE LOG (SILENCIOSA: A CADA 20%) ---
                # Calcula porcentagem atual
                current_percent = int((frame_idx / total_frames_in) * 100)
                
                # Regra: Loga no inicio (0%), a cada 20%, e no final
                # Também loga SEMPRE que houver um ALERTA DE FURTO (prioridade máxima) via override na detecção
                should_log = (current_percent % 20 == 0 and current_percent > last_logged_percent) or (frame_idx == 0)
                
                if should_log:
                    last_logged_percent = current_percent
                    # Este print vai para o WebSocket via LogBuffer (que o WebService lê)
                    print(f"\r[APP] Progresso: {current_percent}% ({frame_idx}/{total_frames_in})")
                    
                t0 = time.time()
                
                # --- DETECÇÃO ---
                yolo_dets = None
                if USING_DEEPOCSORT:
                    tracks = tracker.track(frame)
                    yolo_dets = tracks
                else:
                    # BoTSORT Manual (Robust Fix)
                    res = yolo_model.predict(
                        source=frame,
                        imgsz=cm.YOLO_IMGSZ,
                        conf=cm.DETECTION_CONF,
                        device=cm.DEVICE,
                        classes=[cm.YOLO_CLASS_PERSON],
                        verbose=False,
                        stream=False
                    )
                    
                    dets = np.empty((0, 6))
                    if len(res) > 0 and len(res[0].boxes) > 0:
                        dets = res[0].boxes.data.cpu().numpy()
                        # Normaliza shapes se necessario (igual yolo_stream)
                        if dets.shape[1] == 4: # [x,y,x,y]
                             r = dets.shape[0]
                             dets = np.hstack((dets, np.full((r, 1), 0.85), np.zeros((r, 1))))
                        elif dets.shape[1] == 5:
                             dets = np.hstack((dets, np.zeros((dets.shape[0], 1))))
                    
                    tracks = tracker_instance.update(dets, frame)
                    yolo_dets = tracks

                t1 = time.time()
                tempos["detector_total"] += (t1 - t0)

                # --- POSE ---
                pose_records, _ = pose_extractor.processar_frame(
                    frame_img=frame,
                    detections_yolo=yolo_dets,
                    frame_idx=frame_idx,
                    desenhar_no_frame=False
                )
                
                t2 = time.time()
                tempos["rtmpose_total"] += (t2 - t1)
                
                # --- CLASSIFICAÇÃO (CÉREBRO) ---
                t3_start = time.time()
                for rec in pose_records:
                    pid = rec["id_persistente"]
                    kps = rec["keypoints"]
                    
                    # O Cérebro processa e diz a probabilidade
                    prob = brain.predict_single(pid, kps)
                    rec['theft_prob'] = round(prob, 2) # [OTIMIZAÇÃO] Arredonda aqui na fonte
                    rec['is_theft'] = prob >= cm.CLASSE2_THRESHOLD
                    
                    if pid not in pred_stats: pred_stats[pid] = 0.0
                    pred_stats[pid] = max(pred_stats[pid], prob)
                    
                    if rec['is_theft']:
                        id_final_preds[pid] = 1
                        # ALERTA DE FURTO: Fura o bloqueio de logs!
                        print(Fore.RED + f"[ALERTA 🚨] Frame {frame_idx}: ID {pid} - FURTO DETECTADO ({prob:.1%})")
                    elif pid not in id_final_preds:
                        id_final_preds[pid] = 0

                t3_end = time.time()
                tempos["temporal_total"] += (t3_end - t3_start)
                
                registros_totais.extend(pose_records)
                
                # --- RENDERIZAÇÃO PADRONIZADA (APP) ---
                viz_frame = frame.copy()

                for rec in pose_records:
                    pid = rec["id_persistente"]
                    bbox = rec["bbox"]
                    kps = np.array(rec["keypoints"])
                    conf = rec["confidence"]
                    prob = rec.get("theft_prob", 0.0)
                    is_theft = rec.get("is_theft", False)
                    
                    # 1. Esqueletos Coloridos (Random ID Color)
                    pid_color = color_for_id(pid)
                    desenhar_esqueleto_unificado(viz_frame, kps, kp_thresh=cm.POSE_CONF_MIN, base_color=pid_color)
                    
                    # 2. Cor da BBox (Verde=Normal, Vermelho=Classe2)
                    color = (0, 0, 255) if is_theft else (0, 255, 0)
                    
                    if bbox is not None:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # 3. Label de 2 Linhas (Fundo Branco, Texto Preto)
                        class_name = cm.CLASSE2 if is_theft else cm.CLASSE1
                        line1 = f"ID: {pid} | Pessoa: {conf:.2f}"
                        line2 = f"Classe: {class_name} ({prob:.1%})"
                        
                        font_scale = 0.6
                        thick = 2
                        
                        (w1, h1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
                        (w2, h2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
                        
                        w_box = max(w1, w2) + 10
                        h_box = h1 + h2 + 20
                        
                        # Retângulo Branco Cheio
                        cv2.rectangle(viz_frame, (x1, y1 - h_box), (x1 + w_box, y1), (255, 255, 255), -1)
                        
                        # Texto Preto
                        cv2.putText(viz_frame, line1, (x1 + 5, y1 - h2 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick)
                        cv2.putText(viz_frame, line2, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick)
                
                # Grava frame (10 FPS)
                writer.write(viz_frame)
                
                # Atualização de Preview para o App Web (Gatekeeper Otimizado)
                if show_preview and state is not None and state.show_preview:
                    state.update_frame(viz_frame)
            
            # Se frame não é múltiplo do skip, ignoramos completamente (não grava)
            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[STOP] Interrompido pelo usuário.")

    finally:
        cap.release()
        writer.release()
        if not USING_DEEPOCSORT and yolo_model: del yolo_model.tracker
        gpu_manager.clear_cache()

    total_time = time.time() - start_time_global
    tempos["video_total"] = total_time
    tempos["total"] = total_time
    tempos["yolo"] = tempos["detector_total"]
    tempos["rtmpose"] = tempos["rtmpose_total"]

    print(Fore.GREEN + f"\n[CONCLUIDO] Processamento Finalizado em {total_time:.2f}s")
    
    # RELATORIO DE TEMPOS (Formato Solicitado)
    # Tempos acumulados durante o loop
    t_norm = tempos["normalizacao"] # Pode ser 0.0 caso não tenha passo de normalização física
    t_yolo = tempos["detector_total"]
    t_pose = tempos["rtmpose_total"]
    t_temp = tempos["temporal_total"]
    
    # Recalcula total com base nos componentes para consistência
    calc_total = t_norm + t_yolo + t_pose + t_temp
    
    model_name = cm.TEMPORAL_MODEL.upper() if getattr(cm, 'TEMPORAL_MODEL', None) else "TEMPORAL MODEL"
    
    print("\n" + "="*60)
    print(f"{'Normalização video 10 FPS':<45} {t_norm:>10.2f} seg")
    print(f"{f'YOLO + {cm.TRACKER_NAME} + OSNet':<45} {t_yolo:>10.2f} seg")
    print(f"{'RTMPose':<45} {t_pose:>10.2f} seg")
    print(f"{model_name:<45} {t_temp:>10.2f} seg")
    print("-" * 60)
    print(f"{'TOTAL':<45} {calc_total:>10.2f} seg")
    print("="*60 + "\n")

    # POS-PROC
    if sanitizar_dados:
        # print(Fore.CYAN + "[INFO] Aplicando Sanitização Final...")
        registros_totais = sanitizar_dados(registros_totais, threshold=150.0)

    # --- SALVA JSON DE POSE PADRÃO ---
    # Formato: <nome do video>_pose.json
    json_pose_name = f"{video_path.stem}_pose.json"
    json_pose_path = jsons_dir / json_pose_name
    
    tracker_key = "deepocsort_id" if USING_DEEPOCSORT else "botsort_id"
    
    records_final = []
    
    for r in registros_totais:
        # Copia e enriquece
        new_r = r.copy()
        new_r[tracker_key] = r["id_persistente"]
        records_final.append(new_r)

    with open(json_pose_path, "w", encoding="utf-8") as f:
        json.dump(records_final, f, indent=2, ensure_ascii=False) # Legível conforme solicitado
        # json.dump(records_final, f, indent=None, separators=(',', ':')) # Minificado

    # --- SALVA JSON DE TRACKING REPORT ---
    # Formato: <nome do video>_tracking.json
    json_tracking_path = jsons_dir / f"{video_path.stem}_tracking.json"
    
    tracking_by_frame = {}
    ids_encontrados = set()
    
    for r in records_final:
        f_idx = str(r["frame"])
        pid = r["id_persistente"]
        ids_encontrados.add(pid)
        
        if f_idx not in tracking_by_frame:
            tracking_by_frame[f_idx] = []
        
        # Objeto de tracking rico (com predição)
        is_theft = r.get("is_theft", False)
        classe_id = 1 if is_theft else 0
        classe_nome = cm.CLASSE2 if is_theft else cm.CLASSE1
        
        track_obj = {
            tracker_key: pid,
            "id_persistente": pid,
            "bbox": r["bbox"],
            "confidence": r["confidence"],
            "classe_id": classe_id,
            "classe_predita": classe_nome
        }
        tracking_by_frame[f_idx].append(track_obj)
    
    id_map = {str(i): i for i in sorted(list(ids_encontrados))}
    
    tracking_report = {
        "video": video_path.name,
        "total_frames": total_frames_in,
        "id_map": id_map,
        "tracking_by_frame": tracking_by_frame
    }
    
    with open(json_tracking_path, "w", encoding="utf-8") as f:
        json.dump(tracking_report, f, indent=2, ensure_ascii=False)
    
    # Return Stats
    ids_predicoes = []
    for pid, pred_cls in id_final_preds.items():
        score = pred_stats.get(pid, 0.0)
        ids_predicoes.append({
            "id": int(pid),
            "classe_id": int(pred_cls),
            f"score_{cm.CLASSE2}": float(score)
        })

    video_pred = 1 if any(v == 1 for v in id_final_preds.values()) else 0
    video_score = max(pred_stats.values()) if pred_stats else 0.0
    
    return {
        "video": str(video_path),
        "pred": int(video_pred),
        f"score_{cm.CLASSE2}": float(video_score),
        "tempos": tempos,
        "ids_predicoes": ids_predicoes
    }