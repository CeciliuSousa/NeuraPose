# ==============================================================
# neurapose_backend/nucleo/visualizacao.py
# ==============================================================
# Módulo centralizado para visualização gráfica (CV2)
# Responsável por desenhar esqueletos, boxes e labels de forma padronizada.
# ==============================================================

import cv2
import numpy as np
import hashlib
import neurapose_backend.config_master as cm

def _hash_to_color(i: int):
    """Gera uma cor consistente baseada no hash do ID."""
    h = int(hashlib.md5(str(i).encode()).hexdigest(), 16)
    hue = h % 180
    sat = 200 + (h // 180) % 55
    val = 200 + (h // (180 * 55)) % 55
    hsv = np.uint8([[[hue, sat, val]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    return tuple(map(int, bgr))


def color_for_id(global_id: int):
    """Retorna cor (B,G,R) para um dado ID global."""
    return _hash_to_color(global_id)


def desenhar_esqueleto_unificado(frame, keypoints, kp_thresh=cm.POSE_CONF_MIN, base_color=(0, 255, 0), edge_color=None):
    """
    Desenha o esqueleto (keypoints e conexões) no frame.
    
    Args:
        frame: Imagem BGR (numpy array).
        keypoints: Array (K, 3) -> [[x, y, conf], ...]
        kp_thresh: Confiança mínima para desenhar o ponto.
        base_color: Cor base (B, G, R) para as juntas.
        edge_color: Cor das conexões (se None, usa a base um pouco mais escura).
        
    Returns:
        frame: Imagem com desenhos.
    """
    if edge_color is None:
        edge_color = tuple(int(c * 0.6) for c in base_color)

    # Desenha conexões (limbs)
    for a, b in cm.PAIRS:
        # Verifica bounds para evitar crash se keypoints for menor que o esperado (ex: COCO 17)
        if a < len(keypoints) and b < len(keypoints):
            if keypoints[a][2] >= kp_thresh and keypoints[b][2] >= kp_thresh:
                pt1 = (int(keypoints[a][0]), int(keypoints[a][1]))
                pt2 = (int(keypoints[b][0]), int(keypoints[b][1]))
                cv2.line(frame, pt1, pt2, edge_color, 1, lineType=cv2.LINE_AA)

    # Desenha pontos (joints)
    for i, (x, y, conf) in enumerate(keypoints):
        if conf >= kp_thresh:
            cv2.circle(frame, (int(x), int(y)), 2, base_color, -1, lineType=cv2.LINE_AA)

    return frame


def desenhar_info_predicao_padrao(frame, bbox, pid, conf, pred_name=None, classe_id=0):
    """
    Desenha bounding box e labels informativos no frame.
    Usado tanto para visualização simples quanto para vídeo final classificado.
    
    Args:
        frame: Imagem BGR.
        bbox: [x1, y1, x2, y2].
        pid: ID persistente/global.
        conf: Confiança da detecção de pessoa.
        pred_name: (Opcional) Nome da classe predita (ex: FURTO).
        classe_id: (Opcional) 0 = Normal (Verde), 1 = Anômalo (Vermelho).
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Cores dinâmicas baseadas na classe
    if classe_id == 1:    
        cor_bbox = (0, 0, 255)       # Vermelho (Anomalia)
        cor_label = (0, 0, 255)      
    else:                
        cor_bbox = (0, 255, 0)       # Verde (Normal)
        cor_label = (0, 255, 0)      

    cv2.rectangle(frame, (x1, y1), (x2, y2), cor_bbox, 2)

    # Label Linha 1: ID e Confiança
    label1 = f"ID: {pid} | Conf: {conf:.2f}"
    
    # Desenha fundo linha 1
    (tw1, th1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, max(0, y1 - th1 - 10)), (x1 + tw1 + 6, y1), (255, 255, 255), -1)
    cv2.putText(frame, label1, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # Label Linha 2 (Opcional): Classe Predita
    if pred_name:
        label2 = f"Classe: {pred_name}"
        (tw2, th2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        
        cv2.rectangle(frame, (x1, y1), (x1 + tw2 + 6, y1 + th2 + 10), cor_label, -1)
        cv2.putText(frame, label2, (x1 + 3, y1 + th2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
from pathlib import Path
from tqdm import tqdm

def gerar_video_predicao(
    video_path: Path,
    registros,
    video_out_path: Path,
    show_preview: bool = False,
    preview_callback=None,
    modelo_nome: str = "CLASSE"
):
    """
    Gera vídeo de saída desenhando esqueletos e labels unificados.
    
    Args:
        video_path: Caminho do vídeo original.
        registros: Lista de dicts com dados processados (bbox, keypoints, classes).
        video_out_path: Caminho de saída.
        show_preview: Se True, tenta mostrar preview (via callback ou cv2.imshow se callback for None).
        preview_callback: Função opcional que recebe o frame (ex: para streamar via API).
        modelo_nome: Texto para exibir na legenda.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(video_out_path),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (W, H),
    )
    
    # Fallback codec se avc1 falhar
    if not writer.isOpened():
        print(f"[AVISO] Codec 'avc1' falhou. Tentando 'mp4v'.")
        writer = cv2.VideoWriter(
            str(video_out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H),
        )

    # Agrupa registros por frame para acesso rápido
    registros_por_frame = {}
    for r in registros:
        f_id = int(r["frame"])
        if f_id not in registros_por_frame:
            registros_por_frame[f_id] = []
        registros_por_frame[f_id].append(r)

    pbar = tqdm(total=total_frames, desc=f"Gerando video {video_path.stem}", leave=False)
    frame_idx = 1 

    while True:
        ok, frame = cap.read()
        if not ok: break

        regs = registros_por_frame.get(frame_idx, [])

        for r in regs:
            # Extrai dados robustamente
            bbox = r.get("bbox")
            if bbox is None: continue
            
            kps = np.array(r["keypoints"], dtype=np.float32)
            pid = int(r.get("id_persistente", r.get("botsort_id", 0)))
            conf = float(r.get("confidence", 0.0))
            
            # Dados de classificação (se existirem)
            classe_id = int(r.get("classe_id", 0))
            pred_name = r.get("classe_predita", None)

            # 1. Desenha Esqueleto
            frame = desenhar_esqueleto_unificado(frame, kps, kp_thresh=cm.POSE_CONF_MIN)

            # 2. Desenha Info (Box + Texto)
            frame = desenhar_info_predicao_padrao(
                frame, 
                bbox, 
                pid, 
                conf, 
                pred_name=pred_name, 
                classe_id=classe_id
            )



        writer.write(frame)

        # Preview Callback (Decoupled)
        if show_preview:
            # Integração com API de Streaming (State Global)
            try:
                from neurapose_backend.globals.state import state
                state.set_frame(frame)
            except ImportError:
                pass

            if preview_callback:
                preview_callback(frame)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    # cv2.destroyAllWindows()
