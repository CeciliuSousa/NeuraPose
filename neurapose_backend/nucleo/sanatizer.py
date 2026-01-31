# neurapose_backend/nucleo/sanatizer.py
"""
Módulo de Sanitização de Dados - Anti-Teleporte (Velocity Gating)

Remove frames onde o ID "saltou" fisicamente para outra pessoa,
baseando-se na velocidade máxima permitida do movimento dos quadris.
"""
import math
from collections import defaultdict

# =============================================================================
# CONFIGURAÇÕES DE FÍSICA
# =============================================================================
HIP_CONF_MIN = 0.5
LEFT_HIP_IDX = 11
RIGHT_HIP_IDX = 12
DEFAULT_THRESHOLD = 150.0  # Pixels máximos permitidos por movimento


def _calcular_centroide(det):
    """Calcula centroide priorizando quadris (Hips) para maior estabilidade."""
    keypoints = det.get("keypoints", [])
    
    # Tenta usar a média dos quadris
    if len(keypoints) >= 13:
        left_hip = keypoints[LEFT_HIP_IDX]
        right_hip = keypoints[RIGHT_HIP_IDX]
        # Verifica confiança (index 2)
        if left_hip[2] > HIP_CONF_MIN and right_hip[2] > HIP_CONF_MIN:
            cx = (left_hip[0] + right_hip[0]) / 2
            cy = (left_hip[1] + right_hip[1]) / 2
            return (cx, cy)
            
    # Fallback: Centro da Bounding Box
    bbox = det.get("bbox", [0, 0, 0, 0])
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return (cx, cy)


def _distancia_euclidiana(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def sanitizar_dados(lista_deteccoes, threshold=DEFAULT_THRESHOLD):
    """
    Filtra frames de teletransporte (erros de tracking).
    
    Args:
        lista_deteccoes (list): Lista plana de dicts com 'frame', 'id_persistente', 'bbox', etc.
        threshold (float): Limite de pixels para movimento válido.
        
    Returns:
        list: Nova lista contendo APENAS os frames válidos.
    """
    print(f"[SANITIZER] Iniciando limpeza Anti-Teleporte (Threshold: {threshold})...")
    
    if not lista_deteccoes:
        return []

    # 1. Agrupar por ID
    tracks = defaultdict(list)
    for det in lista_deteccoes:
        # Tenta pegar id_persistente, senão botsort_id
        pid = det.get("id_persistente", det.get("botsort_id", 0))
        tracks[pid].append(det)

    dados_limpos = []
    frames_removidos = 0
    ids_afetados = set()

    # 2. Processar cada Track
    for pid, track in tracks.items():
        if not track: continue
        
        # ORDENAÇÃO CRÍTICA: Garante que estamos analisando a linha do tempo correta
        track.sort(key=lambda x: x.get("frame", 0))

        # A "Âncora" inicial é o primeiro frame do vídeo para este ID
        valid_track = [track[0]]
        last_valid_center = _calcular_centroide(track[0])
        last_valid_frame = track[0].get("frame", 0)

        for i in range(1, len(track)):
            det = track[i]
            curr_center = _calcular_centroide(det)
            curr_frame = det.get("frame", 0)
            
            # Cálculos
            dist = _distancia_euclidiana(last_valid_center, curr_center)
            frame_gap = max(1, curr_frame - last_valid_frame)
            
            # --- LÓGICA HÍBRIDA (STRICT MODE) ---
            # Se o gap for curto (< 1s), não permitimos que o threshold cresça.
            # Isso barra o erro onde o ID pula 200px em 3 frames.
            if frame_gap < 30:
                limit = threshold
            else:
                # Se sumiu por muito tempo, permitimos movimento maior (adaptativo)
                limit = threshold * frame_gap
            
            if dist > limit:
                # TELETRANSPORTE DETECTADO (LIXO)
                # Ação: Ignorar este frame. 
                # A âncora (last_valid_center) NÃO muda. O ID espera a pessoa voltar.
                frames_removidos += 1
                ids_afetados.add(pid)
            else:
                # MOVIMENTO VÁLIDO
                valid_track.append(det)
                last_valid_center = curr_center
                last_valid_frame = curr_frame
        
        dados_limpos.extend(valid_track)

    # Reordenar a lista final por frame para salvar no JSON corretamente
    dados_limpos.sort(key=lambda x: x.get("frame", 0))
    
    print(f"[SANITIZER] Limpeza concluída. {frames_removidos} frames removidos em {len(ids_afetados)} IDs.")
    return dados_limpos