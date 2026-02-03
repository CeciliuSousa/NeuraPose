# neurapose_backend/nucleo/sanatizer.py
"""
Módulo de Sanitização - Reatribuição Inteligente + Anti-Teleporte

1. Spatial Locking (Trava Espacial): 
   - Se o ID 3 sair da cadeira, ele vira um "Impostor" (ID 903).
   - Se o ID 89 sentar na cadeira, ele é corrigido para ID 3.
2. Velocity Gating: 
   - Remove saltos impossíveis (teletransporte) baseado no FPS do vídeo.
"""
import math
from collections import defaultdict
from colorama import Fore
import neurapose_backend.config_master as cm

# =============================================================================
# CONFIGURAÇÕES FÍSICAS
# =============================================================================
HIP_CONF_MIN = 0.5
LEFT_HIP_IDX = 11
RIGHT_HIP_IDX = 12

# Define quantos frames equivalem a 1 segundo para a lógica de "Teleporte"
# Se FPS_TARGET=10, 10 frames = 1 segundo.
FRAMES_REF = int(cm.FPS_TARGET) 
DEFAULT_THRESHOLD = 150.0  # Pixels máximos permitidos por movimento (ajustável)

# CONFIGURAÇÃO DE IDENTIDADES ESTÁTICAS (TRAVA ESPACIAL)
# Útil para caixas, recepção ou posições fixas.
STATIC_IDENTITIES = {
    # ID Alvo : { 'raio': pixels, 'frames_ancora': qtd_frames_para_media }
    3: {'raio': 100.0, 'frames_ancora': 60} 
}

# Prefixo para IDs de impostores (Ex: ID 3 vira 903)
ID_IMPOSTOR_START = 900 

def _calcular_centroide(det):
    """
    Calcula centroide priorizando a média dos quadris (Hips) para maior estabilidade.
    Se a confiança dos quadris for baixa, usa o centro da Bounding Box.
    """
    keypoints = det.get("keypoints", [])
    
    # 1. Tenta usar a média dos quadris (mais estável que bbox)
    if len(keypoints) >= 13:
        left_hip = keypoints[LEFT_HIP_IDX]
        right_hip = keypoints[RIGHT_HIP_IDX]
        
        # Verifica se a confiança (index 2) é boa
        # Nota: RTMPose output pode ser [x, y, conf] ou [x, y] dependendo do pós-proc.
        # Assumindo formato [x, y, conf].
        if len(left_hip) > 2 and left_hip[2] > HIP_CONF_MIN and right_hip[2] > HIP_CONF_MIN:
            cx = (left_hip[0] + right_hip[0]) / 2
            cy = (left_hip[1] + right_hip[1]) / 2
            return (cx, cy)
            
    # 2. Fallback: Centro da Bounding Box
    bbox = det.get("bbox", [0, 0, 0, 0])
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return (cx, cy)

def _distancia_euclidiana(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def _calibrar_posicao_fixa(lista_deteccoes, target_id, n_frames):
    """Descobre onde o ID estático 'mora' calculando a média dos primeiros N frames."""
    coords = []
    frames_count = 0
    
    # Filtra apenas o ID alvo e ordena por tempo
    det_alvo = [d for d in lista_deteccoes if d.get("id_persistente") == target_id]
    det_alvo.sort(key=lambda x: x.get("frame", 0))
    
    for det in det_alvo:
        coords.append(_calcular_centroide(det))
        frames_count += 1
        if frames_count >= n_frames:
            break
            
    if not coords:
        return None
        
    avg_x = sum(c[0] for c in coords) / len(coords)
    avg_y = sum(c[1] for c in coords) / len(coords)
    return (avg_x, avg_y)

def sanitizar_dados(lista_deteccoes, threshold=DEFAULT_THRESHOLD):
    """
    Pipeline Mestre de Limpeza:
    1. Trava Espacial: Renomeia impostores e recupera IDs perdidos.
    2. Velocity Gating: Remove frames de teletransporte (erro de tracking).
    """
    if not lista_deteccoes: return []

    # print(Fore.CYAN + f"[SANITIZER] Iniciando limpeza (Ref: {FRAMES_REF} frames/seg)...")

    # =========================================================================
    # ETAPA 1: REATRIBUIÇÃO INTELIGENTE (CORREÇÃO DE CRACHÁS)
    # =========================================================================
    
    # 1. Calibrar onde o ID 3 mora (Posição média inicial)
    posicoes_fixas = {}
    for pid, cfg in STATIC_IDENTITIES.items():
        pos = _calibrar_posicao_fixa(lista_deteccoes, pid, cfg['frames_ancora'])
        if pos:
            posicoes_fixas[pid] = pos
            # print(Fore.YELLOW + f"[LOCK] ID {pid} calibrado na posição {int(pos[0])}x{int(pos[1])} (Raio: {cfg['raio']}px)")

    trocas_resgate = 0
    trocas_impostor = 0
    
    for det in lista_deteccoes:
        pid_original = det.get("id_persistente", 0)
        centro = _calcular_centroide(det)
        
        for static_id, pos_ancora in posicoes_fixas.items():
            raio = STATIC_IDENTITIES[static_id]['raio']
            dist = _distancia_euclidiana(centro, pos_ancora)
            
            # CENÁRIO A: É o ID 3, mas saiu da área restrita (IMPOSTOR)
            if pid_original == static_id and dist > raio:
                det["id_persistente"] = ID_IMPOSTOR_START + static_id
                trocas_impostor += 1
            
            # CENÁRIO B: Não é o ID 3, mas entrou na área restrita (RESGATE)
            elif pid_original != static_id and dist <= raio:
                det["id_persistente"] = static_id
                trocas_resgate += 1

    if trocas_impostor > 0 or trocas_resgate > 0:
        pass
        # print(Fore.GREEN + f"[FIX] Trava Espacial: {trocas_resgate} resgates e {trocas_impostor} impostores renomeados.")

    # =========================================================================
    # ETAPA 2: VELOCITY GATING (ANTI-TELEPORTE)
    # =========================================================================
    
    # 1. Reagrupa os dados (agora contendo IDs corrigidos)
    tracks = defaultdict(list)
    for det in lista_deteccoes:
        pid = det.get("id_persistente", det.get("botsort_id", 0))
        tracks[pid].append(det)

    dados_limpos = []
    frames_removidos = 0
    ids_afetados = set()

    for pid, track in tracks.items():
        if not track: continue
        # Ordenação temporal obrigatória
        track.sort(key=lambda x: x.get("frame", 0))

        valid_track = [track[0]]
        last_valid_center = _calcular_centroide(track[0])
        last_valid_frame = track[0].get("frame", 0)

        for i in range(1, len(track)):
            det = track[i]
            curr_center = _calcular_centroide(det)
            curr_frame = det.get("frame", 0)
            
            dist = _distancia_euclidiana(last_valid_center, curr_center)
            frame_gap = max(1, curr_frame - last_valid_frame)
            
            # Lógica Híbrida: Permite movimento maior se houve gap temporal (oclusão),
            # mas é rígido se o gap for pequeno (teleporte).
            if frame_gap < FRAMES_REF: # Ex: menos de 1 segundo
                limit = threshold
            else:
                limit = threshold * (frame_gap / FRAMES_REF) * 1.5 # Tolerância progressiva
            
            if dist > limit:
                # Teleporte detectado: Ignora este frame
                frames_removidos += 1
                ids_afetados.add(pid)
            else:
                # Movimento válido
                valid_track.append(det)
                last_valid_center = curr_center
                last_valid_frame = curr_frame
        
        dados_limpos.extend(valid_track)

    dados_limpos.sort(key=lambda x: x.get("frame", 0))
    # print(f"[SANITIZER] Limpeza concluída. {frames_removidos} frames removidos em {len(ids_afetados)} IDs.")
    
    return dados_limpos