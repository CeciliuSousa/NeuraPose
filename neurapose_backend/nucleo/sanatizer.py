# # neurapose_backend/nucleo/sanatizer.py
# """
# Módulo de Sanitização de Dados - Anti-Teleporte (Velocity Gating)

# Remove frames onde o ID "saltou" fisicamente para outra pessoa,
# baseando-se na velocidade máxima permitida do movimento dos quadris.
# """
# import math
# from collections import defaultdict

# # =============================================================================
# # CONFIGURAÇÕES DE FÍSICA
# # =============================================================================
# HIP_CONF_MIN = 0.5
# LEFT_HIP_IDX = 11
# RIGHT_HIP_IDX = 12
# DEFAULT_THRESHOLD = 150.0  # Pixels máximos permitidos por movimento


# def _calcular_centroide(det):# neurapose_backend/nucleo/sanitizer.py
# """
# Módulo de Sanitização - Reatribuição Inteligente + Anti-Teleporte

# 1. Spatial Locking (Trava Espacial): 
#    - Se o ID 3 sair da cadeira, ele vira um "Impostor" (ID 903).
#    - Se o ID 89 sentar na cadeira, ele é corrigido para ID 3.
# 2. Velocity Gating: 
#    - Remove saltos impossíveis (teletransporte) baseado no FPS do vídeo.
# """
# import math
# from collections import defaultdict
# from colorama import Fore
# import neurapose_backend.config_master as cm

# # =============================================================================
# # CONFIGURAÇÕES
# # =============================================================================
# HIP_CONF_MIN = 0.5
# LEFT_HIP_IDX = 11
# RIGHT_HIP_IDX = 12

# # Define quantos frames equivalem a 1 segundo para a lógica de "Teleporte"
# FRAMES = int(cm.FPS_TARGET) 

# # CONFIGURAÇÃO DE IDENTIDADES ESTÁTICAS (TRAVA ESPACIAL)
# # ID 3: Pessoa sentada. Se sair do raio de 100px, é impostor.
# STATIC_IDENTITIES = {
#     # ID Alvo : { 'raio': pixels, 'frames_ancora': qtd_frames_para_media }
#     3: {'raio': 100.0, 'frames_ancora': 60} 
# }

# # Prefixo para IDs de impostores (Ex: ID 3 vira 903)
# ID_IMPOSTOR_START = 900 

# def _calcular_centroide(det):
#     """Calcula centroide priorizando quadris (Hips)."""
#     keypoints = det.get("keypoints", [])
#     if len(keypoints) >= 13:
#         left_hip = keypoints[LEFT_HIP_IDX]
#         right_hip = keypoints[RIGHT_HIP_IDX]
#         if left_hip[2] > HIP_CONF_MIN and right_hip[2] > HIP_CONF_MIN:
#             return ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
            
#     bbox = det.get("bbox", [0, 0, 0, 0])
#     return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

# def _distancia_euclidiana(p1, p2):
#     return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# def _calibrar_posicao_fixa(lista_deteccoes, target_id, n_frames):
#     """Descobre onde o ID 'mora' (média dos primeiros N frames)."""
#     coords = []
#     frames_count = 0
    
#     # Filtra apenas o ID alvo e ordena
#     det_alvo = [d for d in lista_deteccoes if d.get("id_persistente") == target_id]
#     det_alvo.sort(key=lambda x: x.get("frame", 0))
    
#     for det in det_alvo:
#         coords.append(_calcular_centroide(det))
#         frames_count += 1
#         if frames_count >= n_frames:
#             break
            
#     if not coords:
#         return None
        
#     avg_x = sum(c[0] for c in coords) / len(coords)
#     avg_y = sum(c[1] for c in coords) / len(coords)
#     return (avg_x, avg_y)

# def sanitizar_dados(lista_deteccoes, threshold=150.0):
#     """
#     Pipeline de Limpeza:
#     1. Trava Espacial: Renomeia impostores e recupera IDs perdidos.
#     2. Velocity Gating: Remove frames de teletransporte.
#     """
#     if not lista_deteccoes: return []

#     print(Fore.CYAN + f"[SANITIZER] Iniciando limpeza (Ref: {FRAMES} frames/seg)...")

#     # =========================================================================
#     # ETAPA 1: REATRIBUIÇÃO INTELIGENTE (CORREÇÃO DE CRACHÁS)
#     # =========================================================================
    
#     # 1. Calibrar onde o ID 3 mora (Posição média inicial)
#     posicoes_fixas = {}
#     for pid, cfg in STATIC_IDENTITIES.items():
#         pos = _calibrar_posicao_fixa(lista_deteccoes, pid, cfg['frames_ancora'])
#         if pos:
#             posicoes_fixas[pid] = pos
#             print(Fore.YELLOW + f"[LOCK] ID {pid} calibrado na posição {pos} (Raio: {cfg['raio']}px)")

#     trocas_resgate = 0
#     trocas_impostor = 0
    
#     for det in lista_deteccoes:
#         pid_original = det.get("id_persistente", 0)
#         centro = _calcular_centroide(det)
        
#         for static_id, pos_ancora in posicoes_fixas.items():
#             raio = STATIC_IDENTITIES[static_id]['raio']
#             dist = _distancia_euclidiana(centro, pos_ancora)
            
#             # CENÁRIO A: É o ID 3, mas está longe da cadeira (IMPOSTOR)
#             if pid_original == static_id and dist > raio:
#                 # Renomeia para 900 + ID (Ex: 903). 
#                 # A 4ª pessoa ganha um ID válido e não fica invisível.
#                 det["id_persistente"] = ID_IMPOSTOR_START + static_id
#                 trocas_impostor += 1
            
#             # CENÁRIO B: Não é o ID 3 (ex: 89), mas está na cadeira (ID PERDIDO)
#             elif pid_original != static_id and dist <= raio:
#                 # Recuperamos o crachá: Você É o ID 3!
#                 det["id_persistente"] = static_id
#                 trocas_resgate += 1

#     if trocas_impostor > 0 or trocas_resgate > 0:
#         print(Fore.GREEN + f"[FIX] Trava Espacial: {trocas_resgate} resgates e {trocas_impostor} impostores renomeados.")

#     # =========================================================================
#     # ETAPA 2: VELOCITY GATING (ANTI-TELEPORTE)
#     # =========================================================================
#     # Reagrupa os dados (agora contendo IDs originais e IDs 900+)
#     tracks = defaultdict(list)
#     for det in lista_deteccoes:
#         pid = det.get("id_persistente", det.get("botsort_id", 0))
#         tracks[pid].append(det)

#     dados_limpos = []
#     frames_removidos = 0

#     for pid, track in tracks.items():
#         if not track: continue
#         track.sort(key=lambda x: x.get("frame", 0))

#         valid_track = [track[0]]
#         last_valid_center = _calcular_centroide(track[0])
#         last_valid_frame = track[0].get("frame", 0)

#         for i in range(1, len(track)):
#             det = track[i]
#             curr_center = _calcular_centroide(det)
#             curr_frame = det.get("frame", 0)
            
#             dist = _distancia_euclidiana(last_valid_center, curr_center)
#             frame_gap = max(1, curr_frame - last_valid_frame)
            
#             # Lógica Híbrida usando FRAMES (cm.FPS_TARGET)
#             if frame_gap < FRAMES:
#                 limit = threshold
#             else:
#                 limit = threshold * frame_gap
            
#             if dist > limit:
#                 frames_removidos += 1
#             else:
#                 valid_track.append(det)
#                 last_valid_center = curr_center
#                 last_valid_frame = curr_frame
        
#         dados_limpos.extend(valid_track)

#     dados_limpos.sort(key=lambda x: x.get("frame", 0))
#     return dados_limpos
#     """Calcula centroide priorizando quadris (Hips) para maior estabilidade."""
#     keypoints = det.get("keypoints", [])
    
#     # Tenta usar a média dos quadris
#     if len(keypoints) >= 13:
#         left_hip = keypoints[LEFT_HIP_IDX]
#         right_hip = keypoints[RIGHT_HIP_IDX]
#         # Verifica confiança (index 2)
#         if left_hip[2] > HIP_CONF_MIN and right_hip[2] > HIP_CONF_MIN:
#             cx = (left_hip[0] + right_hip[0]) / 2
#             cy = (left_hip[1] + right_hip[1]) / 2
#             return (cx, cy)
            
#     # Fallback: Centro da Bounding Box
#     bbox = det.get("bbox", [0, 0, 0, 0])
#     cx = (bbox[0] + bbox[2]) / 2
#     cy = (bbox[1] + bbox[3]) / 2
#     return (cx, cy)


# def _distancia_euclidiana(p1, p2):
#     return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


# def sanitizar_dados(lista_deteccoes, threshold=DEFAULT_THRESHOLD):
#     """
#     Filtra frames de teletransporte (erros de tracking).
    
#     Args:
#         lista_deteccoes (list): Lista plana de dicts com 'frame', 'id_persistente', 'bbox', etc.
#         threshold (float): Limite de pixels para movimento válido.
        
#     Returns:
#         list: Nova lista contendo APENAS os frames válidos.
#     """
#     print(f"[SANITIZER] Iniciando limpeza Anti-Teleporte (Threshold: {threshold})...")
    
#     if not lista_deteccoes:
#         return []

#     # 1. Agrupar por ID
#     tracks = defaultdict(list)
#     for det in lista_deteccoes:
#         # Tenta pegar id_persistente, senão botsort_id
#         pid = det.get("id_persistente", det.get("botsort_id", 0))
#         tracks[pid].append(det)

#     dados_limpos = []
#     frames_removidos = 0
#     ids_afetados = set()

#     # 2. Processar cada Track
#     for pid, track in tracks.items():
#         if not track: continue
        
#         # ORDENAÇÃO CRÍTICA: Garante que estamos analisando a linha do tempo correta
#         track.sort(key=lambda x: x.get("frame", 0))

#         # A "Âncora" inicial é o primeiro frame do vídeo para este ID
#         valid_track = [track[0]]
#         last_valid_center = _calcular_centroide(track[0])
#         last_valid_frame = track[0].get("frame", 0)

#         for i in range(1, len(track)):
#             det = track[i]
#             curr_center = _calcular_centroide(det)
#             curr_frame = det.get("frame", 0)
            
#             # Cálculos
#             dist = _distancia_euclidiana(last_valid_center, curr_center)
#             frame_gap = max(1, curr_frame - last_valid_frame)
            
#             # --- LÓGICA HÍBRIDA (STRICT MODE) ---
#             # Se o gap for curto (< 1s), não permitimos que o threshold cresça.
#             # Isso barra o erro onde o ID pula 200px em 3 frames.
#             if frame_gap < 30:
#                 limit = threshold
#             else:
#                 # Se sumiu por muito tempo, permitimos movimento maior (adaptativo)
#                 limit = threshold * frame_gap
            
#             if dist > limit:
#                 # TELETRANSPORTE DETECTADO (LIXO)
#                 # Ação: Ignorar este frame. 
#                 # A âncora (last_valid_center) NÃO muda. O ID espera a pessoa voltar.
#                 frames_removidos += 1
#                 ids_afetados.add(pid)
#             else:
#                 # MOVIMENTO VÁLIDO
#                 valid_track.append(det)
#                 last_valid_center = curr_center
#                 last_valid_frame = curr_frame
        
#         dados_limpos.extend(valid_track)

#     # Reordenar a lista final por frame para salvar no JSON corretamente
#     dados_limpos.sort(key=lambda x: x.get("frame", 0))
    
#     print(f"[SANITIZER] Limpeza concluída. {frames_removidos} frames removidos em {len(ids_afetados)} IDs.")
#     return dados_limpos