# ==============================================================
# neurapose_backend/nucleo/filtros.py
# ==============================================================
# Módulo centralizado para regras de filtragem e limpeza de IDs.
# Deve ser usado igualmente no TREINO e no APP para garantir consistência.
# ==============================================================

import numpy as np
from colorama import Fore
import neurapose_backend.config_master as cm
from neurapose_backend.nucleo.geometria import calcular_deslocamento, calcular_iou

def filtrar_ids_validos_v6(registros, min_frames=cm.MIN_FRAMES_PER_ID, min_dist=50.0, verbose=False):
    """
    Aplica as regras de limpeza V6 (Inteligente) para decidir quais IDs manter.
    
    Regras:
    A) Duração mínima: O ID deve existir por pelo menos 'min_frames'.
    B) Mobilidade física: O ID deve ter se deslocado pelo menos 'min_dist' pixels do início ao fim.
       (Isso remove objetos estáticos classificados erroneamente como pessoas).
    
    Args:
        registros (list): Lista de dicionários de detecção (deve conter 'id_persistente', 'bbox', etc).
        min_frames (int): Mínimo de frames para considerar válido.
        min_dist (float): Distância mínima de deslocamento (pixels).
        verbose (bool): Se True, imprime motivos da remoção no console.
        
    Returns:
        list: Lista de inteiros contendo apenas os IDs persistentes VÁLIDOS.
    """
    if not registros:
        return []

    # print(Fore.YELLOW + "\n[NUCLEO] Iniciando filtragem unificada de IDs (V6)...")

    # 1. Coletar estatísticas de cada ID
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
                "inicio": centro,
                "fim": centro,
                "keypoints": []
            }
        
        stats_id[pid]["frames"] += 1
        stats_id[pid]["fim"] = centro # Atualiza última posição conhecida
        if kps:
            stats_id[pid]["keypoints"].append(kps)

    # 2. Aplicar regras de exclusão
    ids_validos = []
    
    for pid, dados in stats_id.items():
        rejeitado = False
        motivo = ""

        # REGRA A: Duração
        if dados["frames"] < min_frames:
            rejeitado = True
            motivo = f"Curta duracao ({dados['frames']} < {min_frames} frames)"
            
        # REGRA B: Imobilidade (Somente se passou na regra A)
        if not rejeitado:
            distancia = calcular_deslocamento(dados["inicio"], dados["fim"])
            
            # Se deslocou pouco, verifica se esta "vivo" (mexendo braços/pernas)
            if distancia < min_dist:
                activity_score = _calcular_atividade_pose(dados["keypoints"])
                
                # Se alem de parado geograficamente, tambem esta estatico na pose -> REMOVE
                if activity_score < cm.MIN_MEMBER_ACTIVITY:
                    rejeitado = True
                    motivo = f"Estatico (Desloc. {distancia:.1f}px < {min_dist} e Ativ. {activity_score:.1f} < {cm.MIN_MEMBER_ACTIVITY})"
                else:
                    # Salvou pelo gongo (Pose ativa)
                    if verbose:
                        print(Fore.CYAN + f"[NUCLEO] ID {pid} mantido por atividade de pose ({activity_score:.1f}) apesar de estatico ({distancia:.1f}px)")
                        
        # REGRA C: Atividade de Pose (Evita manequins/estatuas)
        # Calcula a variância média das juntas em relação ao centro da pose.
        if not rejeitado:
             # Ja calculado acima se entrou no if de distancia, senao calcula agora para validar manequins que se movem (ex: arrastados)
             # Mas geralmente manequins sao estaticos. Essa regra C original era meio redundante se B for fraca.
             # Vamos manter a C original apenas como sanity check muito baixo
            activity_score = _calcular_atividade_pose(dados["keypoints"])
            # Se a atividade for muito baixa, é provavel que seja um objeto estatico detectado
            if activity_score < cm.MIN_POSE_ACTIVITY: # Esse eh bem baixo (0.8)
                rejeitado = True
                motivo = f"Inanimado (Atividade {activity_score:.2f} < {cm.MIN_POSE_ACTIVITY})"

        if rejeitado:
            if verbose:
                print(Fore.YELLOW + "[NUCLEO]" + Fore.WHITE + f" ID: {pid} removido: {motivo}")
            continue
            
        # Aprovado
        ids_validos.append(pid)

    if verbose:
        # print(Fore.GREEN + f"[OK] IDs Aprovados: {ids_validos}")
        pass

    return ids_validos


def _calcular_atividade_pose(kps_historico):
    """
    Calcula a variância média das juntas em relação ao centro da pose.
    Isso ajuda a identificar se o 'esqueleto' está vivo (movendo-se internamente)
    ou se é um objeto estático (cadeira/manequim).
    """
    if not kps_historico or len(kps_historico) < 5:
        return 0.0

    # Extrai apenas x,y (ignora confiança)
    # kps_historico é lista de lists ou arrays (N, 17, 3)
    try:
        data = np.array(kps_historico)[:, :, :2] 
    except:
        return 0.0
    
    # Centro da pose em cada frame
    centers = np.mean(data, axis=1, keepdims=True)
    
    # Centraliza
    data_centered = data - centers
    
    # Desvio padrão de cada junta ao longo do tempo (indica o quanto ela treme/move)
    # Na verdade, queremos a variância da pose em si? 
    # A implementação original calculava std over axis=0 (tempo) para ver se as juntas mudam de posição relativa.
    stds = np.std(data_centered, axis=0) 
    
    # Média da magnitude dos desvios
    avg_activity = np.mean(np.linalg.norm(stds, axis=1))
    
    return avg_activity


def filtrar_ghosting_v5(records, iou_thresh=0.8):
    """
    Remove IDs duplicados (Ghosting) que ocupam o mesmo espaço físico no mesmo frame.
    Mantém o ID com maior confiança ou, em caso de empate, o menor ID.
    (Migrado do pré-processamento legado).
    
    Args:
        records: Lista de registros (dicts com 'frame', 'bbox', 'confidence').
        iou_thresh: Limiar de IoU para considerar sobreposição.
        
    Returns:
        list: Lista de registros filtrada.
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
        print(Fore.YELLOW + f"[FILTRO V5] Ghosting: {ids_removidos_count} detecções sobrepostas removidas.")
        
    return records_filtrados
