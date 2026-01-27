# ==============================================================
# neurapose_backend/nucleo/sequencia.py
# ==============================================================
# Módulo centralizado para sequenciamento temporal (Sliding Windows).
# Prepara arrays numpy para entrada em modelos LSTM/TFT.
# ==============================================================

import numpy as np
import neurapose_backend.config_master as cm

def montar_sequencia_individual(records, target_id, max_frames=None, min_frames=5):
    """
    Monta a sequência temporal para UM ID específico (target_id).
    
    Args:
        records (list): Lista de todos os registros de detecção do vídeo.
        target_id (int/str): ID do alvo a ser extraído.
        max_frames (int): Tamanho fixo da janela temporal. Se None, usa cm.TIME_STEPS (30).
        min_frames (int): Mínimo de frames necessários para aceitar a sequência.
        
    Returns:
        np.ndarray: Array (2, T, 17) no formato PyTorch (C, T, V) pronto para inferência.
                    Retorna None se não houver frames suficientes.
    """
    if max_frames is None:
        max_frames = cm.TIME_STEPS  # Garante paridade com o Treino (30)

    frames = []

    # Filtra e ordena frames do ID alvo
    # Assumindo que 'records' pode estar desordenado
    registros_alvo = [r for r in records if int(r.get("id_persistente", r.get("id", -1))) == int(target_id)]
    registros_alvo.sort(key=lambda x: int(x["frame"]))

    for r in registros_alvo:
        # Pega apenas os 17 keypoints e converte para float32
        coords = np.array(
            [[kp[0], kp[1]] for kp in r["keypoints"][:17]], dtype=np.float32
        )
        frames.append(coords)

    if len(frames) < min_frames:
        return None

    # Padding ou Truncating para max_frames
    seq = np.zeros((max_frames, 17, 2), dtype=np.float32)
    num = min(len(frames), max_frames)
    seq[:num] = frames[:num]

    # Transpõe para formato do modelo: (T, V, C) -> (C, T, V)
    # T=Time(frames), V=Vertices(joints), C=Channels(x,y)
    # Entrada esperada pelo LSTM: (Batch, Channels, Time, Vertices) ou similar
    # Aqui retornamos (2, T, 17) para ser empilhado depois em Batch
    seq = np.transpose(seq, (2, 0, 1))
    
    return seq


def montar_sequencia_lote(records, ids_validos, max_frames=None, min_frames=5):
    """
    Gera sequências para múltiplos IDs de uma vez (Processamento O(N)).
    
    Args:
        records (list): Lista de todos os registros.
        ids_validos (list): Lista de IDs a processar.
        max_frames (int): Tamanho da janela temporal.
        min_frames (int): Mínimo de frames para aceitar.
        
    Returns:
        dict: {id: np.ndarray}
    """
    if max_frames is None:
        max_frames = cm.TIME_STEPS

    # 1. Agrupamento O(N) usando Hash Map
    # Em vez de iterar records para CADA id (O(N*IDs)), iteramos uma vez
    grupos = {gid: [] for gid in ids_validos}
    
    for r in records:
        gid = int(r.get("id_persistente", r.get("id", -1)))
        if gid in grupos:
            grupos[gid].append(r)
            
    resultados = {}
    
    # 2. Processamento Individual dos Grupos
    for gid, registros_alvo in grupos.items():
        # Ordena cronologicamente
        registros_alvo.sort(key=lambda x: int(x["frame"]))
        
        frames = []
        for r in registros_alvo:
            coords = np.array(
                [[kp[0], kp[1]] for kp in r["keypoints"][:17]], dtype=np.float32
            )
            frames.append(coords)
            
        if len(frames) < min_frames:
            continue
            
        # Padding/Truncating
        seq = np.zeros((max_frames, 17, 2), dtype=np.float32)
        num = min(len(frames), max_frames)
        seq[:num] = frames[:num]
        
        # (T, V, C) -> (C, T, V)
        seq = np.transpose(seq, (2, 0, 1))
        
        resultados[gid] = seq
        
    return resultados
