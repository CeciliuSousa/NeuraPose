# neurapose/pre_processamento/utils/pose_cleaner.py

import numpy as np
from colorama import Fore

def calcular_similaridade_pose(kpts1, kpts2, bbox_size):
    """
    Calcula a distância média entre keypoints normalizada pelo tamanho da pessoa.
    Quanto MENOR o valor, MAIS PARECIDAS as poses.
    """
    if not kpts1 or not kpts2:
        return float('inf')

    # Converter para numpy se não for
    k1 = np.array(kpts1)
    k2 = np.array(kpts2)

    # Pegar apenas x,y (ignorar score de confiança se houver)
    pts1 = k1[:, :2]
    pts2 = k2[:, :2]
    
    # Calcular distância euclidiana entre cada ponto correspondente
    diff = np.linalg.norm(pts1 - pts2, axis=1)
    
    # Normalizar pelo tamanho da bounding box (diagonal ou altura)
    # Isso evita que pessoas perto da câmera tenham "erros" maiores que as de longe
    scale = np.sqrt(bbox_size[0]**2 + bbox_size[1]**2) + 1e-6
    
    mean_dist = np.mean(diff) / scale
    return mean_dist

def corrigir_ids_por_consistencia_pose(pose_records, limiar_mudanca=0.15):
    """
    Analisa a lista de registros e corrige IDs que tiveram mudanças biomecânicas impossíveis.
    """
    print(Fore.YELLOW + "\n[POSE-CLEANER] Iniciando verificação de consistência biomecânica...")
    
    # Organizar dados por frame
    frames_dict = {}
    for r in pose_records:
        f_idx = r['frame']
        if f_idx not in frames_dict:
            frames_dict[f_idx] = []
        frames_dict[f_idx].append(r)
    
    sorted_frames = sorted(frames_dict.keys())
    
    # Histórico da última pose conhecida de cada ID: {id_persistente: {'kpts': [], 'bbox_wh': [], 'frame': 0}}
    historico = {}
    ids_trocados = 0

    for f_idx in sorted_frames:
        registros_no_frame = frames_dict[f_idx]
        
        # Para cada pessoa detectada neste frame
        for pessoa in registros_no_frame:
            pid = pessoa.get('id_persistente')
            kpts = pessoa.get('keypoints')
            bbox = pessoa.get('bbox') # [x1, y1, x2, y2]
            
            if pid is None or not kpts or not bbox:
                continue
                
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            bbox_size = (w, h)

            # Se já conhecemos esse ID, verificar se a pose explodiu
            if pid in historico:
                last_data = historico[pid]
                
                # Só verifica se foi no frame imediatamente anterior ou muito recente (max 5 frames gap)
                if f_idx - last_data['frame'] <= 5:
                    dist = calcular_similaridade_pose(kpts, last_data['kpts'], bbox_size)
                    
                    # SE A DISTÂNCIA FOR MUITO GRANDE, HOUVE UMA MUDANÇA BRUSCA!
                    if dist > limiar_mudanca:
                        # Aqui entra a lógica de recuperação:
                        # O ID 'pid' mudou de pose bruscamente.
                        # Será que essa pose atual pertence, na verdade, a outro ID que "sumiu"?
                        
                        melhor_match_id = None
                        menor_dist = float('inf')

                        # Varre o histórico procurando alguém que "sumiu" e tinha essa pose
                        for h_id, h_data in historico.items():
                            if h_id == pid: continue # não comparar consigo mesmo
                            if f_idx - h_data['frame'] > 10: continue # ignorar IDs muito velhos

                            dist_candidato = calcular_similaridade_pose(kpts, h_data['kpts'], bbox_size)
                            
                            # Se achou alguém muito parecido com essa pose atual
                            if dist_candidato < limiar_mudanca and dist_candidato < menor_dist:
                                menor_dist = dist_candidato
                                melhor_match_id = h_id

                        # Se achamos um dono melhor para essa pose
                        if melhor_match_id is not None:
                            # TROCA O ID!
                            print(Fore.CYAN + f"  [CORREÇÃO] Frame {f_idx}: ID {pid} (mudança brusca: {dist:.3f}) -> Reatribuído para ID {melhor_match_id} (match: {menor_dist:.3f})")
                            pessoa['id_persistente'] = melhor_match_id
                            pid = melhor_match_id # Atualiza variável local para salvar no histórico correto
                            ids_trocados += 1

            # Atualiza o histórico com a pose atual (seja ela original ou corrigida)
            historico[pid] = {
                'kpts': kpts,
                'bbox_wh': bbox_size,
                'frame': f_idx
            }

    print(Fore.GREEN + f"[POSE-CLEANER] Concluído. Total de correções automáticas: {ids_trocados}")
    return pose_records