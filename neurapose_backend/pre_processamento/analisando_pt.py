# AlphaPose/src/analisando_pt.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import time

# --- Configuração da Animação ---
FPS = 30 # Frames por segundo (padrão 30)
DELAY = 1 / FPS # Atraso em segundos entre frames
# Variáveis globais para a figura (usadas na animação)
fig = None
ax = None
scatters = []
lines = []
texts = []

# Caminho para incluir diretório raiz (se precisar de módulos locais)

# Caminho do arquivo .pkl
# file_path = "./datasets/processed_complete_modify/train/processed/data.pkl"
file_path = "./datasets/data-labex/treino/data/data.pt"

primeiraClasse = "normal"
segundaClasse = "furto"

# Conexões dos 17 keypoints (Padrão COCO)
CONNECTIONS = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (2, 6)
]

def setup_animation(all_keypoints):
    global fig, ax
    
    # Determina limites globais para o plot
    all_x = all_keypoints[:, :, 0].flatten()
    all_y = all_keypoints[:, :, 1].flatten()
    
    # Filtra zeros (pontos não detectados costumam ser 0)
    valid_x = all_x[all_x > 0.1]
    valid_y = all_y[all_y > 0.1]
    
    if len(valid_x) == 0: valid_x = all_x
    if len(valid_y) == 0: valid_y = all_y
        
    min_x, max_x = valid_x.min(), valid_x.max()
    min_y, max_y = valid_y.min(), valid_y.max()
    
    margin = 50
    
    plt.ion() # Modo interativo
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Salva limites no ax para reutilizar
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    ax.invert_yaxis() # Y cresce para baixo
    ax.set_aspect('equal')

def animar_pose(keypoints_frame, connections, frame_idx, total_frames, label):
    global ax
    
    # Limpa o eixo mas preserva limites se possível, ou re-seta
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    ax.clear()
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # ax.invert_yaxis() # Se o ylim já estiver invertido (maior, menor), não precisa inverter de novo, mas clear() reseta.
    # Verificando se ylim está invertido
    if ylim[0] < ylim[1]: # Se normal (menor, maior), inverte
        ax.invert_yaxis()
    
    ax.set_aspect('equal')
    
    ax.set_title(f"Frame: {frame_idx}/{total_frames} | Label: {label}")
    
    # Desenha conexões
    for p1, p2 in connections:
        if p1 < len(keypoints_frame) and p2 < len(keypoints_frame):
            x1, y1 = keypoints_frame[p1]
            x2, y2 = keypoints_frame[p2]
            if x1 > 0.1 and y1 > 0.1 and x2 > 0.1 and y2 > 0.1:
                ax.plot([x1, x2], [y1, y2], 'g-', lw=2)
            
    # Desenha pontos
    if keypoints_frame.shape[-1] >= 2:
        for i, (x, y) in enumerate(keypoints_frame):
            if x > 0.1 and y > 0.1:
                ax.scatter(x, y, c='red', s=50) 
                ax.text(x, y, str(i), fontsize=8, color='blue')
                
    plt.draw()
    plt.pause(DELAY)

def visualizar_amostra(dataset_split, idx=0, num_frames=5):
    """Função principal que carrega a amostra e executa a ANIMAÇÃO."""
    if idx >= len(dataset_split):
        print(f"❌ Índice {idx} fora do intervalo. Total de amostras: {len(dataset_split)}")
        return

    # CORREÇÃO: Usa '*rest' para desempacotar o que for extra (mais de 2 elementos)
    keypoints_tensor, label, *rest = dataset_split[idx]
    
    if rest:
        print(f"ℹ️ Metadados extras (ignorados): {rest}")
        
    keypoints_np = keypoints_tensor if isinstance(keypoints_tensor, np.ndarray) else np.array(keypoints_tensor)
    
    print(f"📏 Shape original do tensor da amostra {idx}: {keypoints_np.shape}")
    
    # --- VERIFICAÇÃO E AJUSTE DE DIMENSÕES ---
    
    if keypoints_np.ndim != 3:
        print(f"❌ Erro de Dimensão: Keypoints esperados 3D, mas veio {keypoints_np.ndim}D. Impossível visualizar.")
        return

    # Assumindo que T, V, C seja (frames, 17, 2+)
    T, V, C = keypoints_np.shape 
    
    # Transpõe se o formato for (C, T, V) ou outro com 2 na primeira dimensão
    if T <= 3 and V > T and C > T:
        keypoints_np = np.transpose(keypoints_np, (1, 2, 0)) # Tenta (C, T, V) -> (T, V, C)
        print(f"🔄 Shape ajustado (transposto): {keypoints_np.shape}")
        
    T, V, C = keypoints_np.shape # Re-avalia a forma

    if V != 17 or C < 2:
        print(f"❌ Erro de Shape: Keypoints esperados 17 pontos e 2+ coordenadas, mas veio {V} pontos e {C} coordenadas.")
        return

    # --- INÍCIO DA ANIMAÇÃO ---

    print(f"\n🔍 Amostra {idx} | Classe: {segundaClasse if label == 1 else primeiraClasse}")
    print(f"📏 Shape final: {keypoints_np.shape} (frames, 17, 2+)")

    # Se num_frames for definido como 90, mas a amostra tem 30, exibe 30.
    num_frames_a_exibir = min(num_frames, keypoints_np.shape[0])
    total_frames = keypoints_np.shape[0]

    if num_frames_a_exibir == 0:
        print(f"🖼️ Não exibindo frames, pois 'num_frames' é 0 ou a amostra está vazia.")
        return

    print(f"🎬 Solicitado: {num_frames} frames | Disponível: {total_frames} frames")
    print(f"🎬 Exibindo: {num_frames_a_exibir} frames a {FPS} FPS...")
    
    setup_animation(keypoints_np)
    
    # Loop de Animação
    for t in range(num_frames_a_exibir):
        # Passa apenas as coordenadas (x, y) para a plotagem
        animar_pose(keypoints_np[t, :, :2], CONNECTIONS, t, total_frames, label)
        
    plt.ioff()
    plt.show(block=True) # Bloqueia para manter a última pose na tela

def listar_amostras(data):
    print(f"\n📊 Quantidade de amostras: {len(data)}")

def detalhar_amostras(data):
    # CORREÇÃO: Usa '*rest' para desempacotar o que for extra na iteração
    furto_count = sum(1 for keypoints_tensor, label, *rest in data if label == 1)
    
    normal_count = len(data) - furto_count
    print(f"\n📊 Totais Gerais:")
    print(f" - Total:  {len(data)}")
    print(f" - {primeiraClasse}: {normal_count}")
    print(f" - {segundaClasse}:  {furto_count}")

# =======================================================================
# EXECUÇÃO PRINCIPAL
# =======================================================================

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

try:
    print(f"\n🔍 Carregando arquivo: {file_path}")
    
    raw_data = torch.load(file_path, map_location="cpu")
    data = [] 

    if isinstance(raw_data, dict):
        print("⚠️ O objeto carregado é um dicionário. Tentando separar keypoints e labels...")
        
        # Tentativa de chaves comuns (ajustadas conforme a última iteração)
        keys_to_check = [('keypoints', 'labels'), ('data', 'labels'), ('keypoints', 'label')]
        keypoints_tensor, labels_tensor = None, None
        
        for k_k, k_l in keys_to_check:
            if k_k in raw_data and k_l in raw_data:
                keypoints_tensor = raw_data[k_k]
                labels_tensor = raw_data[k_l]
                break
        
        if keypoints_tensor is None or labels_tensor is None:
            print(f"❌ Não foi possível encontrar as chaves 'keypoints' e 'labels'. Chaves disponíveis: {raw_data.keys()}")
            raise ValueError("Estrutura do arquivo .pt desconhecida.")

        if not isinstance(keypoints_tensor, torch.Tensor) or not isinstance(labels_tensor, torch.Tensor):
            raise TypeError("As chaves encontradas não apontam para torch.Tensor.")

        N = keypoints_tensor.shape[0]
        if N != labels_tensor.shape[0]:
            raise ValueError(f"Número de amostras inconsistente: Keypoints ({N}) vs Labels ({labels_tensor.shape[0]})")
        
        labels_np = labels_tensor.cpu().numpy().flatten().tolist()
        
        # Cria a lista de tuplas (keypoints_tensor, label)
        data = [(keypoints_tensor[i], labels_np[i]) for i in range(N)]
        print(f"✅ Dados reestruturados em {N} amostras.")

    elif isinstance(raw_data, (list, tuple)):
        data = raw_data
    else:
        raise TypeError(f"O objeto carregado ({type(raw_data)}) não é uma lista nem um dicionário reconhecido.")


    if len(data) > 0:
        primeira_amostra = data[0]
        print(f"Tipo da primeira amostra: {type(primeira_amostra)}")
        print(f"Tamanho da primeira amostra: {len(primeira_amostra)}")
    
    listar_amostras(data)
    detalhar_amostras(data)

    visualizar_amostra(data, idx=30, num_frames=114)

except Exception as e:
    print("❌ Erro ao carregar ou processar o arquivo:")
    import traceback
    traceback.print_exc(file=sys.stdout)