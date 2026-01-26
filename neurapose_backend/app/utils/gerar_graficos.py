# ==============================================================
# neurapose-backend/app/utils/gerar_graficos.py
# ==============================================================

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any
from colorama import Fore

import neurapose_backend.config_master as cm


# Configuração de estilo profissional
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Paleta de cores profissional
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#06A77D',
    'danger': '#D62839',
    'warning': '#F77F00',
    'info': '#118AB2',
    'neutral': '#6C757D',
}


def gerar_grafico_matriz_confusao(conf_matrix: np.ndarray, metricas_dir: Path, modelo_nome: str = "ROBUST-LSTM"):
    """
    Gera e salva gráfico da matriz de confusão como heatmap.
    
    Args:
        conf_matrix: Matriz de confusão (2x2)
        metricas_dir: Diretório para salvar o gráfico
        modelo_nome: Nome do modelo para incluir no título
    """
    plt.figure(figsize=(8, 6))
    
    # Criar heatmap com anotações
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=True,
        square=True,
        xticklabels=[cm.CLASSE1, cm.CLASSE2],
        yticklabels=[cm.CLASSE1, cm.CLASSE2],
        linewidths=1,
        linecolor='gray'
    )
    
    plt.title(f'Matriz de Confusão - {modelo_nome}', fontweight='bold', pad=15)
    plt.ylabel('Verdadeiro', fontweight='bold')
    plt.xlabel('Predito', fontweight='bold')
    plt.tight_layout()
    
    output_path = metricas_dir / "confusion_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # print(Fore.GREEN + f"[OK] Gráfico de matriz de confusão salvo: {output_path}")


def gerar_grafico_distribuicao_classes(metricas: Dict[str, Any], metricas_dir: Path, modelo_nome: str = "ROBUST-LSTM"):
    """
    Gera gráfico de distribuição de predições vs ground truth.
    
    Args:
        metricas: Dicionário com métricas incluindo matriz de confusão
        metricas_dir: Diretório para salvar o gráfico
        modelo_nome: Nome do modelo
    """
    conf_matrix = np.array(metricas['confusion_matrix'])
    
    # Se a matriz não for 2x2, ajustar
    # if conf_matrix.shape != (2, 2):
    #     print(Fore.YELLOW + "[AVISO] Matriz de confusão não é 2x2, ajustando...")
    #     return
    
    TN, FP, FN, TP = conf_matrix.ravel()
    
    # Calcular totais
    total_classe2_real = TN + FP
    total_classe1_real = FN + TP
    total_classe1_pred = TN + FN
    total_classe2_pred = FP + TP
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico 1: Ground Truth
    categorias_gt = [cm.CLASSE2, cm.CLASSE1]
    valores_gt = [total_classe2_real, total_classe1_real]
    colors_gt = [COLORS['success'], COLORS['danger']]
    
    # ...

    # Gráfico 2: Predições
    categorias_pred = [cm.CLASSE1, cm.CLASSE2]
    valores_pred = [total_classe1_pred, total_classe2_pred]
    colors_pred = [COLORS['success'], COLORS['danger']]
    
    bars2 = ax2.bar(categorias_pred, valores_pred, color=colors_pred, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_title('Distribuição Predita', fontweight='bold', pad=10)
    ax2.set_ylabel('Quantidade de Vídeos', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Adicionar valores nas barras
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    fig.suptitle(f'Distribuição de Classes - {modelo_nome}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = metricas_dir / "class_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # print(Fore.GREEN + f"[OK] Gráfico de distribuição de classes salvo: {output_path}")


def gerar_grafico_metricas_comparativas(metricas: Dict[str, Any], metricas_dir: Path, modelo_nome: str = cm.TEMPORAL_MODEL):
    """
    Gera gráfico de barras comparando principais métricas.
    
    Args:
        metricas: Dicionário com as métricas calculadas
        metricas_dir: Diretório para salvar o gráfico
        modelo_nome: Nome do modelo
    """
    # Selecionar métricas principais
    metricas_nomes = ['Accuracy', f'Precision\n{cm.CLASSE2}', f'Recall\n{cm.CLASSE2}', 'F1 Macro', 'Balanced\nAccuracy']
    metricas_valores = [
        metricas.get('accuracy', 0),
        metricas.get(f'precision_{cm.CLASSE2}', 0),
        metricas.get(f'recall_{cm.CLASSE2}', 0),
        metricas.get('f1_macro', 0),
        metricas.get('balanced_accuracy', 0)
    ]
    
    # Cores para cada métrica
    cores = [COLORS['primary'], COLORS['success'], COLORS['info'], COLORS['secondary'], COLORS['warning']]
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(metricas_nomes, metricas_valores, color=cores, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    plt.title(f'Métricas de Desempenho - {modelo_nome}', fontweight='bold', pad=15, fontsize=14)
    plt.ylabel('Score', fontweight='bold', fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Adicionar valores nas barras
    for bar, valor in zip(bars, metricas_valores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{valor:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Adicionar linha de referência em 0.5
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (0.5)')
    plt.legend()
    
    plt.tight_layout()
    
    output_path = metricas_dir / "metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # print(Fore.GREEN + f"[OK] Gráfico de métricas comparativas salvo: {output_path}")


def gerar_grafico_distribuicao_confianca(resultados: list, labels: Dict[str, Any], metricas_dir: Path, modelo_nome: str = "ROBUST-LSTM"):
    """
    Gera gráfico de distribuição dos scores de confiança por classe.
    
    Args:
        resultados: Lista de resultados dos vídeos
        labels: Dicionário de labels do ground truth
        metricas_dir: Diretório para salvar o gráfico
        modelo_nome: Nome do modelo
    """
    score_classe1 = []
    score_classe2 = []
    
    for r in resultados:
        stem = Path(r["video"]).stem
        if stem not in labels:
            continue
        
        id_map = labels[stem]
        id_map = labels[stem]
        # GT: 1 se qualquer ID for CLASSE2
        gt = 1 if any(v.lower() == cm.CLASSE2 for v in id_map.values()) else 0
        score = r.get(f"score_{cm.CLASSE2}", float(r["pred"]))
        
        if gt == 0:
            score_classe1.append(score)
        else:
            score_classe2.append(score)
    
    if not score_classe1 and not score_classe2:
        print(Fore.YELLOW + "[AVISO] Sem dados suficientes para gráfico de distribuição de confiança")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histograma Classe 1
    if score_classe1:
        ax1.hist(score_classe1, bins=20, color=COLORS['success'], alpha=0.7, edgecolor='black', linewidth=1)
        ax1.axvline(np.mean(score_classe1), color='darkgreen', linestyle='--', linewidth=2, label=f'Média: {np.mean(score_classe1):.3f}')
        ax1.set_title(f'Distribuição de Scores - {cm.CLASSE1}', fontweight='bold', pad=10)
        ax1.set_xlabel(f'Score de Confiança {cm.CLASSE1}', fontweight='bold')
        ax1.set_ylabel('Frequência', fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
    
    # Histograma Classe 2
    if score_classe2:
        ax2.hist(score_classe2, bins=20, color=COLORS['danger'], alpha=0.7, edgecolor='black', linewidth=1)
        ax2.axvline(np.mean(score_classe2), color='darkred', linestyle='--', linewidth=2, label=f'Média: {np.mean(score_classe2):.3f}')
        ax2.set_title(f'Distribuição de Scores - {cm.CLASSE2}', fontweight='bold', pad=10)
        ax2.set_xlabel(f'Score de Confiança {cm.CLASSE2}', fontweight='bold')
        ax2.set_ylabel('Frequência', fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    
    fig.suptitle(f'Distribuição de Confiança por Classe - {modelo_nome}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = metricas_dir / "confidence_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # print(Fore.GREEN + f"[OK] Gráfico de distribuição de confiança salvo: {output_path}")


def gerar_todos_graficos(metricas_json_path: Path, resultados_json_path: Path, labels_json_path: Path, modelo_nome: str = "ROBUST-LSTM"):
    """
    Função principal para gerar todos os gráficos de análise.
    
    Args:
        metricas_json_path: Caminho para metricas_teste_avancadas.json
        resultados_json_path: Caminho para resultados_teste.json
        labels_json_path: Caminho para labels.json
        modelo_nome: Nome do modelo
    """
    if not metricas_json_path.exists():
        print(Fore.YELLOW + f"[AVISO] Arquivo de métricas não encontrado: {metricas_json_path}")
        return
    
    metricas_dir = metricas_json_path.parent
    
    # Carregar métricas
    with open(metricas_json_path, 'r', encoding='utf-8') as f:
        metricas = json.load(f)
    
    print(Fore.CYAN + f"\n[INFO] GERANDO GRÁFICOS DE ANALISE DO MODELO: {modelo_nome}...\n")
    
    # 1. Matriz de Confusão
    conf_matrix = np.array(metricas['confusion_matrix'])
    gerar_grafico_matriz_confusao(conf_matrix, metricas_dir, modelo_nome)
    
    # 2. Distribuição de Classes
    gerar_grafico_distribuicao_classes(metricas, metricas_dir, modelo_nome)
    
    # 3. Métricas Comparativas
    gerar_grafico_metricas_comparativas(metricas, metricas_dir, modelo_nome)
    
    # 4. Distribuição de Confiança (requer resultados e labels)
    if resultados_json_path.exists() and labels_json_path.exists():
        with open(resultados_json_path, 'r', encoding='utf-8') as f:
            resultados = json.load(f)
        with open(labels_json_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        
        gerar_grafico_distribuicao_confianca(resultados, labels, metricas_dir, modelo_nome)
    else:
        print(Fore.YELLOW + "[AVISO] Arquivos de resultados ou labels não encontrados para gráfico de confiança")
    
    # print(Fore.GREEN + f"\n[OK] Todos os gráficos foram gerados em: {metricas_dir}\n")
