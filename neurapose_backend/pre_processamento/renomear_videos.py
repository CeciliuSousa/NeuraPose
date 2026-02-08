import os
import shutil
from pathlib import Path

def renomear_videos(input_dir_name, classe1, classe2):
    # Definir caminhos relativos ao ROOT do projeto
    # Como este script ficará em neurapose_backend/pre_processamento/,
    # o ROOT é dois níveis acima.
    ROOT = Path(__file__).resolve().parent.parent.parent
    
    input_path = ROOT / input_dir_name
    output_base = ROOT / "neurapose_backend" / "videos"
    output_path = output_base / input_dir_name
    
    # Criar pasta de saída se não existir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Arquivo de log
    log_file = output_path / "log_renomeacao.txt"
    
    # Listas para cada classe
    videos_classe1 = []
    videos_classe2 = []
    
    # Formatos de vídeo aceitos
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
    # 1. Loop para identificar os vídeos
    if not input_path.exists():
        print(f"Erro: Pasta de entrada {input_path} não encontrada.")
        return

    for file in input_path.iterdir():
        if file.suffix.lower() in video_extensions:
            nome_minusculo = file.name.lower()
            if classe1.lower() in nome_minusculo:
                videos_classe1.append(file)
            elif classe2.lower() in nome_minusculo:
                videos_classe2.append(file)
    
    # Ordenar para manter uma sequência lógica
    videos_classe1.sort()
    videos_classe2.sort()
    
    logs = []
    
    # 2. Renomear vídeos da classe 1
    for i, file_path in enumerate(videos_classe1):
        novo_nome = f"cena-{classe1}-{i:05d}{file_path.suffix}"
        destino = output_path / novo_nome
        shutil.copy2(file_path, destino)
        logs.append(f"Video {file_path.name} foi renomeado para {novo_nome}")
        
    # 3. Renomear vídeos da classe 2
    for i, file_path in enumerate(videos_classe2):
        novo_nome = f"cena-{classe2}-{i:05d}{file_path.suffix}"
        destino = output_path / novo_nome
        shutil.copy2(file_path, destino)
        logs.append(f"Video {file_path.name} foi renomeado para {novo_nome}")
        
    # 4. Salvar log
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(logs))
        
    print(f"Processo concluído!")
    print(f"Vídeos salvos em: {output_path}")
    print(f"Log gerado em: {log_file}")

if __name__ == "__main__":
    import sys
    ROOT = Path(__file__).resolve().parent.parent.parent
    if str(ROOT) not in sys.path: sys.path.append(str(ROOT))
    
    import neurapose_backend.config_master as cm

    # Configuração das classes conforme solicitado
    CLASSE1 = cm.CLASSE2.lower() # CLASSE2
    CLASSE2 = cm.CLASSE1.lower() # CLASSE1
    
    # Nome da pasta de entrada
    PASTA_ENTRADA = "labex-furtos-2026"
    
    renomear_videos(PASTA_ENTRADA, CLASSE1, CLASSE2)
