import sys
from pathlib import Path
from neurapose_backend import config_master as cm

# Adiciona o diretório atual ao path para importar config_master
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

def sync_constants():
    # Caminho para o arquivo constants.ts no frontend
    # Ajuste o caminho relativo conforme a estrutura do seu projeto
    frontend_constants_path = current_dir.parent / "neurapose_tauri" / "src" / "constants.ts"
    
    if not frontend_constants_path.parent.exists():
        print(f"ERRO: Diretório do frontend não encontrado em: {frontend_constants_path.parent}")
        return

    # Conteúdo a ser escrito
    ts_content = f"""/**
 * ARQUIVO GERADO AUTOMATICAMENTE - NÃO EDITE MANUALMENTE
 * Sincronizado com neurapose_backend/config_master.py
 * Para alterar estes valores, edite o config_master.py e execute sync_constants.py
 */

export const CLASSE1 = "{cm.CLASSE1}";
export const CLASSE2 = "{cm.CLASSE2}";
"""

    try:
        with open(frontend_constants_path, "w", encoding="utf-8") as f:
            f.write(ts_content)
        
        print(f"SUCESSO: constants.ts atualizado!")
        print(f"Arquivo: {frontend_constants_path}")
        print(f"CLASSE1: {cm.CLASSE1}")
        print(f"CLASSE2: {cm.CLASSE2}")
        
    except Exception as e:
        print(f"ERRO ao escrever constants.ts: {e}")

if __name__ == "__main__":
    sync_constants()
