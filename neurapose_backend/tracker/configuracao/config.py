# ================================================================
# neurapose/tracker/configuracao/config.py
# ================================================================
# ATENCAO: NAO EDITE ESTE ARQUIVO!
# Todas as configuracoes estao em: config_master.py
# ================================================================

from pathlib import Path

# Importa BOT_SORT_CONFIG diretamente do config_master
from neurapose.config_master import BOT_SORT_CONFIG, OSNET_PATH

# Re-exporta para manter compatibilidade com imports existentes
__all__ = ["BOT_SORT_CONFIG", "OSNET_PATH"]
