# ================================================================
# neurapose_backend/tracker/utils/ferramentas.py
# ================================================================

import yaml
import numpy as np
from pathlib import Path

# Importa configuracoes centralizadas
import neurapose_backend.config_master as cm


def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))

def save_temp_tracker_yaml():
    """
    Salva o arquivo YAML de configuracao do BoTSORT.
    O path vem do config_master.py (BOTSORT_YAML_PATH).
    """
    path = cm.BOTSORT_YAML_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cm.BOT_SORT_CONFIG, f, sort_keys=False, allow_unicode=True)
    return str(path)
