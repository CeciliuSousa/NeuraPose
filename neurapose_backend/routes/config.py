import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException

import neurapose_backend.config_master as cm
from neurapose_backend.nucleo.user_config_manager import UserConfigManager
from neurapose_backend.globals.state import state

router = APIRouter()
logger = logging.getLogger("NeuraPoseAPI")

# ==============================================================
# CONFIG ENDPOINTS
# ==============================================================

# Runtime Config Cache (same logic as main.py)
RUNTIME_CONFIG = UserConfigManager.load_config()

# Sync Initial States
if "ROOT" not in RUNTIME_CONFIG:
    RUNTIME_CONFIG["ROOT"] = str(cm.ROOT)

for k, v in cm.BOT_SORT_CONFIG.items():
    if k not in RUNTIME_CONFIG: RUNTIME_CONFIG[k] = v
for k, v in cm.DEEP_OC_SORT_CONFIG.items():
    if k not in RUNTIME_CONFIG: RUNTIME_CONFIG[k] = v
    
# Sync Reflection from CM to Runtime
for k in dir(cm):
    if k.isupper() and not k.startswith("_"):
        val = getattr(cm, k)
        if (not callable(val) and not isinstance(val, (type, type(Path()), type(Path().resolve())))):
             if k not in RUNTIME_CONFIG:
                 RUNTIME_CONFIG[k] = val

# Sync Runtime to CM
for k, v in RUNTIME_CONFIG.items():
    if hasattr(cm, k):
        attr = getattr(cm, k)
        if not callable(attr) and not isinstance(attr, (type(Path()), Path)):
             setattr(cm, k, v)


def update_cm_runtime(updates: Dict[str, Any], persist: bool = True):
    global RUNTIME_CONFIG
    
    # Logic copied from main.py
    if "RTMPOSE_INPUT_SIZE" in updates and isinstance(updates["RTMPOSE_INPUT_SIZE"], str):
        try:
            w, h = map(int, updates["RTMPOSE_INPUT_SIZE"].split('x'))
            updates["RTMPOSE_INPUT_SIZE"] = (w, h)
        except:
            pass

    RUNTIME_CONFIG.update(updates)
    
    # Sync specific dicts
    for k, v in updates.items():
        if k in cm.BOT_SORT_CONFIG: cm.BOT_SORT_CONFIG[k] = v
        if k in cm.DEEP_OC_SORT_CONFIG: cm.DEEP_OC_SORT_CONFIG[k] = v

    if "TRACKER_NAME" in updates:
        cm.TRACKER_NAME = updates["TRACKER_NAME"]

    # Generic Sync
    for k, v in updates.items():
        if hasattr(cm, k):
            attr = getattr(cm, k)
            if not callable(attr) and not isinstance(attr, (Path, type(Path()))):
                setattr(cm, k, v)
    
    # Derived Updates
    if "RTMPOSE_INPUT_SIZE" in updates:
         if isinstance(RUNTIME_CONFIG["RTMPOSE_INPUT_SIZE"], (tuple, list)):
             cm.SIMCC_W = RUNTIME_CONFIG["RTMPOSE_INPUT_SIZE"][0]
             cm.SIMCC_H = RUNTIME_CONFIG["RTMPOSE_INPUT_SIZE"][1]

    if "USE_FP16" in updates:
        cm.DEEP_OC_SORT_CONFIG["half"] = updates["USE_FP16"]

    if "OSNET_MODEL" in updates:
        new_path = cm.OSNET_DIR / updates["OSNET_MODEL"]
        cm.OSNET_PATH = new_path
        cm.DEEP_OC_SORT_CONFIG["reid_weights"] = str(new_path)
        cm.BOT_SORT_CONFIG["model"] = str(new_path)

def recursive_sanitize(obj):
    """Recursively convert Path objects to strings and Numpy types to native Python types."""
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: recursive_sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_sanitize(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_sanitize(v) for v in obj)
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

@router.get("/config")
def api_get_config():
    """Retorna todas as configurações atuais e caminhos do sistema."""
    
    system_paths = {
        "root": str(cm.ROOT),
        "videos": str(cm.PROCESSING_INPUT_DIR),
        "processados": str(cm.PROCESSING_OUTPUT_DIR),
        "reidentificados": str(cm.REID_OUTPUT_DIR),
        "anotados": str(cm.ANNOTATIONS_OUTPUT_DIR),
        "datasets": str(cm.DATASETS_ROOT),
        "models": str(cm.MODELS_DIR),
        "modelos_treinados": str(cm.TRAINED_MODELS_DIR),
        "output": str(cm.PROCESSING_OUTPUT_DIR),
        "relatorios_testes": str(cm.TEST_REPORTS_DIR)
    }

    # Garante que tudo em RUNTIME_CONFIG seja serializável (recursivamente)
    safe_config = recursive_sanitize(RUNTIME_CONFIG)

    return {
        "status": "success", 
        "config": safe_config,
        "paths": system_paths
    }

@router.get("/config/all")
def api_get_all_config():
    return recursive_sanitize(RUNTIME_CONFIG)

@router.post("/config/update")
def api_update_config(updates: Dict[str, Any]):
    try:
        update_cm_runtime(updates)
        return {"status": "success", "message": "Configurações atualizadas na memória."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/config/reset")
def api_reset_config():
    global RUNTIME_CONFIG
    import importlib
    importlib.reload(cm)
    
    UserConfigManager.reset_to_defaults()
    RUNTIME_CONFIG = UserConfigManager.get_default_config()
    
    for k in cm.BOT_SORT_CONFIG:
        if k in RUNTIME_CONFIG:
            cm.BOT_SORT_CONFIG[k] = RUNTIME_CONFIG[k]
            
    return {"status": "success", "message": "Configurações resetadas para os padrões originais."}

@router.post("/set_preview_state/{enabled}")
def api_set_preview_state(enabled: bool):
    """Endpoint para o Frontend forçar o estado do preview."""
    state.show_preview = enabled
    return {"status": "success", "preview_enabled": state.show_preview}

@router.post("/preview/toggle")
def toggle_preview(enabled: bool = True):
    """Liga/desliga o preview em tempo real durante processamento."""
    state.show_preview = enabled
    if not enabled:
        state.current_frame = None
    return {"status": "preview_enabled" if enabled else "preview_disabled", "show_preview": state.show_preview}

@router.get("/preview/status")
def get_preview_status():
    """Retorna o estado atual do preview."""
    return {"show_preview": state.show_preview, "has_frame": state.current_frame is not None}
