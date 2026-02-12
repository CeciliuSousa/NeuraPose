import sys
import os
import shutil
import psutil
import platform
import logging
import torch
from pathlib import Path
from typing import Optional, Dict, List
import subprocess

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

import neurapose_backend.config_master as cm
from neurapose_backend.globals.state import state
from neurapose_backend.globals.hardware_monitor import monitor as hw_monitor
from neurapose_backend.nucleo.log_service import LogBuffer
from neurapose_backend.nucleo.websocket_service import ws_manager
from neurapose_backend.otimizador.cuda import gpu_utils as gpu_opt
from neurapose_backend.otimizador.ram import memory as ram_opt

router = APIRouter()
logger = logging.getLogger("NeuraPoseAPI")

HIDDEN_ENTRIES = {
    "app", "pre_processamento", "detector", "LSTM",
    "neurapose_backend", "neurapose_frontend", "node_modules", 
    ".git", ".next", ".venv", "__pycache__", ".agent", ".gemini", 
    "config_master.py", "main.py", "picker.py", "agente.txt", 
    "user_settings.json", "package.json", "package-lock.json", 
    "tsconfig.json", "next.config.js", ".gitignore", "README.md"
}

# ==============================================================
# SYSTEM ENDPOINTS
# ==============================================================

@router.get("/")
def root():
    return {"status": "ok", "system": "NeuraPose API (Modular)"}

@router.get("/health")
def health_check():
    """Ultra-fast health check - sem I/O."""
    return {
        "status": "healthy",
        "system": "NeuraPose API",
        "version": "1.0.0",
        "device": "gpu" if cm.DEVICE == "cuda" else "cpu",
        "processing": state.is_running,
        "paused": state.is_paused,
        "current_process": state.current_process,
        "process_status": state.process_status
    }

@router.post("/shutdown")
def shutdown_server():
    """Finaliza o servidor e todos os processos filhos à força."""
    logger.info("Encerrando servidor...")
    state.kill_all_processes()
    
    import threading
    import signal
    
    def kill_me():
        import time
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)
        sys.exit(0)
        
    threading.Thread(target=kill_me).start()
    return {"status": "shutting_down"}

# ==============================================================
# LOGS & WEBSOCKETS
# ==============================================================

@router.get("/logs")
def get_logs(category: str = "default"):
    """Retorna logs do backend por categoria."""
    return {"logs": LogBuffer().get_logs(category)}

@router.delete("/logs")
def clear_logs(category: Optional[str] = None):
    """Limpa o buffer de logs do servidor (ou de uma categoria)."""
    LogBuffer().clear(category)
    return {"status": "logs cleared", "category": category or "all"}

@router.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket, category: str = "process"):
    """WebSocket para receber logs em tempo real."""
    await ws_manager.connect(websocket)
    try:
        import asyncio
        while True:
            await ws_manager.send_logs(websocket, category)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)

@router.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    """WebSocket para receber status de processamento em tempo real."""
    await ws_manager.connect(websocket)
    try:
        import asyncio
        while True:
            metrics = hw_monitor.get_metrics()
            status = {
                "is_running": state.is_running,
                "is_paused": state.is_paused,
                "current_process": state.current_process,
                "process_status": state.process_status,
                "hardware": {
                    "cpu_percent": metrics.get("cpu", 0),
                    "ram_used_gb": metrics.get("ram_used", 0),
                    "ram_total_gb": metrics.get("ram_total", 0),
                    "gpu_mem_used_gb": metrics.get("gpu_mem", 0),
                    "gpu_mem_total_gb": metrics.get("gpu_total", 0),
                    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
                }
            }
            await ws_manager.send_status(websocket, status)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)

# ==============================================================
# HARDWARE & GPU
# ==============================================================

@router.get("/system/info")
def get_system_info():
    """Retorna informações de uso de hardware (CPU, RAM, GPU)."""
    metrics = hw_monitor.get_metrics()
    return {
        "cpu_percent": metrics["cpu"],
        "ram_used_gb": metrics["ram_used"],
        "ram_total_gb": metrics["ram_total"],
        "gpu_mem_used_gb": metrics["gpu_mem"],
        "gpu_mem_total_gb": metrics["gpu_total"],
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    }

@router.get("/gpu/info")
def api_gpu_info():
    """Retorna informações detalhadas sobre a memória GPU."""
    # Reimplementação local de get_gpu_memory_info para evitar dependência circular se estivesse no main
    if not torch.cuda.is_available():
        return {"available": False, "error": "CUDA não disponível"}
    try:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        total = props.total_memory
        free = total - reserved
        return {
            "available": True,
            "device_name": props.name,
            "device_index": device,
            "total_gb": round(total / (1024**3), 2),
            "allocated_gb": round(allocated / (1024**3), 2),
            "reserved_gb": round(reserved / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "utilization_percent": round((allocated / total) * 100, 1),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}

@router.post("/gpu/cleanup")
def api_gpu_cleanup(force: bool = True):
    """Libera memória GPU não utilizada."""
    before = api_gpu_info()
    gpu_opt.clear_gpu_cache(force=force)
    ram_opt.force_gc()
    after = api_gpu_info()
    
    freed = before.get("reserved_gb", 0) - after.get("reserved_gb", 0)
    return {
        "success": True,
        "freed_gb": round(freed, 3),
        "before": before,
        "after": after,
        "message": f"Liberado {freed:.3f} GB de memória GPU"
    }

@router.post("/gpu/optimize")
def api_gpu_optimize(required_gb: float = 2.0):
    """Tenta garantir quantidade mínima de memória GPU disponível."""
    info = api_gpu_info()
    if info.get("free_gb", 0) >= required_gb:
        success = True
    else:
        api_gpu_cleanup(force=True)
        info = api_gpu_info()
        success = info.get("free_gb", 0) >= required_gb
        
    return {
        "success": success,
        "required_gb": required_gb,
        "available_gb": info.get("free_gb", 0),
        "message": "Memória suficiente disponível" if success else f"Não foi possível liberar {required_gb}GB"
    }

# ==============================================================
# FILE SYSTEM
# ==============================================================

@router.get("/browse")
def browse_path(path: str = "."):
    """Lista o conteúdo de uma pasta para o explorador de arquivos do frontend."""
    try:
        project_root = cm.ROOT.parent
        p_obj = Path(path)
        
        # Se for relativo, ancora no projeto
        if not p_obj.is_absolute():
            p = (project_root / p_obj).resolve()
        else:
            p = p_obj.resolve()
            
        if not p.exists():
            # Fallback para project root se não existir
            p = project_root
        
        items = []
        for entry in p.iterdir():
            if entry.name in HIDDEN_ENTRIES:
                continue
            items.append({
                "name": entry.name,
                "path": str(entry.absolute()),
                "is_dir": entry.is_dir()
            })
            
        items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
        return {
            "current": str(p.absolute()),
            "parent": str(p.parent.absolute()) if p.parent != p else str(p.absolute()),
            "items": items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ls")
def list_directory(path: Optional[str] = None):
    """Lista diretórios para o explorador de arquivos customizado."""
    try:
        project_root = cm.ROOT.parent
        
        if path:
            p_obj = Path(path)
            if not p_obj.is_absolute():
                root = (project_root / p_obj).resolve()
            else:
                root = p_obj.resolve()
        else:
            root = project_root

        if not root.exists():
            root = project_root

        items = []
        for item in root.iterdir():
            if item.name in HIDDEN_ENTRIES: continue
            if item.is_dir():
                items.append({"name": item.name, "path": str(item), "isDir": True})

        return {"items": sorted(items, key=lambda x: x['name']), "current": str(root)}
    except Exception as e:
        logger.error(f"Erro no /ls: {e}")
        return {"items": [], "current": str(cm.ROOT.parent), "error": str(e)}

@router.get("/pick-folder")
def pick_folder_endpoint(initial_dir: Optional[str] = None):
    """Abre o seletor de pastas nativo do Windows."""
    # Assume que picker.py está na raiz do backend (onde config_master.py está)
    picker_script = str(cm.ROOT / "picker.py")
    
    if not os.path.exists(picker_script):
        # Fallback para tentativa local
        picker_script = str(Path(__file__).parent.parent / "picker.py")
    
    cmd = [sys.executable, picker_script]
    
    # Se initial_dir não for passado, usa a raiz do projeto
    start_dir = initial_dir if initial_dir else str(cm.ROOT.parent)
    cmd.append(start_dir)
        
    try:
        result = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
        logger.info(f"[PICKER] Pasta selecionada: {result}")
        return {"path": result}
    except subprocess.CalledProcessError:
        return {"path": None}
    except Exception as e:
        logger.error(f"Erro ao abrir seletor: {e}")
        raise HTTPException(status_code=500, detail=str(e))
