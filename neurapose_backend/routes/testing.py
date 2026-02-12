import logging
import json
import importlib
import sys
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

import neurapose_backend.config_master as cm
from neurapose_backend.globals.state import state
from neurapose_backend.nucleo.log_service import CaptureOutput

router = APIRouter()
logger = logging.getLogger("NeuraPoseAPI")

# ==============================================================
# MODELS
# ==============================================================

class TestRequest(BaseModel):
    model_name: str
    datasets: List[str]
    device: str = "cuda"
    test_split: bool = True
    batch_size: int = 32

# ==============================================================
# HELPERS
# ==============================================================

def run_testing_task(req_dict: dict):
    state.reset()
    state.is_running = True
    state.current_process = 'test'
    state.process_status = 'processing'
    
    with CaptureOutput(category="test"):
        try:
            # Emulation of command line args for testar_modelo.py
            # TODO: Refactor testar_modelo.py to not rely on sys.argv
            sys.argv = ["testar_modelo.py"]
            sys.argv.append("--model_name")
            sys.argv.append(req_dict["model_name"])
            
            for ds in req_dict["datasets"]:
                sys.argv.append("--datasets")
                sys.argv.append(ds)
                
            if not req_dict.get("test_split", True):
                sys.argv.append("--no_split")
                
            sys.argv.append("--batch_size")
            sys.argv.append(str(req_dict.get("batch_size", 32)))
            
            logger.info(f"Iniciando teste com args: {sys.argv}")
            
            import neurapose_backend.app.testar_modelo as tm
            importlib.reload(tm)
            tm.main()
            
            state.process_status = 'success'
        except Exception as e:
            logger.error(f"Erro no teste: {e}")
            state.process_status = 'error'
        finally:
            state.is_running = False
            state.current_frame = None

# ==============================================================
# ENDPOINTS
# ==============================================================

@router.post("/test")
async def start_testing(req: TestRequest, background_tasks: BackgroundTasks):
    """Inicia teste de modelos."""
    if req.device == "cpu":
        cm.DEVICE = "cpu"
    else:
        import torch
        cm.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
    background_tasks.add_task(run_testing_task, req.dict())
    return {"status": "started", "message": "Fase de testes iniciada."}

@router.post("/test/stop")
async def stop_testing():
    state.stop_requested = True
    logger.info("Solicitação de parada de teste recebida.")
    return {"status": "stopped", "message": "Solicitação de parada enviada."}

@router.get("/test/reports")
def list_test_reports():
    """Lista relatórios de testes gerados."""
    reports_dir = cm.TEST_REPORTS_DIR
    if not reports_dir.exists():
        return []
        
    reports = []
    for report in reports_dir.iterdir():
        if report.is_dir():
            metricas_path = report / "metricas.json"
            if metricas_path.exists():
                with open(metricas_path, "r") as f:
                    data = json.load(f)
                    reports.append({
                        "name": report.name,
                        "date": report.stat().st_mtime,
                        "metrics": data
                    })
    return sorted(reports, key=lambda x: x["date"], reverse=True)
