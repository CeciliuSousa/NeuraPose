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
    model_path: str
    dataset_path: str
    device: str = "cuda"
    show_preview: bool = False

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
            model_path = req_dict["model_path"]
            dataset_path = req_dict["dataset_path"]
            show_preview = req_dict.get("show_preview", False)
            
            # logger.info(f"Iniciando teste dinâmico com: modelo={model_path}, dataset={dataset_path}, show={show_preview}")
            
            import neurapose_backend.app.testar_modelo as tm
            importlib.reload(tm)
            tm.main(
                override_model_dir=model_path,
                override_input_dir=dataset_path,
                override_show=show_preview
            )
            
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
