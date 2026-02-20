import logging
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from colorama import Fore

import neurapose_backend.config_master as cm
from neurapose_backend.globals.state import state
from neurapose_backend.nucleo.log_service import CaptureOutput
from neurapose_backend.routes.config import update_cm_runtime

router = APIRouter()
logger = logging.getLogger("NeuraPoseAPI")

# ==============================================================
# MODELS
# ==============================================================

class TrainRequest(BaseModel):
    epochs: int
    batch_size: int
    learning_rate: float
    model_name: str
    dataset_name: str
    temporal_model: str = "tft" 

class TrainStartRequest(BaseModel):
    dataset_path: str
    model_type: str = "tft"
    epochs: int = 5000
    batch_size: int = 32
    lr: float = 0.0003
    dropout: float = 0.3
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 8
    kernel_size: int = 5
    use_data_augmentation: bool = False
    use_lr_scheduler: bool = True

class TrainRetrainRequest(TrainStartRequest):
    pretrained_path: str 

# ==============================================================
# HELPERS
# ==============================================================

def run_training_task(req: TrainRequest):
    """Executa o pipeline de treinamento em segundo plano."""
    state.reset()
    state.is_running = True
    state.current_process = 'train'  
    
    with CaptureOutput():
        logger.info(f"Iniciando treinamento: {req.model_name} (Dataset: {req.dataset_name})")
        try:
            from neurapose_backend.LSTM.pipeline.treinador import main as start_train
            
            updates = {
                "EPOCHS": req.epochs,
                "BATCH_SIZE": req.batch_size,
                "LEARNING_RATE": req.learning_rate,
                "MODEL_NAME": req.model_name,
                "PROCESSING_DATASET": req.dataset_name,
                "TEMPORAL_MODEL": req.temporal_model
            }
            update_cm_runtime(updates, persist=False)
            
            start_train()
            logger.info(f"Treinamento concluído para {req.model_name}")
            state.process_status = 'success'
        except Exception as e:
            logger.error(f"Erro no treinamento: {e}")
            state.process_status = 'error'
        finally:
            state.is_running = False
            state.current_frame = None

# ==============================================================
# ENDPOINTS
# ==============================================================

@router.post("/train")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """Inicia treinamento."""
    background_tasks.add_task(run_training_task, req)
    return {"status": "started", "detail": f"Training {req.model_name}"}

@router.post("/train/start")
async def train_model_start(req: TrainStartRequest, background_tasks: BackgroundTasks):
    from neurapose_backend.LSTM.pipeline.treinador import main as train_main
    
    dataset_path = Path(req.dataset_path).resolve()
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset não encontrado: {dataset_path}")
    
    data_file = dataset_path / "treino" / "data" / "data.pt"
    if not data_file.exists():
        data_file = dataset_path / "data.pt"
    if not data_file.exists():
        pt_files = list(dataset_path.rglob("*.pt"))
        if pt_files:
            data_file = pt_files[0]
        else:
            raise HTTPException(status_code=400, detail=f"Arquivo .pt não encontrado em {dataset_path}")
    
    dataset_name = dataset_path.name
    
    def run_train():
        state.reset()
        state.is_running = True
        state.current_process = 'train'
        state.process_status = 'processing'
        with CaptureOutput(category="train"):
            try:
                import sys
                cm.USE_DATA_AUGMENTATION = req.use_data_augmentation
                cm.USE_LR_SCHEDULER = req.use_lr_scheduler
                
                sys.argv = [
                    "treinador.py",
                    "--dataset", str(data_file),
                    "--model", req.model_type,
                    "--epochs", str(req.epochs),
                    "--batch_size", str(req.batch_size),
                    "--lr", str(req.lr),
                    "--dropout", str(req.dropout),
                    "--hidden_size", str(req.hidden_size),
                    "--num_layers", str(req.num_layers),
                    "--num_heads", str(req.num_heads),
                    "--name", dataset_name
                ]
                train_main()
                state.process_status = 'success'
            except Exception as e:
                logger.error(f"[ERRO] Treinamento falhou: {e}")
                state.process_status = 'error'
            finally:
                state.is_running = False
    
    background_tasks.add_task(run_train)
    return {"status": "started", "message": f"Treinamento iniciado: {dataset_name} com {req.model_type}"}

@router.post("/train/retrain")
async def train_model_retrain(req: TrainRetrainRequest, background_tasks: BackgroundTasks):
    from neurapose_backend.LSTM.pipeline.treinador import main as train_main
    
    dataset_path = Path(req.dataset_path).resolve()
    pretrained_path = Path(req.pretrained_path).resolve()
    
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset não encontrado: {dataset_path}")
    if not pretrained_path.exists():
        raise HTTPException(status_code=404, detail=f"Modelo pré-treinado não encontrado: {pretrained_path}")
    
    data_file = dataset_path / "treino" / "data" / "data.pt"
    if not data_file.exists():
        pt_files = list(dataset_path.rglob("*.pt"))
        data_file = pt_files[0] if pt_files else None
    if not data_file:
        raise HTTPException(status_code=400, detail=f"Arquivo .pt não encontrado em {dataset_path}")
    
    model_best = pretrained_path / "model_best.pt"
    if not model_best.exists():
        model_best = pretrained_path
    
    dataset_name = dataset_path.name
    
    def run_retrain():
        state.reset()
        state.is_running = True
        state.current_process = 'train'
        state.process_status = 'processing'
        with CaptureOutput(category="train"):
            try:
                import sys
                cm.USE_DATA_AUGMENTATION = req.use_data_augmentation
                cm.USE_LR_SCHEDULER = req.use_lr_scheduler

                sys.argv = [
                    "treinador.py",
                    "--dataset", str(data_file),
                    "--pretrained", str(model_best),
                    "--model", req.model_type,
                    "--epochs", str(req.epochs),
                    "--batch_size", str(req.batch_size),
                    "--lr", str(req.lr),
                    "--dropout", str(req.dropout),
                    "--hidden_size", str(req.hidden_size),
                    "--num_layers", str(req.num_layers),
                    "--num_heads", str(req.num_heads),
                    "--name", f"{dataset_name}-retreinado"
                ]
                train_main()
                state.process_status = 'success'
            except Exception as e:
                logger.error(f"[ERRO] Retreinamento falhou: {e}")
                state.process_status = 'error'
            finally:
                state.is_running = False
    
    background_tasks.add_task(run_retrain)
    return {"status": "started", "message": f"Retreinamento iniciado: {dataset_name}"}

@router.post("/train/stop")
def stop_training():
    state.stop_requested = True
    return {"status": "stop_requested", "message": "Interrupção de treinamento solicitada."}
