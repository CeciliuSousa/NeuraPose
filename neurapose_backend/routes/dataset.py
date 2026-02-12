import logging
import importlib
from pathlib import Path
from typing import Optional
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

class SplitRequest(BaseModel):
    input_dir_process: str
    dataset_name: str
    output_root: Optional[str] = None
    train_split: float = 0.7
    test_split: float = 0.2
    train_ratio: float = 0.8  # train/(train+val)

class ConvertRequest(BaseModel):
    dataset_path: str
    extension: str = ".pt"
    output_name: Optional[str] = None

# ==============================================================
# HELPERS
# ==============================================================

def check_folder_status(folder_path: Path, required_subfolders: list) -> dict:
    """Verifica status de completude de uma pasta."""
    if not folder_path.exists() or not folder_path.is_dir():
        return None
        
    existing = [sf for sf in required_subfolders if (folder_path / sf).exists()]
    has_videos = (folder_path / "videos").exists() and any((folder_path / "videos").glob("*.mp4"))
    has_jsons = (folder_path / "jsons").exists() and any((folder_path / "jsons").glob("*.json"))
    
    # Status: complete, partial, empty
    if len(existing) == len(required_subfolders) and has_videos and has_jsons:
        status = "complete"
    elif len(existing) > 0 or has_videos:
        status = "partial"
    else:
        status = "empty"
        
    return {
        "name": folder_path.name,
        "path": str(folder_path),
        "status": status,
        "subfolders": existing,
        "has_videos": has_videos,
        "has_jsons": has_jsons,
        "has_annotations": (folder_path / "anotacoes").exists(),
        "has_reid": (folder_path / "reid").exists(),
    }

# ==============================================================
# ENDPOINTS
# ==============================================================

@router.get("/datasets/list")
def list_all_datasets():
    """
    Lista todas as pastas de datasets organizadas por categoria:
    - processamentos: Vídeos processados (resultado_processamento/)
    - reidentificacoes: Datasets reidentificados (resultados-reidentificacoes/)
    - datasets: Datasets prontos para treino (datasets/)
    """
    result = {
        "processamentos": [],
        "reidentificacoes": [],
        "datasets": []
    }
    
    # 1. Processamentos
    proc_dir = cm.PROCESSING_OUTPUT_DIR
    if proc_dir.exists():
        for folder in proc_dir.iterdir():
            if folder.is_dir() and not folder.name.startswith('.'):
                info = check_folder_status(folder, ["videos", "jsons", "predicoes"])
                if info:
                    result["processamentos"].append(info)
    
    # 2. Reidentificações
    reid_dir = cm.REID_OUTPUT_DIR
    if reid_dir.exists():
        for folder in reid_dir.iterdir():
            if folder.is_dir() and not folder.name.startswith('.'):
                info = check_folder_status(folder, ["videos", "jsons", "predicoes"])
                if info:
                    result["reidentificacoes"].append(info)
    
    # 3. Datasets (para treinamento)
    datasets_dir = getattr(cm, 'DATASETS_DIR', cm.BACKEND_DIR / "datasets")
    if datasets_dir.exists():
        for folder in datasets_dir.iterdir():
            if folder.is_dir() and not folder.name.startswith('.'):
                has_train = (folder / "train").exists()
                has_test = (folder / "test").exists()
                result["datasets"].append({
                    "name": folder.name,
                    "path": str(folder),
                    "status": "complete" if has_train and has_test else "partial" if has_train or has_test else "empty",
                    "has_train": has_train,
                    "has_test": has_test,
                })
    
    return {"status": "success", "data": result}

@router.post("/dataset/split")
async def split_dataset(req: SplitRequest, background_tasks: BackgroundTasks):
    """Inicia a divisão do dataset para treino e teste em segundo plano."""
    from neurapose_backend.pre_processamento.split_dataset_label import run_split
    
    input_path = Path(req.input_dir_process).resolve()
    output_root = Path(req.output_root).resolve() if req.output_root else cm.TEST_DATASETS_ROOT.parent
    
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"Diretório de entrada não encontrado: {input_path}")
        
    def run_split_task():
        state.reset()
        state.is_running = True
        state.current_process = 'split'
        with CaptureOutput('split'):
            try:
                run_split(
                    root_path=input_path,
                    dataset_name=req.dataset_name,
                    output_root=output_root,
                    train_split=req.train_split,
                    test_split=req.test_split,
                    train_ratio=req.train_ratio,
                    logger=logger
                )
                state.process_status = 'success'
            except Exception as e:
                logger.error(f"Erro no split de dataset: {e}")
                state.process_status = 'error'
            finally:
                state.is_running = False
                
    background_tasks.add_task(run_split_task)
    return {"status": "started", "message": f"Split iniciado para o dataset {req.dataset_name}"}

@router.post("/convert/pt")
async def convert_dataset_to_pt(req: ConvertRequest, background_tasks: BackgroundTasks):
    """Converte JSONs de anotações para formato PyTorch (.pt)."""
    dataset_path = Path(req.dataset_path).resolve()
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset não encontrado: {dataset_path}")
    
    base_search = dataset_path / "treino" if (dataset_path / "treino").exists() else dataset_path
    jsons_dir = base_search / "dados" if (base_search / "dados").exists() else base_search / "jsons"
    annotations_dir = base_search / "anotacoes"
    labels_path = annotations_dir / "labels.json"
    
    if not jsons_dir.exists():
        raise HTTPException(status_code=400, detail=f"Pasta de JSONs ('dados' ou 'jsons') não encontrada em {base_search}")
    if not labels_path.exists():
        raise HTTPException(status_code=400, detail=f"labels.json não encontrado em {annotations_dir}")
    
    def run_conversion():
        state.reset()
        state.is_running = True
        state.current_process = 'convert'
        with CaptureOutput(category="convert"):
            try:
                original_jsons = cm.PROCESSING_JSONS_DIR
                original_labels = cm.PROCESSING_ANNOTATIONS_DIR
                
                if req.output_name:
                    base_dest = getattr(cm, 'DATASETS_DIR', dataset_path.parent)
                    dest_dataset_root = base_dest / req.output_name
                    out_dir = dest_dataset_root / "treino" / "data"
                    dataset_name = req.output_name
                    logger.info(f"[CONVERTE] Criando Novo Dataset em: {dest_dataset_root}")
                else:
                    dataset_name = dataset_path.name
                    out_dir = (dataset_path / "treino" / "data")
                
                out_dir.resolve().mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"data{req.extension}"
                
                import neurapose_backend.pre_processamento.converte_pt as cpt
                importlib.reload(cpt) # Força reload para resetar globais se necessário
                
                # Injection de dependência "suja" mas necessária dada a arquitetura do script original
                cpt.JSONS_DIR = jsons_dir.resolve()
                cpt.LABELS_PATH = labels_path.resolve()
                cpt.OUT_PT = out_file.resolve()
                cpt.LOG_FILE = (out_dir / "frames_invalidos.txt").resolve()
                cpt.DEBUG_LOG = (out_dir / "debug_log.txt").resolve()
                
                logger.info(f"[CONVERTE] Iniciando conversão para {dataset_name}")
                cpt.main()
                
                state.process_status = 'success'
                logger.info(f"[OK] Conversão concluída com sucesso!")
                
            except Exception as e:
                logger.error(f"[ERRO] Conversão falhou: {e}")
                state.process_status = 'error'
            finally:
                state.is_running = False
                # Restaurar se necessário (embora thread morra)
    
    background_tasks.add_task(run_conversion)
    return {"status": "started", "message": f"Conversão iniciada para {dataset_path.name}"}
