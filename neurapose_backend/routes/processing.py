import time
import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from colorama import Fore

import neurapose_backend.config_master as cm
from neurapose_backend.globals.state import state
from neurapose_backend.nucleo.log_service import CaptureOutput
from neurapose_backend.pre_processamento.utils.ferramentas import format_seconds_to_hms, imprimir_banner
from neurapose_backend.pre_processamento.pipeline.processador import processar_video
from neurapose_backend.otimizador.cuda import gpu_utils as gpu_opt

router = APIRouter()
logger = logging.getLogger("NeuraPoseAPI")

# ==============================================================
# MODELS
# ==============================================================

class ProcessRequest(BaseModel):
    input_path: str
    dataset_name: Optional[str] = None
    onnx_path: Optional[str] = None
    show_preview: bool = False
    device: str = "cuda"

# ==============================================================
# HELPERS (Moved from main.py)
# ==============================================================

def run_processing_thread(input_path: Path, output_path: Path, onnx_path: Path, show: bool, device: str = "cuda"):
    state.reset()
    state.is_running = True
    state.current_process = 'process'
    state.process_status = 'processing'
    state.show_preview = show
    
    cm.DEVICE = device if (device == "cpu" or cm.torch.cuda.is_available()) else "cpu"

    with CaptureOutput(category="process"):
        imprimir_banner(onnx_path)
        
        try:
            if input_path.is_file():
                v_name = input_path.stem
                final_out = output_path
                if final_out == Path(cm.PROCESSING_OUTPUT_DIR):
                     final_out = output_path / v_name
                processar_video(input_path, final_out, show=show)
                
            elif input_path.is_dir():
                videos = sorted(input_path.glob("*.mp4"))
                output_path.mkdir(parents=True, exist_ok=True)
                print(f"[INFO] ENCONTRADOS {len(videos)} VIDEOS")

                videos_to_process = []
                processed_count = 0
                props_dir = output_path / "predicoes"
                jsons_dir = output_path / "jsons"

                for v in videos:
                    is_processed = False
                    if props_dir.exists() and any(props_dir.glob(f"{v.stem}*pose.mp4")):
                        is_processed = True
                    if not is_processed and jsons_dir.exists():
                        if any(jsons_dir.glob(f"{v.stem}*tracking.json")): is_processed = True
                        if any(jsons_dir.glob(f"{v.stem}*{cm.FPS_TARGET}fps.json")): is_processed = True
                    
                    if is_processed:
                        processed_count += 1
                    else:
                        videos_to_process.append(v)

                if processed_count > 0:
                    print(Fore.YELLOW + f"[INFO] PASTA DE SAÍDA ENCONTRADA COM {processed_count} VIDEOS PROCESSADOS")
                
                print(Fore.CYAN + f"[INFO] PROCESSANDO OS {len(videos_to_process)} VIDEOS NÃO PROCESSADOS")
                
                start_time_total = time.time()
                for i, v in enumerate(videos_to_process, 1):
                    if state.stop_requested: break
                    print(f"\n[{i}/{len(videos_to_process)}] PROCESSANDO: {v.name}")
                    processar_video(v, output_path, show=show)
                    state.current_frame = None

            if state.stop_requested:
                logger.info("Processamento interrompido pelo usuario.")
                state.process_status = 'idle'
            else:
                elapsed_total = time.time() - start_time_total if 'start_time_total' in locals() else 0
                if 'videos_to_process' in locals() and len(videos_to_process) > 0:
                     print(Fore.CYAN + f"\n[INFO] TEMPO TOTAL DE PROCESSAMENTO DOS {len(videos_to_process)} VIDEOS: {format_seconds_to_hms(elapsed_total)}")
                
                print(Fore.GREEN + "[OK] FINALIZANDO O PROGRAMA DE PROCESSAMENTO..." + Fore.RESET)
                state.process_status = 'success'
        except Exception as e:
            logger.error(f"Erro no processamento: {e}")
            state.process_status = 'error'
        finally:
            state.is_running = False
            state.current_frame = None
            gpu_opt.clear_gpu_cache(force=True)

# ==============================================================
# ENDPOINTS
# ==============================================================

@router.post("/process")
async def start_processing(req: ProcessRequest, background_tasks: BackgroundTasks):
    """Inicia processamento de video(s) via subprocess (thread)."""
    inp = Path(req.input_path)
    
    if not inp.exists():
        raise HTTPException(status_code=404, detail="Pasta de entrada não encontrada")
    
    if req.dataset_name and req.dataset_name.strip():
        dataset_name = req.dataset_name.strip()
    else:
        dataset_name = inp.name if inp.is_dir() else inp.stem
    
    if dataset_name.endswith("-processado"):
        dataset_name = dataset_name[:-11]
    
    output_dir = cm.PROCESSING_OUTPUT_DIR / f"{dataset_name}-processado"
    onnx_path = Path(req.onnx_path) if req.onnx_path else Path(cm.RTMPOSE_PATH)
    
    background_tasks.add_task(
        run_processing_thread, 
        inp, 
        output_dir, 
        onnx_path,
        req.show_preview, 
        req.device
    )
    return {
        "status": "started", 
        "detail": f"Processando {inp.name} -> {output_dir.name}",
        "output_dir": str(output_dir)
    }

@router.post("/process/stop")
def stop_process():
    state.stop_requested = True
    return {"status": "stop_requested"}

@router.post("/process/pause")
def pause_process():
    state.is_paused = True
    return {"status": "paused"}

@router.post("/process/resume")
def resume_process():
    state.is_paused = False
    return {"status": "resumed"}
