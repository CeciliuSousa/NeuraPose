import sys
import os
import psutil
import logging
import platform
import warnings
import atexit
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from colorama import init

# ==============================================================
# ENVIRONMENT SETUP
# ==============================================================
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

for p in [str(CURRENT_DIR), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ==============================================================
# PROJECT IMPORTS
# ==============================================================
try:
    import neurapose_backend.config_master as cm
    from neurapose_backend.globals.hardware_monitor import monitor as hw_monitor
    from neurapose_backend.globals.state import state
    from neurapose_backend.otimizador.cuda.gpu_utils import gpu_manager
    
    # Import Routes
    from neurapose_backend.routes import (
        system, config, processing, training, testing, 
        annotation, dataset
    )
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import project modules.\nImport failed: {e}")
    raise e

# ==============================================================
# LOGGING & INIT
# ==============================================================
init(strip=False)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeuraPoseAPI")
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

app = FastAPI(title="NeuraPose API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi.staticfiles import StaticFiles

# ==============================================================
# ROUTERS
# ==============================================================
# Mount Static Files for Video Playback
if cm.PROCESSING_OUTPUT_DIR.exists():
    app.mount("/videos", StaticFiles(directory=str(cm.PROCESSING_OUTPUT_DIR)), name="videos")
else:
    logger.warning(f"[STATIC] Video directory not found: {cm.PROCESSING_OUTPUT_DIR}")
app.include_router(system.router, tags=["System"])
app.include_router(config.router, tags=["Config"])
app.include_router(processing.router, tags=["Processing"])
app.include_router(training.router, tags=["Training"])
app.include_router(testing.router, tags=["Testing"])
app.include_router(annotation.router, tags=["Annotation"])
app.include_router(dataset.router, tags=["Dataset"])

# ==============================================================
# LIFECYCLE & PROCESS MANAGEMENT
# ==============================================================

def configurar_prioridade_alta():
    """Força o processo atual a ter Alta Prioridade no SO."""
    try:
        p = psutil.Process(os.getpid())
        system_os = platform.system()
        
        logger.info(f"[SYSTEM] Ajustando prioridade do processo PID {p.pid}...")
        
        if system_os == "Windows":
            p.nice(psutil.HIGH_PRIORITY_CLASS)
            logger.info("[SYSTEM] Prioridade definida para: ALTA (Windows)")
            
        elif system_os == "Linux":
            try:
                p.nice(-10)
                logger.info("[SYSTEM] Prioridade definida para: -10 (Linux)")
            except:
                logger.warning("[SYSTEM] Falha ao definir nice (requer sudo), rodando normal.")
                
    except Exception as e:
        logger.error(f"[SYSTEM] Erro ao ajustar prioridade: {e}")

@app.on_event("startup")
def startup_event():
    """Inicializa monitoramento, logs e prioridade."""
    configurar_prioridade_alta()
    hw_monitor.start()
    
    # Inicializa Otimizações GPU
    try:
        gpu_manager.enable_cudnn_benchmarking()
        if cm.USE_FP16:
            gpu_manager.enable_mixed_precision()
        logger.info("[GPU] Otimizações globais ativadas.")
    except Exception as e:
        logger.warning(f"[GPU] Falha ao configurar otimizações: {e}")

    logger.info("NeuraPose Backend Started Successfully (Modular)")

@atexit.register
def cleanup_on_exit():
    """Último recurso para garantir encerramento limpo."""
    logger.info("Cleaning up resources on exit...")
    try:
        hw_monitor.stop()
        state.kill_all_processes()
    except:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
