import sys
import os
from pathlib import Path
import logging
import json
import shutil
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import psutil
import torch

# ==============================================================
# SETUP PATHS
# ==============================================================
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

# Adiciona o diretório atual e o ROOT ao sys.path
for p in [str(CURRENT_DIR), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.log_service import LogBuffer, CaptureOutput
from app.user_config_manager import UserConfigManager


# ==============================================================
# IMPORTS DO PROJETO
# ==============================================================
# Importamos diretamente do config_master que está na mesma pasta
try:
    import config_master as cm
    from pre_processamento.pipeline.processador import processar_video
    from pre_processamento.utils.ferramentas import carregar_sessao_onnx, imprimir_banner

    # Import logic for Manual ReID
    from pre_processamento.reid_manual import (
        aplicar_processamento_completo, 
        renderizar_video_limpo, 
        carregar_pose_records,
        salvar_json,
        indexar_por_frame_e_contar_ids
    )
    # Import logic for Training
    from LSTM.pipeline import treinador
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import project modules.\\nImport failed: {e}")
    raise e

# Forçamos o reload do config_master caso ele tenha sido alterado em disco
import importlib
importlib.reload(cm)


# ==============================================================
# LOGGING
# ==============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeuraPoseAPI")

# ==============================================================
# SEGURANÇA: Pastas e arquivos restritos no explorador
# ==============================================================
HIDDEN_ENTRIES = {
    "app", "pre_processamento", "detector", "LSTM", "tracker", 
    "neurapose_backend", "neurapose_frontend", "node_modules", 
    ".git", ".next", ".venv", "__pycache__", ".agent", ".gemini", 
    "config_master.py", "main.py", "picker.py", "agente.txt", 
    "user_settings.json", "package.json", "package-lock.json", 
    "tsconfig.json", "next.config.js", ".gitignore", "README.md"
}

app = FastAPI(title="NeuraPose API", version="1.0.0")

# ==============================================================
# CORS
# ==============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compressão GZIP para respostas mais rápidas
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=500)

# ==============================================================
# MODELS
# ==============================================================
class ProcessRequest(BaseModel):
    input_path: str
    dataset_name: Optional[str] = None  # Nome do dataset (se vazio, usa nome da pasta de entrada)
    onnx_path: Optional[str] = None
    show_preview: bool = False
    device: str = "cuda"  # "cuda" ou "cpu"


class TrainRequest(BaseModel):
    epochs: int
    batch_size: int
    learning_rate: float
    model_name: str
    dataset_name: str
    temporal_model: str = "tft" # tft or lstm

class ReIDRule(BaseModel):
    src: int
    tgt: int
    start: int
    end: int

class ReIDDelete(BaseModel):
    id: int
    start: int
    end: int

class ReIDCut(BaseModel):
    start: int
    end: int

class ReIDApplyRequest(BaseModel):
    rules: List[ReIDRule] = []
    deletions: List[ReIDDelete] = []
    cuts: List[ReIDCut] = []
    action: str = "process" # 'process' or 'delete'

class AnnotationRequest(BaseModel):
    video_stem: str
    annotations: Dict[str, str]  # { "id_persistente": "classe" }
    output_path: Optional[str] = None

class SplitRequest(BaseModel):
    input_dir_process: str
    dataset_name: str
    output_root: Optional[str] = None
    train_split: str = "treino"
    test_split: str = "teste"

class TestRequest(BaseModel):
    model_path: Optional[str] = None
    dataset_path: Optional[str] = None
    device: str = "cuda"


# ==============================================================
# GPU MEMORY MANAGEMENT
# ==============================================================
import gc

def get_gpu_memory_info() -> Dict[str, Any]:
    """Retorna informações detalhadas sobre memória GPU."""
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


def cleanup_gpu_memory(force: bool = False) -> Dict[str, Any]:
    """Libera memória GPU não utilizada.
    
    Args:
        force: Se True, força limpeza agressiva incluindo cache do garbage collector
    """
    if not torch.cuda.is_available():
        return {"success": False, "message": "CUDA não disponível"}
    
    before = get_gpu_memory_info()
    
    try:
        # Limpa cache do PyTorch
        torch.cuda.empty_cache()
        
        if force:
            # Força coleta de lixo Python
            gc.collect()
            torch.cuda.empty_cache()
            
            # Sincroniza para garantir liberação
            torch.cuda.synchronize()
        
        after = get_gpu_memory_info()
        freed = before.get("reserved_gb", 0) - after.get("reserved_gb", 0)
        
        return {
            "success": True,
            "freed_gb": round(freed, 3),
            "before": before,
            "after": after,
            "message": f"Liberado {freed:.3f} GB de memória GPU"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def ensure_gpu_memory(required_gb: float = 2.0) -> bool:
    """Verifica se há memória GPU suficiente, tentando liberar se necessário.
    
    Args:
        required_gb: Quantidade mínima de memória livre necessária em GB
    
    Returns:
        True se memória suficiente está disponível, False caso contrário
    """
    if not torch.cuda.is_available():
        return False
    
    info = get_gpu_memory_info()
    if info.get("free_gb", 0) >= required_gb:
        return True
    
    # Tenta liberar memória
    cleanup_gpu_memory(force=True)
    
    info = get_gpu_memory_info()
    return info.get("free_gb", 0) >= required_gb

# ==============================================================



# HELPERS
# ==============================================================
from app.state import state

# ==============================================================
# RUNTIME CONFIGURATION (In-memory, resets on restart)
# ==============================================================
# Inicializa RUNTIME_CONFIG com as configurações persistentes do usuário
RUNTIME_CONFIG = UserConfigManager.load_config()

# Garante que caminhos críticos de config_master sejam mantidos se não estiverem no JSON
# (Caminhos como ROOT, MODELS_DIR etc são calculados dinamicamente no config_master)
if "ROOT" not in RUNTIME_CONFIG:
    RUNTIME_CONFIG["ROOT"] = str(cm.ROOT)

# Sincroniza o BOT_SORT_CONFIG do config_master com o que foi carregado no RUNTIME_CONFIG
for k in cm.BOT_SORT_CONFIG:
    if k in RUNTIME_CONFIG:
        cm.BOT_SORT_CONFIG[k] = RUNTIME_CONFIG[k]


def get_all_cm_config():
    return RUNTIME_CONFIG

def update_cm_runtime(updates: Dict[str, Any], persist: bool = True):
    global RUNTIME_CONFIG
    RUNTIME_CONFIG.update(updates)
    
    # Sincroniza BOT_SORT_CONFIG
    for k, v in updates.items():
        if k in cm.BOT_SORT_CONFIG:
            cm.BOT_SORT_CONFIG[k] = v
            
    if persist:
        # Salva apenas as variáveis que queremos persistir (limpa caminhos dinâmicos se houver)
        UserConfigManager.save_config(RUNTIME_CONFIG)


@app.post("/config/reset")
def api_reset_config():
    global RUNTIME_CONFIG
    import importlib
    importlib.reload(cm)
    
    # Remove o arquivo user_settings.json
    UserConfigManager.reset_to_defaults()
    
    # Recarrega o RUNTIME_CONFIG com os padrões do config_master
    RUNTIME_CONFIG = UserConfigManager.get_default_config()
    
    # Sincroniza BOT_SORT_CONFIG novamente
    for k in cm.BOT_SORT_CONFIG:
        if k in RUNTIME_CONFIG:
            cm.BOT_SORT_CONFIG[k] = RUNTIME_CONFIG[k]
            
    return {"status": "success", "message": "Configurações resetadas para os padrões originais."}


# ==============================================================
# HELPERS
# ==============================================================
def run_subprocess_processing(input_path: str, dataset_name: str, show: bool, device: str = "cuda"):
    """
    Executa o processamento via subprocess chamando:
    uv run python -m neurapose_backend.pre_processamento.processar --input-folder "..." [--show]
    
    Captura stdout/stderr em tempo real para o LogBuffer usando thread separada.
    """
    import subprocess
    import threading
    import queue
    
    state.reset()
    state.is_running = True
    log_buffer = LogBuffer()
    
    # Monta o comando
    cmd = [
        "uv", "run", "python", "-m", 
        "neurapose_backend.pre_processamento.processar",
        "--input-folder", input_path,
        "--output-root", str(cm.PROCESSING_OUTPUT_DIR / f"{dataset_name}-processado")
    ]
    
    if show:
        cmd.append("--show")
    
    logger.info(f"Executando comando: {' '.join(cmd)}")
    log_buffer.write(f"[CMD] {' '.join(cmd)}")
    
    try:
        # Executa o comando - SEM bufsize=1 para evitar overhead de I/O
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ROOT_DIR),
            encoding='utf-8',
            errors='replace',
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        )
        
        state.add_process(process)
        
        # Thread para ler output sem bloquear o processo
        output_queue = queue.Queue()
        
        def read_output():
            try:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        output_queue.put(line.rstrip())
                    if process.poll() is not None:
                        break
            except:
                pass
            finally:
                output_queue.put(None)  # Sinaliza fim
        
        reader_thread = threading.Thread(target=read_output, daemon=True)
        reader_thread.start()
        
        # Processa output da queue sem bloquear o subprocess
        while True:
            try:
                line = output_queue.get(timeout=0.1)
                if line is None:  # Fim do output
                    break
                log_buffer.write(line)
            except queue.Empty:
                # Verifica se deve parar
                if state.stop_requested:
                    process.terminate()
                    log_buffer.write("[INFO] Processamento interrompido pelo usuário.")
                    break
                # Verifica se processo terminou
                if process.poll() is not None:
                    # Drena output restante
                    while not output_queue.empty():
                        try:
                            line = output_queue.get_nowait()
                            if line:
                                log_buffer.write(line)
                        except:
                            break
                    break
        
        process.wait()
        
        if process.returncode == 0:
            log_buffer.write("[OK] Processamento concluído com sucesso!")
            logger.info("Processamento concluído com sucesso.")
        elif not state.stop_requested:
            log_buffer.write(f"[ERRO] Processo finalizou com código: {process.returncode}")
            logger.error(f"Processo finalizou com código: {process.returncode}")
            
    except Exception as e:
        logger.error(f"Erro ao executar subprocess: {e}")
        log_buffer.write(f"[ERRO] {str(e)}")
    finally:
        state.reset()
        # Limpa memória GPU após processamento
        cleanup_result = cleanup_gpu_memory(force=True)
        if cleanup_result.get("success"):
            logger.info(f"Memória GPU liberada: {cleanup_result.get('freed_gb', 0):.2f} GB")


# Mantém a função antiga para compatibilidade (pode ser removida futuramente)
def run_processing_task(input_path: Path, output_path: Path, onnx_path: Path, show: bool, device: str = "cuda"):
    state.reset()
    state.is_running = True
    
    # Atualiza o dispositivo no config_master para esta sessão
    cm.DEVICE = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"

    
    # Wrap execution to capture stdout/stderr to LogBuffer
    with CaptureOutput():
        # Imprime banner como faz o processar.py
        imprimir_banner(onnx_path)
        
        logger.info(f"Iniciando processamento: {input_path} -> {output_path}")
        
        try:
            sess, input_name = carregar_sessao_onnx(str(onnx_path))
            
            if input_path.is_file():
                v_name = input_path.stem
                final_out = output_path
                if final_out == Path(cm.PROCESSING_OUTPUT_DIR):
                     final_out = output_path / v_name
                print(f"[INFO] Processando 1 vídeo: {input_path.name}")
                processar_video(input_path, sess, input_name, final_out, show=show)
            elif input_path.is_dir():
                videos = sorted(input_path.glob("*.mp4"))
                output_path.mkdir(parents=True, exist_ok=True)
                print(f"[INFO] Encontrados {len(videos)} vídeos em {input_path}")
                for i, v in enumerate(videos, 1):
                    if state.stop_requested: break
                    print(f"\n[{i}/{len(videos)}] Processando: {v.name}")
                    processar_video(v, sess, input_name, output_path, show=show)

            
            if state.stop_requested:
                logger.info("Processamento interrompido pelo usuario.")
            else:
                logger.info("Processamento concluido.")
        except Exception as e:
            logger.error(f"Erro no processamento: {e}")
            print(f"Erro critico: {e}")
        finally:
            state.reset()

def run_training_task(req: TrainRequest):
    """Executa o pipeline de treinamento em segundo plano."""
    state.reset()
    state.is_running = True
    
    with CaptureOutput():
        logger.info(f"Iniciando treinamento: {req.model_name} (Dataset: {req.dataset_name})")
        try:
            from LSTM.pipeline.treinador import main as start_train
            # NOTA: O treinador atual lê configurações via get_config().
            # Para integração total, precisaríamos que o treinador aceitasse os args do req.
            # Por enquanto, vamos atualizar o RUNTIME_CONFIG que o treinador pode vir a ler.
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
        except Exception as e:
            logger.error(f"Erro no treinamento: {e}")
        finally:
            state.reset()

def run_testing_task(req: Dict[str, Any]):
    """Executa o pipeline de teste em segundo plano."""
    state.reset()
    state.is_running = True
    
    with CaptureOutput():
        logger.info("Iniciando fase de testes e validação...")
        try:
            from app.testar_modelo import main as start_test
            # Similar ao treino, o teste lê de configurações globais
            start_test()
            logger.info("Fase de testes concluída.")
        except Exception as e:
            logger.error(f"Erro nos testes: {e}")
        finally:
            state.reset()



# ==============================================================
# ENDPOINTS
# ==============================================================

@app.get("/")
def root():
    return {"status": "ok", "system": "NeuraPose API"}

@app.get("/logs")
def get_logs():
    """Retorna logs do backend."""
    return {"logs": LogBuffer().get_logs()}

@app.delete("/logs")
def clear_logs():
    """Limpa o buffer de logs do servidor."""
    LogBuffer().clear()
    return {"status": "logs cleared"}

@app.get("/health")
def health_check():
    """Ultra-fast health check - sem I/O."""
    return {
        "status": "healthy",
        "system": "NeuraPose API",
        "version": "1.0.0",
        "device": "gpu" if cm.DEVICE == "cuda" else "cpu",
        "processing": state.is_running,
        "paused": state.is_paused
    }

# Cache para system info (evita chamadas pesadas)
_system_info_cache = {"data": None, "timestamp": 0}
_CACHE_TTL = 2  # segundos

@app.get("/system/info")
def get_system_info():
    """Retorna informações de uso de hardware (CPU, RAM, GPU).
    
    Usa PyTorch para GPU e cache de 2s para otimização.
    """
    import time
    
    # Verifica cache
    now = time.time()
    if _system_info_cache["data"] and (now - _system_info_cache["timestamp"]) < _CACHE_TTL:
        return _system_info_cache["data"]
    
    # CPU sem interval (instantâneo, usa cache interno do psutil)
    cpu_usage = psutil.cpu_percent(interval=0)
    ram = psutil.virtual_memory()
    
    gpu_name = ""
    gpu_mem_used = 0.0
    gpu_mem_total = 0.0
    
    # GPU - usa mem_get_info para uso GLOBAL da GPU (não apenas PyTorch)
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            # mem_get_info retorna (free, total) em bytes
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            
            gpu_mem_total = total_mem / (1024**3)
            # Uso real = Total - Livre
            gpu_mem_used = (total_mem - free_mem) / (1024**3)
    except Exception:
        pass
    
    result = {
        "cpu_percent": cpu_usage,
        "ram_used_gb": ram.used / (1024**3),
        "ram_total_gb": ram.total / (1024**3),
        "gpu_mem_used_gb": gpu_mem_used,
        "gpu_mem_total_gb": gpu_mem_total,
        "gpu_name": gpu_name
    }
    
    # Atualiza cache
    _system_info_cache["data"] = result
    _system_info_cache["timestamp"] = now
    
    return result




@app.get("/config/all")
def api_get_all_config():
    return get_all_cm_config()

@app.get("/config")
def api_get_config():
    """Retorna configurações básicas e caminhos raiz para restrição do explorador."""
    return {
        "status": "success",
        "paths": {
            "videos": str(cm.PROCESSING_INPUT_DIR),
            "processamentos": str(cm.PROCESSING_OUTPUT_DIR),
            "reidentificacoes": str(cm.REID_OUTPUT_DIR),
            "datasets": str(cm.ROOT / "datasets"),
            "modelos": str(cm.TRAINED_MODELS_DIR),
            "anotacoes": str(cm.PROCESSING_ANNOTATIONS_DIR),
            "root": str(cm.ROOT)
        }
    }

@app.get("/browse")
def browse_path(path: str = "."):
    """Lista o conteúdo de uma pasta para o explorador de arquivos do frontend."""
    try:
        p = Path(path).resolve()
        
        if not p.exists():
            logger.warning(f"Caminho solicitado no /browse não existe: {p}. Fallback para videos.")
            p = Path(cm.PROCESSING_INPUT_DIR).resolve()
            if not p.exists(): p = Path.cwd()
        
        # Por segurança, podemos limitar o browse ao ROOT_DIR ou drives específicos
        # Para este projeto, vamos permitir navegar livremente, mas com cautela
        
        items = []
        for entry in p.iterdir():
            if entry.name in HIDDEN_ENTRIES:
                continue
            items.append({
                "name": entry.name,
                "path": str(entry.absolute()),
                "is_dir": entry.is_dir()
            })
            
        # Ordenar: pastas primeiro, depois nomes
        items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))

        
        return {
            "current": str(p.absolute()),
            "parent": str(p.parent.absolute()) if p.parent != p else str(p.absolute()),
            "items": items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ls")
def list_directory(path: Optional[str] = None):
    """Lista diretórios para o explorador de arquivos customizado."""
    try:
        if path:
            root = Path(path).resolve()
        else:
            root = Path.cwd()
            
        # Se o caminho não existe, tenta usar o diretório pai ou o workspace root
        if not root.exists():
            logger.warning(f"Caminho solicitado não existe: {root}. Tentando fallback.")
            root = Path.cwd()

        items = []
        for item in root.iterdir():
            if item.name in HIDDEN_ENTRIES:
                continue
            if item.is_dir():
                items.append({
                    "name": item.name,
                    "path": str(item),
                    "isDir": True
                })

        return {"items": sorted(items, key=lambda x: x['name']), "current": str(root)}
    except Exception as e:
        logger.error(f"Erro no /ls: {e}")
        # Retorna o diretório atual em caso de erro crítico para não travar a UI
        return {"items": [], "current": str(Path.cwd()), "error": str(e)}

@app.post("/config/update")
def api_update_config(updates: Dict[str, Any]):
    try:
        update_cm_runtime(updates)
        return {"status": "success", "message": "Configurações atualizadas na memória."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================
# GPU MEMORY ENDPOINTS
# ==============================================================

@app.get("/gpu/info")
def api_gpu_info():
    """Retorna informações detalhadas sobre a memória GPU."""
    return get_gpu_memory_info()


@app.post("/gpu/cleanup")
def api_gpu_cleanup(force: bool = True):
    """Libera memória GPU não utilizada.
    
    Args:
        force: Se True (padrão), força limpeza agressiva incluindo garbage collector.
    """
    return cleanup_gpu_memory(force=force)


@app.post("/gpu/optimize")
def api_gpu_optimize(required_gb: float = 2.0):
    """Tenta garantir quantidade mínima de memória GPU disponível.
    
    Args:
        required_gb: Quantidade mínima de memória livre necessária em GB (padrão: 2.0)
    
    Returns:
        Status indicando se a memória necessária está disponível.
    """
    success = ensure_gpu_memory(required_gb=required_gb)
    info = get_gpu_memory_info()
    return {
        "success": success,
        "required_gb": required_gb,
        "available_gb": info.get("free_gb", 0),
        "message": "Memória suficiente disponível" if success else f"Não foi possível liberar {required_gb}GB"
    }


@app.post("/shutdown")
def shutdown_server():
    """Finaliza o servidor e todos os processos filhos à força."""
    logger.info("Encerrando servidor...")
    state.kill_all_processes()
    
    # Use a thread to exit after returning response
    import threading
    import os
    import signal
    
    def kill_me():
        import time
        time.sleep(0.5)
        # Força saída do processo Python
        os.kill(os.getpid(), signal.SIGTERM)
        sys.exit(0)
        
    threading.Thread(target=kill_me).start()
    return {"status": "shutting_down"}


# Endpoint /video_feed removido - preview via temp_preview.jpg foi descontinuado


@app.post("/process/stop")
def stop_process():
    state.stop_requested = True
    return {"status": "stop_requested"}

@app.post("/process/pause")
def pause_process():
    state.is_paused = True
    return {"status": "paused"}

@app.post("/process/resume")
def resume_process():
    state.is_paused = False
    return {"status": "resumed"}


@app.post("/process")
async def start_processing(req: ProcessRequest, background_tasks: BackgroundTasks):
    """Inicia processamento de video(s) via subprocess."""
    inp = Path(req.input_path)
    
    if not inp.exists():
        raise HTTPException(status_code=404, detail="Pasta de entrada não encontrada")
    
    # Calcula nome do dataset: se não fornecido, usa o nome da pasta de entrada
    if req.dataset_name and req.dataset_name.strip():
        dataset_name = req.dataset_name.strip()
    else:
        # Usa o nome da última pasta do caminho de entrada
        dataset_name = inp.name if inp.is_dir() else inp.stem
    
    # Garante que o sufixo -processado não seja duplicado
    if dataset_name.endswith("-processado"):
        dataset_name = dataset_name[:-11]  # Remove o sufixo se já existir
    
    output_dir = cm.PROCESSING_OUTPUT_DIR / f"{dataset_name}-processado"
    
    background_tasks.add_task(
        run_subprocess_processing, 
        str(inp), 
        dataset_name, 
        req.show_preview, 
        req.device
    )
    return {
        "status": "started", 
        "detail": f"Processando {inp.name} -> {output_dir.name}",
        "output_dir": str(output_dir)
    }

@app.post("/test")
async def start_testing(req: TestRequest, background_tasks: BackgroundTasks):
    """Inicia fase de testes."""
    # Atualiza dispositivo se necessário
    if req.device:
        cm.DEVICE = req.device if (req.device == "cpu" or torch.cuda.is_available()) else "cpu"
        
    background_tasks.add_task(run_testing_task, req.dict())
    return {"status": "started", "message": "Fase de testes iniciada."}


@app.post("/train")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """Inicia treinamento."""
    background_tasks.add_task(run_training_task, req)
    return {"status": "started", "detail": f"Training {req.model_name}"}

# ==============================================================
# RE-ID MANUAL ENDPOINTS
# ==============================================================

@app.get("/reid/list")
def list_reid_candidates(root_path: Optional[str] = None):
    """Lista videos disponiveis para limpeza (com base na pasta jsons)."""
    try:
        # Resolve path safely
        if root_path:
            root = Path(root_path).resolve()
        else:
            root = cm.PROCESSING_OUTPUT_DIR

        # Se usuário selecionou pasta 'predicoes', sobe um nível
        if root.name == "predicoes" or root.name == "jsons" or root.name == "videos":
            root = root.parent
            
        json_dir = root / "jsons"
        
        if not json_dir.exists():
            return {"videos": [], "root": str(root)}
            
        jsons = sorted(json_dir.glob("*.json"))
        videos = []
        for j in jsons:
            if "_tracking" in j.name: continue
            stem = j.stem
            # Check if video exists
            video_path = root / "videos" / f"{stem.replace('_pose', '')}.mp4"
            if not video_path.exists():
                video_path = root / "videos" / f"{stem}.mp4" # fallback
            
            videos.append({
                "id": stem,
                "json_path": str(j),
                "video_path": str(video_path) if video_path.exists() else None,
                "processed": False 
            })
        return {"videos": videos, "root": str(root)}
    except Exception as e:
        logger.error(f"Erro ao listar videos ReID: {e}")
        return {"videos": [], "error": str(e)}

@app.get("/reid/{video_id}/data")
def get_reid_data(video_id: str, root_path: Optional[str] = None):
    """Retorna dados de poses para o editor."""
    root = Path(root_path).resolve() if root_path else cm.PROCESSING_OUTPUT_DIR
    if root.name in ["predicoes", "jsons", "videos"]: root = root.parent
    
    # Prioriza _tracking.json (com IDs persistentes corretos) sobre o .json (keypoints)
    tracking_path = root / "jsons" / f"{video_id}_tracking.json"
    keypoints_path = root / "jsons" / f"{video_id}.json"
    
    # Decide qual arquivo usar
    use_tracking = tracking_path.exists()
    json_path = tracking_path if use_tracking else keypoints_path
    
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="JSON not found")
    
    raw_data = carregar_pose_records(json_path)
    
    # Handle tracking.json format (object with tracking_by_frame)
    if use_tracking and isinstance(raw_data, dict) and "tracking_by_frame" in raw_data:
        # Converte o formato tracking para lista de records para reusar indexar_por_frame_e_contar_ids
        records = []
        for frame_idx, detections in raw_data["tracking_by_frame"].items():
            for det in detections:
                records.append({
                    "frame": int(frame_idx),
                    "id_persistente": det.get("id_persistente", det.get("botsort_id", -1)),
                    "bbox": det.get("bbox", [0,0,0,0]),
                    "confidence": det.get("confidence", 0)
                })
        
        # Use id_map for accurate unique ID count
        id_map = raw_data.get("id_map", {})
        unique_persistent_ids = set(id_map.values()) if id_map else set()
    else:
        records = raw_data
        unique_persistent_ids = None  # Will be calculated from records
    
    frames_index, id_counter = indexar_por_frame_e_contar_ids(records)
    
    # Converte frames_index para formato JSON-friendly
    frames_clean = {}
    for f, items in frames_index.items():
        frames_clean[str(f)] = []
        for bbox, gid in items:
            frames_clean[str(f)].append({"bbox": bbox, "id": gid})
    
    # For tracking files, use id_map count; otherwise use counter
    if unique_persistent_ids is not None:
        id_counts = {str(pid): id_counter.get(pid, 0) for pid in unique_persistent_ids}
    else:
        id_counts = dict(id_counter.most_common(100))
            
    return {
        "video_id": video_id,
        "frames": frames_clean,
        "id_counts": id_counts
    }

@app.get("/reid/video/{video_id}")
def stream_video(video_id: str, root_path: Optional[str] = None):
    """Serve o arquivo de video."""
    root = Path(root_path).resolve() if root_path else cm.PROCESSING_OUTPUT_DIR
    if root.name in ["predicoes", "jsons", "videos"]: root = root.parent
    
    # Prioridade: Vídeo com correções/marcações (_pose.mp4 na pasta predicoes)
    v_final = None
    
    # 1. Tenta achar na pasta de predicoes (com esqueleto)
    possible_preds = [
        root / "predicoes" / f"{video_id}_pose.mp4",
        root / "predicoes" / f"{video_id}.mp4"
    ]
    
    for p in possible_preds:
        if p.exists():
            v_final = p
            break
            
    # 2. Se não achar, tenta o original (fallback, mas sem marcações)
    if not v_final:
        possible_sources = [
            root / "videos" / f"{video_id.replace('_pose', '')}.mp4",
            root / "videos" / f"{video_id}.mp4"
        ]
        for p in possible_sources:
            if p.exists():
                v_final = p
                break
                
    if not v_final or not v_final.exists():
        raise HTTPException(status_code=404, detail="Video not found")
        
    return FileResponse(v_final, media_type="video/mp4")


# ==============================================================
# REID AGENDA - Persistência em arquivo JSON
# ==============================================================

REID_AGENDA_DIR = CURRENT_DIR / "resultados-reidentificacoes" / "reid"
REID_AGENDA_FILE = REID_AGENDA_DIR / "reid.json"

class ReidSwapRule(BaseModel):
    src_id: int
    tgt_id: int
    frame_start: int = 0
    frame_end: int = 999999

class ReidDeletionRule(BaseModel):
    id: int
    frame_start: int = 0
    frame_end: int = 999999

class ReidCutRule(BaseModel):
    frame_start: int
    frame_end: int

class ReidVideoEntry(BaseModel):
    video_id: str
    action: str = "process"  # "process" ou "delete"
    swaps: List[ReidSwapRule] = []
    deletions: List[ReidDeletionRule] = []
    cuts: List[ReidCutRule] = []

class ReidAgendaRequest(BaseModel):
    source_dataset: str
    video: ReidVideoEntry



# Helper para caminhos dinâmicos
def get_reid_paths_from_source(source_path_str: str):
    source = Path(source_path_str)
    # Se terminar em 'predicoes', sobe um nível para pegar o nome do dataset
    if source.name == 'predicoes':
        dataset_dir = source.parent
    else:
        dataset_dir = source
    
    # Define novo nome: dataset-reidentificado
    new_dataset_name = f"{dataset_dir.name}-reidentificado"
    
    # Salva em resultados-reidentificacoes/[dataset]-reidentificado
    output_root = CURRENT_DIR / "resultados-reidentificacoes" / new_dataset_name
    
    return {
        "root": output_root,
        "reid_dir": output_root / "reid",
        "agenda_file": output_root / "reid" / "reid.json",
        "videos_dir": output_root / "videos",
        "predictions_dir": output_root / "predicoes",
        "jsons_dir": output_root / "jsons"
    }

@app.get("/reid/agenda")
def get_reid_agenda(root_path: Optional[str] = None):
    """Carrega a agenda de ReID e calcula estatísticas de pendência."""
    target_file = None
    agenda = None
    
    if root_path:
        paths = get_reid_paths_from_source(root_path)
        target_file = paths["agenda_file"]
    else:
        target_file = REID_AGENDA_FILE

    # Tenta carregar agenda se existir
    if target_file and target_file.exists():
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                agenda = json.load(f)
        except Exception as e:
            logger.error(f"Erro ao ler agenda ReID: {e}")
            return {"agenda": None, "error": str(e)}

    # Calcula estatísticas
    stats = {"total": 0, "processed": 0, "pending": 0}
    
    if root_path:
        try:
            src_path = Path(root_path)
            if src_path.name == 'predicoes':
                src_root = src_path.parent
            else:
                src_root = src_path
            
            src_videos_dir = src_root / "videos"
            if src_videos_dir.exists():
                # Conta vídeos source (extensões válidas)
                valid_exts = {'.mp4', '.avi', '.mov', '.mkv'}
                # Glob recursivo ou simples? Simples.
                total_videos = sum(1 for f in src_videos_dir.glob("*") if f.suffix.lower() in valid_exts)
                stats["total"] = total_videos
                
                processed_count = 0
                excluded_count = 0
                if agenda and "videos" in agenda:
                    for v in agenda["videos"]:
                        act = v.get("action", "process")
                        if act == "delete":
                            excluded_count += 1
                        else:
                            processed_count += 1
                
                stats["processed"] = processed_count
                stats["excluded"] = excluded_count
                stats["pending"] = max(0, total_videos - (processed_count + excluded_count))
        except Exception as e:
            logger.warning(f"Erro ao calcular stats de ReID: {e}")

    return {"agenda": agenda, "stats": stats}


@app.post("/reid/agenda/save")
def save_reid_agenda(request: ReidAgendaRequest):
    """Salva/atualiza um vídeo na agenda de ReID com caminho dinâmico."""
    from datetime import datetime
    
    # Determina caminhos baseados no source_dataset
    if request.source_dataset:
        paths = get_reid_paths_from_source(request.source_dataset)
        target_dir = paths["reid_dir"]
        target_file = paths["agenda_file"]
    else:
        # Fallback para padrão
        target_dir = REID_AGENDA_DIR
        target_file = REID_AGENDA_FILE
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Carrega agenda existente ou cria nova
    if target_file.exists():
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                agenda = json.load(f)
        except:
            agenda = None
    else:
        agenda = None
    
    if agenda is None:
        agenda = {
            "version": "1.0",
            "source_dataset": request.source_dataset,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "videos": []
        }
    
    # Atualiza source_dataset se diferente
    if agenda.get("source_dataset") != request.source_dataset:
        agenda["source_dataset"] = request.source_dataset
        agenda["videos"] = []  # Limpa vídeos se dataset mudou
    
    # Procura vídeo existente na agenda
    video_data = request.video.model_dump()
    existing_idx = None
    for i, v in enumerate(agenda["videos"]):
        if v["video_id"] == video_data["video_id"]:
            existing_idx = i
            break
    
    # Atualiza ou adiciona
    if existing_idx is not None:
        agenda["videos"][existing_idx] = video_data
    else:
        agenda["videos"].append(video_data)
    
    agenda["updated_at"] = datetime.now().isoformat()
    
    # Salva arquivo
    try:
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(agenda, f, ensure_ascii=False, indent=2)
        return {"status": "success", "message": f"Vídeo '{request.video.video_id}' agendado com sucesso."}
    except Exception as e:
        logger.error(f"Erro ao salvar agenda ReID: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/reid/agenda/{video_id}")
def remove_from_agenda(video_id: str):
    """Remove um vídeo da agenda."""
    if not REID_AGENDA_FILE.exists():
        return {"status": "not_found"}
    
    try:
        with open(REID_AGENDA_FILE, "r", encoding="utf-8") as f:
            agenda = json.load(f)
        
        original_count = len(agenda.get("videos", []))
        agenda["videos"] = [v for v in agenda.get("videos", []) if v["video_id"] != video_id]
        
        with open(REID_AGENDA_FILE, "w", encoding="utf-8") as f:
            json.dump(agenda, f, ensure_ascii=False, indent=2)
        
        removed = original_count > len(agenda["videos"])
        return {"status": "removed" if removed else "not_found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reid/batch-apply")
async def batch_apply_reid(data: Dict[str, Any], background_tasks: BackgroundTasks):
    """Aplica mudanças em vários vídeos em lote."""
    videos = data.get("videos", []) # Lista de {video_id, rules, deletions, cuts}
    root_path = data.get("root_path")
    if root_path:
        r = Path(root_path).resolve()
        if r.name in ["predicoes", "jsons", "videos"]: 
            root_path = str(r.parent)
            
    output_path = data.get("output_path")
    
    if not videos:
        return {"status": "error", "message": "Nenhum vídeo fornecido."}
        
    def run_batch():
        state.is_running = True
        with CaptureOutput():
            try:
                for item in videos:
                    v_id = item["video_id"]
                    req = ReIDApplyRequest(**item)
                    logger.info(f"Processando ReID em lote para: {v_id}")
                    apply_reid_changes(v_id, req, root_path=root_path, output_path=output_path)
                logger.info("Processamento em lote concluído.")
            except Exception as e:
                logger.error(f"Erro no processamento em lote: {e}")
            finally:
                state.reset()
                
    background_tasks.add_task(run_batch)
    return {"status": "started", "count": len(videos)}

@app.post("/reid/{video_id}/apply")
def apply_reid_changes(video_id: str, req: ReIDApplyRequest, root_path: Optional[str] = None, output_path: Optional[str] = None):
    """Aplica as mudancas e gera novo video."""
    try:
        root = Path(root_path) if root_path else cm.PROCESSING_OUTPUT_DIR
        
        # Lógica de EXCLUSÃO DA FONTE
        if req.action == 'delete':
            deleted_files = []
            
            # Paths originais para deletar
            files_to_delete = [
                root / "jsons" / f"{video_id}.json",
                root / "videos" / f"{video_id.replace('_pose', '')}.mp4",
                root / "videos" / f"{video_id}.mp4",
                root / "predicoes" / f"{video_id}_pose.mp4",
                root / "predicoes" / f"{video_id}_tracking.mp4"
            ]
            
            for f in files_to_delete:
                if f.exists():
                    try:
                        os.remove(f)
                        deleted_files.append(f.name)
                    except Exception as e:
                        logger.error(f"Failed to delete {f}: {e}")
            
            return {"status": "deleted", "video_id": video_id, "deleted_files": deleted_files}

        if output_path:
            out_root = Path(output_path)
        else:
            # Padrão: usa REID_OUTPUT_DIR do config_master
            out_root = cm.REID_OUTPUT_DIR
        
        out_json_dir = out_root / "jsons"
        out_pred_dir = out_root / "predicoes"
        out_videos_dir = out_root / "videos" 
        
        for p in [out_json_dir, out_pred_dir, out_videos_dir]:
            p.mkdir(parents=True, exist_ok=True)
            
        # Paths
        json_path = root / "jsons" / f"{video_id}.json"
        v_raw = root / "videos" / f"{video_id.replace('_pose', '')}.mp4"
        if not v_raw.exists(): v_raw = root / "videos" / f"{video_id}.mp4"
        
        if not json_path.exists():
            raise HTTPException(status_code=404, detail=f"JSON não encontrado: {json_path}")
        if not v_raw.exists():
            raise HTTPException(status_code=404, detail=f"Vídeo de entrada não encontrado: {v_raw}")

        out_json_path = out_json_dir / f"{video_id}.json"
        out_v_pred = out_pred_dir / f"{video_id}_pose.mp4"
        
        # Carrega dados
        records = carregar_pose_records(json_path)
        
        # Converte models para formato de lista de dicts
        rules_list = [r.dict() for r in req.rules]
        delete_list = [d.dict() for d in req.deletions]
        cut_list = [c.dict() for c in req.cuts]
        
        # Aplica processamento lógico
        recs_mod, c_ids, d_ids, d_cuts = aplicar_processamento_completo(records, rules_list, delete_list, cut_list)
        
        # Salva JSON
        salvar_json(out_json_path, recs_mod)
        
        # Renderiza Video
        renderizar_video_limpo(v_raw, out_v_pred, recs_mod, cut_list)
        
        # Opcional: Copia o vídeo original se não existir na pasta de saída
        out_v_raw = out_videos_dir / v_raw.name
        if not out_v_raw.exists():
            import shutil
            shutil.copy2(v_raw, out_v_raw)
            
        return {
            "status": "success",
            "video_id": video_id,
            "swaps": c_ids,
            "deletions": d_ids,
            "cuts": d_cuts,
            "output_video": str(out_v_pred),
            "output_json": str(out_json_path)
        }
    except Exception as e:
        logger.error(f"Erro ao aplicar ReID para {video_id}: {e}")
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================
# ANNOTATION ENDPOINTS
# ==============================================================

@app.get("/annotate/list")
def list_videos_to_annotate(root_path: Optional[str] = None):
    """Lista vídeos disponíveis para anotação em resultados-reidentificacoes."""
    from pre_processamento.anotando_classes import listar_jsons, encontrar_video_para_json
    
    root = Path(root_path).resolve() if root_path else cm.REID_OUTPUT_DIR
    json_dir = root / "jsons"
    pred_dir = root / "predicoes"
    labels_path = cm.PROCESSING_ANNOTATIONS_DIR / "labels.json"
    
    if not json_dir.exists():
        return {"videos": []}
        
    labels_existentes = {}
    if labels_path.exists():
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                labels_existentes = json.load(f) or {}
        except: pass
        
    json_files = listar_jsons(json_dir)
    result = []
    
    for j in json_files:
        stem = j.stem
        v_path = encontrar_video_para_json(pred_dir, stem)
        status = "anotado" if stem in labels_existentes else "pendente"
        
        result.append({
            "video_id": stem,
            "status": status,
            "has_video": v_path is not None and v_path.exists(),
            "creation_time": j.stat().st_mtime
        })
        
    return {"videos": sorted(result, key=lambda x: x["creation_time"], reverse=True)}

@app.get("/annotate/{video_id}/details")
def get_annotation_details(video_id: str, root_path: Optional[str] = None):
    """Retorna detalhes dos IDs encontrados no vídeo para anotação."""
    from pre_processamento.anotando_classes import carregar_pose_records, indexar_por_frame
    
    root = Path(root_path).resolve() if root_path else cm.REID_OUTPUT_DIR
    json_path = root / "jsons" / f"{video_id}.json"
    
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="JSON não encontrado")
        
    records = carregar_pose_records(json_path)
    frames_index, id_counter = indexar_por_frame(records)
    
    # Prepara um sumário dos IDs para o front
    ids_info = []
    for gid, count in id_counter.items():
        if count >= cm.MIN_FRAMES_PER_ID:
            ids_info.append({
                "id": gid,
                "frames": count,
                "label": "desconhecido"
            })
            
    return {
        "video_id": video_id,
        "ids": ids_info,
        "total_frames": max(frames_index.keys()) if frames_index else 0
    }

@app.post("/annotate/save")
def save_annotations(req: AnnotationRequest):
    """Salva as anotações no arquivo labels.json global."""
    labels_path = cm.PROCESSING_ANNOTATIONS_DIR / "labels.json"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    
    todas_labels = {}
    if labels_path.exists():
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                todas_labels = json.load(f) or {}
        except: pass
        
    # As anotações vêm como { "ID": "classe" }
    # O formato do labels.json deve ser { "video_stem": { "ID": "classe", ... } }
    todas_labels[req.video_stem] = req.annotations
    
    try:
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(todas_labels, f, indent=4, ensure_ascii=False)
        return {"status": "success", "message": f"Anotações salvas para {req.video_stem}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar: {e}")

@app.post("/dataset/split")
async def split_dataset(req: SplitRequest, background_tasks: BackgroundTasks):
    """Inicia a divisão do dataset para treino e teste em segundo plano."""
    from pre_processamento.split_dataset_label import run_split
    
    input_path = Path(req.input_dir_process).resolve()
    output_root = Path(req.output_root).resolve() if req.output_root else cm.TEST_DATASETS_ROOT.parent
    
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"Diretório de entrada não encontrado: {input_path}")
        
    def run_split_task():
        state.is_running = True
        with CaptureOutput():
            try:
                run_split(
                    root_path=input_path,
                    dataset_name=req.dataset_name,
                    output_root=output_root,
                    train_split=req.train_split,
                    test_split=req.test_split
                )
            except Exception as e:
                logger.error(f"Erro no split de dataset: {e}")
            finally:
                state.reset()
                
    background_tasks.add_task(run_split_task)
    return {"status": "started", "message": f"Split iniciado para o dataset {req.dataset_name}"}


# Models e Endpoints ReID Batch
class ReidBatchApplyRequest(BaseModel):
    videos: Optional[List[Any]] = None
    root_path: str
    output_path: Optional[str] = None

def run_reid_batch_processing(source_dataset: str):
    logger.info(f"Iniciando Batch Process ReID: {source_dataset}")
    with CaptureOutput():
        try:
            paths = get_reid_paths_from_source(source_dataset)
            
            # 1. Cria diretórios
            for k in ["videos_dir", "predictions_dir", "jsons_dir"]:
                paths[k].mkdir(parents=True, exist_ok=True)
                
            # 2. Verifica Agenda
            if not paths["agenda_file"].exists():
                logger.error(f"Agenda não encontrada: {paths['agenda_file']}")
                return
                
            with open(paths["agenda_file"], "r", encoding="utf-8") as f:
                agenda = json.load(f)
                
            # 3. Define Origem
            src_path = Path(source_dataset)
            # Se apontar para .../predicoes, ajusta para root do dataset
            if src_path.name == 'predicoes':
                ds_root = src_path.parent
            else:
                ds_root = src_path
                
            src_videos = ds_root / "videos"
            src_jsons = ds_root / "jsons"
            
            # 4. Copia Vídeos Originais
            logger.info("Copiando vídeos base...")
            if src_videos.exists():
                for vfile in src_videos.glob("*"):
                    if vfile.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                        dst = paths["videos_dir"] / vfile.name
                        if not dst.exists():
                            shutil.copy2(vfile, dst)
                            
            # 5. Processa Vídeos da Agenda
            for vid_entry in agenda.get("videos", []):
                vid_id = vid_entry["video_id"]
                action = vid_entry.get("action", "process")
                
                if action == "delete":
                    continue 
                    
                # Mapeia regras
                swaps = [
                    {"src": s["src_id"], "tgt": s["tgt_id"], "start": s["frame_start"], "end": s["frame_end"]}
                    for s in vid_entry.get("swaps", [])
                ]
                deletions = [
                    {"id": d["id"], "start": d["frame_start"], "end": d["frame_end"]}
                    for d in vid_entry.get("deletions", [])
                ]
                cuts = [
                    {"start": c["frame_start"], "end": c["frame_end"]}
                    for c in vid_entry.get("cuts", [])
                ]
                
                # Carrega JSON original
                json_in = src_jsons / f"{vid_id}.json"
                if not json_in.exists():
                    logger.warning(f"JSON input não achado: {json_in}")
                    continue
                    
                with open(json_in, "r", encoding="utf-8") as f:
                    records = json.load(f)
                    
                # Aplica Lógica (Otimizada)
                processed, _, _, _ = aplicar_processamento_completo(records, swaps, deletions, cuts)
                
                # Salva JSON Filtrado
                json_out = paths["jsons_dir"] / f"{vid_id}.json"
                with open(json_out, "w", encoding="utf-8") as f:
                    json.dump(processed, f, ensure_ascii=False, indent=2)
                    
                # Gera Vídeo Anotado
                vid_in = paths["videos_dir"] / f"{vid_id}.mp4"
                if not vid_in.exists():
                    candidates = list(paths["videos_dir"].glob(f"{vid_id}.*"))
                    if candidates: vid_in = candidates[0]
                    
                if vid_in.exists():
                    vid_out = paths["predictions_dir"] / f"{vid_id}.mp4"
                    logger.info(f"Gerando vídeo anotado: {vid_out}")
                    renderizar_video_limpo(str(vid_in), str(vid_out), processed, cuts)
                else:
                    logger.warning(f"Vídeo base não achado: {vid_id}")
                    
            logger.info("Batch ReID finalizado.")

        except Exception as e:
            logger.error(f"Erro critical batch reid: {e}")

@app.post("/reid/batch-apply")
def batch_apply_reid(request: ReidBatchApplyRequest, background_tasks: BackgroundTasks):
    """Executa o processamento em massa das correções de ReID."""
    background_tasks.add_task(run_reid_batch_processing, request.root_path)
    return {"status": "started", "message": "Iniciando processamento ReID em background."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

