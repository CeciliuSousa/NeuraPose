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
        indexar_por_frame_e_contar_ids,
        renderizar_video_cortado_raw
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
    root_path: str  # Pasta raiz do dataset

class SplitRequest(BaseModel):
    input_dir_process: str
    dataset_name: str
    output_root: Optional[str] = None
    train_split: str = "treino"
    test_split: str = "teste"
    train_ratio: float = 0.85  # Porcentagem de treino (0.0 a 1.0)

class TestRequest(BaseModel):
    model_path: Optional[str] = None
    dataset_path: Optional[str] = None
    device: str = "cuda"

class ConvertRequest(BaseModel):
    dataset_path: str  # Caminho do dataset (datasets/<nome>)
    extension: str = ".pt"  # Extensão de saída (.pt, .pth)

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

class TrainRetrainRequest(TrainStartRequest):
    pretrained_path: str  # Caminho do modelo pré-treinado

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
    category = "process"
    
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
                log_buffer.write(line, category)
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
                                log_buffer.write(line, category)
                        except:
                            break
                    break
        
        process.wait()
        
        if process.returncode == 0:
            log_buffer.write("[OK] Processamento concluído com sucesso!", category)
            logger.info("Processamento concluído com sucesso.")
        elif not state.stop_requested:
            log_buffer.write(f"[ERRO] Processo finalizou com código: {process.returncode}", category)
            logger.error(f"Processo finalizou com código: {process.returncode}")
            
    except Exception as e:
        logger.error(f"Erro ao executar subprocess: {e}")
        log_buffer.write(f"[ERRO] {str(e)}", category)
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
    with CaptureOutput(category="process"):
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
    
    with CaptureOutput(category="test"):
        logger.info("Iniciando fase de testes e validação...")
        try:
            import sys
            # Passa argumentos para o script de teste
            sys.argv = ["testar_modelo.py"]
            if req.get("dataset_path"):
                sys.argv.extend(["--input-dir", req["dataset_path"]])
            if req.get("model_path"):
                sys.argv.extend(["--model-dir", req["model_path"]])
            
            # Recarrega os modulos para garantir que peguem o novo sys.argv
            import importlib
            import app.configuracao.config as config
            import app.utils.ferramentas as tools
            import app.testar_modelo as tm
            
            importlib.reload(config)
            importlib.reload(tools)
            importlib.reload(tm)
            
            tm.main()
            logger.info("Fase de testes concluída.")
        except Exception as e:
            logger.error(f"Erro nos testes: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            state.reset()



# ==============================================================
# ENDPOINTS
# ==============================================================

@app.get("/")
def root():
    return {"status": "ok", "system": "NeuraPose API"}

@app.get("/logs")
def get_logs(category: str = "default"):
    """Retorna logs do backend por categoria."""
    return {"logs": LogBuffer().get_logs(category)}

@app.delete("/logs")
def clear_logs(category: Optional[str] = None):
    """Limpa o buffer de logs do servidor (ou de uma categoria)."""
    LogBuffer().clear(category)
    return {"status": "logs cleared", "category": category or "all"}

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
            "modelos_treinados": str(cm.TRAINED_MODELS_DIR),
            "relatorios_testes": str(cm.TEST_REPORTS_DIR),
            "anotacoes": str(cm.ANNOTATIONS_OUTPUT_DIR),
            "root": str(cm.ROOT)
        },
        "classes": {
            "classe1": cm.CLASSE1,
            "classe2": cm.CLASSE2,
            "class_names": cm.CLASS_NAMES
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

@app.post("/train/stop")
def stop_training():
    state.stop_requested = True
    return {"status": "stop_requested", "message": "Interrupção de treinamento solicitada."}

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


@app.post("/train/start")
async def train_model_start(req: TrainStartRequest, background_tasks: BackgroundTasks):
    """Inicia treinamento de novo modelo com parâmetros configuráveis."""
    from LSTM.pipeline.treinador import main as train_main
    
    dataset_path = Path(req.dataset_path).resolve()
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset não encontrado: {dataset_path}")
    
    # Busca arquivo .pt na estrutura esperada
    data_file = dataset_path / "treino" / "data" / "data.pt"
    if not data_file.exists():
        data_file = dataset_path / "data.pt"  # fallback
    if not data_file.exists():
        # Busca em qualquer lugar
        pt_files = list(dataset_path.rglob("*.pt"))
        if pt_files:
            data_file = pt_files[0]
        else:
            raise HTTPException(status_code=400, detail=f"Arquivo .pt não encontrado em {dataset_path}")
    
    # Nome do dataset para nomenclatura
    dataset_name = dataset_path.name
    
    def run_train():
        state.is_running = True
        with CaptureOutput(category="train"):
            try:
                import sys
                # Configura argumentos para o treinador
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
                
                logger.info(f"[TREINO] Iniciando treinamento")
                logger.info(f"[TREINO] Dataset: {data_file}")
                logger.info(f"[TREINO] Modelo: {req.model_type}")
                logger.info(f"[TREINO] Epochs: {req.epochs}")
                
                train_main()
                
                logger.info(f"[OK] Treinamento concluído!")
            except Exception as e:
                logger.error(f"[ERRO] Treinamento falhou: {e}")
            finally:
                state.reset()
    
    background_tasks.add_task(run_train)
    return {"status": "started", "message": f"Treinamento iniciado: {dataset_name} com {req.model_type}"}


@app.post("/train/retrain")
async def train_model_retrain(req: TrainRetrainRequest, background_tasks: BackgroundTasks):
    """Retreina modelo existente com novos dados ou mais épocas."""
    from LSTM.pipeline.treinador import main as train_main
    
    dataset_path = Path(req.dataset_path).resolve()
    pretrained_path = Path(req.pretrained_path).resolve()
    
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset não encontrado: {dataset_path}")
    if not pretrained_path.exists():
        raise HTTPException(status_code=404, detail=f"Modelo pré-treinado não encontrado: {pretrained_path}")
    
    # Busca arquivo .pt
    data_file = dataset_path / "treino" / "data" / "data.pt"
    if not data_file.exists():
        pt_files = list(dataset_path.rglob("*.pt"))
        data_file = pt_files[0] if pt_files else None
    if not data_file:
        raise HTTPException(status_code=400, detail=f"Arquivo .pt não encontrado em {dataset_path}")
    
    # Busca model_best.pt no modelo pré-treinado
    model_best = pretrained_path / "model_best.pt"
    if not model_best.exists():
        model_best = pretrained_path  # Assume que é o arquivo diretamente
    
    dataset_name = dataset_path.name
    
    def run_retrain():
        state.is_running = True
        with CaptureOutput(category="train"):
            try:
                import sys
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
                
                logger.info(f"[RETRAIN] Iniciando retreinamento")
                logger.info(f"[RETRAIN] Dataset: {data_file}")
                logger.info(f"[RETRAIN] Modelo base: {model_best}")
                
                train_main()
                
                logger.info(f"[OK] Retreinamento concluído!")
            except Exception as e:
                logger.error(f"[ERRO] Retreinamento falhou: {e}")
            finally:
                state.reset()
    
    background_tasks.add_task(run_retrain)
    return {"status": "started", "message": f"Retreinamento iniciado: {dataset_name}"}



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
def remove_from_agenda(video_id: str, root_path: Optional[str] = None):
    """Remove um vídeo da agenda, aceitando root_path dinâmico."""
    target_file = None
    if root_path:
        paths = get_reid_paths_from_source(root_path)
        target_file = paths["agenda_file"]
    else:
        target_file = REID_AGENDA_FILE
        
    if not target_file.exists():
        return {"status": "not_found", "detail": "Agenda file not found"}
    
    try:
        with open(target_file, "r", encoding="utf-8") as f:
            agenda = json.load(f)
        
        original_count = len(agenda.get("videos", []))
        agenda["videos"] = [v for v in agenda.get("videos", []) if v["video_id"] != video_id]
        
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(agenda, f, ensure_ascii=False, indent=2)
        
        removed = original_count > len(agenda["videos"])
        return {"status": "removed" if removed else "not_found"}
    except Exception as e:
        logger.error(f"Erro ao remover da agenda: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint duplicado removido - usar /reid/batch-apply abaixo (linha ~1527)

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
    """Lista vídeos disponíveis para anotação.
    
    Busca vídeos em predicoes/ e verifica labels em {root}/anotacoes/labels.json
    """
    root = Path(root_path).resolve() if root_path else cm.REID_OUTPUT_DIR
    
    # Ajusta se usuário selecionou subpasta
    if root.name in ["predicoes", "jsons", "videos", "reid", "anotacoes"]:
        root = root.parent
    
    # Pastas de origem
    pred_dir = root / "predicoes"
    json_dir = root / "jsons"
    videos_dir = root / "videos"
    
    # Labels na pasta do dataset
    labels_path = root / "anotacoes" / "labels.json"
    
    # Carrega labels existentes
    labels_existentes = {}
    if labels_path.exists():
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                labels_existentes = json.load(f) or {}
        except: 
            pass
    
    result = []
    
    # Lista vídeos de predicoes ou videos (preferência para predicoes)
    source_dir = pred_dir if pred_dir.exists() else videos_dir
    
    if source_dir.exists():
        valid_exts = {'.mp4', '.avi', '.mov', '.mkv'}
        for vfile in source_dir.glob("*"):
            if vfile.suffix.lower() not in valid_exts:
                continue
                
            stem = vfile.stem.replace("_pose", "")
            
            # Verifica se tem JSON correspondente
            json_path = json_dir / f"{stem}.json"
            if not json_path.exists():
                json_path = json_dir / f"{vfile.stem}.json"
            
            status = "anotado" if stem in labels_existentes else "pendente"
            
            result.append({
                "video_id": stem,
                "video_name": vfile.name,
                "status": status,
                "has_json": json_path.exists(),
                "creation_time": vfile.stat().st_mtime
            })
    
    # Ordena por data de criação (mais recente primeiro)
    return {
        "videos": sorted(result, key=lambda x: x["creation_time"], reverse=True),
        "root": str(root),
        "labels_path": str(labels_path)
    }

@app.get("/annotate/{video_id}/details")
def get_annotation_details(video_id: str, root_path: Optional[str] = None):
    """Retorna detalhes dos IDs encontrados no vídeo para anotação."""
    from pre_processamento.anotando_classes import carregar_pose_records, indexar_por_frame
    
    root = Path(root_path).resolve() if root_path else cm.REID_OUTPUT_DIR
    logger.info(f"[ANNOTATE] root_path recebido: {root_path}")
    logger.info(f"[ANNOTATE] root inicial: {root}")
    
    # Ajusta se usuário selecionou subpasta
    if root.name in ["predicoes", "jsons", "videos", "reid", "anotacoes"]:
        root = root.parent
        logger.info(f"[ANNOTATE] root ajustado para pai: {root}")
    
    # Tenta encontrar o JSON (pode ter sufixo _pose ou não)
    json_dir = root / "jsons"
    logger.info(f"[ANNOTATE] Buscando JSON em: {json_dir}")
    
    # Lista arquivos disponíveis para debug
    if json_dir.exists():
        available = [f.name for f in json_dir.glob("*.json")]
        logger.info(f"[ANNOTATE] JSONs disponíveis: {available}")
    
    json_path = json_dir / f"{video_id}.json"
    logger.info(f"[ANNOTATE] Tentando: {json_path.name} | Existe: {json_path.exists()}")
    
    if not json_path.exists():
        # Tenta sem sufixo _pose
        clean_id = video_id.replace("_pose", "")
        json_path = json_dir / f"{clean_id}.json"
        logger.info(f"[ANNOTATE] Tentando sem _pose: {json_path.name} | Existe: {json_path.exists()}")
    
    if not json_path.exists():
        # Tenta com sufixo _pose
        json_path = json_dir / f"{video_id}_pose.json"
        logger.info(f"[ANNOTATE] Tentando com _pose: {json_path.name} | Existe: {json_path.exists()}")
    
    if not json_path.exists():
        logger.error(f"[ANNOTATE] JSON não encontrado para: {video_id}")
        raise HTTPException(status_code=404, detail=f"JSON não encontrado: {video_id}")
        
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
        "total_frames": max(frames_index.keys()) if frames_index else 0,
        "min_frames": cm.MIN_FRAMES_PER_ID
    }

@app.post("/annotate/save")
def save_annotations(req: AnnotationRequest):
    """Salva as anotações no arquivo labels.json na pasta do dataset.
    
    Salva em: {root_path}/anotacoes/labels.json
    """
    root = Path(req.root_path).resolve()
    
    # Ajusta se usuário selecionou subpasta
    if root.name in ["predicoes", "jsons", "videos", "reid", "anotacoes"]:
        root = root.parent
    
    # Define caminho do labels.json na pasta do dataset
    labels_dir = root / "anotacoes"
    labels_path = labels_dir / "labels.json"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Carrega labels existentes
    todas_labels = {}
    if labels_path.exists():
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                todas_labels = json.load(f) or {}
        except: 
            pass
    
    # Atualiza com novas anotações
    # Formato: { "video_stem": { "ID": "classe", ... } }
    todas_labels[req.video_stem] = req.annotations
    
    try:
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(todas_labels, f, indent=4, ensure_ascii=False)
        return {
            "status": "success", 
            "message": f"Anotações salvas para {req.video_stem}",
            "labels_path": str(labels_path)
        }
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
                    test_split=req.test_split,
                    train_ratio=req.train_ratio
                )
            except Exception as e:
                logger.error(f"Erro no split de dataset: {e}")
            finally:
                state.reset()
                
    background_tasks.add_task(run_split_task)
    return {"status": "started", "message": f"Split iniciado para o dataset {req.dataset_name}"}


@app.post("/convert/pt")
async def convert_dataset_to_pt(req: ConvertRequest, background_tasks: BackgroundTasks):
    """Converte JSONs de anotações para formato PyTorch (.pt)."""
    from pre_processamento.converte_pt import main as converte_main
    
    dataset_path = Path(req.dataset_path).resolve()
    
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset não encontrado: {dataset_path}")
    
    # Suporte para estrutura flexível: busca 'treino' ou usa a raiz
    base_search = dataset_path / "treino" if (dataset_path / "treino").exists() else dataset_path
    
    # Busca pasta de JSONs ('dados' ou 'jsons')
    jsons_dir = base_search / "dados" if (base_search / "dados").exists() else base_search / "jsons"
    annotations_dir = base_search / "anotacoes"
    labels_path = annotations_dir / "labels.json"
    
    if not jsons_dir.exists():
        raise HTTPException(status_code=400, detail=f"Pasta de JSONs ('dados' ou 'jsons') não encontrada em {base_search}")
    if not labels_path.exists():
        raise HTTPException(status_code=400, detail=f"labels.json não encontrado em {annotations_dir}")
    
    def run_conversion():
        state.is_running = True
        with CaptureOutput(category="convert"):
            try:
                import config_master as cm
                original_jsons = cm.PROCESSING_JSONS_DIR
                original_labels = cm.PROCESSING_ANNOTATIONS_DIR
                
                # Nome do dataset e caminhos de saída
                dataset_name = dataset_path.name
                out_dir = (dataset_path / "treino" / "data").resolve()
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"data{req.extension}"
                
                # FORÇA a atualização das variáveis dentro do módulo converte_pt
                import pre_processamento.converte_pt as cpt
                
                # Atualiza cm para qualquer outro import interno no converte_pt
                cm.PROCESSING_JSONS_DIR = jsons_dir
                cm.PROCESSING_ANNOTATIONS_DIR = annotations_dir
                
                # Sobrescreve caminhos que o script usa internamente
                cpt.JSONS_DIR = jsons_dir.resolve()
                cpt.LABELS_PATH = labels_path.resolve()
                cpt.OUT_PT = out_file.resolve()
                cpt.LOG_FILE = (out_dir / "frames_invalidos.txt").resolve()
                cpt.DEBUG_LOG = (out_dir / "debug_log.txt").resolve()
                
                logger.info(f"[CONVERTE] Iniciando conversão para {dataset_name}")
                logger.info(f"[CONVERTE] Caminho Base Localizado: {base_search}")
                logger.info(f"[CONVERTE] Usando JSONs em: {cpt.JSONS_DIR}")
                logger.info(f"[CONVERTE] Usando Labels em: {cpt.LABELS_PATH}")
                logger.info(f"[CONVERTE] Destino Final: {cpt.OUT_PT}")
                
                # Executa a função main do script de conversão
                cpt.main()
                
                logger.info(f"[OK] Conversão concluída com sucesso!")
                
            except Exception as e:
                logger.error(f"[ERRO] Conversão falhou: {e}")
            finally:
                # Restaura configs originais
                if 'original_jsons' in locals():
                    cm.PROCESSING_JSONS_DIR = original_jsons
                if 'original_labels' in locals():
                    cm.PROCESSING_ANNOTATIONS_DIR = original_labels
                state.reset()
    
    background_tasks.add_task(run_conversion)
    return {"status": "started", "message": f"Conversão iniciada para {dataset_path.name}"}

# Models e Endpoints ReID Batch
class ReidBatchApplyRequest(BaseModel):
    videos: Optional[List[Any]] = None
    root_path: str
    output_path: Optional[str] = None

def run_reid_batch_processing(source_dataset: str):
    """Processa todos os vídeos agendados em reid.json e salva em resultados-reidentificacoes."""
    state.is_running = True
    logger.info(f"Iniciando Batch Process ReID: {source_dataset}")
    
    with CaptureOutput():
        try:
            paths = get_reid_paths_from_source(source_dataset)
            
            # 1. Cria diretórios de saída
            for k in ["videos_dir", "predictions_dir", "jsons_dir"]:
                paths[k].mkdir(parents=True, exist_ok=True)
            logger.info(f"Pasta de saída: {paths['root']}")
                
            # 2. Verifica Agenda
            if not paths["agenda_file"].exists():
                logger.error(f"Agenda não encontrada: {paths['agenda_file']}")
                return
                
            with open(paths["agenda_file"], "r", encoding="utf-8") as f:
                agenda = json.load(f)
                
            # 3. Define Origem
            src_path = Path(source_dataset)
            if src_path.name in ['predicoes', 'jsons', 'videos']:
                ds_root = src_path.parent
            else:
                ds_root = src_path
                
            src_videos = ds_root / "videos"
            src_jsons = ds_root / "jsons"
            
            # 4. Identifica vídeos para excluir (não copiar)
            videos_to_delete = set()
            videos_to_process = {}
            
            for vid_entry in agenda.get("videos", []):
                vid_id = vid_entry["video_id"]
                action = vid_entry.get("action", "process")
                
                if action == "delete":
                    videos_to_delete.add(vid_id)
                    # Remove _pose suffix para encontrar vídeo original
                    clean_id = vid_id.replace("_pose", "")
                    videos_to_delete.add(clean_id)
                else:
                    videos_to_process[vid_id] = vid_entry
            
            logger.info(f"Vídeos a excluir: {len(videos_to_delete)}")
            logger.info(f"Vídeos a processar: {len(videos_to_process)}")
            
            # 5. Copia Vídeos Originais (exceto os marcados para delete)
            logger.info("Copiando vídeos base...")
            copied_count = 0
            if src_videos.exists():
                for vfile in src_videos.glob("*"):
                    if vfile.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                        # Verifica se deve pular (vídeo marcado para delete)
                        if vfile.stem in videos_to_delete:
                            logger.info(f"Pulando vídeo excluído: {vfile.name}")
                            continue
                            
                        dst = paths["videos_dir"] / vfile.name
                        if not dst.exists():
                            shutil.copy2(vfile, dst)
                            copied_count += 1
            logger.info(f"Copiados {copied_count} vídeos")
            
            # 6. Processa Vídeos da Agenda
            processed_count = 0
            for vid_id, vid_entry in videos_to_process.items():
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
                    logger.warning(f"JSON input não encontrado: {json_in}")
                    continue
                    
                with open(json_in, "r", encoding="utf-8") as f:
                    records = json.load(f)
                
                # Verifica se há alterações reais
                has_changes = len(swaps) > 0 or len(deletions) > 0 or len(cuts) > 0
                
                if has_changes:
                    # Aplica Lógica de processamento
                    processed, _, _, _ = aplicar_processamento_completo(records, swaps, deletions, cuts)
                else:
                    # Sem alterações, apenas copia o JSON original
                    processed = records
                
                # Salva JSON (filtrado ou cópia)
                json_out = paths["jsons_dir"] / f"{vid_id}.json"
                with open(json_out, "w", encoding="utf-8") as f:
                    json.dump(processed, f, ensure_ascii=False, indent=2)
                    
                # Gera Vídeo Anotado apenas se houver alterações
                if has_changes:
                    # Encontra vídeo de entrada
                    vid_stem = vid_id.replace("_pose", "")
                    vid_in = paths["videos_dir"] / f"{vid_stem}.mp4"
                    if not vid_in.exists():
                        vid_in = paths["videos_dir"] / f"{vid_id}.mp4"
                    if not vid_in.exists():
                        candidates = list(paths["videos_dir"].glob(f"{vid_stem}.*"))
                        if candidates: 
                            vid_in = candidates[0]
                        
                    if vid_in.exists():
                        # 1. Vídeo Anotado (com caixas) -> Predições
                        vid_out_pred = paths["predictions_dir"] / f"{vid_id}_pose.mp4"
                        logger.info(f"Gerando vídeo anotado: {vid_out_pred.name}")
                        renderizar_video_limpo(str(vid_in), str(vid_out_pred), processed, cuts)

                        # 2. Vídeo Raw (sem caixas, mas cortado) -> Videos (para consistência)
                        vid_out_raw = paths["videos_dir"] / f"{vid_id}.mp4"
                        logger.info(f"Gerando vídeo raw cortado: {vid_out_raw.name}")
                        renderizar_video_cortado_raw(str(vid_in), str(vid_out_raw), cuts)
                    else:
                        logger.warning(f"Vídeo base não encontrado: {vid_id}")
                
                processed_count += 1
                    
            logger.info(f"Batch ReID finalizado. Processados: {processed_count} vídeos.")

        except Exception as e:
            logger.error(f"Erro crítico no batch reid: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            state.reset()

@app.post("/reid/batch-apply")
def batch_apply_reid(request: ReidBatchApplyRequest, background_tasks: BackgroundTasks):
    """Executa o processamento em massa das correções de ReID."""
    background_tasks.add_task(run_reid_batch_processing, request.root_path)
    return {"status": "started", "message": "Iniciando processamento ReID em background."}


# ==============================================================
# DATASETS MANAGER
# ==============================================================
@app.get("/datasets/list")
def list_all_datasets():
    """
    Lista todas as pastas de datasets organizadas por categoria:
    - processamentos: Vídeos processados (resultado_processamento/)
    - reidentificacoes: Datasets reidentificados (resultados-reidentificacoes/)
    - datasets: Datasets prontos para treino (datasets/)
    
    Retorna status de completude baseado na presença de subpastas.
    """
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
    reid_dir = cm.REID_MANUAL_OUTPUT_DIR
    if reid_dir.exists():
        for folder in reid_dir.iterdir():
            if folder.is_dir() and not folder.name.startswith('.'):
                info = check_folder_status(folder, ["videos", "jsons", "predicoes"])
                if info:
                    result["reidentificacoes"].append(info)
    
    # 3. Datasets (para treinamento)
    datasets_dir = cm.DATASETS_DIR if hasattr(cm, 'DATASETS_DIR') else cm.BACKEND_DIR / "datasets"
    if datasets_dir.exists():
        for folder in datasets_dir.iterdir():
            if folder.is_dir() and not folder.name.startswith('.'):
                # Para datasets, verificamos se tem train/test splits
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

