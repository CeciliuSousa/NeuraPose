import sys
from pathlib import Path
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from app.log_service import LogBuffer, CaptureOutput

# ==============================================================
# SETUP PATHS
# ==============================================================
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

# Adiciona o diretório atual e o ROOT ao sys.path
for p in [str(CURRENT_DIR), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ==============================================================
# IMPORTS DO PROJETO
# ==============================================================
# Importamos diretamente do config_master que está na mesma pasta
try:
    import config_master as cm
    from pre_processamento.pipeline.processador import processar_video
    from pre_processamento.utils.ferramentas import carregar_sessao_onnx
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
    # Se falhar o import direto, tentamos com o prefixo do pacote neurapose_backend
    try:
        import neurapose_backend.config_master as cm
        from neurapose_backend.pre_processamento.pipeline.processador import processar_video
        from neurapose_backend.pre_processamento.utils.ferramentas import carregar_sessao_onnx
        from neurapose_backend.pre_processamento.reid_manual import (
            aplicar_processamento_completo, 
            renderizar_video_limpo, 
            carregar_pose_records,
            salvar_json,
            indexar_por_frame_e_contar_ids
        )
        from neurapose_backend.LSTM.pipeline import treinador
    except ImportError as e2:
        print(f"CRITICAL ERROR: Could not import project modules.\nDirect import failed: {e}\nPackage import failed: {e2}")
        # Re-raise to stop the server if critical modules are missing
        raise e2

# ==============================================================
# LOGGING
# ==============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeuraPoseAPI")

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

# ==============================================================
# MODELS
# ==============================================================
class ProcessRequest(BaseModel):
    input_path: str
    output_path: str
    onnx_path: Optional[str] = None
    show_preview: bool = False

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
    rules: List[ReIDRule]
    deletions: List[ReIDDelete]
    cuts: List[ReIDCut]
    output_path: Optional[str] = None

# ==============================================================
# HELPERS
# ==============================================================
from app.state import state

# ==============================================================
# CONFIG MASTER MANAGER
# ==============================================================
def get_all_cm_config():
    return {
        "YOLO_MODEL": cm.YOLO_MODEL,
        "DETECTION_CONF": cm.DETECTION_CONF,
        "POSE_CONF_MIN": cm.POSE_CONF_MIN,
        "EMA_ALPHA": cm.EMA_ALPHA,
        "TIME_STEPS": cm.TIME_STEPS,
        "BATCH_SIZE": cm.BATCH_SIZE,
        "LEARNING_RATE": cm.LEARNING_RATE,
        "EPOCHS": cm.EPOCHS,
        "FURTO_THRESHOLD": cm.FURTO_THRESHOLD,
        "MAX_FRAMES_PER_SEQUENCE": cm.MAX_FRAMES_PER_SEQUENCE,
        "PROCESSING_DATASET": cm.PROCESSING_DATASET,
    }

def update_cm_file(updates: Dict[str, Any]):
    import re
    config_file = Path(cm.__file__)
    content = config_file.read_text(encoding="utf-8")
    
    for key, value in updates.items():
        if isinstance(value, str):
            # Match KEY = "value" or KEY = 'value'
            pattern = rf'^({key}\s*=\s*["\'])(.*?)(["\'])'
            content = re.sub(pattern, rf'\1{value}\3', content, flags=re.MULTILINE)
        else:
            # Match KEY = 0.5 or KEY = 10
            pattern = rf'^({key}\s*=\s*)([\d\.\-]+)'
            content = re.sub(pattern, rf'\1{value}', content, flags=re.MULTILINE)
            
    config_file.write_text(content, encoding="utf-8")
    # Reload module (experimental, safe for constants)
    import importlib
    importlib.reload(cm)

# ==============================================================
# HELPERS
# ==============================================================
def run_processing_task(input_path: Path, output_path: Path, onnx_path: Path, show: bool):
    state.reset()
    state.is_running = True
    
    # Wrap execution to capture stdout/stderr to LogBuffer
    with CaptureOutput():
        logger.info(f"Iniciando processamento: {input_path} -> {output_path}")
        
        try:
            sess, input_name = carregar_sessao_onnx(str(onnx_path))
            
            if input_path.is_file():
                v_name = input_path.stem
                final_out = output_path
                if final_out == Path(cm.PROCESSING_OUTPUT_DIR):
                     final_out = output_path / v_name
                processar_video(input_path, sess, input_name, final_out, show=show)
            elif input_path.is_dir():
                videos = sorted(input_path.glob("*.mp4"))
                output_path.mkdir(parents=True, exist_ok=True)
                for v in videos:
                    if state.stop_requested: break
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

def pick_folder_via_subprocess():
    import subprocess
    import sys
    
    # Get current python executable to run the script
    python_exe = sys.executable
    picker_script = Path(__file__).parent / "picker.py"
    
    try:
        # Run picker.py and capture stdout
        # Using shell=False for security and robustness
        result = subprocess.check_output(
            [python_exe, str(picker_script)],
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        folder = result.strip()
        return folder if folder else None
    except subprocess.CalledProcessError:
        # This happens if exit code is non-zero (e.g. cancelled)
        return None
    except Exception as e:
        logger.error(f"Error calling picker subprocess: {e}")
        return None

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

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "system": "NeuraPose API",
        "version": "1.0.0",
        "device": "gpu" if cm.DEVICE == "cuda" else "cpu",
        "neurapose_root": str(ROOT_DIR),
        "processing": state.is_running,
        "paused": state.is_paused
    }

@app.get("/config/all")
def api_get_all_config():
    return get_all_cm_config()

@app.get("/config")
def api_get_config():
    """Retorna configurações básicas, compatível com o FileExplorerModal do frontend."""
    return {
        "status": "success",
        "paths": {
            "processing_input": str(cm.PROCESSING_INPUT_DIR),
            "processing_output": str(cm.PROCESSING_OUTPUT_DIR)
        }
    }

@app.get("/browse")
def browse_path(path: str = "."):
    """Lista o conteúdo de uma pasta para o explorador de arquivos do frontend."""
    try:
        p = Path(path).resolve()
        
        # Por segurança, podemos limitar o browse ao ROOT_DIR ou drives específicos
        # Para este projeto, vamos permitir navegar livremente, mas com cautela
        
        items = []
        for entry in p.iterdir():
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

@app.post("/config/update")
def api_update_config(updates: Dict[str, Any]):
    try:
        update_cm_file(updates)
        return {"status": "success", "message": "Configurações atualizadas no config_master.py"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video_feed")
def video_feed():
    def generate():
        import cv2
        while True:
            frame = state.get_frame()
            if frame is not None:
                # Encode as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                # Small delay to not consume CPU
                import time
                time.sleep(0.1)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

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

@app.get("/pick-folder")
def pick_folder():
    """Abre janela nativa do Windows usando subprocesso (mais estável no Windows)."""
    folder = pick_folder_via_subprocess()
        
    if folder:
        return {"path": folder}
    return {"path": None}

@app.post("/process")
async def start_processing(req: ProcessRequest, background_tasks: BackgroundTasks):
    """Inicia processamento de video(s)."""
    inp = Path(req.input_path)
    out = Path(req.output_path)
    onnx = Path(req.onnx_path) if req.onnx_path else cm.RTMPOSE_PREPROCESSING_PATH
    
    if not inp.exists():
        raise HTTPException(status_code=404, detail="Input path not found")
    
    background_tasks.add_task(run_processing_task, inp, out, onnx, req.show_preview)
    return {"status": "started", "detail": f"Processing {inp.name} -> {out.name}"}

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
    root = Path(root_path) if root_path else cm.PROCESSING_OUTPUT_DIR
    json_dir = root / "jsons"
    
    if not json_dir.exists():
        return {"videos": []}
        
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
            "processed": False # TODO: Check if already cleaned
        })
    return {"videos": videos}

@app.get("/reid/{video_id}/data")
def get_reid_data(video_id: str, root_path: Optional[str] = None):
    """Retorna dados de poses para o editor."""
    root = Path(root_path) if root_path else cm.PROCESSING_OUTPUT_DIR
    json_path = root / "jsons" / f"{video_id}.json"
    
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="JSON not found")
    
    records = carregar_pose_records(json_path)
    frames_index, id_counter = indexar_por_frame_e_contar_ids(records)
    
    # Converte frames_index para formato JSON-friendly
    # Dict keys must be strings in JSON
    frames_clean = {}
    for f, items in frames_index.items():
        frames_clean[str(f)] = []
        for bbox, gid in items:
            frames_clean[str(f)].append({"bbox": bbox, "id": gid})
            
    return {
        "video_id": video_id,
        "frames": frames_clean,
        "id_counts": dict(id_counter.most_common(100))
    }

@app.get("/reid/video/{video_id}")
def stream_video(video_id: str, root_path: Optional[str] = None):
    """Serve o arquivo de video."""
    root = Path(root_path) if root_path else cm.PROCESSING_OUTPUT_DIR
    
    # Tenta achar o video original
    v_raw = root / "videos" / f"{video_id.replace('_pose', '')}.mp4"
    if not v_raw.exists():
        # Tenta com nome direto
        v_raw = root / "videos" / f"{video_id}.mp4"
        
    if not v_raw.exists():
        # Tenta achar na pasta de predicoes se nao tiver original?
        v_pred = root / "predicoes" / f"{video_id}_pose.mp4"
        if v_pred.exists():
            v_raw = v_pred
        else:
            raise HTTPException(status_code=404, detail="Video not found")
            
    return FileResponse(v_raw, media_type="video/mp4")

@app.post("/reid/{video_id}/apply")
def apply_reid_changes(video_id: str, req: ReIDApplyRequest, root_path: Optional[str] = None):
    """Aplica as mudancas e gera novo video."""
    root = Path(root_path) if root_path else cm.PROCESSING_OUTPUT_DIR
    
    if req.output_path:
        out_root = Path(req.output_path)
    else:
        out_root = root.parent / (root.name + "-reid")
    
    out_json_dir = out_root / "jsons"
    out_pred_dir = out_root / "predicoes"
    out_videos_dir = out_root / "videos" # Backup/Cpy
    
    for p in [out_json_dir, out_pred_dir, out_videos_dir]:
        p.mkdir(parents=True, exist_ok=True)
        
    # Paths
    json_path = root / "jsons" / f"{video_id}.json"
    v_raw = root / "videos" / f"{video_id.replace('_pose', '')}.mp4"
    if not v_raw.exists(): v_raw = root / "videos" / f"{video_id}.mp4"
    
    out_json_path = out_json_dir / f"{video_id}.json"
    out_v_pred = out_pred_dir / f"{video_id}_pose.mp4"
    
    # Carrega dados
    records = carregar_pose_records(json_path)
    
    # Converte models para formato de lista de dicts (esperado pelas funcoes legadas)
    rules_list = [r.dict() for r in req.rules]
    delete_list = [d.dict() for d in req.deletions]
    cut_list = [c.dict() for c in req.cuts]
    
    # Aplica processamento
    recs_mod, c_ids, d_ids, d_cuts = aplicar_processamento_completo(records, rules_list, delete_list, cut_list)
    
    # Salva JSON
    salvar_json(out_json_path, recs_mod)
    
    # Renderiza Video
    # Isso pode demorar, idealmente seria BackgroundTask, mas o usuario pode querer esperar ou polling
    renderizar_video_limpo(v_raw, out_v_pred, recs_mod, cut_list)
    
    return {
        "status": "success",
        "swaps": c_ids,
        "deletions": d_ids,
        "cuts": d_cuts,
        "output_video": str(out_v_pred),
        "output_json": str(out_json_path)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
