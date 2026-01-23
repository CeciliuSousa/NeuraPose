# ==============================================================
# neurapose_backend/tasks/processing_tasks.py
# ==============================================================
# Tarefas Celery para processamento de vídeos.
# Roda em worker separado do servidor HTTP.
# ==============================================================

from celery import shared_task
from pathlib import Path
import neurapose_backend.config_master as cm


@shared_task(bind=True, name="process_videos")
def process_videos_task(self, input_path: str, dataset_name: str, show: bool = False, device: str = "cuda"):
    """
    Tarefa Celery para processar vídeos.
    
    Esta tarefa roda em um worker separado, não bloqueando o servidor HTTP.
    O progresso é enviado via state updates.
    
    Args:
        input_path: Caminho dos vídeos (arquivo ou pasta)
        dataset_name: Nome do dataset de saída
        show: Se True, gera preview
        device: "cuda" ou "cpu"
    
    Returns:
        Dict com status e informações do processamento
    """
    from neurapose_backend.pre_processamento.pipeline.processador import processar_video
    from neurapose_backend.pre_processamento.utils.ferramentas import imprimir_banner
    
    # Atualiza cm.DEVICE
    cm.DEVICE = device if device == "cuda" else "cpu"
    
    output_path = Path(cm.PROCESSING_OUTPUT_DIR / f"{dataset_name}-processado")
    input_p = Path(input_path)
    
    results = {
        "status": "processing",
        "videos_processed": 0,
        "videos_total": 0,
        "errors": []
    }
    
    try:
        # Imprime banner
        imprimir_banner(cm.RTMPOSE_PATH)
        
        if input_p.is_file():
            results["videos_total"] = 1
            self.update_state(state="PROGRESS", meta={"current": 0, "total": 1, "video": input_p.name})
            
            processar_video(input_p, output_path, show=show)
            results["videos_processed"] = 1
            
        elif input_p.is_dir():
            videos = sorted(input_p.glob("*.mp4"))
            results["videos_total"] = len(videos)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for i, v in enumerate(videos, 1):
                self.update_state(state="PROGRESS", meta={"current": i, "total": len(videos), "video": v.name})
                
                try:
                    processar_video(v, output_path, show=show)
                    results["videos_processed"] += 1
                except Exception as e:
                    results["errors"].append({"video": v.name, "error": str(e)})
        
        results["status"] = "success" if not results["errors"] else "partial"
        
    except Exception as e:
        results["status"] = "error"
        results["errors"].append({"video": "global", "error": str(e)})
    
    return results


@shared_task(bind=True, name="reidentify_videos")
def reidentify_task(self, input_folder: str, output_folder: str):
    """Tarefa Celery para re-identificação de IDs."""
    from neurapose_backend.pre_processamento.reidentificacao.reidentificador import main as run_reid
    
    self.update_state(state="PROGRESS", meta={"step": "reidentification"})
    
    try:
        run_reid(input_folder, output_folder)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
