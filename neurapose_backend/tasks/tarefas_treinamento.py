# ==============================================================
# neurapose_backend/tasks/training_tasks.py
# ==============================================================
# Tarefas Celery para treinamento de modelos.
# ==============================================================

from celery import shared_task
from pathlib import Path
import neurapose_backend.config_master as cm


@shared_task(bind=True, name="train_model")
def train_model_task(self, model_name: str, dataset_name: str, epochs: int = None, batch_size: int = None):
    """
    Tarefa Celery para treinamento de modelo temporal.
    
    Args:
        model_name: Nome do modelo a ser salvo
        dataset_name: Nome do dataset de treino
        epochs: Número de épocas (default: cm.EPOCHS)
        batch_size: Tamanho do batch (default: cm.BATCH_SIZE)
    
    Returns:
        Dict com métricas do treinamento
    """
    from neurapose_backend.LSTM.pipeline.treinador import main as start_train
    
    # Atualiza configs se fornecidas
    if epochs:
        cm.EPOCHS = epochs
    if batch_size:
        cm.BATCH_SIZE = batch_size
    
    cm.TRAINED_MODEL_NAME = model_name
    cm.DATASET_NAME = dataset_name
    
    self.update_state(state="PROGRESS", meta={"step": "training", "epoch": 0, "total_epochs": cm.EPOCHS})
    
    try:
        # Executa treinamento
        start_train()
        
        return {
            "status": "success",
            "model_name": model_name,
            "model_path": str(cm.MODEL_SAVE_DIR)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@shared_task(bind=True, name="test_model")
def test_model_task(self, model_path: str, dataset_path: str, labels_path: str):
    """
    Tarefa Celery para teste de modelo.
    
    Args:
        model_path: Caminho da pasta do modelo
        dataset_path: Caminho dos vídeos de teste
        labels_path: Caminho do labels.json
    """
    from neurapose_backend.app.testar_modelo import main as run_test
    
    self.update_state(state="PROGRESS", meta={"step": "testing"})
    
    try:
        # Atualiza configs
        cm.TEST_MODEL_DIR = Path(model_path)
        cm.TEST_DIR = Path(dataset_path)
        cm.TEST_LABELS_JSON = Path(labels_path)
        
        run_test()
        
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
