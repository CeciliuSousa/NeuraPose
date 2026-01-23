# ==============================================================
# neurapose_backend/celery_app.py
# ==============================================================
# Configuração do Celery para processamento assíncrono.
# Separa tarefas pesadas (YOLO, RTMPose) do servidor HTTP.
# ==============================================================

from celery import Celery
import neurapose_backend.config_master as cm

# Configuração do broker (Redis)
# Para desenvolvimento local: redis://localhost:6379/0
# Para produção: use variável de ambiente CELERY_BROKER_URL
BROKER_URL = "redis://localhost:6379/0"
RESULT_BACKEND = "redis://localhost:6379/1"

# Cria instância do Celery
celery_app = Celery(
    "neurapose",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=[
        "neurapose_backend.tasks.tarefas_processamento",
        "neurapose_backend.tasks.tarefas_treinamento",
    ]
)

# Configurações otimizadas
celery_app.conf.update(
    # Não prefetch muitas tarefas (processamento é pesado)
    worker_prefetch_multiplier=1,
    
    # Timeout para tarefas longas (30 minutos)
    task_time_limit=1800,
    
    # Resultado expira em 1 hora
    result_expires=3600,
    
    # Usa JSON para serialização (compatível com logs)
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    
    # Timezone
    timezone="America/Sao_Paulo",
    
    # Não ack até completar (evita perder tarefa se worker crashar)
    task_acks_late=True,
    
    # Rejeitado vai pro final da fila
    task_reject_on_worker_lost=True,
)

# Para rodar o worker:
# uv run celery -A neurapose_backend.celery_app worker --loglevel=info --pool=solo
