# NeuraPose Backend API

API REST para automação do sistema NeuraPose usando FastAPI.

## Quick Start

```bash
# Instalar dependências (na raiz do projeto)
uv sync --index-strategy unsafe-best-match

# Executar backend (dev)
uv run uvicorn neurapose_backend.main:app --reload --port 8000

# Acessar docs
# http://localhost:8000/docs
```

## Endpoints Principais

| Grupo | Rota | Método | Descrição |
|-------|------|--------|-----------|
| Sistema | `/` | GET | Status da API |
| Sistema | `/health` | GET | Health check com status de processamento |
| Sistema | `/logs` | GET | Logs do backend em tempo real |
| Processamento | `/process` | POST | Processar vídeos com YOLO + RTMPose |
| Processamento | `/process/stop` | POST | Parar processamento |
| Processamento | `/process/pause` | POST | Pausar processamento |
| Processamento | `/process/resume` | POST | Retomar processamento |
| ReID | `/reid/list` | GET | Listar vídeos para re-identificação |
| ReID | `/reid/{video_id}/apply` | POST | Aplicar correções de IDs |
| Anotação | `/annotate/list` | GET | Listar vídeos para anotação |
| Anotação | `/annotate/save` | POST | Salvar anotações |
| Dataset | `/dataset/split` | POST | Dividir dataset em treino/teste |
| Treinamento | `/train` | POST | Treinar modelo LSTM/TFT |
| Teste | `/test` | POST | Testar modelo treinado |
| Config | `/config` | GET | Obter configurações |
| Config | `/config/update` | POST | Atualizar configurações |
| Explorador | `/browse` | GET | Explorar diretórios |
| Preview | `/video_feed` | GET | Stream de vídeo em tempo real |

## Docker

```bash
docker-compose up -d
```

## Documentação Interativa

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health

