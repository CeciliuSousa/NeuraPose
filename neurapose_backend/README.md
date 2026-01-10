# NeuraPose Backend API

API REST para automação do sistema NeuraPose usando FastAPI.

## Quick Start

```bash
# Instalar dependências
./NeuraPose-App/: uv sync --index-strategy unsafe-best-match

# Executar (dev)
uvicorn main:app --reload

# Acessar docs
# http://localhost:8000/docs
```

## Endpoints

| Grupo | Rota | Descrição |
|-------|------|-----------|
| Processing | `/api/v1/processing/videos` | Processar vídeos |
| Annotation | `/api/v1/annotation/save` | Salvar anotações |
| Dataset | `/api/v1/dataset/split` | Dividir dataset |
| Training | `/api/v1/training/start` | Treinar modelo |
| Testing | `/api/v1/testing/run` | Executar testes |
| Config | `/api/v1/config` | Configurações |

## Docker

```bash
docker-compose up -d
```

## Documentação

- Swagger UI: http://localhost:8000/docs
- Health: http://localhost:8000/health
