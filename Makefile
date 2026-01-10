# ================================================================
# NeuraPose - Makefile
# ================================================================
# Comandos úteis para desenvolvimento e deploy

.PHONY: help install dev docker-build docker-up docker-down docker-logs test clean

# Variáveis
BACKEND_DIR = neurapose-backend
PYTHON = python
UVICORN = uvicorn

help:
	@echo "NeuraPose Backend - Comandos disponíveis:"
	@echo ""
	@echo "  make install       - Instala dependências do backend"
	@echo "  make dev           - Inicia servidor de desenvolvimento"
	@echo "  make docker-build  - Build da imagem Docker"
	@echo "  make docker-up     - Sobe containers Docker"
	@echo "  make docker-down   - Para containers Docker"
	@echo "  make docker-logs   - Exibe logs dos containers"
	@echo "  make test          - Executa testes"
	@echo "  make clean         - Limpa arquivos temporários"
	@echo ""

# ================================================================
# Desenvolvimento Local
# ================================================================

install:
	@echo "Instalando dependências do backend..."
	cd $(BACKEND_DIR) && pip install -r requirements.txt
	@echo "Instalação concluída!"

dev:
	@echo "Iniciando servidor de desenvolvimento..."
	cd $(BACKEND_DIR) && $(UVICORN) main:app --reload --host 0.0.0.0 --port 8000

run:
	@echo "Iniciando servidor..."
	cd $(BACKEND_DIR) && $(UVICORN) main:app --host 0.0.0.0 --port 8000

# ================================================================
# Docker
# ================================================================

docker-build:
	@echo "Construindo imagem Docker..."
	cd $(BACKEND_DIR) && docker-compose build

docker-up:
	@echo "Subindo containers..."
	cd $(BACKEND_DIR) && docker-compose up -d
	@echo "API disponível em: http://localhost:8000"
	@echo "Docs em: http://localhost:8000/docs"

docker-down:
	@echo "Parando containers..."
	cd $(BACKEND_DIR) && docker-compose down

docker-logs:
	cd $(BACKEND_DIR) && docker-compose logs -f

docker-restart:
	@echo "Reiniciando containers..."
	cd $(BACKEND_DIR) && docker-compose restart

# ================================================================
# Testes e Qualidade
# ================================================================

test:
	@echo "Executando testes..."
	cd $(BACKEND_DIR) && $(PYTHON) -m pytest tests/ -v

lint:
	@echo "Verificando código..."
	cd $(BACKEND_DIR) && $(PYTHON) -m flake8 app/

format:
	@echo "Formatando código..."
	cd $(BACKEND_DIR) && $(PYTHON) -m black app/

# ================================================================
# Limpeza
# ================================================================

clean:
	@echo "Limpando arquivos temporários..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Limpeza concluída!"

clean-jobs:
	@echo "Limpando jobs antigos..."
	rm -rf $(BACKEND_DIR)/data/jobs/*.json
	@echo "Jobs limpos!"

# ================================================================
# Utilitários
# ================================================================

# Abre documentação da API no navegador
docs:
	@echo "Abrindo documentação..."
	start http://localhost:8000/docs || xdg-open http://localhost:8000/docs || open http://localhost:8000/docs

# Verifica saúde da API
health:
	curl -s http://localhost:8000/health | python -m json.tool

# Exibe configuração atual
config:
	curl -s http://localhost:8000/api/v1/config | python -m json.tool
