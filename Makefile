# ================================================================
# NeuraPose - Makefile
# ================================================================
# Comandos rápidos para desenvolvimento e deploy
#
# Uso: make <comando>
# ================================================================

.PHONY: help dev backend frontend docker-build docker-up docker-down docker-logs clean install

# ================================================================
# AJUDA
# ================================================================
help:
	@echo ""
	@echo "==============================================="
	@echo "  NeuraPose - Comandos Disponíveis"
	@echo "==============================================="
	@echo ""
	@echo "  DESENVOLVIMENTO:"
	@echo "    make dev         - Inicia backend + frontend local"
	@echo "    make backend     - Inicia apenas o backend"
	@echo "    make frontend    - Inicia apenas o frontend"
	@echo "    make tauri       - Inicia app Tauri (desktop)"
	@echo ""
	@echo "  DOCKER:"
	@echo "    make docker-build  - Build das imagens Docker"
	@echo "    make docker-up     - Sobe containers"
	@echo "    make docker-down   - Para containers"
	@echo "    make docker-logs   - Ver logs"
	@echo ""
	@echo "  UTILITÁRIOS:"
	@echo "    make install     - Instala dependências"
	@echo "    make clean       - Limpa arquivos temporários"
	@echo "    make health      - Verifica status do backend"
	@echo ""

# ================================================================
# DESENVOLVIMENTO LOCAL
# ================================================================

# Inicia backend e frontend em paralelo
dev:
	@echo "Iniciando NeuraPose..."
	@make -j2 backend frontend

# Inicia apenas o backend (Python/FastAPI)
backend:
	@echo "[BACKEND] Iniciando servidor FastAPI..."
	uv run uvicorn neurapose_backend.main:app --reload --host 127.0.0.1 --port 8000

# Inicia apenas o frontend (React/Vite)
frontend:
	@echo "[FRONTEND] Iniciando Vite..."
	cd neurapose_tauri && npm run dev

# Inicia o app Tauri (desktop)
tauri:
	@echo "[TAURI] Iniciando aplicativo desktop..."
	cd neurapose_tauri && npm run tauri dev

# Build do app Tauri
tauri-build:
	@echo "[TAURI] Gerando build de produção..."
	cd neurapose_tauri && npm run tauri build

# ================================================================
# INSTALAÇÃO
# ================================================================

# Instala todas as dependências
install:
	@echo "Instalando dependências do backend..."
	uv sync
	@echo ""
	@echo "Instalando dependências do frontend..."
	cd neurapose_tauri && npm install
	@echo ""
	@echo "✓ Instalação concluída!"

# ================================================================
# DOCKER
# ================================================================

docker-build:
	@echo "Construindo imagens Docker..."
	docker-compose build

docker-up:
	@echo "Subindo containers..."
	docker-compose up -d
	@echo ""
	@echo "✓ Backend: http://localhost:8000"
	@echo "✓ Frontend: http://localhost:1420"
	@echo "✓ Docs API: http://localhost:8000/docs"

docker-down:
	@echo "Parando containers..."
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-restart:
	@echo "Reiniciando containers..."
	docker-compose restart

# ================================================================
# UTILITÁRIOS
# ================================================================

# Verifica saúde do backend
health:
	@curl -s http://localhost:8000/health | python -m json.tool 2>/dev/null || echo "Backend offline"

# Limpa arquivos temporários
clean:
	@echo "Limpando arquivos temporários..."
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "node_modules/.cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Limpeza concluída!"

# Limpa resultados de processamento
clean-results:
	@echo "Limpando resultados de processamento..."
	rm -rf neurapose_backend/resultados-processamentos/* 2>/dev/null || true
	rm -rf neurapose_backend/resultados-reidentificacoes/* 2>/dev/null || true
	rm -rf neurapose_backend/relatorios-testes/* 2>/dev/null || true
	@echo "✓ Resultados limpos!"
