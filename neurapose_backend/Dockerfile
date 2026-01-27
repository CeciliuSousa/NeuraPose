# ============================================================
# Dockerfile - NeuraPose Backend (Python/FastAPI + CUDA)
# ============================================================

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv (gerenciador de pacotes Python)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copiar arquivos de dependências (da raiz do contexto)
COPY pyproject.toml uv.lock ./

# Instalar dependências Python
RUN uv sync --frozen

# Copiar apenas o código do backend
COPY neurapose_backend ./neurapose_backend

# Expor porta
EXPOSE 8000

# Comando para iniciar o servidor
CMD ["uv", "run", "uvicorn", "neurapose_backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
