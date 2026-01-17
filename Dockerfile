# ============================================================
# Dockerfile - NeuraPose
# ============================================================
# Imagem com suporte a GPU (CUDA 12.8) usando UV para
# gerenciamento de pacotes Python.
#
# Build:
#   docker build -t neurapose .
#
# Run (com GPU):
#   docker run --gpus all -it --name neurapose neurapose
#
# Run (sem GPU):
#   docker run -it --name neurapose neurapose
# ============================================================

# Base com CUDA 12.8 + cuDNN + Python (compativel com PyTorch cu128)
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Evita prompts interativos durante instalacao
ENV DEBIAN_FRONTEND=noninteractive

# Metadata
LABEL maintainer="Cecilius Afonso Sousa do Carmo <ceciliusousa@gmail.com>"
LABEL description="NeuraPose-teste - Sistema de deteccao de acoes humanas usando pose estimation"
LABEL version="1.0.0"

# ============================================================
# 1. Instalar dependencias do sistema
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    curl \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Define Python 3.10 como padrao
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# ============================================================
# 2. Instalar UV (gerenciador de pacotes)
# ============================================================
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Adiciona UV ao PATH
ENV PATH="/root/.local/bin:${PATH}"

# ============================================================
# 3. Configurar diretorio de trabalho
# ============================================================
WORKDIR /app

# ============================================================
# 4. Copiar arquivos de projeto
# ============================================================
# Copia primeiro apenas arquivos de dependencias para cache
COPY pyproject.toml uv.lock* ./
COPY .python-version ./

# ============================================================
# 5. Criar ambiente virtual e instalar dependencias
# ============================================================
# Cria venv com Python 3.10.11
RUN uv venv --python 3.10.11 .venv

# Ativa venv no PATH
ENV PATH="/app/.venv/bin:${PATH}"
ENV VIRTUAL_ENV="/app/.venv"

# Instala dependencias (usa lock file se existir)
# Instala apenas as dependencias definidas no uv.lock (sem instalar o projeto)
RUN uv sync --no-install-project --index-strategy unsafe-best-match

# ============================================================
# 6. Copiar codigo fonte
# ============================================================
COPY neurapose/ ./neurapose/
COPY main.py ./
COPY README.md ./

# Instala o projeto atual e verifica integridade do ambiente
RUN uv sync --index-strategy unsafe-best-match

# ============================================================
# 7. Configurar variaveis de ambiente
# ============================================================
# Desabilita buffering do Python para logs em tempo real
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Configura CUDA (visibilidade de GPUs)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ============================================================
# 8. Criar diretorios para volumes
# ============================================================
RUN mkdir -p /app/neurapose/videos \
    /app/neurapose/datasets \
    /app/neurapose/modelos-lstm-treinados \
    /app/neurapose/resultado_processamento \
    /app/neurapose/relatorios-testes

# ============================================================
# 9. Entrypoint
# ============================================================
# Por padrao, abre um shell interativo
CMD ["/bin/bash"]
