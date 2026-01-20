# ============================================================
# Dockerfile - NeuraPose Frontend (React/Vite)
# ============================================================

FROM node:20-alpine

WORKDIR /app

# Copiar arquivos de dependências
COPY package*.json ./

# Instalar dependências
RUN npm ci

# Copiar código fonte
COPY . .

# Expor porta do Vite
EXPOSE 1420

# Comando para iniciar o servidor de desenvolvimento
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "1420"]
