# ================================================================
# NeuraPose - Scripts de Desenvolvimento (Windows PowerShell)
# ================================================================
# Uso: .\scripts.ps1 <comando>
# Exemplo: .\scripts.ps1 dev
# ================================================================

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

function Show-Help {
    Write-Host ""
    Write-Host "===============================================" -ForegroundColor Cyan
    Write-Host "  NeuraPose - Comandos Disponíveis" -ForegroundColor Cyan
    Write-Host "===============================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  DESENVOLVIMENTO:" -ForegroundColor Yellow
    Write-Host "    .\scripts.ps1 backend     - Inicia o backend"
    Write-Host "    .\scripts.ps1 frontend    - Inicia o frontend"
    Write-Host "    .\scripts.ps1 tauri       - Inicia app Tauri (desktop)"
    Write-Host ""
    Write-Host "  DOCKER:" -ForegroundColor Yellow
    Write-Host "    .\scripts.ps1 docker-build  - Build das imagens Docker"
    Write-Host "    .\scripts.ps1 docker-up     - Sobe containers"
    Write-Host "    .\scripts.ps1 docker-down   - Para containers"
    Write-Host "    .\scripts.ps1 docker-logs   - Ver logs"
    Write-Host ""
    Write-Host "  UTILITÁRIOS:" -ForegroundColor Yellow
    Write-Host "    .\scripts.ps1 install     - Instala dependências"
    Write-Host "    .\scripts.ps1 health      - Verifica status do backend"
    Write-Host ""
}

function Start-Backend {
    Write-Host "[BACKEND] Iniciando servidor FastAPI..." -ForegroundColor Green
    uv run uvicorn neurapose_backend.main:app --reload --host 127.0.0.1 --port 8000
}

function Start-Frontend {
    Write-Host "[FRONTEND] Iniciando Vite..." -ForegroundColor Green
    Set-Location neurapose_tauri
    npm run dev
    Set-Location ..
}

function Start-Tauri {
    Write-Host "[TAURI] Iniciando aplicativo desktop..." -ForegroundColor Green
    Set-Location neurapose_tauri
    npm run tauri dev
    Set-Location ..
}

function Install-Dependencies {
    Write-Host "Instalando dependências do backend..." -ForegroundColor Cyan
    uv sync
    Write-Host ""
    Write-Host "Instalando dependências do frontend..." -ForegroundColor Cyan
    Set-Location neurapose_tauri
    npm install
    Set-Location ..
    Write-Host ""
    Write-Host "✓ Instalação concluída!" -ForegroundColor Green
}

function Docker-Build {
    Write-Host "Construindo imagens Docker..." -ForegroundColor Cyan
    docker-compose build
}

function Docker-Up {
    Write-Host "Subindo containers..." -ForegroundColor Cyan
    docker-compose up -d
    Write-Host ""
    Write-Host "✓ Backend: http://localhost:8000" -ForegroundColor Green
    Write-Host "✓ Frontend: http://localhost:1420" -ForegroundColor Green
    Write-Host "✓ Docs API: http://localhost:8000/docs" -ForegroundColor Green
}

function Docker-Down {
    Write-Host "Parando containers..." -ForegroundColor Yellow
    docker-compose down
}

function Docker-Logs {
    docker-compose logs -f
}

function Check-Health {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
        $response | ConvertTo-Json
    } catch {
        Write-Host "Backend offline" -ForegroundColor Red
    }
}

# Executar comando
switch ($Command.ToLower()) {
    "help"         { Show-Help }
    "backend"      { Start-Backend }
    "frontend"     { Start-Frontend }
    "tauri"        { Start-Tauri }
    "install"      { Install-Dependencies }
    "docker-build" { Docker-Build }
    "docker-up"    { Docker-Up }
    "docker-down"  { Docker-Down }
    "docker-logs"  { Docker-Logs }
    "health"       { Check-Health }
    default        { 
        Write-Host "Comando desconhecido: $Command" -ForegroundColor Red
        Show-Help
    }
}
