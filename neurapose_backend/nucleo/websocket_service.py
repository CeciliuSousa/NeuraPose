# ==============================================================
# neurapose_backend/app/websocket_service.py
# ==============================================================
# Serviço WebSocket para push de logs em tempo real.
# Elimina polling de /logs e melhora responsividade.
# ==============================================================

import asyncio
import json
from typing import Set
from fastapi import WebSocket, WebSocketDisconnect
from neurapose_backend.nucleo.log_service import LogBuffer


class WebSocketManager:
    """
    Gerenciador de conexões WebSocket para broadcast de logs.
    
    Usage no main.py:
        ws_manager = WebSocketManager()
        
        @app.websocket("/ws/logs")
        async def ws_logs(websocket: WebSocket):
            await ws_manager.connect(websocket)
            try:
                while True:
                    # Envia logs pendentes
                    await ws_manager.send_logs(websocket, category="process")
                    await asyncio.sleep(0.5)
            except WebSocketDisconnect:
                ws_manager.disconnect(websocket)
    """
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._last_log_counts = {}  # {websocket_id: {category: count}}
    
    async def connect(self, websocket: WebSocket):
        """Aceita e registra uma nova conexão WebSocket."""
        await websocket.accept()
        self.active_connections.add(websocket)
        self._last_log_counts[id(websocket)] = {}
    
    def disconnect(self, websocket: WebSocket):
        """Remove uma conexão WebSocket."""
        self.active_connections.discard(websocket)
        self._last_log_counts.pop(id(websocket), None)
    
    async def send_logs(self, websocket: WebSocket, category: str = "default"):
        """
        Envia novos logs para um WebSocket específico.
        Envia apenas logs que ainda não foram enviados.
        """
        log_buffer = LogBuffer()
        all_logs = log_buffer.get_logs(category)
        
        ws_id = id(websocket)
        last_count = self._last_log_counts.get(ws_id, {}).get(category, 0)
        
        # Envia apenas logs novos
        new_logs = all_logs[last_count:]
        
        if new_logs:
            await websocket.send_json({
                "type": "logs",
                "category": category,
                "logs": new_logs,
                "total": len(all_logs)
            })
            
            # Atualiza contador
            if ws_id not in self._last_log_counts:
                self._last_log_counts[ws_id] = {}
            self._last_log_counts[ws_id][category] = len(all_logs)
    
    async def broadcast_logs(self, category: str = "default"):
        """Envia novos logs para todas as conexões ativas."""
        for websocket in list(self.active_connections):
            try:
                await self.send_logs(websocket, category)
            except Exception:
                self.disconnect(websocket)
    
    async def send_status(self, websocket: WebSocket, status: dict):
        """Envia status de processamento para um WebSocket."""
        await websocket.send_json({
            "type": "status",
            **status
        })
    
    async def broadcast_status(self, status: dict):
        """Envia status para todas as conexões ativas."""
        for websocket in list(self.active_connections):
            try:
                await self.send_status(websocket, status)
            except Exception:
                self.disconnect(websocket)


# Instância global para uso no main.py
ws_manager = WebSocketManager()
