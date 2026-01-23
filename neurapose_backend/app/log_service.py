# ==============================================================
# neurapose-backend/app/log_service.py
# ==============================================================

import sys
import re
from collections import deque
from typing import List, Optional

ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

class LogBuffer:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LogBuffer, cls).__new__(cls)
            # Dicionário de buffers por categoria (default, train, test, convert, etc.)
            cls._instance.buffers = {}
        return cls._instance
    
    def _get_buffer(self, category: str):
        if category not in self.buffers:
            self.buffers[category] = deque(maxlen=2000) # Buffer maior configurável
        return self.buffers[category]
    
    def write(self, message: str, category: str = "default"):
        if not message:
            return
        
        # Remove códigos ANSI de cor
        message = ANSI_ESCAPE.sub('', message)
            
        buffer = self._get_buffer(category)
        
        # Lógica para tratamento de progresso (\r)
        if "\r" in message:
            parts = [p for p in message.split("\r") if p.strip()]
            if not parts: return
            
            # Se começa com \r, assume que é uma atualização de linha (como TQDM)
            if message.startswith("\r") and buffer:
                buffer.pop()
            
            for part in parts:
                buffer.append(part)
        elif message.strip():
            buffer.append(message.strip())
            
    def get_logs(self, category: str = "default") -> List[str]:
        return list(self._get_buffer(category))
    
    def clear(self, category: Optional[str] = None):
        if category:
            if category in self.buffers:
                self.buffers[category].clear()
        else:
            for buf in self.buffers.values():
                buf.clear()

class CaptureOutput:
    """Context manager to redirect stdout/stderr to LogBuffer category"""
    def __init__(self, category: str = "default"):
        self.log_buffer = LogBuffer()
        self.category = category
        self.terminal_stdout = sys.stdout
        self.terminal_stderr = sys.stderr
        
    def write(self, message):
        self.terminal_stdout.write(message)
        self.log_buffer.write(message, self.category)
        
    def flush(self):
        self.terminal_stdout.flush()
        
    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.terminal_stdout
        sys.stderr = self.terminal_stderr
