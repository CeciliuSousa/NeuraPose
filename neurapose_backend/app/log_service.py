import sys
import io
from collections import deque
from typing import List
import logging

class LogBuffer:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LogBuffer, cls).__new__(cls)
            cls._instance.logs = deque(maxlen=1000) # Keep last 1000 lines
        return cls._instance
    
    def write(self, message: str):
        if not message:
            return
            
        # tqdm e outros usam \r para atualizar a mesma linha
        if "\r" in message:
            # Dividimos por \r para pegar as atualizações
            parts = [p for p in message.split("\r") if p.strip()]
            if not parts: return
            
            # Se a mensagem começa com \r, removemos a última linha do buffer (atualização)
            if message.startswith("\r") and self._instance.logs:
                self._instance.logs.pop()
            
            # Adicionamos as partes (geralmente só a última nos interessa)
            for part in parts:
                self._instance.logs.append(part)
        elif message.strip():
            self._instance.logs.append(message.strip())
            
    def get_logs(self) -> List[str]:
        return list(self._instance.logs)
    
    def clear(self):
        self._instance.logs.clear()

class CaptureOutput:
    """Context manager to redirect stdout/stderr to LogBuffer"""
    def __init__(self):
        self.log_buffer = LogBuffer()
        self.terminal_stdout = sys.stdout
        self.terminal_stderr = sys.stderr
        
    def write(self, message):
        self.terminal_stdout.write(message)
        self.log_buffer.write(message)
        
    def flush(self):
        self.terminal_stdout.flush()
        
    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.terminal_stdout
        sys.stderr = self.terminal_stderr
