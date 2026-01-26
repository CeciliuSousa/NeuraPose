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
            self.buffers[category] = deque(maxlen=10000) # Buffer aumentado para suportar stream de progresso
        return self.buffers[category]
    
    def write(self, message: str, category: str = "default"):
        if not message:
            return
        
        # Remove códigos ANSI de cor
        message = ANSI_ESCAPE.sub('', message)
            
        buffer = self._get_buffer(category)
        
        # Lógica para tratamento de progresso (\r) - MODIFICADO: Append only (sem pop)
        if "\r" in message:
            parts = message.split('\r')
            for i, part in enumerate(parts):
                if part.strip():
                    # Adiciona \r se não for o primeiro elemento ou se a msg original começava com \r
                    # Isso garante que o frontend receba o sinal de "substituição"
                    is_update = (i > 0) or message.startswith('\r')
                    final_msg = f"\r{part.strip()}" if is_update else part.strip()
                    buffer.append(final_msg)
                    
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
        self._buffer = ""

    def write(self, message):
        self.terminal_stdout.write(message)
        self._buffer += message
        
        # Processa linhas completas ou updates de progresso
        while True:
            nl = self._buffer.find('\n')
            cr = self._buffer.find('\r')
            
            if nl == -1 and cr == -1:
                break
                
            # Determina o separador mais próximo
            if nl != -1 and (cr == -1 or nl < cr):
                limit = nl
                is_cr = False
            else:
                limit = cr
                is_cr = True
            
            # Extrai o "chunk" de texto até o separador
            chunk = self._buffer[:limit]
            
            # Se for CR, repassamos o \r para o LogBuffer tratar a animação
            full_msg = chunk + ('\r' if is_cr else '')
            
            # Envia se tiver conteúdo relevante ou for comando de controle
            if full_msg.strip() or is_cr:
                self.log_buffer.write(full_msg, self.category)
            
            # Remove o processado do buffer (pula o separador)
            self._buffer = self._buffer[limit+1:]

    def flush(self):
        self.terminal_stdout.flush()
        # Se sobrou algo no buffer ao final, envia
        if self._buffer and self._buffer.strip():
            self.log_buffer.write(self._buffer, self.category)
            self._buffer = ""
        
    def __enter__(self):
        # Evita aninhamento de CaptureOutput (prevencao de log duplo)
        if hasattr(sys.stdout, "terminal_stdout"):
            self.terminal_stdout = sys.stdout.terminal_stdout
        else:
            self.terminal_stdout = sys.stdout
            
        if hasattr(sys.stderr, "terminal_stderr"):
            self.terminal_stderr = sys.stderr.terminal_stderr
        else:
            self.terminal_stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.terminal_stdout
        sys.stderr = self.terminal_stderr
