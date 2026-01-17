# ==============================================================
# neurapose-backend/app/state.py
# ==============================================================

import threading

class ProcessingState:
    def __init__(self):
        self.is_running = False
        self.is_paused = False
        self.stop_requested = False
        self.current_frame = None
        self.current_process = None  # 'test', 'train', 'process', 'convert', 'split', etc.
        self.process_status = 'idle'  # 'idle', 'processing', 'success', 'error'
        self._lock = threading.Lock()

    def set_frame(self, frame):
        with self._lock:
            self.current_frame = frame

    def get_frame(self):
        with self._lock:
            return self.current_frame

    def reset(self):
        self.is_running = False
        self.is_paused = False
        self.stop_requested = False
        self.current_frame = None
        # Manter current_process e process_status até próximo processo
        
    def add_process(self, proc):
        """Adiciona um processo para monitoramento/shutdown."""
        if not hasattr(self, '_processes'):
            self._processes = []
        self._processes.append(proc)

    def kill_all_processes(self):
        """Mata todos os processos registrados."""
        if hasattr(self, '_processes'):
            for p in self._processes:
                try:
                    p.terminate()
                    p.kill()
                except:
                    pass
            self._processes = []

state = ProcessingState()
