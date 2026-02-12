# ==============================================================
# neurapose_backend/global/state.py
# ==============================================================
# Módulo global para gerenciamento de estado entre app e pre_processamento

import threading
import os


class ProcessingState:
    """Classe para gerenciar estado global de processamento."""  
    def __init__(self):
        self.is_running = False
        self.is_paused = False
        self.stop_requested = False
        self.show_preview = False
        self.current_frame = None
        self.current_process = None
        self.process_status = 'idle'
        self._lock = threading.Lock()
        self._processes = []
        self._current_thread = None

    def set_frame(self, frame):
        with self._lock:
            self.current_frame = frame
            if not hasattr(self, '_frame_count'):
                self._frame_count = 0
            self._frame_count += 1
            if self._frame_count % 100 == 1:
                pass

    def get_frame(self):
        with self._lock:
            return self.current_frame

    def reset(self):
        self.is_running = False
        self.is_paused = False
        self.stop_requested = False
        self.show_preview = False
        self.current_frame = None
        self._current_thread = None
        
    def set_current_thread(self, thread):
        """Registra a thread atual de processamento."""
        self._current_thread = thread
        
    def add_process(self, proc):
        """Adiciona um processo para monitoramento/shutdown."""
        self._processes.append(proc)

    def kill_all_processes(self):
        """Mata todos os processos registrados."""
        for p in self._processes:
            try:
                p.terminate()
                p.kill()
            except:
                pass
        self._processes = []
    
    def force_stop(self):
        """Força a parada de qualquer processamento de forma agressiva."""
        self.stop_requested = True
        self.is_running = False
        self.process_status = 'idle'
        
        self.kill_all_processes()
        
        self.current_frame = None
        
        print("[FORCE STOP] Parada forçada executada.")
        
    def request_stop(self):
        print(f"[DEBUG] Stop requested! Called from somewhere.")
        self.stop_requested = True
    
    def emergency_exit(self):
        """Último recurso: encerra o processo Python forçadamente."""
        print("[EMERGENCY EXIT] Encerrando processo Python...")
        self.force_stop()
        os._exit(1)

state = ProcessingState()
