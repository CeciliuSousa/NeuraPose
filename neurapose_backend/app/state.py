# ==============================================================
# neurapose-backend/app/state.py
# ==============================================================

import threading
import os

class ProcessingState:
    def __init__(self):
        self.is_running = False
        self.is_paused = False
        self.stop_requested = False
        self.show_preview = False  # Flag para ativar/desativar preview em tempo real
        self.current_frame = None
        self.current_process = None  # 'test', 'train', 'process', 'convert', 'split', etc.
        self.process_status = 'idle'  # 'idle', 'processing', 'success', 'error'
        self._lock = threading.Lock()
        self._processes = []
        self._current_thread = None

    def set_frame(self, frame):
        with self._lock:
            self.current_frame = frame
            # Debug: conta quantos frames foram setados
            if not hasattr(self, '_frame_count'):
                self._frame_count = 0
            self._frame_count += 1
            if self._frame_count % 100 == 1:  # Print a cada 100 frames
                # print(f"[DEBUG] state.set_frame() chamado - frame #{self._frame_count}")

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
        # Manter current_process e process_status até próximo processo
        
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
        
        # Mata processos registrados
        self.kill_all_processes()
        
        # Limpa o frame para parar o stream de vídeo
        self.current_frame = None
        
        print("[FORCE STOP] Parada forçada executada.")
    
    def emergency_exit(self):
        """Último recurso: encerra o processo Python forçadamente."""
        print("[EMERGENCY EXIT] Encerrando processo Python...")
        self.force_stop()
        os._exit(1)

state = ProcessingState()
