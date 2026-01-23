# ==============================================================
# neurapose_backend/nucleo/video_reader.py
# ==============================================================
# Leitor de vídeo otimizado com pre-fetch de frames em thread separada.
# Elimina tempo de espera de I/O durante processamento GPU.
# ==============================================================

import cv2
import threading
import queue
from pathlib import Path
from typing import Optional, Tuple
import neurapose_backend.config_master as cm


class VideoReaderAsync:
    """
    Leitor de vídeo com pre-fetch assíncrono de frames.
    
    Usa uma thread separada para ler frames do disco enquanto
    a thread principal processa o frame atual na GPU.
    
    Usage:
        with VideoReaderAsync(video_path) as reader:
            for frame_idx, frame in reader:
                # Processa frame
                pass
    """
    
    def __init__(self, video_path: Path, buffer_size: int = None):
        """
        Args:
            video_path: Caminho do vídeo.
            buffer_size: Tamanho do buffer de frames (default: cm.PREFETCH_BUFFER_SIZE).
        """
        self.video_path = Path(video_path)
        self.buffer_size = buffer_size or cm.PREFETCH_BUFFER_SIZE
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._queue: Optional[queue.Queue] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        
        # Propriedades do vídeo (preenchidas no __enter__)
        self.fps: float = 0.0
        self.total_frames: int = 0
        self.width: int = 0
        self.height: int = 0
        
    def __enter__(self):
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise IOError(f"Não foi possível abrir o vídeo: {self.video_path}")
        
        # Lê propriedades
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Inicializa queue e thread
        self._queue = queue.Queue(maxsize=self.buffer_size)
        self._stop_event = threading.Event()
        
        if cm.USE_PREFETCH:
            self._thread = threading.Thread(target=self._read_frames, daemon=True)
            self._thread.start()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def _read_frames(self):
        """Thread worker que lê frames e coloca na queue."""
        frame_idx = 0
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                # Sinaliza fim do vídeo
                self._queue.put((None, None))
                break
            
            frame_idx += 1
            try:
                self._queue.put((frame_idx, frame), timeout=1.0)
            except queue.Full:
                # Buffer cheio, espera
                if self._stop_event.is_set():
                    break
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[int, any]:
        if cm.USE_PREFETCH:
            # Modo async: busca da queue
            try:
                frame_idx, frame = self._queue.get(timeout=5.0)
                if frame is None:
                    raise StopIteration
                return frame_idx, frame
            except queue.Empty:
                raise StopIteration
        else:
            # Modo sync: lê diretamente
            ret, frame = self._cap.read()
            if not ret:
                raise StopIteration
            # Contador manual
            if not hasattr(self, '_sync_frame_idx'):
                self._sync_frame_idx = 0
            self._sync_frame_idx += 1
            return self._sync_frame_idx, frame
    
    def close(self):
        """Libera recursos."""
        if self._stop_event:
            self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        
        if self._cap:
            self._cap.release()
            self._cap = None
