import threading

class ProcessingState:
    def __init__(self):
        self.is_running = False
        self.is_paused = False
        self.stop_requested = False
        self.current_frame = None
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

state = ProcessingState()
