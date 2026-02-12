import threading
import time
import psutil
import torch

class HardwareMonitorThread:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.running = False
        self.thread = None
        self.metrics = {
            "cpu": 0.0,
            "ram_used": 0.0,
            "ram_total": 0.0,
            "gpu_mem": 0.0,
            "gpu_total": 0.0
        }
        self.lock = threading.Lock()

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def _run(self):
        while self.running:
            try:
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory()
                gpu_mem = 0.0
                gpu_total = 0.0
                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info()
                    gpu_mem = (total - free) / (1024**3)
                    gpu_total = total / (1024**3)
                
                with self.lock:
                    self.metrics["cpu"] = cpu
                    self.metrics["ram_used"] = ram.used / (1024**3)
                    self.metrics["ram_total"] = ram.total / (1024**3)
                    self.metrics["gpu_mem"] = gpu_mem
                    self.metrics["gpu_total"] = gpu_total
            except Exception:
                pass
            
            time.sleep(self.interval)

    def get_metrics(self):
        with self.lock:
            return self.metrics.copy()

monitor = HardwareMonitorThread(interval=2.0)
