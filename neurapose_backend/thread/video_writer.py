import threading
import cv2
from colorama import Fore
import neurapose_backend.config_master as cm

class VideoWriterThread(threading.Thread):
    def __init__(self, output_path, fps, width, height, queue_in):
        super().__init__()
        self.output_path = str(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        self.queue_in = queue_in
        self.stopped = False
        self.daemon = True
        self.writer = None

    def _init_writer(self):
        if cm.USE_NVENC:
            try:
                pass
            except: pass

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        
        if not self.writer.isOpened():
             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
             self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))

    def run(self):
        self._init_writer()
        try:
            while not self.stopped:
                item = self.queue_in.get()
                if item is None:
                    break
                
                frame = item
                if self.writer is not None:
                    self.writer.write(frame)
                
                self.queue_in.task_done()
        except Exception as e:
            print(Fore.RED + f"[ESCRITA] Erro: {e}")
        finally:
            if self.writer: self.writer.release()

    def stop(self):
        self.stopped = True