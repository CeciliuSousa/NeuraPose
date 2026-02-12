import threading
import cv2
from neurapose_backend.globals.state import state
from colorama import Fore

class FrameReaderThread(threading.Thread):
    def __init__(self, video_path, skip_interval, queue_out, max_frames=None):
        super().__init__()
        self.video_path = str(video_path)
        self.skip_interval = skip_interval
        self.queue_out = queue_out
        self.max_frames = max_frames
        self.stopped = False
        self.daemon = True
        
        self.cap = cv2.VideoCapture(self.video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def run(self):
        frame_idx = 0
        try:
            while self.cap.isOpened() and not self.stopped:
                if frame_idx % self.skip_interval != 0:
                    if not self.cap.grab(): break
                    frame_idx += 1
                    continue
                ret, frame = self.cap.read()
                if not ret: break
                self.queue_out.put((frame_idx, frame))
                frame_idx += 1

                if state and state.stop_requested: break
                
        except Exception as e:
            print(Fore.RED + f"[LEITURA] Erro: {e}")
        finally:
            self.cap.release()
            self.queue_out.put(None)

    def stop(self):
        self.stopped = True