import os
import sys
import cv2
import torch
import logging
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from colorama import Fore

# Importacoes do tracker
from neurapose_backend.tracker.rastreador import CustomBoTSORT, CustomReID, save_temp_tracker_yaml
from neurapose_backend.globals.state import state
import neurapose_backend.config_master as cm

# Suppress Logs
logging.getLogger("ultralytics").setLevel(logging.ERROR)
os.environ["YOLO_VERBOSE"] = "False"

# Monkey Patch Tracker
from ultralytics.trackers import bot_sort
bot_sort.BOTSORT = CustomBoTSORT
bot_sort.ReID = CustomReID

class YoloStreamDetector:
    def __init__(self):
        self.ensure_model()
        self.model = YOLO(str(cm.YOLO_PATH)).to(cm.DEVICE)
        
    def ensure_model(self):
        model_path = cm.YOLO_PATH
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if not model_path.exists():
            model_base = cm.YOLO_MODEL.replace('.pt', '')
            try:
                temp = YOLO(model_base)
                temp.save(str(model_path))
            except Exception as e:
                if model_path.exists():
                    os.remove(model_path)
                raise FileNotFoundError(f"Erro ao baixar {cm.YOLO_PATH}: {e}")

    def stream_video(self, video_path: str, batch_size=None):
        """
        Generator que processa o vídeo em batches e cede (yields) os resultados.
        
        Yields:
            tuple: (batch_index, frames_files, batch_results, track_data_partial)
                 - frames_files: Lista de frames numpy (do batch)
                 - batch_results: Lista de resultados do YOLO (boxes, ids)
        
        Returns (via generator return value logic or final property):
            track_data_complete
        """
        if batch_size is None:
            batch_size = cm.YOLO_BATCH_SIZE

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Init Tracker
        tracker = CustomBoTSORT(frame_rate=int(fps))
        self.model.tracker = tracker
        tracker_yaml_path = save_temp_tracker_yaml()
        
        print(Fore.CYAN + f"[INFO] STREAMING VIDEO YOLO ({batch_size} batch)...")
        
        track_data = {}
        frame_idx_global = 0
        last_progress = 0
        
        try:
            while True:
                if state.stop_requested:
                    print(Fore.YELLOW + "[STOP] Detecção interrompida.")
                    break
                    
                frames_batch = []
                for _ in range(batch_size):
                    ret, frame = cap.read()
                    if not ret: break
                    frames_batch.append(frame)
                    
                if not frames_batch:
                    break
                
                # Inference
                batch_results = self.model.track(
                    source=frames_batch,
                    imgsz=cm.YOLO_IMGSZ,
                    conf=cm.DETECTION_CONF,
                    device=cm.DEVICE,
                    persist=True,
                    tracker=str(tracker_yaml_path),
                    classes=[cm.YOLO_CLASS_PERSON],
                    verbose=False,
                    half=True,
                    stream=False
                )
                
                # Processamento leve para extrair features e montar track_data
                processed_batch_results = []
                
                for i, r in enumerate(batch_results):
                    current_frame_idx = frame_idx_global + i
                    boxes_data = None
                    
                    if r.boxes is not None and len(r.boxes) > 0:
                        boxes_data = r.boxes.data.cpu().numpy()
                        ids = boxes_data[:, 4] if boxes_data.shape[1] > 4 else None
                        current_time = current_frame_idx / fps
                        
                        if ids is not None:
                            for tid_raw in ids:
                                if tid_raw is None or tid_raw < 0: continue
                                tid = int(tid_raw)
                                
                                # Feature extraction seguro
                                try:
                                    trk = tracker.tracks[tid]
                                    f = trk.feat.copy() if hasattr(trk, 'feat') else np.zeros(512)
                                except:
                                    f = np.zeros(512, dtype=np.float32)
                                
                                if tid not in track_data:
                                    track_data[tid] = {
                                        "start": current_time,
                                        "end": current_time,
                                        "features": [f],
                                        "frames": {current_frame_idx},
                                    }
                                else:
                                    track_data[tid]["end"] = current_time
                                    track_data[tid]["frames"].add(current_frame_idx)
                                    if len(track_data[tid]["features"]) < 20:
                                        track_data[tid]["features"].append(f)
                    
                    # Armazena apenas o necessário para o consumidor (RTMPose)
                    # O 'r' original segura tensores na GPU, idealmente liberamos ou enviamos clones?
                    # O YOLO result 'r' tem .orig_img e outros metadados.
                    # Vamos passar 'boxes_data' limpo para evitar overhead de memória
                    processed_batch_results.append({
                        "boxes": boxes_data,
                        # Passamos o frame original para o RTMPose usar
                        "frame": frames_batch[i] 
                    })

                # Yield Batch
                yield (frame_idx_global, processed_batch_results)
                
                # Update loop state
                frame_idx_global += len(frames_batch)
                
                # Log Progresso YOLO
                prog = int((frame_idx_global / (total_frames or 1)) * 100)
                if prog >= last_progress + 10:
                    sys.stdout.write(f"\r{Fore.YELLOW}[YOLO Stream]{Fore.WHITE} Progresso: {prog} %")
                    sys.stdout.flush()
                    last_progress = prog
                    
        finally:
            cap.release()
            del self.model.tracker
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Retorna metadados finais (track_data para o merge posterior)
        # Como é um generator, podemos lançar isso via exceção controlada ou propriedade,
        # mas o python generator return value é acessível via StopIteration.
        # Uma forma mais limpa é o caller ter acesso ao objeto ou um yield final especial.
        # Vamos fazer um yield final especial.
        yield ("DONE", track_data, fps)
