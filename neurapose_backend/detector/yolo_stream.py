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
        
        # Seleciona o modelo correto (Engine ou PT)
        if cm.USE_TENSORRT and self.engine_path.exists():
            print(Fore.GREEN + f"[INFO] Usando Modelo Otimizado (TensorRT): {self.engine_path.name}")
            # TensorRT deve ser carregado diretamente para a task 'track' ou 'predict'
            # Ultralytics carrega .engine automaticamente via YOLO('model.engine')
            self.model = YOLO(str(self.engine_path), task="detect")
        else:
            print(Fore.YELLOW + f"[INFO] Usando Modelo Padrão (PyTorch): {cm.YOLO_PATH.name}")
            self.model = YOLO(str(cm.YOLO_PATH)).to(cm.DEVICE)
        
    def ensure_model(self):
        # 1. Garante modelo PT base
        pt_path = cm.YOLO_PATH
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not pt_path.exists():
            print(Fore.CYAN + f"[INFO] Baixando modelo base: {pt_path.name}...")
            model_base = cm.YOLO_MODEL.replace('.pt', '')
            try:
                temp = YOLO(model_base)
                temp.save(str(pt_path))
            except Exception as e:
                if pt_path.exists(): os.remove(pt_path)
                raise FileNotFoundError(f"Erro ao baixar {cm.YOLO_PATH}: {e}")

        self.engine_path = pt_path.with_name(f"{pt_path.stem}_b{cm.YOLO_BATCH_SIZE}.engine")
        
        if cm.USE_TENSORRT:
            if not self.engine_path.exists():
                print(Fore.MAGENTA + "="*60)
                print(f"[OTIMIZAÇÃO] Gerando Motor TensorRT para {pt_path.name}...")
                print(f"Isso pode levar de 5 a 10 minutos. Por favor aguarde...")
                print("="*60 + Fore.RESET)
                try:
                    model = YOLO(str(pt_path))

                    batch_size = cm.YOLO_BATCH_SIZE
                    print(Fore.CYAN + f"[TRT] Exportando com Batch Size = {batch_size}...")
                    
                    model.export(
                        format="engine",
                        device=0,
                        half=True,
                        verbose=False,
                        batch=batch_size,
                        workspace=4         # Aumenta workspace para otimização (GB)
                    )
                    
                    # Renomeia Engine e ONNX para incluir o batch size
                    # Ultralytics gera: yolov8l.engine e yolov8l.onnx
                    # Queremos: yolov8l_b32.engine e yolov8l_b32.onnx
                    
                    default_engine = pt_path.with_suffix('.engine')
                    default_onnx = pt_path.with_suffix('.onnx')
                    
                    target_onnx = self.engine_path.with_suffix('.onnx')
                    
                    # Renomeia Engine force-overwrite
                    if default_engine.exists() and default_engine != self.engine_path:
                        if self.engine_path.exists(): self.engine_path.unlink()
                        default_engine.rename(self.engine_path)
                        
                    # Renomeia ONNX force-overwrite
                    if default_onnx.exists() and default_onnx != target_onnx:
                        if target_onnx.exists(): target_onnx.unlink()
                        default_onnx.rename(target_onnx)
                        
                    print(Fore.GREEN + "[SUCESSO] Modelo TensorRT gerado e versionado!")
                except Exception as e:
                    print(Fore.RED + f"[ERRO] Falha na exportação TensorRT: {e}")
                    print(Fore.YELLOW + "[INFO] Fallback para PyTorch.")
                    # Continua sem engine (vai usar .pt no init)
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
                
                # TensorRT Batch Padding Logic
                # Se o batch atual for menor que o esperado pelo Engine (ex: final do vídeo), preenchemos.
                actual_batch_size = len(frames_batch)
                padded_frames = frames_batch
                
                if cm.USE_TENSORRT and actual_batch_size < batch_size:
                    padding_needed = batch_size - actual_batch_size
                    # Cria frames pretos (zeros) com mesmo shape
                    # Assumindo que todos frames tem mesmo shape do primeiro
                    h, w, c = frames_batch[0].shape
                    black_frame = np.zeros((h, w, c), dtype=np.uint8)
                    padded_frames = frames_batch + [black_frame] * padding_needed

                # Inference
                batch_results = self.model.track(
                    source=padded_frames,
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
                
                # Remove padding results se necessário
                if cm.USE_TENSORRT and actual_batch_size < batch_size:
                    batch_results = batch_results[:actual_batch_size]
                
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
