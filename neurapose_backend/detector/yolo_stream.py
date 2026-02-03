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
from neurapose_backend.tracker.rastreador import CustomBoTSORT, CustomReID, CustomDeepOCSORT, save_temp_tracker_yaml
from neurapose_backend.globals.state import state
import neurapose_backend.config_master as cm

# Suppress Logs
logging.getLogger("ultralytics").setLevel(logging.ERROR)
os.environ["YOLO_VERBOSE"] = "False"

# Monkey Patch Tracker
from ultralytics.trackers import bot_sort
bot_sort.BOTSORT = CustomBoTSORT
bot_sort.ReID = CustomReID

import threading
import queue
import time

class ThreadedVideoLoader:
    def __init__(self, path, buffer_size=128):
        self.cap = cv2.VideoCapture(str(path))
        self.q = queue.Queue(maxsize=buffer_size)
        self.stopped = False
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Start reading thread
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()

    def update(self):
        while True:
            if self.stopped:
                break
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    # Push None to signal end
                    self.q.put(None) 
                    break
                self.q.put(frame)
            else:
                time.sleep(0.005) # avoid busy wait
        self.cap.release()

    def read(self):
        # returns ret, frame like cv2
        if self.q.empty() and self.stopped:
            return False, None
        
        try:
            # wait with timeout to allow checking stopped status
            frame = self.q.get(timeout=1.0) 
        except queue.Empty:
            return False, None

        if frame is None:
            return False, None
            
        return True, frame

    def release(self):
        self.stopped = True
        if self.t.is_alive():
            self.t.join()
        if self.cap.isOpened():
            self.cap.release()


class YoloStreamDetector:
    def __init__(self):
        self.ensure_model()
        
        # Seleciona o modelo correto (Engine ou PT)
        if cm.USE_TENSORRT and self.engine_path.exists():
            print(Fore.GREEN + f"[INFO] MODELO OTIMIZADO (TensorRT): {self.engine_path.name}")
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
                
                # Cleanup: YOLO baixa no CWD, então removemos o arquivo solto
                local_file = Path(f"{model_base}.pt")
                if local_file.exists():
                    local_file.unlink()
                    print(Fore.CYAN + f"[INFO] Arquivo temporário removido: {local_file.name}")
                    
            except Exception as e:
                if pt_path.exists(): os.remove(pt_path)
                raise FileNotFoundError(f"Erro ao baixar {cm.YOLO_PATH}: {e}")

        self.engine_path = pt_path.with_name(f"{pt_path.stem}-batch{cm.YOLO_BATCH_SIZE}.engine")
        
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
                    
                    # Renomeia Engine e ONNX para o padrão: modelo-batchN.engine/onnx
                    # Ultralytics gera: yolov8l.engine e yolov8l.onnx
                    # Queremos: yolov8l-batch32.engine e yolov8l-batch32.onnx
                    
                    default_engine = pt_path.with_suffix('.engine')
                    default_onnx = pt_path.with_suffix('.onnx')
                    
                    target_onnx = self.engine_path.with_suffix('.onnx')
                    
                    # Renomeia Engine force-overwrite
                    if default_engine.exists() and default_engine != self.engine_path:
                        if self.engine_path.exists(): self.engine_path.unlink()
                        default_engine.rename(self.engine_path)
                        print(Fore.GREEN + f"[TRT] Engine salvo: {self.engine_path.name}")
                        
                    # Renomeia ONNX force-overwrite
                    if default_onnx.exists() and default_onnx != target_onnx:
                        if target_onnx.exists(): target_onnx.unlink()
                        default_onnx.rename(target_onnx)
                        print(Fore.GREEN + f"[TRT] ONNX salvo: {target_onnx.name}")
                        
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

        # Escolha do Loader (Assíncrono vs Síncrono)
        buffer_size = getattr(cm, 'ASYNC_BUFFER_SIZE', 128)
        use_async = getattr(cm, 'USE_ASYNC_LOADER', True)
        
        if use_async:
            print(Fore.CYAN + f"[INFO] INICIANDO LEITURA ASSÍNCRONA (Buffer={buffer_size})...")
            loader = ThreadedVideoLoader(str(video_path), buffer_size=buffer_size)
            fps = loader.fps
            total_frames = loader.total_frames
            # Mantemos interface 'read' compatível
            cap_read = loader.read 
            cap_release = loader.release
        else:
            print(Fore.YELLOW + f"[INFO] Iniciando Leitura Síncrona (OpenCV)...")
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_read = cap.read
            cap_release = cap.release
        
        # =========================================================================
        # SELEÇÃO DINÂMICA DE TRACKER
        # =========================================================================
        USING_DEEPOCSORT = (cm.TRACKER_NAME.upper() == "DEEPOCSORT")
        
        deep_tracker = None
        tracker = None # BoTSORT

        if USING_DEEPOCSORT:
             # DeepOCSORT roda frame a frame
            deep_tracker = CustomDeepOCSORT()
        else:
             # BoTSORT Injetado
            tracker = CustomBoTSORT(frame_rate=int(fps))
            self.model.tracker = tracker
            tracker_yaml_path = save_temp_tracker_yaml()
        
        print(Fore.CYAN + f"[INFO] STREAMING VIDEO YOLO ({batch_size} batch) | Tracker: {cm.TRACKER_NAME}...")
        
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
                    ret, frame = cap_read()
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

                # Skip-Frame Logic
                processed_batch_results = []
                skip_val = getattr(cm, 'YOLO_SKIP_FRAME_INTERVAL', 1)
                
                # ============================================================
                # BATCH INFERENCE LOGIC (TensorRT vs PyTorch)
                # ============================================================
                # TensorRT: Batch fixo obrigatório. Coletamos frames, fazemos padding, inferência única.
                # PyTorch: Batch dinâmico. Podemos processar frame a frame.
                
                # Pré-computa quais frames precisam de YOLO
                yolo_frame_indices = []
                yolo_frames = []
                for i, frame in enumerate(frames_batch):
                    current_frame_idx = frame_idx_global + i
                    do_yolo = (skip_val <= 1) or (current_frame_idx % skip_val == 0)
                    if do_yolo:
                        yolo_frame_indices.append(i)
                        yolo_frames.append(frame)
                
                # Mapa de detecções: índice local -> dets
                detections_map = {}
                
                if yolo_frames:
                    try:
                        if cm.USE_TENSORRT:
                            # TensorRT: Batch fixo obrigatório
                            # Padding para completar o batch size do engine
                            trt_batch_size = batch_size  # Engine foi compilado com este valor
                            num_yolo_frames = len(yolo_frames)
                            
                            if num_yolo_frames < trt_batch_size:
                                # Padding com frames pretos
                                h, w, c = yolo_frames[0].shape
                                black_frame = np.zeros((h, w, c), dtype=np.uint8)
                                padded_frames = yolo_frames + [black_frame] * (trt_batch_size - num_yolo_frames)
                            else:
                                padded_frames = yolo_frames[:trt_batch_size]
                            
                            if not USING_DEEPOCSORT:
                                res_batch = self.model.predict(
                                    source=padded_frames,
                                    imgsz=cm.YOLO_IMGSZ,
                                    conf=cm.DETECTION_CONF,
                                    device=cm.DEVICE,
                                    classes=[cm.YOLO_CLASS_PERSON],
                                    verbose=False,
                                    stream=False
                                )
                            else:
                                # DeepOCSORT roda YOLO internamente frame-a-frame
                                res_batch = []
                                
                            # Inicializa mapa de detecções vazio para evitar erro de referência
                            # detections_map[local_i] = np.empty((0, 6)) # REMOVIDO: Erro de indentação e lógica deslocada
                            
                            # Preenche results para BoTSORT
                            if not USING_DEEPOCSORT:
                                # Extrai detecções apenas dos frames reais (não padding)
                                for idx, local_i in enumerate(yolo_frame_indices):
                                    if idx < len(res_batch) and len(res_batch[idx].boxes) > 0:
                                        raw_data = res_batch[idx].boxes.data.cpu().numpy()
                                        # Normaliza para 6 colunas se necessário
                                        if raw_data.shape[1] == 4:
                                            rows = raw_data.shape[0]
                                            confs = np.full((rows, 1), 0.85, dtype=np.float32)
                                            clss = np.zeros((rows, 1), dtype=np.float32)
                                            raw_data = np.hstack((raw_data, confs, clss))
                                        elif raw_data.shape[1] == 5:
                                            rows = raw_data.shape[0]
                                            clss = np.zeros((rows, 1), dtype=np.float32)
                                            raw_data = np.hstack((raw_data, clss))
                                        detections_map[local_i] = raw_data
                                    else:
                                        detections_map[local_i] = np.empty((0, 6))

                        else:
                            # PyTorch: Processa frame a frame (dinâmico)
                            for idx, local_i in enumerate(yolo_frame_indices):
                                frame = yolo_frames[idx]
                                
                                # SE BOTSORT: Roda YOLO Detect
                                if not USING_DEEPOCSORT:
                                    res = self.model.predict(
                                        source=frame,
                                        imgsz=cm.YOLO_IMGSZ,
                                        conf=cm.DETECTION_CONF,
                                        device=cm.DEVICE,
                                        classes=[cm.YOLO_CLASS_PERSON],
                                        verbose=False,
                                        stream=False
                                    )
                                    if len(res) > 0 and len(res[0].boxes) > 0:
                                        detections_map[local_i] = res[0].boxes.data.cpu().numpy()
                                    else:
                                        detections_map[local_i] = np.empty((0, 6))
                                else:
                                    # Se DEEPOCSORT, não rodamos YOLO aqui, pois o tracker roda internamente
                                    pass
                    except Exception as e:
                        print(Fore.RED + f"[ERRO] Falha na inferência YOLO batch: {e}")
                        # Fallback: todos os frames sem detecção
                        for local_i in yolo_frame_indices:
                            detections_map[local_i] = np.empty((0, 6))
                
                # ============================================================
                # TRACKING LOOP (Sequencial por frame para manter consistência temporal)
                # ============================================================
                for i, frame in enumerate(frames_batch):
                    current_frame_idx = frame_idx_global + i
                    tracks = np.empty((0, 7))
                    
                    try:
                        # Obtém detecções (do batch ou vazio se foi skip)
                        dets = detections_map.get(i, np.empty((0, 6)))
                        
                        # ============================================================
                        # NORMALIZAÇÃO DE COLUNAS (TensorRT retorna 4, Tracker exige 6)
                        # ============================================================
                        if isinstance(dets, np.ndarray) and len(dets) > 0:
                            ncols = dets.shape[1]
                            
                            # Se o array tiver formato (N, 4) -> [x1, y1, x2, y2]
                            if ncols == 4:
                                rows = dets.shape[0]
                                # Criar coluna de Confiança (0.85 padrão para TensorRT filtrado)
                                confs = np.full((rows, 1), 0.85, dtype=np.float32)
                                # Criar coluna de Classe (0 -> Pessoa)
                                clss = np.zeros((rows, 1), dtype=np.float32)
                                # Juntar: Agora temos (N, 6)
                                dets = np.hstack((dets, confs, clss))
                                
                            # Se o array tiver formato (N, 5) -> falta classe
                            elif ncols == 5:
                                rows = dets.shape[0]
                                clss = np.zeros((rows, 1), dtype=np.float32)
                                dets = np.hstack((dets, clss))
                        
                        # ============================================================
                        # SMARTSKIP: Usa Kalman Prediction em frames pulados
                        # ============================================================
                        is_yolo_frame = i in detections_map and len(detections_map.get(i, [])) > 0
                        
                        if USING_DEEPOCSORT:
                            # --- DeepOCSORT (Frame-a-Frame Inteiro) ---
                            # O usuário configurou CustomDeepOCSORT para rodar YOLO + Tracker em um passo
                            # Então passamos o frame cru e recebemos os tracks
                            if is_yolo_frame: # Respeita o skip-frame?
                                 # Sim, só roda se for frame de yolo. Nos frames pulados, o que fazemos?
                                 # DeepOCSORT nativo boxmot não tem predict_only explícito público fácil na classe wrapper do usuário
                                 # Mas vamos assumir que chamamos 'update' com vazio se skipar ou repetimos?
                                 # Vamos chamar .track(frame)
                                 
                                 # Nota: CustomDeepOCSORT.track retorna ndarray [x,y,x,y,id,conf,cls]
                                 tracks = deep_tracker.track(frame)
                            else:
                                 # Skip frame logic para DeepOCSORT
                                 # Se o usuário não implementou predict_only, passamos vazio ou repetimos?
                                 # Vamos passar vazio para manter vivo
                                 # O wrapper do usuário trata vazio: return self.tracker.update(np.empty((0, 6)), frame)
                                 # Mas aqui não temos detecções.
                                 # Idealmente, DeepOCSORT deve rodar em TODOS os frames para Kalman funcionar bem.
                                 # Ignorar YOLO_SKIP_FRAME_INTERVAL para DeepOCSORT é mais seguro.
                                 tracks = deep_tracker.track(frame)
                        else:
                            # --- BoTSORT (Padrão) ---
                            if is_yolo_frame:
                                # Frame com detecção YOLO: Update completo
                                tracks = tracker.update(dets, frame)
                            elif len(dets) == 0 and hasattr(tracker, 'predict_only'):
                                # Frame PULADO: Usa Kalman Predict (Interpolação Suave)
                                tracks = tracker.predict_only()
                            else:
                                # Fallback: Update com dets vazio
                                tracks = tracker.update(dets, frame)
                        
                    except Exception as e:
                        print(Fore.RED + f"[ERRO] Falha no tracking frame {current_frame_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

                    # 3. Processa Resultados para Compatibilidade
                    boxes_data = None
                    if len(tracks) > 0:
                        # BoTSORT retorna [x1, y1, x2, y2, id, conf, cls]
                        boxes_data = tracks
                        
                    # Extração de Features e Track Data
                    if boxes_data is not None:
                        ids = boxes_data[:, 4]
                        current_time = current_frame_idx / fps
                        
                        for idx_in_batch, row in enumerate(boxes_data):
                            tid = int(row[4])
                            if tid < 0: continue
                            
                            # Feature extraction seguro
                            if not USING_DEEPOCSORT:
                                try:
                                    trk = tracker.tracks[tid]
                                    f = trk.feat.copy() if hasattr(trk, 'feat') else np.zeros(512)
                                except:
                                    f = np.zeros(512, dtype=np.float32)
                            else:
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
                    processed_batch_results.append({
                        "boxes": boxes_data,
                        # Passamos o frame original para o RTMPose usar
                        "frame": frame 
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
            cap_release()
            if not USING_DEEPOCSORT:
                 del self.model.tracker
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Retorna metadados finais (track_data para o merge posterior)
        # Como é um generator, podemos lançar isso via exceção controlada ou propriedade,
        # mas o python generator return value é acessível via StopIteration.
        # Uma forma mais limpa é o caller ter acesso ao objeto ou um yield final especial.
        # Vamos fazer um yield final especial.
        yield ("DONE", track_data, fps)
