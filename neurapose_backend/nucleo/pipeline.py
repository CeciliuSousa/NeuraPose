import time
import cv2
import torch
import sys
import threading
import queue
import traceback
from pathlib import Path
from colorama import Fore
import numpy as np
import random

import neurapose_backend.config_master as cm
from neurapose_backend.detector.yolo_stream import YoloStreamDetector
from neurapose_backend.detector.yolo_detector import merge_tracks
from neurapose_backend.nucleo.filtros import filtrar_ids_validos_v6, filtrar_ghosting_v5

# Import opcional para feedback de estado no App
try:
    from neurapose_backend.globals.state import state
except ImportError:
    state = None

class PipelineParalelo:
    def __init__(self, pose_extractor, verbose=True):
        self.pose_extractor = pose_extractor
        self.verbose = verbose
        # Fila limita memory usage se YOLO for muito mais rapido que Pose
        self.queue = queue.Queue(maxsize=4) 
        self.records = []
        self.error = None
        
    def worker_rtmpose(self):
        """Consome batches da fila e processa poses"""
        try:
            while True:
                item = self.queue.get()
                if item is None: # Sentinel
                    self.queue.task_done()
                    break
                
                # Desempacota
                frame_idx_start, batch_data = item
                
                # Processa cada frame do batch
                for i, data in enumerate(batch_data):
                    current_idx = frame_idx_start + i
                    frame_img = data["frame"]
                    boxes = data["boxes"]
                    
                    # ID Map temporario vazio pois o merge final ocorre depois
                    # RTMPose vai usar o ID cru do YOLO/BoTSORT
                    frame_regs, _ = self.pose_extractor.processar_frame(
                        frame_img=frame_img,
                        detections_yolo=boxes,
                        frame_idx=current_idx,
                        id_map={}, 
                        desenhar_no_frame=False
                    )
                    
                    self.records.extend(frame_regs)
                
                self.queue.task_done()
                
        except Exception as e:
            self.error = e
            traceback.print_exc()

    def executar(self, video_path, batch_size):
        # Inicializa Detector com Stream
        streamer = YoloStreamDetector()
        
        # Inicia Thread Consumidora
        t_consumer = threading.Thread(target=self.worker_rtmpose)
        t_consumer.start()
        
        generator = streamer.stream_video(video_path, batch_size)
        
        track_data_final = None
        fps = 30.0
        final_frame_idx = 0
        
        try:
            for item in generator:
                # O generator do yolo_stream yielda tuplas normais ou o sinal final
                if len(item) == 3 and item[0] == "DONE":
                    _, track_data_final, fps = item
                    continue
                
                # Se houver erro na thread, paramos
                if self.error:
                    break
                    
                # Yield normal: (idx, batch_results)
                # Envia para consumidor
                self.queue.put(item)
                final_frame_idx = item[0] + len(item[1])

            # Sinal de parada
            self.queue.put(None)
            
            # Aguarda fim do processamento de poses
            t_consumer.join()
            
            if self.error:
                raise self.error
                
        except Exception as e:
            # Em caso de crash, tenta limpar
            self.queue.put(None) # Libera thread se estiver presa
            t_consumer.join(timeout=2)
            raise e
            
        return self.records, track_data_final, final_frame_idx

def executar_pipeline_extracao(
    video_path_norm: Path,
    pose_extractor,
    batch_size: int = None,
    verbose: bool = True
):
    """
    Executa o pipeline otimizado com paralelismo (Producer-Consumer).
    """
    
    # Fixa seeds para determinismo
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    
    if batch_size is None:
        batch_size = cm.YOLO_BATCH_SIZE

    tempos = {"yolo": 0.0, "rtmpose": 0.0}

    # --------------------------------------------------------
    # 1. EXECUÇÃO PARALELA (YOLO + RTMPose)
    # --------------------------------------------------------
    if verbose:
        print(Fore.CYAN + f"[INFO] INICIANDO PIPELINE PARALELO (threads)...")
    
    t0_global = time.time()
    
    pipeline = PipelineParalelo(pose_extractor, verbose)
    
    # O "custo" de YOLO e RTMPose agora é entrelaçado.
    # Vamos medir o tempo total desse bloco
    
    try:
        raw_records, track_data, total_frames = pipeline.executar(video_path_norm, batch_size)
    except Exception as e:
        print(Fore.RED + f"[ERRO] Erro no pipeline paralelo: {e}")
        return [], {}, [], 0, tempos

    t_total_extracao = time.time() - t0_global
    
    # Para compatibilidade de logs, dividimos o tempo igualmente ou deixamos unificado
    # Como rodou junto, o "gargalo" define o tempo.
    # Vamos atribuir 50% para cada visualmente ou apenas registrar total
    tempos["yolo"] = t_total_extracao / 2  # Estimativa
    tempos["rtmpose"] = t_total_extracao / 2 # Estimativa

    if not raw_records or not track_data:
        if verbose: print(Fore.RED + "[AVISO] Nenhuma detecção ou track valido.")
        return [], {}, [], total_frames, tempos

    if verbose:
        print(Fore.GREEN + "\n[OK]" + Fore.WHITE + f" EXTRAÇÃO PARALELA CONCLUÍDA em {t_total_extracao:.2f}s!")
        print(Fore.CYAN + f"[INFO] CONSOLIDANDO IDs (Merge Tracks)...")

    # --------------------------------------------------------
    # 2. PÓS-PROCESSAMENTO (Merge Tracks + Remap IDs)
    # --------------------------------------------------------
    # Executa a fusão de IDs (BoTSORT logic para gaps)
    # Precisamos da função merge_tracks do yolo_detector
    merged_tracks, id_map_full = merge_tracks(track_data)
    
    # Atualiza IDs nos registros de pose
    # RTMPose rodou com IDs "crus", agora aplicamos o mapa final
    count_remapped = 0
    
    for r in raw_records:
        old_id = r["id_persistente"]
        if old_id in id_map_full:
            new_id = id_map_full[old_id]
            if new_id != old_id:
                r["id_persistente"] = new_id
                r["id"] = new_id # Atualiza ambos por segurança
                count_remapped += 1
                
    if verbose and count_remapped > 0:
        print(Fore.BLUE + f"[INFO] {count_remapped} registros tiveram IDs unificados.")

    # --------------------------------------------------------
    # 3. FILTRAGEM FINAL
    # --------------------------------------------------------
    if verbose:
        print(Fore.CYAN + f"[INFO] FILTRANDO GHOSTING E VALIDADE...")

    # A) Anti-Ghosting
    records_clean = filtrar_ghosting_v5(raw_records, iou_thresh=0.8)
    
    # B) Filtros de Validade
    ids_validos = filtrar_ids_validos_v6(
        registros=records_clean,
        min_frames=cm.MIN_FRAMES_PER_ID,
        min_dist=50.0,
        verbose=verbose
    )
    
    if verbose:
        print(Fore.YELLOW + "[NUCLEO]" + Fore.WHITE + f" IDs APROVADOS: {sorted(ids_validos)}")
    
    # C) Filtra registros finais
    registros_finais = [r for r in records_clean if r["id_persistente"] in ids_validos]
    
    return registros_finais, id_map_full, ids_validos, total_frames, tempos

