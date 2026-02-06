# # neurapose_backend/detector/yolo_detector.py

# import os
# import sys
# import cv2
# import torch
# import logging
# import numpy as np
# from pathlib import Path
# from ultralytics import YOLO

# # Importacoes do tracker
# from neurapose_backend.tracker.rastreador import CustomBoTSORT, CustomReID, CustomDeepOCSORT, save_temp_tracker_yaml

# from neurapose_backend.globals.state import state
# from colorama import Fore

# import neurapose_backend.config_master as cm

# logging.getLogger("ultralytics").setLevel(logging.ERROR)
# os.environ["YOLO_VERBOSE"] = "False"

# from ultralytics.trackers import bot_sort

# bot_sort.BOTSORT = CustomBoTSORT
# bot_sort.ReID = CustomReID


# # 1. Fusao de IDs
# # ================================================================
# def merge_tracks(track_data, gap_thresh=1.5):
#     """
#     track_data: dict[track_id] -> {start, end, frames, features}
#     Retorna:
#       merged_tracks: dict[id_original] -> {start, end, aliases}
#       id_map: dict[id_qualquer] -> id_original

#     Regra:
#       - Nao usa embedding para decidir fusao.
#       - So usa tempo (start/end):
#           overlap = not (end_a < start_b or end_b < start_a)
#           gap = start_b - end_a
#           if not overlap and 0 < gap < gap_thresh -> funde id_b em id_a
#     """
#     merged_tracks = {}
#     used = set()

#     # Ordena por instante de inicio
#     ids_sorted = sorted(track_data.keys(), key=lambda tid: track_data[tid]["start"])

#     for i, id_a in enumerate(ids_sorted):
#         if id_a in used:
#             continue

#         data_a = track_data[id_a]
#         merged_tracks[id_a] = {
#             "start": data_a["start"],
#             "end": data_a["end"],
#             "aliases": []
#         }
#         used.add(id_a)

#         # Compara com os IDs seguintes na lista (id_b comeca depois de id_a)
#         for id_b in ids_sorted[i + 1:]:
#             if id_b in used:
#                 continue

#             data_b = track_data[id_b]

#             start_a = merged_tracks[id_a]["start"]
#             end_a = merged_tracks[id_a]["end"]
#             start_b = data_b["start"]
#             end_b = data_b["end"]

#             # Mesmo criterio de overlap do codigo original
#             overlap = not (end_a < start_b or end_b < start_a)
#             gap = start_b - end_a  # importante: b depois de a

#             if not overlap and 0 < gap < gap_thresh:
#                 merged_tracks[id_a]["aliases"].append(id_b)
#                 merged_tracks[id_a]["end"] = max(end_a, end_b)
#                 used.add(id_b)

#     # Cria o mapa id_atual -> id_persistente
#     id_map = {}
#     for orig, data in merged_tracks.items():
#         id_map[int(orig)] = int(orig)
#         for alias in data["aliases"]:
#             id_map[int(alias)] = int(orig)

#     return merged_tracks, id_map


# # ================================================================
# # 2. YOLO + BoTSORT + coleta de IDs + fusao de IDs
# # ================================================================
# def yolo_detector_botsort(videos_dir=None, batch_size=None):
#     """
#     Roda YOLOv8x + CustomBoTSORT em varios videos com Processamento em LOTE.
#     Retorna lista com:
#         - video
#         - fps
#         - track_data
#         - merged_tracks
#         - id_map
#         - results (frame a frame)
#     """

#     videos_path = Path(videos_dir or (cm.ROOT / "videos"))

#     if batch_size is None:
#         batch_size = cm.YOLO_BATCH_SIZE

#     # ================================================================
#     # MODELO YOLO - Usar caminho centralizado (config_master) e baixar se nao existir
#     # ================================================================
#     model_path = cm.YOLO_PATH
#     model_path.parent.mkdir(parents=True, exist_ok=True)

#     if not model_path.exists():
#         model_base = cm.YOLO_MODEL.replace('.pt', '')
#         try:
#             temp_model = YOLO(model_base, task='detect')
#             temp_model.save(str(model_path))
            
#             # Cleanup: YOLO baixa no CWD, então removemos o arquivo solto
#             local_file = Path(f"{model_base}.pt")
#             if local_file.exists():
#                 local_file.unlink()
#                 print(Fore.CYAN + f"[INFO] Arquivo temporário removido: {local_file.name}")
                
#         except Exception as e:
#             if model_path.exists():
#                 os.remove(model_path)
#             raise FileNotFoundError(f"Erro ao baixar {cm.YOLO_PATH}: {e}")
    
#     # Carrega o modelo
#     model = YOLO(str(model_path), task='detect').to(cm.DEVICE)

#     # Lista de videos
#     if videos_path.is_file():
#         videos = [videos_path]
#     else:
#         videos = [
#             v for v in videos_path.glob("*")
#             if v.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]
#         ]

#     if not videos:
#         print("[WARN] Nenhum video encontrado.")
#         return []

#     resultados_finais = []

#     for video in videos:
#         cap = cv2.VideoCapture(str(video))
#         fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#         # Reset Tracker State for new video
#         tracker = CustomBoTSORT(frame_rate=int(fps))
#         model.tracker = tracker
        
#         # Configure Tracker YAML
#         tracker_yaml_path = save_temp_tracker_yaml()
        
#         print(Fore.CYAN + f"[INFO] PROCESSANDO VIDEO COM YOLO + {cm.TRACKER_NAME.upper()}...")
#         sys.stdout.flush()

#         # =========================================================================
#         # SELEÇÃO DINÂMICA DE TRACKER
#         # =========================================================================
#         USING_DEEPOCSORT = (cm.TRACKER_NAME.upper() == "DEEPOCSORT")
        
#         deep_tracker = None
#         if USING_DEEPOCSORT:
#             print(Fore.MAGENTA + "[TRACKER] Inicializando DeepOCSORT (BoxMOT)...")
#             # DeepOCSORT roda frame a frame, não injetado no YOLO
#             deep_tracker = CustomDeepOCSORT()
#             # YOLO model inside DeepOCSORT wrapper handles inference
#             # We will use deep_tracker.track(frame) inside the loop
#         else:
#             # BoTSORT Default (Injetado no YOLO)
#             tracker = CustomBoTSORT(frame_rate=int(fps))
#             model.tracker = tracker


#         results = []
        
#         # ... (Loop code not shown here, assumed handled by caller or context) ...
#         # But we need to update the PRINT inside the loop if we can see it in previous views (we can).
#         # Wait, I need to use 'view_file' first to be safe or rely on recent edits.
#         # I saw lines 252 in previous outputs.
#         # I'll replacing the start block first.

#         # Note: The `replace_file_content` below targets the progress print which is inside the loop.
#         # I'll try to target the progress block if I can match context.

#         track_data = {}
        
#         frame_idx_global = 0
#         last_progress = 0
        
#         # Spacer para a barra de progresso não apagar o header
#         print("")

#         while True:
#             # Check Stop
#             if state.stop_requested:
#                 print(Fore.YELLOW + "[STOP] Detecção interrompida.")
#                 break

#             # 1. Carregar Batch de Frames
#             frames_batch = []
#             for _ in range(batch_size):
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 frames_batch.append(frame)

#             if not frames_batch:
#                 break # Fim do vídeo

#             # =========================================================================
#             # INFERÊNCIA (BRANCH CONFORME TRACKER)
#             # =========================================================================
            
#             # --- OPÇÃO A: DEEPOCSORT (FRAME-A-FRAME) ---
#             if USING_DEEPOCSORT:
#                 # Adaptação: DeepOCSORT espera 1 frame por vez (ou lista de frames para loop interno)
#                 # Nosso wrapper CustomDeepOCSORT.track aceita 1 frame.
#                 # Precisamos iterar o batch manualmente aqui.
                
#                 batch_results_fake = [] # Estrutura para manter compatibilidade com loop abaixo
                
#                 for frame in frames_batch:
#                     # Roda rastreador (YOLO interno + OCSORT)
#                     # Retorna np.array: [[x1, y1, x2, y2, id, conf, cls, ...]]
#                     tracks = deep_tracker.track(frame)
                    
#                     # Empacota em um objeto mock que imita o resultado do YOLO para o loop de baixo
#                     # Precisamos de algo com .boxes.data
#                     class MockResult:
#                         def __init__(self, data):
#                             self.boxes = self.MockBoxes(data)
#                         class MockBoxes:
#                             def __init__(self, data):
#                                 self.data = torch.from_numpy(data) if data.shape[0] > 0 else torch.empty(0, 6)
                                
#                     batch_results_fake.append(MockResult(tracks))
                
#                 batch_results = batch_results_fake

#             # --- OPÇÃO B: BOTSORT (BATCH NATIVO YOLO) ---
#             else:
#                 # Inferencia com Tracker Default
#                 batch_results = model.track(
#                     source=frames_batch,
#                     imgsz=cm.YOLO_IMGSZ,
#                     conf=cm.DETECTION_CONF,
#                     device=cm.DEVICE,
#                     persist=True,
#                     tracker=str(tracker_yaml_path),
#                     classes=[cm.YOLO_CLASS_PERSON],
#                     verbose=False,
#                     half=True,
#                     stream=False,
#                     task="detect",
#                     persist=True
#                 )


#             # 3. Processar Resultados do Batch (COMUM AOS DOIS)
#             for i, r in enumerate(batch_results):
#                 current_frame_idx = frame_idx_global + i
                
#                 # Feedback visual para o frontend (opcional, pega o último do batch)
#                 if i == len(batch_results) - 1:
#                      # Apenas marcamos progresso, não enviamos imagem para não travar
#                      pass

#                 # Extrai dados leves
#                 frame_res = {"boxes": None}
                
#                 if r.boxes is not None and len(r.boxes) > 0:
#                     boxes_data = r.boxes.data.cpu().numpy()
#                     frame_res["boxes"] = boxes_data
                    
#                     ids = boxes_data[:, 4] if boxes_data.shape[1] > 4 else None
#                     current_time = current_frame_idx / fps

#                     if ids is not None:
#                         for tid_raw in ids:
#                             if tid_raw is None or tid_raw < 0: continue
#                             tid = int(tid_raw)
                            
#                             # Feature extraction (se disponível)
#                             # DeepOCSORT não expõe 'feat' facilmente no output padrão boxmot
#                             # BoTSORT expõe via tracker.tracks
#                             f = np.zeros(512, dtype=np.float32)

#                             if not USING_DEEPOCSORT:
#                                 try:
#                                     trk = tracker.tracks[tid]
#                                     f = trk.feat.copy() if hasattr(trk, 'feat') else np.zeros(512)
#                                 except:
#                                     pass
                            
#                             # Se DeepOCSORT implementasse extração, pegariamos aqui. 
#                             # Por hora, ele foca no rastreamento robusto.

#                             if tid not in track_data:
#                                 track_data[tid] = {
#                                     "start": current_time,
#                                     "end": current_time,
#                                     "features": [f],
#                                     "frames": {current_frame_idx},
#                                 }
#                             else:
#                                 track_data[tid]["end"] = current_time
#                                 track_data[tid]["frames"].add(current_frame_idx)
#                                 if len(track_data[tid]["features"]) < 20:
#                                     track_data[tid]["features"].append(f)
                
#                 results.append(frame_res)

#             # Atualiza Indices
#             frame_idx_global += len(frames_batch)
            
#             prog = int((frame_idx_global / total_frames) * 100)
#             if prog >= last_progress + 10:
#                 sys.stdout.write(f"\r{Fore.YELLOW}[YOLO]{Fore.WHITE} Progresso: {prog} %")
#                 sys.stdout.flush()
#                 last_progress = prog

#         cap.release()
#         sys.stdout.write('\n')
#         sys.stdout.flush()
        
#         # Merge Tracks
#         if state.stop_requested:
#             break

#         if not track_data:
#             print("[WARN] Nenhum track valido identificado.")
#             continue

#         merged_tracks, id_map = merge_tracks(track_data)

#         resultados_finais.append({
#             "video": str(video),
#             "fps": fps,
#             "track_data": track_data,
#             "merged_tracks": merged_tracks,
#             "id_map": id_map,
#             "results": results,
#         })
        
#         # Clean Memory
#         if not USING_DEEPOCSORT:
#             del model.tracker
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     print(Fore.GREEN + "[OK] DETECÇÃO E IDENTIFICAÇÃO CONCLUIDA!")
#     return resultados_finais
