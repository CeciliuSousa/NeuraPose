# import os
# import json
# import logging
# import shutil
# from pathlib import Path
# from typing import Optional, List, Dict, Any
# from fastapi import APIRouter, HTTPException, BackgroundTasks
# from pydantic import BaseModel

# import neurapose_backend.config_master as cm
# from neurapose_backend.globals.state import state
# from neurapose_backend.nucleo.log_service import CaptureOutput
# from neurapose_backend.pre_processamento.reid_manual import (
#     aplicar_processamento_completo, 
#     carregar_pose_records, 
#     renderizar_video_limpo, 
#     salvar_json,
#     renderizar_video_cortado_raw
# )

# router = APIRouter()
# logger = logging.getLogger("NeuraPoseAPI")

# REID_AGENDA_DIR = cm.REID_OUTPUT_DIR / "reid"
# REID_AGENDA_FILE = REID_AGENDA_DIR / "reid.json"

# # ==============================================================
# # MODELS
# # ==============================================================

# class ReidSwapRule(BaseModel):
#     src_id: int
#     tgt_id: int
#     frame_start: int = 0
#     frame_end: int = 999999

# class ReidDeletionRule(BaseModel):
#     id: int
#     frame_start: int = 0
#     frame_end: int = 999999

# class ReidCutRule(BaseModel):
#     frame_start: int
#     frame_end: int

# class ReidVideoEntry(BaseModel):
#     video_id: str
#     action: str = "process"
#     swaps: List[ReidSwapRule] = []
#     deletions: List[ReidDeletionRule] = []
#     cuts: List[ReidCutRule] = []

# class ReidAgendaRequest(BaseModel):
#     source_dataset: str
#     video: ReidVideoEntry
    
# class ReidBatchApplyRequest(BaseModel):
#     videos: Optional[List[Any]] = None
#     root_path: str
#     output_path: Optional[str] = None

# class ReIDApplyRequest(BaseModel):
#     """Modelo antigo para apply individual (compatibilidade)."""
#     action: str = "process"
#     rules: List[ReidSwapRule] = []
#     deletions: List[ReidDeletionRule] = [] 
#     cuts: List[ReidCutRule] = []

# # ==============================================================
# # HELPERS
# # ==============================================================

# def get_reid_paths_from_source(source_path_str: str):
#     source = Path(source_path_str)
#     # Se terminar em 'predicoes', sobe um nível para pegar o nome do dataset
#     if source.name == 'predicoes':
#         dataset_dir = source.parent
#     else:
#         dataset_dir = source
    
#     # Define novo nome: dataset-reidentificado
#     new_dataset_name = f"{dataset_dir.name}-reidentificado"
    
#     # Salva em resultados-reidentificacoes/[dataset]-reidentificado
#     output_root = cm.REID_OUTPUT_DIR / new_dataset_name
    
#     return {
#         "root": output_root,
#         "reid_dir": output_root / "reid",
#         "agenda_file": output_root / "reid" / "reid.json",
#         "videos_dir": output_root / "videos",
#         "predictions_dir": output_root / "predicoes",
#         "jsons_dir": output_root / "jsons"
#     }

# def run_reid_batch_processing(source_dataset: str):
#     """Processa todos os vídeos agendados em reid.json e salva em resultados-reidentificacoes."""
#     state.is_running = True
#     logger.info(f"Iniciando Batch Process ReID: {source_dataset}")
    
#     with CaptureOutput():
#         try:
#             paths = get_reid_paths_from_source(source_dataset)
            
#             # 1. Cria diretórios de saída
#             for k in ["videos_dir", "predictions_dir", "jsons_dir"]:
#                 paths[k].mkdir(parents=True, exist_ok=True)
#             logger.info(f"Pasta de saída: {paths['root']}")
                
#             # 2. Verifica Agenda
#             if not paths["agenda_file"].exists():
#                 logger.error(f"Agenda não encontrada: {paths['agenda_file']}")
#                 return
                
#             with open(paths["agenda_file"], "r", encoding="utf-8") as f:
#                 agenda = json.load(f)
                
#             # 3. Define Origem
#             src_path = Path(source_dataset)
#             if src_path.name in ['predicoes', 'jsons', 'videos']:
#                 ds_root = src_path.parent
#             else:
#                 ds_root = src_path
                
#             src_videos = ds_root / "videos"
#             src_jsons = ds_root / "jsons"
            
#             # 4. Identifica vídeos para excluir (não copiar)
#             videos_to_delete = set()
#             videos_to_process = {}
            
#             for vid_entry in agenda.get("videos", []):
#                 vid_id = vid_entry["video_id"]
#                 action = vid_entry.get("action", "process")
                
#                 if action == "delete":
#                     videos_to_delete.add(vid_id)
#                     # Remove _pose suffix para encontrar vídeo original
#                     clean_id = vid_id.replace("_pose", "")
#                     videos_to_delete.add(clean_id)
#                 else:
#                     videos_to_process[vid_id] = vid_entry
            
#             logger.info(f"Vídeos a excluir: {len(videos_to_delete)}")
#             logger.info(f"Vídeos a processar: {len(videos_to_process)}")
            
#             # 5. Copia Vídeos Originais (exceto os marcados para delete)
#             logger.info("Copiando vídeos base...")
#             copied_count = 0
#             if src_videos.exists():
#                 for vfile in src_videos.glob("*"):
#                     if vfile.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
#                         # Verifica se deve pular (vídeo marcado para delete)
#                         if vfile.stem in videos_to_delete:
#                             logger.info(f"Pulando vídeo excluído: {vfile.name}")
#                             continue
                            
#                         dst = paths["videos_dir"] / vfile.name
#                         if not dst.exists():
#                             shutil.copy2(vfile, dst)
#                             copied_count += 1
#             logger.info(f"Copiados {copied_count} vídeos")
            
#             # 6. Processa Vídeos da Agenda
#             processed_count = 0
#             for vid_id, vid_entry in videos_to_process.items():
#                 # Mapeia regras
#                 swaps = [
#                     {"src": s["src_id"], "tgt": s["tgt_id"], "start": s["frame_start"], "end": s["frame_end"]}
#                     for s in vid_entry.get("swaps", [])
#                 ]
#                 deletions = [
#                     {"id": d["id"], "start": d["frame_start"], "end": d["frame_end"]}
#                     for d in vid_entry.get("deletions", [])
#                 ]
#                 cuts = [
#                     {"start": c["frame_start"], "end": c["frame_end"]}
#                     for c in vid_entry.get("cuts", [])
#                 ]
                
#                 # Carrega JSON original
#                 json_in = src_jsons / f"{vid_id}.json"
#                 if not json_in.exists():
#                     logger.warning(f"JSON input não encontrado: {json_in}")
#                     continue
                    
#                 with open(json_in, "r", encoding="utf-8") as f:
#                     records = json.load(f)
                
#                 # Verifica se há alterações reais
#                 has_changes = len(swaps) > 0 or len(deletions) > 0 or len(cuts) > 0
                
#                 if has_changes:
#                     # Aplica Lógica de processamento
#                     processed, _, _, _ = aplicar_processamento_completo(records, swaps, deletions, cuts)
#                 else:
#                     # Sem alterações, apenas copia o JSON original
#                     processed = records
                
#                 # Salva JSON (filtrado ou cópia)
#                 json_out = paths["jsons_dir"] / f"{vid_id}.json"
#                 with open(json_out, "w", encoding="utf-8") as f:
#                     json.dump(processed, f, ensure_ascii=False, indent=2)
                    
#                 # Gera Vídeo Anotado apenas se houver alterações
#                 if has_changes:
#                     # Encontra vídeo de entrada
#                     vid_stem = vid_id.replace("_pose", "")
#                     vid_in = paths["videos_dir"] / f"{vid_stem}.mp4"
#                     if not vid_in.exists():
#                         vid_in = paths["videos_dir"] / f"{vid_id}.mp4"
#                     if not vid_in.exists():
#                         candidates = list(paths["videos_dir"].glob(f"{vid_stem}.*"))
#                         if candidates: 
#                             vid_in = candidates[0]
                        
#                     if vid_in.exists():
#                         # 1. Vídeo Anotado (com caixas) -> Predições
#                         vid_out_pred = paths["predictions_dir"] / f"{vid_id}_pose.mp4"
#                         logger.info(f"Gerando vídeo anotado: {vid_out_pred.name}")
#                         renderizar_video_limpo(str(vid_in), str(vid_out_pred), processed, cuts)

#                         # 2. Vídeo Raw (sem caixas, mas cortado) -> Videos (para consistência)
#                         vid_out_raw = paths["videos_dir"] / f"{vid_id}.mp4"
#                         logger.info(f"Gerando vídeo raw cortado: {vid_out_raw.name}")
#                         renderizar_video_cortado_raw(str(vid_in), str(vid_out_raw), cuts)
#                     else:
#                         logger.warning(f"Vídeo base não encontrado: {vid_id}")
                
#                 processed_count += 1
                    
#             logger.info(f"Batch ReID finalizado. Processados: {processed_count} vídeos.")

#         except Exception as e:
#             logger.error(f"Erro crítico no batch reid: {e}")
#             import traceback
#             logger.error(traceback.format_exc())
#         finally:
#             state.reset()

# # ==============================================================
# # ENDPOINTS
# # ==============================================================

# @router.get("/reid/agenda")
# def get_reid_agenda(root_path: Optional[str] = None):
#     """Carrega a agenda de ReID e calcula estatísticas de pendência."""
#     target_file = None
#     agenda = None
    
#     if root_path:
#         paths = get_reid_paths_from_source(root_path)
#         target_file = paths["agenda_file"]
#     else:
#         target_file = REID_AGENDA_FILE

#     # Tenta carregar agenda se existir
#     if target_file and target_file.exists():
#         try:
#             with open(target_file, "r", encoding="utf-8") as f:
#                 agenda = json.load(f)
#         except Exception as e:
#             logger.error(f"Erro ao ler agenda ReID: {e}")
#             return {"agenda": None, "error": str(e)}

#     # Calcula estatísticas
#     stats = {"total": 0, "processed": 0, "pending": 0}
    
#     if root_path:
#         try:
#             src_path = Path(root_path)
#             if src_path.name == 'predicoes':
#                 src_root = src_path.parent
#             else:
#                 src_root = src_path
            
#             src_videos_dir = src_root / "videos"
#             if src_videos_dir.exists():
#                 # Conta vídeos source (extensões válidas)
#                 valid_exts = {'.mp4', '.avi', '.mov', '.mkv'}
#                 # Glob recursivo ou simples? Simples.
#                 total_videos = sum(1 for f in src_videos_dir.glob("*") if f.suffix.lower() in valid_exts)
#                 stats["total"] = total_videos
                
#                 processed_count = 0
#                 excluded_count = 0
#                 if agenda and "videos" in agenda:
#                     for v in agenda["videos"]:
#                         act = v.get("action", "process")
#                         if act == "delete":
#                             excluded_count += 1
#                         else:
#                             processed_count += 1
                
#                 stats["processed"] = processed_count
#                 stats["excluded"] = excluded_count
#                 stats["pending"] = max(0, total_videos - (processed_count + excluded_count))
#         except Exception as e:
#             logger.warning(f"Erro ao calcular stats de ReID: {e}")

#     return {"agenda": agenda, "stats": stats}

# @router.post("/reid/agenda/save")
# def save_reid_agenda(request: ReidAgendaRequest):
#     """Salva/atualiza um vídeo na agenda de ReID com caminho dinâmico."""
#     from datetime import datetime
    
#     # Determina caminhos baseados no source_dataset
#     if request.source_dataset:
#         paths = get_reid_paths_from_source(request.source_dataset)
#         target_dir = paths["reid_dir"]
#         target_file = paths["agenda_file"]
#     else:
#         # Fallback para padrão
#         target_dir = REID_AGENDA_DIR
#         target_file = REID_AGENDA_FILE
    
#     target_dir.mkdir(parents=True, exist_ok=True)
    
#     # Carrega agenda existente ou cria nova
#     if target_file.exists():
#         try:
#             with open(target_file, "r", encoding="utf-8") as f:
#                 agenda = json.load(f)
#         except:
#             agenda = None
#     else:
#         agenda = None
    
#     if agenda is None:
#         agenda = {
#             "version": "1.0",
#             "source_dataset": request.source_dataset,
#             "created_at": datetime.now().isoformat(),
#             "updated_at": datetime.now().isoformat(),
#             "videos": []
#         }
    
#     # Atualiza source_dataset se diferente
#     if agenda.get("source_dataset") != request.source_dataset:
#         agenda["source_dataset"] = request.source_dataset
#         agenda["videos"] = []  # Limpa vídeos se dataset mudou
    
#     # Procura vídeo existente na agenda
#     video_data = request.video.model_dump()
#     existing_idx = None
#     for i, v in enumerate(agenda["videos"]):
#         if v["video_id"] == video_data["video_id"]:
#             existing_idx = i
#             break
    
#     # Atualiza ou adiciona
#     if existing_idx is not None:
#         agenda["videos"][existing_idx] = video_data
#     else:
#         agenda["videos"].append(video_data)
    
#     agenda["updated_at"] = datetime.now().isoformat()
    
#     # Salva arquivo
#     try:
#         with open(target_file, "w", encoding="utf-8") as f:
#             json.dump(agenda, f, ensure_ascii=False, indent=2)
#         return {"status": "success", "message": f"Vídeo '{request.video.video_id}' agendado com sucesso."}
#     except Exception as e:
#         logger.error(f"Erro ao salvar agenda ReID: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @router.delete("/reid/agenda/{video_id}")
# def remove_from_agenda(video_id: str, root_path: Optional[str] = None):
#     """Remove um vídeo da agenda, aceitando root_path dinâmico."""
#     target_file = None
#     if root_path:
#         paths = get_reid_paths_from_source(root_path)
#         target_file = paths["agenda_file"]
#     else:
#         target_file = REID_AGENDA_FILE
        
#     if not target_file.exists():
#         return {"status": "not_found", "detail": "Agenda file not found"}
    
#     try:
#         with open(target_file, "r", encoding="utf-8") as f:
#             agenda = json.load(f)
        
#         original_count = len(agenda.get("videos", []))
#         agenda["videos"] = [v for v in agenda.get("videos", []) if v["video_id"] != video_id]
        
#         with open(target_file, "w", encoding="utf-8") as f:
#             json.dump(agenda, f, ensure_ascii=False, indent=2)
        
#         removed = original_count > len(agenda["videos"])
#         return {"status": "removed" if removed else "not_found"}
#     except Exception as e:
#         logger.error(f"Erro ao remover da agenda: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @router.post("/reid/batch-apply")
# def batch_apply_reid(request: ReidBatchApplyRequest, background_tasks: BackgroundTasks):
#     """Executa o processamento em massa das correções de ReID."""
#     background_tasks.add_task(run_reid_batch_processing, request.root_path)
#     return {"status": "started", "message": "Iniciando processamento ReID em background."}

# # Endpoint "apply" individual legado (mantido para compatibilidade se necessário)
# @router.post("/reid/{video_id}/apply")
# def apply_reid_changes(video_id: str, req: ReIDApplyRequest, root_path: Optional[str] = None, output_path: Optional[str] = None):
#     # (Lógica idêntica ao main.py - omitida aqui para priorizar batch, mas pode ser adicionada se o front usar)
#     # Recomendo que o front migre para save_agenda e batch-apply
#     return {"status": "deprecated", "message": "Use /reid/agenda/save e /reid/batch-apply"}

# from fastapi.responses import FileResponse
# @router.get("/reid/video/{video_id}")
# def stream_reid_video(video_id: str, root_path: Optional[str] = None, source: str = "default"):
#     """Serve o arquivo de video para o ReID player."""
#     root = Path(root_path).resolve() if root_path else cm.PROCESSING_OUTPUT_DIR
#     if root.name in ["predicoes", "jsons", "videos"]: root = root.parent
    
#     # ... Lógica de busca de vídeo (cópia exata do main.py) ...
#     v_final = None
#     if source == 'raw':
#         clean_id = video_id.replace('_pose', '').replace('_pred', '')
#         possible_raw = [root / "videos" / f"{clean_id}.mp4", root / "videos" / f"{video_id}.mp4"]
#         for p in possible_raw:
#              if p.exists():
#                  v_final = p
#                  break

#     if not v_final:
#         possible_preds = [root / "predicoes" / f"{video_id}.mp4"]
#         for p in possible_preds:
#              if p.exists():
#                  v_final = p
#                  break
            
#     if not v_final:
#         possible_sources = [root / "videos" / f"{video_id.replace('_pose', '')}.mp4", root / "videos" / f"{video_id}.mp4"]
#         for p in possible_sources:
#              if p.exists():
#                  v_final = p
#                  break
                
#     if not v_final or not v_final.exists():
#         raise HTTPException(status_code=404, detail="Video not found")
        
#     return FileResponse(v_final, media_type="video/mp4")
