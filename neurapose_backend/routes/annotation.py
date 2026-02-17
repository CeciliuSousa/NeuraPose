import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import neurapose_backend.config_master as cm

router = APIRouter()
logger = logging.getLogger("NeuraPoseAPI")

class AnnotationRequest(BaseModel):
    video_stem: str
    annotations: Dict[str, Any]
    root_path: str

class BatchAnnotationRequest(BaseModel):
    root_path: str
    default_class: str = cm.CLASSE1

@router.get("/annotate/list")
def list_videos_to_annotate(root_path: Optional[str] = None):
    root = Path(root_path).resolve() if root_path else cm.REID_OUTPUT_DIR
    
    if root.name in ["predicoes", "jsons", "videos", "reid", "anotacoes"]:
        root = root.parent
    
    pred_dir = root / "predicoes"
    json_dir = root / "jsons"
    videos_dir = root / "videos"
    labels_path = root / "anotacoes" / "labels.json"
    
    labels_existentes = {}
    if labels_path.exists():
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                labels_existentes = json.load(f) or {}
        except: 
            pass
    
    result = []
    source_dir = videos_dir if videos_dir.exists() else pred_dir
    
    if source_dir.exists():
        valid_exts = {'.mp4', '.avi', '.mov', '.mkv'}
        for vfile in source_dir.glob("*"):
            if vfile.suffix.lower() not in valid_exts:
                continue
                
            stem = vfile.stem.replace("_pose", "").replace("_pred", "")
            json_path = json_dir / f"{stem}.json"
            if not json_path.exists(): json_path = json_dir / f"{stem}_pose.json"
            if not json_path.exists(): json_path = json_dir / f"{vfile.stem}.json"
            
            is_annotated = (stem in labels_existentes) or (vfile.stem in labels_existentes)
            status = "anotado" if is_annotated else "pendente"
            
            result.append({
                "video_id": stem,
                "video_name": vfile.name,
                "status": status,
                "has_json": json_path.exists(),
                "creation_time": vfile.stat().st_mtime
            })
    
    return {
        "videos": sorted(result, key=lambda x: x["video_id"]),
        "root": str(root),
        "labels_path": str(labels_path)
    }

@router.get("/annotate/{video_id}/details")
def get_annotation_details(video_id: str, root_path: Optional[str] = None):
    from neurapose_backend.pre_processamento.anotando_classes import carregar_pose_records, indexar_por_frame
    
    root = Path(root_path).resolve() if root_path else cm.REID_OUTPUT_DIR
    if root.name in ["predicoes", "jsons", "videos", "reid", "anotacoes"]:
        root = root.parent
    
    json_dir = root / "jsons"
    json_path = json_dir / f"{video_id}.json"
    
    if not json_path.exists():
        clean_id = video_id.replace("_pose", "").replace("_pred", "")
        json_path = json_dir / f"{clean_id}.json"

    if not json_path.exists():
        json_path = json_dir / f"{clean_id}_pose.json"
        
    if not json_path.exists():
        json_path = json_dir / f"{clean_id}_tracking.json"
    
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"JSON não encontrado: {video_id}")
    
    labels_path = root / "anotacoes" / "labels.json"
    saved_labels = {}
    video_stem = video_id.replace("_pose", "").replace(f"_{cm.FPS_TARGET}fps", "")
    
    if labels_path.exists():
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                all_labels = json.load(f) or {}
                saved_labels = all_labels.get(video_stem, {})
                if not saved_labels:
                    alt_stem = video_stem.replace(f"_{cm.FPS_TARGET}fps", "")
                    saved_labels = all_labels.get(alt_stem, {})
        except:
            pass
        
    records = carregar_pose_records(json_path)
    frames_index, id_counter = indexar_por_frame(records)
    
    ids_info = []
    for gid, count in id_counter.items():
        if count >= cm.MIN_FRAMES_PER_ID:
            saved_label = saved_labels.get(str(gid), None)
            if isinstance(saved_label, dict):
                label = saved_label.get("classe", "desconhecido")
            elif saved_label:
                label = saved_label
            else:
                label = "desconhecido"
            
            ids_info.append({
                "id": gid,
                "frames": count,
                "label": label
            })
            
    return {
        "video_id": video_id,
        "ids": ids_info,
        "total_frames": max(frames_index.keys()) if frames_index else 0,
        "min_frames": cm.MIN_FRAMES_PER_ID
    }

@router.post("/annotate/save")
def save_annotations(req: AnnotationRequest):
    root = Path(req.root_path).resolve()
    if root.name in ["predicoes", "jsons", "videos", "reid", "anotacoes"]:
        root = root.parent
    
    labels_dir = root / "anotacoes"
    labels_path = labels_dir / "labels.json"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    todas_labels = {}
    if labels_path.exists():
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                todas_labels = json.load(f) or {}
        except: 
            pass
    
    clean_annotations = {}
    for lid, lval in req.annotations.items():
        if isinstance(lval, BaseModel):
             clean_annotations[lid] = lval.dict()
        else:
             clean_annotations[lid] = lval

    todas_labels[req.video_stem] = clean_annotations
    
    try:
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(todas_labels, f, indent=4, ensure_ascii=False)
        return {
            "status": "success", 
            "message": f"Anotações salvas para {req.video_stem}",
            "labels_path": str(labels_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar: {e}")

@router.post("/annotate/save-all-pending")
def save_all_pending(req: BatchAnnotationRequest):
    root = Path(req.root_path).resolve()
    if root.name in ["predicoes", "jsons", "videos", "reid", "anotacoes"]:
        root = root.parent
    
    jsons_path = root / "jsons"
    if not jsons_path.exists(): raise HTTPException(404, "Pasta jsons não encontrada")
    
    labels_dir = root / "anotacoes"
    labels_path = labels_dir / "labels.json"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    todas_labels = {}
    if labels_path.exists():
        try:
            with open(labels_path, "r", encoding="utf-8") as f: todas_labels = json.load(f) or {}
        except: pass
        
    videos_dir = root / "videos"
    if not videos_dir.exists():
        source_files = list(jsons_path.glob("*.json"))
        is_video_source = False
    else:
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        source_files = [p for p in videos_dir.iterdir() if p.suffix.lower() in video_exts]
        is_video_source = True
        
    saved_count = 0
    import re
    
    for src_file in source_files:
        if is_video_source: video_stem = src_file.stem
        else: video_stem = re.sub(r'(_pose|_pred|_tracking)+$', '', src_file.stem)
        
        if video_stem in todas_labels: continue
        
        candidate_jsons = [jsons_path / f"{video_stem}_pose.json", jsons_path / f"{video_stem}.json", jsons_path / f"{video_stem}_pred.json"]
        target_json = next((c for c in candidate_jsons if c.exists()), None)
        
        if not target_json: continue
        
        try:
            with open(target_json, "r", encoding="utf-8") as f: records = json.load(f)
            from collections import Counter
            id_counter = Counter()
            for r in records:
                gid = r.get("id_persistente") or r.get("BoTSORT_id") or r.get("track_id")
                if gid is not None and int(gid) >= 0: id_counter[int(gid)] += 1
            
            annotations = {}
            for gid, count in id_counter.items():
                if count >= cm.MIN_FRAMES_PER_ID: annotations[str(gid)] = req.default_class
            
            if annotations:
                todas_labels[video_stem] = annotations
                saved_count += 1
        except: continue


    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(todas_labels, f, indent=4, ensure_ascii=False)
        
    return {"status": "success", "saved_count": saved_count}

@router.get("/annotate/video/{video_id}")
def stream_annotation_video(video_id: str, root_path: Optional[str] = None):
    root = Path(root_path).resolve() if root_path else cm.REID_OUTPUT_DIR
    if root.name in ["predicoes", "jsons", "videos", "reid", "anotacoes"]:
        root = root.parent
    
    # Busca 1: Nome exato na pasta de vídeos (Original)
    video_path = root / "videos" / f"{video_id}.mp4"
    
    # Busca 2: Nome exato na pasta de predicoes (Processado)
    if not video_path.exists():
        video_path = root / "predicoes" / f"{video_id}.mp4"
    
    # Busca 3: Tentativa de limpar sufixos _pose, _pred
    if not video_path.exists():
        clean_id = video_id.replace("_pose", "").replace("_pred", "")
        video_path = root / "videos" / f"{clean_id}.mp4"
        
    # Busca 4: Limpo na pasta de predicoes
    if not video_path.exists():
        clean_id = video_id.replace("_pose", "").replace("_pred", "")
        video_path = root / "predicoes" / f"{clean_id}_pose.mp4"

    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")
        

    return FileResponse(video_path, media_type="video/mp4")

@router.get("/annotate/{video_id}/data")
def get_annotation_data(video_id: str, root_path: Optional[str] = None):
    """Retorna dados detalhados frame a frame para o player (BBoxes, IDs)."""
    from neurapose_backend.pre_processamento.anotando_classes import carregar_pose_records
    
    root = Path(root_path).resolve() if root_path else cm.REID_OUTPUT_DIR
    if root.name in ["predicoes", "jsons", "videos", "reid", "anotacoes"]:
        root = root.parent
    
    json_dir = root / "jsons"
    json_path = json_dir / f"{video_id}.json"
    
    if not json_path.exists():
        clean_id = video_id.replace("_pose", "").replace("_pred", "")
        json_path = json_dir / f"{clean_id}.json"

    if not json_path.exists():
        json_path = json_dir / f"{clean_id}_pose.json"
        
    if not json_path.exists():
        json_path = json_dir / f"{clean_id}_tracking.json"
    
    if not json_path.exists():
         # Retorna vazio se não tiver JSON, para não quebrar o player
        return {"video_id": video_id, "frames": {}, "id_counts": {}}
    
    records = carregar_pose_records(json_path)
    
    frames_data = {}
    id_counts = {}
    
    for r in records:
        fid = str(int(r.get("frame_idx", r.get("frame", -1))))
        gid = int(r.get("id_persistente") if r.get("id_persistente") is not None else (r.get("BoTSORT_id") if r.get("BoTSORT_id") is not None else -1))
        
        if gid < 0: continue
            
        bbox = r.get("bbox", [])
        
        if fid not in frames_data: frames_data[fid] = []
        frames_data[fid].append({
            "id": gid,
            "bbox": bbox,
            "keypoints": r.get("keypoints", [])
        })
        
        id_counts[str(gid)] = id_counts.get(str(gid), 0) + 1
            
    return {
        "video_id": video_id,
        "frames": frames_data,
        "id_counts": id_counts
    }
