# ==============================================================
# neurapose_backend/nucleo/tracking_utils.py
# ==============================================================
# Módulo utilitário para geração de relatórios de tracking.
# Centraliza a lógica de criação do _tracking.json.
# ==============================================================

import json
from pathlib import Path
from typing import List, Dict, Any

def gerar_relatorio_tracking(
    registros: List[Dict[str, Any]],
    id_map: Dict[Any, Any],
    ids_validos: List[int],
    total_frames: int,
    video_name: str,
    output_path: Path
):
    """
    Gera e salva o relatório de tracking (tracking.json).
    
    Args:
        registros: Lista de registros filtrados/processados.
        id_map: Mapa original de IDs do BoTSORT.
        ids_validos: Lista de IDs considerados válidos após filtragem.
        total_frames: Número total de frames do vídeo.
        video_name: Nome do arquivo de vídeo.
        output_path: Caminho completo para salvar o arquivo .json.
    """
    
    # Filtra e limpa o mapa de IDs para manter apenas os válidos
    id_map_limpo = {str(k): int(v) for k, v in id_map.items() if v in ids_validos}
    
    tracking_analysis = {
        "video": video_name,
        "total_frames": total_frames,
        "id_map": id_map_limpo,
        "tracking_by_frame": {}
    }
    
    # Organiza registros por frame
    for reg in registros:
        f_id = reg["frame"]
        if f_id not in tracking_analysis["tracking_by_frame"]:
            tracking_analysis["tracking_by_frame"][f_id] = []
        
        tracking_analysis["tracking_by_frame"][f_id].append({
            "botsort_id": reg.get("botsort_id"),
            "id_persistente": reg["id_persistente"],
            "bbox": reg["bbox"],
            "confidence": reg["confidence"],
             # Inclui a classe se estiver disponível (para testes)
            "classe_id": reg.get("classe_id"),
            "classe_predita": reg.get("classe_predita")
        })
        
    # Garante que o diretório pai existe
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tracking_analysis, f, indent=2, ensure_ascii=False)
    
    return tracking_analysis
