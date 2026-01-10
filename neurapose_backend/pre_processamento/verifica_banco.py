# ===========================================================
# neurapose_backend/pre_processamento/verifica_banco.py
# ===========================================================

import json
import re
from pathlib import Path
from datetime import datetime

# ===================== CONFIGURAÇÕES =====================
data_name = "data-labex"

base_dir = Path(f"resultado_processamento/{data_name}")
labels_path = base_dir / "anotacoes" / "labels.json"
jsons_dir = base_dir / "jsons"
report_dir = base_dir / "relatorios"
report_dir.mkdir(parents=True, exist_ok=True)
debug_path = report_dir / "debug.txt"

# ===================== FUNÇÃO DE LOG =====================
def log(msg: str, out_file):
    ts = datetime.now().strftime("[%H:%M:%S]")
    line = f"{ts} {msg}"
    print(line)
    out_file.write(line + "\n")

# ===================== FUNÇÃO DE BUSCA =====================
def find_json(base_dir: Path, stem: str):
    """Procura JSON tolerando variações com ordem de prioridade fixa."""
    stem_norm = stem.lower().replace("-", "_").replace(" ", "")
    
    # 1. Tenta a correspondência exata conforme esperado pelo labels.json 
    variants = [
        stem_norm,
        re.sub(r"([a-z])(\d)", r"\1_\2", stem_norm),
        stem_norm.replace("_", ""),
    ]
    
    for v in variants:
        path = base_dir / f"{v}.json"
        if path.exists():
            return path

    matches = sorted(list(base_dir.glob(f"*{stem_norm.replace('_', '')}*.json")))
    return matches[0] if matches else None

# ===================== EXECUÇÃO PRINCIPAL =====================
if not labels_path.exists():
    raise FileNotFoundError(f"labels.json não encontrado em: {labels_path}")

with open(labels_path, "r", encoding="utf-8") as f:
    labels = json.load(f)

with open(debug_path, "w", encoding="utf-8") as debug:
    log("🚀 Iniciando verificação de integridade (pré-split)", debug)
    log(f"📁 Diretório base: {base_dir}", debug)
    log(f"🔢 Total de vídeos no labels.json: {len(labels)}", debug)
    log("-" * 90, debug)

    total_missing_json = 0
    total_missing_ids = 0
    total_extra_ids = 0

    for i, (video, ids_dict) in enumerate(labels.items(), 1):
        log(f"\n[{i}/{len(labels)}] 🎬 Vídeo: {video}", debug)
        expected_ids = list(ids_dict.keys())
        expected_classes = list(ids_dict.values())

        # Tenta localizar o JSON
        json_path = find_json(jsons_dir, video)
        if not json_path:
            log(f"   ⚠️ JSON não encontrado para '{video}'", debug)
            total_missing_json += 1
            continue

        log(f"   ✅ JSON localizado: {json_path.name}", debug)

        # Lê o JSON de keypoints
        try:
            with open(json_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)
        except Exception as e:
            log(f"   ❌ Erro ao abrir {json_path.name}: {e}", debug)
            continue

        # Coleta IDs detectados
        found_ids = set()
        for item in data:
            if "id" in item:
                found_ids.add(str(item["id"]))
            elif "id_persistente" in item:
                found_ids.add(str(item["id_persistente"]))

        missing_ids = [i for i in expected_ids if i not in found_ids]
        extra_ids = [i for i in found_ids if i not in expected_ids]

        # Logs detalhados
        resumo = ", ".join([f"id={pid}({cls})" for pid, cls in zip(expected_ids, expected_classes)])
        log(f"   🧾 IDs esperados ({len(expected_ids)}): {resumo}", debug)
        log(f"   🧩 IDs encontrados ({len(found_ids)}): {', '.join(sorted(found_ids)) or 'nenhum'}", debug)

        if missing_ids:
            total_missing_ids += len(missing_ids)
            log(f"   ⚠️ IDs ausentes no JSON: {', '.join(missing_ids)}", debug)
        if extra_ids:
            total_extra_ids += len(extra_ids)
            log(f"   ⚠️ IDs extras (não listados no labels.json): {', '.join(extra_ids)}", debug)

        if not missing_ids and not extra_ids:
            log("   ✅ Todos os IDs esperados estão presentes.", debug)

    # ===================== RESUMO FINAL =====================
    log("\n" + "=" * 90, debug)
    log("📊 RESUMO FINAL DE VERIFICAÇÃO", debug)
    log(f" - Total de vídeos analisados: {len(labels)}", debug)
    log(f" - JSONs não encontrados: {total_missing_json}", debug)
    log(f" - IDs ausentes no JSON: {total_missing_ids}", debug)
    log(f" - IDs extras (não esperados): {total_extra_ids}", debug)
    log("=" * 90, debug)
    log(f"💾 Relatório salvo em: {debug_path}", debug)

print("\n✅ Verificação concluída! Veja o relatório em:")
print(f"   {debug_path}")
