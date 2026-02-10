# neurapose-backend/app/testar_modelo.py
# Script principal para teste do modelo treinado.

import json
import time
from pathlib import Path
from colorama import Fore, init as colorama_init
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix)
import warnings
warnings.filterwarnings("ignore")

from neurapose_backend.app.configuracao.config import (
    DATASET_DIR, MODEL_DIR, LABELS_TEST_PATH, args as config_args
)
import neurapose_backend.config_master as cm
from neurapose_backend.app.utils.ferramentas import verificar_recursos, imprimir_banner
from neurapose_backend.app.pipeline.processador import processar_video
from neurapose_backend.app.utils.gerar_graficos import gerar_todos_graficos
from neurapose_backend.LSTM.modulos.fabrica_modelo import ClassifierFactory
from neurapose_backend.globals.state import state

# colorama_init(autoreset=True) - Removido para evitar conflito com main.py
args = config_args
def carregar_labels_videos(labels_path: Path):
    with open(labels_path, 'r', encoding='utf-8') as f: data = json.load(f)
    return data

def format_seconds_to_hms(seconds):
    """Formata segundos para H:M:S ou apenas Segundos se < 60."""
    if seconds < 60:
        return f"{seconds:.2f} seg"
    
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    
    parts = []
    if h > 0: parts.append(f"{int(h)}h")
    if m > 0: parts.append(f"{int(m)}m")
    parts.append(f"{s:.2f}s")
    
    return " ".join(parts)

def main():
    # 1. Resolver paths PRIMEIRO para poder exibir no banner corretamente
    arg_model = getattr(args, 'model_dir', '')
    model_dir = Path(arg_model) if arg_model else MODEL_DIR
    best_model_path = model_dir / "model_best.pt"
    
    if model_dir.is_file():
        best_model_path = model_dir
        model_dir = model_dir.parent
    elif not best_model_path.exists():
        candidates = list(model_dir.glob("*.pt"))
        if candidates: best_model_path = candidates[0]
        else:
             # Se nao achou nada, mantem para falhar no verificar_recursos ou tratar depois
             pass

    # 2. Detectar nome amigável do modelo
    folder_name = model_dir.name.lower()
    display_name = None
    
    MODEL_DISPLAY_MAP = {
        "tcn": "Temporal Convolutional Network (TCN)",
        "bilstm": "BiLSTM (Bidirectional LSTM)",
        "attention": "AttentionLSTM",
        "robust": "RobustLSTM",
        "pooled": "PooledLSTM",
        "wavenet": "WaveNet",
        "transformer": "Transformer",
        "tft": "Temporal Fusion Transformer",
        "lstm": "LSTM (Long Short-Term Memory)"
    }
    
    # Ordem de prioridade (ex: bilstm > lstm)
    for key, name in MODEL_DISPLAY_MAP.items():
        # Verifica padroes comuns: "_tcn", "-tcn", "modelo_tcn"
        if key in folder_name:
             # Refinamento para evitar clash (ex: 'lstm' em 'bilstm')
             if key == "lstm" and "bilstm" in folder_name: continue
             display_name = name
             break
             
    if not display_name:
         display_name = "Temporal Fusion Transformer" if cm.TEMPORAL_MODEL == "tft" else "LSTM / BiLSTM"

    # 3. Validar recursos e Imprimir
    checks = verificar_recursos()
    checks['model_name_display'] = display_name
    # Update modelo_temporal check based on actual resolved path
    checks['modelo_temporal'] = best_model_path.exists()
    
    imprimir_banner(checks)
    
    # Se nao achou modelo, sai agora que ja avisou no banner
    if not best_model_path.exists(): return

    video_input = Path(args.input_dir) if args.input_dir else DATASET_DIR
    if video_input.is_dir():
        if (video_input / "teste" / "videos").exists(): video_input = video_input / "teste" / "videos"
        elif (video_input / "videos").exists(): video_input = video_input / "videos"
             
    dataset_root = video_input
    if video_input.name == "videos":
        dataset_root = video_input.parent
        if dataset_root.name == "teste": dataset_root = dataset_root.parent

    labels_gt_path = LABELS_TEST_PATH
    possible_labels = [
        dataset_root / "teste" / "anotacoes" / "labels.json",
        dataset_root / "anotacoes" / "labels.json",
        video_input.parent / "anotacoes" / "labels.json",
        LABELS_TEST_PATH
    ]
    for p in possible_labels:
        if p.exists():
            labels_gt_path = p
            break

    if not cm.RTMPOSE_PATH.exists(): return

    lstm_model, norm_stats = ClassifierFactory.load(model_dir, device=cm.DEVICE)
    lstm_model.eval()

    if video_input.is_file(): video_list = [video_input]
    else: video_list = sorted(video_input.glob("*.mp4")) + sorted(video_input.glob("*.m4v"))
    
    if not video_list: return
    
    print(Fore.BLUE + f"[INFO] ENCONTRADOS {len(video_list)} VIDEOS")


    labels_gt = {}
    if labels_gt_path.exists(): labels_gt = carregar_labels_videos(labels_gt_path)

    out_report_dir = cm.TEST_REPORTS_DIR / model_dir.name
    out_report_dir.mkdir(parents=True, exist_ok=True)
    out_metricas_dir = out_report_dir / "metricas"
    out_metricas_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = {}
    total_videos = len(video_list)
    total_time_all = 0.0
    start_time_total = time.time()
    
    for i, video_path in enumerate(video_list):
        if state.stop_requested: break
        print(Fore.BLUE + f"[{i+1}/{total_videos}] PROCESSANDO: {video_path.name}")
        
        predictions = processar_video(video_path, lstm_model, norm_stats.get("mu"), norm_stats.get("sigma"), show_preview=args.show, output_dir=out_report_dir, labels_path=labels_gt_path)
        
        if predictions and 'tempos' in predictions and 'total' in predictions['tempos']:
            total_time_all += predictions['tempos']['total']

        all_predictions[video_path.stem] = predictions
        
        print(Fore.BLUE + f"[INFO] SALVANDO O PROCESSAMENTO: {video_path.name}...") 
        print(Fore.GREEN + f"[OK]" + Fore.WHITE + f" SALVAMENTO CONCLUIDO!!")

    if state.stop_requested: return

    elapsed_total = time.time() - start_time_total
    print(Fore.CYAN + f"\n[INFO] TEMPO TOTAL DE TESTE DOS {len(video_list)} VIDEOS: {format_seconds_to_hms(elapsed_total)}")
    
    print(Fore.BLUE + "[INFO] SALVANDO RELATÓRIO COMPLETO DE TESTE DO MODELO...")

    # Metricas
    if labels_gt:
        y_true, y_pred = [], []
        for video_stem, preds in all_predictions.items():
            if not preds: continue
            gt_map = labels_gt.get(video_stem, {})
            if not gt_map: continue
            for p_info in preds.get("ids_predicoes", []):
                pid = str(p_info["id"])
                if pid in gt_map:
                    gt_id = cm.CLASS_TO_ID.get(gt_map[pid], 0)
                    y_true.append(gt_id)
                    y_pred.append(p_info["classe_id"])

        if y_true:
            metricas = {
                "accuracy": accuracy_score(y_true, y_pred),
                "f1_macro": f1_score(y_true, y_pred, average='macro'),
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
            }
            with open(out_metricas_dir / "metricas.json", "w") as f: json.dump(metricas, f, indent=2)
            
            lista_res = [p for p in all_predictions.values() if p]
            with open(out_metricas_dir / "resultados.json", "w") as f: json.dump(lista_res, f, indent=2)

            # =========================================================
            # NOVO: Relatório Detalhado por Vídeo (Tabela)
            # =========================================================
            detalhes_videos = {}
            
            for video_stem, preds in all_predictions.items():
                if not preds: continue
                
                # Mapa de GT para este video
                gt_map = labels_gt.get(video_stem, {})
                # Se não achar por stem, tenta por nome completo se estiver no json
                if not gt_map:
                     for k in labels_gt:
                        if k in video_stem or video_stem in k:
                            gt_map = labels_gt[k]
                            break
                
                video_details = []
                
                # Itera sobre predições e cruza com GT
                for p_info in preds.get("ids_predicoes", []):
                    pid = str(p_info["id"])
                    
                    # Classe Predita
                    pred_cls_name = cm.CLASSE2 if p_info["classe_id"] == 1 else cm.CLASSE1
                    score_val = p_info.get(f"score_{cm.CLASSE2}", 0.0)
                    if p_info["classe_id"] == 0: 
                        score_val = 1.0 - score_val # Confiança da classe 0
                    
                    # Classe Real
                    real_cls_name = "?"
                    if pid in gt_map:
                        val = gt_map[pid]
                        if isinstance(val, dict): val = val.get("classe", "?")
                        real_cls_name = str(val).upper()
                    
                    # Verifica acerto
                    is_ok = False
                    if real_cls_name != "?":
                        is_ok = (real_cls_name == pred_cls_name)
                    
                    video_details.append({
                        "id": int(pid),
                        "real": real_cls_name,
                        "predito": pred_cls_name,
                        "conf": round(score_val * 100, 2),
                        "ok": is_ok,
                        "status_symbol": "✓" if is_ok else "✗" if real_cls_name != "?" else "-"
                    })
                
                # Ordena por ID
                video_details.sort(key=lambda x: x["id"])
                detalhes_videos[video_stem] = video_details
            
            # Salva o arquivo detalhado
            with open(out_metricas_dir / "detalhes_predicoes.json", "w", encoding='utf-8') as f:
                json.dump(detalhes_videos, f, indent=2, ensure_ascii=False)
                
            print(Fore.BLUE + f"[INFO] RELATÓRIO DETALHADO SALVO EM: detalhes_predicoes.json")

            
            gerar_todos_graficos(out_metricas_dir / "metricas.json", out_metricas_dir / "resultados.json", labels_gt_path, model_dir.name)

    print(Fore.GREEN + "[OK]" + Fore.WHITE + " FINALIZANDO O PROGRAMA DE TESTE DO MODELO...")

if __name__ == "__main__":
    main()