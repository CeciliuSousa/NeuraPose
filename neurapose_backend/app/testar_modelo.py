# ==============================================================
# neurapose-backend/app/testar_modelo.py
# ==============================================================

"""
Script principal para teste do modelo treinado.
Processa videos, gera predicoes e calcula metricas.
"""

import json
from pathlib import Path
from colorama import Fore, init as colorama_init
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, balanced_accuracy_score, matthews_corrcoef
)

import warnings
warnings.filterwarnings("ignore")

from neurapose_backend.app.configuracao.config import (
    DATASET_DIR, MODEL_DIR,
    LABELS_TEST_PATH,
    args as config_args
)
import neurapose_backend.config_master as cm


from neurapose_backend.app.utils.ferramentas import verificar_recursos, imprimir_banner
from neurapose_backend.app.pipeline.processador import processar_video

from neurapose_backend.app.utils.gerar_graficos import gerar_todos_graficos
from neurapose_backend.LSTM.modulos.fabrica_modelo import ClassifierFactory

# Estado global para controle de parada
from neurapose_backend.globals.state import state


colorama_init(autoreset=True)
args = config_args


def carregar_labels_videos(labels_path: Path):
    """Carrega labels ground-truth dos videos."""
    with open(labels_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def classificar_video_por_label(video_label):
    """Classifica video: 1 se tem CLASSE1, 0 se CLASSE2."""
    if isinstance(video_label, dict):
        for label in video_label.values():
            if label.lower() == cm.CLASSE1:
                return 1
        return 0
    return 0


def main():
    # Banner
    checks = verificar_recursos()
    imprimir_banner(checks)
    
    # Se fornecido via argumento, usamos. Caso contrário, config_master
    # Se fornecido via argumento, usamos. Caso contrário, config_master
    # from neurapose_backend.config_master import TEST_REPORTS_DIR, RTMPOSE_PATH as CM_RTMPOSE (Using cm)
    
    # Determina diretório do modelo
    arg_model = getattr(args, 'model_dir', '')
    if arg_model and str(arg_model).strip():
        model_dir = Path(arg_model)
    else:
        model_dir = MODEL_DIR

    best_model_path = model_dir / "model_best.pt"
    
    # 1. Lógica do Modelo: Se for arquivo, usa. Se for pasta, busca model_best.pt
    if model_dir.is_file():
        best_model_path = model_dir
        model_dir = model_dir.parent
    elif not best_model_path.exists():
        # Fallback: Tenta buscar qualquer .pt se o model_best.pt nao existir
        candidates = list(model_dir.glob("*.pt"))
        if candidates:
             best_model_path = candidates[0]
             print(Fore.YELLOW + f"[AVISO] model_best.pt não encontrado. Usando {best_model_path.name}")
        else:
             print(Fore.RED + f"[ERRO] Modelo model_best.pt nao encontrado em {model_dir}")
             return

    # 2. Lógica do Dataset: Se for pasta do dataset, busca subpastas padrao
    video_input = Path(args.input_dir) if args.input_dir else DATASET_DIR
    
    # Se o usuario passou a raiz do dataset (ex: datasets/meu-dataset), ajustamos para vídeos
    if video_input.is_dir():
        # Prioridade 1: teste/videos
        if (video_input / "teste" / "videos").exists():
             video_input = video_input / "teste" / "videos"
        # Prioridade 2: videos
        elif (video_input / "videos").exists():
             video_input = video_input / "videos"
             
    # 3. Lógica das Labels: Busca relativa ao dataset (não necessariamente aos videos)
    # Voltar para a raiz do dataset se estivermos na pasta de videos
    dataset_root = video_input
    if video_input.name == "videos":
        dataset_root = video_input.parent # volta para 'teste' ou raiz
        if dataset_root.name == "teste":
            dataset_root = dataset_root.parent # volta para raiz do dataset

    # Tenta caminhos padrao de labels
    possible_labels = [
        dataset_root / "teste" / "anotacoes" / "labels.json",
        dataset_root / "anotacoes" / "labels.json",
        video_input.parent / "anotacoes" / "labels.json",  # Caso esteja em teste/videos
        LABELS_TEST_PATH
    ]
    
    labels_gt_path = LABELS_TEST_PATH # Fallback final
    for p in possible_labels:
        if p.exists():
            labels_gt_path = p
            break

    # Caminho do RTMPose (config_master)
    rtmpose_p = cm.RTMPOSE_PATH
    if not rtmpose_p.exists():
        print(Fore.RED + f"[ERRO] RTMPose nao encontrado em {rtmpose_p}")
        return

    # Carrega modelo LSTM/TFT
    # print(Fore.CYAN + f"\n[MODELO] Carregando modelo de: {model_dir}")
    lstm_model, norm_stats = ClassifierFactory.load(model_dir, device=cm.DEVICE)
    lstm_model.eval()

    # Lista videos
    if video_input.is_file():
        video_list = [video_input]
    else:
        video_list = sorted(video_input.glob("*.mp4"))
        video_list += sorted(video_input.glob("*.m4v"))
    
    if not video_list:
        print(Fore.YELLOW + f"[AVISO] Nenhum vídeo encontrado em {video_input}")
        return

    print(Fore.CYAN + f"[INFO] {len(video_list)} videos encontrados para testes...")

    # Carrega labels ground-truth
    labels_gt = {}
    if labels_gt_path.exists():
        # print(Fore.BLUE + f"[LABELS] Usando Ground Truth: {labels_gt_path}")
        labels_gt = carregar_labels_videos(labels_gt_path)

    # Output Dir para Relatórios: relatorios-testes / <nome_do_modelo>
    out_report_dir = cm.TEST_REPORTS_DIR / model_dir.name
    out_report_dir.mkdir(parents=True, exist_ok=True)
    out_metricas_dir = out_report_dir / "metricas"
    out_metricas_dir.mkdir(parents=True, exist_ok=True)

    # Processa videos
    all_predictions = {}
    for video_path in video_list:
        # Verifica se foi solicitada parada
        if state.stop_requested:
            print(Fore.YELLOW + "[STOP] Teste interrompido pelo usuário.")
            break

        print(Fore.MAGENTA + f"\n[VIDEO] {video_path.name}")
        
        predictions = processar_video(
            video_path=video_path,
            model=lstm_model,
            mu=norm_stats.get("mu"),
            sigma=norm_stats.get("sigma"),
            show_preview=args.show,
            output_dir=out_report_dir
        )
        
        all_predictions[video_path.stem] = predictions

    # Verifica se deve abortar antes de calcular métricas
    if state.stop_requested:
        return

    # Calcula metricas
    if labels_gt:
        y_true, y_pred = [], []
        
        for video_stem, preds in all_predictions.items():
            if preds is None:
                continue
            
            # Buscar mapa de labels para este vídeo (ex: {"1": "normal", "6": "furto"})
            gt_map = labels_gt.get(video_stem, {})
            # Se não tiver labels para o vídeo, ignora
            if not gt_map:
                continue

            # Iterar sobre cada ID predito no vídeo
            for p_info in preds.get("ids_predicoes", []):
                pid = str(p_info["id"])
                
                # Só avalia se existe label para este ID específico
                if pid in gt_map:
                    label_str = gt_map[pid]
                    # Converter label string para ID numérico (ex: "furto" -> 1)
                    gt_id = cm.CLASS_TO_ID.get(label_str, 0)
                    
                    pred_id = p_info["classe_id"]
                    
                    y_true.append(gt_id)
                    y_pred.append(pred_id)
                    
                    classe_pred_str = cm.CLASSE2 if pred_id == 1 else cm.CLASSE1
                    print(Fore.CYAN + f"[ID {pid}] Real: {label_str.upper()} | Pred: {classe_pred_str}")

        if y_true:
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            bacc = balanced_accuracy_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            conf_matrix = confusion_matrix(y_true, y_pred)

            print(Fore.GREEN + "\n" + "="*50)
            print("METRICAS DE VALIDAÇÃO")
            print("="*50)
            print(f"Acuracia: {acc:.4f}")
            print(f"F1 Macro: {f1:.4f}")
            print(f"Precisao: {prec:.4f}")
            print(f"Recall: {rec:.4f}")
            print(f"Balanced Acc: {bacc:.4f}")
            print(f"MCC: {mcc:.4f}")
            print(f"\nMatriz Confusao:\n{conf_matrix}")

            # Salva metricas
            metricas = {
                "accuracy": acc, "f1_macro": f1, "precision": prec,
                "recall": rec, "balanced_accuracy": bacc, "mcc": mcc,
                "confusion_matrix": conf_matrix.tolist(),
                "model_name": model_dir.name,
                "dataset_videos": str(video_input)
            }
            
            metricas_path = out_metricas_dir / "metricas.json"
            with open(metricas_path, "w", encoding="utf-8") as f:
                json.dump(metricas, f, indent=2)
            
            # Salva resultados brutos para graficos
            resultados_path = out_metricas_dir / "resultados.json"
            # Converter keys do all_predictions para lista ou dict serializavel se necessario
            # all_predictions é dict {stem: predictions_dict}
            # O grafico espera lista de resultados (video, pred, score...)
            lista_resultados = []
            for video_stem, preds in all_predictions.items():
                if preds:
                    lista_resultados.append(preds)
            
            with open(resultados_path, "w", encoding="utf-8") as f:
                json.dump(lista_resultados, f, indent=2)

            # Gera graficos
            gerar_todos_graficos(metricas_path, resultados_path, labels_gt_path, model_dir.name)
            
            # print(Fore.GREEN + f"\n[OK] Relatorio salvo em: {out_report_dir}")

if __name__ == "__main__":
    main()