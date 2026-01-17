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

# Configuracoes
from app.configuracao.config import (
    CLASSE1, CLASSE2, DATASET_DIR, RTMPOSE_PATH, MODEL_DIR, METRICAS_DIR,
    LABELS_TEST_PATH, DEVICE,
    MODEL_NAME, BEST_MODEL_PATH,
    args as config_args
)


from app.utils.ferramentas import verificar_recursos, imprimir_banner, carregar_sessao_onnx
from app.pipeline.processador_video import processar_video

from app.utils.gerar_graficos import gerar_todos_graficos
from LSTM import ClassifierFactory


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
            if label.lower() == CLASSE1:
                return 1
        return 0
    return 0


def main():
    # Banner
    checks = verificar_recursos()
    imprimir_banner(checks)
    
    # Se fornecido via argumento, usamos. Caso contrário, config_master
    from config_master import TEST_REPORTS_DIR, RTMPOSE_PATH as CM_RTMPOSE
    
    # Determina diretório do modelo
    model_dir = Path(getattr(args, 'model_dir', '')) or MODEL_DIR
    best_model_path = model_dir / "model_best.pt"
    if not best_model_path.exists():
        # Tenta o próprio path se for um arquivo
        if model_dir.is_file():
            best_model_path = model_dir
            model_dir = model_dir.parent
        else:
            print(Fore.RED + f"[ERRO] Modelo model_best.pt nao encontrado em {model_dir}")
            return
            
    # Determina diretório de vídeos (dataset)
    # User quer teste/videos/ se for um dataset
    video_input = Path(args.input_dir) if args.input_dir else DATASET_DIR
    if video_input.is_dir():
        if (video_input / "teste" / "videos").exists():
            video_input = video_input / "teste" / "videos"
        elif (video_input / "videos").exists():
            video_input = video_input / "videos"
            
    # Labels ground-truth: busca automática
    labels_gt_path = video_input.parent / "anotacoes" / "labels.json"
    if not labels_gt_path.exists():
        # Tenta na raiz do que foi selecionado
        labels_gt_path = video_input / "teste" / "anotacoes" / "labels.json"
    if not labels_gt_path.exists():
        labels_gt_path = video_input / "anotacoes" / "labels.json"
    if not labels_gt_path.exists():
        labels_gt_path = LABELS_TEST_PATH

    # Caminho do RTMPose (config_master)
    rtmpose_p = CM_RTMPOSE
    if not rtmpose_p.exists():
        print(Fore.RED + f"[ERRO] RTMPose nao encontrado em {rtmpose_p}")
        return

    # Carrega modelo LSTM/TFT
    print(Fore.CYAN + f"\n[MODELO] Carregando modelo de: {model_dir}")
    lstm_model, norm_stats = ClassifierFactory.load(model_dir, device=DEVICE)
    lstm_model.eval()

    # Carrega sessao ONNX
    sess, input_name = carregar_sessao_onnx(str(rtmpose_p))

    # Lista videos
    if video_input.is_file():
        video_list = [video_input]
    else:
        video_list = sorted(video_input.glob("*.mp4"))
        video_list += sorted(video_input.glob("*.m4v"))
    
    if not video_list:
        print(Fore.YELLOW + f"[AVISO] Nenhum vídeo encontrado em {video_input}")
        return

    print(Fore.CYAN + f"[INFO] {len(video_list)} videos para processar")

    # Carrega labels ground-truth
    labels_gt = {}
    if labels_gt_path.exists():
        print(Fore.BLUE + f"[LABELS] Usando Ground Truth: {labels_gt_path}")
        labels_gt = carregar_labels_videos(labels_gt_path)

    # Output Dir para Relatórios: relatorios-testes / <nome_do_modelo>
    out_report_dir = TEST_REPORTS_DIR / model_dir.name
    out_report_dir.mkdir(parents=True, exist_ok=True)
    out_metricas_dir = out_report_dir / "metricas"
    out_metricas_dir.mkdir(parents=True, exist_ok=True)

    # Processa videos
    all_predictions = {}
    for video_path in video_list:
        print(Fore.MAGENTA + f"\n[VIDEO] {video_path.name}")
        
        predictions = processar_video(
            video_path=video_path,
            sess=sess,
            input_name=input_name,
            model=lstm_model,
            mu=norm_stats.get("mu"),
            sigma=norm_stats.get("sigma"),
            show_preview=args.show
        )
        
        all_predictions[video_path.stem] = predictions

    # Calcula metricas
    if labels_gt:
        y_true, y_pred = [], []
        
        for video_stem, preds in all_predictions.items():
            if video_stem in labels_gt:
                gt = classificar_video_por_label(labels_gt[video_stem])
                pred = 1 if any(p.get(CLASSE2, False) for p in preds) else 0
                y_true.append(gt)
                y_pred.append(pred)

        if y_true:
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            bacc = balanced_accuracy_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)

            print(Fore.GREEN + "\n" + "="*50)
            print("METRICAS DE VALIDAÇÃO")
            print("="*50)
            print(f"Acuracia: {acc:.4f}")
            print(f"F1 Macro: {f1:.4f}")
            print(f"Precisao: {prec:.4f}")
            print(f"Recall: {rec:.4f}")
            print(f"Balanced Acc: {bacc:.4f}")
            print(f"MCC: {mcc:.4f}")
            print(f"\nMatriz Confusao:\n{cm}")

            # Salva metricas
            metricas = {
                "accuracy": acc, "f1_macro": f1, "precision": prec,
                "recall": rec, "balanced_accuracy": bacc, "mcc": mcc,
                "confusion_matrix": cm.tolist(),
                "model_name": model_dir.name,
                "dataset_videos": str(video_input)
            }
            
            metricas_path = out_metricas_dir / "metricas.json"
            with open(metricas_path, "w", encoding="utf-8") as f:
                json.dump(metricas, f, indent=2)
            
            # Gera graficos
            gerar_todos_graficos(y_true, y_pred, out_metricas_dir)
            
            print(Fore.GREEN + f"\n[OK] Relatorio salvo em: {out_report_dir}")

    print(Fore.GREEN + "\n[FIM] Teste concluido!")


if __name__ == "__main__":
    main()