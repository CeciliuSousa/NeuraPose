# neurapose/app/testar_modelo.py
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
from neurapose_backend.app.configuracao.config import (
    CLASSE1, CLASSE2, DATASET_DIR, RTMPOSE_PATH, MODEL_DIR, METRICAS_DIR,
    LABELS_TEST_PATH, DEVICE,
    MODEL_NAME, BEST_MODEL_PATH,
    args as config_args
)

from neurapose_backend.app.utils.ferramentas import verificar_recursos, imprimir_banner, carregar_sessao_onnx
from neurapose_backend.app.pipeline.processador_video import processar_video
from neurapose_backend.app.utils.gerar_graficos import gerar_todos_graficos
from neurapose_backend.LSTM import ClassifierFactory

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
    
    # Verificacoes
    if not BEST_MODEL_PATH.exists():
        print(Fore.RED + f"[ERRO] Modelo nao encontrado: {BEST_MODEL_PATH}")
        return
        
    if not RTMPOSE_PATH.exists():
        print(Fore.RED + f"[ERRO] RTMPose nao encontrado: {RTMPOSE_PATH}")
        return

    # Carrega modelo LSTM/TFT
    print(Fore.CYAN + f"\n[MODELO] Carregando {MODEL_NAME} de {MODEL_DIR}")
    lstm_model, norm_stats = ClassifierFactory.load(MODEL_DIR, device=DEVICE)
    lstm_model.eval()

    # Carrega sessao ONNX
    sess, input_name = carregar_sessao_onnx(str(RTMPOSE_PATH))

    # Lista videos
    if DATASET_DIR.is_file():
        video_list = [DATASET_DIR]
    else:
        video_list = sorted(DATASET_DIR.glob("*.mp4"))
        video_list += sorted(DATASET_DIR.glob("*.m4v"))
    
    print(Fore.CYAN + f"[INFO] {len(video_list)} videos para processar")

    # Carrega labels ground-truth
    labels_gt = {}
    if LABELS_TEST_PATH.exists():
        labels_gt = carregar_labels_videos(LABELS_TEST_PATH)

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
            print("METRICAS")
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
                "confusion_matrix": cm.tolist()
            }
            
            metricas_path = METRICAS_DIR / "metricas.json"
            with open(metricas_path, "w", encoding="utf-8") as f:
                json.dump(metricas, f, indent=2)
            
            # Gera graficos
            gerar_todos_graficos(y_true, y_pred, METRICAS_DIR)
            
            print(Fore.GREEN + f"\n[OK] Metricas salvas em {METRICAS_DIR}")

    print(Fore.GREEN + "\n[FIM] Teste concluido!")


if __name__ == "__main__":
    main()