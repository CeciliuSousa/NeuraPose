# ======================================================
# neurapose_backend/pre_processamento/processar.py
# ======================================================

import sys
import argparse
from pathlib import Path
from colorama import Fore, init as colorama_init

# Importa configuracoes centralizadas
import neurapose_backend.config_master as cm
try:
    from neurapose_backend.app.user_config_manager import UserConfigManager
    # Carrega configs do usuário e atualiza CM
    user_conf = UserConfigManager.load_config()
    for k, v in user_conf.items():
        if hasattr(cm, k): setattr(cm, k, v)
    
    # Recalcula derivadas criticas (SIMCC)
    if hasattr(cm, "RTMPOSE_INPUT_SIZE") and isinstance(cm.RTMPOSE_INPUT_SIZE, (tuple, list)) and len(cm.RTMPOSE_INPUT_SIZE) == 2:
        cm.SIMCC_W = cm.RTMPOSE_INPUT_SIZE[0]
        cm.SIMCC_H = cm.RTMPOSE_INPUT_SIZE[1]
        
    print(f"[INFO] Configurações de usuário carregadas. RTMPose: {cm.RTMPOSE_MODEL} ({cm.SIMCC_W}x{cm.SIMCC_H})")
except ImportError:
    print(Fore.YELLOW + "[AVISO] Não foi possível carregar UserConfigManager. Usando padrões do config_master.")
except Exception as e:
    print(Fore.RED + f"[ERRO] Falha ao aplicar configurações do usuário: {e}")


from neurapose_backend.pre_processamento.utils.ferramentas import imprimir_banner, carregar_sessao_onnx
from neurapose_backend.pre_processamento.pipeline.processador import processar_video


import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Cython evaluation")
warnings.filterwarnings("ignore", message="Initializing zero-element tensors")

colorama_init(autoreset=True)

# ======================================================
# MAIN
# ======================================================
def main():
    parser = argparse.ArgumentParser()

    # Args para sobrescrever paths do config_master
    parser.add_argument("--input-video", type=str, help="Video unico para processar")
    parser.add_argument("--input-folder", type=str, default=str(cm.PROCESSING_INPUT_DIR), 
                        help=f"Pasta de videos (default: {cm.PROCESSING_INPUT_DIR})")
    parser.add_argument("--output-root", type=str, default=str(cm.PROCESSING_OUTPUT_DIR),
                        help=f"Pasta de saida (default: {cm.PROCESSING_OUTPUT_DIR})")
    parser.add_argument("--show", action="store_true", help="Mostrar preview")
    parser.add_argument("--onnx", type=str, default=str(cm.RTMPOSE_PREPROCESSING_PATH),
                        help=f"Caminho do modelo ONNX (default: {cm.RTMPOSE_PREPROCESSING_PATH})")

    args = parser.parse_args()

    if not args.input_video and not args.input_folder:
        print(Fore.RED + "[ERRO] Use --input-video ou --input-folder")
        sys.stdout.flush()
        sys.exit(1)

    onnx_path = Path(args.onnx)
    out_root = Path(args.output_root)
    
    imprimir_banner(onnx_path)
    sys.stdout.flush()

    sess, input_name = carregar_sessao_onnx(str(onnx_path))
    sys.stdout.flush()

    if args.input_video:
        v = Path(args.input_video)
        out_root = out_root / v.stem if out_root == Path(args.output_root) else out_root
        out_root.mkdir(parents=True, exist_ok=True)

        preds_dir = out_root / "predicoes"
        json_dir = out_root / "jsons"
        already_processed = any(preds_dir.glob(f"{v.stem}*pose.mp4")) or any(json_dir.glob(f"{v.stem}*tracking.json"))
        if already_processed:
            print(Fore.YELLOW + f"[SKIP] Video já processado: {v.name}")
            sys.stdout.flush()
            return

        processar_video(v, sess, input_name, out_root, show=args.show)
        return

    if args.input_folder:
        folder = Path(args.input_folder)
        out_root.mkdir(parents=True, exist_ok=True)

        videos = sorted(folder.glob("*.mp4"))
        print(Fore.CYAN + f"[INFO] Encontrados {len(videos)} videos em {folder}")
        sys.stdout.flush()
        
        # Acumulador de tempos
        total_times = {"yolo": 0, "rtmpose": 0, "total": 0}
        videos_processados = 0
        
        for v in videos:
            preds_dir = out_root / "predicoes"
            json_dir = out_root / "jsons"
            # Verifica se o vídeo já foi processado (pred video ou tracking json existe)
            already_processed = any(preds_dir.glob(f"{v.stem}*pose.mp4")) or any(json_dir.glob(f"{v.stem}*tracking.json"))
            if already_processed:
                print(Fore.YELLOW + f"[SKIP] Video já processado: {v.name}")
                continue

            times = processar_video(v, sess, input_name, out_root, show=args.show)
            if times:
                total_times["yolo"] += times["yolo"]
                total_times["rtmpose"] += times["rtmpose"]
                total_times["total"] += times["total"]
                videos_processados += 1
        
        # Tabela final com soma de todos os vídeos
        if videos_processados > 1:
            print(Fore.MAGENTA + "\n" + "="*60)
            print(Fore.MAGENTA + f"  TEMPO TOTAL - {videos_processados} VIDEOS PROCESSADOS")
            print(Fore.MAGENTA + "="*60)
            print(Fore.WHITE + f"  {'Etapa':<30} {'Tempo':>15}")
            print(Fore.WHITE + "-"*60)
            print(Fore.YELLOW + f"  {'YOLO + BoTSORT + OSNet':<30} {total_times['yolo']:>12.2f} seg")
            print(Fore.YELLOW + f"  {'RTMPose (Inferencia)':<30} {total_times['rtmpose']:>12.2f} seg")
            print(Fore.WHITE + "-"*60)
            print(Fore.GREEN + f"  {'TEMPO TOTAL GERAL':<30} {total_times['total']:>12.2f} seg")
            # Converte para minutos se maior que 60
            if total_times['total'] > 60:
                mins = total_times['total'] / 60
                print(Fore.GREEN + f"  {'':<30} {mins:>12.2f} min")
            print(Fore.MAGENTA + "="*60 + "\n")
            sys.stdout.flush() 


if __name__ == "__main__":
    # from neurapose_backend import config_master as cm
    # cm.imprimir_configs_yolo_botsort()
    main()
