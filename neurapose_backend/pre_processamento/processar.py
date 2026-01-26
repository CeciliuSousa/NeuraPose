# neurapose_backend/pre_processamento/processar.py
# Script principal de pre-processamento.

import sys
import os
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import argparse
from pathlib import Path
from colorama import Fore, init as colorama_init
import neurapose_backend.config_master as cm

try:
    from neurapose_backend.nucleo.user_config_manager import UserConfigManager
    user_conf = UserConfigManager.load_config()
    for k, v in user_conf.items():
        if hasattr(cm, k): setattr(cm, k, v)
    
    if hasattr(cm, "RTMPOSE_INPUT_SIZE") and isinstance(cm.RTMPOSE_INPUT_SIZE, (tuple, list)) and len(cm.RTMPOSE_INPUT_SIZE) == 2:
        cm.SIMCC_W = cm.RTMPOSE_INPUT_SIZE[0]
        cm.SIMCC_H = cm.RTMPOSE_INPUT_SIZE[1]
except ImportError:
    pass
except Exception as e:
    print(Fore.RED + f"[ERRO] Config user: {e}")

from neurapose_backend.pre_processamento.utils.ferramentas import imprimir_banner
from neurapose_backend.pre_processamento.pipeline.processador import processar_video

import warnings
warnings.filterwarnings("ignore")

colorama_init(autoreset=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-video", type=str, help="Video unico")
    parser.add_argument("--input-folder", type=str, default=str(cm.PROCESSING_INPUT_DIR))
    parser.add_argument("--output-root", type=str, default=str(cm.PROCESSING_OUTPUT_DIR))
    parser.add_argument("--show", action="store_true", help="Preview")
    parser.add_argument("--onnx", type=str, default=str(cm.RTMPOSE_PREPROCESSING_PATH))

    args = parser.parse_args()

    if not args.input_video and not args.input_folder:
        print(Fore.RED + "[ERRO] Use --input-video ou --input-folder")
        sys.exit(1)

    onnx_path = Path(args.onnx)
    if onnx_path.is_dir():
        onnx_files = list(onnx_path.glob("**/*.onnx"))
        if not onnx_files:
            sys.exit(1)
        onnx_path = onnx_files[0]
        
    out_root = Path(args.output_root)
    imprimir_banner(onnx_path)
    sys.stdout.flush()

    # Logica de Processamento
    if args.input_video:
        v_list = [Path(args.input_video)]
    else:
        folder = Path(args.input_folder)
        v_list = sorted(folder.glob("*.mp4"))
        print(Fore.BLUE + f"[OK] DIRETORIO DE VIDEOS ENCONTRADO, TOTAL DE VIDEOS: {len(v_list)}")
        print("")

    out_root.mkdir(parents=True, exist_ok=True)
    
    total_videos = len(v_list)
    for i, v in enumerate(v_list):
        print(Fore.MAGENTA + f"[VIDEO] PROCESSANDO: {v.name} - [{i+1} / {total_videos}]")
        
        # Output especifico por video (se for pasta)
        if args.input_video:
            curr_out = out_root / v.stem if out_root == Path(args.output_root) else out_root
        else:
            curr_out = out_root

        curr_out.mkdir(parents=True, exist_ok=True)
        preds_dir = curr_out / "predicoes"
        json_dir = curr_out / "jsons"
        
        processed = any(preds_dir.glob(f"{v.stem}*pose.mp4")) or any(json_dir.glob(f"{v.stem}*tracking.json"))
        if processed:
            print(Fore.YELLOW + f"[SKIP] Video já processado.")
            continue

        processar_video(v, curr_out, show=args.show)
        print(Fore.GREEN + f"[OK] SALVAMENTO CONCLUIDO!!\n")
        sys.stdout.flush()

    print(Fore.GREEN + "[OK] FINALIZANDO O PROGRAMA DE PROCESSAMENTO...")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    # from neurapose_backend import config_master as cm
    # cm.imprimir_configs_yolo_botsort()
    main()
