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
        print(Fore.CYAN + f"[INFO] ENCONTRADOS {len(v_list)} VIDEOS")
        print("")

    out_root.mkdir(parents=True, exist_ok=True)
    
    total_videos = len(v_list)
    for i, v in enumerate(v_list):
        print(Fore.CYAN + f"[{i+1}/{total_videos}] PROCESSANDO: {v.name}")
        
        # Output especifico por video (se for pasta)
        if args.input_video:
            # Se entrada for video unico
            curr_out = out_root / v.stem if out_root.resolve() == Path(args.output_root).resolve() else out_root
        else:
             # Se for modo pasta, cria pasta irma com sufixo -processado se o output for o default
             # Verifica se out_root eh o default (resultados-processamentos)
            if out_root.resolve() == Path(cm.PROCESSING_OUTPUT_DIR).resolve():
                 parent_processed = out_root / f"{folder.name}-processado"
                 parent_processed.mkdir(parents=True, exist_ok=True)
                 curr_out = parent_processed
            else:
                 # Se o usuario especificou um output customizado, usa ele
                 curr_out = out_root

        curr_out.mkdir(parents=True, exist_ok=True)
        preds_dir = curr_out / "predicoes"
        json_dir = curr_out / "jsons"
        
        # Verifica se existe output: Video de pose, JSON de tracking ou JSON final (30fps)
        # Debug paths
        # print(Fore.YELLOW + f"[DEBUG] Verificando processados em: {curr_out}")
        # print(Fore.YELLOW + f"[DEBUG] Glob preds: {list(preds_dir.glob(f'{v.stem}*pose.mp4'))}")
        # print(Fore.YELLOW + f"[DEBUG] Glob json final: {list(json_dir.glob(f'{v.stem}*_30fps.json'))}")

        exists_pose = any(preds_dir.glob(f"{v.stem}*pose.mp4"))
        exists_tracking = any(json_dir.glob(f"{v.stem}*tracking.json"))
        exists_json_final = any(json_dir.glob(f"{v.stem}*_30fps.json"))
        
        processed = exists_pose or exists_tracking or exists_json_final
        
        if processed:
            print(Fore.MAGENTA + f"[SKIP]" + Fore.WHITE + f" Video já processado ({v.name}).")
            continue

        processar_video(v, curr_out, show=args.show)
        print(Fore.CYAN + f"[INFO] SALVANDO O PROCESSAMENTO: {v.name}...")
        print(Fore.GREEN + f"[OK]" + Fore.WHITE + f" SALVAMENTO CONCLUIDO!!\n")
        sys.stdout.flush()

    print(Fore.GREEN + "[OK]" + Fore.WHITE + " ENCERRANDO PROGRAMA DE PROCESSAMENTO...")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    # from neurapose_backend import config_master as cm
    # cm.imprimir_configs_yolo_botsort()
    main()
