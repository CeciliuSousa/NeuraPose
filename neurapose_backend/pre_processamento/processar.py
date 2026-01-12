# ======================================================
# neurapose_backend/pre_processamento/processar.py
# ======================================================

import sys
import argparse
from pathlib import Path
from colorama import Fore, init as colorama_init

# Importa configuracoes centralizadas
from neurapose_backend.config_master import (
    PROCESSING_INPUT_DIR,
    PROCESSING_OUTPUT_DIR,
    RTMPOSE_PREPROCESSING_PATH,
)


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
    parser.add_argument("--input-folder", type=str, default=str(PROCESSING_INPUT_DIR), 
                        help=f"Pasta de videos (default: {PROCESSING_INPUT_DIR})")
    parser.add_argument("--output-root", type=str, default=str(PROCESSING_OUTPUT_DIR),
                        help=f"Pasta de saida (default: {PROCESSING_OUTPUT_DIR})")
    parser.add_argument("--show", action="store_true", help="Mostrar preview")
    parser.add_argument("--onnx", type=str, default=str(RTMPOSE_PREPROCESSING_PATH),
                        help=f"Caminho do modelo ONNX (default: {RTMPOSE_PREPROCESSING_PATH})")

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
        
        for v in videos:
            preds_dir = out_root / "predicoes"
            json_dir = out_root / "jsons"
            # Verifica se o vídeo já foi processado (pred video ou tracking json existe)
            already_processed = any(preds_dir.glob(f"{v.stem}*pose.mp4")) or any(json_dir.glob(f"{v.stem}*tracking.json"))
            if already_processed:
                print(Fore.YELLOW + f"[SKIP] Video já processado: {v.name}")
                continue

            processar_video(v, sess, input_name, out_root, show=args.show) 


if __name__ == "__main__":
    from neurapose_backend import config_master as cm
    cm.imprimir_configs_yolo_botsort()
    main()
