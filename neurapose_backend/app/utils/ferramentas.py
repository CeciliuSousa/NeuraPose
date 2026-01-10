# ==============================================================
# src/utils/ferramentas.py (COMPLETO E CORRIGIDO)
# ==============================================================

import torch
import onnxruntime as ort
from pathlib import Path
from yt_dlp import YoutubeDL
from colorama import Fore
from neurapose_backend.app.configuracao.config import (
    YOLO_PATH,
    OSNET_PATH,
    RTMPOSE_PATH,
    BEST_MODEL_PATH,
    LABELS_TEST_PATH,
    DATASET_DIR,
    TRACKER_NAME,
    DEVICE,
    MODEL_NAME, 
    DATASET_NAME 
)

def status_str(ok: bool):
    """Retorna uma string colorida indicando sucesso ou falha."""
    return Fore.GREEN + "[OK]" if ok else Fore.RED + "[ERRO]"


def verificar_recursos():
    """
    Verifica se todos os arquivos e diretórios necessários existem.
    Retorna um dicionário com o status de cada recurso.
    """
    modelo_default = f"{MODEL_NAME}-{DATASET_NAME}"
    
    return {
        "yolo": YOLO_PATH.exists(),
        "osnet": OSNET_PATH.exists(),
        "rtmpose": RTMPOSE_PATH.exists(),
        "modelo_temporal": BEST_MODEL_PATH.exists(),
        "labels": LABELS_TEST_PATH.exists(),
        "dataset": DATASET_DIR.exists(),
        
        "modelo_temporal_nome": modelo_default, 
        "dataset_path": str(DATASET_DIR),
    }


def imprimir_banner(checks):
    """
    Imprime o banner inicial do sistema com o status dos recursos,
    garantindo o alinhamento.
    """
    print("\n======================================================================")
    print("SISTEMA DE DETECÇÃO — NEURAPOSE AI")
    print("======================================================================")
    
    print(f"YOLO                : {status_str(checks['yolo'])} {YOLO_PATH.name}")
    print(f"TRACKER             : {status_str(True)} {TRACKER_NAME}")
    print(f"OSNet ReID          : {status_str(checks['osnet'])} {OSNET_PATH.name}")
    print(
        f"RTMPose-l           : {status_str(checks['rtmpose'])} "
        f"{RTMPOSE_PATH.parent.name}/{RTMPOSE_PATH.name}"
    )
    print(
        f"Modelo Temporal     : {status_str(checks['modelo_temporal'])} {checks['modelo_temporal_nome']}"
    )
    print(
        f"Labels de Teste     : {status_str(checks['labels'])} {LABELS_TEST_PATH.name}"
    )
    print("----------------------------------------------------------------------")
    
    print(f"Dataset de vídeos de teste: {checks['dataset_path']}")
    if DEVICE.startswith("cuda"):
        try:
            print(Fore.GREEN + f"GPU detectada: {torch.cuda.get_device_name(0)}")
        except Exception:
            print(Fore.GREEN + "GPU detectada (não foi possível obter o nome).")
    else:
        print(Fore.YELLOW + "Executando em CPU.")
    print("======================================================================\n")


def carregar_sessao_onnx(model_path: str | Path):
    """
    Carrega uma sessão de inferência ONNX (RTMPose), tentando CUDA e depois CPU.
    Retorna (sessão, nome_input).
    """
    tentativas = []
    providers_list = [
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"]
    ]

    for providers in providers_list:
        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(str(model_path), providers=providers, sess_options=opts)
            input_name = sess.get_inputs()[0].name
            
            # VERIFICAÇÃO DE PROVIDER ATIVO
            active_providers = sess.get_providers()
            if 'CUDAExecutionProvider' in active_providers:
                print(Fore.GREEN + "RTMPose usando GPU (CUDAExecutionProvider)")
            else:
                print(Fore.YELLOW + f"RTMPose usando CPU ({active_providers[0]})")
            
            return sess, input_name
        except Exception as e:
            tentativas.append((providers, str(e)))
            print(Fore.RED + f"Falha com providers {providers}: {e}")

    raise RuntimeError(f"Não foi possível iniciar sessão ONNX. Tentativas: {tentativas}")


def baixar_video_ytdlp(url: str, pasta_saida: Path) -> Path:
    """
    Baixa um vídeo do YouTube usando yt-dlp.
    Retorna o caminho do arquivo baixado.
    """
    pasta_saida.mkdir(parents=True, exist_ok=True)
    print(Fore.CYAN + f"Baixando vídeo do YouTube...\n {url}")
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "outtmpl": str(pasta_saida / "%(title)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": False
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        output_file = pasta_saida / f"{info['title']}.mp4"
    print(Fore.GREEN + f"Download concluído: {output_file}")
    return output_file