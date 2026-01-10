# ==============================================================
# pre_processamento/utils/ferramentas.py
# ==============================================================
# Funções utilitárias gerais: download de vídeos, carregamento
# de modelos ONNX e formatação de logs.

import torch
import onnxruntime as ort
from pathlib import Path
from yt_dlp import YoutubeDL
from colorama import Fore

# Importa do config_master via pre_processamento config (usar import absoluto)
from neurapose.pre_processamento.configuracao.config import TRACKER_NAME

# Importa do config_master (absoluto)
from neurapose.config_master import YOLO_MODEL, OSNET_PATH, DATASET_NAME, ROOT as PROJECT_ROOT


def status_str(ok: bool):
    """Retorna string colorida [OK] ou [ERRO]."""
    return Fore.GREEN + "[OK]" if ok else Fore.RED + "[ERRO]"


def imprimir_banner(onnx_path: Path):
    """Imprime banner informativo sobre o ambiente e modelos."""
    print("\n======================================================================")
    print("PRÉ-PROCESSAMENTO — NEURAPOSE AI")
    print("======================================================================")

    # YOLO (usa config_master)
    yolopath = PROJECT_ROOT / "detector" / "modelos" / YOLO_MODEL
    yolo_name = YOLO_MODEL.replace('.pt', '')
    print(f"YOLO                  : {status_str(yolopath.exists())} {yolo_name}")

    # Tracker
    print(f"TRACKER               : {status_str(True)} {TRACKER_NAME}")

    # OSNet (usa config_master)
    print(f"OSNet ReID            : {status_str(OSNET_PATH.exists())} {OSNET_PATH.name}")

    # RTMPose
    print(
        f"RTMPose-l             : {status_str(onnx_path.exists())} "
        f"{onnx_path.parent.name}/{onnx_path.name}"
    )

    print("----------------------------------------------------------------------")
    # Dataset
    # print(f"Dataset               : {DATASET_NAME}")

    # GPU Info
    if torch.cuda.is_available():
        print(Fore.GREEN + f"GPU detectada         : {torch.cuda.get_device_name(0)}")
    else:
        print(Fore.YELLOW + "Dispositivo           : CPU (sem GPU)")
    
    print("======================================================================\n")


def baixar_video_ytdlp(url: str, pasta_saida: Path) -> Path:
    """
    Baixa um vídeo do YouTube usando yt-dlp.
    Retorna o caminho do arquivo baixado.
    """
    pasta_saida.mkdir(parents=True, exist_ok=True)
    print(Fore.CYAN + f"🎬 Baixando vídeo do YouTube...\n🔗 {url}")
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "outtmpl": str(pasta_saida / "%(title)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": False
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        output_file = pasta_saida / f"{info['title']}.mp4"
    print(Fore.GREEN + f"✅ Download concluído: {output_file}")
    return output_file


def carregar_sessao_onnx(model_path: str | Path):
    """
    Carrega uma sessão de inferência ONNX (RTMPose), tentando CUDA e depois CPU.
    Retorna (sessão, nome_input).
    """
    tentativas = []
    # Tenta primeiro com CUDA, depois só CPU
    providers_list = [
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"]
    ]

    for providers in providers_list:
        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(str(model_path), providers=providers, sess_options=opts)
            
            # ✅ VERIFICAÇÃO DE PROVIDER ATIVO
            active_providers = sess.get_providers()
            if 'CUDAExecutionProvider' in active_providers:
                print(Fore.GREEN + "✓ RTMPose usando GPU (CUDAExecutionProvider)")
            else:
                print(Fore.YELLOW + f"⚠️ RTMPose usando CPU ({active_providers[0]})")
            
            input_name = sess.get_inputs()[0].name
            return sess, input_name
        except Exception as e:
            tentativas.append((providers, str(e)))
            # Mensagem de erro só se falhar com todos os providers

    raise RuntimeError(f"Não foi possível iniciar sessão ONNX. Tentativas: {tentativas}")
