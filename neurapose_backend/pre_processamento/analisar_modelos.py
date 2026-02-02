# ===========================================================
# neurapose_backend/pre_processamento/analisar_modelos.py
# ===========================================================

import os
import re
from colorama import Fore, init
from datetime import datetime

init(autoreset=True)

BASE_DIR = "neurapose_backend/modelos-temporais"
OUT_PATH = "neurapose_backend/modelos-temporais/Ranking_Modelos.txt"

# -----------------------------------------------------------
# Extração das métricas principais do relatório
# -----------------------------------------------------------
def extract_metrics_from_txt(file_path):
    metrics = {"F1": None, "Acc": None, "Loss": None, "conf_matrix": None}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # busca métricas gerais
        f1_match = re.search(r"F1(?: Macro)?[^:\d]*[: ]+([\d\.]+)", text)
        acc_match = re.search(r"Acc[^:\d]*[: ]+([\d\.]+)", text)
        loss_match = re.search(r"Loss[^:\d]*[: ]+([\d\.]+)", text)
        epoch_match = re.search(r"Melhor epoca: (\d+)", text)
        
        # tenta extrair matriz de confusão (formato textual)
        matrix_match = re.search(
            r"Matriz de Confusão.*?normal.*?(\d+).*?(\d+).*?furto.*?(\d+).*?(\d+)",
            text,
            flags=re.S | re.I
        )

        if f1_match:
            metrics["F1"] = float(f1_match.group(1))
        if acc_match:
            metrics["Acc"] = float(acc_match.group(1))
        if loss_match:
            metrics["Loss"] = float(loss_match.group(1))
        if matrix_match:
            metrics["conf_matrix"] = {
                "normal": [int(matrix_match.group(1)), int(matrix_match.group(2))],
                "furto": [int(matrix_match.group(3)), int(matrix_match.group(4))]
            }

        # Se Loss for None, tenta buscar no JSON de histórico usando a melhor época
        if metrics["Loss"] is None and epoch_match:
            best_epoch = int(epoch_match.group(1))
            json_path = file_path.replace("relatorio_", "historico_").replace(".txt", ".json")
            
            if os.path.exists(json_path):
                try:
                    import json
                    with open(json_path, "r", encoding="utf-8") as jf:
                        history = json.load(jf)
                        # Busca a entrada correspondente à melhor época
                        for entry in history:
                            if entry.get("epoch") == best_epoch:
                                val_loss = entry.get("val_loss")
                                if val_loss is not None:
                                    metrics["Loss"] = float(val_loss)
                                break
                except Exception as json_err:
                    print(Fore.YELLOW + f"Aviso: erro ao ler json {json_path}: {json_err}")

    except Exception as e:
        print(Fore.RED + f"Erro ao ler {file_path}: {e}")
    return metrics

# -----------------------------------------------------------
# Funções auxiliares
# -----------------------------------------------------------
def colorize_metric(value, is_loss=False):
    if value is None:
        return Fore.WHITE + "N/A"
    if is_loss:
        if value < 0.4: return Fore.GREEN + f"{value:.4f}"
        if value < 0.6: return Fore.YELLOW + f"{value:.4f}"
        return Fore.RED + f"{value:.4f}"
    else:
        if value >= 0.85: return Fore.GREEN + f"{value:.4f}"
        if value >= 0.70: return Fore.YELLOW + f"{value:.4f}"
        return Fore.RED + f"{value:.4f}"

def find_report_files(model_dir):
    for fname in os.listdir(model_dir):
        if fname.startswith("relatorio_") and fname.endswith(".txt"):
            return os.path.join(model_dir, fname)
    return None

def find_image_files(model_dir):
    imgs = []
    for fname in os.listdir(model_dir):
        if fname.endswith(".png"):
            imgs.append(os.path.join(model_dir, fname))
    return imgs

def classificar_f1(f1):
    if f1 is None:
        return "Sem dados"
    elif f1 >= 0.85:
        return "Excelente desempenho"
    elif f1 >= 0.70:
        return "Bom desempenho"
    else:
        return "Desempenho insatisfatório"

# -----------------------------------------------------------
# Programa principal
# -----------------------------------------------------------
def main():
    models_info = []

    # percorre todas as subpastas de "meus-modelos"
    for folder in sorted(os.listdir(BASE_DIR)):
        full_path = os.path.join(BASE_DIR, folder)
        if not os.path.isdir(full_path):
            continue

        report_path = find_report_files(full_path)
        if not report_path:
            continue

        metrics = extract_metrics_from_txt(report_path)
        imgs = find_image_files(full_path)

        models_info.append({
            "name": folder,
            "report": report_path,
            "metrics": metrics,
            "imgs": imgs
        })

    # ordena do melhor F1 para o pior
    models_info.sort(key=lambda m: (m["metrics"]["F1"] or 0), reverse=True)

    # gera o arquivo de ranking
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        out.write("==== RANKING DE MODELOS - LABEX ====\n")
        out.write(f"Data de geração: {timestamp}\n")
        out.write(f"Total de modelos encontrados: {len(models_info)}\n\n")

        for rank, m in enumerate(models_info, start=1):
            name = m["name"]
            f1 = m["metrics"]["F1"]
            acc = m["metrics"]["Acc"]
            loss = m["metrics"]["Loss"]
            conf = m["metrics"].get("conf_matrix")

            out.write("="*60 + "\n")
            out.write(f"#{rank} - {name.upper()}\n")
            out.write("="*60 + "\n")
            out.write(f"  F1 Macro : {f'{f1:.4f}' if f1 else 'N/A'}\n")
            out.write(f"  Accuracy : {f'{acc:.4f}' if acc else 'N/A'}\n")
            out.write(f"  Loss     : {f'{loss:.4f}' if loss else 'N/A'}\n")
            out.write(f"  Avaliação: {classificar_f1(f1)}\n")

            if conf:
                out.write("\n  --- Matriz de Confusão (Validação) ---\n")
                out.write(f"        normal | TP={conf['normal'][0]}  FN={conf['normal'][1]}\n")
                out.write(f"         furto | FP={conf['furto'][0]}  TN={conf['furto'][1]}\n")

            out.write(f"\n  Relatório: {m['report']}\n")

            if m["imgs"]:
                out.write("  Imagens  :\n")
                for img in m["imgs"]:
                    out.write(f"    - {img}\n")
            out.write("\n")

    # exibe resumo no terminal
    print(Fore.CYAN + f"\n==== RANKING DE MODELOS ({len(models_info)} encontrados) ====")
    for rank, m in enumerate(models_info, start=1):
        name = m["name"]
        f1 = m["metrics"]["F1"]
        acc = m["metrics"]["Acc"]
        loss = m["metrics"]["Loss"]
        conf = m["metrics"].get("conf_matrix")

        print(f"{Fore.WHITE}#{rank:02d} {name:<25} | "
              f"F1: {colorize_metric(f1)} | "
              f"Acc: {colorize_metric(acc)} | "
              f"Loss: {colorize_metric(loss, is_loss=True)}")

        if conf:
            print(f"      {Fore.MAGENTA}Matriz Confusão -> normal:[{conf['normal'][0]},{conf['normal'][1]}]  "
                  f"furto:[{conf['furto'][0]},{conf['furto'][1]}]")

    print(Fore.GREEN + f"\n📊 Ranking completo salvo em: {OUT_PATH}")

# -----------------------------------------------------------
if __name__ == "__main__":
    main()
