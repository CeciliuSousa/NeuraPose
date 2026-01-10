# ==========================================================
# neurapose_backend/pre_processamento/anotando_classes.py
# ==========================================================

"""
Ferramenta interativa para anotacao de classes.
Le pastas de predicoes e jsons, exibe videos e permite
classificar cada ID1 como normal ou furto.
"""

import os
import cv2
import json
from pathlib import Path
from collections import defaultdict, Counter
from colorama import Fore, init as colorama_init

# Adiciona root ao path


# Configuracoes centralizadas
from neurapose_backend.config_master import (
    PROCESSING_OUTPUT_DIR,
    PROCESSING_ANNOTATIONS_DIR,
    MIN_FRAMES_PER_ID,
    CLASSE1,
    CLASSE2
)  

colorama_init(autoreset=True)


def listar_jsons(json_dir: Path):
    """Lista arquivos JSON (exceto tracking)."""
    return sorted([p for p in json_dir.glob("*.json") 
                   if p.is_file() and not p.name.endswith("_tracking.json")])


def encontrar_video_para_json(pred_dir: Path, stem: str):
    """Encontra video correspondente ao JSON."""
    candidatos = [pred_dir / f"{stem}_pose.mp4", pred_dir / f"{stem}.mp4"]
    candidatos += list(pred_dir.glob(f"{stem}*.mp4"))
    for c in candidatos:
        if c.exists():
            return c
    return None


def carregar_pose_records(json_path: Path):
    """Carrega registros de pose do JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def indexar_por_frame(records):
    """Indexa registros por frame e conta ocorrencias por ID."""
    frames = defaultdict(list)
    id_counter = Counter()

    for r in records:
        gid = r.get("id_persistente", -1)
        if gid is None or gid < 0:
            continue

        bbox = r.get("bbox", {})
        if isinstance(bbox, list) and len(bbox) == 4:
            bbox = {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}

        frame_idx = int(r.get("frame", 0))
        frames[frame_idx].append((bbox, int(gid)))
        id_counter[int(gid)] += 1

    return frames, id_counter


def exibir_video(video_path: Path, title="Preview"):
    """Exibe video sem overlays."""
    if video_path is None or not video_path.exists():
        print(Fore.YELLOW + f"[AVISO] Video nao encontrado")
        return

    cap = cv2.VideoCapture(str(video_path))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > 960:
            esc = 960 / w
            frame = cv2.resize(frame, (960, int(h * esc)))
        cv2.imshow(title, frame)
        if cv2.waitKey(15) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def exibir_video_com_bboxes(video_path: Path, frames_index, present_ids:set=None, removed_ids:set=None, kept_color=(0,255,0), removed_color=(0,215,255), title="Review"):
    """Exibe video com bounding boxes sobrepostos.
    `present_ids` aparecem em verde; `removed_ids` em amarelo.
    Não desenha legendas extras na imagem (somente bboxes e ID).
    """
    if video_path is None or not video_path.exists():
        print(Fore.YELLOW + f"[AVISO] Video nao encontrado")
        return

    present_ids = set(present_ids) if present_ids else set()
    removed_ids = set(removed_ids) if removed_ids else set()

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > 960:
            esc = 960 / w
            frame = cv2.resize(frame, (960, int(h * esc)))
            scale_x = frame.shape[1] / w
            scale_y = frame.shape[0] / h
        else:
            scale_x = scale_y = 1.0

        entries = frames_index.get(frame_idx, [])
        for bbox, gid in entries:
            try:
                x1 = int(bbox.get("x1", 0) * scale_x)
                y1 = int(bbox.get("y1", 0) * scale_y)
                x2 = int(bbox.get("x2", 0) * scale_x)
                y2 = int(bbox.get("y2", 0) * scale_y)
            except Exception:
                continue

            if gid not in present_ids and gid not in removed_ids:
                continue

            color = kept_color if gid in present_ids else removed_color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Nenhuma legenda/overlay de texto extra exibida (apenas bboxes e IDs)
        cv2.imshow(title, frame)
        if cv2.waitKey(15) & 0xFF == 27:
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


def limpar_json_mantendo_ids(records, kept_ids):
    """Filtra registros mantendo apenas IDs selecionados."""
    kept = set(int(i) for i in kept_ids)
    return [r for r in records if int(r.get("id_persistente", -1)) in kept]


def salvar_json(path: Path, data):
    """Salva JSON de forma atomica."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Anotar IDs por video")
    parser.add_argument("--root", default=str(PROCESSING_OUTPUT_DIR),
                        help=f"Root de processamento (default: {PROCESSING_OUTPUT_DIR})")
    parser.add_argument("--input-dir-process", "--input-dir", dest="root",
                        help="Alias para --root (diretório de processamento)")

    parser.add_argument("--min-frames", type=int, default=MIN_FRAMES_PER_ID,
                        help=f"Minimo de frames por ID (default: {MIN_FRAMES_PER_ID})")
    parser.add_argument("--positive-class", default=CLASSE2.lower(),
                        help="Classe positiva (default: furto)")
    parser.add_argument("--negative-class", default=CLASSE1.lower(),
                        help="Classe negativa (default: normal)")

    parser.add_argument("--labels-out", default=str(PROCESSING_ANNOTATIONS_DIR / "labels.json"),
                        help=f"Caminho para labels de saida (default: {PROCESSING_ANNOTATIONS_DIR / 'labels.json'})")
    parser.add_argument("--output-dir-annotations", dest="labels_out",
                        help="Alias para --labels-out (arquivo de anotacoes de saida)")

    args = parser.parse_args()

    root = Path(args.root).resolve()
    pred_dir = root / "predicoes"
    json_dir = root / "jsons"

    print(Fore.BLUE + f"[CONFIG] Root: {root}")
    print(Fore.BLUE + f"[CONFIG] Classes: {args.negative_class} / {args.positive_class}")

    if not pred_dir.exists() or not json_dir.exists():
        raise FileNotFoundError(f"Pastas nao encontradas: {pred_dir}, {json_dir}")

    labels_out = Path(args.labels_out)
    labels_out.parent.mkdir(parents=True, exist_ok=True)

    # carrega labels existentes (se houver), para pular videos ja anotados
    if labels_out.exists():
        try:
            with open(labels_out, "r", encoding="utf-8") as f:
                todas_labels = json.load(f) or {}
            print(Fore.BLUE + f"[INFO] Labels existentes carregados: {labels_out} ({len(todas_labels)} entradas)")
        except Exception as e:
            print(Fore.YELLOW + f"[AVISO] Falha ao carregar {labels_out}: {e}")
            todas_labels = {}
    else:
        todas_labels = {}

    json_files = listar_jsons(json_dir)
    if not json_files:
        print(Fore.YELLOW + f"[AVISO] Nenhum JSON em {json_dir}")
        return

    print(Fore.MAGENTA + f"[INFO] {len(json_files)} JSONs encontrados")

    for jpath in json_files:
        stem = jpath.stem
        vpath = encontrar_video_para_json(pred_dir, stem)

        try:
            records = carregar_pose_records(jpath)
        except Exception as e:
            print(Fore.RED + f"[ERRO] {jpath.name}: {e}")
            continue

        frames_index, id_counter = indexar_por_frame(records)

        if not vpath:
            print(Fore.RED + f"\n[VIDEO] {stem}   [STATUS: NAO ENCONTRADO]")
            print(Fore.YELLOW + "  Video nao encontrado")
            continue

        if stem in todas_labels:
            print(Fore.BLUE + f"\n[VIDEO] {stem}   [STATUS: ANOTADO]")
            print(Fore.WHITE + f"  Anotacao existente: {todas_labels.get(stem)}")
            continue

        print(Fore.GREEN + f"\n[VIDEO] {stem}   [STATUS: ENCONTRADO]")
        print(Fore.WHITE + "\n  Reproduzindo video (ESC para sair)...")
        exibir_video(vpath, title=f"Preview - {stem}")

        def _find_and_remove(glob_dir: Path, stem_pattern: str, description: str):
            removed = []
            for p in glob_dir.glob(stem_pattern):
                try:
                    p.unlink()
                    removed.append(p)
                except Exception as e:
                    print(Fore.YELLOW + f"  Falha ao apagar {p}: {e}")
            if removed:
                print(Fore.YELLOW + f"  {description} removidos:")
                for p in removed:
                    print(f"    {p}")
            else:
                print(Fore.YELLOW + f"  Nenhum {description} encontrado para remover.")

        while True:
            print(Fore.YELLOW + "\n  IDs encontrados:")
            for pid, cnt in id_counter.most_common():
                print(f"    ID {pid}: {cnt} frames")

            # Filtra IDs com frames suficientes
            candidatos = {pid: cnt for pid, cnt in id_counter.items() if cnt >= args.min_frames}
            if not candidatos:
                print(Fore.YELLOW + f"  Nenhum ID com {args.min_frames}+ frames")
                break

            # Pergunta ao usuario se deseja repetir o video antes de selecionar IDs
            while True:
                repetir = input("\nDeseja repetir o vídeo?\n  1 - Sim\n  2 - Não\n\nEscolha a opção: ").strip()
                if repetir == "1":
                    print(Fore.WHITE + "\n  Reproduzindo video (ESC para sair)...")
                    exibir_video(vpath, title=f"Preview - {stem}")
                    # Ao terminar a reprodução, pergunta novamente (permite repetir quantas vezes desejar)
                    continue
                elif repetir == "2" or repetir == "":
                    break
                else:
                    print(Fore.YELLOW + "  Opcao invalida. Digite 1 para Sim ou 2 para Não.")

            user_input = input(f"\n  IDs a manter (ex: 1,3) ou Enter para todos: ").strip()
            if user_input == "":
                kept_ids = list(sorted(candidatos.keys()))
            else:
                kept_ids = [int(x) for x in user_input.split(",") if x.strip().isdigit()]
                kept_ids = [pid for pid in kept_ids if pid in candidatos]

            if not kept_ids:
                print(Fore.YELLOW + "  Nenhum ID valido")
                break

            # Preview com bboxes (mantidos = verde, removidos = amarelo)
            records_filtrados_preview = limpar_json_mantendo_ids(records, kept_ids)
            present_ids = set(int(r.get("id_persistente", -1)) for r in records_filtrados_preview)
            removed_ids = set(candidatos.keys()) - present_ids
            exibir_video_com_bboxes(vpath, frames_index, present_ids=present_ids, removed_ids=removed_ids,
                                    title=f"Review - {stem}")

            print(Fore.YELLOW + "\n  IDs mantidos:")
            for pid in sorted(kept_ids):
                print(f"    ID {pid}: {candidatos[pid]} frames")

            # Classificacao entre mantidos
            video_labels = {str(pid): args.negative_class for pid in kept_ids}
            pos_in = input(f"\n  Escolha os IDs para anotar como {args.positive_class.upper()} entre os \"IDs mantidos\": ").strip()
            if pos_in:
                pos_ids = [int(x.strip()) for x in pos_in.split(",")
                           if x.strip().isdigit() and int(x.strip()) in kept_ids]
                for pid in pos_ids:
                    video_labels[str(pid)] = args.positive_class

            print(Fore.YELLOW + "\n  IDs anotados:")
            for pid in sorted(kept_ids):
                print(f"    ID {pid}: {video_labels[str(pid)]}")

            # Menu com Repetir
            print("\n  Deseja salvar essa anotacao no labels.json?\n")
            print("    1 - Confirmar")
            print("    2 - Cancelar")
            print("    3 - Repetir")
            choice = input("\n  Escolha uma opcao: ").strip()

            if choice == "1":
                total_antes = len(records)
                records_filtrados = limpar_json_mantendo_ids(records, kept_ids)
                salvar_json(jpath, records_filtrados)
                todas_labels[stem] = video_labels
                salvar_json(labels_out, todas_labels)
                print(Fore.GREEN + f"\n  Resumo: {total_antes} -> {len(records_filtrados)} registros")

                # Mostra as classes de cada ID que foram salvas em labels.json
                print(Fore.CYAN + "\n  labels:")
                for pid in sorted(video_labels.keys(), key=lambda x: int(x)):
                    print(f"    ID {pid}: {video_labels[pid]}")

                print(Fore.GREEN + f"\n  Anotacao para {stem} salva em {labels_out}")
                break

            elif choice == "2":
                videos_dir_candidates = [root / "videos", root.parent / "videos"]
                for vd in videos_dir_candidates:
                    if vd.exists():
                        _find_and_remove(vd, f"{stem}*.mp4", "video(s) na pasta videos")

                _find_and_remove(pred_dir, f"{stem}*.mp4", "video(s) em predicoes")

                try:
                    jpath.unlink()
                    print(Fore.YELLOW + f"  JSON removido: {jpath}")
                except Exception:
                    print(Fore.YELLOW + f"  JSON nao encontrado ou erro ao remover: {jpath}")
                tracking = jpath.with_name(stem + "_tracking.json")
                if tracking.exists():
                    try:
                        tracking.unlink()
                        print(Fore.YELLOW + f"  JSON tracking removido: {tracking}")
                    except Exception:
                        print(Fore.YELLOW + f"  Erro ao remover tracking: {tracking}")

                print(Fore.YELLOW + f"\n  Operacao cancelada e arquivos removidos para {stem}.")
                break

            elif choice == "3":
                print(Fore.BLUE + "\n  Repetindo analise do mesmo video...\n")
                continue

            else:
                print(Fore.YELLOW + "\n  Opcao invalida. Voltando ao menu do mesmo video.")
                continue

    salvar_json(labels_out, todas_labels)
    print(Fore.GREEN + f"\n[OK] Labels salvos: {labels_out}")


if __name__ == "__main__":
    main()
