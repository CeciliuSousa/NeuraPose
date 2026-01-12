# ==========================================================
# neurapose_backend/pre_processamento/anotando_classes_temp.py
# ==========================================================

import os
import cv2
import json
from pathlib import Path
from collections import Counter, defaultdict
from colorama import Fore, init as colorama_init
import argparse
import numpy as np

# Tenta usar Tkinter para obter o tamanho da tela para posicionamento
try:
    import tkinter as tk
    root = tk.Tk()
    SCREEN_W = root.winfo_screenwidth()
    SCREEN_H = root.winfo_screenheight()
    root.withdraw() 
except:
    # Valores padrão de fallback se tkinter não funcionar ou não for usado
    SCREEN_W = 1920 
    SCREEN_H = 1080

colorama_init(autoreset=True)

# Configuracoes centralizadas (usar mesmos defaults que em anotando_classes.py)
from config_master import (
    PROCESSING_OUTPUT_DIR,
    PROCESSING_ANNOTATIONS_DIR,
    MIN_FRAMES_PER_ID,
    CLASSE1,
    CLASSE2,
)


# ==========================================================
# 1. FUNÇÕES DE SUPORTE
# ==========================================================

def carregar_json(path: Path):
    """Carrega dados JSON de um arquivo."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        print(Fore.RED + f"❌ Erro ao decodificar JSON em {path.name}. Retornando vazio.")
        return {}

def salvar_json(path: Path, data):
    """Salva dados em JSON com garantia de escrita."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def indexar_por_frame(records):
    """Conta a ocorrência de IDs persistentes."""
    id_counter = Counter()
    for r in records:
        gid = r.get("id_persistente", -1)
        if gid is not None and gid >= 0:
            id_counter[int(gid)] += 1
    return id_counter


def indexar_por_frame_records(records):
    """Indexa registros por frame e conta ocorrencias por ID (para desenhar bboxes)."""
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


def filtrar_registros_por_ids(records, kept_ids):
    """Retorna somente os registros cujos ids persistentes estão em kept_ids."""
    kept = set(int(i) for i in kept_ids)
    return [r for r in records if int(r.get("id_persistente", -1)) in kept]


def exibir_video_com_bboxes(video_path: Path, frames_index, present_ids:set=None, removed_ids:set=None, kept_color=(0,255,0), removed_color=(0,215,255), title="Review"):
    """Exibe video com bounding boxes sobrepostos.
    present_ids aparecem em verde; removed_ids em amarelo.
    Não desenha legendas extras na imagem (somente bboxes).
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

            # Apenas desenhar para IDs que sejam mantidos ou removidos (candidatos)
            if gid not in present_ids and gid not in removed_ids:
                continue

            color = kept_color if gid in present_ids else removed_color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.imshow(title, frame)
        if cv2.waitKey(15) & 0xFF == 27:
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

def encontrar_video_para_json(pred_dir: Path, stem: str):
    """Localiza o arquivo de vídeo associado ao JSON."""
    candidatos = [
        pred_dir / f"{stem}_pose.mp4",
        pred_dir / f"{stem}.mp4",
    ]
    for c in candidatos:
        if c.exists():
            return c
    return None

# ==========================================================
# 2. FUNÇÃO DE STATUS
# ==========================================================

def desenhar_status_com_fundo(frame, status_data, window_w):
    """Desenha o status do anotador com fundo preto e todos os campos solicitados."""
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    line_height = 30
    padding = 10
    
    # Linhas de Status Solicitadas
    status_lines = [
        f"Frame: {status_data['frame']}/{status_data['total_frames']} [{status_data['mode']}]",
        f"ID Foco: {status_data['target_id']}",
        f"Inicio Furto: {status_data['start_frame'] if status_data['start_frame'] is not None else 'N/A'}",
        f"Fim Furto: {status_data['end_frame'] if status_data['end_frame'] is not None else 'N/A'}",
        f"Comandos: (ESPAÇO) Play/Pause | (Q) CONFIRMAR | (R) Resetar"
    ]
    
    color_text = (255, 255, 255)  # Branco
    color_bg = (0, 0, 0)          # Preto
    
    # Desenho do Fundo
    bg_height = len(status_lines) * line_height + padding
    cv2.rectangle(frame, (0, 0), (window_w, bg_height), color_bg, -1)
    
    # Desenho do Texto
    is_ready_to_confirm = status_data['start_frame'] is not None and status_data['end_frame'] is not None

    for i, line in enumerate(status_lines):
        y = (i + 1) * line_height - padding
        
        # Cor de destaque verde se pronto para confirmar
        if i == 4 and is_ready_to_confirm:
             color_line = (0, 255, 0) 
        else:
            color_line = color_text
            
        cv2.putText(frame, line, (padding, y), font, font_scale, color_line, font_thickness)
    
    return frame

# ==========================================================
# 3. PLAYER FLUIDO E ANOTAÇÃO DE TEMPO (CORRIGIDO)
# ==========================================================

def mark_action_frames(video_path: Path, video_size, total_frames: int, target_id: int):
    """
    Player fluido para marcação de INÍCIO/FIM do evento de ação para um ID específico.
    A janela é redimensionada para metade da tela e movida para a direita.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(Fore.RED + "Erro: Não foi possível abrir o vídeo.")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay_ms = int(1000 / fps) if fps > 0 else 30
    
    # 1. CÁLCULOS DE DIMENSÃO E POSICIONAMENTO (Metade da tela, à direita)
    target_w = SCREEN_W // 2
    original_w, original_h = video_size
    ratio = original_h / original_w
    target_h = int(target_w * ratio)
    
    # Posição: Meio da tela em X, topo da tela em Y
    pos_x = target_w  
    pos_y = 0         
    
    current_frame = 0
    start_frame = None
    end_frame = None
    playing = False # Começa PAUSADO

    window_title = f"ANOTAR FURTO - ID: {target_id} - {video_path.stem}"
    
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, target_w, target_h)
    cv2.moveWindow(window_title, pos_x, pos_y) # Mover para a direita

    print(Fore.YELLOW + f"\n--- MARCANDO AÇÃO PARA ID {target_id} ---")
    print("===COMANDOS===\n\n"
    "(ESPAÇO) Play/Pause\n"
    "(I) Marcar INÍCIO\n"
    "(O) Marcar FIM\n"
    "(A/D) Frame -/+1\n"
    "(W/S) Frame -/+10\n"
    "(R) Resetar\n"
    "(Q) Confirmar/Salvar\n")

    while True:
        # 2. Leitura e Redimensionamento do Frame
        if playing:
            ret, frame = cap.read()
            if not ret:
                current_frame = total_frames - 1
                playing = False
                continue
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret: 
                break
        
        # Redimensiona o frame para o tamanho da janela (metade da tela)
        frame_resized = cv2.resize(frame, (target_w, target_h))
        
        # 3. Desenho do Status
        status_data = {
            'frame': current_frame,
            'total_frames': total_frames,
            'mode': "REPRODUZINDO" if playing else "PAUSADO",
            'target_id': target_id,
            'start_frame': start_frame,
            'end_frame': end_frame
        }
        frame_final = desenhar_status_com_fundo(frame_resized, status_data, target_w)

        cv2.imshow(window_title, frame_final)

        # 4. Captura do Key
        wait_time = delay_ms if playing else 1 
        key = cv2.waitKey(wait_time) & 0xFF 

        # 5. Lógica de Controle
        
        # CONFIRMAR E SAIR (só se I e O estiverem marcados)
        if key == ord('q'):
            if start_frame is not None and end_frame is not None:
                break
            else:
                print(Fore.RED + "❌ Erro: INÍCIO e FIM devem ser marcados antes de confirmar (Q).")
        
        # Play/Pause
        elif key == ord(' '): 
            playing = not playing
            print(Fore.CYAN + f"-> Modo: {'Reproduzir' if playing else 'Pausar'}")
            
        # Marcar INÍCIO (Só permite marcar se for o primeiro)
        elif key == ord('i'):
             if start_frame is None:
                start_frame = current_frame
                print(Fore.GREEN + f"-> INÍCIO marcado: {start_frame}")
             else:
                print(Fore.RED + f"-> INÍCIO já marcado em {start_frame}. Use (R) para resetar.")
            
        # Marcar FIM (Só permite marcar se for o primeiro)
        elif key == ord('o'):
            if end_frame is None:
                end_frame = current_frame
                print(Fore.GREEN + f"-> FIM marcado: {end_frame}")
            else:
                print(Fore.RED + f"-> FIM já marcado em {end_frame}. Use (R) para resetar.")
            
        elif key == ord('r'): # Resetar
            start_frame, end_frame = None, None
            print(Fore.YELLOW + "-> Marcação RESETADA.")
        
        # Navegação (Funciona pausando automaticamente)
        elif key in [ord('s'), ord('w'), ord('d'), ord('a')]:
            playing = False # Garante que está pausado ao navegar
            
            if key == ord('s'): current_frame = min(current_frame + 10, total_frames - 1)
            elif key == ord('w'): current_frame = max(0, current_frame - 10)
            elif key == ord('d'): current_frame = min(current_frame + 1, total_frames - 1)
            elif key == ord('a'): current_frame = max(0, current_frame - 1)

            # Forçar a leitura do novo frame
            current_frame = min(max(0, current_frame), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            
    cap.release()
    cv2.destroyAllWindows()
    
    # Certificar ordem lógica
    if start_frame is not None and end_frame is not None and start_frame > end_frame:
        start_frame, end_frame = end_frame, start_frame
    
    return start_frame, end_frame


# ==========================================================
# 4. FLUXO PRINCIPAL DO ANOTADOR
# ==========================================================

def main():
    parser = argparse.ArgumentParser(description="Anotador Multi-ID (Tempo + Classe)")
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
    labels_out = Path(args.labels_out)

    if not pred_dir.exists() or not json_dir.exists():
        raise FileNotFoundError(f"As pastas esperadas não foram encontradas:\n - {pred_dir}\n - {json_dir}")

    todas_anotacoes = carregar_json(labels_out)
    json_files = sorted([p for p in json_dir.glob("*.json") if p.is_file() and not p.name.endswith("_tracking.json")])

    for jpath in json_files:
        stem = jpath.stem
        vpath = encontrar_video_para_json(pred_dir, stem)

        if stem in todas_anotacoes and todas_anotacoes[stem].get('status') == 'COMPLETO':
            print(Fore.CYAN + f"\n⏩ Pulando {stem}: já marcado como COMPLETO.")
            continue
        
        # --- Obtém o tamanho e total de frames do vídeo ---
        cap_temp = cv2.VideoCapture(str(vpath))
        if not cap_temp.isOpened(): continue
        video_w = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_temp.release()
        video_size = (video_w, video_h)
        
        try:
            records = carregar_json(jpath)
        except Exception:
            continue

        if not vpath or not records:
            continue

        # Indexa registros por frame (usado para desenhar bboxes no replay) e conta ids
        frames_index, id_counter = indexar_por_frame_records(records)
        candidatos = {pid: cnt for pid, cnt in id_counter.items() if cnt >= args.min_frames}
        
        if not candidatos:
            print(Fore.YELLOW + f"Nenhum ID com pelo menos {args.min_frames} frames para {stem}.")
            continue

        print(Fore.CYAN + f"\n\n=======================================================")
        print(Fore.CYAN + f"🎬 Iniciando: {stem}")
        print(Fore.YELLOW + "IDs persistentes encontrados:")
        for pid, cnt in id_counter.most_common():
            print(f" - ID {pid}: {cnt} frames")
        
        # Exibe o vídeo (Pré-visualização)
        print(Fore.WHITE + "\n▶️ Reproduzindo vídeo (ESC para sair)...")
        cap_preview = cv2.VideoCapture(str(vpath))
        
        # Pre-cálculo do tamanho da preview (metade da tela, à esquerda)
        preview_w = SCREEN_W // 2
        ratio = video_h / video_w
        preview_h = int(preview_w * ratio)
        
        cv2.namedWindow(f"Preview - {stem}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"Preview - {stem}", preview_w, preview_h)
        cv2.moveWindow(f"Preview - {stem}", 0, 0) # Posiciona à esquerda
        
        while True:
            ok, frame = cap_preview.read()
            if not ok: break
            frame_resized = cv2.resize(frame, (preview_w, preview_h))
            cv2.imshow(f"Preview - {stem}", frame_resized)
            if cv2.waitKey(15) & 0xFF == 27: break 
        cap_preview.release()
        cv2.destroyAllWindows()
        
        # --- Lógica Interativa (IDs) ---
        kept_ids_input = input(f"\nIDs a manter (ex.: 1,2,3): ").strip()
        kept_ids = [int(x) for x in kept_ids_input.split(",") if x.strip().isdigit() and int(x) in candidatos]
        
        if not kept_ids:
             print(Fore.RED + "Nenhum ID válido selecionado. Pulando vídeo.")
             continue
             
        # Repetir vídeo mostrando mantidos em verde e removidos em amarelo
        records_filtrados_preview = filtrar_registros_por_ids(records, kept_ids)
        present_ids = set(int(r.get("id_persistente", -1)) for r in records_filtrados_preview)
        removed_ids = set(candidatos.keys()) - present_ids
        exibir_video_com_bboxes(vpath, frames_index, present_ids=present_ids, removed_ids=removed_ids, title=f"Review - {stem}")

        furto_ids_input = input(Fore.MAGENTA + f"\nIDs que são '{args.positive_class}' (entre {sorted(kept_ids)}): ").strip()
        furto_ids = [int(x) for x in furto_ids_input.split(",") if x.strip().isdigit() and int(x) in kept_ids]


        # PASSO 2: MARCAÇÃO DE TEMPO POR ID
        video_anotacoes = {
            "Total_frames": total_frames,
            "ids_mantidos": kept_ids,
            "acoes": []
        }
        
        # 2a. Anotar IDs de FURTO
        for target_id in furto_ids:
            # Chama a função de anotação com o tamanho correto do vídeo
            start, end = mark_action_frames(vpath, video_size, total_frames, target_id)
            
            if start is not None and end is not None:
                video_anotacoes['acoes'].append({
                    "id": target_id,
                    "classe": args.positive_class,
                    "inicio": start,
                    "fim": end
                })
            else:
                 print(Fore.RED + f"❌ Marcação de tempo para ID {target_id} foi ignorada ou incompleta.")
        
        # 2b. Anotar IDs NORMAIS (FUNDO)
        normal_ids = [pid for pid in kept_ids if pid not in furto_ids]
        
        for target_id in normal_ids:
             video_anotacoes['acoes'].append({
                "id": target_id,
                "classe": args.negative_class,
                "inicio": 0,
                "fim": total_frames
            })

        # Finalização e Salvamento
        if not furto_ids and normal_ids:
            video_anotacoes['status'] = 'COMPLETO (SOMENTE NORMAL)'
        elif video_anotacoes['acoes']:
             video_anotacoes['status'] = 'COMPLETO'
        else:
            video_anotacoes['status'] = 'INCOMPLETO'

        # Pergunta para salvar a anotacao desse video de forma imediata
        print("\n  Deseja salvar essa anotacao no labels.json?\n")
        print("    1 - Confirmar")
        print("    2 - Cancelar")
        choice = input("\n  Escolha uma opcao: ").strip()
        if choice == "1":
            total_antes = len(records)
            records_filtrados = filtrar_registros_por_ids(records, kept_ids)
            salvar_json(jpath, records_filtrados)

            todas_anotacoes[stem] = video_anotacoes
            salvar_json(labels_out, todas_anotacoes)

            print(Fore.GREEN + f"\n  Resumo: {total_antes} -> {len(records_filtrados)} registros")
            print(Fore.CYAN + "\n  labels:")
            # mostra as classes por ID
            labels_map = {str(a['id']): a['classe'] for a in video_anotacoes.get('acoes', [])}
            for pid in sorted(labels_map.keys(), key=lambda x: int(x)):
                print(f"    ID {pid}: {labels_map[pid]}")

            print(Fore.GREEN + f"\n  Anotacao para {stem} salva em {labels_out}")
        else:
            print(Fore.YELLOW + f"\n  Operacao cancelada. Nenhuma alteracao salva para {stem}")

    print(Fore.CYAN + "\n=======================================================")
    print(Fore.GREEN + f"✅ PROCESSO DE ANOTAÇÃO CONCLUÍDO.")
    print(Fore.CYAN + "=======================================================")

if __name__ == "__main__":
    main()