# ==============================================================================
# neurapose_backend/pre_processamento/reid_manual.py
# ==============================================================================

"""
Ferramenta interativa para correção manual de re-identificação (id_persistente).
- Suporte a Intervalos de Frames.
- Navegação Frame a Frame (Setas) SOMENTE no modo PAUSE.
- Opção de Excluir IDs (Lixo/Cadeira).
- Opção de CORTAR (Trim) trechos do vídeo (ex: remover início/fim).
- Redesenho de vídeo limpo (Apenas ID_P e Confiança sobre fundo branco).
- Estrutura de pastas espelhada (-reid).
"""

import os
import sys
import json
import shutil
import cv2
import numpy as np
from pathlib import Path
from collections import Counter
from colorama import Fore, init as colorama_init

from neurapose_backend.config_master import (
    PROCESSING_OUTPUT_DIR,
    REID_MANUAL_SUFFIX,
    REID_MANUAL_LABELS_FILENAME,
)

from neurapose_backend.pre_processamento.anotando_classes import (
    listar_jsons,
    encontrar_video_para_json,
    carregar_pose_records,
    salvar_json,
)

# Visualizacao de esqueletos (utilitários prontos)
from neurapose_backend.pre_processamento.utils.visualizacao import desenhar_esqueleto, color_for_id
from neurapose_backend.pre_processamento.configuracao.config import POSE_CONF_MIN

# ==============================================================================
# 0. FUNÇÕES LÓGICAS DE CORTE E REGRAS
# ==============================================================================

def verificar_corte(frame_idx, cut_list):
    """Retorna True se o frame estiver em uma área de corte."""
    for cut in cut_list:
        if cut['start'] <= frame_idx <= cut['end']:
            return True
    return False

def aplicar_processamento_completo(records, rules_list, delete_list, cut_list):
    """
    Aplica na ordem:
    1. Cortes (Remove frames e realinha a contagem dos frames restantes).
    2. Exclusões de ID.
    3. Trocas de ID.
    """
    processed_records = []
    
    # Ordenar cortes para calcular o shift corretamente
    # Ex: Cortar 1-10 e 20-30
    cut_list.sort(key=lambda x: x['start'])

    # Mapa de tradução de Frame Antigo -> Frame Novo
    # Se frame foi cortado, map é None
    max_frame = 0
    if records:
        max_frame = max(int(r['frame']) for r in records) + 100 # Margem segura
    
    frame_map = {}
    current_shift = 0
    next_new_frame = 1
    
    # Criar mapa de frames frame-a-frame (método seguro)
    # Simula a leitura do vídeo
    for f in range(1, max_frame + 1):
        if verificar_corte(f, cut_list):
            frame_map[f] = None # Frame será deletado
        else:
            frame_map[f] = next_new_frame
            next_new_frame += 1

    changed_ids = 0
    deleted_ids = 0
    cut_records = 0

    for r in records:
        old_frame = int(r.get("frame", -1))
        pid = int(r.get("id_persistente", -1))
        
        # 1. Verifica Corte
        new_frame = frame_map.get(old_frame)
        if new_frame is None:
            cut_records += 1
            continue # Pula registro pois o frame foi cortado

        # Atualiza o frame para o novo índice sequencial
        r["frame"] = new_frame

        # 2. Verifica Exclusão de ID (usando frame original para lógica ou novo? 
        # Geralmente a regra foi criada olhando o vídeo original, então usamos old_frame na condição, 
        # mas cuidado: se o usuário define regra baseada no visual, é frame original)
        should_delete = False
        for d in delete_list:
            if pid == d['id'] and d['start'] <= old_frame <= d['end']:
                should_delete = True
                break
        
        if should_delete:
            deleted_ids += 1
            continue

        # 3. Verifica Troca de ID
        for rule in rules_list:
            if pid == rule['src'] and rule['start'] <= old_frame <= rule['end']:
                if pid != rule['tgt']:
                    r["id_persistente"] = rule['tgt']
                    changed_ids += 1
                break 
        
        processed_records.append(r)
                
    return processed_records, changed_ids, deleted_ids, cut_records

# ==============================================================================
# 1. FUNÇÃO DE REDESENHO LIMPO (Visual Profissional)
# ==============================================================================
def renderizar_video_limpo(video_in, video_out, registros_processados, cut_list):
    """
    Gera um vídeo novo.
    - Pula frames que estão na cut_list.
    - Usa o JSON já processado (onde frames foram renumerados para 1, 2, 3...).
    """
    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        print(Fore.RED + f"[ERRO] Falha ao abrir vídeo para renderização: {video_in}")
        return
    
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    writer = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, H)
    )

    # Indexa registros processados pelo NOVO número de frame
    frames_map = {}
    for r in registros_processados:
        f = int(r["frame"])
        if f not in frames_map:
            frames_map[f] = []
        frames_map[f].append(r)

    input_frame_idx = 1
    output_frame_idx = 1 # O vídeo de saída começa do 1 e é contínuo
    
    while True:
        ok, frame = cap.read()
        if not ok: break

        # Se este frame original está na lista de corte, pulamos
        if verificar_corte(input_frame_idx, cut_list):
            input_frame_idx += 1
            continue

        # Se o frame foi mantido, desenhamos as infos do JSON correspondente (output_frame_idx)
        if output_frame_idx in frames_map:
            for reg in frames_map[output_frame_idx]:
                pid = reg["id_persistente"]
                bbox = reg["bbox"] # [x1, y1, x2, y2]
                conf = reg.get("confidence", 0.0)
                
                x1, y1, x2, y2 = map(int, bbox)
                
                # Desenhar Esqueleto (se keypoints estiverem disponíveis)
                kps = reg.get("keypoints")
                if kps:
                    try:
                        kps_arr = np.array(kps)
                        base_color = color_for_id(pid)
                        frame = desenhar_esqueleto(frame, kps_arr, kp_thresh=POSE_CONF_MIN, base_color=base_color)
                    except Exception:
                        pass

                # BBox Verde (Espessura 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Texto formatado: ID_P e Confiança
                label = f"ID_P: {pid} | Pessoa: {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Fundo Branco do Texto
                cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 10, y1), (255, 255, 255), -1)
                
                # Texto Preto
                cv2.putText(frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        writer.write(frame)
        
        input_frame_idx += 1
        output_frame_idx += 1

    cap.release()
    writer.release()

# ==============================================================================
# 2. PLAYER INTERATIVO
# ==============================================================================
def exibir_video_interativo(video_path: Path, frames_index, title="Review", max_width=960, ids_para_apagar=None, cut_list=None):
    if video_path is None or not video_path.exists():
        print(Fore.YELLOW + f"[AVISO] Video nao encontrado: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = int(1000 / fps)

    playing = False 
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    if not ok: return

    # Teclas
    LEFT_KEYS = {81, 2424832, 65361, 75, ord('a')}
    RIGHT_KEYS = {83, 2555904, 65363, 77, ord('d')}
    
    # ids_para_apagar can be either a set (global IDs) or a list of dicts {'id','start','end'}
    if ids_para_apagar is None:
        ids_para_apagar = []
    if cut_list is None: cut_list = []

    while True:
        frame_show = frame.copy()
        h, w = frame_show.shape[:2]
        
        if w > max_width:
            s = max_width / w
            frame_show = cv2.resize(frame_show, (max_width, int(h * s)))
            scale = s
        else:
            scale = 1.0

        curr_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        display_idx = max(1, curr_pos)

        # Verifica se o frame atual está marcado para corte
        is_cut = verificar_corte(display_idx, cut_list)

        # Efeito visual para frame cortado (Escurecer + Texto X)
        if is_cut:
            overlay = frame_show.copy()
            cv2.rectangle(overlay, (0,0), (frame_show.shape[1], frame_show.shape[0]), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.7, frame_show, 0.3, 0, frame_show)
            cv2.putText(frame_show, "CORTADO", (int(w*scale)//2 - 100, int(h*scale)//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)

        # Desenhar BBoxes e IDs
        entries = frames_index.get(display_idx, [])
        for bbox, gid in entries:
            try:
                x1, y1 = int(bbox["x1"] * scale), int(bbox["y1"] * scale)
                x2, y2 = int(bbox["x2"] * scale), int(bbox["y2"] * scale)
                
                # Cores
                color = (0, 255, 0) # Verde (Normal)
                label = f"ID:{gid}"
                
                # Determina se este ID foi marcado para deletar NESTE FRAME
                is_deleted = False
                if isinstance(ids_para_apagar, set):
                    if gid in ids_para_apagar:
                        is_deleted = True
                else:
                    for d in ids_para_apagar:
                        try:
                            if gid == d['id'] and d['start'] <= display_idx <= d['end']:
                                is_deleted = True
                                break
                        except Exception:
                            continue

                if is_deleted:
                    color = (0, 0, 255) # Vermelho (Deletar ID)
                    label = f"DEL:{gid}"
                
                # Se frame está cortado, tudo fica cinza
                if is_cut: color = (100, 100, 100)

                cv2.rectangle(frame_show, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_show, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            except: pass

        # Infos na tela
        status_cut = " [AREA DE CORTE]" if is_cut else ""
        state_str = "PLAY" if playing else "PAUSE"
        info = f"{state_str}{status_cut} | FRAME: {display_idx} / {total_frames}"
        
        cv2.rectangle(frame_show, (0, 0), (max_width, 40), (0, 0, 0), -1)
        cv2.putText(frame_show, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow(title, frame_show)

        if playing:
            key = cv2.waitKey(delay_ms) & 0xFF
        else:
            key = cv2.waitKey(0) & 0xFF

        if key == 27 or key == ord('q'): # Sair
            break
        elif key == 32 or key == ord('p'): # Espaço: Toggle
            playing = not playing
            continue 
        
        if playing:
            ok, next_frame = cap.read()
            if ok: frame = next_frame
            else:
                playing = False
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                cap.read()
        else:
            if key in LEFT_KEYS:
                target = max(0, curr_pos - 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, next_frame = cap.read()
                if ok: frame = next_frame
            elif key in RIGHT_KEYS:
                if curr_pos < total_frames:
                    ok, next_frame = cap.read()
                    if ok: frame = next_frame

    cap.release()
    cv2.destroyWindow(title)

# ==============================================================================
# 3. FUNÇÕES AUXILIARES
# ==============================================================================
def indexar_por_frame_e_contar_ids(records):
    frames = {}
    id_counter = Counter()
    for r in records:
        gid = int(r.get("id_persistente", -1))
        if gid is None or gid < 0: continue
        bbox = r.get("bbox", {})
        if isinstance(bbox, list) and len(bbox) == 4:
            bbox = {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}
        frame_idx = int(r.get("frame", 0))
        frames.setdefault(frame_idx, []).append((bbox, gid))
        id_counter[gid] += 1
    return frames, id_counter

def parse_range(rng_str, max_val=999999):
    """Auxiliar para ler strings '1-35' ou enter vazio"""
    rng_str = rng_str.strip()
    if not rng_str: return 0, max_val
    if "-" in rng_str:
        try:
            s, e = map(int, rng_str.split("-"))
            return s, e
        except: return 0, max_val
    if rng_str.isdigit():
        return int(rng_str), int(rng_str) # Apenas um frame
    return 0, max_val

# ==============================================================================
# 4. MAIN
# ==============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="ReID manual: corrigir, limpar e recortar vídeos")
    parser.add_argument("--input-dir", dest="root", default=str(PROCESSING_OUTPUT_DIR))
    parser.add_argument("--output-dir", dest="out", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if args.out:
        out_root = Path(args.out).resolve()
    else:
        out_root = root.parent / (root.name + REID_MANUAL_SUFFIX)

    # Definição de pastas espelhadas
    json_dir, pred_dir, videos_dir = root / "jsons", root / "predicoes", root / "videos"
    out_json_dir, out_pred_dir, out_videos_dir = out_root / "jsons", out_root / "predicoes", out_root / "videos"

    for p in [out_json_dir, out_pred_dir, out_videos_dir]: p.mkdir(parents=True, exist_ok=True)
    labels_path = out_root / REID_MANUAL_LABELS_FILENAME

    print(Fore.BLUE + f"[CONFIG] Input: {root}")
    print(Fore.BLUE + f"[CONFIG] Output: {out_root}")

    json_files = listar_jsons(json_dir)
    print(Fore.MAGENTA + f"[INFO] {len(json_files)} JSONs encontrados")

    labels_reid = {}
    if labels_path.exists():
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                labels_reid = json.load(f) or {}
        except: pass

    for jpath in json_files:
        if "_tracking" in jpath.name: continue
        stem = jpath.stem
        out_json_path = out_json_dir / jpath.name
        
        if out_json_path.exists():
            print(Fore.BLUE + f"\n[VIDEO] {stem} [JA PROCESSADO]")
            continue

        v_pred = encontrar_video_para_json(pred_dir, stem)
        v_raw = videos_dir / f"{stem.replace('_pose', '')}.mp4"
        if not v_raw.exists(): v_raw = videos_dir / f"{stem}.mp4"
        
        # Player usa pred se tiver, ou raw
        v_player = v_pred if v_pred and v_pred.exists() else v_raw

        print(Fore.GREEN + f"\n[VIDEO] {stem} [ENCONTRADO]")
        records = carregar_pose_records(jpath)
        frames_index, id_counter = indexar_por_frame_e_contar_ids(records)

        # Listas de ações
        rules_list = []
        delete_list = []
        cut_list = []
        
        apagados_visual = set() 

        while True:
            exibir_video_interativo(
                v_player, 
                frames_index, 
                title=f"Editor: {stem}", 
                ids_para_apagar=delete_list,
                cut_list=cut_list
            )
            
            print(Fore.YELLOW + f"\n  IDs Presentes: {dict(id_counter.most_common(15))}")
            if apagados_visual:
                print(Fore.RED + f"  IDs Deletados (todos): {sorted(list(apagados_visual))}")
            if delete_list:
                print(Fore.RED + "  Exclusoes Agendadas:")
                for d in delete_list:
                    print(Fore.RED + f"    - ID {d['id']} : {d['start']}-{d['end']}")
            if cut_list: print(Fore.MAGENTA + f"  Cortes Agendados: {cut_list}")
            
            print("\n  O que deseja fazer?")
            print("  1 - Corrigir ID (Trocar)")
            print("  2 - Excluir ID (Lixo/Cadeira)")
            print("  3 - CORTAR TRECHO DE VÍDEO (Trim)")
            print("  4 - Rever vídeo")
            print("  5 - SALVAR e finalizar")
            print("  6 - Pular")
            
            opt = input("\n  Escolha: ").strip()

            if opt == "4": continue
            elif opt == "6": break

            elif opt == "1": # Troca
                try:
                    src = int(input("  ID ORIGINAL (Errado): "))
                    tgt = int(input("  ID NOVO (Correto): "))
                    rng = input("  Intervalo (Enter=Todo, ou 1-35): ").strip()
                    s, e = parse_range(rng)
                    rules_list.append({'src': src, 'tgt': tgt, 'start': s, 'end': e})
                    print(Fore.CYAN + "  Regra de troca agendada.")
                except: print(Fore.RED + "  Entrada invalida.")

            elif opt == "2": # Exclusão ID
                try:
                    did = int(input(Fore.RED + "  ID para EXCLUIR: "))
                    rng = input("  Intervalo (Enter=Todo, ou 1-35): ").strip()
                    s, e = parse_range(rng)
                    delete_list.append({'id': did, 'start': s, 'end': e})
                    # Marca visualmente como deletado apenas se o usuário deixou vazio (toda a duração)
                    if not rng:
                        apagados_visual.add(did)
                        print(Fore.RED + f"  ID {did} marcado para EXCLUSAO TOTAL.")
                    else:
                        print(Fore.RED + f"  ID {did} marcado para exclusão no intervalo {s}-{e}.")
                except: print(Fore.RED + "  Entrada invalida.")

            elif opt == "3": # Corte de Vídeo
                try:
                    print(Fore.MAGENTA + "  [CORTAR VÍDEO] O intervalo selecionado será REMOVIDO.")
                    rng = input("  Intervalo para REMOVER (ex: 1-35): ").strip()
                    if not rng: 
                        print("  Cancelado.")
                        continue
                    s, e = parse_range(rng, max_val=0) # max_val 0 força erro se vazio
                    if s > 0:
                        cut_list.append({'start': s, 'end': e})
                        print(Fore.MAGENTA + f"  Trecho {s}-{e} será removido do vídeo final.")
                    else:
                        print(Fore.RED + "  Intervalo inválido.")
                except: print(Fore.RED + "  Entrada invalida.")

            elif opt == "5": # Salvar
                # 1. Processar JSON (aplica cortes, delete e swap)
                recs_mod, c_ids, d_ids, d_cuts = aplicar_processamento_completo(records, rules_list, delete_list, cut_list)
                
                print(Fore.GREEN + f"  Resumo: {c_ids} Trocas, {d_ids} Exclusões ID, {d_cuts} Records cortados.")
                
                # 2. Salvar JSON novo
                salvar_json(out_json_path, recs_mod)
                
                # 3. Copiar vídeo bruto para pasta out (opcional, backup)
                # shutil.copy2(v_raw, out_videos_dir / v_raw.name)
                
                # 4. Renderizar vídeo final LIMPO e CORTADO
                out_v_pred = out_pred_dir / f"{stem}_pose.mp4"
                print(Fore.WHITE + "  Renderizando novo video limpo e cortado...")
                
                renderizar_video_limpo(v_raw, out_v_pred, recs_mod, cut_list)
                
                # Log
                from datetime import datetime
                labels_reid[stem] = {
                    "rules": rules_list, 
                    "deletions": delete_list, 
                    "cuts": cut_list,
                    "date": datetime.now().isoformat()
                }
                salvar_json(labels_path, labels_reid)
                break

    print(Fore.GREEN + f"\n[OK] Processamento manual finalizado.")

if __name__ == "__main__":
    colorama_init(autoreset=True)
    main()