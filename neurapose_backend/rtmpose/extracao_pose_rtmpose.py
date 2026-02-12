# ==============================================================
# neurapose_backend/rtmpose/extracao_pose_rtmpose.py
# ==============================================================
# Módulo unificado para extração de pose usando RTMPose (Modelos Top-Down).
# Substitui implementações duplicadas em 'modulos/rtmpose.py' e 'modulos/extracao_pose.py'.
# ==============================================================

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from colorama import Fore

import neurapose_backend.config_master as cm
from neurapose_backend.nucleo.geometria import (
    calcular_centro_escala, 
    get_affine_transform, 
    transform_preds
)
from neurapose_backend.nucleo.suavizacao import EmaSmoother
from neurapose_backend.nucleo.visualizacao import desenhar_esqueleto, color_for_id

class ExtratorPoseRTMPose:
    def __init__(self, model_path=None, device='cuda'):
        """
        Inicializa o extrator RTMPose.
        
        Args:
            model_path (str/Path): Caminho para o arquivo .onnx. Se None, usa do config_master.
            device (str): 'cuda' ou 'cpu'.
        """
        self.model_path = str(model_path) if model_path else str(cm.RTMPOSE_PATH)
        self.device = device
        
        # Carrega Sessão ONNX
        self.sess, self.input_name = self._carregar_sessao()
        
        # Inicializa suavizador
        self.smoother = EmaSmoother()
        
        # print(Fore.CYAN + f"[RTMPOSE] Inicializado. Modelo: {Path(self.model_path).name}")

    def _carregar_sessao(self):
        """Configura e carrega a sessão ONNXRuntime."""
        providers = []
        if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
            
        try:
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            
            sess = ort.InferenceSession(self.model_path, providers=providers, sess_options=sess_options)
            input_name = sess.get_inputs()[0].name
            return sess, input_name
        except Exception as e:
            print(Fore.RED + f"[ERRO_CRITICO] Falha ao carregar RTMPose: {e}")
            raise e

    def _preprocessar_input(self, bgr_crop):
        """
        Prepara recorte da imagem para o modelo (Resize -> BGR2RGB -> Norm -> NCHW).
        """
        # 1. Resize
        resized = cv2.resize(bgr_crop, (cm.SIMCC_W, cm.SIMCC_H))
        
        # 2. BGR -> RGB e Float32
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # 3. Normalização (ImageNet Mean/Std)
        rgb = (rgb - cm.MEAN) / cm.STD
        
        # 4. Transpose HWC -> CHW e Add Batch Dim -> NCHW
        chw = rgb.transpose(2, 0, 1)[None] 
        
        return chw.astype(np.float32)

    def _decodificar_simcc(self, simcc_x, simcc_y):
        """
        Decodifica saída SimCC para coordenadas no espaço do crop (192, 256).
        """
        # Argmax nos eixos de classificação
        x_idx = np.argmax(simcc_x, axis=2)
        y_idx = np.argmax(simcc_y, axis=2)

        # Confiança baseada no pico máximo
        conf = np.sqrt(np.max(simcc_x, axis=2) * np.max(simcc_y, axis=2)).astype(np.float32)

        # Rescale usando o fator de split (geralmente 2.0 para SimCC)
        x = (x_idx.astype(np.float32) / cm.SIMCC_SPLIT_RATIO)
        y = (y_idx.astype(np.float32) / cm.SIMCC_SPLIT_RATIO)

        coords = np.stack([x, y], axis=2).astype(np.float32)
        return coords, conf

    def processar_frame(self, frame_img, detections_yolo, frame_idx, id_map=None, desenhar_no_frame=False):
        """
        Processa um frame inteiro: para cada detecção YOLO, extrai a pose em BATCH.
        
        Args:
            frame_img (np.array): Imagem original BGR.
            detections_yolo (list): Lista de detecções ou objeto Boxes do YOLO.
            frame_idx (int): Indice do frame atual.
            id_map (dict): Mapa de re-identificação (opcional).
            desenhar_no_frame (bool): Se True, desenha o esqueleto no frame.
            
        Returns:
            list[dict]: Lista de registros processados (keypoints, bbox, ids...).
            np.array: Frame com desenho (se solicitado, senão retorna o original).
        """
        id_map = id_map or {}
        registros = []
        
        # Validacao de Detecções (Igual ao original)
        if detections_yolo is None: return registros, frame_img

        boxes_list = []
        confs_list = []
        track_ids_list = []

        is_numpy = isinstance(detections_yolo, np.ndarray)
        if is_numpy:
            # Proteção contra arrays 1D ou vazios (IndexError no shape[1])
            if detections_yolo.ndim < 2:
                if detections_yolo.size == 0: 
                    return registros, frame_img
                else: 
                     # Tenta recuperar se for (N,) expandindo dims? Não, assumir erro.
                     print(Fore.YELLOW + f"[RTMPOSE] WARN: Array 1D ignorado: {detections_yolo.shape}")
                     return registros, frame_img

            if detections_yolo.shape[1] >= 6:
                boxes_list = detections_yolo[:, :4]
                track_ids_list = detections_yolo[:, 4]
                confs_list = detections_yolo[:, 5]
            else: return registros, frame_img
        else:
            try:
                boxes_list = detections_yolo.xyxy.cpu().numpy()
                confs_list = detections_yolo.conf.cpu().numpy()
                if detections_yolo.id is None: return registros, frame_img
                track_ids_list = detections_yolo.id.cpu().numpy()
            except AttributeError: return registros, frame_img

        if len(boxes_list) == 0: return registros, frame_img

        # -----------------------------------------------------------
        # 1. PREPARAÇÃO (Crops) - Coleta de dados O(N) na CPU
        # -----------------------------------------------------------
        valid_crops = []
        valid_metadata = [] # (box, conf, id_persistente, center, scale)

        for box, conf, raw_tid in zip(boxes_list, confs_list, track_ids_list):
            pid = int(id_map.get(int(raw_tid), int(raw_tid)))
            x1, y1, x2, y2 = map(int, box)

            # Recorte Inteligente
            center, scale = calcular_centro_escala(x1, y1, x2, y2, (cm.SIMCC_W, cm.SIMCC_H))
            trans = get_affine_transform(center, scale, 0, (cm.SIMCC_W, cm.SIMCC_H))
            crop = cv2.warpAffine(frame_img, trans, (cm.SIMCC_W, cm.SIMCC_H), flags=cv2.INTER_LINEAR)
            
            # Pre-processamento individual do crop (Resize+Norm -> CHW)
            # Otimização: _preprocessar_input poderia ser vetorizado, mas o resize/warp é cv2.
            # Vamos usar _preprocessar_input que retorna (1, C, H, W) e concatenar depois.
            inp_tensor = self._preprocessar_input(crop) 
            
            valid_crops.append(inp_tensor)
            valid_metadata.append({
                "pid": pid,
                "box": [x1, y1, x2, y2],
                "conf": float(conf),
                "raw_tid": int(raw_tid),
                "center": center,
                "scale": scale
            })

        if not valid_crops: return registros, frame_img

        # -----------------------------------------------------------
        # 2. INFERÊNCIA EM BATCH
        # -----------------------------------------------------------
        # Concatena todos os crops em um batch tensor: (N, 3, 256, 192)
        full_batch = np.concatenate(valid_crops, axis=0)
        total_samples = len(full_batch)
        
        # Split em mini-batches se exceder MAX_BATCH (segurança de VRAM)
        max_bs = getattr(cm, 'RTMPOSE_MAX_BATCH_SIZE', 10) 
        
        simcc_x_all = []
        simcc_y_all = []

        for i in range(0, total_samples, max_bs):
            batch_chunk = full_batch[i : i + max_bs]
            
            # Inferência ONNX (Sess.run) - O(1) pesado na GPU
            outputs = self.sess.run(None, {self.input_name: batch_chunk})
            simcc_x_all.append(outputs[0])
            simcc_y_all.append(outputs[1])

        # Reconstrói arrays completos (output do modelo para N pessoas)
        simcc_x_full = np.concatenate(simcc_x_all, axis=0) # (N, K, W*2)
        simcc_y_full = np.concatenate(simcc_y_all, axis=0) # (N, K, H*2)

        # -----------------------------------------------------------
        # 3. DECODIFICAÇÃO E POS-PROCESSAMENTO
        # -----------------------------------------------------------
        # Decodifica SimCC (Argmax vetorizado pelo numpy)
        # _decodificar_simcc lida com (Batch, K, Valls)
        coords_batch, confs_batch = self._decodificar_simcc(simcc_x_full, simcc_y_full) 

        # Itera resultados para transformar de volta e desenhar
        for i, meta in enumerate(valid_metadata):
            coords_crop = coords_batch[i] # (K, 2)
            kp_conf = confs_batch[i]      # (K, 1)

            # Transformação Reversa
            coords_orig = transform_preds(coords_crop, meta["center"], meta["scale"], (cm.SIMCC_W, cm.SIMCC_H))
            
            # Montagem 
            # --- CORREÇÃO DE SHAPE: Garante (K, 1) ---
            if kp_conf.ndim == 1:
                kp_conf = kp_conf[:, None]

            kps_final = np.concatenate([coords_orig, kp_conf], axis=1) # (K, 3)

            # Suavização
            kps_suavizados = self.smoother.step(meta["pid"], kps_final)
            
            # [OTIMIZAÇÃO] Arredondamento Vetorizado antes de converter para lista
            # Muito mais rápido que iterar depois
            kps_rounded = np.round(kps_suavizados, 2).tolist()
            
            # Registro
            registros.append({
                "frame": frame_idx,
                f"{cm.TRACKER_NAME}_id": meta["raw_tid"],
                "id_persistente": meta["pid"],
                "bbox": [round(x, 2) for x in meta["box"]],
                "confidence": round(float(meta["conf"]), 2),
                "keypoints": kps_rounded
            })

            # Desenho (Opcional)
            if desenhar_no_frame:
                base_color = color_for_id(meta["pid"])
                frame_img = desenhar_esqueleto(frame_img, kps_suavizados, kp_thresh=cm.POSE_CONF_MIN, base_color=base_color)
                # Opcional: Desenhar bbox e ID
                x1, y1, x2, y2 = meta["box"]
                cv2.rectangle(frame_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_img, f"ID: {meta['pid']}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        return registros, frame_img
