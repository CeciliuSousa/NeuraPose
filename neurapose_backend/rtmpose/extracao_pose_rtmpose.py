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
from neurapose_backend.nucleo.visualizacao import desenhar_esqueleto_unificado, color_for_id

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
            sess = ort.InferenceSession(self.model_path, providers=providers)
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
        Processa um frame inteiro: para cada detecção YOLO, extrai a pose.
        
        Args:
            frame_img (np.array): Imagem original BGR.
            detections_yolo (list): Lista de detecções ou objeto Boxes do YOLO.
                                    Espera-se ter .xyxy, .conf e .id
            frame_idx (int): Indice do frame atual.
            id_map (dict): Mapa de re-identificação (opcional).
            desenhar_no_frame (bool): Se True, desenha o esqueleto no frame.
            
        Returns:
            list[dict]: Lista de registros processados (keypoints, bbox, ids...).
            np.array: Frame com desenho (se solicitado, senão retorna o original).
        """
        id_map = id_map or {}
        registros = []
        
        # Se não houver detecções, retorna vazio
        if detections_yolo is None:
            return registros, frame_img



        boxes = []
        confs = []
        track_ids = []

        is_numpy = isinstance(detections_yolo, np.ndarray)
        if is_numpy:
            # Assumindo formato [x1, y1, x2, y2, id, conf] do yolo_detector output "boxes"
            # O yolo_detector retorna boxes_data = [x1, y1, x2, y2, id, conf, ...]
            if detections_yolo.shape[1] >= 6:
                boxes = detections_yolo[:, :4]
                track_ids = detections_yolo[:, 4]
                confs = detections_yolo[:, 5]
            else:
                 # Formato inválido ou sem IDs
                 return registros, frame_img
        else:
            # Fallback para objeto YOLOv8 .boxes original (se usado em outro lugar)
            try:
                boxes = detections_yolo.xyxy.cpu().numpy()
                confs = detections_yolo.conf.cpu().numpy()
                if detections_yolo.id is None:
                    return registros, frame_img
                track_ids = detections_yolo.id.cpu().numpy()
            except AttributeError:
                # Se não for nem numpy nem objeto YOLO válido
                return registros, frame_img

        for box, conf, raw_tid in zip(boxes, confs, track_ids):
            # Resolve ID persistente via ReID map
            pid = int(id_map.get(int(raw_tid), int(raw_tid)))
            
            x1, y1, x2, y2 = map(int, box)
            
            # 1. Recorte Inteligente (Geometry)
            center, scale = calcular_centro_escala(x1, y1, x2, y2, (cm.SIMCC_W, cm.SIMCC_H))
            
            # Matriz de transformação
            trans = get_affine_transform(center, scale, 0, (cm.SIMCC_W, cm.SIMCC_H))
            
            # Crop do frame
            crop = cv2.warpAffine(frame_img, trans, (cm.SIMCC_W, cm.SIMCC_H), flags=cv2.INTER_LINEAR)
            
            # 2. Pré-processamento
            inp_tensor = self._preprocessar_input(crop)
            
            # 3. Inferência ONNX
            # O nome do output geralmente não precisa, run retorna lista na ordem
            outputs = self.sess.run(None, {self.input_name: inp_tensor})
            simcc_x, simcc_y = outputs[0], outputs[1]
            
            # 4. Decodificação (Crop Space)
            coords_crop, conf_arr = self._decodificar_simcc(simcc_x, simcc_y)
            
            # 5. Transformação Reversa (Original Space)
            coords_orig = transform_preds(coords_crop[0], center, scale, (cm.SIMCC_W, cm.SIMCC_H))
            
            # 6. Montagem dos Keypoints Finais (x, y, conf)
            kps_final = np.concatenate([coords_orig, conf_arr[0][:, None]], axis=1)
            
            # 7. Suavização Temporal (EMA)
            kps_suavizados = self.smoother.step(pid, kps_final)
            
            # 8. Visualização (Opcional)
            if desenhar_no_frame:
                base_color = color_for_id(pid)
                frame_img = desenhar_esqueleto_unificado(frame_img, kps_suavizados, kp_thresh=cm.POSE_CONF_MIN, base_color=base_color)
                # Opcional: Desenhar bbox e ID
                cv2.rectangle(frame_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_img, f"ID: {pid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # 9. Registro
            registros.append({
                "frame": frame_idx,
                "botsort_id": int(raw_tid),
                "id_persistente": pid,
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "keypoints": kps_suavizados.tolist()
            })
            
        return registros, frame_img
