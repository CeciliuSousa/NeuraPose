import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

# Importação direta e robusta da classe real
from boxmot.trackers.deepocsort.deepocsort import DeepOcSort

# Importa suas configs
import neurapose_backend.config_master as cm

class CustomDeepOCSORT:  # Mudei o nome para não conflitar
    def __init__(self):
        # 1. Carrega Detector (YOLO)
        # # Usa o path do config para garantir que pegou o modelo certo
        # print(f"[INIT] Carregando YOLO: {cm.YOLO_PATH}")
        self.model = YOLO(cm.YOLO_PATH) 
        
        # 2. Carrega Tracker (DeepOCSORT)
        # Mapeia as configs do seu dicionário para os argumentos da classe
        cfg = cm.DEEP_OC_SORT_CONFIG  # <--- Certifique-se de ter criado esse dict no config_master
        
        # print(f"[INIT] Carregando DeepOCSORT com ReID: {cfg['model_weights']}")
        
        # BoxMOT espera device index (0) ou 'cpu', nao 'cuda' string generic
        device_arg = 0 if cm.DEVICE == 'cuda' else 'cpu'
        
        self.tracker = DeepOcSort(
            reid_weights=cm.OSNET_PATH, # Peso do OSNet
            device=device_arg,
            half=cm.USE_FP16,
            
            # Parâmetros Anti-Oclusão (Vêm do seu config)
            det_thresh=cfg['det_thresh'],
            max_age=cfg['max_age'],
            min_hits=cfg['min_hits'],
            iou_thresh=cfg['iou_thresh'],
            delta_t=cfg['delta_t'],
            asso_func=cfg['asso_func'],
            inertia=cfg['inertia'],
            w_association_emb=cfg['w_association_emb'],
        )

    def track(self, frame):
        """
        Recebe um frame (imagem cv2) e retorna os tracks [x1, y1, x2, y2, id, conf, cls, ...]
        """
        
        # 1. Inferência YOLO (Rápida)
        # verbose=False limpa o terminal
        # conf=0.35 para pegar pessoas meio escondidas
        results = self.model.predict(frame, conf=cm.DETECTION_CONF, verbose=False)
        
        # 2. Prepara dados para o Tracker
        # O BoxMOT exige um array Numpy no formato: [x1, y1, x2, y2, conf, cls]
        detecções = results[0].boxes.data.cpu().numpy()
        
        # FILTRO DE CLASSE: MANTER APENAS PESSOAS (Classe 0)
        # Assumindo que YOLO_CLASS_PERSON é 0
        if len(detecções) > 0:
            detecções = detecções[detecções[:, 5] == 0]
        
        # Se não detectou ninguém, retorna vazio para não quebrar o tracker e limpar memória
        if len(detecções) == 0:
             try:
                 # Passa array vazio com shape correto (0, 6)
                 return self.tracker.update(np.empty((0, 6)), frame)
             except Exception:
                 return np.empty((0, 7))

        # 3. Atualiza o DeepOCSORT
        # Ele faz a mágica de ReID + Kalman Filter aqui
        try:
            tracks = self.tracker.update(detecções, frame)
        except IndexError:
            # Captura erro de "axis 0 with size 1" do Kalman Filter (xysr_kf.py)
            # Geralmente acontece em oclusão total ou inicialização ruim
            # Retorna vazio neste frame para evitar crash
            return np.empty((0, 7))
        except Exception as e:
            print(f"[DeepOCSORT] Erro inesperado no update: {e}")
            return np.empty((0, 7))
        
        # O tracks retorna algo como: [[x1, y1, x2, y2, id, conf, cls, ...]]
        # Vamos garantir que as coordenadas sejam inteiros (Otimização Regra 3)
        if len(tracks) > 0:
            # Arredonda apenas as 4 primeiras colunas (coords)
            tracks[:, :4] = np.round(tracks[:, :4])
            
        return tracks