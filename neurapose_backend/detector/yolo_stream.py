# neurapose_backend/detector/yolo_stream.py
# ==============================================================================
# WRAPPER CENTRALIZADO DE DETECÇÃO E RASTREAMENTO (YOLO + TRACKERS)
# ==============================================================================
# Este módulo encapsula a lógica de escolha do Tracker (BoTSORT/DeepOCSORT) e
# a inicialização do modelo YOLO, servindo como ponto único de entrada para
# detecção de pessoas no sistema.
# ==============================================================================

import numpy as np
import torch
from ultralytics import YOLO
from colorama import Fore

import neurapose_backend.config_master as cm
from neurapose_backend.tracker.rastreador import CustomBoTSORT, CustomDeepOCSORT, save_temp_tracker_yaml

# Módulos de Otimização
from neurapose_backend.otimizador.cuda.gpu_utils import gpu_manager
from neurapose_backend.otimizador.cpu import core as cpu_opt
from neurapose_backend.otimizador.ram import memory as ram_opt

class YoloDetectorPerson:
    """
    Wrapper unificado para detecção de pessoas e rastreamento.
    Gerencia automaticamente:
    1. Carga do Modelo YOLO (com path do config_master)
    2. Escolha e Instanciação do Tracker (BoTSORT ou DeepOCSORT)
    3. Lógica de inferência (predict -> update ou track direto)
    4. Otimização de Recursos (CPU/GPU/RAM)
    """

    def __init__(self, target_fps=cm.FPS_TARGET):
        """
        Inicializa o detector e o rastreador conforme configurações globais.
        
        Args:
            target_fps (int): FPS alvo para configuração inicial do Tracker.
                              Geralmente cm.FPS_TARGET (10).
        """
        self.target_fps = target_fps
        self.using_tracker = (cm.TRACKER_NAME.upper() == "DEEPOCSORT")
        
        self.model = None
        self.tracker = None
        self.tracker_instance = None
        
        self._init_system()

    def _init_system(self):
        print(Fore.CYAN + f"[DETECTOR] Inicializando pipeline de rastreamento: {cm.TRACKER_NAME}")
        
        if self.using_tracker:
            self.tracker = CustomDeepOCSORT()
            print(Fore.GREEN + "[DETECTOR] DeepOCSORT carregado com sucesso.")
        else:
            print(Fore.YELLOW + f"[DETECTOR] Carregando YOLO: {cm.YOLO_PATH}")
            try:
                self.model = YOLO(str(cm.YOLO_PATH), task='detect').to(cm.DEVICE)
                
                self.tracker_instance = CustomBoTSORT(frame_rate=int(self.target_fps))
                
                self.model.tracker = self.tracker_instance
                save_temp_tracker_yaml()
                
                print(Fore.GREEN + "[DETECTOR] YOLO + BoTSORT carregados com sucesso.")
            except Exception as e:
                print(Fore.RED + f"[ERRO] Falha fatal ao carregar YOLO/BoTSORT: {e}")
                raise e

    @gpu_manager.inference_mode()
    def process_frame(self, frame: np.ndarray, conf_threshold: float = None, frame_idx: int = None):
        """
        Processa um único frame e retorna as trilhas (tracks).
        
        Args:
            frame (np.ndarray): Imagem BGR (Opencv).
            conf_threshold (float, optional): Sobrescreve cm.DETECTION_CONF se fornecido.
            frame_idx (int, optional): Índice do frame para controle de GC e throttling.
            
        Returns:
            np.ndarray: Array de tracks no formato [x1, y1, x2, y2, id, conf, cls]
                        Retorna array vazio (0, 7) se nada for detectado.
        """
        if frame_idx is not None:
            cpu_opt.throttle()
            ram_opt.smart_cleanup(frame_idx)

        conf = conf_threshold if conf_threshold is not None else cm.DETECTION_CONF
        
        try:
            if self.using_tracker:
                tracks = self.tracker.track(frame)
                return tracks
            
            else:
                res = self.model.predict(
                    source=frame,
                    imgsz=cm.YOLO_IMGSZ,
                    conf=conf,
                    device=cm.DEVICE,
                    classes=[cm.YOLO_CLASS_PERSON],
                    verbose=False, 
                    stream=False
                )
                
                dets = np.empty((0, 6))
                
                if len(res) > 0 and len(res[0].boxes) > 0:
                    dets = res[0].boxes.data.cpu().numpy()
                    
                    if dets.shape[1] == 4:
                        r = dets.shape[0]
                        dets = np.hstack((dets, np.full((r, 1), 0.85), np.zeros((r, 1))))
                    elif dets.shape[1] == 5:
                        dets = np.hstack((dets, np.zeros((dets.shape[0], 1))))
                
                tracks = self.tracker_instance.update(dets, frame)
                
                if len(tracks) == 0:
                    return np.empty((0, 7))
                    
                return tracks

        except Exception as e:
            print(Fore.RED + f"[DETECTOR] Erro no processamento do frame: {e}")
            return np.empty((0, 7))

        try:
            results = []
            for i, frame in enumerate(frames):
                results.append(self.process_frame(frame, frame_idx=i))
            return results
        except Exception as e:
             print(Fore.RED + f"[DETECTOR] Erro no processamento de batch: {e}")
             return []

    def cleanup(self):
        """
        Libera recursos de GPU e RAM.
        Deve ser chamado ao finalizar o uso do detector.
        """
        print(Fore.CYAN + "[DETECTOR] Liberando recursos...")
        if self.model:
            del self.model
            self.model = None
            
        if self.tracker:
            del self.tracker
            self.tracker = None
            
        if self.tracker_instance:
             del self.tracker_instance
             self.tracker_instance = None

        gpu_manager.clear_cache()
        ram_opt.force_gc()
        print(Fore.GREEN + "[DETECTOR] Recursos liberados.")