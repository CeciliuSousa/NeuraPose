import numpy as np
from ultralytics import YOLO

from boxmot.trackers.deepocsort.deepocsort import DeepOcSort

import neurapose_backend.config_master as cm

try:
    from loguru import logger
    logger.remove()
except ImportError:
    pass

class CustomDeepOCSORT:
    def __init__(self):
        # print(f"[INIT] Carregando YOLO: {cm.YOLO_PATH}")
        self.model = YOLO(cm.YOLO_PATH, task='detect') 
        
        cfg = cm.DEEP_OC_SORT_CONFIG
        
        # print(f"[INIT] Carregando DeepOCSORT com ReID: {cfg['model_weights']}")

        device_arg = 0 if cm.DEVICE == 'cuda' else 'cpu'
        
        self.tracker = DeepOcSort(
            reid_weights=cm.OSNET_PATH,
            device=device_arg,
            half=cm.USE_FP16,
            
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
        
        try:
            results = self.model.predict(frame, conf=cm.DETECTION_CONF, verbose=False)
        except Exception as e:
            print(f"[DeepOCSORT] Erro no YOLO predict: {e}")
            return np.empty((0, 7))

        if not results:
            return np.empty((0, 7))

        try:
            if results[0].boxes is None or len(results[0].boxes) == 0:
                 try:
                     return self.tracker.update(np.empty((0, 6)), frame)
                 except:
                     return np.empty((0, 7))

            detecções = results[0].boxes.data.cpu().numpy()
            
            if len(detecções) > 0:
                detecções = detecções[detecções[:, 5] == 0]
            
            if len(detecções) == 0:
                 try:
                     return self.tracker.update(np.empty((0, 6)), frame)
                 except:
                     return np.empty((0, 7))

            tracks = self.tracker.update(detecções, frame)
            
        except IndexError as e:
            # print(f"[DeepOCSORT] Erro CRÍTICO de Índice (Tuple/List): {e}")
            return np.empty((0, 7))
            
        except Exception as e:
            print(f"[DeepOCSORT] Erro genérico no update: {e}")
            try:
                return self.tracker.update(np.empty((0, 6)), frame)
            except:
                return np.empty((0, 7))
        
        if len(tracks) > 0:
            tracks[:, :4] = np.round(tracks[:, :4])
            
        return tracks