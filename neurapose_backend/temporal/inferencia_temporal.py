import torch
import numpy as np
from collections import deque, defaultdict
import neurapose_backend.config_master as cm

class ClassificadorAcao:
    def __init__(self, model_path, window_size=30, num_joints=17):
        self.device = cm.DEVICE
        self.window_size = window_size
        self.input_dim = num_joints * 2 # 34 features (X, Y)
        
        print(f"[CEREBRO] Carregando modelo temporal de: {model_path}")
        try:
            self.model = torch.load(model_path, map_location=self.device)
            if hasattr(self.model, 'eval'): self.model.eval()
        except Exception as e:
            print(f"[ERRO] Falha ao carregar modelo temporal: {e}")
            self.model = None

        # Buffer: {track_id: deque([[x,y...], ...])}
        self.buffers = defaultdict(lambda: deque(maxlen=window_size))
        
    def _flatten_kps(self, keypoints):
        """Converte [[x,y,c]..] -> [x,y,x,y..] (34 floats)."""
        flat = []
        for kp in keypoints:
            flat.extend(kp[:2]) 
        # Padding se necessário
        if len(flat) < self.input_dim:
            flat.extend([0.0] * (self.input_dim - len(flat)))
        return flat[:self.input_dim]

    def predict_single(self, track_id, keypoints_raw):
        """Retorna probabilidade de furto (0.0 a 1.0)"""
        if self.model is None: return 0.0
        
        # 1. Pipeline de Dados
        features = self._flatten_kps(keypoints_raw)
        self.buffers[track_id].append(features)
        
        # 2. Cold Start Check
        if len(self.buffers[track_id]) < self.window_size: return 0.0
            
        # 3. Inferência
        seq = np.array(list(self.buffers[track_id]), dtype=np.float32)
        tensor_in = torch.tensor(seq).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor_in)
            # Softmax (Assume classe 1 = Furto)
            if output.shape[-1] >= 2:
                prob = torch.softmax(output, dim=1)[0][cm.POSITIVE_CLASS_ID].item()
            else:
                prob = torch.sigmoid(output).item()
        return prob
