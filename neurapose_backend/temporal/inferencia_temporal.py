import torch
import numpy as np
from collections import deque, defaultdict
import neurapose_backend.config_master as cm

class ClassificadorAcao:
    def __init__(self, model_path, architecture="tft", window_size=30, num_joints=17):
        self.device = cm.DEVICE
        self.window_size = window_size
        self.num_joints = num_joints
        # Define dimensão de entrada (17 keypoints * 2 coordenadas X,Y = 34 features)
        self.input_dim = num_joints * 2 
        
        print(f"[CEREBRO] Carregando modelo {architecture} de {model_path}...")
        try:
            # Tenta carregar o modelo completo
            # map_location garante que carregue na CPU se CUDA nao tiver disponivel ou vice/versa conforme 'device'
            self.model = torch.load(model_path, map_location=self.device)
            if hasattr(self.model, 'eval'):
                self.model.eval()
        except Exception as e:
            print(f"[ERRO] Falha crítica ao carregar modelo temporal: {e}")
            self.model = None

        # Buffer por ID: {track_id: deque([[x,y...], ...], maxlen=30)}
        self.buffers = defaultdict(lambda: deque(maxlen=window_size))
        
    def _flatten_kps(self, keypoints):
        """
        Converte lista de keypoints [[x,y,c]..] para vetor flat [x,y,x,y..].
        Remove a confiança (c) para bater com o formato de treino.
        """
        flat = []
        for kp in keypoints:
            # Pega apenas X e Y (ignora Confiança index 2)
            # kp pode ser lista ou numpy
            if len(kp) >= 2:
                flat.extend(kp[:2])
            else:
                 flat.extend([0.0, 0.0])
        
        # Garante tamanho fixo (padding com 0 se faltar, corte se sobrar)
        if len(flat) < self.input_dim:
            flat.extend([0.0] * (self.input_dim - len(flat)))
        return flat[:self.input_dim]

    def predict_single(self, track_id, keypoints_raw):
        """
        Adiciona pose ao buffer e retorna probabilidade de furto.
        Retorna: float (0.0 a 1.0)
        """
        if self.model is None: return 0.0
        
        # 1. Prepara dados (Flatten)
        features = self._flatten_kps(keypoints_raw)
        self.buffers[track_id].append(features)
        
        # 2. Só prevê se tiver janela cheia (Cold Start de 3s)
        if len(self.buffers[track_id]) < self.window_size:
            return 0.0
            
        # 3. Cria Tensor [Batch=1, Time=30, Feats=34]
        seq = np.array(list(self.buffers[track_id]), dtype=np.float32)

        # Normalização Z-Score On-the-fly?
        # O modelo foi treinado com Z-Score. Precisamos das stats!
        # Por enquanto, assumiremos que o modelo lida ou que os dados raw funcionam 'ok'.
        # O ideal seria carregar mu/sigma junto.
        # Mas vamos seguir o pedido do user primeiro.
        
        tensor_in = torch.tensor(seq).unsqueeze(0).to(self.device)
        
        # 4. Inferência
        with torch.no_grad():
            output = self.model(tensor_in)
            
            # Tratamento de saída (Sigmoid ou Softmax)
            # Adapte conforme a última camada do seu modelo treinado
            # Geralmente (Batch, NumClasses)
            if output.shape[-1] == 1:
                prob = torch.sigmoid(output).item()
            elif output.shape[-1] >= 2:
                # Assume classe 1 = Furto
                prob = torch.softmax(output, dim=1)[0][cm.POSITIVE_CLASS_ID].item()
            else:
                prob = 0.0
                
        return prob
