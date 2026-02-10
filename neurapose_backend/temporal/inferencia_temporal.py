import torch
import numpy as np
from collections import deque, defaultdict
import neurapose_backend.config_master as cm

class ClassificadorAcao:
    def __init__(self, model_path, model_instance=None, window_size=None, num_joints=cm.NUM_JOINTS, mu=None, sigma=None):
        self.device = cm.DEVICE
        self.window_size = window_size if window_size is not None else cm.TIME_STEPS
        self.input_dim = num_joints * 2 # 34 features (X, Y)
        
        # Normalização
        self.mu = mu.to(self.device) if mu is not None else None
        self.sigma = sigma.to(self.device) if sigma is not None else None
        
        # [RESOLUÇÃO] Estima escala do treino baseada no MU (Média das coordenadas X e Y)
        # Se max(mu_x) > 1.0, assumimos coordenadas em pixels.
        self.train_mean_x = 1.0
        self.fix_resolution = False
        
        if self.mu is not None:
            # mu tem shape (1, 1, 34) ou (1, 34) dependendo
            # Assumindo flatten (X...X, Y...Y) com 17 joints
            # Indices 0..16 = X, 17..33 = Y
            try:
                flat_mu = self.mu.flatten()
                if len(flat_mu) >= 17:
                    # Pega a média dos primeiros 17 valores (X)
                    self.train_mean_x = flat_mu[:17].mean().item()
                    
                    # Heurística: Se média X > 100, é pixel.
                    if self.train_mean_x > 100:
                        self.fix_resolution = True
                        # print(Fore.BLUE + f"[TCN] Escala de Treino Detectada: X_mean={self.train_mean_x:.1f}")
            except Exception as e:
                print(f"[AVISO] Falha ao estimar escala de treino: {e}")

        if model_instance is not None:
             # print(f"[CEREBRO] Usando modelo pré-carregado em memória.")
             self.model = model_instance
             self.model.to(self.device)
             self.model.eval()
        else:
             # ... (Legacy loading code)
             pass
             try:
                # Tenta carregar objeto completo (Legacy)
                self.model = torch.load(model_path, map_location=self.device)
                if isinstance(self.model, dict):
                    print(f"[ERRO] CHECKPOINT CONTEM APENAS STATE_DICT! Use a Factory para carregar.")
                    self.model = None
                elif hasattr(self.model, 'eval'): 
                    self.model.eval()
             except Exception as e:
                print(f"[ERRO] Falha ao carregar modelo temporal: {e}")
                self.model = None

        # Buffer: {track_id: deque([[x,y...], ...])}
        self.buffers = defaultdict(lambda: deque(maxlen=window_size))
        
    def _flatten_kps(self, keypoints):
        """
        Converte keypoints para o formato do treino: [x0...x16, y0...y16] (34 floats).
        Treino: (N, C, T, V) -> permute(0,2,1,3) -> (N, T, C, V) -> reshape(N, T, C*V)
        Isso agrupa por Canal primeiro (todos X, depois todos Y).
        """
        xs = []
        ys = []
        for kp in keypoints:
            xs.append(kp[0])
            ys.append(kp[1])
            
        # Concatena [x0..x16, y0..y16]
        flat = xs + ys
        
        # Padding se faltar (ex: 17 juntas * 2 = 34)
        if len(flat) < self.input_dim:
            flat.extend([0.0] * (self.input_dim - len(flat)))
        return flat[:self.input_dim]

    def predict_single(self, track_id, keypoints_raw):
        """Retorna probabilidade da CLASSE2 (0.0 a 1.0)"""
        if self.model is None: return 0.0
        
        # 1. Pipeline de Dados
        features = self._flatten_kps(keypoints_raw)
        self.buffers[track_id].append(features)
        
        # 2. Prepare Sequence (Pad if needed for Cold Start)
        buffer_len = len(self.buffers[track_id])
        if buffer_len < 1: return 0.0
        
        # Se buffer não está cheio, repete o primeiro frame (padding à esquerda ou repetição)
        # Strategy: Repeat content to fill window_size
        if buffer_len < self.window_size:
            missing = self.window_size - buffer_len
            # Opção A: Pad com zeros (pode ser estranho para TCN)
            # Opção B: Repetir o primeiro frame (mais estável para pose parada)
            first_frame = self.buffers[track_id][0]
            padded_seq = [first_frame] * missing + list(self.buffers[track_id])
            seq = np.array(padded_seq, dtype=np.float32)
        else:
            seq = np.array(list(self.buffers[track_id]), dtype=np.float32)

        # 3. Inferência
        tensor_in = torch.tensor(seq).unsqueeze(0).to(self.device) # (1, T, F)
        
        # [RESOLUÇÃO] Aplica escala se necessário
        if self.fix_resolution:
            # Estima escala do input atual
            input_x = tensor_in[:, :, :17]
            input_mean_x = input_x[input_x > 0].float().mean() # Ignora zeros
            
            if not torch.isnan(input_mean_x) and input_mean_x > 10: # Se for válido
                 ratio_x = self.train_mean_x / input_mean_x.item()
                 # Se a diferença for significativa (>20%), aplica correção
                 if ratio_x < 0.8 or ratio_x > 1.2:
                      tensor_in = tensor_in * ratio_x
        
        # Aplica normalização se disponível
        if self.mu is not None and self.sigma is not None:
            tensor_in = (tensor_in - self.mu) / self.sigma
        
        with torch.no_grad():
            output = self.model(tensor_in)
            if output.shape[-1] >= 2:
                prob = torch.softmax(output, dim=1)[0][cm.POSITIVE_CLASS_ID].item()
            else:
                prob = torch.sigmoid(output).item()
        return prob

    def predict_batch(self, track_ids, keypoints_list):
        """
        Realiza inferência em BATCH para N pessoas simultaneamente (O(1) GPU overhead).
        Args:
            track_ids (list): Lista de IDs.
            keypoints_list (list): Lista de keypoints raw [(17,2), ...].
        Returns:
            dict: {track_id: prob}
        """
        if self.model is None or not track_ids: return {}
        
        valid_seqs = []
        valid_ids = []
        
        for tid, kps in zip(track_ids, keypoints_list):
            features = self._flatten_kps(kps)
            self.buffers[tid].append(features)
            
            buffer_len = len(self.buffers[tid])
            if buffer_len < 1: continue 

            if buffer_len < self.window_size:
                missing = self.window_size - buffer_len
                first_frame = self.buffers[tid][0]
                padded_seq = [first_frame] * missing + list(self.buffers[tid])
                seq = np.array(padded_seq, dtype=np.float32)
            else:
                seq = np.array(list(self.buffers[tid]), dtype=np.float32)
            
            valid_seqs.append(seq)
            valid_ids.append(tid)

        if not valid_seqs: return {}

        tensor_batch = torch.tensor(np.array(valid_seqs)).to(self.device)
        
        # [RESOLUÇÃO] Aplica escala dinâmica se detectado descasamento
        if self.fix_resolution:
             # Calcula média X do batch atual (considerando apenas > 0 para ignorar padding/miss)
             batch_x = tensor_batch[:, :, :17]
             input_mean_x = batch_x[batch_x > 1].float().mean()
             
             if not torch.isnan(input_mean_x) and input_mean_x > 10:
                  ratio_x = self.train_mean_x / input_mean_x.item()
                  
                  # Aplica se desvio > 20%
                  if ratio_x < 0.8 or ratio_x > 1.2:
                       # Multiplica todo o tensor pelo ratio (assumindo escala uniforme X/Y)
                       # print(f"[DEBUG] Scaling input by {ratio_x:.2f} (Train={self.train_mean_x:.0f}, Input={input_mean_x:.0f})")
                       tensor_batch = tensor_batch * ratio_x

        # Normalização Batch
        if self.mu is not None and self.sigma is not None:
             tensor_batch = (tensor_batch - self.mu) / self.sigma

        probs_map = {}
        with torch.no_grad():
            output = self.model(tensor_batch)
            
            if output.shape[-1] >= 2:
                probs = torch.softmax(output, dim=1)[:, cm.POSITIVE_CLASS_ID].cpu().numpy()
            else:
                 probs = torch.sigmoid(output).flatten().cpu().numpy()
            
            for tid, p in zip(valid_ids, probs):
                probs_map[tid] = float(p)
                
        return probs_map
