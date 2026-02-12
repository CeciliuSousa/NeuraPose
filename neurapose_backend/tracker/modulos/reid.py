# ================================================================
# neurapose_backend/tracker/modulos/reid.py
# ================================================================

import torch
import torch.nn as nn
import cv2
import numpy as np
from ultralytics.utils.ops import xywh2xyxy

import torchvision.transforms as T
import neurapose_backend.config_master as cm
from neurapose_backend.otimizador.cuda.gpu_utils import gpu_manager
from colorama import Fore


class OSNetAIN(nn.Module):
    def __init__(self, state_dict, model_name="osnet_x1_0"):
        super().__init__()
        
        # Importação Dinâmica baseada no nome
        # Ex: "osnet_x0_5" -> from torchreid.reid.models import osnet_x0_5
        import torchreid.reid.models as models
        
        if hasattr(models, model_name):
            builder = getattr(models, model_name)
        else:
            print(Fore.RED + f"[ReID] Arquitetura '{model_name}' não encontrada. Usando 'osnet_x1_0' como fallback.")
            builder = models.osnet_x1_0
            
        # Instancia sem pre-treino (vamos carregar os nossos)
        self.model = builder(num_classes=0, pretrained=False)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def forward(self, x):
        return self.model(x)


class CustomReID:
    def __init__(self, model_path):
        ckpt = torch.load(model_path, map_location=cm.DEVICE)
        state_dict = ckpt.get("state_dict", ckpt)
        
        # Detecção de arquitetura pelo nome do arquivo
        fname = str(model_path).lower()
        model_name = "osnet_x1_0" # fallback padrao
        
        # Lista de arquiteturas conhecidas do torchreid (na ordem do mais especifico para o mais geral)
        known_archs = [
            "osnet_ain_x1_0", "osnet_ain_x0_75", "osnet_ain_x0_5", "osnet_ain_x0_25",
            "osnet_ibn_x1_0", "osnet_x1_0", "osnet_x0_75", "osnet_x0_5", "osnet_x0_25"
        ]
        
        for arch in known_archs:
            if arch in fname:
                model_name = arch
                # print(f"[ReID] Arquitetura detectada pelo nome: {model_name}")
                break
        
        # Remove chaves do classificador (geram erro de tamanho pois num_classes=0)
        # O checkpoint tem classifier para 4101 classes (MSMT17), nos usamos 0 (Feature Extractor)
        keys_to_remove = ["classifier.weight", "classifier.bias"]
        for k in keys_to_remove:
            if k in state_dict:
                del state_dict[k]
        
        self.model = OSNetAIN(state_dict, model_name=model_name)
        self.device = cm.DEVICE
        self.model.to(self.device)
        
        # Configuração de Precisão (FP16 vs FP32)
        # Só ativa FP16 se o usuário quiser E se estivermos rodando em CUDA
        self.use_fp16 = getattr(cm, 'USE_FP16', False) and (self.device == 'cuda')  
        
        if self.use_fp16:
            # print(Fore.CYAN + f"[ReID] Otimização FP16 Ativada (Half-Precision)")
            self.model.half() # Converte pesos e buffers para FP16
        else:
            # print(Fore.YELLOW + f"[ReID] Rodando em FP32 (Padrão)")
            pass

        self.norm = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img, dets):
        if dets is None or len(dets) == 0:
            return []

        boxes_xyxy = xywh2xyxy(torch.from_numpy(dets[:, :4]))
        
        # Batching images
        img_h, img_w = img.shape[:2]
        batch_crops = []
        valid_indices = []
        
        for i, box in enumerate(boxes_xyxy):
            # Recorte manual eficiente
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                # Resize direto
                crop_resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
                # Normalização e Permutação (sempre em float32/64 primeiro para precisão no pré-proc)
                crop_t = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0
                crop_t = self.norm(crop_t)
                batch_crops.append(crop_t)
                valid_indices.append(i)
        
        if not batch_crops:
            return [np.zeros(512, dtype=np.float32) for _ in range(len(boxes_xyxy))]

        # Stack into a single tensor [B, C, H, W]
        batch_t = torch.stack(batch_crops).to(self.device)
        
        # Conversão para FP16 se ativado
        if self.use_fp16:
            batch_t = batch_t.half()

        with gpu_manager.inference_mode():
            batch_feats = self.model(batch_t)
        
        # Se estiver em FP16, converte de volta para Float32 antes de sair
        # Isso garante compatibilidade com o resto do sistema (BoTSORT/NumPy)
        if self.use_fp16:
            batch_feats = batch_feats.float()
        
        # Move back to CPU and convert to numpy
        batch_feats = batch_feats.cpu().numpy()

        # Reassemble features in original order
        final_feats = [None] * len(boxes_xyxy)
        for idx, i in enumerate(valid_indices):
            final_feats[i] = batch_feats[idx]
            
        # Fill None with zero vectors
        for i in range(len(final_feats)):
            if final_feats[i] is None:
                final_feats[i] = np.zeros(512, dtype=np.float32)

        return final_feats
