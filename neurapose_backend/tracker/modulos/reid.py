# ================================================================
# neurapose_backend/tracker/modulos/reid.py
# ================================================================

import torch
import torch.nn as nn
import cv2
import numpy as np
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.plotting import save_one_box

import torchvision.transforms as T
import neurapose_backend.config_master as cm


class OSNetAIN(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        from torchreid.reid.models import osnet_ain_x1_0
        self.model = osnet_ain_x1_0(num_classes=0, pretrained=False)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def forward(self, x):
        return self.model(x)


class CustomReID:
    def __init__(self, model_path):
        ckpt = torch.load(model_path, map_location=cm.DEVICE)
        state_dict = ckpt.get("state_dict", ckpt)

        self.model = OSNetAIN(state_dict)
        self.device = cm.DEVICE
        self.model.to(self.device)

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
                # Normalização e Permutação
                crop_t = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0
                crop_t = self.norm(crop_t)
                batch_crops.append(crop_t)
                valid_indices.append(i)
        
        if not batch_crops:
            return [np.zeros(512, dtype=np.float32) for _ in range(len(boxes_xyxy))]

        # Stack into a single tensor [B, C, H, W]
        batch_t = torch.stack(batch_crops).to(self.device)

        with torch.no_grad():
            batch_feats = self.model(batch_t)
        
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
