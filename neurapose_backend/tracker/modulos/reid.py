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
from neurapose_backend.config_master import DEVICE


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
        ckpt = torch.load(model_path, map_location=DEVICE)
        state_dict = ckpt.get("state_dict", ckpt)

        self.model = OSNetAIN(state_dict)
        self.device = DEVICE
        self.model.to(self.device)

        self.norm = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img, dets):
        if dets is None or len(dets) == 0:
            return []

        boxes_xyxy = xywh2xyxy(torch.from_numpy(dets[:, :4]))
        crops = [save_one_box(box, img, save=False) for box in boxes_xyxy]

        feats = []
        for crop in crops:
            if crop is None or crop.size == 0:
                feats.append(np.zeros(512, dtype=np.float32))
                continue

            crop_resized = cv2.resize(crop, (128, 256))
            crop_t = torch.from_numpy(crop_resized.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            crop_t = self.norm(crop_t)

            crop_t = crop_t.to(self.device)

            with torch.no_grad():
                f = self.model(crop_t)
            feats.append(f.squeeze().cpu().numpy())

        return feats
