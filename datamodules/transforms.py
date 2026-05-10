import random
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Any, Dict, Union

# from torchvision import transforms as transforms

class GTBasedRandomCrop(A.BBoxSafeRandomCrop):
    def __init__(self, keep_hw_ratio = True):
        super().__init__()

        # self.keep_hw_ratio = keep_hw_ratio

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Union[int, float]]:
        img_h, img_w = params["image"].shape[:2]
        if len(params["bboxes"]) == 0:  # less likely, this class is for use with bboxes.
            raise ValueError("len(bboxes) must over than 0")
        # chosse random box

        bboxes = np.array(params["bboxes"])
        x, y, x2, y2 = random.choice(bboxes[np.array(params["bboxes"], np.int32)[:,4] == 0])[:4]
        
        # find bigger region
        bx, by = x * random.random(), y * random.random()
        bx2, by2 = x2 + (1 - x2) * random.random(), y2 + (1 - y2) * random.random()

        bw, bh = bx2 - bx, by2 - by
        crop_height = img_h if bh >= 1.0 else int(img_h * bh)
        crop_width = img_w if bw >= 1.0 else int(img_w * bw)
        h_start = np.clip(0.0 if bh >= 1.0 else by / (1.0 - bh), 0.0, 1.0)
        w_start = np.clip(0.0 if bw >= 1.0 else bx / (1.0 - bw), 0.0, 1.0)
        return {"h_start": h_start, "w_start": w_start, "crop_height": crop_height, "crop_width": crop_width}

def get_transforms(size):
    return {
        "default": default_transform(size),
        "minimum": minimum_transform(),
    }

def default_transform(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def minimum_transform():
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def large_transform():
    return A.Compose([
        A.Resize(1536, 1536),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])