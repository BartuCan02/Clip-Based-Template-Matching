import os
import json

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import glob

from ..transforms import large_transform

class RPINE_Dataset(Dataset):
    def __init__(self, root, transform, max_exemplars = 1, scale_factor=32, split="test", now_eval=False):
        print("@@@@@@@@@@@@@@@@@@@@ Custom RPINE Dataset @@@@@@@@@@@@@@@@@@@@")
        print("This data is RPINE dataset")
        data_path = root

        self.split = split
        self.label_path = os.path.join(data_path, 'labels')
        self.image_path = os.path.join(data_path, 'images')
        self.exemplars_path = os.path.join(data_path, 'exemplars.json')
        self.max_exemplars = max_exemplars
        self.scale_factor = scale_factor

        self.img_urls = {}
        self.labels = sorted(glob.glob(self.label_path + '/*'))
        self.transform = transform
        self.eval = now_eval

        with open(self.exemplars_path, "r") as exemplars_json:
            self.exemplars_dict = json.load(exemplars_json)

        print("This data contains: {} images".format(len(self.labels)))
        print("--------------------------------------------------")

    def __len__(self):
        return len(self.labels)
    
    def get_bboxes(self, label_file):
        bboxes = []
        with open(label_file,'r') as f:
            for line in f:
                line = line.rstrip('\n').split()
                x1, y1, x2, y2 = line
                bboxes.append([int(x1),int(y1),int(x2),int(y2)])
        bboxes = np.array(bboxes, dtype=np.float32)

        return bboxes
    
    def img_url_finder(self, img_name):
        img_url = None

        if img_name in self.img_urls.keys():
            img_url = self.img_urls[img_name]
        else:
            for ex in ['.jpg', '.jpeg', '.png']:
                if os.path.exists(os.path.join(self.image_path, img_name + ex)):
                    img_name += ex
                    break

            img_url = os.path.join(self.image_path, img_name)
            self.img_urls[img_name] = img_url
        return img_url
    
    def box_coords_encoder(self, bboxes, exemplars): # for albumentations style box transformation
        all_boxes = []
        epsilon = 1e-7
        for box in bboxes:
            x1, y1 = min(1., max(0., box[0])), min(1., max(0., box[1]))
            x2, y2 = min(1., max(0., box[2] + epsilon)), min(1., max(0., box[3] + epsilon))
            all_boxes.append([x1, y1, x2, y2, 0]) # 0 for indicate bboxes
        for box in exemplars:
            x1, y1 = min(1., max(0., box[0])), min(1., max(0., box[1]))
            x2, y2 = min(1., max(0., box[2] + epsilon)), min(1., max(0., box[3] + epsilon))
            all_boxes.append([x1, y1, x2, y2, 1]) # 1 for indicate exempalrs

        return all_boxes
    
    def box_coords_decoder(self, all_boxes): # for albumentations style box transformation
        bboxes = []
        exemplars = []

        for box in all_boxes:
            if box[4] == 0:
                bboxes.append([box[0], box[1], box[2], box[3]])
            else:
                exemplars.append([box[0], box[1], box[2], box[3]])

        bboxes = np.array(bboxes, dtype=np.float32)
        exemplars = np.array(exemplars, dtype=np.float32)

        return bboxes, exemplars
    
    def _pre_transform_hook(self, image, scaled_boxes, scaled_exemplars):
        return image, scaled_boxes, scaled_exemplars
    
    def __getitem__(self, idx):
        label_file = self.labels[idx]

        # get img_name, img_url
        img_name = os.path.basename(label_file).split('.')[0]
        img_url = self.img_url_finder(img_name)
        
        # image, boxes, exemplars
        image = Image.open(img_url).convert("RGB")
        img_w, img_h = image.size

        bboxes = self.get_bboxes(label_file) # xyxy format

        exemplars = self.exemplars_dict[img_name]
        exemplars = exemplars[:min(self.max_exemplars, len(exemplars))] # get only predefined number of exemplars
        exemplars = np.array(exemplars, dtype=np.float32) # xyxy format

        # box scaling
        img_size = np.array([img_w, img_h])
        img_res = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)

        scaled_boxes = bboxes / img_res[None, :]
        scaled_exemplars = exemplars / img_res[None, :]

        # transform
        image = np.array(image)
        image, scaled_boxes, scaled_exemplars = self._pre_transform_hook(image, scaled_boxes, scaled_exemplars)
        if self.split == "test" and self.eval and (bboxes[:, 2] - bboxes[:, 0]).min() < 25 and (bboxes[:, 3] - bboxes[:, 1]).min() < 25:
            augment = large_transform()(image = image)
            image = augment['image']
        elif 'bboxes' in self.transform.processors.keys(): # if albumentations transform has bbox transformation
            all_boxes = self.box_coords_encoder(scaled_boxes, scaled_exemplars)
            augment = self.transform(image = image, bboxes=all_boxes) # only for albumentations box parameters == "albumentations" : normalized[x,y,x,y]

            image = augment['image']
            scaled_boxes, scaled_exemplars = self.box_coords_decoder(augment['bboxes'])
        else:
            augment = self.transform(image = image)
            image = augment['image']

        ret = {
            "image": image,
            "boxes": scaled_boxes,
            "exemplars": scaled_exemplars,

            "img_name": img_name,
            "img_url": img_url,
            "img_id": idx,
            "img_size": img_size,
            "orig_boxes": bboxes,
            "orig_exemplars": exemplars
        }
        return ret
    