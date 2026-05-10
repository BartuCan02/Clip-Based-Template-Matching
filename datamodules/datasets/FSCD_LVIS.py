import os
import json

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import glob

from ..transforms import large_transform

class FSCD_LVIS_Dataset(Dataset):
    def __init__(self, root, transform, max_exemplars = 1, scale_factor=32, split="train", now_eval=False, LVIS_split_unseen = False):

        self.split = split

        if LVIS_split_unseen == False:
            # For FSCD-LVIS Seen split
            count_json_file = "count_train.json" if split == 'train' else "count_test.json"
            instances_json_file = "instances_train.json" if split == 'train' else "instances_test.json"
        else:
            # For FSCD-LVIS Unseen split
            count_json_file = "unseen_count_train.json" if split == 'train' else "unseen_count_test.json"
            instances_json_file = "unseen_instances_train.json" if split == 'train' else "unseen_instances_test.json"

        print("@@@@@@@@@@@@@@@@@@@@ FSCD-LVIS @@@@@@@@@@@@@@@@@@@@")
        print("This data is fscd LVIS dataset, split: {}".format(split))
        data_path = root

        self.im_dir = os.path.join(data_path, "images")
        self.count_file = os.path.join(data_path, "annotations", count_json_file)
        self.instances_file = os.path.join(data_path, "annotations", instances_json_file)
        self.max_exemplars = max_exemplars
        self.scale_factor = scale_factor

        if self.max_exemplars > 3:
            raise ValueError("FSCD147 has maximum 3 exemplars per images")
        
        self.label_instance = COCO(self.instances_file)
        self.image_ids = self.label_instance.getImgIds()

        count_annos = self.load_json(self.count_file)
        self.count_anno = self.label_organizer(count_annos)
        self.transform = transform
        self.eval = now_eval

        print("This data contains: {} images".format(len(self.image_ids)))
        print("--------------------------------------------------")
        
    def __len__(self):
        return len(self.image_ids)

    def load_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        return data
    
    def label_organizer(self, annotations):
        label_library = {}

        img_infos = annotations["images"] # id, width, height, file_name
        anno_infos = annotations['annotations'] # id, image_id, boxes, points, file_name

        for img_info in img_infos:  
            label_library[img_info["id"]] = img_info 
        for anno_info in anno_infos:
            id = anno_info["id"]
            label_library[id]['boxes'] = anno_info['boxes']
            label_library[id]['points'] = anno_info['points']
            label_library[id]['image_id'] = anno_info['image_id']

        label_list = {}
        for key in label_library.keys():
            now_label = label_library[key]
            label_list[now_label['image_id']] = now_label

        return label_list

    def get_bboxes(self, img_id):
        bboxes = []

        anno_ids = self.label_instance.getAnnIds([img_id])
        annos = self.label_instance.loadAnns(anno_ids)

        for anno in annos:
            x1, y1, w, h = anno["bbox"]
            bboxes.append([int(x1),int(y1),int(x1 + w),int(y1 + h)])

        bboxes = np.array(bboxes, dtype=np.float32)
        return bboxes
    
    def get_exemplars(self, ex_boxes):
        bboxes = []
        for bbox in ex_boxes[:self.max_exemplars]: # xywh
            x1, y1, w, h = bbox
            bboxes.append([int(x1),int(y1),int(x1 + w),int(y1 + h)])
        bboxes = np.array(bboxes, dtype=np.float32)
        return bboxes

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
        img_id = self.image_ids[idx]
        now_anno = self.count_anno[img_id]

        # get img_name, img_url
        img_name = now_anno['file_name']
        img_url = "{}/{}".format(self.im_dir, img_name)

        # image, boxes, exemplars
        image = Image.open(img_url)
        image = image.convert("RGB")
        img_w, img_h = image.size

        bboxes = self.get_bboxes(img_id) # xyxy format
        exemplars = self.get_exemplars(now_anno['boxes']) # xyxy format

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
