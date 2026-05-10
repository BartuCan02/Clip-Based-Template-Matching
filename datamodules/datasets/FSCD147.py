import json
import os

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import transforms as transforms

from ..transforms import large_transform

class FSCD147_Dataset(Dataset):
    def __init__(self, root, transform, max_exemplars = 1, scale_factor=32, split="val", now_eval=False):

        instances_json_file = {
            "train": "instances_train.json",
            "val": "instances_val.json",
            "test": "instances_test.json"
        }[split]

        print("@@@@@@@@@@@@@@@@@@@@ FSCD-147 @@@@@@@@@@@@@@@@@@@@")
        print("This data is fscd 147 dataset, split: {}".format(split))
        data_path = root
        
        self.split = split
        self.anno_file = os.path.join(data_path, "annotations", "annotation_FSC147_384.json")
        self.data_split_file = os.path.join(data_path, "annotations", "Train_Test_Val_FSC_147.json")
        self.im_dir = os.path.join(data_path, "images_384_VarV2")
        self.instance_file = os.path.join(data_path, "annotations", instances_json_file)
        self.max_exemplars = max_exemplars
        self.scale_factor = scale_factor

        if self.max_exemplars > 3:
            raise ValueError("FSCD147 has maximum 3 exemplars per images")

        self.annotations = self.load_json(self.anno_file)
        self.data_split = self.data_split_loader(split)
        self.label_instance = COCO(self.instance_file)
        self.img_name_to_ori_id = self.map_img_name_to_ori_id()
        self.transform = transform
        self.eval = now_eval

        print("This data contains: {} images".format(len(self.data_split)))
        print("--------------------------------------------------")

    def __len__(self):
        return len(self.data_split)

    def load_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        return data
    
    def data_split_loader(self, split):
        data_split = self.load_json(self.data_split_file)[split]
        return data_split

    def map_img_name_to_ori_id(self,):
        all_coco_imgs = self.label_instance.imgs
        map_name_2_id = dict()
        for k, v in all_coco_imgs.items():
            img_id = v["id"]
            img_name = v["file_name"]
            map_name_2_id[img_name] = img_id
        return map_name_2_id
    
    def get_bboxes(self, img_name):
        bboxes = []

        coco_im_id = self.img_name_to_ori_id[img_name]
        anno_ids = self.label_instance.getAnnIds([coco_im_id])
        annos = self.label_instance.loadAnns(anno_ids)

        for anno in annos:
            x1, y1, w, h = anno["bbox"]
            bboxes.append([int(x1),int(y1),int(x1 + w),int(y1 + h)])

        bboxes = np.array(bboxes, dtype=np.float32)
        return bboxes
    
    def get_exemplars(self, img_name):
        exemplars = []
        ori_exemplar_boxes = self.annotations[img_name]["box_examples_coordinates"][:self.max_exemplars]

        for exemplar_box in ori_exemplar_boxes:
            y1 = exemplar_box[0][1]
            x1 = exemplar_box[0][0]
            x2 = exemplar_box[2][0]
            y2 = exemplar_box[2][1]
            exemplars.append([x1, y1, x2, y2])
        exemplars = np.array(exemplars, dtype=np.float32)
        return exemplars

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
        # get img_name, img_url
        img_name = self.data_split[idx]
        img_url = "{}/{}".format(self.im_dir, img_name)

        # image, boxes, exemplars
        image = Image.open(img_url)
        img_w, img_h = image.size

        bboxes = self.get_bboxes(img_name) # xyxy format
        exemplars = self.get_exemplars(img_name) # xyxy format

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
