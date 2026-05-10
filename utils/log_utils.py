import os
import cv2
import copy
import json
import shutil
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchmetrics.detection import MeanAveragePrecision

IMG_LOG_PATH = 'logged_datas'
IMG_VIS_PATH = 'image_visualize'
PR_VIS_PATH = 'PR_visualize'
GTS_NAME_FORMAT = 'instances'
PRED_NAME_FORMAT = 'predictions'

def image_info_collector(log_path, stage, batch, pred_logits, pred_boxes, ref_points):
    log_path = os.path.join(log_path, IMG_LOG_PATH)
    log_path = os.path.join(log_path, stage)
    os.makedirs(log_path, exist_ok=True)

    for bidx in range(len(batch['img_name'])):
        logits          = pred_logits[bidx]
        bboxes          = pred_boxes[bidx] # scaled xyxy format
        points          = ref_points[bidx]
        
        img_name        = batch['img_name'][bidx]
        img_url         = batch["img_url"][bidx]
        img_id          = batch["img_id"][bidx]
        img_size        = batch["img_size"][bidx].tolist()
        orig_boxes      = batch["orig_boxes"][bidx] # non-scaled xyxy format
        orig_exemplars  = batch["orig_exemplars"][bidx] # non-scaled xyxy format

        orig_boxes, orig_exemplars = gt_refinery(orig_boxes, orig_exemplars)
        logits, bboxes, points = pred_refinery(logits, bboxes, points, img_size)

        with open(os.path.join(log_path, f"{img_id}.json"), "w") as img_file:
            json.dump({
                "img_name":         img_name,
                "img_url":          img_url,
                "img_id":           img_id,
                "img_size":         img_size,
                "orig_boxes":       orig_boxes, # non-scaled xywh format
                "orig_exemplars":   orig_exemplars, # non-scaled xywh format
                "logits":           logits,
                "bboxes":           bboxes, # non-scaled xywh format
                "points":           points,
            }, img_file, indent=4)

def gt_refinery(orig_boxes, orig_exemplars):
    # orig_boxes xyxy -> xywh
    xys = orig_boxes[:,:2]
    whs = orig_boxes[:,2:] - orig_boxes[:,:2]
    new_boxes = np.concatenate((xys, whs), axis=1)

    # orig_exemplars xyxy -> xywh
    exys = orig_exemplars[:,:2]
    ewhs = orig_exemplars[:,2:] - orig_exemplars[:,:2]
    new_exemplars = np.concatenate((exys, ewhs), axis=1)

    # np array -> list
    new_boxes = np.round(new_boxes).astype(int).tolist()
    new_exemplars = np.round(new_exemplars).astype(int).tolist()

    return new_boxes, new_exemplars

def pred_refinery(logits, bboxes, ref_points, img_size):
    img_w, img_h = img_size

    # select boxes over threshold (logits)
    prob = logits
    object_prob = prob[..., 0]
    threshold = 0.
    obj_pos = torch.where(object_prob >= threshold)

    logits = logits[obj_pos]
    bboxes = bboxes[obj_pos]
    ref_points = ref_points[obj_pos]

    # points refinement
    ref_points = ref_points.detach().cpu().numpy().astype(np.float32)
    ref_points[..., 0] *= np.array(img_w, np.float32)
    ref_points[..., 1] *= np.array(img_h, np.float32)
    ref_points = np.round(ref_points).astype(int)

    # boxes refinement
    bboxes = bboxes.detach().cpu().numpy().astype(np.float32)
    bboxes[..., 0] *= np.array(img_w, np.float32)
    bboxes[..., 1] *= np.array(img_h, np.float32)
    bboxes[..., 2] *= np.array(img_w, np.float32)
    bboxes[..., 3] *= np.array(img_h, np.float32)
    bboxes = np.round(bboxes).astype(int)

    # xyxy -> xywh
    xys = bboxes[:,:2]
    whs = bboxes[:,2:] - bboxes[:,:2]
    bboxes = np.concatenate((xys, whs), axis=1)

    # type conversion
    logits = logits.tolist()
    bboxes = bboxes.tolist()
    ref_points = ref_points.tolist()

    return logits, bboxes, ref_points

def Get_MAE_RMSE(log_path, stage):
    gt_coco_api = COCO(os.path.join(log_path, f"{GTS_NAME_FORMAT}_{stage}.json"))
    pred_coco_api = COCO(os.path.join(log_path, f"{PRED_NAME_FORMAT}_{stage}.json"))

    error = 0
    squared_error = 0

    f_txt = open(os.path.join(log_path, f"MAE_RMSE_{stage}.txt"), 'w')

    img_ids = pred_coco_api.getImgIds()
    for img_id in img_ids:
        num_objects_gt = len(gt_coco_api.getAnnIds([img_id]))
        num_objects_pred = len(pred_coco_api.getAnnIds([img_id]))

        error += abs(num_objects_gt - num_objects_pred)
        squared_error += (num_objects_gt - num_objects_pred) ** 2

        img_info = pred_coco_api.loadImgs([img_id])
        img_name = img_info[0]['file_name']
        f_txt.write(f"{img_name}\t\t{num_objects_gt}\t\t{num_objects_pred}\t\t{abs(num_objects_gt - num_objects_pred)}\t\t{(num_objects_gt - num_objects_pred) ** 2}\n")

    MAE = error / len(img_ids)
    RMSE = np.sqrt(squared_error / len(img_ids))

    f_txt.close()

    return MAE, RMSE

def Get_AP_scores(log_path, stage, img_visualize = False):
    coco_eval = coco_evaluate(log_path, stage, img_visualize)

    metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
    results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else 0)
            for idx, metric in enumerate(metrics)
    }

    AP = results["AP"]
    AP50 = results["AP50"]
    AP75 = results["AP75"]
    return AP, AP50, AP75

def coco_evaluate(log_path, stage, img_visualize = False):
    gt_coco_api = COCO(os.path.join(log_path, f"{GTS_NAME_FORMAT}_{stage}.json"))
    pred_coco_api = COCO(os.path.join(log_path, f"{PRED_NAME_FORMAT}_{stage}.json"))

    vis_img_path = os.path.join(log_path, f"{IMG_VIS_PATH}_{stage}")
    if img_visualize:
        os.makedirs(vis_img_path, exist_ok=True)

    # information gathering for AP calculation, and results visualization
    predictions = []
    img_ids = pred_coco_api.getImgIds()
    for img_id in img_ids:
        anno_ids = pred_coco_api.getAnnIds([img_id])
        pred_annos = pred_coco_api.loadAnns(anno_ids)
        img_info = pred_coco_api.loadImgs([img_id])

        if img_visualize:
            gt_anno_ids = gt_coco_api.getAnnIds([img_id])
            gt_annos = gt_coco_api.loadAnns(gt_anno_ids)

            img_name = img_info[0]['file_name']
            img = image_visualization(img_info, gt_annos, pred_annos)

            img_save_path = os.path.join(vis_img_path, f"{img_name}_{img_id}.jpg")
            cv2.imwrite(img_save_path, img)

        # predictions informations collection
        prediction = {"image_id": img_id, "instances": []}
        for anno in pred_annos:
            result = {
                "image_id": anno["image_id"],
                "category_id": anno["category_id"],
                "bbox": anno["bbox"],
                "score": anno["score"],
            }
            prediction["instances"].append(result)
        predictions.append(prediction)
    coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

    coco_pred = gt_coco_api.loadRes(coco_results)
    coco_eval = COCOevalMaxDets(gt_coco_api, coco_pred, 'bbox')
    coco_eval.params.maxDets = [900, 1000, 1100]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    if img_visualize:
        PR_curve_vis_path = os.path.join(log_path, f"Sub_Debug_{PR_VIS_PATH}_{stage}")
        os.makedirs(PR_curve_vis_path, exist_ok=True)

        Draw_PR_curves(coco_eval, PR_curve_vis_path)

    return coco_eval

def del_img_log_path(log_path, stage):
    img_log_path = os.path.join(log_path, IMG_LOG_PATH)
    img_log_path = os.path.join(img_log_path, stage)

    if os.path.exists(img_log_path):
        shutil.rmtree(img_log_path)

def coco_style_annotation_generator(log_path, stage):
    img_log_path = os.path.join(log_path, IMG_LOG_PATH)
    img_log_path = os.path.join(img_log_path, stage)
    img_datas = os.listdir(img_log_path)

    predictions = {
        "categories": [{"name": "fg", "id": 1}], 
        "images": [], 
        "annotations": [], 
        'anno_id': 1
    }

    gts = {
        "categories": [{"name": "fg", "id": 1}], 
        "images": [], 
        "annotations": [], 
        'anno_id': 1
    }

    # coco style data informations collecting sequence
    for img_file in img_datas:
        img_path = os.path.join(img_log_path, img_file)
        with open(img_path, 'r') as json_file:
            img_data = json.load(json_file)

        # collecting image informations
        img_info = {
            "id":               img_data["img_id"],
            "height":           img_data["img_size"][1],
            "width":            img_data["img_size"][0],
            "file_name":        img_data["img_name"],

            "img_url":          img_data["img_url"],
            "exemplar_boxes":   img_data["orig_exemplars"],
        }

        # collecting gt boxes informations
        for gt_box in img_data["orig_boxes"]:
            x, y, w, h = gt_box
            anno = {
                "id": gts['anno_id'],
                "image_id": img_info['id'],
                "area": int(w * h),
                "iscrowd": 0,
                "bbox": [int(x), int(y), int(w), int(h)],
                "category_id": 1,
            }

            gts['annotations'].append(anno)
            gts['anno_id'] += 1
        gts['images'].append(img_info)

        # collecting pred boxes informations
        pred_scores = img_data["logits"]
        pred_boxes = img_data["bboxes"]
        pred_points = img_data["points"]
        for pred_score, pred_box, pred_point in zip(pred_scores, pred_boxes, pred_points):
            score = pred_score[0]
            x, y, w, h = pred_box
            cx, cy = pred_point

            anno = {
                "id": predictions['anno_id'],
                "image_id": img_info['id'],
                "area": int(w * h),
                "bbox": [int(x), int(y), int(w), int(h)],
                "category_id": 1,
                "score": float(score),
                "point": [int(cx), int(cy)],
            }

            predictions['annotations'].append(anno)
            predictions['anno_id'] += 1
        predictions['images'].append(img_info)

        # add dumy file
        if len(predictions['annotations']) == 0:
            predictions['annotations'].append({
                "id": predictions['anno_id'],
                "image_id": img_info['id'],
                "area": 0,
                "bbox": [0, 0, 0, 0],
                "category_id": 1,
                "score": float(0),
                "point": [int(0), int(0)],
            })

    # save generated coco style annotation files
    gts_path = os.path.join(log_path, f"{GTS_NAME_FORMAT}_{stage}.json")
    predictions_path = os.path.join(log_path, f"{PRED_NAME_FORMAT}_{stage}.json")

    with open(gts_path, 'w') as file:
        json.dump(gts, file, indent=4)

    with open(predictions_path, 'w') as file:
        json.dump(predictions, file, indent=4)

def image_visualization(img_info, gt_annos, pred_annos):
    # load image
    img_url = img_info[0]['img_url']
    img = cv2.imread(img_url)

    # resize image
    Max_Width = 512
    H, W, _ = img.shape
    R = Max_Width / W # ratio
    img = cv2.resize(img, (int(W*R), int(H*R)))

    img_gt = copy.deepcopy(img)
    img_pred = copy.deepcopy(img)

    # draw predicted bboxes
    preds = []
    for anno in pred_annos:
        x, y, w, h = anno['bbox']
        x, y, w, h = int(x*R), int(y*R), int(w*R), int(h*R)

        img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        img_pred = cv2.rectangle(img_pred, (x,y), (x+w, y+h), (0,255,0), 2)

        preds.append({
            'bbox': anno['bbox'],
            'score': anno['score']
        })

    # draw GT bboxes
    gts = []
    for anno in gt_annos:
        x, y, w, h = anno['bbox']
        x, y, w, h = int(x*R), int(y*R), int(w*R), int(h*R)

        img = cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        img_gt = cv2.rectangle(img_gt, (x,y), (x+w, y+h), (255,0,0), 2)

        gts.append({
            'bbox': anno['bbox']
        })

    # draw exemplars
    for exemplar in img_info[0]['exemplar_boxes']:
        x, y, w, h = exemplar
        x, y, w, h = int(x*R), int(y*R), int(w*R), int(h*R)

        img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        img_gt = cv2.rectangle(img_gt, (x,y), (x+w, y+h), (0,0,255), 2)
        img_pred = cv2.rectangle(img_pred, (x,y), (x+w, y+h), (0,0,255), 2)

    # get mAP scores of each image
    metric = single_image_mean_average_precision_torchmetric(preds, gts)
    mAP_score = metric['map'] * 100.
    AP50_score = metric['map_50'] * 100.
    AP75_score = metric['map_75'] * 100.
    ap_image = np.zeros((80,img.shape[1]*3,3), np.uint8)
    ap_image = cv2.putText(ap_image, f"mAP: {mAP_score:.2f}   AP50: {AP50_score:.2f}   AP75: {AP75_score:.2f}", (10, 70), cv2.FONT_ITALIC, 1, (255,255,255), 1, cv2.LINE_AA)

    # write name of each img          
    ap_image = cv2.putText(ap_image, "GT", (img.shape[1]*0 + 10, 30), cv2.FONT_ITALIC, 1, (255,255,255), 1, cv2.LINE_AA)
    ap_image = cv2.putText(ap_image, "Pred", (img.shape[1]*1 + 10, 30), cv2.FONT_ITALIC, 1, (255,255,255), 1, cv2.LINE_AA)
    ap_image = cv2.putText(ap_image, "GT + Pred", (img.shape[1]*2 + 10, 30), cv2.FONT_ITALIC, 1, (255,255,255), 1, cv2.LINE_AA)

    img = cv2.hconcat([img_gt, img_pred, img])
    img = cv2.vconcat([img, ap_image])

    return img

class COCOevalMaxDets(COCOeval):
    """
    Modified version of COCOeval for evaluating AP with a custom
    maxDets (by default for COCO, maxDets is 100)
    """

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results given
        a custom value for  max_dets_per_image
        """

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=1000):
            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1]) if iouThr is None else "{:0.2f}".format(iouThr)
            )
            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            # print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            # Evaluate AP using the custom limit on maximum detections per image
            stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
            stats[1] = _summarize(1, iouThr=0.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=0.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng="small", maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng="large", maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng="small", maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng="large", maxDets=self.params.maxDets[2])
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")
        self.stats = _summarizeDets()

    def __str__(self):
        self.summarize()

def Get_PR_curve(coco_eval, iouThr):
    aind = [0] # [all, small, medium large] -> "all"
    mind = [2] # [900, 1000, 1100] -> 1100

    precisions = coco_eval.eval['precision']

    Thrs_idx = np.where(iouThr == coco_eval.params.iouThrs)[0]
    precision = precisions[Thrs_idx][:, :, :, aind, mind]

    if len(precision[precision > -1]) == 0:
        precision = np.array([0. for i in range(101)])

    precision = precision.squeeze()

    return precision

def Draw_PR_curves(coco_eval, PR_curve_vis_path):
    iouThrs = np.array([0.5 + 0.1 * i for i in range(5)])
    recall = np.array([0.01 * i for i in range(101)])
    precisions = []
    for iouThr in iouThrs:
        precision = Get_PR_curve(coco_eval, iouThr)
        precisions.append(precision)

        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0., 1.])
        plt.ylim([0., 1.])

        fig_path = os.path.join(PR_curve_vis_path, f"PR_Curve_{iouThr:.2f}.jpg")
        plt.savefig(fig_path, format="jpeg")
        plt.clf()

    for iouThr, precision in zip(iouThrs, precisions):
        plt.plot(recall, precision, label=f"{iouThr:.2f}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.legend()

    fig_path = os.path.join(PR_curve_vis_path, f"PR_Curve_All.jpg")
    plt.savefig(fig_path, format="jpeg")
    plt.clf()

class Modified_MeanAveragePrecision(MeanAveragePrecision):
    @property
    def cocoeval(self) -> object:
        cocoeval = COCOevalMaxDets
        return cocoeval

def single_image_mean_average_precision_torchmetric(preds, gts):
    metric = Modified_MeanAveragePrecision(box_format='xywh', iou_type='bbox', max_detection_thresholds=[900, 1000, 1100], sync_on_compute=False, dist_sync_on_step=False)

    boxes = []
    scores = []
    labels = []
    for pred in preds:
        x, y, w, h = pred['bbox']
        score = pred['score']
        boxes.append([x, y, w, h])
        scores.append(score)
        labels.append(0)

    predictions = [{
        "boxes": torch.tensor(boxes, dtype=float),
        "scores": torch.tensor(scores, dtype=float),
        "labels": torch.tensor(labels, dtype=int),
    }]

    boxes = []
    labels = []
    for gt in gts:
        x, y, w, h = gt['bbox']
        boxes.append([x, y, w, h])
        labels.append(0)

    targets = [{
        "boxes": torch.tensor(boxes, dtype=float),
        "labels": torch.tensor(labels, dtype=int),
    }]

    metric.update(predictions, targets)
    return metric.compute()