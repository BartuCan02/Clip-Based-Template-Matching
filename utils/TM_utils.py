import math
import torch
import numpy as np
import torch.nn.functional as F

from torchvision.ops import nms
from torchvision.ops.boxes import box_area

def calc_area(box):
    x1, y1, x2, y2 = box
    return (x2-x1)*(y2-y1)

def Make_Template_size_predictions(centers):
    xy = torch.zeros_like(centers)
    wh = torch.zeros_like(centers)

    box_regression = torch.concat([xy, wh], dim=1)
    return box_regression

class GT_map():
    def __init__(self, args):
        self.positive_threshold = args.positive_threshold
        self.negative_threshold = args.negative_threshold
        self.box_reg = not args.ablation_no_box_regression

    def get_template(self, h, w, device, is_center = True):
        xs = (torch.arange(0, w, step=1, dtype=torch.float32, device=device) + (0.5 if is_center else 0.)) / w
        ys = (torch.arange(0, h, step=1, dtype=torch.float32, device=device) + (0.5 if is_center else 0.)) / h
        ys, xs = torch.meshgrid(ys, xs, indexing='ij')
        xs = xs.reshape(-1)
        ys = ys.reshape(-1)
        template = torch.stack([xs, ys], dim=1)

        return template
    
    def Get_not_in_boundary(self, H, W, exemplar, device):
        x1, y1, x2, y2 = exemplar
        x1, y1 = min(1., max(0., x1)), min(1., max(0., y1))
        x2, y2 = min(1., max(0., x2)), min(1., max(0., y2))

        x1, x2 = math.floor(x1*W), math.ceil(x2*W)
        y1, y2 = math.floor(y1*H), math.ceil(y2*H)

        if (x2-x1) % 2 == 0: x2 -= 1
        if (y2-y1) % 2 == 0: y2 -= 1

        pad_x = (x2 - x1) // 2
        pad_y = (y2 - y1) // 2

        not_in_boundary = torch.zeros((H, W), dtype=torch.bool, device=device)
        not_in_boundary[pad_y:H-pad_y, pad_x:W-pad_x] = True
        not_in_boundary = not_in_boundary.reshape(-1)

        return not_in_boundary

    def Get_is_center(self, centers, boxes, device):
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        cx, cy = (x2 + x1) / 2, (y2 + y1) / 2
        cxs, cys = centers[:, 0], centers[:, 1]

        relative_x = torch.abs(cxs[:, None] - cx[None])
        relative_y = torch.abs(cys[:, None] - cy[None])

        is_center = torch.zeros_like(relative_x, dtype=torch.bool, device=device)
        now_centers = torch.argmin(relative_x + relative_y, dim=0)
        is_center[now_centers, range(len(now_centers))] = True
        return is_center

    def Get_cxcywh(self, boxes):
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        cx, cy = (x2 + x1) / 2, (y2 + y1) / 2
        w, h = x2 - x1, y2 - y1

        xywh = torch.stack([cx, cy, w, h], dim=1)
        return xywh

    def Get_is_in_out_positive(self, centers, boxes, device=None):
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        cx, cy = (x2 + x1) / 2, (y2 + y1) / 2
        ws, hs = x2 - x1, y2 - y1
        cxs, cys = centers[:, 0], centers[:, 1]

        relative_x = torch.abs(cxs[:, None] - cx[None])
        relative_y = torch.abs(cys[:, None] - cy[None])

        ratio = -hs / ws
        bias_p = ((1 - self.positive_threshold) / (1 + self.positive_threshold)) * hs
        bias_n = ((1 - self.negative_threshold) / (1 + self.negative_threshold)) * hs

        is_in_positive = ratio * relative_x + bias_p >= relative_y
        is_in_negative = ratio * relative_x + bias_n < relative_y
        return is_in_positive, is_in_negative

    def Get_pred_gts(self, pred_objectness, pred_regressions, gt_boxes, exemplars, batch):
        preds = {
            "objectness": [],
            "regressions": [],
        }
        gts = {
            "objectness": [],
            "regressions": [],
            "weight_objectness": [],
            "weight_regressions": [],
        }
        gt_maps = []

        num_layer = len(pred_objectness)
        for layer_idx, (objectness_map, regressions_map) in enumerate(zip(pred_objectness, pred_regressions)):
            preds_layer = {
                "objectness": [],
                "regressions": [],
            }
            gts_layer = {
                "objectness": [],
                "regressions": [],
                "weight_objectness": [],
                "weight_regressions": [],
            }
            gt_maps_layer = []
            weight_maps_layer = []

            _, _, H, W = objectness_map.shape

            centers = self.get_template(H, W, device=objectness_map.device, is_center=False)
            cxs, cys = centers[:, 0], centers[:, 1]

            for bidx, boxes in enumerate(gt_boxes):
                ex1, ey1, ex2, ey2 = exemplars[bidx][0]
                ex1, ey1 = min(1., max(0., ex1)), min(1., max(0., ey1))
                ex2, ey2 = min(1., max(0., ex2)), min(1., max(0., ey2))
                ex_w, ex_h = ex2 - ex1, ey2 - ey1
                ex_w_h, ex_h_h = (ex2-ex1)/2, (ey2-ey1)/2

                if batch['regression_ablation_b']:
                    ex_w, ex_h = 1., 1.

                with torch.no_grad():
                    is_center = self.Get_is_center(centers, boxes, objectness_map.device)
                    GT_xywh = self.Get_cxcywh(boxes)
                    try:
                        is_in_positive, is_in_negative = self.Get_is_in_out_positive(centers, boxes, objectness_map.device)
                    except:
                        is_in_positive = is_center
                        is_in_negative = ~is_center

                    if self.positive_threshold == 1.0: is_in_positive = is_center
                    if self.negative_threshold == 1.0: is_in_negative = ~is_center

                    not_in_boundary = self.Get_not_in_boundary(H, W, exemplars[bidx][0], objectness_map.device)
                    not_in_boundary = not_in_boundary[:,None].repeat(1,len(boxes))

                    if layer_idx == num_layer - 1:
                        is_center_or_in_positive = (is_center == True) | (is_in_positive == True)
                    else:
                        is_center_or_in_positive = (is_in_positive == True)

                    is_in_negative = is_in_negative | (is_center_or_in_positive & ~not_in_boundary)
                    is_center_or_in_positive = is_center_or_in_positive & not_in_boundary

                    # box selection
                    area = torch.tensor(list(map(calc_area, boxes)), dtype=torch.float32, device=objectness_map.device)
                    box_area_of_locations = area[None].repeat(len(centers), 1)
                    box_area_of_locations[is_center_or_in_positive == False] = 100000000.
                    _, box_targets_id = torch.min(box_area_of_locations, dim=1) # choose smallest area box
                    box_targets = GT_xywh[box_targets_id].to(torch.float32).to(objectness_map.device)

                    # map refinary
                    positive_map = torch.max(is_center_or_in_positive, dim=1)[0].reshape(H, W)
                    ignore_map = (torch.max(is_center_or_in_positive == False, dim=1)[0] & torch.max(is_in_negative == False, dim=1)[0] & torch.max(not_in_boundary, dim=1)[0]).reshape(H, W)
                    negative_map = ~(positive_map | ignore_map)

                pred_positive = objectness_map[bidx][positive_map.unsqueeze(0)]
                pred_negative = objectness_map[bidx][negative_map.unsqueeze(0)]
                gt_positive = torch.ones_like(pred_positive)
                gt_negative = torch.zeros_like(pred_negative)

                if self.box_reg:
                    now_pred_regression = regressions_map[bidx]
                    now_pred_regression = now_pred_regression.permute(1,2,0)
                else:
                    now_pred_regression = Make_Template_size_predictions(centers).reshape(H,W,4)

                if batch['regression_ablation_c']:
                    pred_xy = centers.reshape(H,W,2) + now_pred_regression[...,:2] * torch.tensor([1.,1.], dtype=now_pred_regression.dtype, device=now_pred_regression.device)
                else:
                    pred_xy = centers.reshape(H,W,2) + now_pred_regression[...,:2] * torch.tensor([ex_w,ex_h], dtype=now_pred_regression.dtype, device=now_pred_regression.device)
                pred_wh = torch.exp(now_pred_regression[...,2:]) * torch.tensor([ex_w,ex_h], dtype=now_pred_regression.dtype, device=now_pred_regression.device)
                
                pred_xywh = torch.concat([pred_xy, pred_wh], dim=-1)
                gt_xywh = box_targets.reshape(H,W,4)

                pred_xywh = pred_xywh[positive_map]
                gt_xywh = gt_xywh[positive_map]
                
                gt_map = positive_map * 0.5 + (negative_map * -0.5 + 0.5)
                gt_map = gt_map.unsqueeze(0)

                pred_sample = torch.concat([pred_positive, pred_negative])
                gt_sample = torch.concat([gt_positive, gt_negative])

                if pred_xywh.shape[0] == 0:
                    pred_xywh = torch.tensor([[0., 0., 1e-14, 1e-14]], dtype=pred_xywh.dtype, device=pred_xywh.device)
                    gt_xywh = torch.tensor([[0., 0., 1e-14, 1e-14]], dtype=gt_xywh.dtype, device=gt_xywh.device)

                preds_layer['objectness'].append(pred_sample)
                preds_layer['regressions'].append(pred_xywh)
                gts_layer['objectness'].append(gt_sample)
                gts_layer['regressions'].append(gt_xywh)
                gts_layer['weight_objectness'].append(None)
                gts_layer['weight_regressions'].append(None)
                gt_maps_layer.append(gt_map)
                weight_maps_layer.append(None)

            preds['objectness'].append(torch.concat(preds_layer['objectness']))
            preds['regressions'].append(torch.concat(preds_layer['regressions']))
            gts['objectness'].append(torch.concat(gts_layer['objectness']))
            gts['regressions'].append(torch.concat(gts_layer['regressions']))
            gts['weight_objectness'].append(None)
            gts['weight_regressions'].append(None)
            gt_maps.append(torch.stack(gt_maps_layer))

        return preds, gts, gt_maps

def Get_pred_boxes(pred_objectness, pred_regressions, exemplars, batch, cls_ths = 0.1, box_reg = True):  
    cls_threshold = cls_ths

    pred_boxes, ref_points, pred_logits = [], [], []

    dtype = pred_objectness[-1].dtype
    device = pred_objectness[-1].device

    for bidx in range(len(pred_objectness[0])):
        bboxes, refs, logits = [], [], []

        # box size setting (cuz. box regression is not contained in this setting)
        x1, y1, x2, y2 = exemplars[bidx][0]
        x1, y1 = min(1., max(0., x1)), min(1., max(0., y1))
        x2, y2 = min(1., max(0., x2)), min(1., max(0., y2))
        box_w, box_h = x2 - x1, y2 - y1
        box_w_h, box_h_h = (x2-x1)/2, (y2-y1)/2

        if batch['regression_ablation_b']:
            box_w, box_h = 1., 1.

        for level in range(len(pred_objectness)):
            pred = pred_objectness[level][bidx].sigmoid()
            pred = pred.squeeze(0)

            H, W = pred.shape
            img_res = torch.tensor([W, H], dtype=dtype, device=device)

            kernel = adaptive_kernel_generater([y2-y1, x2-x1], [H, W])
            local_peak = (custom_shape_3x3_maxpool2d(pred[None,None,...], kernel) == pred[None,None,...])[0,0]
            centers = torch.where((pred >= cls_threshold) & local_peak)

            # refs
            now_refs = torch.flip(torch.stack(centers).T, dims=(-1,)) / img_res # normalized [[cx, cy], ...] format

            # logits
            now_logits = pred[centers]
            now_logits = torch.stack([now_logits, torch.zeros_like(now_logits)]).T

            # boxes
            if box_reg:
                regresisons = pred_regressions[level][bidx]
                regresisons = regresisons.permute(1,2,0)

                if batch['regression_ablation_c']:
                    pred_xy = now_refs + regresisons[centers][...,:2] * torch.tensor([1.,1.], dtype=regresisons.dtype, device=regresisons.device)
                else:
                    pred_xy = now_refs + regresisons[centers][...,:2] * torch.tensor([box_w,box_h], dtype=regresisons.dtype, device=regresisons.device)
                pred_wh = torch.exp(regresisons[centers][...,2:]) * torch.tensor([box_w,box_h], dtype=regresisons.dtype, device=regresisons.device)
            else:
                regresisons = Make_Template_size_predictions(now_refs)
                pred_xy = now_refs + regresisons[...,:2] * torch.tensor([box_w,box_h], dtype=regresisons.dtype, device=regresisons.device)
                pred_wh = torch.exp(regresisons[...,2:]) * torch.tensor([box_w,box_h], dtype=regresisons.dtype, device=regresisons.device)
            pred_xywh = torch.concat([pred_xy, pred_wh], dim=-1)
            now_bboxes = torch.concat([pred_xywh[:, :2] - pred_xywh[:, 2:] / 2, pred_xywh[:, :2] + pred_xywh[:, 2:] / 2], dim=1)       

            bboxes.append(now_bboxes)
            refs.append(now_refs)
            logits.append(now_logits)
        
        bboxes = torch.concat(bboxes, dim=0)
        refs = torch.concat(refs, dim=0)
        logits = torch.concat(logits, dim=0)

        if len(bboxes) == 0:
            pred_logits.append(torch.tensor([[0., 0.]], dtype=dtype, device=device))
            pred_boxes.append(torch.tensor([[0., 0., 1e-14, 1e-14]], dtype=dtype, device=device))
            ref_points.append(torch.tensor([[0., 0.]], dtype=dtype, device=device))
        else:
            pred_logits.append(logits)
            pred_boxes.append(bboxes)
            ref_points.append(refs)

    pred_logits = [torch.as_tensor(pred_logits[i], dtype=dtype, device=device) for i in range(len(pred_logits))]
    pred_boxes = [torch.as_tensor(pred_boxes[i], dtype=dtype, device=device) for i in range(len(pred_boxes))]
    ref_points = [torch.as_tensor(ref_points[i], dtype=dtype, device=device) for i in range(len(ref_points))]

    # pred_logits # bs size list of tensor (n, 2) for probability of (object, nonobject)
    # pred_boxes # bs size list of tensor (n, 4) for box coord (x_min, y_min, x_max, y_max) in 0~1 range
    # ref_points # bs size list of tensor (n, 2) (a, b) any value 0~1

    return pred_logits, pred_boxes, ref_points

def NMS(pred_logits, pred_boxes, ref_points, iou_threshold = 0.15):
    # NMS
    for bidx in range(len(pred_logits)):
        nms_idxes = NMS_process(pred_boxes[bidx], pred_logits[bidx], iou_threshold=iou_threshold)
        pred_logits[bidx] = pred_logits[bidx][nms_idxes]
        pred_boxes[bidx] = pred_boxes[bidx][nms_idxes]
        ref_points[bidx] = ref_points[bidx][nms_idxes]

    return pred_logits, pred_boxes, ref_points

def NMS_process(boxes, logits, iou_threshold = 0.65):
    # logits align
    object_prob = logits[..., 0]

    xyxy_boxes = boxes
    nms_idxes = nms(xyxy_boxes, object_prob, iou_threshold=iou_threshold)
    return nms_idxes

def map_normalization(img):
    if torch.is_tensor(img):
        maxv = torch.max(img)
        minv = torch.min(img)
    else:
        maxv = np.max(img)
        minv = np.min(img)

    img = (img - minv) / (maxv - minv + 1e-14)

    return img

def custom_shape_3x3_maxpool2d(x: torch.Tensor, kernel: list) -> torch.Tensor:
    """
    Perform a custom 3x3 max pooling on input x (N, C, H, W).
    EX. The cross shape is:
            0 1 0
            1 1 1
            0 1 0
    """
    # Define the cross mask (3x3) as a boolean tensor
    # 1 (True) means "use this position"
    # 0 (False) means "ignore this position"
    mask_3x3 = torch.tensor(kernel, dtype=torch.bool, device=x.device)  # put on same device as x
    
    flat_mask = mask_3x3.flatten()
    patches = F.unfold(x, kernel_size=3, padding=1)

    N, C_times_9, HW = patches.shape
    H, W = x.shape[2], x.shape[3]
    patches = patches.view(N, -1, 9, HW)  # -> (N, C, 9, H*W)
    # Next step: separate the H*W dimension
    patches = patches.view(N, -1, 9, H, W)  # -> (N, C, 9, H, W)
    selected_positions = patches[:, :, flat_mask, :, :]  # (N, C, 5, H, W) if 5 positions are True
    pooled = selected_positions.max(dim=2)[0]  # shape (N, C, H, W)

    return pooled

def adaptive_kernel_generater(ex_size, pred_size):
    needy_size_h, needy_size_w = 1 / pred_size[0], 1 / pred_size[1]
    ex_h, ex_w = ex_size

    if ex_h >= (needy_size_h * 3) and ex_w >= (needy_size_w * 3):
        return [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    else:
        if ex_h < (needy_size_h * 2) and ex_w < (needy_size_w * 2):
            return [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        elif ex_h < (needy_size_h * 2) and ex_w >= (needy_size_w * 2):
            return [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
        elif ex_h >= (needy_size_h * 2) and ex_w < (needy_size_w * 2):
            return [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        else:
            return [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    
