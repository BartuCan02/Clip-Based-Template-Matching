import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F

from pytorch_lightning import LightningModule

from models import build_model
from criterion import build_criterion
from utils.TM_utils import Get_pred_boxes, GT_map, NMS
from utils.box_refine import SAM_box_refiner
from utils.log_utils import image_info_collector, Get_AP_scores, coco_style_annotation_generator, del_img_log_path, Get_MAE_RMSE
from utils.clip_utils import CLIPReranker, apply_clip_reranking

class Matching_Trainer(LightningModule):
    def __init__(self, args, datamodule):
        super().__init__()

        self.args = args

        self.model = build_model(args)
        self.criterion = build_criterion(args)
        self.datamodule = datamodule

        self.GT_map_generator = GT_map(args)

        self.AP_term = args.AP_term
        self.AP_log = False
        self.result_log = {'train': None, 'val': None, 'test': None}

        if self.args.num_exemplars > 1:
            if self.args.eval:
                self.each_step = self.each_step_multi_exemplars
            else:
                raise ValueError("Multi-exemplar testing is only available in evaluation mode.")

        self.refiner = None
        if self.args.refine_box:
            if self.args.eval:
                from models.backbone.sam.sam import Sam_Backbone
                self.temp_sam = Sam_Backbone(requires_grad=False, model_type = "vit_h")
                self.refiner = SAM_box_refiner() 
            else:
                raise ValueError("SAM decoder box refinement is only available in evaluation mode.")

        # CLIP initialization
        self.clip_reranker = None
        if self.args.use_clip:
            if self.args.eval and self.args.text_prompt is not None:
                try:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self.clip_reranker = CLIPReranker(model_name=self.args.clip_model, device=device)
                    print(f"CLIP reranker initialized with model: {self.args.clip_model}")
                    print(f"Positive prompt: '{self.args.text_prompt}'")
                    if self.args.negative_prompt:
                        print(f"Negative prompt: '{self.args.negative_prompt}'")
                except Exception as e:
                    print(f"Failed to initialize CLIP reranker: {e}")
                    self.clip_reranker = None
            elif not self.args.eval:
                print("CLIP reranking is only available in evaluation mode.")
            elif self.args.text_prompt is None:
                print("Text prompt is required for CLIP reranking. Use --text_prompt argument.")

    def training_step(self, batch, batch_idx):
        return self.each_step(batch, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.each_step(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        return self.each_step(batch, 'test')
    
    def on_train_epoch_end(self):
        self.result_log['train'] = self.each_epoch_end(stage='train')
        if self.result_log['train'] != None and self.result_log['val'] != None:
            print(self.result_log['train'] + '\n' + self.result_log['val'])

    def on_validation_epoch_end(self):
        self.result_log['val'] = self.each_epoch_end(stage='val')

    def on_test_epoch_end(self):
        self.result_log['test'] = self.each_epoch_end(stage='test')
        if self.result_log['test'] != None:
            print(self.result_log['test'])
    
    def on_train_epoch_start(self):
        epoch = self.trainer.current_epoch
        if (epoch == 0) or (epoch % self.AP_term == (self.AP_term-1)):
            self.AP_log = True
        else:
            self.AP_log = False

    def each_step_multi_exemplars(self, batch, stage):
        image = batch["image"]
        gt_boxes = batch['boxes']
        multi_exemplars = batch["exemplars"]

        if len(multi_exemplars) != 1:
            raise ValueError("Multi-exemplar testing is only available for batchsize == 1.")

        batch['regression_ablation_a'] = self.args.ablation_no_box_regression
        batch['regression_ablation_b'] = self.args.regression_scaling_imgsize
        batch['regression_ablation_c'] = self.args.regression_scaling_WH_only

        losses = {
            'loss_ce': [],
            'loss_giou': [],
            'loss': []
        }
        pred_logits = []
        pred_boxes = []
        ref_points = []
        multi_exemplars = [[exemplars.unsqueeze(0)] for exemplars in multi_exemplars[0]]
        for exemplars in multi_exemplars:
            pred_objectness, pred_regressions, matching_feature, _ = self.model(image, exemplars)
            preds, gts, vis_gt_map = self.GT_map_generator.Get_pred_gts(pred_objectness, pred_regressions, gt_boxes, exemplars, batch)

            loss_dict = self.criterion(preds, gts)
            loss_dict['loss'] = loss_dict['loss_ce'] + loss_dict['loss_giou']
            losses['loss_ce'].append(loss_dict['loss_ce'])
            losses['loss_giou'].append(loss_dict['loss_giou'])
            losses['loss'].append(loss_dict['loss'])

            _pred_logits, _pred_boxes, _ref_points = Get_pred_boxes(pred_objectness, pred_regressions, exemplars, batch, self.args.NMS_cls_threshold, not batch['regression_ablation_a'])
            pred_logits.append(_pred_logits[0])
            pred_boxes.append(_pred_boxes[0])
            ref_points.append(_ref_points[0])

        pred_logits = [torch.concat(pred_logits)]
        pred_boxes = [torch.concat(pred_boxes)]
        ref_points = [torch.concat(ref_points)]
        
        if self.args.refine_box:
            backbone_feature = self.temp_sam(image)
            pred_logits, pred_boxes, ref_points = self.refiner(pred_logits, pred_boxes, ref_points, image, backbone_feature)
        
        # Apply CLIP reranking before NMS
        if self.clip_reranker is not None and stage in ['val', 'test']:
            pred_logits, pred_boxes, ref_points = apply_clip_reranking(
                pred_logits, pred_boxes, ref_points, image,
                self.args.text_prompt, self.clip_reranker,
                negative_prompt=self.args.negative_prompt,
                alpha=self.args.clip_alpha,
                beta=self.args.clip_beta,
                top_k=self.args.clip_topk,
                threshold=self.args.clip_threshold
            )
        
        pred_logits, pred_boxes, ref_points = NMS(pred_logits, pred_boxes, ref_points, self.args.NMS_iou_threshold)
        image_info_collector(self.args.logpath, stage, batch, pred_logits, pred_boxes, ref_points)

        return {'loss': sum(losses['loss'])}

    def each_step(self, batch, stage):
        image = batch["image"]
        gt_boxes = batch['boxes']
        exemplars = batch["exemplars"]

        batch['regression_ablation_a'] = self.args.ablation_no_box_regression
        batch['regression_ablation_b'] = self.args.regression_scaling_imgsize
        batch['regression_ablation_c'] = self.args.regression_scaling_WH_only

        pred_objectness, pred_regressions, matching_feature, _ = self.model(image, exemplars)
        preds, gts, vis_gt_map = self.GT_map_generator.Get_pred_gts(pred_objectness, pred_regressions, gt_boxes, exemplars, batch)


        loss_dict = self.criterion(preds, gts)
        loss_dict['loss'] = loss_dict['loss_ce'] + loss_dict['loss_giou']

        new_loss_dict = {}
        for key in loss_dict.keys():
            new_loss_dict[f"{stage}/{key}"] = loss_dict[key]

        if (self.AP_log and stage == 'val') or stage == 'test':
            pred_logits, pred_boxes, ref_points = Get_pred_boxes(pred_objectness, pred_regressions, exemplars, batch, self.args.NMS_cls_threshold, not batch['regression_ablation_a'])

            if self.args.refine_box:
                backbone_feature = self.temp_sam(image)
                pred_logits, pred_boxes, ref_points = self.refiner(pred_logits, pred_boxes, ref_points, image, backbone_feature)
            
            # Apply CLIP reranking before NMS
            if self.clip_reranker is not None and stage in ['val', 'test']:
                pred_logits, pred_boxes, ref_points = apply_clip_reranking(
                    pred_logits, pred_boxes, ref_points, image,
                    self.args.text_prompt, self.clip_reranker,
                    negative_prompt=self.args.negative_prompt,
                    alpha=self.args.clip_alpha,
                    beta=self.args.clip_beta,
                    top_k=self.args.clip_topk,
                    threshold=self.args.clip_threshold
                )
            
            pred_logits, pred_boxes, ref_points = NMS(pred_logits, pred_boxes, ref_points, self.args.NMS_iou_threshold)
            image_info_collector(self.args.logpath, stage, batch, pred_logits, pred_boxes, ref_points)

        self.log_dict(new_loss_dict, on_step=False, on_epoch=True, sync_dist=True if self.args.multi_gpu else False, batch_size=self.args.batch_size)
        return {'loss': loss_dict['loss']}

    def print_presence_map(self, img_names, pred_map, gt_map, stage):
        pred_path = os.path.join(self.args.logpath, 'Debug_presence_pred')
        gt_path = os.path.join(self.args.logpath, 'Debug_presence_gt')
        os.makedirs(pred_path, exist_ok=True)
        os.makedirs(gt_path, exist_ok=True)

        pred_map = [pred.sigmoid() for pred in pred_map]
        for l in range(len(pred_map)):
            for bi in range(len(pred_map[l])):
                P = pred_map[l][bi].permute(1,2,0).detach().cpu().numpy()
                P = (P * 254.).astype(np.uint8)
                G = gt_map[l][bi].permute(1,2,0).detach().cpu().numpy()
                G = (G * 254.).astype(np.uint8)

                cv2.imwrite(os.path.join(pred_path, f"pred_{l}_{img_names[bi]}_{stage}.jpg"), P)
                cv2.imwrite(os.path.join(gt_path, f"gt_{l}_{img_names[bi]}.jpg"), G)

    def each_epoch_end(self, stage):
        epoch = self.trainer.current_epoch
        result = None

        if self.trainer.global_rank == 0:
            metrics = self.trainer.logged_metrics
            result = f"Epoch {epoch}:"
            result = result + " | " + " | ".join([f"{key.split('_epoch')[0]}: {metrics[key]:.4f}" for key in metrics.keys() if ((stage in key) and ('step' not in key) and ('AP' not in key))])

        if ((self.AP_log and stage == 'val') or stage == 'test'):
            self.trainer.strategy.barrier()

            if self.trainer.global_rank == 0:
                coco_style_annotation_generator(self.args.logpath, stage)

            self.trainer.strategy.barrier()

            MAE, RMSE = Get_MAE_RMSE(self.args.logpath, stage)
            AP, AP50, AP75 = Get_AP_scores(self.args.logpath, stage, self.args.visualize)

            self.log(f'{stage}/AP', AP, sync_dist=True if self.args.multi_gpu else False)
            self.log(f'{stage}/AP50', AP50, sync_dist=True if self.args.multi_gpu else False)
            self.log(f'{stage}/AP75', AP75, sync_dist=True if self.args.multi_gpu else False)

            self.log(f'{stage}/MAE', MAE, sync_dist=True if self.args.multi_gpu else False)
            self.log(f'{stage}/RMSE', RMSE, sync_dist=True if self.args.multi_gpu else False)

            self.trainer.strategy.barrier()

            if self.trainer.global_rank == 0:
                result += f"\nEpoch {epoch}: | {stage}/AP: {AP:.2f} | {stage}/AP50: {AP50:.2f} | {stage}/AP75: {AP75:.2f}"
                result += f" | {stage}/MAE: {MAE:.2f} | {stage}/RMSE: {RMSE:.2f}"
                del_img_log_path(self.args.logpath, stage)

        return result

    def configure_optimizers(self):

        param_dicts = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not match_name_keywords(n, ['backbone']) and p.requires_grad
                ],
                "lr": self.args.lr
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if match_name_keywords(n, ['backbone']) and p.requires_grad
                ],
                "lr": self.args.lr_backbone
            }
        ]
        
        milestones = []
        if self.args.lr_drop:
            milestones = [int(self.args.max_epochs * 0.6)]
        else:
            milestones = [self.args.max_epochs + 1]

        optimizer = torch.optim.AdamW(param_dicts, lr=self.args.lr, weight_decay=self.args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out