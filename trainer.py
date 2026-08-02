import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F

from pytorch_lightning import LightningModule
import json

from models import build_model
from criterion import build_criterion
from utils.TM_utils import Get_pred_boxes, GT_map, NMS
from utils.box_refine import SAM_box_refiner
from utils.log_utils import image_info_collector, Get_AP_scores, coco_style_annotation_generator, del_img_log_path, Get_MAE_RMSE
from models.naclip_wrapper import NACLIPHeatmap
from utils.clip_utils import CLIPReranker, apply_clip_reranking

class Matching_Trainer(LightningModule):
    def __init__(self, args, datamodule):
        super().__init__()

        self.args = args

        self.model = build_model(args)

        if self.args.finetune_decoders_and_heads_only:
            print("Fine-tuning decoders and heads")

            for name, param in self.model.named_parameters():
                param.requires_grad = False

            trainable_keywords = [
                "decoder_o",
                "decoder_b",
                "objectness_head",
                "ltrbs_head",
                "no_box_token",
                "no_text_token",
            ]

            for name, param in self.model.named_parameters():
                if any(k in name for k in trainable_keywords):
                    param.requires_grad = True

            print("Trainable parameters:")
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(name)

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


        self.naclip = None
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.args.use_naclip_heatmap:
            self.naclip = NACLIPHeatmap(
                clip_path="ViT-B/16",
                device=device,
                arch="reduced",
                attn_strategy="naclip",
                gaussian_std=5.0,
                logit_scale=40,
            )

        # CLIP semantic re-ranking (approach 1), eval only
        self.clip_reranker = None
        if self.args.use_clip:
            if self.args.eval and self.args.text_prompt is not None:
                try:
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
        """Eval-only variant of each_step for multiple exemplars per image.

        Selected in __init__ (it replaces self.each_step) when --num_exemplars > 1
        together with --eval. Each exemplar is run through the model separately
        and the per-exemplar predictions are concatenated before box refinement,
        CLIP re-ranking and NMS, so the exemplars compete in a single NMS pass
        rather than producing independent detection sets. Requires batch_size 1.

        The text path is not exercised here: this variant predates the 1025-channel
        work and calls the model without a naclip_heatmap, so the 1025th channel
        always falls back to the learned no_text_token placeholder. Use each_step
        for anything text-conditioned.

        Args:
            batch: dict with "image", "boxes" and "exemplars".
            stage: one of 'train', 'val', 'test'; gates re-ranking and logging.

        Returns:
            dict with the summed loss under 'loss'.
        """
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

        # Apply CLIP re-ranking before NMS
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
        """Main train/val/test step: the Approach 3 (1025-channel) path.

        Decides the query mode, builds the NaCLIP heatmap for it, runs TMR,
        computes losses, and on val/test decodes boxes and logs them.

        Query mode. During training with --use_modality_dropout, each sample
        draws uniformly over box_and_text / box_only / text_only, which is what
        lets one checkpoint serve all three modes at eval. Otherwise the mode is
        fixed by --input_mode. use_box / use_text are passed to the model, which
        substitutes its learned no_box_token / no_text_token placeholder for
        whichever modality is absent, so the head always sees a complete
        1025-channel tensor.

        Text conditioning. When --use_naclip_heatmap is set and text is in play,
        the per-sample label from batch["label"] is turned into a dense
        [B, 1, H, W] heatmap by a two-class softmax against "background". The
        heatmap is built per sample in a loop, so this scales linearly with
        batch size and dominates step time.

        Note the asymmetry: use_text=False skips NaCLIP entirely, but
        --use_naclip_heatmap=False also skips it even in a text mode, in which
        case text_only degenerates to the placeholder alone.

        Args:
            batch: dict with "image", "boxes", "exemplars" and "label".
            stage: one of 'train', 'val', 'test'. Only val/test decode boxes,
                run CLIP re-ranking and NMS, and write image logs.

        Returns:
            dict with the summed loss under 'loss'.
        """
        image = batch["image"]

        # Ground-truth bounding boxes of ALL target objects in the image
        # Used for supervision/loss computation
        gt_boxes = batch['boxes']

        # Support exemplar box(es)
        # These define WHAT pattern/object TMR should search for
        exemplars = batch["exemplars"]

        if stage == "train" and self.args.use_modality_dropout:
            r = torch.rand(1).item()

            if r < 1/3:
                use_box = True
                use_text = True
            elif r < 2/3:
                use_box = True
                use_text = False
            else:
                use_box = False
                use_text = True
        else:
            use_box = self.args.input_mode in ["box_only", "box_and_text"]
            use_text = self.args.input_mode in ["text_only", "box_and_text"]

        naclip_heatmap = None

        if self.args.use_naclip_heatmap and use_text:
            heatmaps = []

            for i, label in enumerate(batch["label"]):
                heatmap_i = self.naclip.target_heatmap(
                    image[i:i+1],
                    class_names=["background", label],
                    target_idx=1,
                    out_size=None
                )
                heatmaps.append(heatmap_i)

            naclip_heatmap = torch.cat(heatmaps, dim=0).to(image.device)

        # Ablation settings for different regression experiments
        # a: disable learned box regression completely
        # b: scale regression with image size
        # c: scale only width/height
        batch['regression_ablation_a'] = self.args.ablation_no_box_regression
        batch['regression_ablation_b'] = self.args.regression_scaling_imgsize
        batch['regression_ablation_c'] = self.args.regression_scaling_WH_only


        # Main TMR forward pass
        #
        # pred_objectness:
        # Dense presence/confidence map
        # Shape roughly: [B, 1, H, W]
        # Predicts how likely the exemplar pattern exists at each spatial location
        #
        # pred_regressions:
        # Dense box regression map
        # Shape roughly: [B, 4, H, W]
        # Predicts box offsets/scaling for each location
        #
        # matching_feature:
        # Template matching feature map F_TM
        # Represents similarity between image regions and exemplar feature
        #
        # _:

        pred_objectness, pred_regressions, matching_feature, _ = self.model(image,exemplars,naclip_heatmap=naclip_heatmap,use_box=use_box,use_text=use_text)

        # Convert predictions + GT boxes into loss-compatible format
        #
        # preds:
        # Model predictions formatted for criterion/loss
        #
        # gts:
        # Ground-truth presence maps + regression targets
        #
        # vis_gt_map:
        # Visualization version of GT presence map
        preds, gts, vis_gt_map = self.GT_map_generator.Get_pred_gts(
            pred_objectness,
            pred_regressions,
            gt_boxes,
            exemplars,
            batch
        )

        # Compute TMR losses
        #
        # loss_ce:
        # Presence/objectness classification loss
        #
        # loss_giou:
        # Bounding box regression loss
        loss_dict = self.criterion(preds, gts)


        # Final total loss used for backpropagation
        # TMR loss = classification loss + regression loss
        loss_dict['loss'] = loss_dict['loss_ce'] + loss_dict['loss_giou']

        new_loss_dict = {}
        for key in loss_dict.keys():
            new_loss_dict[f"{stage}/{key}"] = loss_dict[key]

        # During validation/testing:
        # Convert dense TMR prediction maps into actual bounding boxes
        # pred_logits:
        # Confidence/objectness scores for predicted boxes
        # pred_boxes:
        # Final predicted bounding boxes in xyxy format
        # ref_points:
        # Spatial reference points on the feature map from which boxes are decoded
        # Get_pred_boxes converts:
        #   pred_objectness [B,1,H,W]
        #   pred_regressions [B,4,H,W]
        # into:
        #   real image-space bounding boxes
        # It also removes low-confidence predictions using NMS_cls_threshold
        if (self.AP_log and stage == 'val') or stage == 'test':

            pred_logits, pred_boxes, ref_points = Get_pred_boxes(
                pred_objectness,
                pred_regressions,
                exemplars,
                batch,
                self.args.NMS_cls_threshold,
                not batch['regression_ablation_a']
            )

            # Optional SAM decoder refinement step
            # TMR already predicted boxes before this step
            # SAM does NOT create new boxes here
            # It only refines/improves existing TMR box coordinates
            # backbone_feature:
            # Feature map extracted from SAM backbone
            # refiner():
            # Uses SAM decoder to adjust box boundaries
            if self.args.refine_box:

                backbone_feature = self.temp_sam(image)

                pred_logits, pred_boxes, ref_points = self.refiner(
                    pred_logits,
                    pred_boxes,
                    ref_points,
                    image,
                    backbone_feature
                )

            # Apply CLIP re-ranking before NMS
            if self.clip_reranker is not None and stage in ['val', 'test']:
                pred_logits, pred_boxes, ref_points = apply_clip_reranking(
                    pred_logits,
                    pred_boxes,
                    ref_points,
                    image,
                    self.args.text_prompt,
                    self.clip_reranker,
                    negative_prompt=self.args.negative_prompt,
                    alpha=self.args.clip_alpha,
                    beta=self.args.clip_beta,
                    top_k=self.args.clip_topk,
                    threshold=self.args.clip_threshold
                )

            # Non-Maximum Suppression (NMS)
            #
            # Removes duplicate overlapping bounding boxes
            #
            # Keeps boxes with highest confidence score
            # Removes lower-scoring overlapping boxes
            #
            pred_logits, pred_boxes, ref_points = NMS(
                pred_logits,
                pred_boxes,
                ref_points,
                self.args.NMS_iou_threshold
            )


            # Save final predictions/logs for:
            #   AP / AP50 / AP75 evaluation
            #   MAE / RMSE computation
            #   visualization
            #   COCO-style evaluation
            image_info_collector(
                self.args.logpath,
                stage,
                batch,
                pred_logits,
                pred_boxes,
                ref_points
            )
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