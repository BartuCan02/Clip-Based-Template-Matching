import torch
from torch import nn
from torch.nn import functional as F
from .segment_anything.modeling import MaskDecoder, TwoWayTransformer, PromptEncoder

def xyxy_to_ltrb(box):
    cx, cy = (box[:,0] + box[:,2]) / 2, (box[:,1] + box[:,3]) / 2
    l, t, r, b = cx - box[:,0], cy - box[:,1], box[:,2] - cx, box[:,3] - cy
    
    ltrb = torch.stack([l, t, r, b], dim=-1)
    center = torch.stack([cx, cy], dim=-1)
    return ltrb, center


def ltrb_to_xyxy(ltrb, center):
    cx, cy = center[:,0], center[:,1]
    x1, y1, x2, y2 = cx - ltrb[:,0], cy - ltrb[:,1], cx + ltrb[:,2], cy + ltrb[:,3]

    xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
    return xyxy

class SAM_box_refiner(nn.Module):
    def __init__(self):
        super(SAM_box_refiner, self).__init__()

        self.step = 50
        self.prompt_embed_dim = 256
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        checkpoint = torch.hub.load_state_dict_from_url(
            'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
            map_location="cpu"
        )
        state_dict = {k.replace("mask_decoder.", ""): v for k, v in checkpoint.items() if "mask_decoder" in k}
        self.mask_decoder.load_state_dict(state_dict)

    def prompt_encoder_init(self, image_size, emb_size):
        prompt_encoder_sam = PromptEncoder(
            embed_dim=self.prompt_embed_dim,
            image_embedding_size=emb_size,
            input_image_size=image_size,
            mask_in_chans=16,
        )
        checkpoint = torch.hub.load_state_dict_from_url(
            'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
            map_location="cpu"
        )
        state_dict = {k.replace("prompt_encoder.", ""): v for k, v in checkpoint.items() if "prompt_encoder" in k}
        prompt_encoder_sam.load_state_dict(state_dict)

        return prompt_encoder_sam

    def forward_refine(self, pred_logits, pred_boxes, ref_points, image, exemplars, features):
        new_pred_logits = []
        new_pred_boxes = []
        new_pred_ref_points = []

        for bidx in range(len(pred_logits)):
            if len(pred_boxes[bidx]) == 0:
                new_pred_logits.append(pred_logits[bidx])
                new_pred_boxes.append(pred_boxes[bidx])
                new_pred_ref_points.append(ref_points[bidx])
                continue

            emb_shape = features[bidx].shape[-2:]
            img_shape = image[bidx].shape[-2:]
            img_h, img_w = img_shape
            img_res = torch.tensor([img_w, img_h, img_w, img_h], dtype=image.dtype, device=image.device)

            prompt_encoder_sam = self.prompt_encoder_init(img_shape, emb_shape).to(image.device)

            # 1. exemplar processing
            x1, y1, x2, y2 = exemplars[bidx][0]
            ex_box = torch.tensor([[x1, y1, x2, y2]], dtype=image.dtype, device=image.device) * img_res # xyxy

            # Prepare prompt embeddings
            sparse_embeddings, dense_embeddings = prompt_encoder_sam(
                points=None,
                boxes=ex_box,
                masks=None,
            )

            # SAM mask decoder
            masks_, iou_predictions_ = self.mask_decoder(
                image_embeddings=features[bidx].unsqueeze(0),
                image_pe=prompt_encoder_sam.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks_ = F.interpolate(masks_, img_shape, mode="bilinear", align_corners=True)
            masks_ = masks_ > 0
            masks_ = masks_[0, 0]
            
            ys, xs = torch.where(masks_ != 0)
            exemplar_mask_bboxes = torch.tensor([[torch.min(xs), torch.min(ys), torch.max(xs), torch.max(ys)]], dtype=image.dtype, device=image.device) / img_res
            ltrb, center = xyxy_to_ltrb(exemplar_mask_bboxes)

            cx, cy = center[:,0], center[:,1]
            l, t, r, b = ltrb[:,0], ltrb[:,1], ltrb[:,2], ltrb[:,3]
            le, te, re, be = cx - x1, cy - y1, x2 - cx, y2 - cy
            scaler = torch.concat([le/l, te/t, re/r, be/b], dim=-1)

            # import pdb; pdb.set_trace()
            # import numpy as np
            # import cv2

            # # canvas = np.zeros((new_img_h, new_img_w, 3), dtype=np.uint8)
            # canvas = cv2.imread("/home/eunchan/datasets/FSC147/images_384_VarV2/2922.jpg")
            # new_img_shape = canvas.shape[:2]
            # nh, nw = new_img_shape
            # new_img_res = torch.tensor([nw, nh, nw, nh], dtype=image.dtype, device=image.device)

            # canvas = cv2.rectangle(canvas, (int(x1 * nw), int(y1 * nh)), (int(x2 * nw), int(y2 * nh)), (0, 0, 255), 2)

            # nx, ny, nx2, ny2 = exemplar_mask_bboxes[0] * new_img_res
            # canvas = cv2.rectangle(canvas, (int(nx), int(ny)), (int(nx2), int(ny2)), (0, 255, 0), 2)
            # cv2.imwrite(f"./exemplar_debug.jpg", canvas)

            # 2. box refinement

            new_logits = []
            new_boxes = []
            new_ref_points = []
            
            for box_i in range(self.step, len(pred_boxes[bidx]) + self.step, self.step):
                box = pred_boxes[bidx][(box_i - self.step):box_i] * img_res # xyxy

                # Prepare prompt embeddings
                sparse_embeddings, dense_embeddings = prompt_encoder_sam(
                    points=None,
                    boxes=box,
                    masks=None,
                )

                # SAM mask decoder
                masks_, iou_predictions_ = self.mask_decoder(
                    image_embeddings=features[bidx].unsqueeze(0),
                    image_pe=prompt_encoder_sam.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                # Get corrected bounding boxes
                masks_ = F.interpolate(masks_, img_shape, mode="bilinear", align_corners=True)
                masks_ = masks_ > 0
                corrected_bboxes = torch.zeros((masks_.shape[0], 4), dtype=pred_boxes[bidx].dtype, device=pred_boxes[bidx].device)
                masks_ = masks_[:, 0]
                for index, mask_i in enumerate(masks_):
                    y, x = torch.where(mask_i != 0)
                    if y.shape[0] > 0 and x.shape[0] > 0:
                        corrected_bboxes[index, 0] = torch.min(x)
                        corrected_bboxes[index, 1] = torch.min(y)
                        corrected_bboxes[index, 2] = torch.max(x)
                        corrected_bboxes[index, 3] = torch.max(y)

                ltrb, center = xyxy_to_ltrb(corrected_bboxes/img_res)
                ltrb = ltrb * scaler
                corrected_bboxes = ltrb_to_xyxy(ltrb, center)

                step_logits = torch.concat([iou_predictions_, torch.zeros_like(iou_predictions_)], dim=-1)
                step_boxes = corrected_bboxes
                step_ref_points = torch.stack([(step_boxes[:, 0] + step_boxes[:, 2]) / 2, (step_boxes[:, 1] + step_boxes[:, 3]) / 2], dim=-1)

                new_logits.append(step_logits)
                new_boxes.append(step_boxes)
                new_ref_points.append(step_ref_points)
            
            # new_pred_logits.append(torch.cat(new_logits, dim=0)) # type 1, score = IoU score
            new_pred_logits.append(torch.cat(new_logits, dim=0) * pred_logits[bidx]) # type 2, score = IoU score * original score
            # new_pred_logits.append(pred_logits[bidx]) # type 3, score = original score
            new_pred_boxes.append(torch.cat(new_boxes, dim=0))
            new_pred_ref_points.append(torch.cat(new_ref_points, dim=0))

        return new_pred_logits, new_pred_boxes, new_pred_ref_points

    def forward(self, pred_logits, pred_boxes, ref_points, image, features):
        new_pred_logits = []
        new_pred_boxes = []
        new_pred_ref_points = []

        for bidx in range(len(pred_logits)):
            if len(pred_boxes[bidx]) == 0:
                new_pred_logits.append(pred_logits[bidx])
                new_pred_boxes.append(pred_boxes[bidx])
                new_pred_ref_points.append(ref_points[bidx])
                continue

            emb_shape = features[bidx].shape[-2:]
            img_shape = image[bidx].shape[-2:]
            img_h, img_w = img_shape
            img_res = torch.tensor([img_w, img_h, img_w, img_h], dtype=image.dtype, device=image.device)

            prompt_encoder_sam = self.prompt_encoder_init(img_shape, emb_shape).to(image.device)
            new_logits = []
            new_boxes = []
            new_ref_points = []
            
            for box_i in range(self.step, len(pred_boxes[bidx]) + self.step, self.step):
                box = pred_boxes[bidx][(box_i - self.step):box_i] * img_res # xyxy

                # Prepare prompt embeddings
                sparse_embeddings, dense_embeddings = prompt_encoder_sam(
                    points=None,
                    boxes=box,
                    masks=None,
                )

                # SAM mask decoder
                masks_, iou_predictions_ = self.mask_decoder(
                    image_embeddings=features[bidx].unsqueeze(0),
                    image_pe=prompt_encoder_sam.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                # Get corrected bounding boxes
                masks_ = F.interpolate(masks_, img_shape, mode="bilinear", align_corners=True)
                masks_ = masks_ > 0
                corrected_bboxes = torch.zeros((masks_.shape[0], 4), dtype=pred_boxes[bidx].dtype, device=pred_boxes[bidx].device)
                masks_ = masks_[:, 0]
                for index, mask_i in enumerate(masks_):
                    y, x = torch.where(mask_i != 0)
                    if y.shape[0] > 0 and x.shape[0] > 0:
                        corrected_bboxes[index, 0] = torch.min(x)
                        corrected_bboxes[index, 1] = torch.min(y)
                        corrected_bboxes[index, 2] = torch.max(x)
                        corrected_bboxes[index, 3] = torch.max(y)

                step_logits = torch.concat([iou_predictions_, torch.zeros_like(iou_predictions_)], dim=-1)
                step_boxes = corrected_bboxes / img_res
                step_ref_points = torch.stack([(step_boxes[:, 0] + step_boxes[:, 2]) / 2, (step_boxes[:, 1] + step_boxes[:, 3]) / 2], dim=-1)

                new_logits.append(step_logits)
                new_boxes.append(step_boxes)
                new_ref_points.append(step_ref_points)

            # new_pred_logits.append(torch.cat(new_logits, dim=0)) # type 1, score = IoU score
            new_pred_logits.append(torch.cat(new_logits, dim=0) * pred_logits[bidx]) # type 2, score = IoU score * original score
            # new_pred_logits.append(pred_logits[bidx]) # type 3, score = original score
            new_pred_boxes.append(torch.cat(new_boxes, dim=0))
            new_pred_ref_points.append(torch.cat(new_ref_points, dim=0))

        return new_pred_logits, new_pred_boxes, new_pred_ref_points
    
    def save_masks(self, pred_logits, pred_boxes, ref_points, image, features, log_path, batch):
        import os
        import cv2
        import numpy as np

        MASK_PATH = 'masks'
        log_path = os.path.join(log_path, MASK_PATH)
        os.makedirs(log_path, exist_ok=True)

        for bidx in range(len(pred_logits)):
            emb_shape = features[bidx].shape[-2:]
            img_shape = image[bidx].shape[-2:]
            img_h, img_w = img_shape
            img_res = torch.tensor([img_w, img_h, img_w, img_h], dtype=image.dtype, device=image.device)

            prompt_encoder_sam = self.prompt_encoder_init(img_shape, emb_shape).to(image.device)
            
            masks = []
            for box_i in range(self.step, len(pred_boxes[bidx]) + self.step, self.step):
                box = pred_boxes[bidx][(box_i - self.step):box_i] * img_res # xyxy

                # Prepare prompt embeddings
                sparse_embeddings, dense_embeddings = prompt_encoder_sam(
                    points=None,
                    boxes=box,
                    masks=None,
                )

                # SAM mask decoder
                masks_, iou_predictions_ = self.mask_decoder(
                    image_embeddings=features[bidx].unsqueeze(0),
                    image_pe=prompt_encoder_sam.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                # Get corrected bounding boxes
                masks_ = F.interpolate(masks_, img_shape, mode="bilinear", align_corners=True)
                masks_ = masks_ > 0
                masks.append(masks_)

            masks = torch.max(torch.cat(masks, dim=0), dim=0)[0]
            masks = masks.permute(1,2,0).detach().cpu().numpy()
            masks = (masks * 255).astype(np.uint8)

            img_name = batch['img_name'][bidx]
            cv2.imwrite(os.path.join(log_path, f"{img_name}.png"), masks)

            # import time
            # cv2.imwrite(f"./masks_debug_{time.time():.2f}.jpg", masks)