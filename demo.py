import cv2
import torch
import gradio
import argparse
import numpy as np

from torch import nn
from PIL import Image
from gradio_bbox_annotator import BBoxAnnotator
import os
import json

from models import build_model
from utils.TM_utils import Get_pred_boxes, GT_map, NMS
from utils.box_refine import SAM_box_refiner
from models.backbone.sam.sam import Sam_Backbone
from utils.clip_utils import CLIPReranker, apply_clip_reranking

def config_parser():
    parser = argparse.ArgumentParser(description="TMR Demo")
    
    parser.add_argument('--ckpt', default="", metavar="FILE", help='path to ckpt', required=True)
    parser.add_argument('--port', default=6099, type=int)

    # model setting
    parser.add_argument('--modeltype', type=str, default="matching_net", help='Type of model')

    parser.add_argument('--emb_dim', default=512, type=int, help='Embedding dimension')
    parser.add_argument("--no_matcher", type=bool, default=False, help="If true, we don't use matching module")
    parser.add_argument("--squeeze", type=bool, default=False, help="If true, we use matching feature with channel 1")
    parser.add_argument("--fusion", type=bool, default=True, help="If true, we use a fusion layer to combine the features from the backbone and the template matching module")
    parser.add_argument("--positive_threshold", default=0.5, type=float, help="Threshold for positive samples")
    parser.add_argument("--negative_threshold", default=0.5, type=float, help="Threshold for negative samples")
    parser.add_argument("--NMS_cls_threshold", default=0.7, type=float, help="Threshold for NMS classificaiton score")
    parser.add_argument("--NMS_iou_threshold", default=0.5, type=float, help="Threshold for NMS Iou")
    parser.add_argument("--ablation_no_box_regression", type=bool, default=False, help="If true, we don't regress box parameters. Insted we use template size as box width, height parameter")
    parser.add_argument('--template_type', type=str, default='roi_align', help='template extraction algorithm Type')
    parser.add_argument("--feature_upsample", type=bool, default=True, help="If true, feature upsample for template matching")
    parser.add_argument('--eval_multi_scale', type=bool, default=False, help='multi scale processing for evaluation')
    parser.add_argument('--regression_scaling_imgsize', type=bool, default=False)
    parser.add_argument('--regression_scaling_WH_only', type=bool, default=False)
    parser.add_argument("--focal_loss", type=bool, default=False, help='Flag to use focal loss')

    # model - backbone setting
    parser.add_argument("--backbone", default="sam", type=str, help="Name of the backbone to use")
    parser.add_argument("--encoder", default="original", type=str, help="Name of the encoder type to use")
    parser.add_argument("--dilation", default=True, help="If true, we replace stride with dilation in the last convolutional block (DC5)")

    # model - head setting
    parser.add_argument("--decoder_num_layer", default=1, type=int)
    parser.add_argument("--decoder_kernel_size", default=3, type=int)

    # CLIP setting
    parser.add_argument("--use_clip", action="store_true")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--text_prompt", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--clip_alpha", type=float, default=1.0)
    parser.add_argument("--clip_beta", type=float, default=0.5)
    parser.add_argument("--clip_topk", type=int, default=100)
    parser.add_argument("--clip_threshold", type=float, default=0.0)

    args = parser.parse_args()

    return args

class Inference(nn.Module):
    def __init__(self, args):
        super(Inference, self).__init__()

        self.args = args
        self.model = build_model(args)
        self.is_cuda = torch.cuda.is_available()

        self.temp_sam = Sam_Backbone(requires_grad=False, model_type = "vit_h")
        self.refiner = SAM_box_refiner()

        self.clip_reranker = None
        if self.args.use_clip:
            if self.args.text_prompt is not None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.clip_reranker = CLIPReranker(
                    model_name=self.args.clip_model,
                    device=device
                )
                print(f"CLIP reranker initialized with model: {self.args.clip_model}")
                print(f"Positive prompt: {self.args.text_prompt}")
                if self.args.negative_prompt:
                    print(f"Negative prompt: {self.args.negative_prompt}")
            else:
                print("CLIP is enabled, but no text prompt was given.")
    
    def save_debug_json(self, debug_info):
        os.makedirs("debug_logs", exist_ok=True)

        with open("debug_logs/clip_debug.jsonl", "a") as f:
            f.write(json.dumps(debug_info) + "\n")


    def save_debug_json(self, debug_info):
        os.makedirs("debug_logs", exist_ok=True)

        with open("debug_logs/clip_debug.jsonl", "a") as f:
            f.write(json.dumps(debug_info) + "\n")

    def create_debug_info(
        self,
        location,
        image,
        pred_logits,
        pred_boxes,
        ref_points
    ):

        debug = {
            "location": location,
            "clip_enabled": self.clip_reranker is not None,
            "text_prompt": self.args.text_prompt,
            "negative_prompt": self.args.negative_prompt,
        }

        # BOXES
        if len(pred_boxes) > 0 and len(pred_boxes[0]) > 0:

            boxes = pred_boxes[0].detach().float().cpu()

            debug["boxes"] = {
                "shape": list(boxes.shape),
                "first_5_boxes": boxes[:5].tolist(),
            }

        # LOGITS
        if len(pred_logits) > 0 and len(pred_logits[0]) > 0:

            logits = pred_logits[0].detach().float().cpu()

            # BEFORE CLIP → pure TMR scores
            if location == "before_clip":

                debug["logits"] = {
                    "shape": list(logits.shape),

                    # only object score column
                    "tmr_first_5_scores":
                        logits[:5, 0].tolist()
                }

            # AFTER CLIP → fused CLIP scores
            else:

                mapped_scores = logits[:5, 0]

                raw_cosine_scores = (mapped_scores * 2) - 1

                debug["logits"] = {
                    "shape": list(logits.shape),

                    "fused_first_5_scores":
                        mapped_scores.tolist(),

                    "clip_raw_cosine_first_5":
                        raw_cosine_scores.tolist()
                }

        return debug

    def preprocess(self, image_input):

        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        def default_transform(size):
            return A.Compose([
                A.Resize(size, size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])

        img_url = image_input[0]
        exemplars = [[int(p[0]), int(p[1]), int(p[2]), int(p[3])] for p in image_input[1]]

        ori_image = Image.open(img_url).convert("RGB")
        img_w, img_h = ori_image.size

        # Keep original RGB image for CLIP
        clip_image = np.array(ori_image).astype(np.uint8)

        exemplars = np.array(exemplars, dtype=np.float32)  # xyxy format

        # box scaling for TMR
        img_res = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
        scaled_exemplars = exemplars / img_res[None, :]
        scaled_exemplars = torch.tensor(scaled_exemplars, dtype=torch.float32)

        if self.is_cuda:
            scaled_exemplars = scaled_exemplars.cuda()

        scaled_exemplars = [scaled_exemplars]

        # Normalized image for TMR
        image = np.array(ori_image)
        image = default_transform(1024)(image=image)['image'].unsqueeze(0)

        if self.is_cuda:
            image = image.cuda()

        return img_url, image, scaled_exemplars, clip_image
    
    @torch.no_grad()
    def infer(self, image_input, refine_box,*args, **kwargs):

        img_url, image, exemplars, clip_image = self.preprocess(image_input)
        exemplars = [[exemplars.unsqueeze(0)] for exemplars in exemplars[0]]

        pred_logits = []
        pred_boxes = []
        ref_points = []
        for exemplar in exemplars:
            pred_objectness, pred_regressions, matching_feature, _ = self.model(image, exemplar)
            dummy = {
                'regression_ablation_b': False,
                'regression_ablation_c': False,
            }
            _pred_logits, _pred_boxes, _ref_points = Get_pred_boxes(pred_objectness, pred_regressions, exemplar, dummy, self.args.NMS_cls_threshold, True)

            pred_logits.append(_pred_logits[0])
            pred_boxes.append(_pred_boxes[0])
            ref_points.append(_ref_points[0])

        pred_logits = [torch.concat(pred_logits)]
        pred_boxes = [torch.concat(pred_boxes)]
        ref_points = [torch.concat(ref_points)]

        if refine_box:
            backbone_feature = self.temp_sam(image)
            pred_logits, pred_boxes, ref_points = self.refiner(pred_logits, pred_boxes, ref_points, image, backbone_feature)
        

        before_clip_debug = self.create_debug_info(
            location="before_clip",
            image=image,
            pred_logits=pred_logits,
            pred_boxes=pred_boxes,
            ref_points=ref_points
        )

        self.save_debug_json(before_clip_debug)

        if self.clip_reranker is not None:
            pred_logits, pred_boxes, ref_points = apply_clip_reranking(
                pred_logits,
                pred_boxes,
                ref_points,
                clip_image,
                self.args.text_prompt,
                self.clip_reranker,
                negative_prompt=self.args.negative_prompt,
                alpha=self.args.clip_alpha,
                beta=self.args.clip_beta,
                top_k=self.args.clip_topk,
                threshold=self.args.clip_threshold
            )
            
        after_clip_debug = self.create_debug_info(
                location="after_clip",
                image=image,
                pred_logits=pred_logits,
                pred_boxes=pred_boxes,
                ref_points=ref_points
            )

        self.save_debug_json(after_clip_debug)
        
        # Final thresholding applied to the fused pred_logits and pred_boxed(so to beta * clip + alpha * tmr scores)
        final_score_threshold = 0.9

        filtered_logits = []
        filtered_boxes = []
        filtered_ref_points = []

        for logits, boxes, refs in zip(pred_logits, pred_boxes, ref_points):
            scores = logits[:, 0] if logits.dim() > 1 else logits
            keep = scores >= final_score_threshold

            filtered_logits.append(logits[keep])
            filtered_boxes.append(boxes[keep])
            filtered_ref_points.append(refs[keep])

        pred_logits = filtered_logits
        pred_boxes = filtered_boxes
        ref_points = filtered_ref_points
            
        pred_logits, pred_boxes, ref_points = NMS(pred_logits, pred_boxes, ref_points, self.args.NMS_iou_threshold)

        return self.visualize(img_url, pred_boxes[0].cpu().numpy())

    def visualize(self, img_url, pred_boxes):

        img = cv2.imread(img_url)
        H, W, _ = img.shape

        Max_Width = 1024
        R = Max_Width / W # ratio
        img = cv2.resize(img, (int(W*R), int(H*R)))

        for box in pred_boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = x1 * W, y1 * H, x2 * W, y2 * H
            x1, y1, x2, y2 = int(x1*R), int(y1*R), int(x2*R), int(y2*R)
            img = cv2.rectangle(img, (x1,y1), (x2, y2), (0,0,255), 2)

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img

def main(args):
    Infer = Inference(args)    
    state_dict = torch.load(args.ckpt, map_location='cpu')['state_dict']
    Infer.load_state_dict(state_dict, strict=False)
    Infer.eval()
    if torch.cuda.is_available():
        Infer = Infer.cuda()

    demo = gradio.Blocks()
    image_input = BBoxAnnotator(label="Target Image", categories=["support exemplars"])
    image_output = gradio.components.Image(label="Output Image", type="pil")

    with demo:
        gradio.Markdown("# Template Matching and Regression Demo")
        with gradio.Row():
            with gradio.Column(scale=5.0):
                image_input.render()
                refine_box_checkbox = gradio.Checkbox(label="SAM decoder box refinement")
                with gradio.Row(scale=2.0):
                    clearBtn = gradio.ClearButton(components=[image_input, refine_box_checkbox])
                    runBtn = gradio.Button("Run")
            with gradio.Column(scale=5.0):
                image_output.render()

                example = gradio.Examples(
                    examples=[
                        ["demo/1.jpg"],
                        ["demo/2.jpg"],
                        ["demo/3.jpg"],
                        ["demo/4.jpg"],
                        ["demo/5.jpg"],
                        ["demo/6.jpg"],
                    ],
                    inputs=image_input,
                    cache_examples=False,
                )

        runBtn.click(
            fn=lambda image_input, refine_box: Infer.infer(image_input, refine_box=refine_box),
            inputs=[image_input, refine_box_checkbox],
            outputs=[image_output]
        )

    demo.queue().launch(share=True,server_port=args.port)

if __name__ == "__main__":
    args = config_parser()
    main(args)

# I want to add a button for 'refine_box' option, between image_input and [clearBtn, runBtn]. How can I do this?