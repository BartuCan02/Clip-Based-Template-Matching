# preprocess before running this code:
# 1. copy paste code from matching_net_simpleFusion into matching_net.py
# 2. make sure to clone NACLIP repo, and the best_model.ckpt and sam
# 3. run these: 
#   !pip install ftfy regex yapf==0.40.1
#   !pip install -r requirements.txt
# 4. example usage in colab:
#   !python inference_simpleFusion.py \
#  --image /content/drive/MyDrive/nlp-prac/Template-Matching-and-Regression/demo/5.jpg \
#  --ckpt /content/drive/MyDrive/nlp-prac/Template-Matching-and-Regression/weights/TMR_RPINE/best_model.ckpt \
#  --naclip-repo /content/drive/MyDrive/nlp-prac//NACLIP \
#  --target egg \
#  --exemplar-box 100 100 200 200 \
#  --output outputs/simple_fusion_demo.jpg




import argparse
from pathlib import Path
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from argparse import Namespace
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.nn.functional as F

from models import build_model
from utils.TM_utils import Get_pred_boxes, NMS

def config_parser():
    parser = argparse.ArgumentParser(
        description="TMR inference with simple NACLIP heatmap fusion."
    )

    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--ckpt", required=True, help="Path to TMR checkpoint.")
    parser.add_argument("--naclip-repo", required=True, help="Path to cloned NACLIP repo.")
    parser.add_argument("--target", required=True, help='Target word, e.g. "egg".')

    parser.add_argument(
        "--exemplar-box",
        required=True,
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Exemplar box in original image pixel coordinates.",
    )

    parser.add_argument("--output", default="outputs/simple_fusion_visualization.jpg")
    parser.add_argument("--image-size", default=1024, type=int)
    parser.add_argument("--fusion-size", default=128, type=int)
    parser.add_argument("--cls-threshold", default=0.7, type=float)
    parser.add_argument("--score-threshold", default=0.5, type=float)
    parser.add_argument("--nms-iou-threshold", default=0.5, type=float)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--log-stats",
        action="store_true",
        help="Print per-layer f_cat means before/after modulation (the statistics "
             "behind the report's Approach 2 distribution-shift analysis).",
    )

    return parser.parse_args()

def get_tmr_args():
    return Namespace(
        modeltype="matching_net",
        emb_dim=512,
        no_matcher=False,
        squeeze=False,
        fusion=True,
        template_type="roi_align",
        feature_upsample=True,
        ablation_no_box_regression=False,
        decoder_num_layer=1,
        decoder_kernel_size=3,
        backbone="sam",
        encoder="original",
        dilation=True,
        positive_threshold=0.5,
        negative_threshold=0.5,
        NMS_cls_threshold=0.25,
        NMS_iou_threshold=0.5,
        eval_multi_scale=False,
        regression_scaling_imgsize=False,
        regression_scaling_WH_only=False,
        focal_loss=False,
    )

def load_tmr(ckpt_path, device):
    args = get_tmr_args()

    model = build_model(args).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            clean_state_dict[k[len("model."):]] = v
        else:
            clean_state_dict[k] = v

    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)

    print("Missing keys:", missing[:10])
    print("Unexpected keys:", unexpected[:10])

    model.eval()
    return model

def load_naclip(naclip_repo, device):
    import sys
    from pathlib import Path

    naclip_repo = str(Path(naclip_repo).expanduser().resolve())

    if naclip_repo not in sys.path:
        sys.path.insert(0, naclip_repo)

    from models.naclip_wrapper import NACLIPHeatmap

    return NACLIPHeatmap(
        clip_path="ViT-B/16",
        device=device,
        arch="reduced",
        attn_strategy="naclip",
        gaussian_std=5.0,
    )


def preprocess_image(image_path, device, image_size=1024):

    pil_img = Image.open(image_path).convert("RGB")
    original_size = pil_img.size 

    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

    image_tensor = transform(image=np.array(pil_img))["image"].unsqueeze(0)
    return image_tensor.to(device), pil_img, original_size

def make_naclip_map(pil_img, naclip_heatmapper, target_class, device,):
    clip_img = naclip_heatmapper.preprocess(pil_img).unsqueeze(0).to(device)

    heat = naclip_heatmapper.target_heatmap(
        clip_img,
        class_names=["background", target_class],
        target_idx=1,

    )

    return heat

def make_exemplar_box(box_xyxy, original_size, device):
    original_w, original_h = original_size

    x1, y1, x2, y2 = box_xyxy

    box_norm = [
        x1 / original_w,
        y1 / original_h,
        x2 / original_w,
        y2 / original_h,
    ]

    exemplar_tensor = torch.tensor(
        [box_norm],
        dtype=torch.float32,
        device=device,
    )

    return [exemplar_tensor]

def postprocess(
    pred_objectness,
    pred_regressions,
    exemplars,
    cls_threshold,
    score_threshold,
    nms_iou_threshold,
):

    batch_info = {
        "regression_ablation_a": False,
        "regression_ablation_b": False,
        "regression_ablation_c": False,
    }

    pred_logits, pred_boxes, ref_points = Get_pred_boxes(
        pred_objectness,
        pred_regressions,
        exemplars,
        batch_info,
        cls_ths=cls_threshold,
        box_reg=True,
    )

    pred_logits, pred_boxes, ref_points = NMS(
        pred_logits,
        pred_boxes,
        ref_points,
        iou_threshold=nms_iou_threshold,
    )

    scores = pred_logits[0][:, 0]
    boxes = pred_boxes[0]

    keep = scores >= score_threshold
    return boxes[keep], scores[keep]

def normalize_heatmap(x):

    x = x.detach().float().cpu()
    x = x - x.min()
    x = x / (x.max() + 1e-6)
    return x

def draw_boxes_on_image(pil_img, boxes, scores=None, color=(255, 0, 0), thickness=3):
    img_np = np.array(pil_img).copy()
    height, width = img_np.shape[:2]

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box.detach().cpu().numpy()

        x1 = int(np.clip(x1, 0, 1) * width)
        y1 = int(np.clip(y1, 0, 1) * height)
        x2 = int(np.clip(x2, 0, 1) * width)
        y2 = int(np.clip(y2, 0, 1) * height)

        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, thickness)

        if scores is not None:
            score = scores[idx].detach().cpu().item()
            cv2.putText(
                img_np,
                f"{score:.2f}",
                (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    return img_np

def save_visualization(
    image_path,
    pil_img,
    exemplar_box,
    naclip_heat,
    pred_objectness,
    pred_boxes,
    pred_scores,
    target,
    output_path,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    naclip_vis = normalize_heatmap(naclip_heat[0, 0])

    presence_prob = torch.sigmoid(pred_objectness[0])[0, 0]
    presence_vis = normalize_heatmap(presence_prob)

    exemplar_norm = torch.tensor([exemplar_box], dtype=torch.float32) / torch.tensor(
        [pil_img.width, pil_img.height, pil_img.width, pil_img.height],
        dtype=torch.float32,
    )

    exemplar_img = draw_boxes_on_image(
        pil_img,
        exemplar_norm,
        color=(0, 255, 0),
        thickness=4,
    )

    pred_img = draw_boxes_on_image(
        pil_img,
        pred_boxes,
        scores=pred_scores,
        color=(255, 0, 0),
        thickness=3,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].imshow(exemplar_img)
    axes[0, 0].set_title("Input image + exemplar box")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pil_img.resize((naclip_vis.shape[1], naclip_vis.shape[0])))
    axes[0, 1].imshow(naclip_vis, cmap="jet", alpha=0.45)
    axes[0, 1].set_title(f'NACLIP heatmap: "{target}"')
    axes[0, 1].axis("off")

    axes[1, 0].imshow(pil_img.resize((presence_vis.shape[1], presence_vis.shape[0])))
    axes[1, 0].imshow(presence_vis, cmap="jet", alpha=0.45)
    axes[1, 0].set_title("TMR presence after simple fusion")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pred_img)
    axes[1, 1].set_title(f"Predicted boxes: {len(pred_boxes)}")
    axes[1, 1].axis("off")

    fig.suptitle(str(image_path), fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f"Saved visualization to {output_path}")

def main(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tmr_model = load_tmr(args.ckpt, device)
    print(f"tmr Model loaded")

    naclip_heatmapper = load_naclip(args.naclip_repo, device)
    print(f"naclip_heatmapper loaded")

    image_tensor, pil_img, original_size = preprocess_image(
        args.image,
        device=device,
        image_size=args.image_size,
    )
    print(f"image preprocessing finished")

    exemplars = make_exemplar_box(
        box_xyxy=args.exemplar_box,
        original_size=original_size,
        device=device,
    )
    print(f"exemplar created")

    naclip_heat = make_naclip_map(
        pil_img,
        naclip_heatmapper,
        target_class=args.target,
        device=device,
    )
    print(f"naclip heatmap generated")

    naclip_fusion_map = F.interpolate(
        naclip_heat,
        size=(args.fusion_size, args.fusion_size),
        mode="bilinear",
        align_corners=False,
    )
    print(f"naclip heatmap interpolated")

    with torch.no_grad():
        pred_objectness, pred_regressions, f_TMs, backbone_feature = tmr_model(
            image_tensor,
            exemplars,
            naclip_map=naclip_fusion_map,
            log_stats=args.log_stats,
        )

        pred_boxes, pred_scores = postprocess(
            pred_objectness,
            pred_regressions,
            exemplars,
            cls_threshold=args.cls_threshold,
            score_threshold=args.score_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
        )

    print("NACLIP heatmap shape:", tuple(naclip_heat.shape))
    print("Presence shape:", tuple(pred_objectness[0].shape))
    print("Predicted boxes:", len(pred_boxes))

    save_visualization(
        image_path=args.image,
        pil_img=pil_img,
        exemplar_box=args.exemplar_box,
        naclip_heat=naclip_heat,
        pred_objectness=pred_objectness,
        pred_boxes=pred_boxes,
        pred_scores=pred_scores,
        target=args.target,
        output_path=args.output,
    )


if __name__ == "__main__":
    main(config_parser())
