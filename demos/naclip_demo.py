"""
NaCLIP heatmap demo — no checkpoint required.

Matches the notebook's NACLIPHeatmap pipeline:
  - Prompt ensembling over multiple templates
  - Two-class softmax (background vs target) with logit_scale=40

Usage
-----
    python naclip_demo.py --image demo/4.jpg --text "person"
    python naclip_demo.py --image demo/4.jpg --text "car" --save out.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import clip as openai_clip
from models.backbone.clip import build_naclip_backbone

PROMPT_TEMPLATES = [
    'a photo of a {}.', 'a photo of the {}.', 'a close-up photo of a {}.',
    'a photo of one {}.', 'there is a {} in the scene.', 'a {} in a photo.',
    'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.',
    'a photo of the hard to see {}.', 'a low resolution photo of the {}.',
    'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
    'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.',
    'a photo of a hard to see {}.', 'a bright photo of a {}.',
    'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
    'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.',
    'a photo of the cool {}.', 'a black and white photo of the {}.',
    'a painting of the {}.', 'a painting of a {}.', 'a pixelated photo of the {}.',
    'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
    'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.',
    'a {} in a video game.', 'a doodle of a {}.', 'the origami {}.',
    'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.',
    'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.',
    'a rendition of the {}.', 'a photo of the clean {}.', 'a photo of a large {}.',
    'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.',
    'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.',
    'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.',
    'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.',
    'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.',
    'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.',
    'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.',
    'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.',
    'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.',
]


def encode_text_ensemble(clip_model, class_name: str, device) -> torch.Tensor:
    # Faster vectorized version
    feats = []
    texts = [template.format(class_name) for template in PROMPT_TEMPLATES]
    tokens = openai_clip.tokenize(texts).to(device)

    with torch.inference_mode():
        feats = clip_model.encode_text(tokens).float()
        feats = F.normalize(feats, dim=-1)

    mean = feats.mean(dim=0, keepdim=True)
    return F.normalize(mean, dim=-1)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone   = build_naclip_backbone(requires_grad=False).to(device).eval()
    clip_model, _ = openai_clip.load("ViT-B/16", device=device)
    clip_model.eval()

    text_emb = encode_text_ensemble(clip_model, args.text, device)       # [1, 512]
    bg_emb   = encode_text_ensemble(clip_model, args.background, device) # [1, 512]

    pil = Image.open(args.image).convert("RGB")
    x   = backbone.clip_preprocess(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        patch_feats = backbone(x)                       # [1, 512, 14, 14]
        B, D, H, W  = patch_feats.shape

        pf = patch_feats.flatten(2).permute(0, 2, 1)   # [1, 196, 512]
        pf = F.normalize(pf, dim=-1)

        # Two-class softmax: [background, target]
        class_embs = torch.cat([bg_emb, text_emb], dim=0)  # [2, 512]
        logits = pf @ class_embs.T                          # [1, 196, 2]
        logits = logits.permute(0, 2, 1).reshape(1, 2, H, W)
        probs  = torch.softmax(logits * args.logit_scale, dim=1)

        prob14  = probs[0, 1].cpu().numpy()             # [14, 14]
        prob224 = F.interpolate(
            probs[:, 1:2], size=(224, 224), mode="bilinear", align_corners=False,
        ).squeeze().cpu().numpy()

    img224 = pil.resize((224, 224))
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(f'Prompt: "{args.text}" vs "{args.background}"  |  logit_scale={args.logit_scale}', fontsize=13)

    axes[0].imshow(img224)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    im2 = axes[1].imshow(prob14, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Patch probs (14×14)")
    for r in range(H):
        for c in range(W):
            axes[1].text(c, r, f"{prob14[r, c]:.2f}", ha="center", va="center", fontsize=4, color="white")
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    axes[2].imshow(img224)
    axes[2].imshow(prob224, cmap="jet", alpha=0.55, vmin=0, vmax=1)
    patch_size = 224 // H

    for i in range(1, H):
        axes[2].axhline(i * patch_size, color="white", linewidth=0.4, alpha=0.6)

    for i in range(1, W):
        axes[2].axvline(i * patch_size, color="white", linewidth=0.4, alpha=0.6)
    axes[2].set_title("Overlay + patch grid")
    axes[2].axis("off")

    axes[3].imshow(img224)
    top5 = sorted([(prob14[r, c], r, c) for r in range(14) for c in range(14)], reverse=True)[:5]
    for rank, (score, r, c) in enumerate(top5):
        rect = mpatches.Rectangle(
            (c * 16, r * 16), 16, 16,
            linewidth=1.5, edgecolor="red", facecolor="red", alpha=0.35,
        )
        axes[3].add_patch(rect)
        axes[3].text(c * 16 + 8, r * 16 + 8, str(rank + 1),
                     ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    axes[3].set_title("Top-5 matching patches")
    axes[3].axis("off")

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"Saved to {args.save}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",       required=True,         help="Path to input image")
    parser.add_argument("--text",        required=True,         help="Target class name")
    parser.add_argument("--background",  default="background",  help="Background class name")
    parser.add_argument("--logit_scale", type=float, default=40, help="Softmax temperature scale")
    parser.add_argument("--save",        default="",            help="Optional path to save the figure")
    main(parser.parse_args())
