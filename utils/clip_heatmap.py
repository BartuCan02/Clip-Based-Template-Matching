import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import clip



class CLIPHeatmap(nn.Module):
    def __init__(self, model_name="ViT-B/16", device="cuda"):
        super().__init__()

        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_text(self, text_prompt):
        tokens = clip.tokenize([text_prompt]).to(self.device)

        text_features = self.model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)

        return text_features  # [1, 512]

    @torch.no_grad()
    def get_patch_features(self, image_pil):
        """
        image_pil: original RGB PIL image
        """

        image = self.preprocess(image_pil).unsqueeze(0).to(
          device=self.device,
          dtype=self.model.visual.conv1.weight.dtype
        )

        # CLIP visual forward manually to get patch tokens
        visual = self.model.visual

        x = visual.conv1(image)  # [B, width, grid, grid] 224/16 = 14 = grid

        B, C, H, W = x.shape

        x = x.reshape(B, C, H * W)
        x = x.permute(0, 2, 1)  # [B, HW, C]

        class_embedding = visual.class_embedding.to(x.dtype)
        class_embedding = class_embedding + torch.zeros(
            B, 1, C, dtype=x.dtype, device=x.device
        )

        x = torch.cat([class_embedding, x], dim=1)  # [B, 1+HW, C]

        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # [1+HW, B, C]
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # [B, 1+HW, C]

        # remove CLS token
        patch_tokens = x[:, 1:, :]  # [B, HW, C]

        patch_tokens = visual.ln_post(patch_tokens)

        if visual.proj is not None:
            patch_tokens = patch_tokens @ visual.proj

        patch_tokens = F.normalize(patch_tokens, dim=-1)

        grid_size = int(H)

        patch_features = patch_tokens.reshape(B, grid_size, grid_size, -1)

        return patch_features  # ViT-B/16: [1, 14, 14, 512]

    @torch.no_grad()
    def forward(self, image_pil, text_prompt, target_size=(128, 128)):
        text_features = self.encode_text(text_prompt)  # [1, 512]

        patch_features = self.get_patch_features(image_pil)
        # [1, 14, 14, 512] for ViT-B/16

        B, Hc, Wc, D = patch_features.shape

        text_features = text_features.view(1, 1, 1, D)

        heatmap = (patch_features * text_features).sum(dim=-1, keepdim=True)
        # [1, 14, 14, 1]

        heatmap = heatmap.permute(0, 3, 1, 2)
        # [1, 1, 14, 14]

        heatmap_up = F.interpolate(
            heatmap,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )
        # [1, 1, 128, 128]

        return heatmap_up

if __name__ == "__main__":

    import os
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_url = "/content/drive/MyDrive/Colab Notebooks/NLP_PROJECT/Template-Matching-and-Regression/demo/5.jpg"

    image_pil = Image.open(img_url).convert("RGB")

    clip_heatmap_model = CLIPHeatmap(
        model_name="ViT-B/16",
        device=device
    )

    prompts = [
        "a photo of egg",
        "a photo of car"
    ]

    heatmaps = {}

    for prompt in prompts:
        clip_heatmap = clip_heatmap_model(
            image_pil=image_pil,
            text_prompt=prompt,
        )

        heatmap = clip_heatmap[0, 0].detach().float().cpu().numpy()
        heatmaps[prompt] = heatmap

        print(f"\nPrompt: {prompt}")
        print("shape:", clip_heatmap.shape)
        print("min:", heatmap.min())
        print("max:", heatmap.max())
        print("mean:", heatmap.mean())

    # Shared color scale for fair comparison
    vmin = min(h.min() for h in heatmaps.values())
    vmax = max(h.max() for h in heatmaps.values())

    # Original image
    img = cv2.imread(img_url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape

    os.makedirs("debug_logs", exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row_idx, prompt in enumerate(prompts):
        heatmap = heatmaps[prompt]

        # Resize heatmap to original image size, but DO NOT normalize independently
        heatmap_resized = cv2.resize(heatmap, (W, H))

        # For overlay color mapping, use shared vmin/vmax
        heatmap_for_color = (heatmap_resized - vmin) / (vmax - vmin + 1e-8)
        heatmap_for_color = np.clip(heatmap_for_color, 0, 1)

        heatmap_color = plt.cm.jet(heatmap_for_color)[..., :3]

        overlay = (
            0.6 * (img / 255.0)
            + 0.4 * heatmap_color
        )
        overlay = np.clip(overlay, 0, 1)

        axes[row_idx, 0].imshow(img)
        axes[row_idx, 0].set_title("Original Image")
        axes[row_idx, 0].axis("off")

        im = axes[row_idx, 1].imshow(
            heatmap_resized,
            cmap="jet",
            vmin=vmin,
            vmax=vmax
        )
        axes[row_idx, 1].set_title(f"CLIP Heatmap: {prompt}")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title(f"Overlay: {prompt}")
        axes[row_idx, 2].axis("off")

    fig.colorbar(im, ax=axes[:, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    plt.savefig(
        "debug_logs/egg_vs_car_clip_heatmap_shared_scale.png",
        bbox_inches="tight",
        dpi=200
    )

    plt.show()



