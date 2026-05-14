import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation


class CLIPSegHeatmap:
    def __init__(self, model_name="CIDAS/clipseg-rd64-refined", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = CLIPSegProcessor.from_pretrained(model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(self.device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False


    @torch.no_grad()
    def __call__(self, image_pil, text_prompt, target_size=(128, 128)):
        inputs = self.processor(
            text=[text_prompt],
            images=[image_pil],
            return_tensors="pt",
            padding=True
        ).to(self.device)

        outputs = self.model(**inputs)

        # Raw segmentation logits
        logits = outputs.logits  # [1, H, W]

        # Convert to probability-like heatmap
        heatmap = torch.sigmoid(logits).unsqueeze(1)  # [1, 1, H, W]

        # Resize to TMR feature map size
        heatmap_up = torch.nn.functional.interpolate(
            heatmap,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

        return heatmap_up  # [1, 1, 128, 128]


if __name__ == "__main__":

    import os
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs("debug_logs", exist_ok=True)

    img_url = "/content/drive/MyDrive/Colab Notebooks/NLP_PROJECT/Template-Matching-and-Regression/demo/5.jpg"

    image_pil = Image.open(img_url).convert("RGB")

    clipseg = CLIPSegHeatmap()

    prompt = "egg"

    # Run CLIPSeg
    heatmap = clipseg(
        image_pil=image_pil,
        text_prompt=prompt,
        target_size=(128, 128)
    )

    # [128,128]
    h = heatmap[0, 0].detach().float().cpu().numpy()

    # Original image
    img = cv2.imread(img_url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    H, W, _ = img.shape

    # Resize heatmap to image resolution
    h_resized = cv2.resize(h, (W, H))

    # Normalize for visualization
    h_norm = (h_resized - h_resized.min()) / (
        h_resized.max() - h_resized.min() + 1e-8
    )

    # Create colored heatmap
    heatmap_color = plt.cm.jet(h_norm)[..., :3]

    # Overlay
    overlay = (
        0.6 * (img / 255.0)
        + 0.4 * heatmap_color
    )
    overlay = np.clip(overlay, 0, 1)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    im = axes[1].imshow(h_resized, cmap="jet")
    axes[1].set_title(f"CLIPSeg Heatmap: {prompt}")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Heatmap Overlay")
    axes[2].axis("off")

    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    plt.savefig(
        "debug_logs/clipseg_egg_heatmap.png",
        bbox_inches="tight",
        dpi=200
    )

    plt.show()








if __name__ == "__main__": 

  device = "cuda" if torch.cuda.is_available() else "cpu"

  img_url = "/content/drive/MyDrive/Colab Notebooks/NLP_PROJECT/Template-Matching-and-Regression/demo/5.jpg"

  image_pil = Image.open(img_url).convert("RGB")

  clip_heatmap_model = CLIPHeatmap(
      model_name="ViT-B/16",
      device=device
  )


  clip_heatmap = clip_heatmap_model(
      image_pil=image_pil,
      text_prompt="a photo of egg",
  )

  print(clip_heatmap.shape)