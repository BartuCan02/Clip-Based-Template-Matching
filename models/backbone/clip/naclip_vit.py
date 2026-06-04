import importlib.util
import os
import sys

import torch
import torch.nn as nn

# Path to the cloned official NACLIP repo
_NACLIP_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "third_party", "naclip",
))


def _load_naclip(sigma: float):
    """
    Build NACLIP's patched VisionTransformer and load OpenAI CLIP weights into it.

    Uses importlib to load third_party/naclip/clip/model.py directly — it has
    no relative imports so this works without any sys.path / sys.modules tricks.
    The OpenAI CLIP weights (already cached) are extracted via state_dict() and
    fed into NACLIP's build_model, which creates their VisionTransformer with
    the set_params / custom_attn API.
    """
    # Load NACLIP's model module directly from the file (avoids import conflicts)
    spec = importlib.util.spec_from_file_location(
        "_naclip_model_module",
        os.path.join(_NACLIP_ROOT, "clip", "model.py"),
    )
    naclip_model = importlib.util.module_from_spec(spec)
    sys.modules["_naclip_model_module"] = naclip_model
    spec.loader.exec_module(naclip_model)

    # Get weights from OpenAI's CLIP (uses the cached download)
    import clip as openai_clip
    oa_model, preprocess = openai_clip.load("ViT-B/16", device="cpu", jit=False)
    state_dict = oa_model.state_dict()

    # Build NACLIP's CLIP model (their VisionTransformer has set_params + custom_attn)
    model = naclip_model.build_model(state_dict)
    model.float()  # ensure fp32 on CPU (NACLIP's build_model converts to fp16)

    model.visual.set_params("reduced", "naclip", sigma)
    return model, preprocess


class NaClipViT16Backbone(nn.Module):
    """
    CLIP ViT-B/16 with the official NACLIP attention strategy.

    Uses sinahmr/NACLIP's VisionTransformer directly (third_party/naclip),
    configured with set_params("reduced", "naclip", sigma):
      - blocks 0 … N-2  : standard CLIP attention
      - block N-1 (last): K@K.T + Gaussian spatial window, no residual, no MLP

    Input  : [B, 3, 224, 224]  — CLIP-preprocessed images
    Output : [B, 512, 14, 14]  — spatially coherent patch features
    """

    def __init__(self, requires_grad: bool = False, sigma: float = 5.0) -> None:
        super().__init__()

        model, preprocess = _load_naclip(sigma)

        self.visual = model.visual
        self._preprocess = preprocess
        self.num_channels: int = self.visual.output_dim  # 512

        for p in self.parameters():
            p.requires_grad_(requires_grad)

    @property
    def clip_preprocess(self):
        return self._preprocess

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x       : [B, 3, H, W]
        returns : [B, 512, H//16, W//16]
        """
        patch_size = self.visual.conv1.kernel_size[0]
        B, _, H, W = x.shape
        H_p, W_p = H // patch_size, W // patch_size

        # NACLIP's forward with return_all=True → [B, N+1, D]  (ln_post + proj applied)
        all_tokens = self.visual(x, return_all=True)
        patch_tokens = all_tokens[:, 1:]               # drop CLS → [B, N, D]
        D = patch_tokens.shape[-1]
        return patch_tokens.permute(0, 2, 1).reshape(B, D, H_p, W_p)
