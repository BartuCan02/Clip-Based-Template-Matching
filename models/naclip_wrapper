import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from prompts.imagenet_template import openai_imagenet_template

class NACLIPHeatmap:
    def __init__(
        self,
        clip_path="ViT-B/16",
        device="cuda",
        arch="reduced",
        attn_strategy="naclip",
        gaussian_std=5.0,
        logit_scale=40,
    ):
        self.device = device
        self.net, self.preprocess = clip.load(
            clip_path,
            device=device,
            jit=False,
        )

        self.net.visual.set_params(
            arch,
            attn_strategy,
            gaussian_std,
        )

        self.net.eval()
        self.logit_scale = logit_scale
        self.align_corners = False

    @torch.no_grad()
    def encode_prompts(self, class_names):
        query_features = []

        for name in class_names:
            texts = [template(name) for template in openai_imagenet_template]
            tokens = clip.tokenize(texts).to(self.device)

            feat = self.net.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)

            feat = feat.mean(dim=0)
            feat = feat / feat.norm()

            query_features.append(feat.unsqueeze(0))

        query_features = torch.cat(query_features, dim=0)
        return query_features

    @torch.no_grad()
    def forward_logits(self, img, class_names):
        """
        img: [B, 3, H, W], already normalized like NACLIP/CLIP expects.
        class_names: list[str], e.g. ["background", "brick pattern"]
        return: [B, K, H, W]
        """
        query_features = self.encode_prompts(class_names)

        image_features = self.net.encode_image(img, return_all=True)
        image_features = image_features[:, 1:]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = image_features @ query_features.T

        patch_size = self.net.visual.patch_size
        H_img, W_img = img.shape[-2:]
        H_patch = H_img // patch_size
        W_patch = W_img // patch_size
        K = logits.shape[-1]

        logits = logits.permute(0, 2, 1).reshape(-1, K, H_patch, W_patch)

        logits = F.interpolate(
            logits,
            size=img.shape[-2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )

        return logits

    @torch.no_grad()
    def target_heatmap(self, img, class_names, target_idx=1, out_size=None):
        logits = self.forward_logits(img, class_names)

        probs = torch.softmax(logits * self.logit_scale, dim=1)
        heat = probs[:, target_idx:target_idx + 1]

        if out_size is not None:    #upsampling
            heat = F.interpolate(
                heat,
                size=out_size,
                mode="bilinear",
                align_corners=False,
            )

        return heat