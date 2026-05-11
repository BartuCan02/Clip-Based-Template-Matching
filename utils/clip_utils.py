"""
CLIP-based semantic reranking utilities for TMR few-shot pattern detection.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


class CLIPReranker:
    """
    Handles CLIP-based semantic reranking of TMR predictions.
    """

    def __init__(self, model_name="ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.text_embeddings = {}

        try:
            import clip
            self.clip = clip
        except ImportError:
            raise ImportError(
                "OpenAI CLIP is required. Install with:\n"
                "pip install git+https://github.com/openai/CLIP.git"
            )

        self.model, self.preprocess = self.clip.load(model_name, device=device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    def encode_text(self, prompt, cache_key=None):
        """
        Encode a text prompt using CLIP text encoder.
        """
        if cache_key and cache_key in self.text_embeddings:
            return self.text_embeddings[cache_key]

        with torch.no_grad():
            text_tokens = self.clip.tokenize([prompt]).to(self.device)
            text_embedding = self.model.encode_text(text_tokens)
            text_embedding = F.normalize(text_embedding, dim=-1)

        if cache_key:
            self.text_embeddings[cache_key] = text_embedding

        return text_embedding

    def crop_and_encode_image(self, image_tensor, boxes, top_k=None, save_debug_crops=True):
        """
        Crop predicted regions from image and encode them with CLIP.

        image_tensor can be:
        - torch.Tensor: (C, H, W), (H, W, C), or (B, C, H, W) after slicing
        - np.ndarray: (H, W, C) or (C, H, W)

        boxes:
        - expected xyxy format
        - can be normalized [0, 1] or pixel coordinates
        """

        # Move image to CPU if it is a torch tensor
        if isinstance(image_tensor, torch.Tensor):
            if image_tensor.is_cuda:
                image_tensor = image_tensor.detach().cpu()
            else:
                image_tensor = image_tensor.detach()

            if image_tensor.dim() == 3:
                if image_tensor.shape[0] == 3:
                    # CHW -> HWC
                    image_np = image_tensor.permute(1, 2, 0).numpy()
                else:
                    # already HWC
                    image_np = image_tensor.numpy()
            else:
                raise ValueError(f"Unexpected image tensor shape: {image_tensor.shape}")

        elif isinstance(image_tensor, np.ndarray):
            image_np = image_tensor

            if image_np.ndim != 3:
                raise ValueError(f"Unexpected numpy image shape: {image_np.shape}")

            # If CHW, convert to HWC
            if image_np.shape[0] == 3 and image_np.shape[-1] != 3:
                image_np = np.transpose(image_np, (1, 2, 0))

        else:
            raise TypeError(f"Unsupported image type: {type(image_tensor)}")

        # Handle image value range
        if image_np.dtype == np.uint8:
            image_uint8 = image_np
        else:
            image_np = image_np.astype(np.float32)

            # If image looks normalized like ImageNet/SAM, this is not ideal.
            # But we safely map it to [0, 255] to avoid crashes.
            if image_np.min() < 0 or image_np.max() > 1:
                image_np = image_np - image_np.min()
                if image_np.max() > 0:
                    image_np = image_np / image_np.max()

            image_uint8 = (image_np * 255).clip(0, 255).astype(np.uint8)

        image_pil = Image.fromarray(image_uint8)
        H, W = image_uint8.shape[:2]

        # Prepare boxes
        if isinstance(boxes, torch.Tensor):
            boxes_pixel = boxes.detach().float().cpu().clone()
        else:
            boxes_pixel = torch.tensor(boxes, dtype=torch.float32)

        if boxes_pixel.numel() == 0:
            output_dim = self.model.visual.output_dim
            return torch.empty((0, output_dim), device=self.device), torch.empty((0,), dtype=torch.long)

        # If boxes are normalized, convert to pixel coordinates
        if boxes_pixel.max() <= 1.5:
            boxes_pixel[:, 0] *= W
            boxes_pixel[:, 1] *= H
            boxes_pixel[:, 2] *= W
            boxes_pixel[:, 3] *= H

        # Limit to top-k
        if top_k is not None and len(boxes_pixel) > top_k:
            valid_indices = torch.arange(top_k, dtype=torch.long)
            boxes_pixel = boxes_pixel[valid_indices]
        else:
            valid_indices = torch.arange(len(boxes_pixel), dtype=torch.long)

        image_embeddings = []

        if save_debug_crops:
            os.makedirs("debug_logs/crops", exist_ok=True)

        with torch.no_grad():
            for idx, box in enumerate(boxes_pixel):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                # Clamp to image bounds
                x1 = max(0, min(x1, W - 1))
                y1 = max(0, min(y1, H - 1))
                x2 = max(x1 + 1, min(x2, W))
                y2 = max(y1 + 1, min(y2, H))

                crop = image_pil.crop((x1, y1, x2, y2))

                if save_debug_crops and idx < 20:
                    crop.save(f"debug_logs/crops/crop_{idx}.jpg")

                crop_processed = self.preprocess(crop).unsqueeze(0).to(self.device)

                embedding = self.model.encode_image(crop_processed)
                embedding = F.normalize(embedding, dim=-1)

                image_embeddings.append(embedding)

        if image_embeddings:
            image_embeddings = torch.cat(image_embeddings, dim=0)
        else:
            output_dim = self.model.visual.output_dim
            image_embeddings = torch.empty((0, output_dim), device=self.device)

        return image_embeddings, valid_indices

    def compute_similarity(self, image_embeddings, text_embedding):
        """
        Compute cosine similarity between image and text embeddings.
        """
        image_embeddings = image_embeddings.to(self.device)
        text_embedding = text_embedding.to(self.device)

        similarities = torch.mm(image_embeddings, text_embedding.T).squeeze(-1)

        # Convert cosine similarity from [-1, 1] to [0, 1]
        similarities = (similarities + 1) / 2

        return similarities

    def compute_semantic_score(self, image_embeddings, positive_prompt, negative_prompt=None):
        """
        Compute semantic score using positive and optional negative prompt.
        """
        pos_embedding = self.encode_text(
            positive_prompt,
            cache_key=f"positive_{positive_prompt}"
        )

        pos_similarity = self.compute_similarity(image_embeddings, pos_embedding)

        if negative_prompt is not None:
            neg_embedding = self.encode_text(
                negative_prompt,
                cache_key=f"negative_{negative_prompt}"
            )

            neg_similarity = self.compute_similarity(image_embeddings, neg_embedding)

            semantic_score = pos_similarity - 0.3 * neg_similarity
            semantic_score = torch.clamp(semantic_score, min=0.0, max=1.0)
        else:
            semantic_score = pos_similarity

        return semantic_score


def fuse_scores(tmr_scores, clip_scores, alpha=0.7, beta=0.3):
    """
    Fuse TMR confidence scores with CLIP semantic scores.
    """

    if tmr_scores.dim() > 1:
        tmr_prob = tmr_scores[:, 0]
    else:
        tmr_prob = tmr_scores

    clip_scores = clip_scores.to(tmr_prob.device)

    fused = alpha * tmr_prob + beta * clip_scores

    # Keep score in probability-like range
    fused = torch.clamp(fused, min=0.0, max=1.0)

    return fused


def apply_clip_reranking(
    pred_logits,
    pred_boxes,
    ref_points,
    image,
    text_prompt,
    clip_reranker,
    negative_prompt=None,
    alpha=0.7,
    beta=0.3,
    top_k=100,
    threshold=0.0
):
    """
    Apply CLIP-based reranking to TMR predictions.

    Supports image as:
    - torch.Tensor: (B, C, H, W) or (C, H, W)
    - np.ndarray: (H, W, C) or (B, H, W, C)
    """

    pred_logits_reranked = []
    pred_boxes_reranked = []
    ref_points_reranked = []

    # Determine batch size safely
    if isinstance(image, torch.Tensor):
        batch_size = image.shape[0] if image.dim() == 4 else 1
    elif isinstance(image, np.ndarray):
        batch_size = image.shape[0] if image.ndim == 4 else 1
    else:
        batch_size = 1

    for bidx in range(batch_size):

        # Get image for this batch item
        if isinstance(image, torch.Tensor):
            img_tensor = image[bidx] if image.dim() == 4 else image

        elif isinstance(image, np.ndarray):
            img_tensor = image[bidx] if image.ndim == 4 else image

        else:
            img_tensor = image

        boxes = pred_boxes[bidx]
        logits = pred_logits[bidx]
        refs = ref_points[bidx]

        # Apply initial TMR threshold
        if threshold > 0:
            if logits.dim() > 1:
                tmr_scores = logits[:, 0]
            else:
                tmr_scores = logits

            tmr_mask = tmr_scores >= threshold

            boxes = boxes[tmr_mask]
            logits = logits[tmr_mask]
            refs = refs[tmr_mask]

        if len(boxes) == 0:
            pred_logits_reranked.append(logits)
            pred_boxes_reranked.append(boxes)
            ref_points_reranked.append(refs)
            continue

        # Encode cropped regions with CLIP
        image_embeddings, valid_indices = clip_reranker.crop_and_encode_image(
            img_tensor,
            boxes,
            top_k=top_k,
            save_debug_crops=True
        )

        if len(valid_indices) == 0 or image_embeddings.shape[0] == 0:
            pred_logits_reranked.append(logits)
            pred_boxes_reranked.append(boxes)
            ref_points_reranked.append(refs)
            continue

        clip_scores = clip_reranker.compute_semantic_score(
            image_embeddings,
            text_prompt,
            negative_prompt=negative_prompt
        )

        # Move indices to correct device for indexing logits
        valid_indices_device = valid_indices.to(logits.device)

        tmr_scores_subset = logits[valid_indices_device]

        fused_scores = fuse_scores(
            tmr_scores_subset,
            clip_scores,
            alpha=alpha,
            beta=beta
        )

        logits_updated = logits.clone()

        if logits_updated.dim() > 1:
            logits_updated[valid_indices_device, 0] = fused_scores
        else:
            logits_updated[valid_indices_device] = fused_scores

        pred_logits_reranked.append(logits_updated)
        pred_boxes_reranked.append(boxes)
        ref_points_reranked.append(refs)

    return pred_logits_reranked, pred_boxes_reranked, ref_points_reranked