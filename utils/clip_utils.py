"""
CLIP-based semantic reranking utilities for TMR few-shot pattern detection.

This module provides functions to load pretrained CLIP models, encode text prompts,
crop and preprocess image regions, compute semantic similarity scores, and fuse
CLIP scores with TMR confidence scores.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class CLIPReranker:
    """
    Handles CLIP-based semantic reranking of TMR predictions.
    
    Attributes:
        device: torch device (cuda or cpu)
        model: pretrained CLIP model
        preprocess: image preprocessing pipeline for CLIP
        text_embeddings: cached text embeddings for positive/negative prompts
    """
    
    def __init__(self, model_name="ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize CLIP model and preprocessing pipeline.
        
        Args:
            model_name: CLIP model identifier (e.g., "ViT-B/32", "ViT-L/14")
            device: torch device for computation
        """
        self.device = device
        self.text_embeddings = {}
        
        try:
            import clip
            self.clip = clip
        except ImportError:
            raise ImportError(
                "clip-py is required for CLIP reranking. Install with: "
                "pip install git+https://github.com/openai/CLIP.git"
            )
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        
        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def encode_text(self, prompt, cache_key=None):
        """
        Encode a text prompt using CLIP text encoder.
        
        Args:
            prompt: text string to encode
            cache_key: optional key for caching embeddings
            
        Returns:
            normalized text embedding of shape (1, 512) or (1, 768) depending on model
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
    
    def crop_and_encode_image(self, image_tensor, boxes, top_k=None):
        """
        Crop predicted regions from the original image and encode with CLIP.
        
        Args:
            image_tensor: input image tensor, shape (H, W, 3) in [0, 1] range or uint8
            boxes: predicted bounding boxes in format (x_min, y_min, x_max, y_max),
                   normalized to [0, 1] range. Shape: (N, 4)
            top_k: if specified, only encode top-k boxes (after sorting by area or confidence)
        
        Returns:
            image_embeddings: tensor of shape (N, embedding_dim) normalized embeddings
            valid_indices: indices of boxes that were successfully encoded
        """
        # Ensure image is in the right format
        if image_tensor.is_cuda:
            image_tensor = image_tensor.cpu()
        
        # Convert to numpy if needed
        if isinstance(image_tensor, torch.Tensor):
            # Assume shape is (C, H, W) or (H, W, C)
            if image_tensor.dim() == 3:
                if image_tensor.shape[0] == 3:  # (C, H, W)
                    image_np = image_tensor.permute(1, 2, 0).numpy()
                else:  # (H, W, C)
                    image_np = image_tensor.numpy()
            else:
                raise ValueError(f"Unexpected image tensor shape: {image_tensor.shape}")
        else:
            image_np = image_tensor
        
        # Normalize to [0, 1] if needed
        if image_np.dtype == np.uint8 or image_np.max() > 1:
            image_np = image_np.astype(np.float32) / 255.0
        
        # Convert to PIL Image
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        H, W = image_np.shape[:2]
        
        # Denormalize boxes from [0, 1] to pixel coordinates
        boxes_pixel = boxes.clone()
        boxes_pixel[:, 0] = boxes[:, 0] * W  # x_min
        boxes_pixel[:, 1] = boxes[:, 1] * H  # y_min
        boxes_pixel[:, 2] = boxes[:, 2] * W  # x_max
        boxes_pixel[:, 3] = boxes[:, 3] * H  # y_max
        
        # Limit to top-k if specified
        if top_k is not None and len(boxes_pixel) > top_k:
            # Keep top-k boxes by some criterion (e.g., first top_k)
            indices_to_keep = torch.arange(min(top_k, len(boxes_pixel)))
            boxes_pixel = boxes_pixel[indices_to_keep]
            valid_indices = indices_to_keep
        else:
            valid_indices = torch.arange(len(boxes_pixel))
        
        # Crop regions and encode
        image_embeddings = []
        
        with torch.no_grad():
            for idx, box in enumerate(boxes_pixel):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                
                # Ensure box coordinates are within image bounds
                x1 = max(0, min(x1, W - 1))
                y1 = max(0, min(y1, H - 1))
                x2 = max(x1 + 1, min(x2, W))
                y2 = max(y1 + 1, min(y2, H))
                
                # Crop image region
                crop = image_pil.crop((x1, y1, x2, y2))
                
                # Preprocess according to CLIP requirements
                crop_processed = self.preprocess(crop).unsqueeze(0).to(self.device)
                
                # Encode with CLIP image encoder
                embedding = self.model.encode_image(crop_processed)
                embedding = F.normalize(embedding, dim=-1)
                image_embeddings.append(embedding)
        
        if image_embeddings:
            image_embeddings = torch.cat(image_embeddings, dim=0)
        else:
            image_embeddings = torch.empty((0, self.model.visual.output_dim), device=self.device)
        
        return image_embeddings, valid_indices
    
    def compute_similarity(self, image_embeddings, text_embedding):
        """
        Compute cosine similarity between image and text embeddings.
        
        Args:
            image_embeddings: tensor of shape (N, embedding_dim)
            text_embedding: tensor of shape (1, embedding_dim)
        
        Returns:
            similarities: tensor of shape (N,) with similarity scores in [0, 1]
        """
        image_embeddings = image_embeddings.to(self.device)
        text_embedding = text_embedding.to(self.device)
        
        # Cosine similarity via normalized dot product
        similarities = torch.mm(image_embeddings, text_embedding.T).squeeze(-1)
        
        # Normalize to [0, 1]
        similarities = (similarities + 1) / 2  # [-1, 1] -> [0, 1]
        
        return similarities
    
    def compute_semantic_score(self, image_embeddings, positive_prompt, negative_prompt=None):
        """
        Compute semantic score combining positive and optional negative prompts.
        
        Args:
            image_embeddings: tensor of shape (N, embedding_dim)
            positive_prompt: positive text prompt string
            negative_prompt: optional negative prompt string
        
        Returns:
            semantic_scores: tensor of shape (N,) in [0, 1]
        """
        # Encode positive prompt
        pos_embedding = self.encode_text(positive_prompt, cache_key=f"positive_{positive_prompt}")
        pos_similarity = self.compute_similarity(image_embeddings, pos_embedding)
        
        if negative_prompt is not None:
            # Encode negative prompt
            neg_embedding = self.encode_text(negative_prompt, cache_key=f"negative_{negative_prompt}")
            neg_similarity = self.compute_similarity(image_embeddings, neg_embedding)
            
            # Semantic score = positive - weight * negative
            semantic_score = pos_similarity - 0.3 * neg_similarity
            semantic_score = torch.clamp(semantic_score, min=0.0, max=1.0)
        else:
            semantic_score = pos_similarity
        
        return semantic_score


def fuse_scores(tmr_scores, clip_scores, alpha=0.7, beta=0.3):
    """
    Fuse TMR confidence scores with CLIP semantic scores.
    
    Args:
        tmr_scores: original TMR objectness scores, shape (N,) or (N, 2)
        clip_scores: CLIP semantic similarity scores, shape (N,)
        alpha: weight for TMR score (default 0.7)
        beta: weight for CLIP score (default 0.3)
    
    Returns:
        fused_scores: weighted combination, shape (N,)
    """
    # Extract objectness probability if tmr_scores is (N, 2)
    if tmr_scores.dim() > 1:
        tmr_prob = tmr_scores[:, 0]
    else:
        tmr_prob = tmr_scores
    
    # Ensure both are on the same device
    device = tmr_prob.device
    clip_scores = clip_scores.to(device)
    
    # Weighted fusion
    fused = alpha * tmr_prob + beta * clip_scores
    
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
    
    This function processes each batch item independently, crops regions from the
    original image, encodes them with CLIP, computes semantic similarity, and
    fuses scores before final NMS.
    
    Args:
        pred_logits: list of tensors, shape [(N_i, 2)] for batch size B
        pred_boxes: list of tensors, shape [(N_i, 4)] for batch size B  
        ref_points: list of tensors, shape [(N_i, 2)] for batch size B
        image: input image tensor, shape (B, C, H, W)
        text_prompt: positive text prompt string
        clip_reranker: CLIPReranker instance
        negative_prompt: optional negative prompt string
        alpha: weight for TMR score
        beta: weight for CLIP score
        top_k: limit CLIP processing to top-k predictions per image
        threshold: minimum score threshold (applied before fusion)
    
    Returns:
        pred_logits_reranked: list of reranked logits tensors
        pred_boxes_reranked: list of boxes tensors (same as input, just filtered)
        ref_points_reranked: list of reference points (same as input, just filtered)
    """
    pred_logits_reranked = []
    pred_boxes_reranked = []
    ref_points_reranked = []
    
    # Determine batch size
    batch_size = image.shape[0] if image.dim() == 4 else 1
    
    for bidx in range(batch_size):
        # Get image for this batch item
        if image.dim() == 4:
            img_tensor = image[bidx]  # shape (C, H, W)
        else:
            img_tensor = image  # already shape (C, H, W)
        
        # Get predictions for this batch item
        boxes = pred_boxes[bidx]
        logits = pred_logits[bidx]
        refs = ref_points[bidx]
        
        # Apply initial threshold on TMR scores
        if threshold > 0:
            tmr_mask = logits[:, 0] >= threshold
            boxes = boxes[tmr_mask]
            logits = logits[tmr_mask]
            refs = refs[tmr_mask]
        
        if len(boxes) == 0:
            # No predictions above threshold
            pred_logits_reranked.append(logits)
            pred_boxes_reranked.append(boxes)
            ref_points_reranked.append(refs)
            continue
        
        # Crop and encode with CLIP
        image_embeddings, valid_indices = clip_reranker.crop_and_encode_image(
            img_tensor, boxes, top_k=top_k
        )
        
        if len(valid_indices) == 0 or image_embeddings.shape[0] == 0:
            # CLIP encoding failed, use original scores
            pred_logits_reranked.append(logits)
            pred_boxes_reranked.append(boxes)
            ref_points_reranked.append(refs)
            continue
        
        # Compute CLIP semantic scores
        clip_scores = clip_reranker.compute_semantic_score(
            image_embeddings, text_prompt, negative_prompt=negative_prompt
        )
        
        # Fuse scores
        tmr_scores_subset = logits[valid_indices]
        fused_scores = fuse_scores(tmr_scores_subset, clip_scores, alpha=alpha, beta=beta)
        
        # Update logits with fused scores
        logits_updated = logits.clone()
        logits_updated[valid_indices, 0] = fused_scores
        
        pred_logits_reranked.append(logits_updated)
        pred_boxes_reranked.append(boxes)
        ref_points_reranked.append(refs)
    
    return pred_logits_reranked, pred_boxes_reranked, ref_points_reranked
