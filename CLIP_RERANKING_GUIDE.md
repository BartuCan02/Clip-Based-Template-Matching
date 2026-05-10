# CLIP Semantic Reranking Guide

## Overview

This document explains how to use CLIP-based semantic reranking to enhance TMR (Template Matching and Regression) few-shot pattern detection. CLIP reranking combines TMR's spatial localization confidence with semantic understanding of image regions using text prompts.

## Installation

First, install the additional CLIP dependency:

```bash
pip install -r requirements.txt
```

The requirements now include the official OpenAI CLIP library via:
```
git+https://github.com/openai/CLIP.git
```

## Usage

### Basic Usage with Positive Prompt Only

To evaluate the TMR model with CLIP semantic reranking on the RPINE dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --project_name "Few-Shot Pattern Detection with CLIP" \
  --datapath /path/to/RPINE \
  --logpath ./weights/TMR_RPINE_CLIP \
  --modeltype matching_net \
  --template_type roi_align \
  --dataset RPINE \
  --num_workers 1 \
  --batch_size 1 \
  --num_exemplars 1 \
  --backbone sam \
  --encoder original \
  --emb_dim 512 \
  --decoder_num_layer 1 \
  --decoder_kernel_size 3 \
  --feature_upsample \
  --positive_threshold 0.5 \
  --negative_threshold 0.5 \
  --NMS_cls_threshold 0.4 \
  --NMS_iou_threshold 0.5 \
  --fusion \
  --visualize \
  --nowandb \
  --eval \
  --use_clip \
  --text_prompt "apple tree" \
  --clip_model "ViT-B/32"
```

### Advanced Usage with Negative Prompt

For improved filtering of visually similar categories, use an optional negative prompt:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --project_name "Few-Shot Pattern Detection with CLIP" \
  --datapath /path/to/RPINE \
  --logpath ./weights/TMR_RPINE_CLIP \
  --modeltype matching_net \
  --template_type roi_align \
  --dataset RPINE \
  --num_workers 1 \
  --batch_size 1 \
  --num_exemplars 1 \
  --backbone sam \
  --encoder original \
  --emb_dim 512 \
  --decoder_num_layer 1 \
  --decoder_kernel_size 3 \
  --feature_upsample \
  --positive_threshold 0.5 \
  --negative_threshold 0.5 \
  --NMS_cls_threshold 0.4 \
  --NMS_iou_threshold 0.5 \
  --fusion \
  --visualize \
  --nowandb \
  --eval \
  --use_clip \
  --text_prompt "apple tree" \
  --negative_prompt "pear tree" \
  --clip_model "ViT-B/32" \
  --clip_alpha 0.7 \
  --clip_beta 0.3 \
  --clip_topk 100
```

## Command-Line Arguments

### CLIP Reranking Arguments

- `--use_clip`: Enable CLIP semantic reranking (eval/test only)
  - Default: False
  
- `--clip_model`: CLIP model variant to use
  - Default: "ViT-B/32"
  - Options: "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"
  
- `--text_prompt` (REQUIRED if using CLIP): Positive text prompt
  - Example: "apple tree", "crack in concrete", "flower bud"
  - This describes what you want to detect
  
- `--negative_prompt` (OPTIONAL): Negative prompt for disambiguation
  - Example: "pear tree" (when positive is "apple tree")
  - Helps reduce false positives from visually similar objects
  
- `--clip_alpha`: Weight for TMR confidence score in fusion
  - Default: 0.7
  - Range: [0, 1]
  - Higher values rely more on TMR localization
  
- `--clip_beta`: Weight for CLIP semantic score in fusion
  - Default: 0.3
  - Range: [0, 1]
  - Higher values rely more on CLIP semantics
  - Note: `alpha + beta` should ideally sum to 1.0
  
- `--clip_topk`: Number of top TMR predictions to process with CLIP
  - Default: 100
  - Lower values = faster inference, higher values = more comprehensive
  - Set to large number to process all predictions
  
- `--clip_threshold`: Minimum TMR score before CLIP processing
  - Default: 0.0
  - Can be used to filter out very low-confidence TMR predictions

## Score Fusion

The final score used for NMS and thresholding is computed as:

$$\text{final\_score} = \alpha \cdot \text{TMR\_score} + \beta \cdot \text{CLIP\_score}$$

where:
- `TMR_score` = TMR objectness probability (original confidence)
- `CLIP_score` = cosine similarity between image crop and text prompt
- `alpha` = weight for TMR score (default 0.7)
- `beta` = weight for CLIP score (default 0.3)

### With Negative Prompt

When a negative prompt is provided:

$$\text{CLIP\_score} = \max(0, \text{pos\_sim} - 0.3 \cdot \text{neg\_sim})$$

This ensures that regions similar to the negative prompt are penalized.

## Workflow

1. **TMR Detection**: Run standard TMR pipeline to get candidate boxes with confidence scores
2. **Top-k Filtering**: Keep only top-k predictions (default 100) for efficiency
3. **Box Cropping**: Extract image regions corresponding to predicted boxes
4. **CLIP Encoding**: 
   - Encode each cropped region with CLIP image encoder
   - Encode text prompt(s) with CLIP text encoder
5. **Similarity Computation**: Calculate cosine similarity between each crop and text
6. **Score Fusion**: Combine TMR and CLIP scores with learned weights
7. **Final Filtering**: Apply threshold and NMS using fused scores

## Performance Considerations

- **Inference Speed**: CLIP processing adds computational cost. Use `--clip_topk` to limit processing to top predictions
- **GPU Memory**: CLIP models require GPU memory. Use smaller models (ViT-B/32) for limited memory
- **Text Prompts**: More descriptive prompts generally work better (e.g., "red apple hanging from branch" vs. "apple")

## Expected Improvements

CLIP reranking is most beneficial when:
- ✅ Multiple visually similar categories exist in the dataset
- ✅ Dense, repeated patterns create false positives
- ✅ Cluttered backgrounds interfere with localization
- ✅ Clear semantic descriptions can distinguish targets

CLIP reranking may have limited impact when:
- ❌ Target objects are highly unique/distinctive
- ❌ Background is already clean and uncluttered
- ❌ Text descriptions are ambiguous or difficult to describe

## Model Selection

Different CLIP models provide different speed/accuracy tradeoffs:

| Model | Speed | Accuracy | Memory |
|-------|-------|----------|--------|
| ViT-B/32 | Fast | Good | Low |
| ViT-B/16 | Medium | Better | Medium |
| ViT-L/14 | Slow | Best | High |
| ViT-L/14@336px | Very Slow | Best+ | Very High |

## Debugging & Troubleshooting

### ImportError with CLIP

```
ImportError: clip-py is required for CLIP reranking
```

**Solution**: Install CLIP library directly:
```bash
pip install git+https://github.com/openai/CLIP.git
```

### CUDA Out of Memory

**Solution**: 
- Reduce `--clip_topk` to process fewer boxes
- Use smaller CLIP model: `--clip_model "ViT-B/32"`
- Reduce batch size to 1

### Poor Results with CLIP

**Check**:
1. Is the `--text_prompt` descriptive enough?
2. Try different weights: `--clip_alpha 0.5 --clip_beta 0.5`
3. Add a `--negative_prompt` to disambiguate
4. Increase `--clip_topk` to include more candidates

## Citation

If you use CLIP reranking with TMR, please cite:

```bibtex
@inproceedings{jo2025tmr,
  title={Few-Shot Pattern Detection via Template Matching and Regression},
  author={Jo, Eunchan and Kang, Dahyun and Kim, Sanghyun and Choi, Yunseon and Cho, Minsu},
  booktitle={ICCV},
  year={2025}
}

@inproceedings{radford2021clip,
  title={Learning transferable models for computer vision tasks},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  booktitle={ICML},
  year={2021}
}
```

## Next Steps

Possible future enhancements:
- [ ] Fine-tune CLIP on domain-specific data
- [ ] Learn adaptive weights (alpha, beta) per dataset
- [ ] Support for multiple positive prompts
- [ ] Integration with prompt engineering techniques
- [ ] Evaluation metrics specific to semantic reranking
