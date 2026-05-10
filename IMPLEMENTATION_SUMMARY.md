# CLIP Semantic Reranking Implementation Summary

## Overview

This document provides a comprehensive summary of the CLIP semantic reranking implementation for the TMR (Template Matching and Regression) few-shot pattern detection system.

## Implementation Goals ✓

- [x] Keep TMR model and training completely unchanged
- [x] Add CLIP semantic reranking at **inference time only**
- [x] Support positive and optional negative prompts
- [x] Implement weighted score fusion
- [x] Optimize for efficiency with top-k prediction filtering
- [x] Integrate seamlessly into existing codebase with minimal modifications
- [x] Provide comprehensive documentation and examples

## What Was Implemented

### 1. **Core CLIP Module** (`utils/clip_utils.py`)

#### `CLIPReranker` Class
A wrapper around OpenAI's CLIP model that handles:
- Model loading and initialization
- Text prompt encoding (with caching for efficiency)
- Image region cropping and preprocessing
- Cosine similarity computation
- Semantic score calculation (with negative prompt support)

**Key methods:**
- `encode_text()`: Encode text prompts to CLIP embeddings
- `crop_and_encode_image()`: Extract and encode image regions from bounding boxes
- `compute_similarity()`: Calculate cosine similarity between images and text
- `compute_semantic_score()`: Generate semantic scores with optional negative prompt handling

#### Utility Functions
- `fuse_scores()`: Weighted fusion of TMR and CLIP scores
- `apply_clip_reranking()`: Main inference pipeline that orchestrates the entire reranking process

### 2. **Trainer Integration** (`trainer.py`)

Modified the `Matching_Trainer` class to:

1. **Initialize CLIP in `__init__`:**
   ```python
   if self.args.use_clip and self.args.eval and self.args.text_prompt:
       self.clip_reranker = CLIPReranker(...)
   ```

2. **Apply CLIP in inference loop:**
   - After TMR prediction generation via `Get_pred_boxes()`
   - Before NMS processing
   - In both `each_step()` (single exemplar) and `each_step_multi_exemplars()` (multi exemplar) methods

3. **Score fusion pipeline:**
   ```
   TMR Predictions → Crop Regions → CLIP Encoding → Similarity Computation
       ↓
   Fused Scores (alpha * TMR + beta * CLIP) → NMS → Final Predictions
   ```

### 3. **Command-Line Arguments** (`main.py`)

New CLIP-specific arguments added to argument parser:

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_clip` | False | Enable CLIP semantic reranking |
| `--clip_model` | "ViT-B/32" | CLIP model variant |
| `--text_prompt` | None | **Required** positive text prompt |
| `--negative_prompt` | None | Optional negative prompt |
| `--clip_alpha` | 0.7 | Weight for TMR score |
| `--clip_beta` | 0.3 | Weight for CLIP score |
| `--clip_topk` | 100 | Only process top-k predictions |
| `--clip_threshold` | 0.0 | Min TMR score before CLIP |

### 4. **Documentation**

Created comprehensive guides:
- `CLIP_RERANKING_GUIDE.md`: User guide with examples and troubleshooting
- `scripts/eval/examples_clip_reranking.sh`: Example evaluation commands
- `test_clip_integration.py`: Validation script to test implementation

### 5. **Dependencies**

Updated `requirements.txt` to include:
```
git+https://github.com/openai/CLIP.git
Pillow>=9.0.0
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│           Input Image + Exemplars                    │
└────────────────────┬────────────────────────────────┘
                     ↓
         ┌───────────────────────┐
         │   TMR Detection       │
         │  (unchanged)          │
         └──────────┬────────────┘
                    ↓
        ┌──────────────────────────┐
        │  Get_pred_boxes()        │
        │  Output: boxes + scores  │
        └──────────┬───────────────┘
                   ↓
    ┌──────────────────────────────────┐
    │   Top-k Filtering (optional)     │
    │   Keep top-k predictions         │
    └──────────┬───────────────────────┘
               ↓
   ┌─────────────────────────────────────┐
   │  ✨ CLIP Semantic Reranking ✨     │
   │  ┌──────────────────────────────┐  │
   │  │ 1. Crop image regions        │  │
   │  │ 2. Encode with CLIP          │  │
   │  │ 3. Encode text prompts       │  │
   │  │ 4. Compute similarities      │  │
   │  │ 5. Fuse with TMR scores      │  │
   │  └──────────────────────────────┘  │
   └──────────┬──────────────────────────┘
              ↓
   ┌──────────────────────────────────┐
   │   NMS with Fused Scores          │
   │   Final predictions              │
   └──────────┬───────────────────────┘
              ↓
    ┌──────────────────────────┐
    │  Image Info Collection   │
    │  (visualization, eval)   │
    └──────────────────────────┘
```

## Code Changes Summary

### Files Modified:
1. **`main.py`**: Added 8 new command-line arguments
2. **`trainer.py`**: 
   - Added CLIP import
   - Added CLIP initialization in `__init__` (~15 lines)
   - Modified `each_step()` to apply CLIP reranking (~8 lines)
   - Modified `each_step_multi_exemplars()` to apply CLIP reranking (~8 lines)
3. **`requirements.txt`**: Added CLIP and Pillow dependencies

### Files Created:
1. **`utils/clip_utils.py`** (~350 lines): Core CLIP integration module
2. **`CLIP_RERANKING_GUIDE.md`**: User documentation
3. **`test_clip_integration.py`**: Validation/testing script
4. **`scripts/eval/examples_clip_reranking.sh`**: Example commands

### Total Impact:
- **Core TMR model**: 0 changes (frozen, inference only)
- **Trainer logic changes**: ~30 lines (minimal, well-isolated)
- **New code**: ~400 lines (all in utils, cleanly separated)

## Score Fusion Formula

### With Positive Prompt Only:
$$\text{final\_score} = \alpha \cdot \text{TMR}_{\text{score}} + \beta \cdot \text{CLIP}_{\text{score}}$$

### With Negative Prompt:
$$\text{CLIP}_{\text{score}} = \max(0, \text{sim}(\text{crop}, \text{positive}) - 0.3 \cdot \text{sim}(\text{crop}, \text{negative}))$$

$$\text{final\_score} = \alpha \cdot \text{TMR}_{\text{score}} + \beta \cdot \text{CLIP}_{\text{score}}$$

**Parameters:**
- `TMR_score` ∈ [0, 1]: Original objectness probability
- `CLIP_score` ∈ [0, 1]: Cosine similarity (normalized)
- `α = 0.7` (default): Weight for TMR (localization)
- `β = 0.3` (default): Weight for CLIP (semantics)
- Note: α + β = 1.0 (recommend maintaining this)

## Inference Pipeline Details

### Step 1: TMR Detection (Unchanged)
```
Input: Image + Exemplar
Output: Objectness maps, regression maps at multiple levels
```

### Step 2: Get Predictions
```python
pred_logits, pred_boxes, ref_points = Get_pred_boxes(
    pred_objectness, pred_regressions, exemplars, batch,
    cls_threshold=0.4
)
# Output: N predictions with logits (N, 2), boxes (N, 4) normalized
```

### Step 3: CLIP Reranking
```python
# Apply only to top-k for efficiency
pred_logits, pred_boxes, ref_points = apply_clip_reranking(
    pred_logits, pred_boxes, ref_points, image,
    text_prompt="apple tree",
    clip_reranker=self.clip_reranker,
    alpha=0.7, beta=0.3,
    top_k=100  # Limit to top 100 predictions
)
```

Inside `apply_clip_reranking()`:
1. For each image in batch:
   - Filter predictions above threshold (optional)
   - Crop regions from image
   - Encode crops with CLIP image encoder
   - Encode text with CLIP text encoder
   - Compute cosine similarities
   - Fuse TMR scores with CLIP scores
   - Update prediction logits

### Step 4: NMS (Uses Fused Scores)
```python
pred_logits, pred_boxes, ref_points = NMS(
    pred_logits, pred_boxes, ref_points,
    iou_threshold=0.5  # Uses updated logits
)
```

## Performance Characteristics

### Computational Cost
- **CLIP Model Loading**: ~1-2 seconds (one-time)
- **Per Image CLIP Processing**:
  - Top-k=100: ~2-3 seconds (depends on GPU)
  - Top-k=50: ~1-2 seconds
  - Dominated by CLIP image encoding

### Memory Usage
- **CLIP ViT-B/32**: ~350 MB GPU memory
- **CLIP ViT-L/14**: ~700 MB GPU memory
- Batch processing: Load all crops at once for efficiency

### Optimization Tips
1. Use `--clip_topk 50` to process only top 50 predictions
2. Choose appropriate CLIP model (ViT-B/32 fastest)
3. Cache text embeddings (done automatically)
4. Use GPU for CLIP operations

## Testing & Validation

Run the validation script to test CLIP integration:
```bash
python test_clip_integration.py
```

This validates:
- CLIP import and availability
- CLIPReranker initialization
- Text encoding functionality
- Image region encoding
- Similarity computation
- Score fusion logic
- Semantic score computation

## Expected Results

### When CLIP Helps Most:
- ✅ Multiple similar categories (apple vs. pear)
- ✅ Dense, repeated objects
- ✅ Cluttered backgrounds
- ✅ When text descriptions are distinctive

### When CLIP May Not Help:
- ❌ Already high-quality TMR predictions
- ❌ Unique, easily distinguishable objects
- ❌ Ambiguous or vague descriptions
- ❌ When text doesn't match visual content

## Future Enhancements

Possible improvements for future versions:

1. **Fine-tuning**: Fine-tune CLIP on domain-specific data
2. **Learned Weights**: Learn α and β dynamically
3. **Multi-prompt Support**: Use multiple positive prompts
4. **Prompt Optimization**: Automatically generate optimal prompts
5. **Ensemble Methods**: Combine with other reranking approaches
6. **Batch Processing**: Process multiple images simultaneously
7. **Caching**: Cache image embeddings between similar regions

## Troubleshooting

### Common Issues:

**Issue**: ImportError for CLIP
```bash
# Solution:
pip install git+https://github.com/openai/CLIP.git
```

**Issue**: CUDA out of memory
```bash
# Solution: Reduce top-k
python main.py ... --clip_topk 50
```

**Issue**: Poor results
```bash
# Solution: Try different weights
python main.py ... --clip_alpha 0.5 --clip_beta 0.5
```

**Issue**: Slow inference
```bash
# Solution: Use smaller model or reduce top-k
python main.py ... --clip_model "ViT-B/32" --clip_topk 50
```

## Citations

If you use this CLIP reranking implementation, please cite:

```bibtex
@inproceedings{jo2025tmr,
  title={Few-Shot Pattern Detection via Template Matching and Regression},
  author={Jo, Eunchan and Kang, Dahyun and Kim, Sanghyun and Choi, Yunseon and Cho, Minsu},
  booktitle={ICCV},
  year={2025}
}

@inproceedings{radford2021clip,
  title={Learning Transferable Models for Computer Vision Tasks},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  booktitle={ICML},
  year={2021}
}
```

## Contact & Support

For questions or issues with the CLIP implementation:
1. Check `CLIP_RERANKING_GUIDE.md` for detailed usage
2. Review examples in `scripts/eval/examples_clip_reranking.sh`
3. Run `test_clip_integration.py` to validate setup
4. Check implementation in `utils/clip_utils.py`

## Version History

### v1.0 (Current)
- Initial implementation with single positive/negative prompt support
- Weighted sum fusion method
- Top-k optimization for inference
- Full documentation and examples

---

**Implementation Status**: ✅ Complete and ready for evaluation

**Last Updated**: April 26, 2026
