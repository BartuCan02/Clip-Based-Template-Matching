CUDA_VISIBLE_DEVICES=0 python demo.py \
--ckpt weights/TMR_RPINE/best_model.ckpt \
--port 6099 \
--text_prompt "arched windows" \
--clip_model "ViT-B/32" \
--clip_alpha 1.0 \
--clip_beta 0.5 \
--clip_topk 100 \
--clip_threshold 0.0