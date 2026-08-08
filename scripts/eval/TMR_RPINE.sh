export PYTHONPATH="${NACLIP_PATH:-third_party/naclip}:$PYTHONPATH"

# Baseline (no text): the flags below must match scripts/train/TMR_RPINE.sh, so
# no --use_naclip_heatmap here -- the baseline decoder takes 1024 channels, not
# 1025, and strict checkpoint loading fails if the two disagree.
CKPT="${TMR_BASELINE_CKPT:-weights/TMR_RPINE_baseline/best_model.ckpt}"
LOGPATH=./weights/TMR_RPINE_baseline_eval
mkdir -p "$LOGPATH"
cp -n "$CKPT" "$LOGPATH/best_model.ckpt"

CUDA_VISIBLE_DEVICES=0 python main.py \
--project_name "RPINE-CLIP-TMR" \
--datapath "${RPINE_DATA:-data/RPINE}" \
--logpath "$LOGPATH" \
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
--NMS_cls_threshold 0.1 \
--NMS_iou_threshold 0.5 \
--fusion \
--visualize \
--eval
