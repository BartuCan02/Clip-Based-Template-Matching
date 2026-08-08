export PYTHONPATH="${NACLIP_PATH:-third_party/naclip}:$PYTHONPATH"

# Evaluates the single-mode fine-tune from scripts/train/TMR_RPINE_finetune.sh
# (text always present, no modality dropout). main.py loads the checkpoint by
# listing --logpath, so stage it into this run's output directory first.
CKPT="${TMR_FINETUNE_CKPT:-weights/TMR_RPINE_finetune_decoder_and_heads_30epoch/best_model.ckpt}"
LOGPATH=./weights/TMR_RPINE_finetune_decoder_and_heads_30epoch_eval
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
--use_naclip_heatmap \
--visualize \
--eval \
--input_mode box_and_text
