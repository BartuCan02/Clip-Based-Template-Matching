export PYTHONPATH="${NACLIP_PATH:-third_party/naclip}:$PYTHONPATH"

# main.py loads the checkpoint by listing --logpath, so stage it into this run's
# output directory first. Point $TMR_MULTIMODAL_CKPT at your own checkpoint if it
# does not live in the default training logpath.
CKPT="${TMR_MULTIMODAL_CKPT:-weights/TMR_RPINE_Multimodal_finetune_150epoch/best_model.ckpt}"
LOGPATH=./weights/TMR_RPINE_Multimodal_finetune_150epoch_text_only_eval
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
--input_mode text_only
