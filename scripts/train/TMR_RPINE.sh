export PYTHONPATH="/home/zhox/NACLIP:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--project_name "RPINE-CLIP-TMR" \
--datapath /home/zhox/Clip-Based-Template-Matching/data/RPINE \
--logpath ./weights/TMR_RPINE_baseline_trainval \
--modeltype matching_net \
--template_type roi_align \
--dataset RPINE \
--num_workers 4 \
--max_epochs 200 \
--batch_size 4 \
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
--lr 1e-4 \
--lr_backbone 0 \
--lr_drop \
--visualize \
#--multi_gpu