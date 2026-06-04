import os
import argparse
import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger,WandbLogger

from callbacks import CustomCheckpoint
from datamodules import build_datamodule
from trainer import Matching_Trainer

def config_parser():
    parser = argparse.ArgumentParser(description="Matching Network code")

    # seed
    parser.add_argument('--seed', default=42, type=int)

    # log setting (wandb)
    parser.add_argument('--project_name', type=str, default="RPINE-TMR", help='Name of project (for wandb)')
    parser.add_argument("--logpath", type=str, default="./outputs/default", help="/Path/to/output/logs/and/checkpoints")
    parser.add_argument('--nowandb', action='store_true', help='Flag not to use wandb')
    parser.add_argument("--AP_term", default=5, type=int, help='If this value is N, AP is calculated every N epochs')
    parser.add_argument('--best_model_count', action='store_true', help='Flag to save best model using counting metric (MAE)')

    # dataset setting
    parser.add_argument('--datapath', type=str, default='/home/', help='Dataset path')
    parser.add_argument('--dataset', type=str, default='RPINE', help='Dataset type, e.g., RPINE, FSCD147')
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--num_exemplars", default=1, type=int, help='Number of support exemplars')
    parser.add_argument("--image_size", default=1024, type=int)

    # training setting
    parser.add_argument('--resume', action='store_true', help='Flag for resume')
    parser.add_argument("--max_epochs", default=30, type=int)
    parser.add_argument("--finetune_from", type=str, default=None)
    parser.add_argument('--multi_gpu', action='store_true', help='Flag to use multi_gpu')
    parser.add_argument("--finetune_decoders_and_heads_only", action="store_true")
    

    # optimizer setting
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument("--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm")
    parser.add_argument('--lr_drop', action='store_true')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)

    # test setting
    parser.add_argument('--eval', action='store_true', help='Flag for evaluation')

    # visualization setting
    parser.add_argument('--visualize', action='store_true', help='Flag to store visual outputs')

    # model setting
    parser.add_argument('--modeltype', type=str, default="matching_net", help='Type of model')

    parser.add_argument('--emb_dim', default=512, type=int, help='Embedding dimension')
    parser.add_argument("--no_matcher", action='store_true', help="If true, we don't use matching module")
    parser.add_argument("--squeeze", action='store_true', help="If true, we use matching feature with channel 1")
    parser.add_argument("--fusion", action='store_true', help="If true, we use a fusion layer to combine the features from the backbone and the template matching module")
    parser.add_argument("--positive_threshold", default=0.7, type=float, help="Threshold for positive samples")
    parser.add_argument("--negative_threshold", default=0.7, type=float, help="Threshold for negative samples")
    parser.add_argument("--NMS_cls_threshold", default=0.1, type=float, help="Threshold for NMS classificaiton score")
    parser.add_argument("--NMS_iou_threshold", default=0.15, type=float, help="Threshold for NMS Iou")
    parser.add_argument("--refine_box", action='store_true', help="If true, we use SAM decoder for box refinement")
    parser.add_argument("--ablation_no_box_regression", action='store_true', help="If true, we don't regress box parameters. Insted we use template size as box width, height parameter")
    parser.add_argument('--template_type', type=str, default='roi_align', help='template extraction algorithm Type')
    parser.add_argument("--feature_upsample", action='store_true', help="If true, feature upsample for template matching")
    parser.add_argument('--eval_multi_scale', action='store_true', help='multi scale processing for evaluation')
    parser.add_argument('--regression_scaling_imgsize', action='store_true')
    parser.add_argument('--regression_scaling_WH_only', action='store_true')
    parser.add_argument("--focal_loss", action='store_true', help='Flag to use focal loss')

    # model - backbone setting
    parser.add_argument("--backbone", default="resnet50", type=str, help="Name of the backbone to use")
    parser.add_argument("--encoder", default="original", type=str, help="Name of the encoder type to use")
    parser.add_argument("--dilation", default=True, help="If true, we replace stride with dilation in the last convolutional block (DC5)")

    # model - head setting
    parser.add_argument("--decoder_num_layer", default=1, type=int)
    parser.add_argument("--decoder_kernel_size", default=3, type=int)

    # NACLIP setting
    parser.add_argument('--use_naclip_heatmap', action='store_true',help='Use NaCLIP heatmap fusion')
    parser.add_argument('--clip_model', type=str, default="ViT-B/32", help='CLIP model variant (e.g., ViT-B/32, ViT-L/14)')
    parser.add_argument("--naclip_logit_scale", type=float, default=40.0)
    parser.add_argument("--naclip_gaussian_std", type=float, default=5.0)
    parser.add_argument('--text_prompt', type=str, default=None, help='Positive text prompt for CLIP semantic reranking')
    parser.add_argument('--negative_prompt', type=str, default=None, help='Optional negative text prompt for CLIP semantic reranking')

    # CLIP semantic reranking setting
    parser.add_argument('--use_clip', action='store_true', help='Flag to enable CLIP semantic reranking (eval only)')
    parser.add_argument('--clip_alpha', type=float, default=0.7, help='Weight for TMR confidence in score fusion (default 0.7)')
    parser.add_argument('--clip_beta', type=float, default=0.3, help='Weight for CLIP semantic score in fusion (default 0.3)')
    parser.add_argument('--clip_topk', type=int, default=100, help='Only apply CLIP to top-k TMR predictions for efficiency')
    parser.add_argument('--clip_threshold', type=float, default=0.0, help='Minimum TMR score threshold before CLIP processing')

    args = parser.parse_args()
    return args

def main(args):
    seed_everything(args.seed)
    torch.set_float32_matmul_precision('medium')

    # log setting
    project_name = args.project_name
    run_name = os.path.basename(args.logpath)

    # Callbacks setting
    Checkpoint_callback = CustomCheckpoint(args)
    LR_Monitor_callback = LearningRateMonitor(logging_interval='step')
    Progress_bar_callback = TQDMProgressBar(refresh_rate=10)

    callbacks = [LR_Monitor_callback, Progress_bar_callback, Checkpoint_callback]

    # Load datamodule
    Datamodule = build_datamodule(args)

    # Load model
    Model = Matching_Trainer(args, Datamodule)
    M_checkpoint = None

    os.makedirs(os.path.join(args.logpath,'wandb'), exist_ok=True)
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        strategy="ddp" if args.multi_gpu else "auto",
        devices=-1 if args.multi_gpu else 1 if torch.cuda.is_available() else torch.cpu.device_count(),
        logger=CSVLogger(save_dir=args.logpath) if args.nowandb else WandbLogger(name=run_name,entity="nlp-project-tum", save_dir=args.logpath, project=project_name),
        callbacks=callbacks,
        num_sanity_val_steps=0,
        gradient_clip_val=args.clip_max_norm,
        deterministic=False if (args.refine_box or args.template_type == 'roi_align' or args.feature_upsample) else True,
        sync_batchnorm=True if args.multi_gpu else False
    )

    # Evaluate mode
    if args.eval:
        if args.refine_box:
            M_checkpoint = Checkpoint_callback.modelpath
            Model = Matching_Trainer.load_from_checkpoint(M_checkpoint, args=args, datamodule=Datamodule, strict=False)
            trainer.test(model=Model, datamodule=Datamodule)
        else:
            M_checkpoint = Checkpoint_callback.modelpath
            Model = Matching_Trainer.load_from_checkpoint(M_checkpoint, args=args, datamodule=Datamodule, strict=True)
            trainer.test(model=Model, datamodule=Datamodule)
    # Train mode
    else:
        if args.finetune_from is not None:
            M_checkpoint = args.finetune_from
            Model = Matching_Trainer.load_from_checkpoint(M_checkpoint,args=args,datamodule=Datamodule,strict=False)
            trainer.fit(model=Model,datamodule=Datamodule,ckpt_path=None)

        elif args.resume:
            M_checkpoint = Checkpoint_callback.lastmodelpath
            Model = Matching_Trainer.load_from_checkpoint(M_checkpoint,args=args,datamodule=Datamodule,strict=True)
            trainer.fit(model=Model,datamodule=Datamodule,ckpt_path=M_checkpoint)
        
        else:
            trainer.fit(model=Model,datamodule=Datamodule,ckpt_path=None)

if __name__ == "__main__":
    args = config_parser()
    main(args)