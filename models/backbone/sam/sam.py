import os
import torch
from torch import nn
import torch.nn.functional as F

from .sam_ViT import ImageEncoderViT
from functools import partial

class Sam_Backbone(nn.Module):
    def __init__(
            self,
            requires_grad: bool,
            model_path: str = "./checkpoints",
            model_type: str = "vit_h",
    ):

        super(Sam_Backbone, self).__init__()

        # vit_h parameters
        if model_type == "vit_h":
            self.encoder_embed_dim=1280
            encoder_depth=32
            encoder_num_heads=16
            encoder_global_attn_indexes=[7, 15, 23, 31]
        # vit_b parameters
        elif model_type == "vit_b":
            self.encoder_embed_dim=768
            encoder_depth=12
            encoder_num_heads=12
            encoder_global_attn_indexes=[2, 5, 8, 11]

        # common parameters
        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size
        vit_encoder = ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=self.encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )
        self.backbone = vit_encoder

        self.num_channels = prompt_embed_dim

        if model_path is not None:
            if model_type == "vit_h":
                checkpoint_path = os.path.join(model_path, "sam_hq_vit_h.pth")
            elif model_type == "vit_b":
                checkpoint_path = os.path.join(model_path, "sam_hq_vit_b.pth")
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state_dict = {k.replace("image_encoder.", ""): v for k, v in checkpoint.items() if "image_encoder" in k}
            self.backbone.load_state_dict(state_dict)

            for n, param in self.named_parameters():
                param.requires_grad_(requires_grad)

    def forward(self, x):
        x = self.backbone.patch_embed(x)
        if self.backbone.pos_embed is not None:
            if self.backbone.pos_embed.shape[1:] != x.shape[1:]:
                pos_emb = F.interpolate(self.backbone.pos_embed.permute(0, 3, 1, 2), size=x.shape[1:3], mode='bilinear').permute(0, 2, 3, 1)
            else:
                pos_emb = self.backbone.pos_embed

            # if self.backbone.pos_embed.shape[1:] < x.shape[1:]:
            #     upsample_pos_emb = nn.UpsamplingBilinear2d(scale_factor=1.5)
            #     pos_emb = upsample_pos_emb(self.backbone.pos_embed.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            # elif self.backbone.pos_embed.shape[1:] > x.shape[1:]:
            #     downsample_pos_emb = nn.MaxPool2d(kernel_size=2)
            #     pos_emb = downsample_pos_emb(self.backbone.pos_embed.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            # else:
            #     pos_emb = self.backbone.pos_embed
            x = x + pos_emb

        interm_embeddings = []
        for blk in self.backbone.blocks:
            x = blk(x)
            if blk.window_size == 0:
                interm_embeddings.append(x)
        image_embeddings = self.backbone.neck(x.permute(0, 3, 1, 2))

        return image_embeddings
    
    def forward_interm(self, x):
        x = self.backbone.patch_embed(x)
        if self.backbone.pos_embed is not None:
            if self.backbone.pos_embed.shape[1:] != x.shape[1:]:
                pos_emb = F.interpolate(self.backbone.pos_embed.permute(0, 3, 1, 2), size=x.shape[1:3], mode='bilinear').permute(0, 2, 3, 1)
            else:
                pos_emb = self.backbone.pos_embed

            x = x + pos_emb
        
        interm_embeddings = []
        for blk in self.backbone.blocks:
            x = blk(x)
            if blk.window_size == 0:
                interm_embeddings.append(x)

        return interm_embeddings
