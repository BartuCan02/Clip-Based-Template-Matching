import torch
from torch import nn
import torch.nn.functional as F

from .template_matching import TemplateMatching
from .regression_head import Decoder_model, ObjectnessHead, BboxesHead
from .encoders import build_encoder

# Concatenates naclip heatmapper with the tmr's feature map and tmr's heatmap

class matching_net(nn.Module):
    def __init__(self, backbone, args):
        super(matching_net, self).__init__()

        self.args = args
        self.emb_dim = args.emb_dim
        self.fusion = args.fusion
        self.box_reg = not args.ablation_no_box_regression
        self.encoder = build_encoder(args)(backbone, args.emb_dim)
        self.decoder_model = Decoder_model

        self.feature_upsample = args.feature_upsample

        if args.no_matcher:
            self.matcher = None
        else:
            self.matcher = TemplateMatching(args.template_type, args.squeeze)

        if isinstance(self.encoder.num_channels, list):
            self.input_proj = nn.ModuleList([nn.Conv2d(channel, self.emb_dim, kernel_size=1) for channel in self.encoder.num_channels])
        else:
            self.input_proj = nn.ModuleList([nn.Conv2d(self.encoder.num_channels, self.emb_dim, kernel_size=1)])

        decoder_num_layer = args.decoder_num_layer
        decoder_kernel_size = args.decoder_kernel_size
        if args.squeeze:
            decoder_in_channels = 1 + self.emb_dim if self.fusion else 1
            if args.use_naclip_heatmap:
                decoder_in_channels += 1
            
            self.decoder_o = self.decoder_model(decoder_in_channels, decoder_num_layer, decoder_kernel_size)
            self.decoder_b = self.decoder_model(decoder_in_channels, decoder_num_layer, decoder_kernel_size) if self.box_reg else None
        else:
            decoder_in_channels = 2 * self.emb_dim if self.fusion else self.emb_dim
            if args.use_naclip_heatmap:
                decoder_in_channels += 1

            self.decoder_o = self.decoder_model(decoder_in_channels,decoder_num_layer,decoder_kernel_size)
            self.decoder_b = self.decoder_model(decoder_in_channels,decoder_num_layer,decoder_kernel_size) if self.box_reg else None

        self.objectness_head = ObjectnessHead(self.decoder_o.out_channels)
        self.ltrbs_head = BboxesHead(self.decoder_b.out_channels) if self.box_reg else None

    def forward(self, sample, exemplars, naclip_heatmap= None, **kwargs):
        """
        Forward pass of TMR.

        sample:
            Input image tensor.
            Shape: [B, 3, 1024, 1024]

        exemplars:
            Support exemplar boxes.
        """

        # 1. Extract backbone feature map from the input image.
        # f: [B, 256, 64, 64]
        f = self.encoder(sample)

        # The code supports both single-scale and multi-scale features.
        # If the encoder returns only one tensor, wrap it into a list.
        # After this:
        # f = [feature_level_0]
        # f[0]: [B, 256, 64, 64]
        if not isinstance(f, list):
            f = [f]

        # 2. Upsample feature maps spatially.
        # The paper upsamples SAM features from 64x64 to 128x128
        # to get denser predictions.
        # f[0]: [B, 256, 128, 128]
        if self.feature_upsample:
            f = [
                F.interpolate(
                    f_,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=False
                )
                for f_ in f
            ]

        # Output lists for each feature level.
        # os:
        #   objectness / presence maps
        #   [B, 1, H, W]
        # bs:
        #   box regression maps
        #   [B, 4, H, W]
        # f_TMs:
        #   template matching feature maps for debugging/visualization
        #   [B, emb_dim, H, W]
        
        os, bs, f_TMs = [], [], []

        # 3. Loop over feature levels.
        for i in range(len(f)):

            # 4. Project backbone features to embedding dimension.
            # input_proj is a 1x1 Conv2d.
            # With emb_dim=512:
            # fp: [B, 512, 128, 128]
            # This is the projected image feature F.
            fp = self.input_proj[i](f[i])

            # 5. Compute template matching feature.
            # f_TM: [B, 512, H, W]
            if self.matcher is None:
                f_TM = fp
            else:
                f_TM = self.matcher(fp, exemplars)

            # 6. Fuse original image feature and template matching feature.
            # fp:   [B, 512, H, W]
            # f_TM: [B, 512, H, W]
            # f_cat:
            # [B, 1024, H, W]
            
            if self.fusion:
                f_cat = torch.cat([fp, f_TM], dim=1)
            else:
                f_cat = f_TM
            
            # 7. Concatenate the NaCLIP heatmap to f_cat.
            # If fusion=True:  f_cat = [fp, f_TM, naclip_heatmap]
            # If fusion=False: f_cat = [f_TM, naclip_heatmap]
            
            if naclip_heatmap is not None:
                naclip_heatmap = naclip_heatmap.to(f_cat.device)
                
                # Makes sure correct dim (ToDo: If this line is unncessary remove it aftet the dataset test) 
                if naclip_heatmap.dim() == 3:
                    naclip_heatmap = naclip_heatmap.unsqueeze(1)
                
                # Makes sure correct dim (ToDo: If this line is unncessary remove it aftet the dataset test) 
                if naclip_heatmap.shape[-2:] != f_cat.shape[-2:]:
                    naclip_heatmap = F.interpolate(
                        naclip_heatmap,
                        size=f_cat.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )

                # f_cat:
                # [B, 1025, H, W]
                f_cat = torch.cat([f_cat, naclip_heatmap], dim=1)

            # 8. Box regression branch.
            # b: [B, 4, H, W]
            if self.box_reg:
                f_box = self.decoder_b(f_cat)
                b = self.ltrbs_head(f_box)
            else:
                b = None

            # 9. Objectness / presence branch.
            # o: [B, 1, H, W]
            f_obj = self.decoder_o(f_cat)
            o = self.objectness_head(f_obj)

            os.append(o)
            bs.append(b)

            # Store template matching feature for visualization/debugging.
            # ReLU removes negative values.
            #
            # f_TM:
            # [B, 512, H, W]
            f_TMs.append(F.relu(f_TM))

        # Return:
        #
        # os:
        #   list of objectness maps
        #   usually one element: [[B, 1, H, W]]
        #
        # bs:
        #   list of box regression maps
        #   usually one element: [[B, 4, H, W]]
        #
        # f_TMs:
        #   list of template matching feature maps
        #   usually one element: [[B, 512, H, W]]
        #
        # f[0]:
        #   original encoder feature after optional upsampling,
        #   before input projection.
        #   Example: [B, 256, 128, 128]

        return os, bs, f_TMs, f[0]