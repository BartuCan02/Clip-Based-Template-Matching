"""Approach 2: training-free multiplicative modulation of the TMR feature map.

This is the report's Approach 2 (Eq. 1, "multiplicative heatmap modulation"). It
gates the fused feature with a dense NaCLIP text-similarity heatmap M:

    F_cat <- F_cat * (1 + gamma * M),    gamma = GAMMA = 2

The intent is a soft spatial attention: locations NaCLIP considers consistent
with the prompt get their matching response amplified by up to 3x, others are
left unchanged. It needs no training, preserves the channel count, and reduces
to the identity when M == 0, so it can be dropped onto a frozen, pre-trained
TMR checkpoint.

This is a NEGATIVE RESULT and is kept for reproducibility, not for use. It
degrades detections. The reason is a distribution shift the head was never
trained for: the pre-trained decoders expect F_cat at a particular scale, and a
spatially varying gain in [1, 3] acts on a linear-plus-sigmoid presence head as
a spatially varying temperature rather than as attention, making the score map
more confident wherever the prompt matches regardless of the correlation
evidence. It is also applied to all 1024 channels, including the projected
image feature, so it corrupts the appearance pathway as well as the matching
one. Approach 3 (models/matching_net.py) supersedes it by concatenating M as a
1025th channel and re-training the head instead.

This module is a drop-in replacement for models/matching_net.py: it defines the
same `matching_net` class, so copy it over that file before running
inference_simpleFusion.py.
"""

import torch
from torch import nn
import torch.nn.functional as F

from .template_matching import TemplateMatching
from .regression_head import Decoder_model, ObjectnessHead, BboxesHead
from .encoders import build_encoder

# Modulation strength gamma in F_cat <- F_cat * (1 + gamma * M). The report
# evaluates gamma = 2, i.e. a gain of up to 3x at fully prompt-consistent
# locations.
GAMMA = 2.0


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
            self.decoder_o = self.decoder_model(1 + self.emb_dim if self.fusion else 1, decoder_num_layer, decoder_kernel_size)
            self.decoder_b = self.decoder_model(1 + self.emb_dim if self.fusion else 1, decoder_num_layer, decoder_kernel_size) if self.box_reg else None
        else:
            self.decoder_o = self.decoder_model(2 * self.emb_dim if self.fusion else self.emb_dim, decoder_num_layer, decoder_kernel_size)
            self.decoder_b = self.decoder_model(2 * self.emb_dim if self.fusion else self.emb_dim, decoder_num_layer, decoder_kernel_size) if self.box_reg else None

        self.objectness_head = ObjectnessHead(self.decoder_o.out_channels)
        self.ltrbs_head = BboxesHead(self.decoder_b.out_channels) if self.box_reg else None

    def forward(self, sample, exemplars, naclip_map=None, log_stats=False, **kwargs):
        """Run TMR, modulating the fused feature by the NaCLIP heatmap.

        Args:
            sample: input image batch, [B, 3, H, W].
            exemplars: support exemplar box(es) defining the pattern to find.
            naclip_map: dense NaCLIP text-similarity heatmap M, broadcastable to
                f_cat's spatial size, values in [0, 1]. When None, no modulation
                is applied and this reduces to plain TMR.
            log_stats: print per-layer f_cat means before and after modulation.
                These are the statistics reported in the paper's Approach 2
                analysis (f_cat means rising by roughly 2x on prompt-consistent
                images); off by default because it prints on every forward pass.

        Returns:
            (os, bs, f_TMs, f[0]): objectness maps, box regression maps,
            template-matching features, and the first backbone feature level.
        """
        f = self.encoder(sample)
        if not isinstance(f, list):
            f = [f]

        if self.feature_upsample:
            f = [F.interpolate(f_, scale_factor=2, mode='bilinear', align_corners=False) for f_ in f]       

        os, bs, f_TMs = [], [], []
        for i in range(len(f)):
            
            fp = self.input_proj[i](f[i])

            if self.matcher is None:
                f_TM = fp
            else:
                f_TM = self.matcher(fp, exemplars)

            if self.fusion:
                f_cat = torch.cat([fp, f_TM], dim=1)
            else:
                f_cat = f_TM

            # Eq. 1: F_cat <- F_cat * (1 + gamma * M). Note this hits all
            # channels of f_cat, the projected image feature included, which is
            # part of why the approach fails (see module docstring).
            if naclip_map is not None:
                mean_before = f_cat.mean().item() if log_stats else None

                f_cat = f_cat * (1 + GAMMA * naclip_map.to(f_cat.device))

                if log_stats:
                    print(f"--- Layer {i} ---")
                    print(f"f_cat mean (before): {mean_before:.6f}")
                    print(f"f_cat mean (after):  {f_cat.mean().item():.6f}")
                    print(f"NaCLIP map mean:     {naclip_map.mean().item():.6f}")

            if self.box_reg:
                f_box = self.decoder_b(f_cat)
                b = self.ltrbs_head(f_box)
            else:
                b = None

            f_obj = self.decoder_o(f_cat)
            o = self.objectness_head(f_obj)

            os.append(o)
            bs.append(b)
            f_TMs.append(F.relu(f_TM))



        return os, bs, f_TMs, f[0]