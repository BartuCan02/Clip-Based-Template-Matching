"""
Approach 2, training-free multiplicative modulation of the TMR feature map.

Gates the fused feature with the dense NaCLIP text-similarity heatmap M:

    F_cat <- F_cat * (1 + gamma * M),    gamma = 2

The idea is a soft spatial attention. Locations NaCLIP thinks match the prompt
get their response amplified by up to 3x, the rest stay as they are. No training
needed, channel count unchanged, and it is the identity when M == 0, so it drops
straight onto a frozen TMR checkpoint.

It does not work, and is kept here only so the result can be reproduced. The
problem is a distribution shift the head was never trained for. The decoders
expect F_cat at a particular scale, and a spatially varying gain in [1, 3] hits
a linear-plus-sigmoid presence head as a varying temperature rather than as
attention, so the score map turns confident wherever the prompt matches whether
or not the correlation evidence is there. It also hits all 1024 channels, the
projected image feature included, so the appearance pathway is corrupted along
with the matching one. matching_net.py supersedes this by concatenating M as a
1025th channel and re-training the head.

Drop-in replacement for models/matching_net.py, same `matching_net` class, so
copy it over that file before running inference_simpleFusion.py.
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
        """
        Run TMR, modulating the fused feature by the NaCLIP heatmap.

        naclip_map is the dense text-similarity map, broadcastable to f_cat's
        spatial size with values in [0, 1]. Pass None to skip modulation, which
        reduces this to plain TMR.

        log_stats prints the per-layer f_cat means before and after modulation.
        Those are the numbers behind the report's distribution-shift analysis,
        off by default because they fire on every forward pass.

        Returns (os, bs, f_TMs, f[0]): objectness maps, box regression maps,
        template-matching features and the first backbone level.
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