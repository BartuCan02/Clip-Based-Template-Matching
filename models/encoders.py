from torch import nn
import torch.nn.functional as F

from .backbone.sam.common import LayerNorm2d

def build_encoder(args):
    if args.encoder == 'original':
        return Backbone_Encoder

class Backbone_Encoder(nn.Module):
    def __init__(self, backbone, emb_dim):
        super(Backbone_Encoder, self).__init__()

        self.backbone = backbone
        self.num_channels = backbone.num_channels

    def forward(self, x):
        return self.backbone(x)
