from .backbone import build_backbone
from .matching_net import matching_net

def build_model(args):
    backbone = build_backbone(args)

    if args.modeltype == 'matching_net':
        model = matching_net(backbone, args)

    return model