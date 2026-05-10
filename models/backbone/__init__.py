from .resnet import *
from .sam.sam import Sam_Backbone

def build_backbone(args):
    if args.backbone == 'resnet50':
        backbone = resnet50(args.dilation)
    elif args.backbone == 'resnet50_layer1':
        backbone = resnet50_layer1(args.dilation)
    elif args.backbone == 'resnet50_layer2':
        backbone = resnet50_layer2(args.dilation)
    elif args.backbone == 'resnet50_layer3':
        backbone = resnet50_layer3(args.dilation)

    elif args.backbone == 'resnet50_layer1_FRZ':
        backbone = resnet50_layer1_FRZ(args.dilation)
    elif args.backbone == 'resnet50_layer2_FRZ':
        backbone = resnet50_layer2_FRZ(args.dilation)
    elif args.backbone == 'resnet50_layer3_FRZ':
        backbone = resnet50_layer3_FRZ(args.dilation)

    elif args.backbone == 'sam':
        backbone = Sam_Backbone(requires_grad=False, model_type = "vit_h")

    return backbone