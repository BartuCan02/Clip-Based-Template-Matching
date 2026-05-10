import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

from collections import OrderedDict

class resnet50(nn.Module):
    def __init__(self, dilation: bool):
        super(resnet50, self).__init__()

        resnet = models.resnet50(
            replace_stride_with_dilation=[False, False, dilation], 
            weights=models.ResNet50_Weights.IMAGENET1K_V1, 
            norm_layer=FrozenBatchNorm2d
        )

        self.backbone = resnet
        self.num_channels = 2048

        self.backbone.fc.requires_grad_(False)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        return x
    
class resnet50_layer1(nn.Module):
    def __init__(self, dilation: bool):
        super(resnet50_layer1, self).__init__()

        resnet = models.resnet50(
            replace_stride_with_dilation=[False, False, dilation], 
            weights=models.ResNet50_Weights.IMAGENET1K_V1, 
            norm_layer=FrozenBatchNorm2d
        )

        self.backbone = resnet
        self.num_channels = 256

        self.backbone.layer2.requires_grad_(False)
        self.backbone.layer3.requires_grad_(False)
        self.backbone.layer4.requires_grad_(False)
        self.backbone.fc.requires_grad_(False)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)

        return x
    
class resnet50_layer2(nn.Module):
    def __init__(self, dilation: bool):
        super(resnet50_layer2, self).__init__()

        resnet = models.resnet50(
            replace_stride_with_dilation=[False, False, dilation], 
            weights=models.ResNet50_Weights.IMAGENET1K_V1, 
            norm_layer=FrozenBatchNorm2d
        )

        self.backbone = resnet
        self.num_channels = 512

        self.backbone.layer3.requires_grad_(False)
        self.backbone.layer4.requires_grad_(False)
        self.backbone.fc.requires_grad_(False)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)

        return x
    
class resnet50_layer3(nn.Module):
    def __init__(self, dilation: bool):
        super(resnet50_layer3, self).__init__()

        resnet = models.resnet50(
            replace_stride_with_dilation=[False, False, dilation], 
            weights=models.ResNet50_Weights.IMAGENET1K_V1, 
            norm_layer=FrozenBatchNorm2d
        )

        self.backbone = resnet
        self.num_channels = 1024

        self.backbone.layer4.requires_grad_(False)
        self.backbone.fc.requires_grad_(False)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)

        return x
    
class resnet50_layer1_FRZ(resnet50_layer1):
    def __init__(self, dilation: bool):
        super(resnet50_layer1_FRZ, self).__init__(dilation)

        for n, param in self.named_parameters():
            param.requires_grad_(False)

class resnet50_layer2_FRZ(resnet50_layer2):
    def __init__(self, dilation: bool):
        super(resnet50_layer2_FRZ, self).__init__(dilation)
        for n, param in self.named_parameters():
            param.requires_grad_(False)

class resnet50_layer3_FRZ(resnet50_layer3):
    def __init__(self, dilation: bool):
        super(resnet50_layer3_FRZ, self).__init__(dilation)
        for n, param in self.named_parameters():
            param.requires_grad_(False)
