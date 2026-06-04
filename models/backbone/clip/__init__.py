from .naclip_vit import NaClipViT16Backbone


def build_naclip_backbone(
    requires_grad: bool = False,
    sigma: float = 5.0,
) -> NaClipViT16Backbone:
    return NaClipViT16Backbone(requires_grad=requires_grad, sigma=sigma)
