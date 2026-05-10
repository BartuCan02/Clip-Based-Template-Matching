import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import generalized_box_iou_loss

# cx, cy, w, h version gIoU loss
def gIoU_loss(pred, target, reduction="sum"):
    # pred, target: [cx, cy ,w, h]
    pred_xyxy = torch.cat([pred[:, :2] - pred[:, 2:] / 2, pred[:, :2] + pred[:, 2:] / 2], dim=1)
    target_xyxy = torch.cat([target[:, :2] - target[:, 2:] / 2, target[:, :2] + target[:, 2:] / 2], dim=1)

    loss = generalized_box_iou_loss(pred_xyxy, target_xyxy, reduction=reduction, eps=1e-13)
    return loss

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        # alpha: the weight assigned to the rare class (definition in paper)
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        alpha = torch.tensor([1-self.alpha, self.alpha], dtype=inputs.dtype, device=inputs.device)
        targets = targets.type(torch.long)
        at = alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss

class SetCriterion_TM(nn.Module):
    def __init__(self, use_focal_loss = False):
        super().__init__()
        if use_focal_loss:
            self.loss_ce = WeightedFocalLoss(alpha=.25, gamma=2)
        else:
            self.loss_ce = nn.BCEWithLogitsLoss(reduction="none")
        self.loss_giou = gIoU_loss

    def forward(self, outputs, targets):
        ce_losses = []
        bboxes_losses = []
        for level in range(len(outputs['objectness'])):
            num_positive = len(targets['regressions'][level])
            ce_loss = self.loss_ce(outputs['objectness'][level], targets['objectness'][level])
            bbox_loss = self.loss_giou(outputs['regressions'][level], targets['regressions'][level], reduction="none")

            ce_loss = ce_loss.sum() / num_positive
            bbox_loss = bbox_loss.sum() / num_positive

            ce_losses.append(ce_loss)
            bboxes_losses.append(bbox_loss)
        
        result = {
            "loss_ce" : torch.stack(ce_losses).mean(),
            "loss_giou": torch.stack(bboxes_losses).mean()
        }
        return result