"""Unchanged segmentation and direction/position auxiliary losses."""
import torch
from torch.nn import functional as F

def segmentation_loss(logits, target, valid):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp_min(1)
    probability = logits.sigmoid() * valid
    target = target * valid
    dice = (2 * (probability * target).sum((1, 2, 3)) + 1) / (
        probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + (1 - dice).mean()

def improved_loss(model, logits, target, valid, direction, active):
    seg = segmentation_loss(logits, target, valid)
    token_valid = F.avg_pool2d(valid, 8, 8)
    mask = active[:, 0] * (token_valid[:, 0] >= .99)
    ce = -(direction * model.last_direction_logits.log_softmax(1)).sum(1)
    direction_loss = (ce * mask).sum() / mask.sum().clamp_min(1)
    position = F.max_pool2d(target, 8, 8)
    position_valid = token_valid >= .99
    probability = model.last_position_logits.sigmoid() * position_valid
    position = position * position_valid
    position_loss = (1 - (2 * (probability * position).sum((1, 2, 3)) + 1) /
                     (probability.sum((1, 2, 3)) + position.sum((1, 2, 3)) + 1)).mean()
    return seg + .1 * direction_loss + .1 * position_loss
