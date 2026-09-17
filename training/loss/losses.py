"""Segmentation loss functions."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    """Per-image equally weighted class Dice with squared denominator.

    Includes background; reductions are float32 even under autocast. Per-image
    reduction makes sample-weighted gradient accumulation well-defined.
    """

    def __init__(self, n_classes=4, smooth=1e-5):
        super().__init__()
        if not isinstance(n_classes, int) or n_classes < 2:
            raise ValueError("n_classes must be an integer >= 2")
        if not math.isfinite(smooth) or smooth <= 0:
            raise ValueError("Dice smooth must be finite and positive")
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, inputs, target, weight=None, softmax=False):
        probabilities = inputs.float().softmax(1) if softmax else inputs.float()
        labels = F.one_hot(target.long(), self.n_classes).movedim(-1, 1).float()
        if labels.shape != probabilities.shape:
            raise ValueError(f"Logits/target shape mismatch: {inputs.shape}, {target.shape}")
        dims = tuple(range(2, inputs.ndim))
        scores = (2 * (probabilities * labels).sum(dims) + self.smooth) / (
            probabilities.square().sum(dims) + labels.sum(dims) + self.smooth
        )
        losses = 1 - scores
        if weight is not None:
            weights = torch.as_tensor(weight, device=inputs.device, dtype=torch.float32)
            if (weights.shape != (self.n_classes,) or not torch.isfinite(weights).all()
                    or (weights < 0).any() or weights.sum() <= 0):
                raise ValueError("Dice weights must be nonnegative with positive sum.")
            return (losses * weights).sum(1).mean() / weights.sum()
        return losses.mean()


class SegmentationLoss(nn.Module):
    """Combined Cross-Entropy and Soft Dice loss."""

    def __init__(self, n_classes=4, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        if (not all(math.isfinite(w) and w >= 0 for w in (ce_weight, dice_weight))
                or ce_weight + dice_weight <= 0):
            raise ValueError("Loss weights must be finite, nonnegative, with positive sum")
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss(n_classes)

    def forward(self, logits, target):
        ce = F.cross_entropy(logits.float(), target.long())
        dice = self.dice(logits, target, softmax=True)
        return {
            "loss": self.ce_weight * ce + self.dice_weight * dice,
            "ce": ce,
            "dice_loss": dice,
        }


from training.loss.anatomical_loss import (
    AnatomicalSegmentationLoss,
    ClinicalAnatomicalInclusionLoss,
)

