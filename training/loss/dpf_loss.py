"""M3-DPF loss v1. Exclusive edema, patient-independent per-image reductions."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def present_dice(probability, target):
    """Average over nonempty target images; empty images receive CE/BCE gradients."""
    dims = tuple(range(1, probability.ndim))
    mass = target.sum(dims)
    score = (2 * (probability * target).sum(dims) + 1e-5) / (
        probability.sum(dims) + mass + 1e-5)
    valid = mass > 0
    return ((1 - score) * valid).sum() / valid.sum().clamp_min(1)


class DPFLoss(nn.Module):
    requires_aux = True

    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        if (not all(math.isfinite(w) and w >= 0 for w in (ce_weight, dice_weight))
                or ce_weight + dice_weight <= 0):
            raise ValueError("Loss weights must be finite, nonnegative, with positive sum")
        self.ce_weight, self.dice_weight = ce_weight, dice_weight
        self.set_epoch(10)

    def set_epoch(self, epoch):
        self.ramp = min(max(epoch / 10.0, 0.0), 1.0)

    def forward(self, output, target):
        logits, auxiliary = output["logits"].float(), output["aux_logits"].float()
        p = logits.softmax(1)
        y = F.one_hot(target.long(), 4).movedim(-1, 1).float()
        ce = F.cross_entropy(logits, target.long())
        dice = sum(w * present_dice(p[:, c], y[:, c])
                   for c, w in ((1, 0.2), (2, 0.4), (3, 0.4)))
        hierarchy = 0.5 * (present_dice(p[:, 1:].sum(1), y[:, 1:].sum(1)) +
                           present_dice(p[:, 2:].sum(1), y[:, 2:].sum(1)))
        soft_target = F.adaptive_avg_pool2d(y[:, [3, 2]], auxiliary.shape[2:])
        aux = F.binary_cross_entropy_with_logits(auxiliary, soft_target)
        return {"loss": self.ce_weight * ce + self.dice_weight * dice +
                        self.ramp * (0.2 * hierarchy + 0.1 * aux),
                "ce": ce, "dice_loss": dice, "hierarchy_loss": hierarchy, "aux_loss": aux}
