"""Pixel-pooled confusion matrix and metrics."""
from __future__ import annotations

import numpy as np
import torch

from training.dataset.data_contract import CLASS_NAMES


class ConfusionMeter:
    """Pixel-pooled metrics. Absent in both prediction/target is undefined."""

    def __init__(self, num_classes=4, device="cpu"):
        if not isinstance(num_classes, int) or not 2 <= num_classes <= len(CLASS_NAMES):
            raise ValueError(f"num_classes must be between 2 and {len(CLASS_NAMES)}")
        self.num_classes = num_classes
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target must have identical shapes")
        for value in (prediction, target):
            if value.is_floating_point() or value.is_complex():
                raise TypeError("ConfusionMeter requires integer class IDs, not logits")
            if value.numel() and ((value < 0).any() or (value >= self.num_classes).any()):
                raise ValueError("Class IDs are outside the confusion matrix range")
        prediction = prediction.to(self.matrix.device)
        target = target.to(self.matrix.device)
        values = self.num_classes * target.reshape(-1).long() + prediction.reshape(-1).long()
        self.matrix += torch.bincount(values, minlength=self.num_classes**2).reshape(self.matrix.shape)

    def reset(self):
        self.matrix.zero_()

    def compute(self):
        matrix = self.matrix.double().cpu().numpy()
        tp, predicted, actual = np.diag(matrix), matrix.sum(0), matrix.sum(1)
        result = {}
        for name, numerator, denominator in (
            ("precision", tp, predicted),
            ("recall", tp, actual),
            ("dice", 2 * tp, predicted + actual),
            ("iou", tp, predicted + actual - tp),
        ):
            scores = np.divide(numerator, denominator, out=np.full_like(tp, np.nan), where=denominator > 0)
            for i, score in enumerate(scores):
                result[f"{name}/{CLASS_NAMES[i]}"] = float(score)
            valid = scores[1:][np.isfinite(scores[1:])]
            result[f"mean_{name}"] = float(valid.mean()) if len(valid) else float("nan")
        result["pixel_accuracy"] = float(tp.sum() / matrix.sum()) if matrix.sum() else float("nan")
        return result
