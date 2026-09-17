"""Clinical Anatomical Inclusion Loss (Hướng 1) for CMR Myocardial Pathology Segmentation.

Enforces two cardiovascular anatomical priors:
1. Scar in Edema constraint (L_se):
   - L_scar_out: Penalizes predicted scar leaking outside ground truth edema.
   - L_edema_cov: Penalizes predicted edema failing to cover ground truth scar.
2. Lesion heart confinement constraint (L_leak):
   - L_leak: Penalizes predicted lesion (scar + edema) leaking outside the whole myocardium.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from training.loss.losses import DiceLoss


class ClinicalAnatomicalInclusionLoss(nn.Module):
    """Anatomical prior loss for cardiac lesion segmentation.

    Args:
        myo_idx: Class index for myocardium (default 1).
        scar_idx: Class index for scar. If None, derived from label_order.
        edema_idx: Class index for edema. If None, derived from label_order.
        alpha: Weight for scar-in-edema constraint L_se (default 0.1).
        beta: Weight for outside-heart leakage penalty L_leak (default 0.05).
        eps: Small numerical stability constant (default 1e-6).
        label_order: 'canonical' (edema=2, scar=3) or 'legacy' (scar=2, edema=3).
    """

    def __init__(
        self,
        myo_idx: int = 1,
        scar_idx: int | None = None,
        edema_idx: int | None = None,
        alpha: float = 0.1,
        beta: float = 0.05,
        eps: float = 1e-6,
        label_order: str = "canonical",
    ):
        super().__init__()
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError(f"alpha must be finite and nonnegative, got {alpha}")
        if not math.isfinite(beta) or beta < 0:
            raise ValueError(f"beta must be finite and nonnegative, got {beta}")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError(f"eps must be finite and positive, got {eps}")

        self.myo_idx = int(myo_idx)
        if scar_idx is not None and edema_idx is not None:
            self.scar_idx = int(scar_idx)
            self.edema_idx = int(edema_idx)
        elif label_order.lower() == "legacy":
            self.scar_idx = 2
            self.edema_idx = 3
        else:  # canonical: 0:bg, 1:myo, 2:edema, 3:scar
            self.scar_idx = 3
            self.edema_idx = 2

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        self.label_order = label_order

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the clinical anatomical inclusion loss.

        Args:
            pred_logits: Tensor of shape (B, C, H, W), raw logits before softmax.
            target: Tensor of shape (B, H, W) with integer class labels.

        Returns:
            Dictionary containing 'loss_inc', 'loss_se', 'loss_leak',
            'loss_scar_out', and 'loss_edema_cov'.
        """
        if pred_logits.ndim != 4:
            raise ValueError(f"Expected pred_logits of shape (B, C, H, W), got ndim={pred_logits.ndim}")
        if target.ndim != 3:
            raise ValueError(f"Expected target of shape (B, H, W), got ndim={target.ndim}")
        if pred_logits.shape[0] != target.shape[0] or pred_logits.shape[2:] != target.shape[1:]:
            raise ValueError(
                f"Shape mismatch: logits {pred_logits.shape} vs target {target.shape}"
            )

        probs = F.softmax(pred_logits.float(), dim=1)
        p_scar = probs[:, self.scar_idx, :, :]
        p_edema = probs[:, self.edema_idx, :, :]

        y_myo = (target == self.myo_idx).float()
        y_scar = (target == self.scar_idx).float()
        y_edema = (target == self.edema_idx).float()

        # 1. Ràng buộc Scar nằm trong Edema (L_se)
        # Trong MyoPS, tổn thương thực tế (Area at Risk) bao gồm cả Edema (2) và Scar (3)
        y_lesion = torch.clamp(y_edema + y_scar, 0.0, 1.0)
        non_lesion = 1.0 - y_lesion
        denom_non_lesion = torch.sum(non_lesion) + self.eps

        # Phạt nếu Scar dự đoán rò rỉ ra ngoài vùng tổn thương (vào cơ tim lành hoặc ngoài tim)
        loss_scar_out = -torch.sum(
            non_lesion * torch.log(torch.clamp(1.0 - p_scar, min=self.eps, max=1.0))
        ) / denom_non_lesion

        num_scar = torch.sum(y_scar)
        p_lesion = torch.clamp(p_scar + p_edema, 0.0, 1.0)
        if num_scar > 0:
            # Bắt buộc vùng tổn thương dự đoán (Edema + Scar) phải bao bọc các pixel Scar chuẩn
            loss_edema_cov = -torch.sum(
                y_scar * torch.log(torch.clamp(p_lesion, min=self.eps, max=1.0))
            ) / (num_scar + self.eps)
        else:
            loss_edema_cov = torch.tensor(0.0, device=pred_logits.device, dtype=torch.float32)

        l_se = loss_scar_out + loss_edema_cov

        # 2. Ràng buộc Tổn thương không rò rỉ ra ngoài cơ tim (L_leak)
        y_myo_total = torch.clamp(y_myo + y_lesion, 0.0, 1.0)
        outside_heart = 1.0 - y_myo_total
        denom_outside = torch.sum(outside_heart) + self.eps
        l_leak = -torch.sum(
            outside_heart * torch.log(torch.clamp(1.0 - p_lesion, min=self.eps, max=1.0))
        ) / denom_outside

        l_inc = self.alpha * l_se + self.beta * l_leak

        return {
            "loss_inc": l_inc,
            "loss_se": l_se,
            "loss_leak": l_leak,
            "loss_scar_out": loss_scar_out,
            "loss_edema_cov": loss_edema_cov,
        }


class AnatomicalSegmentationLoss(nn.Module):
    """Compound segmentation loss: Cross-Entropy + Soft Dice + Anatomical Inclusion Loss.

    Total loss:
        L_total = ce_weight * L_ce + dice_weight * L_dice + L_inc
    """

    def __init__(
        self,
        n_classes: int = 4,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        alpha: float = 0.1,
        beta: float = 0.05,
        label_order: str = "canonical",
        myo_idx: int = 1,
        scar_idx: int | None = None,
        edema_idx: int | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        if (
            not all(math.isfinite(w) and w >= 0 for w in (ce_weight, dice_weight))
            or ce_weight + dice_weight <= 0
        ):
            raise ValueError("ce_weight and dice_weight must be finite, nonnegative, with positive sum")

        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.dice = DiceLoss(n_classes)
        self.inclusion = ClinicalAnatomicalInclusionLoss(
            myo_idx=myo_idx,
            scar_idx=scar_idx,
            edema_idx=edema_idx,
            alpha=alpha,
            beta=beta,
            eps=eps,
            label_order=label_order,
        )

    def forward(self, output: torch.Tensor | dict[str, Any], target: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = output["logits"] if isinstance(output, dict) else output
        ce = F.cross_entropy(logits.float(), target.long())
        dice = self.dice(logits, target, softmax=True)
        inc_dict = self.inclusion(logits, target)
        total = self.ce_weight * ce + self.dice_weight * dice + inc_dict["loss_inc"]

        return {
            "loss": total,
            "ce": ce,
            "dice_loss": dice,
            "l_inc": inc_dict["loss_inc"],
            "l_se": inc_dict["loss_se"],
            "l_leak": inc_dict["loss_leak"],
            "loss_scar_out": inc_dict["loss_scar_out"],
            "loss_edema_cov": inc_dict["loss_edema_cov"],
        }
