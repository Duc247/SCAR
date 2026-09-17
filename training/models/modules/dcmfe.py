"""Deformable Cross-Modal Feature Exchange (D-CMFE) for Multi-modal CMR.

Replaces the global static All-to-All Cosine Similarity in original CMFE with
a learned spatial displacement field (Deformable Offset) to compensate for
inter-sequence respiratory motion misalignment before local feature fusion.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DeformableOffsetNet(nn.Module):
    """Predicts 2D spatial offset (delta_x, delta_y) and samples deformed features."""

    def __init__(self, in_channels: int, max_offset: float = 2.0):
        super().__init__()
        if not math.isfinite(max_offset) or max_offset <= 0:
            raise ValueError(f"max_offset must be positive and finite, got {max_offset}")
        self.max_offset = float(max_offset)
        hidden = max(in_channels // 2, 1)

        # Mạng dự đoán độ lệch không gian (delta_x, delta_y)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=3, padding=1, bias=True),
        )

    def forward(
        self, feat_anchor: torch.Tensor, feat_target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align feat_target to feat_anchor coordinates via learned deformable sampling."""
        B, C, H, W = feat_anchor.shape

        concat_feat = torch.cat([feat_anchor, feat_target], dim=1)
        offset = torch.tanh(self.net(concat_feat)) * self.max_offset

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=feat_anchor.device, dtype=feat_anchor.dtype),
            torch.linspace(-1, 1, W, device=feat_anchor.device, dtype=feat_anchor.dtype),
            indexing="ij",
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

        norm_offset = torch.zeros_like(base_grid)
        norm_offset[..., 0] = offset[:, 0, :, :] / (W / 2.0)
        norm_offset[..., 1] = offset[:, 1, :, :] / (H / 2.0)

        deformed_grid = (base_grid + norm_offset).to(feat_target.dtype)
        aligned = F.grid_sample(
            feat_target, deformed_grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return aligned, offset


class DeformableCrossModalFusion(nn.Module):
    """Pairwise Deformable Cross-Modal alignment and fusion module.

    Learns spatial displacement (delta_x, delta_y) between an anchor modality m
    and target modality m', then samples deformed features and projects.

    Args:
        channels: Feature channel count (e.g. 1024 at bottleneck).
        max_offset: Maximum displacement in pixels (default 2.0).
    """

    def __init__(self, channels: int = 1024, max_offset: float = 2.0):
        super().__init__()
        self.align_net = DeformableOffsetNet(channels, max_offset=max_offset)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    @property
    def offset_net(self) -> nn.Sequential:
        return self.align_net.net

    @property
    def max_offset(self) -> float:
        return self.align_net.max_offset

    def deform_align(
        self, feat_m: torch.Tensor, feat_other: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.align_net(feat_m, feat_other)

    def forward(self, feat_m: torch.Tensor, feat_other: torch.Tensor) -> torch.Tensor:
        """Pairwise fusion of anchor and aligned target modality."""
        feat_other_aligned, _ = self.deform_align(feat_m, feat_other)
        return self.fusion_conv(torch.cat([feat_m, feat_other_aligned], dim=1))


class DCMFE_Fusion(nn.Module):
    """Deformable Cross-Modal Feature Exchange (D-CMFE) for 3 CMR modalities.

    CINE serves as the anatomical reference anchor. PSIR/LGE and T2w are aligned
    to CINE via separate learned displacement offset fields to eliminate inter-sequence
    motion artifacts, followed by 1x1 multi-modal projection.

    Args:
        in_channels: Input channels per modality (default 1024).
        out_channels: Output fused channels (default 512).
        max_offset: Maximum displacement in pixels (default 2.0).
    """

    def __init__(self, in_channels: int = 1024, out_channels: int = 512, max_offset: float = 2.0):
        super().__init__()
        self.align_psir = DeformableOffsetNet(in_channels, max_offset=max_offset)
        self.align_t2w = DeformableOffsetNet(in_channels, max_offset=max_offset)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(3 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        """Fuse CINE, PSIR, and T2w with deformable cross-modal alignment."""
        psir_aligned, _ = self.align_psir(cine, psir)
        t2w_aligned, _ = self.align_t2w(cine, t2w)
        fused = self.fusion_conv(torch.cat([cine, psir_aligned, t2w_aligned], dim=1))
        return fused


class CMFE_Fusion(nn.Module):
    """Original Cross-Modal Feature Exchange (CMFE) using all-to-all cosine similarity.

    Used as an exact baseline comparison against D-CMFE:
        sim_{m -> m'} = |cos((F_m^flat)^T, F_{m'}^flat)| in R^{B x N x N}
    """

    def __init__(self, in_channels: int = 1024, out_channels: int = 512, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(3 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _cosine_enhance(self, f_anchor: torch.Tensor, f_other: torch.Tensor) -> torch.Tensor:
        B, C, H, W = f_anchor.shape
        a = f_anchor.flatten(2)  # (B, C, N)
        b = f_other.flatten(2)   # (B, C, N)

        a_norm = F.normalize(a, p=2, dim=1, eps=self.eps)
        b_norm = F.normalize(b, p=2, dim=1, eps=self.eps)

        sim = torch.bmm(a_norm.transpose(1, 2), b_norm).abs()
        sim = F.softmax(sim, dim=-1)

        enhanced = torch.bmm(b, sim.transpose(1, 2)).reshape(B, C, H, W)
        return enhanced

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        psir_enh = self._cosine_enhance(cine, psir)
        t2w_enh = self._cosine_enhance(cine, t2w)
        return self.fusion_conv(torch.cat([cine, psir_enh, t2w_enh], dim=1))


class DCMSPA_Fusion(nn.Module):
    """Deformable Cross-Modal Strip Pathology Attention (D-CMSPA) Fusion.

    Unifies Hướng 4 (Deformable Inter-Sequence Motion Compensation) and
    Hướng 3 (Cross-Modal Strip Pathology Attention):
    1. Dynamic Deformable Alignment: CINE is the spatial-anatomical anchor.
       PSIR/LGE and T2w are dynamically aligned to CINE coordinates via learned
       2D displacement vector fields (delta_x, delta_y).
    2. CINE Strip Anatomy Guidance Gate:
       A = conv_strip(Mean_W(CINE) + Mean_H(CINE))
    3. Aligned Pathology Feedback Gate:
       P = conv_patho(std_channel(PSIR_aligned) + std_channel(T2w_aligned))
    4. Cross-Modality Modulated Fusion:
       Fused = Conv1x1(Concat([CINE + CINE * P, PSIR_aligned * A, T2w_aligned * A]))
    """

    def __init__(self, in_channels: int = 1024, out_channels: int = 512, max_offset: float = 2.0):
        super().__init__()
        from training.models.modules.cmspa import channel_std
        self._channel_std = channel_std

        self.align_psir = DeformableOffsetNet(in_channels, max_offset=max_offset)
        self.align_t2w = DeformableOffsetNet(in_channels, max_offset=max_offset)

        hidden = max(in_channels // 4, 1)
        self.conv_strip = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )
        self.conv_patho = nn.Sequential(
            nn.Conv2d(1, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(3 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        # Step 1: Deformable spatial alignment of target modalities to CINE anchor
        psir_aligned, _ = self.align_psir(cine, psir)
        t2w_aligned, _ = self.align_t2w(cine, t2w)

        # Step 2: CINE strip anatomy guidance gate
        anatomy = self.conv_strip(
            cine.mean(dim=3, keepdim=True) + cine.mean(dim=2, keepdim=True)
        )

        # Step 3: Aligned pathology feedback gate
        style = (self._channel_std(psir_aligned) + self._channel_std(t2w_aligned)).to(cine.dtype)
        pathology = self.conv_patho(style)

        # Step 4: Modulated cross-modal fusion
        fused = self.fusion_conv(
            torch.cat(
                [cine + cine * pathology, psir_aligned * anatomy, t2w_aligned * anatomy],
                dim=1,
            )
        )
        return fused

