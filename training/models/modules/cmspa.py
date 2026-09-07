"""Cross-Modal Strip Pathology Attention (CMSPA) fusion module."""
from __future__ import annotations

import torch
from torch import nn

from training.models.modules.sspanet import _statistics_input


def channel_std(x, epsilon=1e-6):
    """Per-pixel channel population std; also defined when C=1."""
    return (_statistics_input(x).var(dim=1, keepdim=True, unbiased=False) + epsilon).sqrt()


class CMSPA_Fusion(nn.Module):
    """CINE strip anatomy guide and learned PSIR/T2W style feedback.

    Anatomy uses a sum of broadcast height/width strips. Pathology uses
    std_channels(PSIR) + std_channels(T2W), shape (B,1,H,W); its learnable
    projection accepts ONE input channel. These gates are learned priors,
    not guaranteed tissue masks.
    """

    def __init__(self, in_channels=1024, out_channels=512):
        super().__init__()
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
            nn.Conv2d(3 * in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cine, psir, t2w):
        anatomy = self.conv_strip(cine.mean(dim=3, keepdim=True)
                                  + cine.mean(dim=2, keepdim=True))
        style = (channel_std(psir) + channel_std(t2w)).to(cine.dtype)
        pathology = self.conv_patho(style)
        fused = torch.cat((cine + cine * pathology,
                           psir * anatomy, t2w * anatomy), dim=1)
        return self.fusion_conv(fused)
