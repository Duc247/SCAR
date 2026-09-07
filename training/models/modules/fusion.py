"""Modality fusion modules for multi-modal feature representations."""
from __future__ import annotations

import torch
from torch import nn


class ConcatFusion(nn.Module):
    def __init__(self, in_channels=1024, out_channels=512):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cine, psir, t2w):
        return self.fusion_conv(torch.cat((cine, psir, t2w), dim=1))


class Fusion_Embed(ConcatFusion):
    """Concatenate modalities, then project to the original skip width."""

    def __init__(self, embed_dim):
        super().__init__(embed_dim, embed_dim)


class CrossAttention_Fusion(nn.Module):
    """M2 control: CINE queries, mean(PSIR,T2W) keys and values."""

    def __init__(self, in_channels=1024, out_channels=512, num_heads=8):
        super().__init__()
        if in_channels % num_heads:
            raise ValueError("Cross-attention channels must be divisible by num_heads")
        self.mha = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.proj = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cine, psir, t2w):
        batch, channels, height, width = cine.shape
        query = cine.flatten(2).transpose(1, 2)
        pathology = ((psir + t2w) * 0.5).flatten(2).transpose(1, 2)
        # Avoid a materialized attention matrix and enable efficient SDPA.
        attention, _ = self.mha(query, pathology, pathology, need_weights=False)
        attention = attention.transpose(1, 2).reshape(batch, channels, height, width)
        return self.proj(torch.cat((cine, attention), dim=1))
