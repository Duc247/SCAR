"""SSPANet spatial and channel attention modules.

Adapted from Md Jahid Hasan's demonstration (DOI 10.1016/j.bspc.2025.108636).
Strip statistic is RMS, sqrt(E[x**2] + eps).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _statistics_input(x):
    """Accumulate squares and variance in FP32 when using AMP."""
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def strip_rms(x, dim, epsilon=1e-6):
    """Author's uncentered RMS statistic, retaining the pooled dimension."""
    return (_statistics_input(x).square().mean(dim=dim, keepdim=True) + epsilon).sqrt()


class SSPA_BasicConv(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 relu=True, bn=True, bias=False):
        layers = [nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride,
                            padding=padding, bias=bias)]
        if bn:
            layers.append(nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01))
        if relu:
            layers.append(nn.ReLU())
        super().__init__(*layers)


class SSPA_ZPool(nn.Module):
    def forward(self, x):
        # Reduce C, not H/W: (B,C,H,W) -> (B,2,H,W).
        return torch.cat((x.max(dim=1, keepdim=True).values,
                          x.mean(dim=1, keepdim=True)), dim=1)


class SSPA_ChannelAttention(nn.Module):
    """Author's name retained; channel pooling produces a spatial gate."""

    def __init__(self):
        super().__init__()
        self.compress = SSPA_ZPool()
        self.conv = SSPA_BasicConv(2, 1, 7, padding=3, relu=False)

    def forward(self, x):
        return x * torch.sigmoid(self.conv(self.compress(x)))


class SSPA_SpatialAttention(nn.Module):
    def __init__(self, inplanes, outplanes=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        outplanes = inplanes if outplanes is None else outplanes
        if inplanes != outplanes:
            raise ValueError("SSPA multiplicative attention must preserve channels")
        self.conv1 = nn.Conv2d(inplanes, outplanes, (3, 1), padding=(1, 0), bias=False)
        self.bn1 = norm_layer(outplanes)
        self.conv2 = nn.Conv2d(inplanes, outplanes, (1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(outplanes)
        self.conv3 = nn.Conv2d(outplanes, outplanes, 1)

    def forward(self, x):
        rms_h = strip_rms(x, dim=3).to(x.dtype)
        rms_w = strip_rms(x, dim=2).to(x.dtype)
        strip_h = self.bn1(self.conv1(rms_h))
        strip_w = self.bn2(self.conv2(rms_w))
        # A 1x1 convolution is linear: project BEFORE expanding each strip.
        # This is algebraically identical to conv3(strip_h + strip_w), and
        # saves C*C*(H*W-H-W) MACs. Add the bias once, never once per strip.
        projected_h = F.conv2d(strip_h, self.conv3.weight, self.conv3.bias)
        projected_w = F.conv2d(strip_w, self.conv3.weight)
        return x * torch.sigmoid(projected_h + projected_w)


class SSPANet_Block(nn.Module):
    def __init__(self, in_channels=1024):
        super().__init__()
        self.channel = SSPA_ChannelAttention()
        self.spatial = SSPA_SpatialAttention(in_channels)

    def forward(self, x):
        return x + x * torch.sigmoid(self.channel(x) + self.spatial(x))
