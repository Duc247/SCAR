"""Dual-pathology fusion: modality routing, local/strip context and residual fusion."""
import math

import torch
from torch import nn

from training.models.modules.fusion import ConcatFusion


def local_block(channels):
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
        nn.GroupNorm(1, channels), nn.GELU(),
        nn.Conv2d(channels, channels, 1), nn.GELU(),
    )


class DualPathologyFusion(nn.Module):
    def __init__(self, in_channels, out_channels, width, auxiliary=False):
        super().__init__()
        self.base = ConcatFusion(in_channels, out_channels)
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, width, 1, bias=False),
                          nn.GroupNorm(1, width), nn.GELU()) for _ in range(3)
        ])
        self.routers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * width, width, 1), nn.GELU(),
                          local_block(width), nn.Conv2d(width, 3, 1)) for _ in range(2)
        ])
        self.strip = nn.Conv2d(width, width, 1)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2 * width, width, 1), nn.GELU(),
                          local_block(width)) for _ in range(2)
        ])
        self.output = nn.Conv2d(2 * width, out_channels, 1)
        self.eta_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        self.heads = nn.ModuleList([nn.Conv2d(width, 1, 1) for _ in range(2)]) if auxiliary else None

    def forward(self, cine, lge, t2w, return_aux=False):
        features = (cine, lge, t2w)
        z = [project(x) for project, x in zip(self.projections, features)]
        joined = torch.cat(z, dim=1)
        strip = self.strip(z[0].mean(3, keepdim=True) + z[0].mean(2, keepdim=True))
        experts = []
        for router, expert in zip(self.routers, self.experts):
            weights = router(joined).softmax(1)
            mixed = sum(weights[:, i:i + 1] * x for i, x in enumerate(z))
            experts.append(expert(torch.cat((mixed, strip), dim=1)))
        fused = self.base(*features) + self.eta_logit.sigmoid() * self.output(torch.cat(experts, 1))
        if return_aux:
            if self.heads is None:
                raise ValueError("This fusion has no auxiliary heads")
            # Canonical auxiliary channel order: scar, exclusive edema.
            return fused, torch.cat([head(x) for head, x in zip(self.heads, experts)], 1)
        return fused
