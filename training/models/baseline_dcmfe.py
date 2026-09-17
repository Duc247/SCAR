"""Baseline with Deformable Cross-Modal Feature Exchange (D-CMFE) and Baseline CMFE.

Implements Hướng 4 on top of the prompt-free paper baseline:
- 3 independent ResNetV2 encoders (CINE, PSIR/LGE, T2w)
- D-CMFE bottleneck cross-modal fusion (learning inter-modality displacement offsets)
- 3-level skip feature fusion
- TransUNet-style decoder cascade
"""
from __future__ import annotations

from torch import nn

from training.models.cmspa_net import CMSPANet
from training.models.modules.dcmfe import CMFE_Fusion, DCMFE_Fusion, DCMSPA_Fusion
from training.models.modules.sspanet import SSPANet_Block


class DCMSPANet(CMSPANet):
    """Ultimate Proposed Architecture: D-CMSPA-Net (Full 4-Way Union).

    Combines:
    - Hướng 1: Clinical Anatomical Inclusion Loss (supervised during training)
    - Hướng 2: SSPANet RMS Strip Pooling Attention on all 3 encoder branches
    - Hướng 3 + 4: D-CMSPA (Deformable Cross-Modal Strip Pathology Attention)
      which first eliminates respiratory motion displacement via learned 2D deformable
      sampling, then applies CINE strip anatomy gating and contrast pathology feedback.
    - 3-level skip feature fusion
    - TransUNet-style decoder cascade
    """

    def __init__(self, config=None, max_offset: float = 2.0, **kwargs):
        kwargs.setdefault("ablation", "M3")
        super().__init__(config, **kwargs)

        self.config.architecture = "dcmspa_net"
        self.config.max_offset = float(self.config.get("max_offset", max_offset))
        self.config.use_sspanet = True

        width = int(64 * self.config.resnet.width_factor)
        channels = 16 * width

        # Hướng 2: SSPANet on all 3 encoders
        self.sspanet_cine = SSPANet_Block(channels)
        self.sspanet_psir = SSPANet_Block(channels)
        self.sspanet_t2w = SSPANet_Block(channels)

        # Hướng 4 + Hướng 3: Deformable Cross-Modal Strip Pathology Attention
        self.cross_fusion = DCMSPA_Fusion(
            in_channels=channels,
            out_channels=self.config.fused_channels,
            max_offset=self.config.max_offset,
        )


class BaselineDCMFE(CMSPANet):
    """Paper baseline network equipped with Deformable Cross-Modal Fusion (D-CMFE).

    Args:
        config: Model configuration dictionary.
        max_offset: Maximum displacement in pixels (default 2.0).
        use_sspanet: Whether to use SSPANet attention (default False, preserving paper baseline).
    """

    def __init__(self, config=None, max_offset: float = 2.0, use_sspanet: bool = False, **kwargs):
        # Default ablation to M0 (identity attention) unless use_sspanet is True
        forced_ablation = "M1" if use_sspanet or kwargs.get("use_sspanet") else "M0"
        kwargs.setdefault("ablation", forced_ablation)
        super().__init__(config, **kwargs)

        self.config.architecture = "baseline_dcmfe"
        self.config.max_offset = float(self.config.get("max_offset", max_offset))
        self.config.use_sspanet = bool(use_sspanet or self.config.get("use_sspanet", False))

        width = int(64 * self.config.resnet.width_factor)
        channels = 16 * width

        if self.config.use_sspanet:
            self.sspanet_cine = SSPANet_Block(channels)
            self.sspanet_psir = SSPANet_Block(channels)
            self.sspanet_t2w = SSPANet_Block(channels)
        else:
            self.sspanet_cine = nn.Identity()
            self.sspanet_psir = nn.Identity()
            self.sspanet_t2w = nn.Identity()

        self.cross_fusion = DCMFE_Fusion(
            in_channels=channels,
            out_channels=self.config.fused_channels,
            max_offset=self.config.max_offset,
        )


class BaselineCMFE(CMSPANet):
    """Paper baseline network with original Cosine Similarity CMFE."""

    def __init__(self, config=None, use_sspanet: bool = False, **kwargs):
        forced_ablation = "M1" if use_sspanet or kwargs.get("use_sspanet") else "M0"
        kwargs.setdefault("ablation", forced_ablation)
        super().__init__(config, **kwargs)

        self.config.architecture = "baseline_cmfe"
        self.config.use_sspanet = bool(use_sspanet or self.config.get("use_sspanet", False))

        width = int(64 * self.config.resnet.width_factor)
        channels = 16 * width

        if self.config.use_sspanet:
            self.sspanet_cine = SSPANet_Block(channels)
            self.sspanet_psir = SSPANet_Block(channels)
            self.sspanet_t2w = SSPANet_Block(channels)
        else:
            self.sspanet_cine = nn.Identity()
            self.sspanet_psir = nn.Identity()
            self.sspanet_t2w = nn.Identity()

        self.cross_fusion = CMFE_Fusion(
            in_channels=channels,
            out_channels=self.config.fused_channels,
        )

