"""Original single-input SCAR architectures for standalone baseline experiments.

These accept one tensor and are deliberately outside the three-modality M0-M3
training registry. Their channel/label configuration requires a matching dataset.
"""
from training.models.baselines.resunet_plus_plus import ResUNetPlusPlus2D
from training.models.baselines.unet_2d import UNet2D
from training.models.baselines.unet_3d import UNet3D

__all__ = ["ResUNetPlusPlus2D", "UNet2D", "UNet3D"]
