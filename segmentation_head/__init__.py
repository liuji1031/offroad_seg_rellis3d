"""Segmentation heads for BEV semantic segmentation.

This module provides various segmentation head architectures for dense prediction
on BEV feature maps, with special support for polar grids.
"""

from .aspp import ASPP, PolarASPP
from .segmentation_head import (
    ASPPSegmentationHead,
    PolarASPPSegmentationHead,
    VanillaSegmentationHead,
    build_segmentation_head_from_config,
)
from .losses import (
    focal_loss,
    dice_loss,
    lovasz_softmax,
    combined_loss,
)

__all__ = [
    # ASPP modules
    'ASPP',
    'PolarASPP',
    # Segmentation heads
    'ASPPSegmentationHead',
    'PolarASPPSegmentationHead',
    'VanillaSegmentationHead',
    'build_segmentation_head_from_config',
    # Loss functions
    'focal_loss',
    'dice_loss',
    'lovasz_softmax',
    'combined_loss',
]

