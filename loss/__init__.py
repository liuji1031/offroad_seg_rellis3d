"""Loss computation module for segmentation.

Provides basic loss functions (focal, dice, lovász) and a unified
SegmentationLoss class for per-class loss computation with class weights.
"""

from .losses import (
    focal_loss,
    dice_loss,
    lovasz_softmax,
    combined_loss,
    FocalLoss,
    DiceLoss,
    CombinedLoss,
)
from .segmentation_loss import SegmentationLoss


__all__ = [
    # Basic loss functions
    'focal_loss',
    'dice_loss',
    'lovasz_softmax',
    'combined_loss',
    
    # Loss modules
    'FocalLoss',
    'DiceLoss',
    'CombinedLoss',
    
    # Segmentation-specific
    'SegmentationLoss',
]

