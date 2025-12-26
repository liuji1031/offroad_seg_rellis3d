"""Segmentation head implementations for BEV semantic segmentation.

Provides complete segmentation heads that combine feature extraction (ASPP)
with classification layers. Loss computation is handled separately by the
loss module.
"""

from typing import Dict, List, Optional, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .aspp import ASPP, PolarASPP


__all__ = [
    'VanillaSegmentationHead',
    'ASPPSegmentationHead',
    'PolarASPPSegmentationHead',
    'build_segmentation_head_from_config',
]


class VanillaSegmentationHead(nn.Module):
    """Simple segmentation head (similar to BEVFusion).
    
    2 conv blocks + 1×1 classifier. Fast but limited receptive field.
    
    Args:
        in_channels: Number of input BEV feature channels
        num_classes: Number of segmentation classes
    
    Note:
        Loss computation should be handled separately using the loss module.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, num_classes, 1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input BEV features [B, C, H, W]
            return_logits: If True, return logits; if False, return probabilities
            
        Returns:
            Predictions [B, num_classes, H, W] (logits or probabilities)
        """
        # Classify
        logits = self.classifier(x)
        
        if return_logits:
            return logits
        else:
            # Return probabilities
            return torch.sigmoid(logits)


class ASPPSegmentationHead(nn.Module):
    """ASPP-based segmentation head.
    
    Uses ASPP for multi-scale context aggregation followed by classification.
    Better than vanilla head for handling diverse terrain types.
    
    Args:
        in_channels: Number of input BEV feature channels
        num_classes: Number of segmentation classes
        aspp_channels: ASPP output channels (default: 256)
        aspp_rates: Dilation rates for ASPP (default: [6, 12, 18])
        dropout: Dropout rate in ASPP (default: 0.5)
    
    Note:
        Loss computation should be handled separately using the loss module.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        aspp_channels: int = 256,
        aspp_rates: List[int] = [6, 12, 18],
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # ASPP module
        self.aspp = ASPP(
            in_channels=in_channels,
            out_channels=aspp_channels,
            rates=aspp_rates,
            dropout=dropout,
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(aspp_channels, aspp_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_channels // 2),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(aspp_channels // 2, num_classes, 1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input BEV features [B, C, H, W]
            return_logits: If True, return logits; if False, return probabilities
            
        Returns:
            Predictions [B, num_classes, H, W] (logits or probabilities)
        """
        # Extract multi-scale features
        x = self.aspp(x)
        
        # Classify
        logits = self.classifier(x)
        
        if return_logits:
            return logits
        else:
            return torch.sigmoid(logits)


class PolarASPPSegmentationHead(nn.Module):
    """Polar-aware ASPP segmentation head.
    
    Adapted for polar BEV grids with:
    - Separate radial/angular dilation rates
    - Circular padding for angular dimension
    - Optimized for small polar grids (e.g., 42×10)
    
    Args:
        in_channels: Number of input BEV feature channels
        num_classes: Number of segmentation classes
        aspp_channels: ASPP output channels
        rates_radial: Dilation rates for radial dimension
        rates_angular: Dilation rates for angular dimension
        use_circular_pad: Use circular padding for angular dim
        dropout: Dropout rate
    
    Note:
        Loss computation should be handled separately using the loss module.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        aspp_channels: int = 256,
        rates_radial: tuple[int, ...] = (2, 4, 6),
        rates_angular: tuple[int, ...] = (1, 2, 3),
        use_circular_pad: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Polar ASPP module
        self.aspp = PolarASPP(
            in_channels=in_channels,
            out_channels=aspp_channels,
            rates_radial=rates_radial,
            rates_angular=rates_angular,
            use_circular_pad=use_circular_pad,
            dropout=dropout,
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(aspp_channels, aspp_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_channels // 2),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(aspp_channels // 2, num_classes, 1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input BEV features [B, C, H_radial, W_angular]
            return_logits: If True, return logits; if False, return probabilities
            
        Returns:
            Predictions [B, num_classes, H, W] (logits or probabilities)
        """
        # Extract polar multi-scale features
        x = self.aspp(x)
        
        # Classify
        logits = self.classifier(x)
        
        if return_logits:
            return logits
        else:
            return torch.sigmoid(logits)


def build_segmentation_head_from_config(config: Dict) -> nn.Module:
    """Build segmentation head from config."""
    _config = config.copy()
    seg_head_type = _config.pop('type')
    if seg_head_type == 'vanilla':
        return VanillaSegmentationHead(**_config)
    elif seg_head_type == 'aspp':
        return ASPPSegmentationHead(**_config)
    elif seg_head_type == 'polar_aspp':
        return PolarASPPSegmentationHead(**_config)
    else:
        raise ValueError(f"Unknown seg_head_type: {seg_head_type}")