"""Complete image encoding pipeline combining backbone and neck.

This module provides a high-level wrapper that combines the backbone and FPN
into a complete image encoder, ready to feed features to the BEV transform.
"""

from typing import Dict, Any, Optional
import torch
from torch import nn

from .resnet import ResNetBackbone
from .fpn import LSSFPN, SECONDFPN, GeneralizedLSSFPN


__all__ = ["ImageEncoder"]


class ImageEncoder(nn.Module):
    """Complete image encoding pipeline: Backbone + Neck.

    This encoder takes raw images and produces multi-scale features suitable
    for input to the BEV transform module. It follows the BEVFusion design:

    1. Backbone: extract hierarchical features (e.g., ResNet)
    2. Neck: fuse multi-scale features (e.g., FPN)

    The output is a feature tensor ready for the Lift-Splat-Shoot transform.

    Args:
        backbone_config: Configuration dict for backbone
        neck_config: Configuration dict for neck

    """

    # Registry of available components
    BACKBONES = {
        "ResNetBackbone": ResNetBackbone,
    }

    NECKS = {
        "LSSFPN": LSSFPN,
        "SECONDFPN": SECONDFPN,
        "GeneralizedLSSFPN": GeneralizedLSSFPN,
    }

    def __init__(
        self,
        backbone_config: Dict[str, Any],
        neck_config: Dict[str, Any],
    ):
        super().__init__()

        # Build backbone
        backbone_type = backbone_config["type"]
        assert backbone_type in self.BACKBONES, (
            f"Unknown backbone: {backbone_type}"
        )
        _config = backbone_config.copy()
        _config.pop("type")
        self.backbone = self.BACKBONES[backbone_type](
            **_config
        )

        # Build neck
        neck_type = neck_config["type"]
        assert neck_type in self.NECKS, f"Unknown neck: {neck_type}"
        _config = neck_config.copy()
        _config.pop("type")
        self.neck = self.NECKS[neck_type](**_config)

        # Store info for debugging
        self.backbone_type = backbone_type
        self.neck_type = neck_type

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and neck.

        Args:
            img: Input images, either:
                - Single camera: [B, 3, H, W]
                - Multi-camera: [B, N, 3, H, W]

        Returns:
            Feature tensor:
                - Single camera: [B, C, H', W']
                - Multi-camera: [B, N, C, H', W']
        """
        # Handle multi-camera input
        is_multi_camera = img.ndim == 5
        if is_multi_camera:
            B, N, C, H, W = img.shape
            img = img.view(B * N, C, H, W)

        # Extract features with backbone
        features = self.backbone(img)

        # Fuse features with neck
        x = self.neck(features)

        # Handle output format (some necks return list, some return tensor)
        if isinstance(x, (list, tuple)):
            x = x[0]

        # Restore multi-camera shape
        if is_multi_camera:
            BN, C_out, H_out, W_out = x.shape
            x = x.view(B, N, C_out, H_out, W_out)

        return x

    def get_output_channels(self) -> int:
        """Get number of output feature channels."""
        if hasattr(self.neck, "out_channels"):
            return self.neck.out_channels
        else:
            # For SECONDFPN which returns concatenated features
            if isinstance(self.neck, SECONDFPN):
                return sum(self.neck.out_channels)
            else:
                raise ValueError(
                    f"Cannot determine output channels for neck type: {self.neck_type}"
                )

    def __repr__(self) -> str:
        return (
            f"ImageEncoder(\n"
            f"  backbone={self.backbone_type},\n"
            f"  neck={self.neck_type},\n"
            f"  out_channels={self.get_output_channels()}\n"
            f")"
        )


def build_image_encoder_from_config(config: dict[str, Any]) -> ImageEncoder:
    """Build ImageEncoder from config."""
    return ImageEncoder(
        backbone_config=config["backbone_config"],
        neck_config=config["neck_config"],
    )


def create_resnet50_lssfpn(
    pretrained: bool = True,
    out_channels: int = 256,
    image_size: tuple = (256, 704),
) -> ImageEncoder:
    """Create a ResNet50 + LSSFPN encoder (common BEVFusion config).

    This is one of the most common configurations used in BEVFusion for
    camera-based perception.

    Args:
        pretrained: Whether to load ImageNet pretrained weights
        out_channels: Number of output channels from FPN
        image_size: Input image size (H, W)

    Returns:
        Configured ImageEncoder
    """
    # ResNet50 outputs channels: [256, 512, 1024, 2048] at strides [4, 8, 16, 32]
    # We'll use the last two for LSSFPN
    encoder = ImageEncoder(
        backbone_config={
            "type": "ResNetBackbone",
            "depth": 50,
            "pretrained": pretrained,
            "num_stages": 4,
            "out_indices": (1, 2, 3),  # Get stride 8, 16, 32 features
            "frozen_stages": -1,
            "norm_eval": False,
        },
        neck_config={
            "type": "LSSFPN",
            "in_indices": (1, 2),  # Use stride 16 and 32
            "in_channels": (1024, 2048),
            "out_channels": out_channels,
            "scale_factor": 1,
        },
    )
    return encoder


def create_resnet50_secondfpn(
    pretrained: bool = True,
    out_channels_per_level: int = 128,
) -> ImageEncoder:
    """Create a ResNet50 + SECONDFPN encoder.

    This configuration concatenates features from all ResNet stages,
    commonly used when you need richer multi-scale information.

    Args:
        pretrained: Whether to load ImageNet pretrained weights
        out_channels_per_level: Output channels for each feature level

    Returns:
        Configured ImageEncoder
    """
    encoder = ImageEncoder(
        backbone_config={
            "type": "ResNetBackbone",
            "depth": 50,
            "pretrained": pretrained,
            "num_stages": 4,
            "out_indices": (0, 1, 2, 3),  # All stages
            "frozen_stages": -1,
            "norm_eval": False,
        },
        neck_config={
            "type": "SECONDFPN",
            "in_channels": [256, 512, 1024, 2048],
            "out_channels": [out_channels_per_level] * 4,
            "upsample_strides": [0.25, 0.5, 1, 2],  # Upsample to stride 16
        },
    )
    return encoder
