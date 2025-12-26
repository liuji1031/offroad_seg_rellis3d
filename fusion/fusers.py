"""Feature fusion modules for multi-modal BEV features.

Implements fusion strategies from BEVFusion for combining camera and LiDAR
BEV representations.
"""

from typing import List

import torch
import torch.nn as nn
from einops import einsum, rearrange

__all__ = [
    "ConvFuser",
    "AddFuser",
    "MaxFuser",
    "create_conv_fuser",
    "create_add_fuser",
    "create_max_fuser",
    "build_fuser_from_config",
]


class ConvFuser(nn.Module):
    """Concatenation-based feature fusion with convolution.

    This is the primary fusion strategy in BEVFusion. It:
    1. Concatenates features along channel dimension
    2. Applies 3x3 convolution + BN + ReLU
    3. Reduces concatenated channels to output channels

    Args:
        in_channels: List of input channel counts for each modality
        out_channels: Number of output channels after fusion

    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        total_in_channels = sum(in_channels)

        self.fusion = nn.Sequential(
            nn.Conv2d(
                total_in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Fuse features by concatenation + convolution.

        Args:
            features: List of BEV features [B, C_i, H, W] for each modality

        Returns:
            Fused features [B, out_channels, H, W]
        """
        assert len(features) == len(self.in_channels), (
            f"Expected {len(self.in_channels)} features, got {len(features)}"
        )

        # Verify channel counts
        for i, (feat, expected_ch) in enumerate(
            zip(features, self.in_channels)
        ):
            assert feat.shape[1] == expected_ch, (
                f"Feature {i}: expected {expected_ch} channels, got {feat.shape[1]}"
            )

        # Concatenate along channel dimension
        x = torch.cat(features, dim=1)  # [B, sum(C_i), H, W]

        # Fuse with convolution
        x = self.fusion(x)  # [B, out_channels, H, W]

        return x


class AddFuser(nn.Module):
    """Addition-based feature fusion with channel-wise learned weights.

    This fusion strategy:
    1. Projects each modality to same channel count
    2. Applies channel-wise weighted addition (learnable or uniform)
    3. Supports channel-wise dropout during training for robustness

    **Key Feature**: Uses channel-wise fusion weights [num_modalities, out_channels]
    instead of scalar weights, allowing fine-grained control. Each output channel
    can learn different modality importance.

    **Dropout Behavior**: Unlike traditional modality dropout that drops entire
    sensors, this applies element-wise dropout to the fusion weights matrix.
    This means individual channels can have different modalities dropped,
    improving robustness at a finer granularity.

    Args:
        in_channels: List of input channel counts for each modality
        out_channels: Number of output channels after fusion
        dropout: Probability of dropping individual channel-modality weights
        learnable_weights: Use learnable channel-wise fusion weights (vs uniform)

    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        dropout: float = 0.0,
        learnable_fusion_weights: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.learnable_weights = learnable_fusion_weights

        # Transform each modality to output channels
        self.transforms = nn.ModuleList()
        for in_ch in in_channels:
            self.transforms.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Learnable fusion weights (channel-wise)
        # Shape: [num_modalities, out_channels] for fine-grained control
        self.num_modalities = len(in_channels)
        if learnable_fusion_weights:
            # Initialize with uniform weights (will be normalized by softmax)
            self.fusion_weights = nn.Parameter(
                torch.ones(self.num_modalities, out_channels)
            )
        else:
            self.register_buffer(
                "fusion_weights", torch.ones(self.num_modalities, out_channels)
            )

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Fuse features by weighted addition.

        Args:
            features: List of BEV features [B, C_i, H, W]

        Returns:
            Fused features [B, out_channels, H, W]
        """
        assert len(features) == len(self.in_channels)

        # Project to common channel count
        projected = []
        for transform, feat in zip(self.transforms, features):
            projected.append(transform(feat))

        # Get fusion weights: [num_modalities, out_channels]
        weights = self.fusion_weights

        # Apply dropout to features (not weights) for robustness
        if self.training and self.dropout > 0:
            for i in range(len(projected)):
                projected[i] = self.dropout_layer(projected[i])

        # Normalize weights across modalities using softmax
        # This ensures weights sum to 1 across modalities for each channel
        softmax_weights = nn.functional.softmax(
            weights, dim=0
        )  # [num_modalities, out_channels]

        # Rearrange features: [B, C, H, W] -> [B, H, W, C]
        for i in range(len(projected)):
            projected[i] = rearrange(projected[i], "b c h w -> b h w c")

        # Stack modalities: [B, H, W, C] x num_modalities -> [B, H, W, C, num_modalities]
        p_stack = torch.stack(projected, dim=-1)

        # Apply channel-wise weighted sum using einsum
        # For each channel c: output[c] = sum_n(feature[c, n] * weight[n, c])
        fused = einsum(
            p_stack, softmax_weights.t(), "b h w c n, c n -> b h w c"
        )

        # Rearrange back: [B, H, W, C] -> [B, C, H, W]
        fused = rearrange(fused, "b h w c -> b c h w")

        return fused


class MaxFuser(nn.Module):
    """Max pooling-based feature fusion.

    Takes element-wise maximum across modalities after projecting
    each to the same channel count.

    Args:
        in_channels: List of input channel counts
        out_channels: Number of output channels

    Example:
        >>> fuser = MaxFuser(in_channels=[80, 64], out_channels=128)
        >>> fused = fuser([cam_bev, lidar_bev])
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Transform each modality
        self.transforms = nn.ModuleList()
        for in_ch in in_channels:
            self.transforms.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Fuse features by element-wise max.

        Args:
            features: List of BEV features [B, C_i, H, W]

        Returns:
            Fused features [B, out_channels, H, W]
        """
        # Project to common channels
        projected = []
        for transform, feat in zip(self.transforms, features):
            projected.append(transform(feat))

        # Stack and take max
        stacked = torch.stack(projected, dim=0)  # [N_mod, B, C, H, W]
        fused, _ = torch.max(stacked, dim=0)  # [B, C, H, W]

        return fused


def create_conv_fuser(
    camera_channels: int,
    lidar_channels: int,
    out_channels: int = 256,
) -> ConvFuser:
    """Create a ConvFuser for camera-LiDAR fusion.

    Args:
        camera_channels: Number of camera BEV feature channels
        lidar_channels: Number of LiDAR BEV feature channels
        out_channels: Number of fused output channels

    Returns:
        Configured ConvFuser
    """
    return ConvFuser(
        in_channels=[camera_channels, lidar_channels],
        out_channels=out_channels,
    )


def create_add_fuser(
    camera_channels: int,
    lidar_channels: int,
    out_channels: int = 256,
    dropout: float = 0.2,
    learnable_fusion_weights: bool | str = True,
) -> AddFuser:
    """Create an AddFuser for camera-LiDAR fusion.

    Args:
        camera_channels: Number of camera BEV features
        lidar_channels: Number of LiDAR BEV features
        out_channels: Number of fused output channels
        dropout: Modality dropout probability
        learnable_weights: Use learnable fusion weights

    Returns:
        Configured AddFuser

    """
    return AddFuser(
        in_channels=[camera_channels, lidar_channels],
        out_channels=out_channels,
        dropout=dropout,
        learnable_fusion_weights=learnable_fusion_weights
        if isinstance(learnable_fusion_weights, bool)
        else learnable_fusion_weights.lower() == "true",
    )


def create_max_fuser(
    camera_channels: int,
    lidar_channels: int,
    out_channels: int = 256,
) -> MaxFuser:
    """Create a MaxFuser for camera-LiDAR fusion."""
    return MaxFuser(
        in_channels=[camera_channels, lidar_channels],
        out_channels=out_channels,
    )


def build_fuser_from_config(config: dict) -> nn.Module:
    """Build a fuser from config."""
    _config = config.copy()
    fuser_type = _config.pop("type")
    if fuser_type == "conv":
        return create_conv_fuser(**_config)
    elif fuser_type == "add":
        return create_add_fuser(**_config)
    elif fuser_type == "max":
        return create_max_fuser(**_config)
    else:
        raise ValueError(f"Unknown fuser_type: {fuser_type}")

