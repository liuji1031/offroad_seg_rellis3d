"""Feature Pyramid Network (FPN) implementations for multi-scale feature fusion.

Adapted from BEVFusion's FPN variants. These necks take multi-scale backbone
features and fuse them into a unified representation for the BEV transform.
"""

from typing import List, Tuple, Any, Union
import torch
from torch import nn
from torch.nn import functional as F


__all__ = ['LSSFPN', 'SECONDFPN', 'GeneralizedLSSFPN']


class LSSFPN(nn.Module):
    """Simple FPN used in LSS (Lift-Splat-Shoot).
    
    Takes two feature maps at different scales, upsamples and concatenates them,
    then applies convolutions for fusion. Optionally upsamples the final output.
    
    Args:
        in_indices: Indices of input features to use (must be length 2)
        in_channels: Input channel numbers for each feature
        out_channels: Output channel number
        scale_factor: Final upsampling factor (1 = no upsampling)
    """
    
    def __init__(
        self,
        in_indices: Tuple[int, int],
        in_channels: Tuple[int, int],
        out_channels: int,
        scale_factor: int = 1,
    ):
        super().__init__()
        
        assert len(in_indices) == 2, "LSSFPN requires exactly 2 input features"
        assert len(in_channels) == 2, "in_channels must match in_indices length"
        
        self.in_indices = in_indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        
        # Fusion: concatenate upsampled features and apply conv
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels[0] + in_channels[1], out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        
        # Optional upsampling
        if scale_factor > 1:
            self.upsample = nn.Sequential(
                nn.Upsample(
                    scale_factor=scale_factor,
                    mode='bilinear',
                    align_corners=True,
                ),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.upsample = None
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass.
        
        Args:
            features: List of feature tensors from backbone
            
        Returns:
            Fused feature tensor
        """
        # Select features
        x1 = features[self.in_indices[0]]
        x2 = features[self.in_indices[1]]
        
        assert x1.shape[1] == self.in_channels[0], f"Expected {self.in_channels[0]} channels, got {x1.shape[1]}"
        assert x2.shape[1] == self.in_channels[1], f"Expected {self.in_channels[1]} channels, got {x2.shape[1]}"
        
        # Upsample x1 to match x2 spatial size
        x1 = F.interpolate(x1, size=x2.shape[-2:], mode='bilinear', align_corners=True)
        
        # Concatenate and fuse
        x = torch.cat([x1, x2], dim=1)
        x = self.fuse(x)
        
        # Optional final upsampling
        if self.upsample is not None:
            x = self.upsample(x)
        
        return x


class SECONDFPN(nn.Module):
    """FPN used in SECOND/PointPillars.
    
    Upsamples each input feature to a common resolution and concatenates them.
    Each feature goes through deconv/conv + BN + ReLU before concatenation.
    
    Args:
        in_channels: Input channel numbers for each feature
        out_channels: Output channel numbers for each feature
        upsample_strides: Upsampling factors for each feature
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        upsample_strides: List[float],
    ):
        super().__init__()
        
        assert len(in_channels) == len(out_channels) == len(upsample_strides)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_strides = upsample_strides
        
        # Build upsampling blocks
        deblocks = []
        for in_channel, out_channel, stride in zip(in_channels, out_channels, upsample_strides):
            if stride >= 1:
                # Upsample with transposed convolution
                kernel_size = int(stride)
                deblock = nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channel,
                        out_channel,
                        kernel_size=kernel_size,
                        stride=kernel_size,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(inplace=True),
                )
            else:
                # Downsample with regular convolution
                kernel_size = int(1 / stride)
                deblock = nn.Sequential(
                    nn.Conv2d(
                        in_channel,
                        out_channel,
                        kernel_size=kernel_size,
                        stride=kernel_size,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(inplace=True),
                )
            deblocks.append(deblock)
        
        self.deblocks = nn.ModuleList(deblocks)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass.
        
        Args:
            features: List of feature tensors from backbone
            
        Returns:
            List with single concatenated tensor
        """
        assert len(features) == len(self.in_channels)
        
        # Upsample each feature
        ups = [deblock(feat) for deblock, feat in zip(self.deblocks, features)]
        
        # Concatenate along channel dimension
        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        
        return [out]


class GeneralizedLSSFPN(nn.Module):
    """Generalized LSS FPN with top-down pathway.
    
    Similar to standard FPN but fuses features in a top-down manner by 
    upsampling higher-level features and concatenating with lower-level ones.
    
    Args:
        in_channels: Input channel numbers for each feature level
        out_channels: Output channel number (same for all levels)
        start_level: Starting feature level index
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        start_level: int = 0,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_level = start_level
        self.num_ins = len(in_channels)
        
        # Build lateral and output convolutions
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for i in range(start_level, self.num_ins - 1):
            # Lateral conv: fuse upsampled high-level with current level
            l_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels[i] + (in_channels[i + 1] if i == self.num_ins - 2 else out_channels),
                    out_channels,
                    1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            
            # Output conv: refine fused features
            fpn_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
    
    def forward(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Forward pass.
        
        Args:
            inputs: List of feature tensors from backbone
            
        Returns:
            Tuple of fused feature tensors
        """
        assert len(inputs) == len(self.in_channels)
        
        # Build laterals (just copy for now)
        laterals = [inputs[i + self.start_level] for i in range(len(inputs))]
        
        # Build top-down pathway
        used_backbone_levels = len(laterals) - 1
        
        for i in range(used_backbone_levels - 1, -1, -1):
            # Upsample higher-level feature
            x_up = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                mode='bilinear',
                align_corners=True,
            )
            
            # Concatenate with current level
            laterals[i] = torch.cat([laterals[i], x_up], dim=1)
            
            # Apply convolutions
            laterals[i] = self.lateral_convs[i](laterals[i])
            laterals[i] = self.fpn_convs[i](laterals[i])
        
        # Return requested outputs
        outs = [laterals[i] for i in range(used_backbone_levels)]
        return tuple(outs)


def build_fpn_from_config(config: dict[str, Any]) -> Union[LSSFPN, SECONDFPN, GeneralizedLSSFPN]:
    """Build FPN from config."""
    fpn_type = config.pop("type")
    if fpn_type.lower() == "lssfpn":
        return LSSFPN(**config)
    elif fpn_type.lower() == "secondfpn":
        return SECONDFPN(**config)
    elif fpn_type.lower() == "generalizedlssfpn":
        return GeneralizedLSSFPN(**config)
    else:
        raise ValueError(f"Unknown FPN type: {fpn_type}")

