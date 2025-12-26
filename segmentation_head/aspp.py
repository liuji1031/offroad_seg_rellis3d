"""ASPP (Atrous Spatial Pyramid Pooling) modules.

Adapted from DeepLabV3+ with enhancements for polar BEV grids.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['ASPP', 'PolarASPP']


class ASPPConv(nn.Module):
    """ASPP convolution block: Conv + BN + ReLU."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ASPPPooling(nn.Module):
    """ASPP global pooling branch."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        x = self.pool(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module.
    
    Multi-scale context aggregation using parallel atrous convolutions
    with different dilation rates.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (typically 256)
        rates: Dilation rates for atrous convolutions (e.g., [6, 12, 18])
        dropout: Dropout probability after fusion (0 to disable)
        
    Example:
        >>> aspp = ASPP(in_channels=512, out_channels=256, rates=[6, 12, 18])
        >>> x = torch.randn(2, 512, 42, 10)
        >>> out = aspp(x)  # [2, 256, 42, 10]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        rates: List[int] = [6, 12, 18],
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rates = rates
        
        # Branch 1: 1×1 convolution
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # Branches 2-N: Atrous convolutions
        self.aspp_convs = nn.ModuleList()
        for rate in rates:
            self.aspp_convs.append(
                ASPPConv(in_channels, out_channels, dilation=rate)
            )
        
        # Branch N+1: Global pooling
        self.global_pool = ASPPPooling(in_channels, out_channels)
        
        # Fusion: Combine all branches
        num_branches = len(rates) + 2  # 1×1 + atrous + global
        self.project = nn.Sequential(
            nn.Conv2d(
                out_channels * num_branches,
                out_channels,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features [B, C_in, H, W]
            
        Returns:
            Output features [B, C_out, H, W]
        """
        outputs = []
        
        # 1×1 convolution
        outputs.append(self.conv1x1(x))
        
        # Atrous convolutions
        for aspp_conv in self.aspp_convs:
            outputs.append(aspp_conv(x))
        
        # Global pooling
        outputs.append(self.global_pool(x))
        
        # Concatenate and fuse
        x = torch.cat(outputs, dim=1)
        x = self.project(x)
        
        return x


class PolarASPPConv(nn.Module):
    """ASPP convolution with separate radial and angular dilation rates.
    
    For polar grids, we want different dilation rates for the radial (H)
    and angular (W) dimensions since they have different characteristics.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation_radial: int = 1,
        dilation_angular: int = 1,
        use_circular_pad: bool = True,
    ):
        super().__init__()
        self.dilation_radial = dilation_radial
        self.dilation_angular = dilation_angular
        self.use_circular_pad = use_circular_pad
        
        # Calculate padding
        padding_radial = dilation_radial  # For 3×3 kernel
        padding_angular = dilation_angular
        
        # Conv without padding (we'll add it manually for circular padding)
        if use_circular_pad:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=(padding_radial, 0),  # Only pad radial, we'll handle angular
                dilation=(dilation_radial, dilation_angular),
                bias=False,
            )
            self.padding_angular = padding_angular
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=(padding_radial, padding_angular),
                dilation=(dilation_radial, dilation_angular),
                bias=False,
            )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply circular padding on angular dimension if needed
        if self.use_circular_pad and self.padding_angular > 0:
            # Circular padding on width (angular dimension)
            x = F.pad(x, (self.padding_angular, self.padding_angular, 0, 0), 
                     mode='circular')
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PolarASPP(nn.Module):
    """ASPP adapted for polar BEV grids.
    
    This variant allows independent control of dilation rates for radial
    and angular dimensions, and supports circular padding for the angular
    dimension (wraparound at 2π).
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        rates_radial: Dilation rates for radial dimension
        rates_angular: Dilation rates for angular dimension
        use_circular_pad: Use circular padding for angular dimension
        dropout: Dropout probability after fusion
        
    Example:
        >>> # For 42 radial × 10 angular grid
        >>> aspp = PolarASPP(
        ...     in_channels=512,
        ...     out_channels=256,
        ...     rates_radial=[2, 4, 6],    # Smaller rates for larger dimension
        ...     rates_angular=[1, 2, 3],   # Even smaller for tiny dimension
        ...     use_circular_pad=True,
        ... )
        >>> x = torch.randn(2, 512, 42, 10)
        >>> out = aspp(x)  # [2, 256, 42, 10]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        rates_radial: tuple[int, ...] = (2, 4, 6),
        rates_angular: tuple[int, ...] = (1, 2, 3),
        use_circular_pad: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rates_radial = rates_radial
        self.rates_angular = rates_angular
        
        assert len(rates_radial) == len(rates_angular), \
            "Must have same number of radial and angular rates"
        
        # Branch 1: 1×1 convolution
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # Branches 2-N: Polar atrous convolutions
        self.polar_convs = nn.ModuleList()
        for rate_r, rate_a in zip(rates_radial, rates_angular):
            self.polar_convs.append(
                PolarASPPConv(
                    in_channels,
                    out_channels,
                    dilation_radial=rate_r,
                    dilation_angular=rate_a,
                    use_circular_pad=use_circular_pad,
                )
            )
        
        # Branch N+1: Global pooling
        self.global_pool = ASPPPooling(in_channels, out_channels)
        
        # Fusion
        num_branches = len(rates_radial) + 2
        self.project = nn.Sequential(
            nn.Conv2d(
                out_channels * num_branches,
                out_channels,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features [B, C_in, H_radial, W_angular]
            
        Returns:
            Output features [B, C_out, H_radial, W_angular]
        """
        outputs = []
        
        # 1×1 convolution
        outputs.append(self.conv1x1(x))
        
        # Polar atrous convolutions
        for polar_conv in self.polar_convs:
            outputs.append(polar_conv(x))
        
        # Global pooling
        outputs.append(self.global_pool(x))
        
        # Concatenate and fuse
        x = torch.cat(outputs, dim=1)
        x = self.project(x)
        
        return x
