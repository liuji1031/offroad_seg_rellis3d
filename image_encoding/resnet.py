"""ResNet backbone for image feature extraction.

Adapted from BEVFusion's GeneralizedResNet but simplified for standalone use.
Supports standard ResNet architectures (ResNet18, ResNet50, etc.) via torchvision.
"""

from typing import List, Tuple, Optional, Any
import torch
from torch import nn
from torchvision.models import resnet18, resnet50, resnet101
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights


__all__ = ['ResNetBackbone']


class ResNetBackbone(nn.Module):
    """ResNet backbone for extracting multi-scale image features.
    
    This backbone extracts features at multiple resolutions from input images,
    which are then fused by a neck (FPN) before being fed to the BEV transform.
    
    Args:
        depth: ResNet depth (18, 50, or 101)
        pretrained: Whether to load ImageNet pretrained weights
        num_stages: Number of ResNet stages to use (1-4)
        out_indices: Indices of stages to output features from
        frozen_stages: Number of stages to freeze during training (-1 means none)
        norm_eval: Whether to set norm layers to eval mode
    
    Returns:
        List of feature tensors at different scales
    """
    
    def __init__(
        self,
        depth: int = 50,
        pretrained: bool = True,
        num_stages: int = 4,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        frozen_stages: int = -1,
        norm_eval: bool = False,
    ):
        super().__init__()
        
        assert depth in [18, 50, 101], f"Unsupported ResNet depth: {depth}"
        assert num_stages in [1, 2, 3, 4], f"num_stages must be 1-4, got {num_stages}"
        assert max(out_indices) < num_stages, f"out_indices {out_indices} exceed num_stages {num_stages}"
        
        self.depth = depth
        self.num_stages = num_stages
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        
        # Load pretrained ResNet from torchvision
        if depth == 18:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = resnet18(weights=weights)
            self.stage_channels = [64, 64, 128, 256, 512]
        elif depth == 50:
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            resnet = resnet50(weights=weights)
            self.stage_channels = [64, 256, 512, 1024, 2048]
        else:  # 101
            weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            resnet = resnet101(weights=weights)
            self.stage_channels = [64, 256, 512, 1024, 2048]
        
        # Extract ResNet layers
        # Stage 0: conv1 + bn1 + relu + maxpool
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Stage 1-4: layer1-layer4
        self.layer1 = resnet.layer1 if num_stages >= 1 else nn.Identity()
        self.layer2 = resnet.layer2 if num_stages >= 2 else nn.Identity()
        self.layer3 = resnet.layer3 if num_stages >= 3 else nn.Identity()
        self.layer4 = resnet.layer4 if num_stages >= 4 else nn.Identity()
        
        self._freeze_stages()
        
    def _freeze_stages(self):
        """Freeze stages according to frozen_stages parameter."""
        if self.frozen_stages >= 0:
            # Freeze stem (conv1, bn1)
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False
        
        # Freeze layer1-4 as specified
        for i in range(1, self.frozen_stages + 1):
            if i <= self.num_stages:
                layer = getattr(self, f'layer{i}')
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False
    
    def train(self, mode: bool = True):
        """Override train to handle norm_eval and frozen_stages."""
        super().train(mode)
        self._freeze_stages()
        
        if mode and self.norm_eval:
            # Set all BN layers to eval mode
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        
        return self
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through ResNet.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            List of feature tensors at different scales specified by out_indices
        """
        # Stage 0: Stem
        x = self.conv1(x)      # H/2, W/2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # H/4, W/4
        
        # Collect features from requested stages
        outputs = []
        
        # Stage 1-4
        stages = [self.layer1, self.layer2, self.layer3, self.layer4]
        for i, layer in enumerate(stages[:self.num_stages]):
            x = layer(x)
            if i in self.out_indices:
                outputs.append(x)
        
        return outputs
    
    def get_out_channels(self) -> List[int]:
        """Get output channels for each stage in out_indices.
        
        Returns:
            List of channel numbers corresponding to out_indices
        """
        # Note: stage_channels[0] is stem output, stage_channels[1-4] are layer1-4
        return [self.stage_channels[i + 1] for i in self.out_indices]


def build_resnet_backbone_from_config(config: dict[str, Any]) -> ResNetBackbone:
    """Build ResNet backbone from config."""
    _config = config.copy()
    backbone_type = _config.pop("type")
    assert backbone_type == "ResNetBackbone", f"Unknown backbone type: {backbone_type}"
    return ResNetBackbone(**_config)
