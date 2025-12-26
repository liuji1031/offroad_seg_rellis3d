"""Image Encoding Module for Camera Feature Extraction.

This module provides the image encoding pipeline that extracts features from 
raw camera images, serving as input to the BEV transform module.

Components:
- ResNet backbone: Feature extraction from images
- FPN necks: Multi-scale feature fusion
- Image encoder wrapper: Complete end-to-end encoding pipeline
"""

from .resnet import ResNetBackbone
from .fpn import LSSFPN, SECONDFPN, GeneralizedLSSFPN
from .image_encoder import (
    ImageEncoder,
    create_resnet50_lssfpn,
    create_resnet50_secondfpn,
    build_image_encoder_from_config,
)

__all__ = [
    'ResNetBackbone',
    'LSSFPN',
    'SECONDFPN', 
    'GeneralizedLSSFPN',
    'ImageEncoder',
    'create_resnet50_lssfpn',
    'create_resnet50_secondfpn',
    'build_image_encoder_from_config',
]

