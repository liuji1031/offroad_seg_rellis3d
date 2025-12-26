"""Fusion module for combining multi-modal BEV features.

This module provides various fusion strategies for combining camera and LiDAR
BEV features, following the BEVFusion architecture.
"""

from .fusers import (
    ConvFuser,
    AddFuser,
    MaxFuser,
    create_conv_fuser,
    create_add_fuser,
    create_max_fuser,
    build_fuser_from_config,
)

__all__ = [
    'ConvFuser',
    'AddFuser',
    'MaxFuser',
    'create_conv_fuser',
    'create_add_fuser',
    'create_max_fuser',
    'build_fuser_from_config',
]

