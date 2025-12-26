"""Complete BEVFusion model for RELLIS-3D off-road segmentation.

This module provides the end-to-end model that integrates:
- Image encoding (camera)
- Point cloud encoding (LiDAR)
- Multi-modal fusion
- Segmentation head
"""

from .rellis_polar_bev_fusion import RellisPolarBevFusion

__all__ = [
    'RellisPolarBevFusion',
]



