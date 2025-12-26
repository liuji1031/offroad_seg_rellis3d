"""Point Cloud Encoding Module for LiDAR Feature Extraction.

This module provides the point cloud encoding pipeline that extracts BEV features
from raw LiDAR point clouds, parallel to the image encoding module.

Components:
- Voxelization: Convert sparse point clouds to voxel/pillar representation (Cartesian or Polar)
- PointPillars encoder: Efficient point cloud feature extraction
- Scatter: Convert voxel features to dense BEV grid
- Point cloud encoder wrapper: Complete end-to-end encoding pipeline
"""

from .voxelization import (
    DynamicVoxelization,
    HardVoxelization,
    PolarVoxelization,
    HardPolarVoxelization,
)
from .pointpillars import (
    PointPillarFeatureNet,
    PointPillarScatter,
    PolarPointPillarFeatureNet,
)
from .point_cloud_encoder import (
    PointCloudEncoder,
    create_pointpillars_encoder,
    create_polar_pointpillars_encoder,
)

__all__ = [
    'DynamicVoxelization',
    'HardVoxelization',
    'PolarVoxelization',
    'HardPolarVoxelization',
    'PointPillarFeatureNet',
    'PointPillarScatter',
    'PolarPointPillarFeatureNet',
    'PointCloudEncoder',
    'create_pointpillars_encoder',
    'create_polar_pointpillars_encoder',
]

