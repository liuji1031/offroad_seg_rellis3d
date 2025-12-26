"""Complete point cloud encoding pipeline.

This module provides a high-level wrapper that combines voxelization, feature encoding,
and scattering into a complete point cloud encoder.
"""

import json
import os
from typing import Any, Dict, List, Tuple

import torch
from torch import nn

from .pointpillars import (
    PointPillarFeatureNet,
    PointPillarScatter,
    PolarPointPillarFeatureNet,
)
from .voxelization import (
    DynamicVoxelization,
    HardPolarVoxelization,
    HardVoxelization,
    PolarVoxelization,
)

__all__ = [
    "PointCloudEncoder",
    "create_pointpillars_encoder",
    "create_polar_pointpillars_encoder",
]


class PointCloudEncoder(nn.Module):
    """Complete point cloud encoding pipeline.

    This encoder takes raw LiDAR point clouds and produces dense BEV features:

    1. **Voxelization**: Convert sparse points to voxels/pillars
    2. **Feature Encoding**: Extract features from each pillar
    3. **Scatter**: Create dense BEV grid

    Args:
        voxelize_config: Configuration dict for voxelization
        encoder_config: Configuration dict for feature encoder
        scatter_config: Configuration dict for scatter module
    """

    # Registry of available components
    VOXELIZERS = {
        "DynamicVoxelization": DynamicVoxelization,
        "HardVoxelization": HardVoxelization,
        "PolarVoxelization": PolarVoxelization,
        "HardPolarVoxelization": HardPolarVoxelization,
    }

    ENCODERS = {
        "PointPillarFeatureNet": PointPillarFeatureNet,
        "PolarPointPillarFeatureNet": PolarPointPillarFeatureNet,
    }

    SCATTERS = {
        "PointPillarScatter": PointPillarScatter,
    }

    def __init__(
        self,
        voxelize_config: Dict[str, Any],
        encoder_config: Dict[str, Any],
        scatter_config: Dict[str, Any],
    ):
        super().__init__()

        # Build voxelizer
        voxelize_type = voxelize_config.pop("type")
        assert voxelize_type in self.VOXELIZERS, (
            f"Unknown voxelizer: {voxelize_type}"
        )
        self.voxelizer = self.VOXELIZERS[voxelize_type](**voxelize_config)

        # Build encoder
        encoder_type = encoder_config.pop("type")
        assert encoder_type in self.ENCODERS, f"Unknown encoder: {encoder_type}"
        self.encoder = self.ENCODERS[encoder_type](**encoder_config)

        # Build scatter
        scatter_type = scatter_config.pop("type")
        assert scatter_type in self.SCATTERS, f"Unknown scatter: {scatter_type}"
        self.scatter = self.SCATTERS[scatter_type](**scatter_config)

        # Store types for debugging
        self.voxelize_type = voxelize_type
        self.encoder_type = encoder_type
        self.scatter_type = scatter_type

    def forward(
        self,
        points: List[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass through full pipeline.

        Args:
            points: List of [N_i, C] point clouds (one per batch element)
                Each point cloud has C features (typically 4: x, y, z, intensity)

        Returns:
            bev_features: [B, C_out, H, W] dense BEV features
        """
        batch_size = len(points)

        # Check if using polar voxelization
        is_polar = isinstance(
            self.voxelizer, (PolarVoxelization, HardPolarVoxelization)
        )

        # 1. Voxelize all point clouds
        voxels_list = []
        coords_list = []
        num_points_list = []
        voxel_centers_list = []

        for batch_idx, pts in enumerate(points):
            if isinstance(
                self.voxelizer, (HardVoxelization, HardPolarVoxelization)
            ):
                # Hard voxelization returns 3 or 4 values
                voxelizer_output = self.voxelizer(pts)

                if isinstance(self.voxelizer, HardPolarVoxelization):
                    # Polar: returns (voxels, coords, num_points, voxel_centers)
                    voxels, coords, num_points, voxel_centers = voxelizer_output
                    voxel_centers_list.append(voxel_centers)
                else:
                    # Standard: returns (voxels, coords, num_points)
                    voxels, coords, num_points = voxelizer_output

            elif isinstance(self.voxelizer, PolarVoxelization):
                # Polar dynamic voxelization returns (points, coords)
                # Note: coords include voxel indices but we don't have voxel_centers here
                # This case is currently not fully supported - would need enhancement
                raise NotImplementedError(
                    "PolarVoxelization (dynamic) is not yet supported in PointCloudEncoder. "
                    "Use HardPolarVoxelization instead."
                )
            else:  # DynamicVoxelization
                voxel_features, coords = self.voxelizer(pts)
                # For dynamic, we don't have structured voxels
                # Just use the points directly
                voxels = voxel_features.unsqueeze(1)  # [N, 1, C]
                num_points = torch.ones(
                    len(voxels), dtype=torch.int32, device=pts.device
                )

            # Add batch index to coordinates
            batch_indices = torch.full(
                (len(coords), 1),
                batch_idx,
                dtype=torch.int32,
                device=coords.device,
            )
            coords_with_batch = torch.cat([batch_indices, coords], dim=1)

            voxels_list.append(voxels)
            coords_list.append(coords_with_batch)
            num_points_list.append(num_points)

        # Concatenate all batches
        voxels = torch.cat(voxels_list, dim=0)
        coords = torch.cat(coords_list, dim=0)
        num_points = torch.cat(num_points_list, dim=0)
        voxel_centers = (
            torch.cat(voxel_centers_list, dim=0) if is_polar else None
        )

        # 2. Encode features
        if isinstance(self.encoder, PolarPointPillarFeatureNet):
            # Polar encoder needs voxel_centers instead of coords
            assert voxel_centers is not None, (
                "Polar encoder requires voxel_centers from polar voxelization"
            )
            voxel_features = self.encoder(voxels, num_points, voxel_centers)
        else:
            # Standard encoder uses coords
            voxel_features = self.encoder(voxels, num_points, coords)

        # 3. Scatter to BEV
        bev_features = self.scatter(voxel_features, coords, batch_size)

        return bev_features

    def get_output_shape(
        self, batch_size: int = 1
    ) -> Tuple[int, int, int, int]:
        """Get output BEV feature shape.

        Args:
            batch_size: Batch size

        Returns:
            (B, C, H, W) output shape
        """
        return (
            batch_size,
            self.scatter.in_channels,
            self.scatter.nx,
            self.scatter.ny,
        )

    def __repr__(self) -> str:
        return (
            f"PointCloudEncoder(\n"
            f"  voxelizer={self.voxelize_type},\n"
            f"  encoder={self.encoder_type},\n"
            f"  scatter={self.scatter_type},\n"
            f"  out_shape={self.get_output_shape()}\n"
            f")"
        )


def create_pointpillars_encoder(
    voxel_size: Tuple[float, float, float] = (0.2, 0.2, 8),
    point_cloud_range: Tuple[float, float, float, float, float, float] = (
        0,
        -25.6,
        -3,
        51.2,
        25.6,
        5,
    ),
    max_points_per_voxel: int = 32,
    max_voxels: int = 16000,
    feat_channels: Tuple[int, ...] = (64,),
    in_channels: int = 4,
) -> PointCloudEncoder:
    """Create a standard PointPillars encoder (common BEVFusion config).

    This is one of the most common configurations used in autonomous driving
    for efficient point cloud processing.

    Args:
        voxel_size: Size of each voxel/pillar (x, y, z) in meters
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        max_points_per_voxel: Maximum points to keep per voxel
        max_voxels: Maximum number of voxels
        feat_channels: Feature dimensions for PFN layers
        in_channels: Number of input point features

    Returns:
        Configured PointCloudEncoder
    """
    # Calculate output shape
    nx = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
    ny = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

    encoder = PointCloudEncoder(
        voxelize_config={
            "type": "HardVoxelization",
            "voxel_size": voxel_size,
            "point_cloud_range": point_cloud_range,
            "max_points_per_voxel": max_points_per_voxel,
            "max_voxels": max_voxels,
        },
        encoder_config={
            "type": "PointPillarFeatureNet",
            "in_channels": in_channels,
            "feat_channels": feat_channels,
            "voxel_size": voxel_size,
            "point_cloud_range": point_cloud_range,
        },
        scatter_config={
            "type": "PointPillarScatter",
            "in_channels": feat_channels[-1],
            "output_shape": (nx, ny),
        },
    )

    return encoder


def create_polar_pointpillars_encoder(
    polar_grid_config: str,
    point_cloud_range_z: Tuple[float, float] = (-3, 5),
    max_points_per_voxel: int = 32,
    max_voxels: int = 16000,
    feat_channels: Tuple[int, ...] = (64,),
    in_channels: int = 4,
) -> PointCloudEncoder:
    """Create a Polar PointPillars encoder for polar BEV grids.

    This configuration is designed for datasets like RELLIS-3D that use
    non-uniform polar grids for BEV segmentation.

    Args:
        polar_grid_config: Path to JSON file with polar grid configuration
        point_cloud_range_z: Z range [z_min, z_max] in meters
        max_points_per_voxel: Maximum points per pillar
        max_voxels: Maximum number of pillars
        feat_channels: Output channels for PFN layers
        in_channels: Input feature channels (typically 4: x, y, z, intensity)

    Returns:
        Configured PointCloudEncoder with polar grid support
    """
    # Load polar grid config to get dimensions
    with open(polar_grid_config, "r", encoding="utf-8") as f:
        config = json.load(f)

    n_radial = len(config["radial_edges"]) - 1
    n_angular = len(config["angle_edges"]) - 1

    encoder = PointCloudEncoder(
        voxelize_config={
            "type": "HardPolarVoxelization",
            "polar_grid_config": polar_grid_config,
            "point_cloud_range_z": point_cloud_range_z,
            "max_points_per_voxel": max_points_per_voxel,
            "max_voxels": max_voxels,
        },
        encoder_config={
            "type": "PolarPointPillarFeatureNet",
            "in_channels": in_channels,
            "feat_channels": feat_channels,
            "with_distance": False,
        },
        scatter_config={
            "type": "PointPillarScatter",
            "in_channels": feat_channels[-1],
            "output_shape": (n_radial, n_angular),
        },
    )

    return encoder
