"""Voxelization modules for converting point clouds to voxel/pillar representation.

Adapted from BEVFusion and MMDetection3D for standalone use.
Includes support for polar grid voxelization.
"""

from typing import Tuple, Union, Dict
import json
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np


__all__ = ['DynamicVoxelization', 'HardVoxelization', 'PolarVoxelization', 'HardPolarVoxelization']


class DynamicVoxelization(nn.Module):
    """Dynamic voxelization that handles variable number of points per voxel.
    
    This is more flexible than hard voxelization as it doesn't limit points per voxel,
    but requires dynamic scatter operations.
    
    Args:
        voxel_size: Size of each voxel (x, y, z) in meters
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        
    Example:
        >>> voxelizer = DynamicVoxelization(
        ...     voxel_size=(0.2, 0.2, 8),  # Pillars: infinite height
        ...     point_cloud_range=(0, -25.6, -3, 51.2, 25.6, 5),
        ... )
        >>> points = torch.randn(10000, 4)  # [N, 4] (x, y, z, intensity)
        >>> voxel_features, voxel_coords = voxelizer(points)
    """
    
    def __init__(
        self,
        voxel_size: Tuple[float, float, float],
        point_cloud_range: Tuple[float, float, float, float, float, float],
    ):
        super().__init__()
        
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        
        # Calculate grid size
        self.grid_size = [
            int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
            int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
            int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]),
        ]
        
        # Register as buffers (will move with model to GPU)
        self.register_buffer(
            'voxel_size_tensor',
            torch.tensor(voxel_size, dtype=torch.float32)
        )
        self.register_buffer(
            'pc_range_tensor',
            torch.tensor(point_cloud_range, dtype=torch.float32)
        )
    
    @torch.no_grad()
    def forward(
        self, 
        points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Voxelize points.
        
        Args:
            points: [N, C] point cloud (C >= 3 for x, y, z)
            
        Returns:
            voxel_features: [M, C] features for each voxel
            voxel_coords: [M, 3] integer coordinates (x_idx, y_idx, z_idx)
        """
        # Get xyz coordinates
        points_xyz = points[:, :3]
        
        # Filter points outside range
        mask = (
            (points_xyz[:, 0] >= self.point_cloud_range[0]) &
            (points_xyz[:, 0] < self.point_cloud_range[3]) &
            (points_xyz[:, 1] >= self.point_cloud_range[1]) &
            (points_xyz[:, 1] < self.point_cloud_range[4]) &
            (points_xyz[:, 2] >= self.point_cloud_range[2]) &
            (points_xyz[:, 2] < self.point_cloud_range[5])
        )
        
        points = points[mask]
        points_xyz = points_xyz[mask]
        
        if len(points) == 0:
            # Return empty tensors with correct shape
            return (
                torch.zeros((0, points.shape[1]), dtype=points.dtype, device=points.device),
                torch.zeros((0, 3), dtype=torch.int32, device=points.device),
            )
        
        # Compute voxel coordinates
        voxel_coords = torch.floor(
            (points_xyz - self.pc_range_tensor[:3]) / self.voxel_size_tensor
        ).int()
        
        # Clamp to grid bounds
        voxel_coords[:, 0] = torch.clamp(voxel_coords[:, 0], 0, self.grid_size[0] - 1)
        voxel_coords[:, 1] = torch.clamp(voxel_coords[:, 1], 0, self.grid_size[1] - 1)
        voxel_coords[:, 2] = torch.clamp(voxel_coords[:, 2], 0, self.grid_size[2] - 1)
        
        return points, voxel_coords


class HardVoxelization(nn.Module):
    """Hard voxelization with fixed maximum points per voxel.
    
    This limits the number of points per voxel, making it more memory-efficient
    but potentially losing information if voxels are very dense.
    
    Args:
        voxel_size: Size of each voxel (x, y, z) in meters
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        max_points_per_voxel: Maximum number of points to keep per voxel
        max_voxels: Maximum number of voxels to generate
        
    Example:
        >>> voxelizer = HardVoxelization(
        ...     voxel_size=(0.2, 0.2, 8),
        ...     point_cloud_range=(0, -25.6, -3, 51.2, 25.6, 5),
        ...     max_points_per_voxel=32,
        ...     max_voxels=16000,
        ... )
        >>> points = torch.randn(10000, 4)
        >>> voxels, coords, num_points = voxelizer(points)
    """
    
    def __init__(
        self,
        voxel_size: Tuple[float, float, float],
        point_cloud_range: Tuple[float, float, float, float, float, float],
        max_points_per_voxel: int = 32,
        max_voxels: int = 16000,
    ):
        super().__init__()
        
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels
        
        # Calculate grid size
        self.grid_size = [
            int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
            int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
            int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]),
        ]
        
        self.register_buffer(
            'voxel_size_tensor',
            torch.tensor(voxel_size, dtype=torch.float32)
        )
        self.register_buffer(
            'pc_range_tensor',
            torch.tensor(point_cloud_range, dtype=torch.float32)
        )
    
    @torch.no_grad()
    def forward(
        self,
        points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Voxelize points with hard limit.
        
        Args:
            points: [N, C] point cloud
            
        Returns:
            voxels: [M, max_points_per_voxel, C] voxel features
            voxel_coords: [M, 3] voxel coordinates
            num_points_per_voxel: [M] number of points in each voxel
        """
        # Get xyz coordinates
        points_xyz = points[:, :3]
        
        # Filter points outside range
        mask = (
            (points_xyz[:, 0] >= self.point_cloud_range[0]) &
            (points_xyz[:, 0] < self.point_cloud_range[3]) &
            (points_xyz[:, 1] >= self.point_cloud_range[1]) &
            (points_xyz[:, 1] < self.point_cloud_range[4]) &
            (points_xyz[:, 2] >= self.point_cloud_range[2]) &
            (points_xyz[:, 2] < self.point_cloud_range[5])
        )
        
        points = points[mask]
        points_xyz = points_xyz[mask]
        
        if len(points) == 0:
            return (
                torch.zeros((0, self.max_points_per_voxel, points.shape[1]),
                           dtype=points.dtype, device=points.device),
                torch.zeros((0, 3), dtype=torch.int32, device=points.device),
                torch.zeros((0,), dtype=torch.int32, device=points.device),
            )
        
        # Compute voxel coordinates
        voxel_coords = torch.floor(
            (points_xyz - self.pc_range_tensor[:3]) / self.voxel_size_tensor
        ).int()
        
        # Clamp to grid bounds
        voxel_coords[:, 0] = torch.clamp(voxel_coords[:, 0], 0, self.grid_size[0] - 1)
        voxel_coords[:, 1] = torch.clamp(voxel_coords[:, 1], 0, self.grid_size[1] - 1)
        voxel_coords[:, 2] = torch.clamp(voxel_coords[:, 2], 0, self.grid_size[2] - 1)
        
        # Convert 3D coords to 1D hash
        voxel_hash = (
            voxel_coords[:, 0] * self.grid_size[1] * self.grid_size[2] +
            voxel_coords[:, 1] * self.grid_size[2] +
            voxel_coords[:, 2]
        )
        
        # Sort by voxel hash for grouping
        sorted_indices = torch.argsort(voxel_hash)
        sorted_hash = voxel_hash[sorted_indices]
        sorted_coords = voxel_coords[sorted_indices]
        sorted_points = points[sorted_indices]
        
        # Find unique voxels
        unique_hash, inverse_indices = torch.unique(sorted_hash, return_inverse=True)
        num_unique = len(unique_hash)
        
        # Limit number of voxels
        if num_unique > self.max_voxels:
            num_unique = self.max_voxels
            unique_hash = unique_hash[:self.max_voxels]
        
        # Initialize output tensors
        voxels = torch.zeros(
            (num_unique, self.max_points_per_voxel, points.shape[1]),
            dtype=points.dtype,
            device=points.device
        )
        coords = torch.zeros((num_unique, 3), dtype=torch.int32, device=points.device)
        num_points = torch.zeros((num_unique,), dtype=torch.int32, device=points.device)
        
        # Fill voxels
        for i in range(num_unique):
            voxel_mask = inverse_indices == i
            voxel_points = sorted_points[voxel_mask]
            n_points = min(len(voxel_points), self.max_points_per_voxel)
            
            voxels[i, :n_points] = voxel_points[:n_points]
            coords[i] = sorted_coords[voxel_mask][0]
            num_points[i] = n_points
        
        return voxels, coords, num_points


class PolarVoxelization(nn.Module):
    """Polar voxelization with non-uniform radial bins and uniform angular bins.
    
    Uses polar coordinates (r, theta) instead of Cartesian (x, y). This is more
    efficient for forward-facing sensors as it matches the sensor's natural geometry.
    
    Args:
        polar_grid_config: Path to JSON config file or dict with polar grid parameters
        point_cloud_range_z: [z_min, z_max] vertical range
        
    Example:
        >>> voxelizer = PolarVoxelization(
        ...     polar_grid_config='configs/rellis/bev_seg_config.json',
        ...     point_cloud_range_z=(-3, 5),
        ... )
        >>> points = torch.randn(10000, 4)  # [N, 4] (x, y, z, intensity)
        >>> voxel_features, voxel_coords = voxelizer(points)
    """
    
    def __init__(
        self,
        polar_grid_config: Union[str, Path, Dict],
        point_cloud_range_z: Tuple[float, float] = (-3, 5),
    ):
        super().__init__()
        
        # Load polar grid config
        if isinstance(polar_grid_config, (str, Path)):
            with open(polar_grid_config, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = polar_grid_config
        
        self.radial_edges = config['radial_edges']
        self.angle_edges = config['angle_edges']
        self.z_min, self.z_max = point_cloud_range_z
        
        # Grid size
        self.n_radial = len(self.radial_edges) - 1
        self.n_angular = len(self.angle_edges) - 1
        self.n_z = 1  # Collapsed z dimension (pillar)
        
        # Register as buffers
        self.register_buffer(
            'radial_edges_tensor',
            torch.tensor(self.radial_edges, dtype=torch.float32)
        )
        self.register_buffer(
            'angle_edges_tensor',
            torch.tensor(self.angle_edges, dtype=torch.float32)
        )
    
    @torch.no_grad()
    def forward(
        self,
        points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Voxelize points in polar coordinates.
        
        Args:
            points: [N, C] point cloud (C >= 3 for x, y, z)
            
        Returns:
            voxel_features: [M, C] features for each voxel
            voxel_coords: [M, 3] integer coordinates (r_idx, theta_idx, z_idx)
        """
        # Get xyz coordinates
        points_xyz = points[:, :3]
        
        # Filter points by z range
        z_mask = (
            (points_xyz[:, 2] >= self.z_min) &
            (points_xyz[:, 2] < self.z_max)
        )
        
        points = points[z_mask]
        points_xyz = points_xyz[z_mask]
        
        if len(points) == 0:
            return (
                torch.zeros((0, points.shape[1]), dtype=points.dtype, device=points.device),
                torch.zeros((0, 3), dtype=torch.int32, device=points.device),
            )
        
        # Convert to polar coordinates
        x, y = points_xyz[:, 0], points_xyz[:, 1]
        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)
        
        # Filter by radial and angular range
        r_mask = (r >= self.radial_edges[0]) & (r < self.radial_edges[-1])
        theta_mask = (theta >= self.angle_edges[0]) & (theta < self.angle_edges[-1])
        mask = r_mask & theta_mask
        
        points = points[mask]
        r = r[mask]
        theta = theta[mask]
        
        if len(points) == 0:
            return (
                torch.zeros((0, points.shape[1]), dtype=points.dtype, device=points.device),
                torch.zeros((0, 3), dtype=torch.int32, device=points.device),
            )
        
        # Find bin indices using searchsorted (non-uniform radial bins)
        r_idx = torch.searchsorted(self.radial_edges_tensor, r, right=False) - 1  # type: ignore
        theta_idx = torch.searchsorted(self.angle_edges_tensor, theta, right=False) - 1  # type: ignore
        
        # Clamp to valid range
        r_idx = torch.clamp(r_idx, 0, self.n_radial - 1)
        theta_idx = torch.clamp(theta_idx, 0, self.n_angular - 1)
        
        # Z dimension collapsed to 0 (pillar)
        z_idx = torch.zeros_like(r_idx)
        
        # Stack coordinates
        voxel_coords = torch.stack([r_idx, theta_idx, z_idx], dim=1).int()
        
        return points, voxel_coords


class HardPolarVoxelization(nn.Module):
    """Hard polar voxelization with fixed maximum points per voxel.
    
    Combines polar coordinate system with hard voxelization limits for
    memory-efficient processing.
    
    Args:
        polar_grid_config: Path to JSON config file or dict with polar grid parameters
        point_cloud_range_z: [z_min, z_max] vertical range
        max_points_per_voxel: Maximum number of points to keep per voxel
        max_voxels: Maximum number of voxels to generate
        
    Example:
        >>> voxelizer = HardPolarVoxelization(
        ...     polar_grid_config='configs/rellis/bev_seg_config.json',
        ...     point_cloud_range_z=(-3, 5),
        ...     max_points_per_voxel=32,
        ...     max_voxels=5000,
        ... )
        >>> points = torch.randn(10000, 4)
        >>> voxels, coords, num_points = voxelizer(points)
    """
    
    def __init__(
        self,
        polar_grid_config: Union[str, Path, Dict],
        point_cloud_range_z: Tuple[float, float] = (-3, 5),
        max_points_per_voxel: int = 32,
        max_voxels: int = 5000,
    ):
        super().__init__()
        
        # Load polar grid config
        if isinstance(polar_grid_config, (str, Path)):
            with open(polar_grid_config, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = polar_grid_config
        
        self.radial_edges = config['radial_edges']
        self.angle_edges = config['angle_edges']
        self.z_min, self.z_max = point_cloud_range_z
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels
        
        # Grid size
        self.n_radial = len(self.radial_edges) - 1
        self.n_angular = len(self.angle_edges) - 1
        self.n_z = 1  # Collapsed z dimension
        
        # Register as buffers
        self.register_buffer(
            'radial_edges_tensor',
            torch.tensor(self.radial_edges, dtype=torch.float32)
        )
        self.register_buffer(
            'angle_edges_tensor',
            torch.tensor(self.angle_edges, dtype=torch.float32)
        )
    
    @torch.no_grad()
    def forward(
        self,
        points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Voxelize points in polar coordinates with hard limit.
        
        Args:
            points: [N, C] point cloud
            
        Returns:
            voxels: [M, max_points_per_voxel, C] voxel features
            voxel_coords: [M, 3] voxel coordinates (r_idx, theta_idx, z_idx)
            num_points_per_voxel: [M] number of points in each voxel
            voxel_centers: [M, 2] Cartesian centers (x_center, y_center) for each voxel
        """
        # Get xyz coordinates
        points_xyz = points[:, :3]
        
        # Filter points by z range
        z_mask = (
            (points_xyz[:, 2] >= self.z_min) &
            (points_xyz[:, 2] < self.z_max)
        )
        
        points = points[z_mask]
        points_xyz = points_xyz[z_mask]
        
        if len(points) == 0:
            return (
                torch.zeros((0, self.max_points_per_voxel, points.shape[1]),
                           dtype=points.dtype, device=points.device),
                torch.zeros((0, 3), dtype=torch.int32, device=points.device),
                torch.zeros((0,), dtype=torch.int32, device=points.device),
                torch.zeros((0, 2), dtype=points.dtype, device=points.device),
            )
        
        # Convert to polar coordinates
        x, y = points_xyz[:, 0], points_xyz[:, 1]
        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)
        
        # Filter by radial and angular range
        r_mask = (r >= self.radial_edges[0]) & (r < self.radial_edges[-1])
        theta_mask = (theta >= self.angle_edges[0]) & (theta < self.angle_edges[-1])
        mask = r_mask & theta_mask
        
        points = points[mask]
        r = r[mask]
        theta = theta[mask]
        
        if len(points) == 0:
            return (
                torch.zeros((0, self.max_points_per_voxel, points.shape[1]),
                           dtype=points.dtype, device=points.device),
                torch.zeros((0, 3), dtype=torch.int32, device=points.device),
                torch.zeros((0,), dtype=torch.int32, device=points.device),
                torch.zeros((0, 2), dtype=points.dtype, device=points.device),
            )
        
        # Find bin indices
        r_idx = torch.searchsorted(self.radial_edges_tensor, r, right=False) - 1  # type: ignore
        theta_idx = torch.searchsorted(self.angle_edges_tensor, theta, right=False) - 1  # type: ignore
        
        # Clamp to valid range
        r_idx = torch.clamp(r_idx, 0, self.n_radial - 1)
        theta_idx = torch.clamp(theta_idx, 0, self.n_angular - 1)
        z_idx = torch.zeros_like(r_idx)
        
        # Stack coordinates
        voxel_coords = torch.stack([r_idx, theta_idx, z_idx], dim=1).int()
        
        # Convert 3D coords to 1D hash for grouping
        voxel_hash = (
            r_idx * self.n_angular * self.n_z +
            theta_idx * self.n_z +
            z_idx
        )
        
        # Sort by voxel hash
        sorted_indices = torch.argsort(voxel_hash)
        sorted_hash = voxel_hash[sorted_indices]
        sorted_coords = voxel_coords[sorted_indices]
        sorted_points = points[sorted_indices]
        
        # Find unique voxels
        unique_hash, inverse_indices = torch.unique(sorted_hash, return_inverse=True)
        num_unique = len(unique_hash)
        
        # Limit number of voxels
        if num_unique > self.max_voxels:
            num_unique = self.max_voxels
            unique_hash = unique_hash[:self.max_voxels]
        
        # Initialize output tensors
        voxels = torch.zeros(
            (num_unique, self.max_points_per_voxel, points.shape[1]),
            dtype=points.dtype,
            device=points.device
        )
        coords = torch.zeros((num_unique, 3), dtype=torch.int32, device=points.device)
        num_points = torch.zeros((num_unique,), dtype=torch.int32, device=points.device)
        voxel_centers = torch.zeros((num_unique, 2), dtype=points.dtype, device=points.device)
        
        # Fill voxels and compute polar centers
        for i in range(num_unique):
            voxel_mask = inverse_indices == i
            voxel_points = sorted_points[voxel_mask]
            n_points = min(len(voxel_points), self.max_points_per_voxel)
            
            voxels[i, :n_points] = voxel_points[:n_points]
            coords[i] = sorted_coords[voxel_mask][0]
            num_points[i] = n_points
            
            # Compute voxel center in Cartesian coordinates
            r_idx_val = coords[i, 0].item()
            theta_idx_val = coords[i, 1].item()
            
            # Get bin edges in polar coordinates
            r_low = self.radial_edges[r_idx_val]
            r_high = self.radial_edges[r_idx_val + 1]
            theta_low = self.angle_edges[theta_idx_val]
            theta_high = self.angle_edges[theta_idx_val + 1]
            
            # Compute polar bin center
            r_center = (r_low + r_high) / 2.0
            theta_center = (theta_low + theta_high) / 2.0
            
            # Convert to Cartesian coordinates
            x_center = r_center * np.cos(theta_center)
            y_center = r_center * np.sin(theta_center)
            
            voxel_centers[i, 0] = x_center
            voxel_centers[i, 1] = y_center
        
        return voxels, coords, num_points, voxel_centers


def _test_voxelization():
    """Test voxelization modules."""
    print("Testing DynamicVoxelization...")
    voxelizer_dynamic = DynamicVoxelization(
        voxel_size=(0.2, 0.2, 8),
        point_cloud_range=(0, -25.6, -3, 51.2, 25.6, 5),
    )
    
    # Generate random points
    points = torch.randn(10000, 4)  # [N, 4] (x, y, z, intensity)
    points[:, 0] = points[:, 0] * 20 + 25  # x: 5-45
    points[:, 1] = points[:, 1] * 20  # y: -20 to 20
    points[:, 2] = points[:, 2] * 2  # z: -2 to 2
    
    voxel_features, voxel_coords = voxelizer_dynamic(points)
    print(f"  Input points: {points.shape}")
    print(f"  Voxel features: {voxel_features.shape}")
    print(f"  Voxel coords: {voxel_coords.shape}")
    print(f"  Grid size: {voxelizer_dynamic.grid_size}")
    
    print("\nTesting HardVoxelization...")
    voxelizer_hard = HardVoxelization(
        voxel_size=(0.2, 0.2, 8),
        point_cloud_range=(0, -25.6, -3, 51.2, 25.6, 5),
        max_points_per_voxel=32,
        max_voxels=16000,
    )
    
    voxels, coords, num_points = voxelizer_hard(points)
    print(f"  Input points: {points.shape}")
    print(f"  Voxels: {voxels.shape}")
    print(f"  Coords: {coords.shape}")
    print(f"  Num points per voxel: {num_points.shape}")
    print(f"  Avg points per voxel: {num_points.float().mean().item():.2f}")
    print(f"  Max points per voxel: {num_points.max().item()}")
    
    print("\nTesting PolarVoxelization...")
    # Create a dummy polar grid config
    polar_config = {
        'radial_edges': [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0],
        'angle_edges': list(np.linspace(2.8, 3.5, 11)),  # ~40 degrees
    }
    
    voxelizer_polar = PolarVoxelization(
        polar_grid_config=polar_config,
        point_cloud_range_z=(-3, 5),
    )
    
    voxel_features, voxel_coords = voxelizer_polar(points)
    print(f"  Input points: {points.shape}")
    print(f"  Voxel features: {voxel_features.shape}")
    print(f"  Voxel coords: {voxel_coords.shape}")
    print(f"  Polar grid size: ({voxelizer_polar.n_radial}, {voxelizer_polar.n_angular}, {voxelizer_polar.n_z})")
    
    print("\nTesting HardPolarVoxelization...")
    voxelizer_hard_polar = HardPolarVoxelization(
        polar_grid_config=polar_config,
        point_cloud_range_z=(-3, 5),
        max_points_per_voxel=32,
        max_voxels=5000,
    )
    
    voxels, coords, num_points = voxelizer_hard_polar(points)
    print(f"  Input points: {points.shape}")
    print(f"  Voxels: {voxels.shape}")
    print(f"  Coords: {coords.shape}")
    print(f"  Num points per voxel: {num_points.shape}")
    print(f"  Avg points per voxel: {num_points.float().mean().item():.2f}")
    print(f"  Max points per voxel: {num_points.max().item()}")
    
    print("\n✓ All voxelization tests passed!")


if __name__ == '__main__':
    _test_voxelization()

