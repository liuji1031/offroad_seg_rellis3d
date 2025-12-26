import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn

from offroad_det_seg_rellis.util import get_logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = get_logger(__name__)
try:
    from .bev_pool import bev_pool

    HAS_BEV_POOL = True
    logger.info("bev_pool imported successfully")
except ImportError as e:
    logger.warning(f"bev_pool not imported: {e}")
    HAS_BEV_POOL = False
    logger.warning(
        "bev_pool CUDA extension not available. Using fallback PyTorch implementation."
    )


class DepthNet(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, hidden_channels: int = 64
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.depthnet(x)


class PolarBEVTransform(nn.Module):
    """Transform camera features to BEV using non-uniform polar grid.

    This implements the Lift-Splat-Shoot paradigm:
    1. LIFT: Create 3D frustum from 2D image features with depth prediction
    2. SPLAT: Transform 3D points to polar BEV coordinates and pool into grid
    3. SHOOT: Output BEV features ready for downstream tasks

    The key difference from standard BEVFusion is handling non-uniform polar grids
    where radial bin size grows with distance from sensor origin.

    Args:
        in_channels: Number of input feature channels from camera backbone
        out_channels: Number of output BEV feature channels
        image_size: (H, W) of input images
        feature_size: (H, W) of downsampled feature maps
        polar_grid_config: Path to JSON config or dict with polar grid parameters
        dbound: (min_depth, max_depth, depth_resolution) in meters
        downsample: Downsampling factor for final BEV features
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        polar_grid_config: str | Dict,
        dbound: Tuple[float, float, float] = (1.0, 60.0, 0.5),
        downsample: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.dbound = dbound

        if isinstance(polar_grid_config, (str, Path)):
            with open(polar_grid_config, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = polar_grid_config

        self.angle_edges = torch.tensor(
            config["angle_edges"], dtype=torch.float32
        )
        self.radial_edges = torch.tensor(
            config["radial_edges"], dtype=torch.float32
        )

        # n_theta: Number of angular bins (uniform spacing)
        # n_r: Number of radial bins (NON-uniform, growing with distance)
        self.n_theta = len(self.angle_edges) - 1  # Number of angular bins
        self.n_r = len(self.radial_edges) - 1  # Number of radial bins

        # Store angle range for normalization
        self.angle_min = self.angle_edges[0].item()
        self.angle_max = self.angle_edges[-1].item()
        self.angle_range = self.angle_max - self.angle_min

        # Store radial range
        self.r_min = self.radial_edges[0].item()
        self.r_max = self.radial_edges[-1].item()

        # Register edges as buffers (non-trainable parameters that move with model)
        self.register_buffer("angle_edges_buf", self.angle_edges)
        self.register_buffer("radial_edges_buf", self.radial_edges)

        # The frustum defines all possible 3D points that could be seen by camera
        # For each pixel in feature map, we sample multiple depth hypotheses
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]  # Number of depth bins

        # This network predicts:
        # - Depth distribution (D channels, softmax over depth bins)
        # - BEV features (C channels)
        self.depthnet = DepthNet(in_channels, self.D + out_channels).to(DEVICE)

        # Optional downsampling of final BEV features
        if downsample > 1:
            assert downsample == 2, (
                f"Only downsample=2 supported, got {downsample}"
            )
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

        self.fp16_enabled = False

    def create_frustum(self) -> torch.Tensor:
        """Create camera frustum with depth sampling.

        The frustum is a 4D tensor [D, fH, fW, 3] where:
        - D: number of depth bins
        - fH, fW: feature map height and width
        - 3: (x, y, depth) coordinates in image space

        For each pixel in the downsampled feature map, we create D depth
        hypotheses. These will be lifted to 3D and transformed to BEV.

        Returns:
            frustum: [D, fH, fW, 3] tensor of camera ray samples
        """
        iH, iW = self.image_size  # Original image size
        fH, fW = self.feature_size  # Downsampled feature size

        ds = torch.arange(*self.dbound, dtype=torch.float32)
        ds = ds.view(-1, 1, 1).expand(-1, fH, fW)  # [D, fH, fW]
        D = ds.shape[0]

        xs = torch.linspace(0, iW - 1, fW, dtype=torch.float32)
        xs = xs.view(1, 1, fW).expand(D, fH, fW)  # [D, fH, fW]

        ys = torch.linspace(0, iH - 1, fH, dtype=torch.float32)
        ys = ys.view(1, fH, 1).expand(D, fH, fW)  # [D, fH, fW]

        # Stack into frustum: [D, fH, fW, 3] with (x, y, depth)
        frustum = torch.stack((xs, ys, ds), dim=-1)

        return nn.Parameter(frustum, requires_grad=False).to(DEVICE)

    def build_inv_intrins(self, intrins: torch.Tensor) -> torch.Tensor:
        """Build inverse intrinsic matrix.

        Args:
            intrins: [B, N, 3, 3] intrinsic matrices

        Returns:
            inv_intrins: [B, N, 3, 3] inverse intrinsic matrices
        """
        # instead of using inverse we can construct the inverse directly
        inv_intrins = torch.zeros_like(intrins)
        inv_intrins[..., 0, 0] = 1.0 / intrins[..., 0, 0]
        inv_intrins[..., 1, 1] = 1.0 / intrins[..., 1, 1]
        inv_intrins[..., 0, 2] = -intrins[..., 0, 2] / intrins[..., 0, 0]
        inv_intrins[..., 1, 2] = -intrins[..., 1, 2] / intrins[..., 1, 1]
        inv_intrins[..., 2, 2] = 1.0
        return inv_intrins

    def get_geometry(
        self,
        camera2lidar_rots: torch.Tensor,
        camera2lidar_trans: torch.Tensor,
        intrins: torch.Tensor,
        post_rots: Optional[torch.Tensor] = None,
        post_trans: Optional[torch.Tensor] = None,
        extra_rots: Optional[torch.Tensor] = None,
        extra_trans: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Transform frustum from image space to LiDAR coordinate frame.

        This is the core geometric transformation that "lifts" 2D image points
        to 3D space and transforms them to the LiDAR/vehicle frame.

        Transformation pipeline:
        1. Undo image augmentation (crop, resize, flip, etc.)
        2. Unproject 2D pixels to 3D camera rays using intrinsics
        3. Transform from camera frame to LiDAR frame
        4. Apply optional LiDAR-space augmentation

        Args:
            camera2lidar_rots: [B, N, 3, 3] rotation matrices, i.e., camera rotation expressed in the LiDAR frame
            camera2lidar_trans: [B, N, 3] translation vectors, i.e., camera translation expressed in the LiDAR frame
            intrins: [B, N, 3, 3] camera intrinsic matrices
            post_rots: [B, N, 3, 3] image augmentation rotations
            post_trans: [B, N, 3] image augmentation translations
            extra_rots: [B, 3, 3] additional LiDAR space rotations (optional)
            extra_trans: [B, 3] additional LiDAR space translations (optional)

        Returns:
            points: [B, N, D, fH, fW, 3] - 3D coordinates (x, y, z) in LiDAR frame
                    B: batch size
                    N: number of cameras
                    D: number of depth bins
                    fH, fW: feature map spatial dimensions
                    3: (x, y, z) coordinates
        """
        B, N, _ = camera2lidar_trans.shape

        if post_rots is not None and post_trans is not None:
            points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
            # Apply inverse rotation: R^(-1) @ (points - translation)
            points = (
                torch.inverse(post_rots)
                .view(B, N, 1, 1, 1, 3, 3)
                .matmul(points.unsqueeze(-1))
            )  # [B, N, D, fH, fW, 3, 1]
        else:
            points = self.frustum.repeat(B, N, 1, 1, 1, 1)

        u = points[:, :, :, :, :, 0]
        v = points[:, :, :, :, :, 1]
        d = points[:, :, :, :, :, 2]
        cx = intrins[..., 0, 2].view(B, N, 1, 1, 1)
        cy = intrins[..., 1, 2].view(B, N, 1, 1, 1)
        fx = intrins[..., 0, 0].view(B, N, 1, 1, 1)
        fy = intrins[..., 1, 1].view(B, N, 1, 1, 1)
        points = torch.stack(
            (
                (u - cx) * d / fx,
                (v - cy) * d / fy,
                d,
            ),
            dim=-1,
        )

        points = (
            camera2lidar_rots.view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
            .squeeze(-1)
        )
        # Add translation
        points = points + camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        if extra_rots is not None:
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )

        if extra_trans is not None:
            points = points + extra_trans.view(B, 1, 1, 1, 1, 3).repeat(
                1, N, 1, 1, 1, 1
            )

        return points  # [B, N, D, fH, fW, 3]

    def get_cam_feats(self, x: torch.Tensor) -> torch.Tensor:
        """Predict depth and extract BEV features from camera images.

        This implements the "Lift" part of Lift-Splat-Shoot:
        - Predict depth probability distribution for each pixel
        - Extract BEV-ready features
        - Combine via outer product to get depth-weighted features

        Args:
            x: [B, N, C, fH, fW] camera features from backbone
               B: batch size, N: num cameras, C: channels

        Returns:
            features: [B, N, D, fH, fW, C_out] depth-distributed features
                      Ready to be pooled into BEV grid
        """
        B, N, C, fH, fW = x.shape

        # Merge batch and camera dimensions for processing
        x = x.view(B * N, C, fH, fW)

        # Predict depth + features using 1x1 convolution
        x = self.depthnet(x)  # [B*N, D+C_out, fH, fW]

        # Split into depth and features
        depth = x[:, : self.D]  # [B*N, D, fH, fW]
        features = x[
            :, self.D : (self.D + self.out_channels)
        ]  # [B*N, C_out, fH, fW]

        # Apply softmax to get depth probability distribution
        depth = depth.softmax(dim=1)  # [B*N, D, fH, fW]

        # Reshape for outer product
        depth = depth.unsqueeze(1)  # [B*N, 1, D, fH, fW]
        features = features.unsqueeze(2)  # [B*N, C_out, 1, fH, fW]

        # Outer product via broadcasting
        x = depth * features  # [B*N, C_out, D, fH, fW]

        # Reshape back to batch format and permute to [B, N, D, fH, fW, C]
        x = x.view(B, N, self.out_channels, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)  # [B, N, D, fH, fW, C_out]

        return x

    def polar_bev_pool(
        self, geom_feats: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Pool 3D features into non-uniform polar BEV grid.

        This is the "Splat" part of Lift-Splat-Shoot, adapted for polar grids.

        Key challenge: The radial bins are NOT uniformly spaced!
        - Close to sensor: small bins (e.g., 0.25m)
        - Far from sensor: large bins (e.g., 1.5m)

        We cannot use simple division like: r_idx = r / resolution
        Instead, we use torch.searchsorted to find which bin each point belongs to.

        Args:
            geom_feats: [B, N, D, H, W, 3] - (x, y, z) coordinates in LiDAR frame
            x: [B, N, D, H, W, C] - features to pool

        Returns:
            bev_features: [B, C, n_r, n_theta] - pooled BEV features
        """
        B, N, D, H, W, C = x.shape

        x_coord = geom_feats[..., 0]  # [B, N, D, H, W]
        y_coord = geom_feats[..., 1]  # [B, N, D, H, W]
        # z_coord = geom_feats[..., 2]  # [B, N, D, H, W]

        # Compute polar coordinates
        r = torch.sqrt(x_coord**2 + y_coord**2)  # Radial distance
        theta = torch.atan2(y_coord, x_coord)  # Angle in radians [-π, π]
        theta[theta < 0] += (
            2 * torch.pi
        )  # wrap around to [0, 2π], for our dataset the points are centered around pi

        r_flat = r.reshape(-1)  # Flatten for searchsorted
        r_indices = (
            torch.searchsorted(
                self.radial_edges_buf.contiguous(),
                r_flat.contiguous(),
                right=False,
            )
            - 1
        )  # [B*N*D*H*W]

        # Points beyond r_max will be filtered out later
        r_indices = torch.clamp(r_indices, 0, self.n_r - 1)
        r_indices = r_indices.reshape(B, N, D, H, W)  # Reshape back

        theta_normalized = theta  # Already in radians

        # Simple binning for uniform angular spacing
        theta_indices = torch.floor(
            (theta_normalized - self.angle_min)
            / ((self.angle_max - self.angle_min) / self.n_theta)
        ).long()

        # Clamp to valid range
        theta_indices = torch.clamp(theta_indices, 0, self.n_theta - 1)

        z_indices = torch.zeros_like(
            r_indices, dtype=torch.long
        )  # All in bin 0
        n_z = 1

        Nprime = B * N * D * H * W
        x_flat = x.reshape(Nprime, C)  # [N_total, C]
        r_flat = r_indices.reshape(Nprime)
        theta_flat = theta_indices.reshape(Nprime)
        z_flat = z_indices.reshape(Nprime)

        polar_indices = torch.stack([r_flat, theta_flat, z_flat], dim=1)

        # Add batch index as 4th dimension
        batch_indices = torch.cat(
            [
                torch.full((Nprime // B,), i, device=x.device, dtype=torch.long)
                for i in range(B)
            ]
        )  # [N_total]

        polar_indices = torch.cat(
            [polar_indices, batch_indices.unsqueeze(1)], dim=1
        )  # [N_total, 4]

        r_all = r.reshape(Nprime)
        in_r_bounds = (r_all >= self.r_min) & (r_all <= self.r_max)

        # Check angular bounds
        theta_all = theta.reshape(Nprime)
        in_theta_bounds = (theta_all >= self.angle_min) & (
            theta_all <= self.angle_max
        )

        # Check grid index bounds (safety check)
        in_grid = (
            (polar_indices[:, 0] >= 0)
            & (polar_indices[:, 0] < self.n_r)
            & (polar_indices[:, 1] >= 0)
            & (polar_indices[:, 1] < self.n_theta)
            & (polar_indices[:, 2] >= 0)
            & (polar_indices[:, 2] < n_z)
        )

        # Combine all filters
        kept = in_r_bounds & in_theta_bounds & in_grid

        x_flat = x_flat[kept]
        polar_indices = polar_indices[kept]

        if HAS_BEV_POOL:
            x_pooled = bev_pool(
                x_flat,  # [N_kept, C] features
                polar_indices,  # [N_kept, 4] (r, theta, z, batch) indices
                B,  # batch size
                n_z,  # depth dimension (will collapse)
                self.n_r,  # "height" = radial bins
                self.n_theta,  # "width" = angular bins
            )  # Output: [B, C, n_z, n_r, n_theta]
            x_pooled = x_pooled.squeeze(2)  # [B, C, n_r, n_theta]

        else:
            # Fallback: Pure PyTorch implementation (slower but works without CUDA)
            x_pooled = self._pool_fallback(x_flat, polar_indices, B, n_z, C)

        return x_pooled

    def _pool_fallback(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        B: int,
        n_z: int,
        C: int,
    ) -> torch.Tensor:
        """Fallback pooling implementation using PyTorch scatter_add.

        Used when CUDA bev_pool is not available.
        """
        # Create output grid
        out = torch.zeros(
            (B, n_z, self.n_r, self.n_theta, C),
            device=features.device,
            dtype=features.dtype,
        )

        # Compute linear indices for scatter
        r_idx = indices[:, 0]
        theta_idx = indices[:, 1]
        z_idx = indices[:, 2]
        b_idx = indices[:, 3]

        # Convert to linear index
        linear_idx = (
            b_idx * (n_z * self.n_r * self.n_theta)
            + z_idx * (self.n_r * self.n_theta)
            + r_idx * self.n_theta
            + theta_idx
        )

        # Scatter add features
        out_flat = out.reshape(-1, C)
        out_flat.index_add_(0, linear_idx, features)
        out = out_flat.reshape(B, n_z, self.n_r, self.n_theta, C)

        # Permute to [B, C, n_z, n_r, n_theta]
        out = out.permute(0, 4, 1, 2, 3)

        # Collapse z
        out = out.squeeze(2)  # [B, C, n_r, n_theta]

        return out

    def forward(
        self,
        img: torch.Tensor,
        camera2lidar_rots: torch.Tensor,
        camera2lidar_trans: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        img_aug_matrix: Optional[torch.Tensor] = None,
        lidar_aug_matrix: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass: image features -> polar BEV features.

        Complete pipeline:
        1. Extract camera features and predict depth (Lift)
        2. Transform to 3D LiDAR coordinates (Lift)
        3. Convert to polar coordinates and pool into grid (Splat)
        4. Return BEV features ready for downstream tasks (Shoot)

        Args:
            img: [B, N, C, H, W] input images
            camera2lidar_rots: [B, N, 3, 3] camera to LiDAR rotations
            camera2lidar_trans: [B, N, 3] camera to LiDAR translations
            camera_intrinsics: [B, N, 3, 3] camera intrinsic matrices
            img_aug_matrix: [B, N, 4, 4] image augmentation matrices
            lidar_aug_matrix: [B, 4, 4] LiDAR augmentation matrix (optional)

        Returns:
            bev_feats: [B, C, n_r, n_theta] polar BEV features
        """
        # Extract transformation matrices
        intrins = camera_intrinsics[..., :3, :3]
        if img_aug_matrix is not None:
            post_rots = img_aug_matrix[..., :3, :3]
            post_trans = img_aug_matrix[..., :3, 3]
        else:
            post_rots = None
            post_trans = None

        extra_rots = None
        extra_trans = None
        if lidar_aug_matrix is not None:
            extra_rots = lidar_aug_matrix[..., :3, :3]
            extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots,
            extra_trans,
        )  # [B, N, D, fH, fW, 3]

        x = self.get_cam_feats(img)  # [B, N, D, fH, fW, C]

        bev_feats = self.polar_bev_pool(geom, x)  # [B, C, n_r, n_theta]
        bev_feats = self.downsample(bev_feats)

        return bev_feats


class PolarDepthLSSTransform(PolarBEVTransform):
    """Polar BEV Transform with explicit depth supervision.

    This variant adds depth supervision by projecting LiDAR points to image
    and using them as ground truth for depth prediction.

    This typically improves performance when LiDAR data is available during training.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        polar_grid_config: str | Dict,
        dbound: Tuple[float, float, float] = (1.0, 60.0, 0.5),
        downsample: int = 1,
        depth_channels: int = 64,  # Number of channels for depth encoding
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            polar_grid_config=polar_grid_config,
            dbound=dbound,
            downsample=downsample,
        )

        # Replace simple depthnet with depth-conditioned version
        # Process explicit depth input (from LiDAR projection)
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, depth_channels, 5, stride=2, padding=2),
            nn.BatchNorm2d(depth_channels),
            nn.ReLU(True),
        )

        # Depth prediction network now takes image features + depth encoding
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + depth_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + out_channels, 1),
        )

    def get_cam_feats(
        self, x: torch.Tensor, depth_map: torch.Tensor
    ) -> torch.Tensor:
        """Extract features with explicit depth input.

        Args:
            x: [B, N, C, fH, fW] camera features
            depth_map: [B, N, 1, H, W] depth map from LiDAR projection

        Returns:
            features: [B, N, D, fH, fW, C_out] depth-distributed features
        """
        B, N, C, fH, fW = x.shape

        # Merge batch and camera dimensions
        x = x.view(B * N, C, fH, fW)
        depth_map = depth_map.view(B * N, 1, *self.image_size)

        # Encode depth information
        depth_encoded = self.dtransform(depth_map)  # [B*N, 64, fH, fW]

        # Concatenate with image features
        x = torch.cat([depth_encoded, x], dim=1)

        # Predict depth distribution and features
        x = self.depthnet(x)  # [B*N, D+C_out, fH, fW]

        # Split and compute outer product (same as parent class)
        depth = x[:, : self.D].softmax(dim=1)
        features = x[:, self.D : (self.D + self.out_channels)]

        x = depth.unsqueeze(1) * features.unsqueeze(2)

        x = x.view(B, N, self.out_channels, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def forward(
        self,
        img: torch.Tensor,
        depth_map: torch.Tensor,
        camera2lidar_rots: torch.Tensor,
        camera2lidar_trans: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        img_aug_matrix: torch.Tensor,
        lidar_aug_matrix: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward with depth supervision."""
        # Extract transformation matrices
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]

        extra_rots = None
        extra_trans = None
        if lidar_aug_matrix is not None:
            extra_rots = lidar_aug_matrix[..., :3, :3]
            extra_trans = lidar_aug_matrix[..., :3, 3]

        # Get 3D geometry
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots,
            extra_trans,
        )

        # Extract features with depth input
        x = self.get_cam_feats(img, depth_map)

        # Pool to polar BEV
        bev_feats = self.polar_bev_pool(geom, x)

        # Downsample
        bev_feats = self.downsample(bev_feats)

        return bev_feats
