"""PointPillars encoder for efficient point cloud feature extraction.

Adapted from PointPillars (CVPR 2019) and BEVFusion implementation.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['PointPillarFeatureNet', 'PointPillarScatter', 'PolarPointPillarFeatureNet']


def get_paddings_indicator(actual_num: torch.Tensor, max_num: int) -> torch.Tensor:
    """Create boolean mask for padded tensors.
    
    Args:
        actual_num: [N] actual number of points in each pillar
        max_num: Maximum number of points per pillar
        
    Returns:
        mask: [N, max_num] boolean mask (True for valid points)
    """
    actual_num = actual_num.unsqueeze(1)  # [N, 1]
    max_num_range = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device
    ).unsqueeze(0)  # [1, max_num]
    
    mask = actual_num.int() > max_num_range  # [N, max_num]
    return mask


class PFNLayer(nn.Module):
    """Pillar Feature Net Layer.
    
    Processes features within each pillar using linear + norm + relu.
    Can be stacked to form deeper networks.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        last_layer: If True, only returns max pooled features
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        last_layer: bool = False,
    ):
        super().__init__()
        
        self.last_layer = last_layer
        
        # If not last layer, output half channels for concatenation
        if not last_layer:
            out_channels = out_channels // 2
        
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: [N_pillars, N_points, C_in] pillar features
            
        Returns:
            [N_pillars, N_points, C_out] processed features (if not last layer)
            [N_pillars, 1, C_out] max pooled features (if last layer)
        """
        # Linear
        x = self.linear(inputs)  # [N, P, C]
        
        # BatchNorm expects [N, C, P]
        x = x.permute(0, 2, 1).contiguous()  # [N, C, P]
        x = self.norm(x)
        x = x.permute(0, 2, 1).contiguous()  # [N, P, C]
        
        # ReLU
        x = F.relu(x)
        
        # Max pooling across points
        x_max = torch.max(x, dim=1, keepdim=True)[0]  # [N, 1, C]
        
        if self.last_layer:
            return x_max
        else:
            # Concatenate max with per-point features
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)  # [N, P, C]
            x_concat = torch.cat([x, x_repeat], dim=2)  # [N, P, 2C]
            return x_concat


class PointPillarFeatureNet(nn.Module):
    """PointPillars feature encoder.
    
    Encodes point cloud features within each pillar using learned feature
    augmentation and max pooling.
    
    Args:
        in_channels: Number of input point features (e.g., 4 for x,y,z,intensity)
        feat_channels: Tuple of feature dimensions for each PFN layer
        voxel_size: Size of each voxel/pillar (x, y, z)
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        with_distance: Whether to add distance from origin as feature
        
    Example:
        >>> encoder = PointPillarFeatureNet(
        ...     in_channels=4,
        ...     feat_channels=(64,),
        ...     voxel_size=(0.2, 0.2, 8),
        ...     point_cloud_range=(0, -25.6, -3, 51.2, 25.6, 5),
        ... )
        >>> voxels = torch.randn(10000, 32, 4)  # [N_pillars, N_points, 4]
        >>> coords = torch.randint(0, 256, (10000, 3))  # [N_pillars, 3]
        >>> num_points = torch.randint(1, 33, (10000,))  # [N_pillars]
        >>> features = encoder(voxels, num_points, coords)  # [N_pillars, 64]
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        feat_channels: Tuple[int, ...] = (64,),
        voxel_size: Tuple[float, float, float] = (0.2, 0.2, 8),
        point_cloud_range: Tuple[float, float, float, float, float, float] = (0, -25.6, -3, 51.2, 25.6, 5),
        with_distance: bool = False,
    ):
        super().__init__()
        
        assert len(feat_channels) > 0
        
        self.in_channels = in_channels
        self.with_distance = with_distance
        
        # Feature augmentation adds:
        # - xyz offset from cluster center (3)
        # - xy offset from pillar center (2)
        # - optionally: distance from origin (1)
        aug_channels = in_channels + 3 + 2
        if with_distance:
            aug_channels += 1
        
        # Voxel parameters
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        
        # Build PFN layers
        pfn_layers = []
        for i, out_channels in enumerate(feat_channels):
            in_ch = aug_channels if i == 0 else feat_channels[i - 1]
            is_last = i == len(feat_channels) - 1
            pfn_layers.append(PFNLayer(in_ch, out_channels, last_layer=is_last))
        
        self.pfn_layers = nn.ModuleList(pfn_layers)
    
    def forward(
        self,
        voxels: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            voxels: [N, max_points, C] voxel features
            num_points_per_voxel: [N] number of points in each voxel
            coords: [N, 3] voxel coordinates (x_idx, y_idx, z_idx)
            
        Returns:
            features: [N, C_out] encoded features for each pillar
        """
        device = voxels.device
        dtype = voxels.dtype
        
        # 1. Compute cluster center (mean of points in pillar)
        points_mean = (
            voxels[:, :, :3].sum(dim=1, keepdim=True) / 
            num_points_per_voxel.type_as(voxels).view(-1, 1, 1)
        )  # [N, 1, 3]
        
        # Offset from cluster center
        f_cluster = voxels[:, :, :3] - points_mean  # [N, P, 3]
        
        # 2. Compute pillar center
        f_center = torch.zeros_like(voxels[:, :, :2])  # [N, P, 2]
        f_center[:, :, 0] = voxels[:, :, 0] - (
            coords[:, 0].to(dtype).unsqueeze(1) * self.vx + self.x_offset
        )
        f_center[:, :, 1] = voxels[:, :, 1] - (
            coords[:, 1].to(dtype).unsqueeze(1) * self.vy + self.y_offset
        )
        
        # 3. Combine features
        features_list = [voxels, f_cluster, f_center]
        
        if self.with_distance:
            # Distance from origin
            points_dist = torch.norm(voxels[:, :, :3], 2, 2, keepdim=True)
            features_list.append(points_dist)
        
        features = torch.cat(features_list, dim=-1)  # [N, P, C_aug]
        
        # 4. Apply mask for padded points
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points_per_voxel, voxel_count)
        mask = mask.unsqueeze(-1).type_as(features)  # [N, P, 1]
        features = features * mask
        
        # 5. Forward through PFN layers
        for pfn in self.pfn_layers:
            features = pfn(features)
        
        # Output is [N, 1, C], squeeze to [N, C]
        return features.squeeze(1)


class PointPillarScatter(nn.Module):
    """Scatter pillar features to create dense BEV pseudo-image.
    
    Takes encoded pillar features and their coordinates, then scatters them
    onto a dense 2D grid to create a BEV representation.
    
    Args:
        in_channels: Number of input feature channels
        output_shape: (H, W) shape of output BEV grid
        
    Example:
        >>> scatter = PointPillarScatter(
        ...     in_channels=64,
        ...     output_shape=(256, 256),
        ... )
        >>> features = torch.randn(10000, 64)
        >>> coords = torch.randint(0, 256, (10000, 3))
        >>> batch_size = 4
        >>> bev_features = scatter(features, coords, batch_size)  # [4, 64, 256, 256]
    """
    
    def __init__(
        self,
        in_channels: int,
        output_shape: Tuple[int, int],
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.nx, self.ny = output_shape
    
    def forward(
        self,
        voxel_features: torch.Tensor,
        coords: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Scatter features to 2D BEV grid.
        
        PointPillars collapses the z-dimension, so only x and y indices are used
        to create the 2D BEV representation. The z coordinate (if present) is ignored.
        
        Args:
            voxel_features: [N, C] encoded voxel features
            coords: [N, 3] or [N, 4] voxel coordinates
                If [N, 4]: (batch_idx, x_idx, y_idx, z_idx)
                           Only batch_idx, x_idx, y_idx are used; z_idx is ignored
                If [N, 3]: (x_idx, y_idx, z_idx) - assuming single batch
                           Only x_idx, y_idx are used; z_idx is ignored
                
                For Cartesian grids: x_idx, y_idx are spatial grid indices
                For polar grids: x_idx=r_idx, y_idx=theta_idx
            batch_size: Number of batches
            
        Returns:
            batch_canvas: [B, C, H, W] dense 2D BEV features
        """
        # Handle coordinate format
        if coords.shape[1] == 4:
            # Format: (batch_idx, x_idx, y_idx, z_idx)
            # For Cartesian grids: x, y are spatial indices
            # For polar grids: x=r_idx, y=theta_idx
            # z_idx is always ignored (PointPillars collapses z dimension)
            batch_idx = coords[:, 0].long()
            x_idx = coords[:, 1].long()
            y_idx = coords[:, 2].long()
            # z_idx = coords[:, 3] is ignored
        else:
            # Format: (x_idx, y_idx, z_idx) assuming single batch
            # z_idx is ignored for 2D BEV
            batch_idx = torch.zeros(
                coords.shape[0], dtype=torch.long, device=coords.device
            )
            x_idx = coords[:, 0].long()
            y_idx = coords[:, 1].long()
            # z_idx = coords[:, 2] is ignored
        
        # Initialize output
        batch_canvas = torch.zeros(
            batch_size,
            self.in_channels,
            self.nx,
            self.ny,
            dtype=voxel_features.dtype,
            device=voxel_features.device,
        )
        
        # Scatter features using advanced indexing
        
        # Create indices for all dimensions
        # batch_idx: [N] -> which batch each feature belongs to
        # x_idx, y_idx: [N] -> spatial location in BEV grid
        # We need to expand for the channel dimension
        
        N = voxel_features.shape[0]
        C = self.in_channels
        
        # Create indices for scatter
        # For each of N features, we need to scatter all C channels
        batch_indices = batch_idx.unsqueeze(1).expand(N, C)  # [N, C]
        channel_indices = torch.arange(C, device=voxel_features.device).unsqueeze(0).expand(N, C)  # [N, C]
        x_indices = x_idx.unsqueeze(1).expand(N, C)  # [N, C]
        y_indices = y_idx.unsqueeze(1).expand(N, C)  # [N, C]
        
        # Scatter all features at once, use advanced indexing to scatter the features to the correct location in the batch_canvas
        # canvas[batch_indices[i,j], channel_indices[i,j], x_indices[i,j], y_indices[i,j]] = voxel_features[i,j]
        batch_canvas[batch_indices, channel_indices, x_indices, y_indices] = voxel_features
        
        return batch_canvas


class PolarPointPillarFeatureNet(nn.Module):
    """PointPillar feature encoder adapted for polar voxelization.
    
    This variant is designed to work with polar grid outputs from HardPolarVoxelization,
    using the actual voxel centers in Cartesian coordinates (x, y) to compute pillar 
    offsets properly. The centers are computed from polar bin centers and converted to 
    Cartesian space.
    
    Args:
        in_channels: Number of input channels (e.g., 4 for x,y,z,intensity)
        feat_channels: List of output channels for each PFN layer
        with_distance: Whether to include distance to origin as a feature
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        feat_channels: Tuple[int, ...] = (64,),
        with_distance: bool = False,
    ):
        super().__init__()
        assert len(feat_channels) > 0
        self.in_channels = in_channels
        self.with_distance = with_distance
        
        # Build PFN layers
        in_filters = [in_channels]
        pfn_layers = []
        n_feat_channels = len(feat_channels)
        for i, feat_channel in enumerate(feat_channels):
            # Input features for polar grid:
            # - Original features: in_channels
            # - Cluster offset: 3 (x,y,z offset from cluster center)
            # - Pillar offset: 2 (x,y offset from pillar center in Cartesian)
            # - Distance (optional): 1
            if i == 0:
                num_input_features = in_channels + 3 + 2  # Base features
                if with_distance:
                    num_input_features += 1
            else:
                num_input_features = feat_channels[i - 1]  # last layer's output features
            
            pfn_layers.append(
                PFNLayer(
                    num_input_features,
                    feat_channel,
                    last_layer=(i >= n_feat_channels - 1)
                )
            )
            in_filters.append(feat_channel)
        
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.num_pfn = len(self.pfn_layers)
    
    def forward(
        self,
        voxels: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
        voxel_centers: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for polar voxelized point cloud.
        
        Args:
            voxels: [N, max_points_per_voxel, C] voxelized point cloud
            num_points_per_voxel: [N] number of points per voxel
            voxel_centers: [N, 2] Cartesian centers (x_center, y_center) for each voxel
            
        Returns:
            features: [N, feat_channels[-1]] encoded pillar features
        """
        # 1. Compute cluster center (mean of points in pillar)
        points_mean = (voxels[:, :, :3].sum(dim=1, keepdim=True) /
                       num_points_per_voxel.type_as(voxels).view(-1, 1, 1))
        f_cluster = voxels[:, :, :3] - points_mean
        
        # 2. Compute offset from pillar center (already in Cartesian)
        # voxel_centers: [N, 2] with (x_center, y_center)
        x_center = voxel_centers[:, 0]  # [N]
        y_center = voxel_centers[:, 1]  # [N]
        
        # Compute offset from pillar center
        f_center = torch.zeros_like(voxels[:, :, :2])  # [N, max_points, 2]
        f_center[:, :, 0] = voxels[:, :, 0] - x_center.unsqueeze(1)  # x offset
        f_center[:, :, 1] = voxels[:, :, 1] - y_center.unsqueeze(1)  # y offset
        
        # 3. Combine features
        features_list = [voxels, f_cluster, f_center]
        if self.with_distance:
            points_dist = torch.norm(voxels[:, :, :3], 2, 2, keepdim=True)
            features_list.append(points_dist)
        features = torch.cat(features_list, dim=-1)
        
        # 4. Apply mask for padded points
        mask = get_paddings_indicator(num_points_per_voxel, features.shape[1])
        features *= mask.unsqueeze(-1).type_as(features)
        
        # 5. Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)
        
        return features.squeeze()

