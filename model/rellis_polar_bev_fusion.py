"""Complete BEVFusion model for RELLIS-3D dataset.

Integrates all components:
- Point cloud encoding (PointPillars with polar voxelization)
- Image encoding (ResNet + FPN)
- BEV transformation (Lift-Splat-Shoot)
- Multi-modal fusion (ConvFuser or AddFuser)
- Segmentation head (Polar ASPP)
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from offroad_det_seg_rellis.point_cloud_encoding import (
    create_polar_pointpillars_encoder,
)
from offroad_det_seg_rellis.image_encoding import (
    build_image_encoder_from_config,
)
from offroad_det_seg_rellis.bev_transform import PolarBEVTransform
from offroad_det_seg_rellis.fusion import (
    build_fuser_from_config,
)
from offroad_det_seg_rellis.segmentation_head import (
    build_segmentation_head_from_config,
)
from offroad_det_seg_rellis.loss import SegmentationLoss


__all__ = ["RellisPolarBevFusion"]


class RellisPolarBevFusion(nn.Module):
    """BEVFusion model for RELLIS-3D off-road terrain segmentation."""

    def __init__(
        self,
        point_cloud_encoder_config: Dict,
        image_encoder_config: Dict,
        bev_transform_config: Dict,
        fusion_config: Dict,
        segmentation_head_config: Dict,
        loss_config: Dict,
        use_camera: bool = True,
        use_lidar: bool = True,
    ):
        super().__init__()
        self.use_camera = use_camera
        self.use_lidar = use_lidar

        assert use_camera or use_lidar, "Must use at least one modality"

        # point cloud encoder
        self.lidar_encoder = create_polar_pointpillars_encoder(
            **point_cloud_encoder_config
        )
        self.lidar_channels = point_cloud_encoder_config["feat_channels"][
            -1
        ]  # final output channels

        # image encoder (Camera -> Image Features)
        if use_camera:
            self.image_encoder = build_image_encoder_from_config(
                image_encoder_config
            )
            self.camera_channels = self.image_encoder.get_output_channels()

        # bev transform (Image Features -> BEV)
        self.bev_transform = PolarBEVTransform(
            **bev_transform_config,
        )

        # camera lidar fusion
        if use_camera and use_lidar:
            self.fuser = build_fuser_from_config(fusion_config)
            self.fusion_channels = self.fuser.out_channels
        else:
            # Single modality - use identity "fusion"
            single_ch = (
                self.camera_channels if use_camera else self.lidar_channels
            )
            self.fuser = nn.Identity()
            self.fusion_channels = single_ch

        # segmentation head
        self.seg_head = build_segmentation_head_from_config(
            segmentation_head_config
        )

        # loss
        self.loss = SegmentationLoss(**loss_config)

    def forward(
        self,
        points: Optional[List[torch.Tensor]] = None,
        images: Optional[torch.Tensor] = None,
        camera_params: Optional[Dict[str, torch.Tensor]] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor|None]:
        """Forward pass.

        Args:
            points: List of point clouds [N_i, 4] (x, y, z, intensity)
            images: Camera images [B, N_cam, 3, H, W] (optional)
            camera_params: Camera parameters dict (optional, required if using camera)

        Returns:
            Training mode: Dict of losses
            Eval mode: Predictions [B, num_classes, H, W] (probabilities)
        """
        bev_features = []

        # encode LiDAR point cloud to BEV
        if self.use_lidar:
            assert points is not None, "Points required when use_lidar=True"
            lidar_bev = self.lidar_encoder(points)  # [B, lidar_ch, H, W]
            bev_features.append(lidar_bev)

        # 2. Encode camera images → BEV
        if self.use_camera:
            assert images is not None, "Images required when use_camera=True"

            # Extract image features
            image_features = self.image_encoder(
                images
            )  # [B, feature_channels, H', W']
            assert camera_params is not None, "camera_params required"
            assert "camera2lidar_rots" in camera_params, "camera2lidar_rots required"
            assert "camera2lidar_trans" in camera_params, "camera2lidar_trans required"
            assert "camera_intrinsics" in camera_params, "camera_intrinsics required"
            
            camera_bev = self.bev_transform(image_features, **camera_params)

            bev_features.append(camera_bev)

        # fuse modalities
        if isinstance(self.fuser, nn.Identity):
            fused_bev = bev_features[0]  # only one modality
        else:
            assert len(bev_features) > 1, "more than one modality required"
            fused_bev = self.fuser(bev_features)  # [B, fused_ch, H, W]
        
        # segmentation
        output = self.seg_head(fused_bev, return_logits=True)

        # compute loss
        if targets is not None:
            loss = self.loss(output, targets)
        else:
            loss = None
        return {"predictions": output, "loss": loss}

    def get_n_params(self) -> Dict[str, int]:
        """Get parameter count for each component."""
        params = {}

        if self.lidar_encoder is not None:
            params["lidar_encoder"] = sum(
                p.numel() for p in self.lidar_encoder.parameters()
            )

        if self.image_encoder is not None:
            params["image_encoder"] = sum(
                p.numel() for p in self.image_encoder.parameters()
            )

        if self.bev_transform is not None:
            params["bev_transform"] = sum(
                p.numel() for p in self.bev_transform.parameters()
            )

        if not isinstance(self.fuser, nn.Identity):
            params["fuser"] = sum(p.numel() for p in self.fuser.parameters())

        params["seg_head"] = sum(p.numel() for p in self.seg_head.parameters())

        params["total"] = sum(p.numel() for p in self.parameters())

        return params


