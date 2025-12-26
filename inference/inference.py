import sys

sys.path.append("/home/ji-liu/GitHub/off_road_nav/")
from pathlib import Path
import torch
import numpy as np
import numpy.typing as npt
from pprint import pprint

from offroad_det_seg_rellis.model.rellis_polar_bev_fusion import (
    RellisPolarBevFusion,
)
from offroad_det_seg_rellis.train.train import load_model_config
from offroad_det_seg_rellis.dataset import (
    load_camera_intrinsics,
    load_transform_from_yaml,
    filter_points_in_fov,
    preprocess_image,
)

TRAIN_DIR = Path(__file__).parent.parent / "train"

def fix_state_dict(state_dict: dict, prefix: str = "_orig_mod.") -> dict:
    keys = list(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            state_dict[key.replace(prefix, "")] = state_dict[key]
            del state_dict[key]
    return state_dict


def create_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
    prefix_to_remove: str = "_orig_mod.",
) -> RellisPolarBevFusion:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    assert "config" in checkpoint, "Config not found in checkpoint"
    config = checkpoint["config"]

    pprint(config)

    # construct model config

    model_config_path = (TRAIN_DIR / config["model_config"]).resolve()
    model_config = load_model_config(str(model_config_path))

    model = RellisPolarBevFusion(
        point_cloud_encoder_config=model_config["point_cloud_encoder"],
        image_encoder_config=model_config["image_encoder"],
        bev_transform_config=model_config["bev_transform"],
        fusion_config=model_config["fusion"],
        segmentation_head_config=model_config["segmentation_head"],
        loss_config=config["loss"],
        use_camera=True,
        use_lidar=True,
    ).to(device)

    # load weights
    assert "model_state_dict" in checkpoint, (
        "Model state dict not found in checkpoint"
    )
    model.load_state_dict(
        fix_state_dict(checkpoint["model_state_dict"], prefix_to_remove)
    )

    return model


class RellisPolarBevFusionInference:
    def __init__(
        self,
        checkpoint_path: str,
        device: str,
        cam_intrin_path: str,
        lidar2cam_path: str,
    ):
        """Create model from checkpoint and save cam and lidar info."""
        self.device = device

        if device.startswith("cuda") and torch.cuda.is_available():
            self.device = device
        else:
            self.device = "cpu"
        self.model = create_model_from_checkpoint(checkpoint_path, device)
        self.model.eval()
        self.cam_intrin = load_camera_intrinsics(cam_intrin_path)
        self.RT_lidar2cam, self.RT_cam2lidar = load_transform_from_yaml(
            lidar2cam_path
        )

        self.camera_params = {
            "camera2lidar_rots": torch.from_numpy(self.RT_cam2lidar[:3, :3])
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device),
            "camera2lidar_trans": torch.from_numpy(self.RT_cam2lidar[:3, 3])
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device),
            "camera_intrinsics": torch.from_numpy(self.cam_intrin)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device),
        }

    def _preprocess_lidar_pts(
        self, points: np.ndarray, img_width: int, img_height: int
    ) -> np.ndarray:
        # filter points to keep the points in the camera field of view
        filtered_points, _, mask = filter_points_in_fov(
            points, img_width, img_height, self.cam_intrin, self.RT_lidar2cam
        )
        return filtered_points

    def _preprocess_images(self, image: np.ndarray) -> npt.NDArray[np.float32]:
        return preprocess_image(image)

    def predict(self, points: np.ndarray, image: np.ndarray):
        points = self._preprocess_lidar_pts(
            points, image.shape[1], image.shape[0]
        )
        image = self._preprocess_images(image)

        points_tensor = torch.from_numpy(points).float().to(self.device)
        image_tensor = torch.from_numpy(image).float().to(self.device)

        # additional batch dimension consisitent with collate fuc
        image_tensor = image_tensor.unsqueeze(0)
        points_lst = [points_tensor]

        output = self.model(
            points=points_lst,
            images=image_tensor,
            camera_params=self.camera_params,
            targets=None,
        )

        logits = output["predictions"].squeeze(0)

        # get the class label
        class_labels = torch.argmax(logits, dim=0, keepdim=False)

        return class_labels.cpu().numpy()
