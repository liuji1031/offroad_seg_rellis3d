import os
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
import yaml
from scipy.spatial.transform import Rotation

from .rellis_classes import RELLIS_CLASS_INDEX_MAP
from .util import (
    filter_points_in_fov,
    fov_angle,
    load_camera_intrinsics,
    load_kitti_bin,
    load_semantic_labels,
    load_transform_from_yaml,
)

RELLIS_DATA_ROOT = os.getenv("RELLIS_DATA_ROOT", str(Path(__file__).parent.parent.parent / "RELLIS" / "Rellis-3D"))

class RellisSequence:
    """Represents a single RELLIS-3D sequence with calibration information.

    Each sequence (e.g., 00000, 00001) contains:
    - camera_info.txt: camera intrinsic parameters [fx, fy, cx, cy]
    - transforms.yaml: extrinsic transform from lidar to camera
    - calib.txt: KITTI-style calibration (not used in this implementation)
    - vel2os1.yaml: transform between Velodyne and Ouster (if available)

    Attributes:
        sequence_dir: Path to the sequence directory
        sequence_id: Sequence ID (e.g., "00000")
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients (5x1)
        RT_lidar2cam: 4x4 transformation matrix from lidar to camera coordinates
        RT_vel2os1: 4x4 transformation matrix from Velodyne to Ouster (optional)
    """

    def __init__(
        self,
        sequence_root: Union[str, Path],
        sequence_id: str,
        load_vel2os1: bool = False,
    ):
        """Initialize sequence and load calibration data.

        Args:
            sequence_dir: Path to sequence directory (e.g., /path/to/Rellis-3D/00000)
            load_vel2os1: Whether to load Velodyne to Ouster transform
        """
        self.sequence_dir = Path(sequence_root) / sequence_id
        if not self.sequence_dir.exists():
            raise FileNotFoundError(
                f"Sequence directory not found: {self.sequence_dir}"
            )

        self.sequence_id = sequence_id
        self.camera_path = self.sequence_dir / "pylon_camera_node"
        self.lidar_path_os1 = self.sequence_dir / "os1_cloud_node_kitti_bin"
        self.lidar_path_vel = self.sequence_dir / "vel_cloud_node_kitti_bin"

        self.n_total_frames = self.get_total_frames()
        # self.data_path_list = self.get_data_path_list()

        # Load camera intrinsics
        self.camera_intrinsics = self._load_camera_intrinsics()

        # Standard distortion coefficients for RELLIS-3D
        self.dist_coeffs = np.array(
            [-0.134313, -0.025905, 0.002181, 0.00084, 0.0]
        ).astype(np.float32)[:, np.newaxis]

        # Load lidar to camera transform
        self.RT_lidar2cam, self.RT_cam2lidar = self._load_lidar2cam_transform()

        # separate out the rotation and translation
        self.R_cam2lidar = self.RT_cam2lidar[:3, :3]
        self.t_cam2lidar = self.RT_cam2lidar[:3, 3]
        self.R_lidar2cam = self.RT_lidar2cam[:3, :3]
        self.t_lidar2cam = self.RT_lidar2cam[:3, 3]

        # Optionally load vel2os1 transform
        self.RT_vel2os1: Optional[npt.NDArray[np.float32]] = None
        if load_vel2os1:
            vel2os1_path = self.sequence_dir / "vel2os1.yaml"
            if vel2os1_path.exists():
                self.RT_vel2os1, _ = self._load_transform_from_yaml(
                    vel2os1_path, key="vel2os1"
                )

        self.fov_x, self.fov_y = self.get_fov_angle()

    def get_total_frames(self) -> int:
        """Get the total number of frames in the sequence."""
        n_total_frames = len(list(self.camera_path.glob("*.jpg")))
        n_total_bin = len(list(self.lidar_path_os1.glob("*.bin")))
        assert n_total_frames == n_total_bin, (
            "Number of frames and binary files do not match"
        )
        return n_total_frames

    def get_data_path_list(self):
        """Return a list of data paths.

        The list contains dictionaries with the keys "image", "lidar_os1",
        "lidar_vel", "label_id", which maps to the file names
        """
        data_path_list = [dict()] * self.n_total_frames
        for frame_idx in range(self.n_total_frames):
            data_path_list[frame_idx] = {
                "image": self.get_image_path(frame_idx),
                "lidar_os1": self.get_lidar_bin_path(frame_idx, "os1"),
                "lidar_vel": self.get_lidar_bin_path(frame_idx, "vel"),
                "camera_label_id": self.get_camera_label_path(frame_idx, "id"),
                "camera_label_color": self.get_camera_label_path(
                    frame_idx, "color"
                ),
                "lidar_os1_label": self.get_lidar_label_path(frame_idx, "os1"),
                "lidar_vel_label": self.get_lidar_label_path(frame_idx, "vel"),
            }
        return data_path_list

    def get_lidar_label_path(
        self, frame_idx: int, lidar_type: str = "os1"
    ) -> Path:
        """Get path to lidar label for given frame index.

        Args:
            frame_idx: Frame index
            lidar_type: Either 'os1' (Ouster) or 'vel' (Velodyne)
        """
        if lidar_type == "os1":
            label_dir = (
                self.sequence_dir / "os1_cloud_node_semantickitti_label_id"
            )
        elif lidar_type == "vel":
            label_dir = (
                self.sequence_dir / "vel_cloud_node_semantickitti_label_id"
            )
        else:
            raise ValueError(
                f"Invalid lidar_type: {lidar_type}, must be 'os1' or 'vel'"
            )
        label_path = label_dir / f"{frame_idx:06d}.label"
        if not label_path.exists():
            raise FileNotFoundError(f"Lidar label file not found: {label_path}")
        return label_path

    def get_image(self, frame_idx: int) -> npt.NDArray[np.uint8]:
        """Get the image for given frame index.

        Args:
            frame_idx: Frame index
        """
        img_path = self.get_image_path(frame_idx)
        assert img_path.exists(), f"Image not found: {img_path}"
        return cv2.imread(str(img_path))  # type: ignore

    def get_lidar_points(
        self, frame_idx: int, lidar_type: str = "os1"
    ) -> npt.NDArray[np.float32]:
        """Get the lidar points for given frame index.

        Args:
            frame_idx: Frame index
            lidar_type: Either 'os1' (Ouster) or 'vel' (Velodyne)
        """
        return load_kitti_bin(self.get_lidar_bin_path(frame_idx, lidar_type))

    def get_lidar_label(
        self, frame_idx: int, lidar_type: str = "os1"
    ) -> npt.NDArray[np.int32]:
        """Get the lidar semantic label for given frame index.

        Args:
            frame_idx: Frame index
            lidar_type: Either 'os1' or 'vel'
        """
        return load_semantic_labels(
            self.get_lidar_label_path(frame_idx, lidar_type),
            ind_remap_dict=RELLIS_CLASS_INDEX_MAP,
        )

    def get_image_size(self) -> Tuple[int, int]:
        """Get the image size from the first image in the sequence.

        Returns:
            Tuple of (height, width)
        """
        img_path = self.get_image_path(0)
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        return image.shape[:2]  # type: ignore

    def _load_camera_intrinsics(self) -> npt.NDArray[np.float32]:
        """Load camera intrinsic parameters from camera_info.txt.

        Returns:
            3x3 camera matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        """
        camera_info_path = self.sequence_dir / "camera_info.txt"
        return load_camera_intrinsics(str(camera_info_path))

    def _load_lidar2cam_transform(
        self,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Load lidar-to-camera extrinsic transform from transforms.yaml.

        Returns:
            4x4 transformation matrix
        """
        transforms_path = self.sequence_dir / "transforms.yaml"
        if not transforms_path.exists():
            raise FileNotFoundError(
                f"transforms.yaml not found in {self.sequence_dir}"
            )

        return load_transform_from_yaml(
            str(transforms_path), key="os1_cloud_node-pylon_camera_node"
        )

    def get_image_path(self, frame_idx: int) -> Path:
        """Get path to camera image for given frame index.

        Args:
            frame_idx: Frame index (e.g., 0, 1, 2, ...)

        Returns:
            Path to image file
        """
        img_dir = self.sequence_dir / "pylon_camera_node"
        pattern = f"{frame_idx:06d}"
        matches = list(img_dir.glob(f"frame{pattern}-*.jpg"))
        if not matches:
            raise FileNotFoundError(
                f"No image found for frame {frame_idx} in {img_dir}"
            )
        return matches[0]

    def get_lidar_bin_path(
        self, frame_idx: int, lidar_type: str = "os1"
    ) -> Path:
        """Get path to KITTI binary point cloud for given frame index.

        Args:
            frame_idx: Frame index (e.g., 0, 1, 2, ...)
            lidar_type: Either 'os1' (Ouster) or 'vel' (Velodyne)

        Returns:
            Path to .bin file
        """
        if lidar_type == "os1":
            bin_dir = self.sequence_dir / "os1_cloud_node_kitti_bin"
        elif lidar_type == "vel":
            bin_dir = self.sequence_dir / "vel_cloud_node_kitti_bin"
        else:
            raise ValueError(
                f"Invalid lidar_type: {lidar_type}, must be 'os1' or 'vel'"
            )

        bin_path = bin_dir / f"{frame_idx:06d}.bin"
        if not bin_path.exists():
            raise FileNotFoundError(f"Binary file not found: {bin_path}")
        return bin_path

    def get_camera_label_path(
        self, frame_idx: int, label_type: str = "color"
    ) -> Path | None:
        """Get path to semantic label for given frame index.

        Args:
            frame_idx: Frame index
            label_type: Either 'color' or 'id'

        Returns:
            Path to label file
        """
        if label_type == "color":
            label_dir = self.sequence_dir / "pylon_camera_node_label_color"
            ext = ".png"
        elif label_type == "id":
            label_dir = self.sequence_dir / "pylon_camera_node_label_id"
            ext = ".png"
        else:
            raise ValueError(f"Invalid label_type: {label_type}")

        pattern = f"{frame_idx:06d}"
        matches = list(label_dir.glob(f"frame{pattern}-*{ext}"))
        if not matches:
            return None
        return matches[0]

    def __repr__(self) -> str:
        return f"RellisSequence(id={self.sequence_id}, dir={self.sequence_dir})"

    def get_fov_angle(self, add_margin: float = 0.0) -> tuple[float, float]:
        """Return the camera field-of-view angles in radians.

        Args:
            add_margin: Extra margin to add to the FOV (interpreted in radians).
        """
        img_height, img_width = self.get_image_size()
        fov_x = fov_angle(
            img_width,
            self.camera_intrinsics[0, 0],
            add_margin=add_margin,
            unit="rad",
        )
        fov_y = fov_angle(
            img_height,
            self.camera_intrinsics[1, 1],
            add_margin=add_margin,
            unit="rad",
        )
        return fov_x, fov_y

    def filter_points_in_fov(
        self, points: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """Filter points that are within the camera field of view.

        Args:
            points: Nx3 array of points in lidar coordinates
        """
        img_height, img_width = self.get_image_size()
        filtered_points, _, mask = filter_points_in_fov(
            points,
            img_width,
            img_height,
            self.camera_intrinsics,
            self.RT_lidar2cam,
        )

        return filtered_points, mask

    def load_bev_seg_config(self, grid_type: str = "polar") -> dict:
        """Load BEV segmentation config from json file.

        Args:
            grid_type: Type of grid to use, either "cartesian" or "polar"

        Returns:
            BEV segmentation config
        """
        config_path = (
            self.sequence_dir / f"bev_seg_{grid_type}" / "bev_seg_config.json"
        )
        if not config_path.exists():
            raise FileNotFoundError(
                f"BEV segmentation config not found: {config_path}"
            )
        return json.load(open(config_path, "r", encoding="utf-8"))

    def load_bev_seg_gt(
        self, frame_idx: int, grid_type: str = "polar"
    ) -> npt.NDArray[np.int32]:
        file_name = f"{frame_idx:06d}.npy"
        file_path = self.sequence_dir / f"bev_seg_{grid_type}" / file_name
        assert file_path.exists(), (
            f"BEV segmentation ground truth not found: {file_path}"
        )
        return np.load(file_path).astype(np.int32)


@lru_cache(maxsize=None)
def get_rellis_sequence(
    sequence_root: Union[str, Path],
    sequence_id: str,
    load_vel2os1: bool = False,
) -> RellisSequence:
    """Return a cached `RellisSequence` instance for the given parameters."""
    normalized_root = Path(sequence_root).expanduser().resolve()
    return RellisSequence(
        normalized_root, sequence_id, load_vel2os1=load_vel2os1
    )


@lru_cache(maxsize=None)
def get_rellis_sequence_by_id(sequence_id: str):
    return get_rellis_sequence(RELLIS_DATA_ROOT, sequence_id)
