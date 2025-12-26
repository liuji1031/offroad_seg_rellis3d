import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Union, Tuple, Any
import yaml
from scipy.spatial.transform import Rotation


def load_model_config(config_path: Union[str, Path]) -> dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def fov_angle(
    img_size: int,
    f: float,
    add_margin: float = np.deg2rad(10.0),
    unit: str = "rad",
) -> float:
    """Return the field of view angle of the camera.

    Args:
        img_size: Image dimension in pixels (width or height).
        f: Focal length.
        add_margin: Extra margin to add to the FOV (interpreted in radians).
        unit: Unit of the returned angle, either ``\"rad\"`` or ``\"deg\"``.
    """
    base_angle = 2 * np.arctan2(img_size / 2, f)
    angle_rad = base_angle + add_margin

    if unit == "rad":
        return angle_rad
    if unit == "deg":
        return np.degrees(angle_rad)
    raise ValueError(f"Invalid unit: {unit}")


def load_kitti_bin(
    bin_path: Union[str, Path], include_reflectivity: bool = True
) -> npt.NDArray[np.float32]:
    """Load KITTI binary point cloud file.

    Args:
        bin_path: Path to .bin file

    Returns:
        Nx3 array of points (x, y, z), reflectivity is ignored
    """
    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(f"Binary file not found: {bin_path}")

    # KITTI format: Nx4 (x, y, z, reflectivity)
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    if include_reflectivity:
        return points
    else:
        return points[:, :3]  # Return only xyz


def load_semantic_labels(
    label_path: Union[str, Path], ind_remap_dict: dict | None = None
) -> npt.NDArray[np.int32]:
    """Load semantic labels from .label file.

    Args:
        label_path: Path to .label file

        ind_remap_dict: Dictionary of index remapping for the labels
    Returns:
        Array of semantic label IDs (N,)
    """
    label_path = Path(label_path)
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    # Load labels as uint32 (lower 16 bits = semantic, upper 16 bits = instance)
    label = np.fromfile(label_path, dtype=np.uint32).reshape(-1)

    # Extract semantic labels (lower 16 bits)
    sem_label = (label & 0xFFFF).astype(np.int32)

    if ind_remap_dict is not None:
        sem_label = np.vectorize(ind_remap_dict.get)(sem_label)
    return sem_label


def depth_color(
    depth: npt.NDArray[np.float32],
    min_d: float = 0.0,
    max_d: float = 120.0,
) -> npt.NDArray[np.uint8]:
    """Map depth values to hue colors (in HSV space).

    Args:
        depth: Depth values
        min_d: Minimum depth for color mapping
        max_d: Maximum depth for color mapping

    Returns:
        Hue values (0-120) as uint8
    """
    max_hue_val = 120  # choice seems arbitrary, but it works
    depth = np.clip(depth, min_d, max_d)
    hue = ((depth - min_d) / (max_d - min_d) * max_hue_val).astype(np.uint8)
    return hue


def filter_points_in_fov(
    points: npt.NDArray[np.float32],
    img_width: int,
    img_height: int,
    camera_matrix: npt.NDArray[np.float32],
    RT_lidar2cam: npt.NDArray[np.float32],
) -> Tuple[
    npt.NDArray[np.float32], npt.NDArray[np.uint8], npt.NDArray[np.bool_]
]:
    """Filter points that are within the camera field of view.

    Args:
        points: Nx3 array of points in lidar coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels
        camera_matrix: 3x3 camera intrinsic matrix
        RT_lidar2cam: 4x4 transformation from lidar to camera

    Returns:
        Tuple of (filtered_points, depth_colors):
            - filtered_points: Mx3 array of points in lidar coordinates
            - depth_colors: Hue values (0-120) for visualization
    """
    # Convert to homogeneous coordinates
    p_l = np.ones((points.shape[0], 4))
    p_l[:, :3] = points[:, :3]  # ignore the reflectivity

    # Transform to camera coordinates
    p_c = (RT_lidar2cam @ p_l.T).T

    x, y, z = p_c[:, 0], p_c[:, 1], p_c[:, 2]

    # Compute field of view
    fov_x = fov_angle(img_width, camera_matrix[0, 0])
    fov_y = fov_angle(img_height, camera_matrix[1, 1])

    # Compute angles (radians)
    x_angle = np.arctan2(x, z)
    y_angle = np.arctan2(y, z)

    # Filter points within FOV
    flag_x = (x_angle > -fov_x / 2) & (x_angle < fov_x / 2)
    flag_y = (y_angle > -fov_y / 2) & (y_angle < fov_y / 2)
    flag_z = z > 0  # Only points in front of camera

    mask = flag_x & flag_y & flag_z
    filtered_points = points[mask]

    # Compute depth colors
    x_f, y_f, z_f = p_c[mask, 0], p_c[mask, 1], p_c[mask, 2]
    dist_f = np.sqrt(x_f**2 + y_f**2 + z_f**2)
    colors = depth_color(dist_f, 0, 70)

    return filtered_points, colors, mask


def load_camera_intrinsics(cam_info_path: str) -> npt.NDArray[np.float32]:
    camera_info_path = Path(cam_info_path)
    if not camera_info_path.exists():
        raise FileNotFoundError(f"camera_info.txt not found in {cam_info_path}")

    # Format: fx fy cx cy
    data = np.loadtxt(camera_info_path)
    P = np.zeros((3, 3), dtype=np.float32)
    P[0, 0] = data[0]  # fx
    P[1, 1] = data[1]  # fy
    P[0, 2] = data[2]  # cx
    P[1, 2] = data[3]  # cy
    P[2, 2] = 1.0
    return P


def quat_to_rotation_matrix(
    q: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix.

    Fallback implementation if scipy is not available.
    """
    x, y, z, w = q
    R = np.array(
        [
            [
                1 - 2 * (y**2 + z**2),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
            ],
            [
                2 * (x * y + w * z),
                1 - 2 * (x**2 + z**2),
                2 * (y * z - w * x),
            ],
            [
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x**2 + y**2),
            ],
        ]
    )
    return R


def load_transform_from_yaml(
    yaml_path: str, key: str = "os1_cloud_node-pylon_camera_node"
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Load 4x4 transformation matrix from YAML file.

    The file stores the transformation cam -> lidar (camera frame expressed in lidar frame).
    We invert it to get the transformation lidar -> cam (lidar frame expressed in camera frame).
    Args:
        yaml_path: Path to YAML file
        key: Key to access in YAML (e.g., 'os1_cloud_node-pylon_camera_node')

    Returns:
        4x4 transformation matrix (inverted to get camera-from-lidar)
        4x4 transformation matrix (from camera to lidar)
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if key not in data:
        raise KeyError(f"Key '{key}' not found in {yaml_path}")

    # Extract quaternion and translation
    q_data = data[key]["q"]
    t_data = data[key]["t"]

    q = np.array([q_data["x"], q_data["y"], q_data["z"], q_data["w"]])
    t = np.array([t_data["x"], t_data["y"], t_data["z"]])

    # Convert quaternion to rotation matrix using scipy
    try:
        R = Rotation.from_quat(q).as_matrix()
    except ImportError:
        # Fallback: manual quaternion to rotation matrix conversion
        R = quat_to_rotation_matrix(q)

    # Build 4x4 transformation matrix
    RT_cam2lidar = np.eye(4, dtype=np.float32)
    RT_cam2lidar[:3, :3] = R
    RT_cam2lidar[:3, 3] = t

    # Invert to get camera-from-lidar
    RT_lidar2cam = np.linalg.inv(RT_cam2lidar)
    return RT_lidar2cam, RT_cam2lidar
