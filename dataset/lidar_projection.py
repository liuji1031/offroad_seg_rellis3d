"""Utilities for projecting LiDAR point clouds onto camera images.

This module provides tools to work with the RELLIS-3D dataset, including
loading calibration data and projecting KITTI binary point clouds onto images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .bev_segmentation import RELLIS_COLOR_MAP
from .rellis_sequence import RellisSequence, get_rellis_sequence
from .util import filter_points_in_fov, load_kitti_bin


def project_points_to_image(
    points: npt.NDArray[np.float32],
    camera_matrix: npt.NDArray[np.float32],
    dist_coeffs: npt.NDArray[np.float32],
    RT_lidar2cam: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Project 3D points onto 2D image plane.

    Args:
        points: Nx3 array of points in lidar coordinates
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: 5x1 distortion coefficients
        RT_lidar2cam: 4x4 transformation from lidar to camera

    Returns:
        2xN array of image coordinates (u, v)
    """
    # Extract rotation and translation for cv2.projectPoints
    R = RT_lidar2cam[:3, :3]
    t = RT_lidar2cam[:3, 3].reshape(3, 1)

    # Convert rotation matrix to Rodrigues vector
    rvec, _ = cv2.Rodrigues(R)
    tvec = t

    # Project points
    img_points, _ = cv2.projectPoints(
        points, rvec, tvec, camera_matrix, dist_coeffs
    )

    # Reshape to 2xN
    img_points = np.squeeze(img_points, 1).T
    return img_points.astype(np.float32)


def overlay_points_on_image(
    image: npt.NDArray[np.uint8],
    img_points: npt.NDArray[np.float32],
    colors: npt.NDArray[np.uint8],
    point_size: int = 2,
) -> npt.NDArray[np.uint8]:
    """Draw colored points on image.

    Args:
        image: HxWx3 BGR image
        img_points: 2xN array of image coordinates
        colors: N-length array of RGB colors
        point_size: Radius of circles to draw

    Returns:
        Image with points overlaid (RGB format)
    """
    image_copy = image.copy()
    # Draw each point
    for i in range(img_points.shape[1]):
        u, v = int(img_points[0, i]), int(img_points[1, i])
        cv2.circle(image_copy, (u, v), point_size, colors[i].tolist(), -1)
    return image_copy


def visualize_lidar_on_image(
    sequence: RellisSequence,
    frame_idx: int,
    lidar_type: str = "os1",
    point_size: int = 2,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> npt.NDArray[np.uint8]:
    """Complete pipeline to visualize LiDAR points on camera image.

    Args:
        sequence: RellisSequence instance
        frame_idx: Frame index to visualize
        lidar_type: Either 'os1' or 'vel'
        point_size: Size of points to draw
        save_path: Optional path to save result image
        show: Whether to display using matplotlib

    Returns:
        RGB image with LiDAR points overlaid
    """
    # Load image and point cloud
    img_path = sequence.get_image_path(frame_idx)
    bin_path = sequence.get_lidar_bin_path(frame_idx, lidar_type=lidar_type)

    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_height, img_width = image.shape[:2]

    points = load_kitti_bin(bin_path)

    # If using Velodyne, transform to Ouster frame first
    if lidar_type == "vel" and sequence.RT_vel2os1 is not None:
        points_hom = np.ones((points.shape[0], 4), dtype=np.float32)
        points_hom[:, :3] = points
        points = (sequence.RT_vel2os1 @ points_hom.T).T[:, :3]

    # Filter points in FOV
    filtered_points, _, mask = filter_points_in_fov(
        points,
        img_width,
        img_height,
        sequence.camera_intrinsics,
        sequence.RT_lidar2cam,
    )

    # load the labels of the filtered points
    labels = sequence.get_lidar_label(frame_idx, lidar_type=lidar_type)[mask]
    colors = np.array(
        [RELLIS_COLOR_MAP[label] for label in labels], dtype=np.uint8
    )
    # Project to image
    img_points = project_points_to_image(
        filtered_points,
        sequence.camera_intrinsics,
        sequence.dist_coeffs,
        sequence.RT_lidar2cam,
    )

    # Overlay on image
    result = overlay_points_on_image(image, img_points, colors, point_size)  # type: ignore

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        cv2.imwrite(str(save_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    # Show if requested
    if show:
        plt.figure(figsize=(20, 12))
        plt.imshow(result)
        plt.title(
            f"Sequence {sequence.sequence_id}, Frame {frame_idx} ({lidar_type})"
        )
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize KITTI binary point clouds on RELLIS-3D images"
    )
    parser.add_argument(
        "sequence_dir",
        help="Path to sequence directory (e.g., /path/to/Rellis-3D/00000)",
    )
    parser.add_argument(
        "sequence_id",
        help="Sequence ID (e.g., 00000)",
    )
    parser.add_argument(
        "frame_idx",
        type=int,
        help="Frame index to visualize (e.g., 104)",
    )
    parser.add_argument(
        "--lidar-type",
        choices=["os1", "vel"],
        default="os1",
        help="LiDAR type: 'os1' (Ouster) or 'vel' (Velodyne)",
    )
    parser.add_argument(
        "--point-size",
        type=int,
        default=2,
        help="Size of points to draw",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save output image",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the result",
    )

    args = parser.parse_args()

    # Load sequence
    load_vel = args.lidar_type == "vel"
    seq = get_rellis_sequence(
        args.sequence_dir, args.sequence_id, load_vel2os1=load_vel
    )

    # Visualize
    visualize_lidar_on_image(
        seq,
        args.frame_idx,
        lidar_type=args.lidar_type,
        point_size=args.point_size,
        save_path=args.save,
        show=not args.no_show,
    )
