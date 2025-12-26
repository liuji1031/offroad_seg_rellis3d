"""BEV Segmentation Map Generation for RELLIS-3D Dataset.

This module generates Bird's Eye View (BEV) segmentation maps from LiDAR point clouds
using KNN-based label assignment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from offroad_det_seg_rellis.util import get_logger

from .rellis_classes import RELLIS_CLASSES, RELLIS_COLOR_MAP
from .rellis_sequence import get_rellis_sequence

logger = get_logger(__name__)

FloatNDArray = npt.NDArray[np.floating[Any]]


def generate_bev_grid(
    x_range: Tuple[float, float] = (-25.0, 25.0),
    y_range: Tuple[float, float] = (-25.0, 25.0),
    resolution: float = 0.25,
) -> Tuple[npt.NDArray[np.float32], Tuple[int, int]]:
    """Generate BEV grid cell centers.

    Args:
        x_range: (x_min, x_max) in meters
        y_range: (y_min, y_max) in meters
        resolution: Grid resolution in meters

    Returns:
        Tuple of (grid_centers, grid_shape):
            - grid_centers: (H*W, 2) array of (x, y) coordinates
            - grid_shape: (H, W) tuple
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Create grid boundaries
    x_edges = np.arange(x_min, x_max + resolution, resolution)
    y_edges = np.arange(y_min, y_max + resolution, resolution)

    # Compute grid cell centers
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0

    # Create meshgrid of centers
    X, Y = np.meshgrid(x_centers, y_centers)

    # Flatten to (H*W, 2)
    grid_centers = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)

    grid_shape = (len(y_centers), len(x_centers))

    return grid_centers, grid_shape


def _compute_angle_edges(
    angle_range: Tuple[float, float], angle_resolution: float
) -> npt.NDArray[np.float32]:
    return np.linspace(
        angle_range[0],
        angle_range[1],
        num=int(np.ceil((angle_range[1] - angle_range[0]) / angle_resolution))
        + 1,
        endpoint=True,
    )


def _compute_radial_edges(
    radial_max: float,
    radial_resolution: float,
    radial_mode: str,
    radial_growth_rate: float,
    radial_min: float = 0.0,
) -> npt.NDArray[np.float32]:
    if radial_resolution <= 0:
        raise ValueError("radial_resolution must be positive")
    if radial_max <= 0:
        raise ValueError("radial_max must be positive")
    if radial_min < 0:
        raise ValueError("radial_min must be non-negative")
    if radial_min >= radial_max:
        raise ValueError("radial_min must be less than radial_max")

    edges = [radial_min]
    i = 0
    current_radius = radial_min
    max_iterations = 10000

    while current_radius < radial_max and i < max_iterations:
        if radial_mode == "linear":
            width = radial_resolution * (1.0 + radial_growth_rate * i)
        elif radial_mode == "exponential":
            width = radial_resolution * (radial_growth_rate**i)
        else:
            raise ValueError(
                f"Unsupported radial_mode '{radial_mode}'. Use 'linear' or 'exponential'."
            )
        current_radius = min(radial_max, current_radius + width)
        edges.append(current_radius)
        i += 1

    if edges[-1] < radial_max:
        edges.append(radial_max)

    return np.array(edges, dtype=np.float32)


def compute_polar_grid_params(
    angle_range: Tuple[float, float],
    angle_resolution: float,
    radial_max: float,
    radial_resolution: float,
    radial_mode: str,
    radial_growth_rate: float,
    radial_min: float = 0.0,
) -> Dict[str, npt.NDArray[np.float32]]:
    angle_edges = _compute_angle_edges(angle_range, angle_resolution)
    radial_edges = _compute_radial_edges(
        radial_max,
        radial_resolution,
        radial_mode,
        radial_growth_rate,
        radial_min,
    )

    if angle_edges.size < 2 or radial_edges.size < 2:
        raise ValueError(
            "Polar grid must have at least 1 bin for angles and radius"
        )

    return {
        "angle_edges": angle_edges,
        "radial_edges": radial_edges,
        "angle_range": np.array(angle_range, dtype=np.float32),
    }


def _unit_vectors_from_angles(angles: FloatNDArray) -> FloatNDArray:
    """Return unit vectors for each input angle."""
    angles32: npt.NDArray[np.float32] = angles.astype(np.float32, copy=False)
    cos_vals: FloatNDArray = np.cos(angles32)
    sin_vals: FloatNDArray = np.sin(angles32)
    return np.stack((cos_vals, sin_vals), axis=-1)


def _cross_product_z(vec_a: FloatNDArray, vec_b: FloatNDArray) -> FloatNDArray:
    """Compute the 2D cross-product (Z component) between pairs of vectors."""
    return vec_a[..., 0] * vec_b[..., 1] - vec_a[..., 1] * vec_b[..., 0]


def _compute_angle_bin_indices_simple(
    angles: FloatNDArray,
    angle_min: float,
    angle_max: float,
    angle_resolution: float,
) -> npt.NDArray[np.int32]:
    """Compute the angle bin indices for a given angle range and resolution."""
    indices = np.floor((angles - angle_min) / angle_resolution).astype(np.int32)
    num_bins = int(np.ceil((angle_max - angle_min) / angle_resolution))
    indices = np.clip(indices, 0, num_bins - 1)
    return indices


def _compute_angle_bin_indices(
    angles: FloatNDArray,
    angle_edges: FloatNDArray,
    tolerance: float = 1e-9,
) -> npt.NDArray[np.int32]:
    """Assign each angle to a polar bin using cross-product direction tests.

    This function determines which angular sector each point angle belongs to by
    checking if the cross products have consistent orientation. For a point angle θ
    to be within sector [φ1, φ2], the following must hold:
        - (φ1 × θ) and (θ × φ2) must have the same sign as (φ1 × φ2)

    This approach handles angle wrap-around naturally because it operates on unit
    vectors rather than raw angle values, avoiding issues with discontinuities
    at ±π from np.arctan2.

    Args:
        angles: Array of point angles in radians (from np.arctan2)
        angle_edges: Monotonically increasing bin edges in radians
        tolerance: Numerical tolerance for cross-product comparisons

    Returns:
        Array of bin indices for each angle (-1 if not assigned to any bin)
    """
    # Handle empty input
    if angles.size == 0:
        return np.empty(0, dtype=np.int32)

    # Convert angles to unit vectors to handle wrap-around naturally
    # Unit vectors remove the ±π discontinuity from arctan2
    edge_vectors = _unit_vectors_from_angles(angle_edges.astype(np.float32))
    angle_vectors = _unit_vectors_from_angles(angles.astype(np.float32))

    # Extract start and end vectors for each angular sector
    # sector_starts[i] is the unit vector for angle_edges[i]
    # sector_ends[i] is the unit vector for angle_edges[i+1]
    sector_starts = edge_vectors[:-1]  # Shape: (num_bins, 2)
    sector_ends = edge_vectors[1:]  # Shape: (num_bins, 2)

    # Compute cross product (φ1 × φ2) for each sector
    # This gives the "reference direction" for the sector
    cross_sectors = _cross_product_z(
        sector_starts, sector_ends
    )  # Shape: (num_bins,)

    # Filter out degenerate sectors where start and end are nearly parallel
    # (occurs when angle bin width is ~0 or ~2π)
    non_degenerate = np.abs(cross_sectors) > tolerance
    if not np.any(non_degenerate):
        return np.full(angles.shape[0], -1, dtype=np.int32)

    # Keep only valid sectors for comparison
    valid_starts = sector_starts[non_degenerate]  # Shape: (num_valid_bins, 2)
    valid_ends = sector_ends[non_degenerate]  # Shape: (num_valid_bins, 2)
    valid_cross = cross_sectors[non_degenerate]  # Shape: (num_valid_bins,)
    valid_bin_indices = np.nonzero(non_degenerate)[0]  # Original bin indices

    # Compute (φ1 × θ) for all sector-point pairs
    # Broadcasting: (num_valid_bins, 1, 2) × (1, num_points, 2) → (num_valid_bins, num_points)
    cross_start_theta = _cross_product_z(
        valid_starts[:, None, :], angle_vectors[None, :, :]
    )

    # Compute (θ × φ2) for all sector-point pairs
    # Broadcasting: (1, num_points, 2) × (num_valid_bins, 1, 2) → (num_valid_bins, num_points)
    cross_theta_end = _cross_product_z(
        angle_vectors[None, :, :], valid_ends[:, None, :]
    )

    # Check if both cross products have the same orientation as (φ1 × φ2)
    # For θ to be in [φ1, φ2]:
    #   - (φ1 × θ) · (φ1 × φ2) >= 0  (θ is counter-clockwise from φ1)
    #   - (θ × φ2) · (φ1 × φ2) > 0   (θ is clockwise from φ2)
    # Shape: (num_valid_bins, num_points)
    same_orientation = (
        cross_start_theta * valid_cross[:, None] >= -tolerance
    ) & (cross_theta_end * valid_cross[:, None] > tolerance)

    # Initialize all bins as unassigned (-1)
    assigned_bins = np.full(angles.shape[0], -1, dtype=np.int32)
    if same_orientation.size == 0:
        return assigned_bins

    # Find points that belong to at least one sector
    membership_mask = np.any(same_orientation, axis=0)  # Shape: (num_points,)

    # For points in multiple sectors (edge cases), assign to the first matching sector
    if np.any(membership_mask):
        candidate_indices = np.argmax(
            same_orientation[:, membership_mask], axis=0
        )
        assigned_bins[membership_mask] = valid_bin_indices[candidate_indices]

    return assigned_bins


def generate_bev_segmentation_map_cartesian(
    points: npt.NDArray[np.float32],
    labels: npt.NDArray[np.int32],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    resolution: float,
) -> Tuple[npt.NDArray[np.int32], dict]:
    """Generate a cartesian BEV segmentation map using majority voting.

    Args:
        points: Nx3 array of points in lidar coordinates
        labels: N array of label IDs
        x_range: (x_min, x_max) in meters
        y_range: (y_min, y_max) in meters
        resolution: Grid resolution in meters
    """

    points_xy = points[:, :2].astype(np.float32)
    x_min, x_max = x_range
    y_min, y_max = y_range
    mask = (
        (points_xy[:, 0] >= x_min)
        & (points_xy[:, 0] < x_max)
        & (points_xy[:, 1] >= y_min)
        & (points_xy[:, 1] < y_max)
    )
    points_xy = points_xy[mask]
    labels = labels[mask]

    grid_centers, grid_shape = generate_bev_grid(x_range, y_range, resolution)
    h, w = grid_shape
    bev_map = np.zeros(grid_shape, dtype=np.int32)

    if len(points_xy) > 0:
        x_indices = ((points_xy[:, 0] - x_min) / resolution).astype(np.int32)
        y_indices = ((points_xy[:, 1] - y_min) / resolution).astype(np.int32)
        x_indices = np.clip(x_indices, 0, w - 1)
        y_indices = np.clip(y_indices, 0, h - 1)

        cart_cell_labels: Dict[Tuple[int, int], list[int]] = {}
        for idx in range(len(points_xy)):
            cell_key = (y_indices[idx], x_indices[idx])
            cart_cell_labels.setdefault(cell_key, []).append(labels[idx])

        for (y_idx, x_idx), cell_label_list in cart_cell_labels.items():
            unique_labels, counts = np.unique(
                cell_label_list, return_counts=True
            )
            bev_map[y_idx, x_idx] = unique_labels[np.argmax(counts)]

    config_data = {
        "x_range": x_range,
        "y_range": y_range,
        "resolution": resolution,
        "grid_centers": grid_centers.tolist(),
        "grid_shape": grid_shape,
    }
    return bev_map, config_data


def generate_bev_segmentation_map_polar(
    points: npt.NDArray[np.float32],
    labels: npt.NDArray[np.int32],
    angle_resolution: float = np.deg2rad(3.0),
    angle_range: Tuple[float, float] = (-np.pi / 2, np.pi / 2),
    radial_resolution: float = 0.25,
    radial_mode: str = "linear",
    radial_growth_rate: float = 0.05,
    radial_max: float = 25.0,
    radial_min: float = 0.0,
    show_point_counts_hist: bool = False,
    wrap_positive: bool = True,
) -> Tuple[npt.NDArray[np.int32], dict]:
    """Generate a polar BEV segmentation map using angle/radius bins.

    Args:
        points: Nx3 array of points in lidar coordinates
        labels: N array of label IDs
        angle_resolution: Angle resolution in radians
        angle_range: Angle range in radians
        radial_resolution: Radial resolution in meters
        radial_mode: Radial mode, either "linear" or "exponential"
        radial_growth_rate: Radial growth rate
        radial_max: Radial maximum in meters
        radial_min: Radial minimum in meters (points closer than this are ignored)
        show_point_counts_hist: Whether to show a histogram of the point counts
        wrap_positive: Whether to wrap the angles to the positive range
    Returns:
        BEV segmentation map (num_angle_bins, num_radial_bins) with label IDs
    """
    points_xy = points[:, :2].astype(np.float32)
    polar_params = compute_polar_grid_params(
        angle_range,
        angle_resolution,
        radial_max,
        radial_resolution,
        radial_mode,
        radial_growth_rate,
        radial_min,
    )

    angle_edges = polar_params["angle_edges"]
    radial_edges = polar_params["radial_edges"]
    num_angle_bins = len(angle_edges) - 1
    num_radial_bins = len(radial_edges) - 1

    bev_map = np.zeros((num_angle_bins, num_radial_bins), dtype=np.int32)
    grid_points_count = np.zeros_like(bev_map, dtype=np.int32)

    if len(points_xy) > 0:
        angles = np.arctan2(points_xy[:, 1], points_xy[:, 0])
        if wrap_positive:
            angles[angles < 0] += 2 * np.pi
        radii = np.linalg.norm(points_xy, axis=1)

        # Filter out points that are too close to the lidar sensor
        radial_mask = (radii >= radial_min) & (radii < radial_max)
        assert np.any(radial_mask), (
            "No points left after filtering by radial_min"
        )

        # Apply radial filter
        points_xy = points_xy[radial_mask]
        labels = labels[radial_mask]
        angles = angles[radial_mask]
        radii = radii[radial_mask]

        angle_idx = _compute_angle_bin_indices_simple(
            angles, angle_range[0], angle_range[1], angle_resolution
        )
        radial_idx = np.searchsorted(radial_edges, radii, side="right") - 1

        valid_mask = (
            (angle_idx >= 0)
            & (angle_idx < num_angle_bins)
            & (radial_idx >= 0)
            & (radial_idx < num_radial_bins)
        )

        if np.any(valid_mask):
            angle_idx = angle_idx[valid_mask]
            radial_idx = radial_idx[valid_mask]
            valid_labels = labels[valid_mask]

            polar_cell_labels: Dict[Tuple[int, int], list[int]] = {}
            for idx in range(len(valid_labels)):
                cell_key = (angle_idx[idx], radial_idx[idx])
                polar_cell_labels.setdefault(cell_key, []).append(
                    valid_labels[idx]
                )

            for (a_idx, r_idx), cell_label_list in polar_cell_labels.items():
                unique_labels, counts = np.unique(
                    cell_label_list, return_counts=True
                )
                bev_map[a_idx, r_idx] = unique_labels[np.argmax(counts)]
                grid_points_count[a_idx, r_idx] = len(cell_label_list)
    # Convert angle_resolution to float if it's a numpy scalar
    angle_res_val = (
        float(angle_resolution)
        if isinstance(angle_resolution, (np.floating, np.integer))
        else angle_resolution
    )
    angle_range_val = [
        float(angle_range[0])
        if isinstance(angle_range[0], (np.floating, np.integer))
        else angle_range[0],
        float(angle_range[1])
        if isinstance(angle_range[1], (np.floating, np.integer))
        else angle_range[1],
    ]
    config_data = {
        "angle_resolution": angle_res_val,
        "angle_range": angle_range_val,
        "radial_resolution": radial_resolution,
        "radial_mode": radial_mode,
        "radial_growth_rate": radial_growth_rate,
        "radial_max": radial_max,
        "radial_min": radial_min,
        "angle_edges": angle_edges.tolist(),
        "radial_edges": radial_edges.tolist(),
    }

    if show_point_counts_hist:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.hist(
            grid_points_count.flatten(),
            bins=np.arange(0, np.max(grid_points_count) + 1),
        )  # type: ignore
        plt.title("Point Counts Histogram")
        plt.xlabel("Point Count")
        plt.ylabel("Frequency")
        plt.subplot(1, 2, 2)
        plt.imshow(grid_points_count, cmap="viridis")
        plt.title("Point Counts Heatmap")
        plt.xlabel("Angular Bin")
        plt.ylabel("Radial Bin")
        plt.colorbar()
        plt.show()
    return bev_map, config_data


def process_sequence(
    sequence_root: Union[str, Path],
    sequence_id: str,
    filter_camera_fov: bool,
    grid_type: str,
    config: dict[str, Any],
    lidar_type: str = "os1",
    output_dir_name: str = "bev_seg",
    frame_indices: list[int] | None = None,
    load_vel2os1: bool = False,
    show_point_counts_hist: bool = False,
) -> None:
    """Process a sequence and generate BEV segmentation maps.

    Args:
        sequence_root: Root directory containing sequences
        sequence_id: Sequence ID
        filter_camera_fov: Whether to filter points in camera FOV
        grid_type: Type of grid to use, either "cartesian" or "polar"
        config: Configuration dictionary
        lidar_type: Type of lidar to use, either "os1" or "vel"
        output_dir_name: Name of the output directory
        frame_indices: List of frame indices to process

    Returns:
        None
    """
    assert lidar_type in ["os1", "vel"], f"Invalid lidar type: {lidar_type}"
    assert grid_type in ["cartesian", "polar"], (
        f"Invalid grid type: {grid_type}"
    )
    # create the rellis sequence object
    sequence = get_rellis_sequence(
        sequence_root, sequence_id, load_vel2os1=load_vel2os1
    )
    if not output_dir_name.endswith(grid_type):
        output_dir_name = f"{output_dir_name}_{grid_type}"
    output_dir = Path(sequence_root) / sequence_id / output_dir_name
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    frame_to_process = (
        frame_indices
        if frame_indices is not None
        else range(sequence.n_total_frames)
    )

    logger.info(
        "Processing sequence %s: %d frames", sequence_id, len(frame_to_process)
    )

    for frame_idx in tqdm(frame_to_process, desc=f"Sequence {sequence_id}"):
        points = sequence.get_lidar_points(frame_idx, lidar_type=lidar_type)
        labels = sequence.get_lidar_label(frame_idx)

        assert len(points) == len(labels), (
            f"Mismatched point/label counts for frame {frame_idx}"
        )

        if filter_camera_fov:
            points, fov_mask = sequence.filter_points_in_fov(points)
            labels = labels[fov_mask]

        if grid_type == "polar":
            fov_x, _ = sequence.get_fov_angle(
                add_margin=config.get("fov_add_margin", 0.0)
            )
            offset_angle = np.pi if lidar_type == "os1" else 0.0
            angle_range = (-fov_x / 2 + offset_angle, fov_x / 2 + offset_angle)
            n_angle_bins = config.get("n_angle_bins", 15)
            angle_resolution = (angle_range[1] - angle_range[0]) / n_angle_bins
            bev_map, config_data = generate_bev_segmentation_map_polar(
                points,
                labels,
                angle_resolution=angle_resolution,
                angle_range=angle_range,
                radial_resolution=config.get("radial_resolution", 0.25),
                radial_mode=config.get("radial_mode", "linear"),
                radial_growth_rate=config.get("radial_growth_rate", 0.05),
                radial_max=config.get("radial_max", 25.0),
                radial_min=config.get("radial_min", 0.0),
                show_point_counts_hist=show_point_counts_hist,
            )
        else:
            bev_map, config_data = generate_bev_segmentation_map_cartesian(
                points,
                labels,
                x_range=config.get("x_range", (-25.0, 25.0)),
                y_range=config.get("y_range", (-25.0, 25.0)),
                resolution=config.get("resolution", 0.25),
            )

        bev_map_path = output_dir / f"{frame_idx:06d}.npy"
        np.save(bev_map_path, bev_map)

    # dump the config as a json file for reproducibility
    config_path = output_dir / "bev_seg_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)


def visualize_bev_map(
    bev_map: npt.NDArray[np.int32],
    color_map: Optional[dict] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 12),
) -> npt.NDArray[np.uint8]:
    """Visualize BEV segmentation map.

    Args:
        bev_map: BEV segmentation map (H, W) with label IDs
        color_map: Dictionary mapping label IDs to BGR colors
        save_path: Optional path to save visualization
        show: Whether to display using matplotlib
        figsize: Figure size for display

    Returns:
        RGB visualization image (H, W, 3)
    """
    if color_map is None:
        color_map = RELLIS_COLOR_MAP

    # Create color image
    h, w = bev_map.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)

    for label_id, color_bgr in color_map.items():
        mask = bev_map == label_id
        color_image[mask] = color_bgr

    # Convert BGR to RGB for display
    rgb_image = color_image[:, :, ::-1].copy()

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        cv2.imwrite(str(save_path), color_image)  # Save as BGR for OpenCV

    # Show if requested
    if show:
        plt.figure(figsize=figsize)
        plt.imshow(rgb_image)
        plt.title("BEV Segmentation Map")
        plt.xlabel("X (forward)")
        plt.ylabel("Y (left)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    return rgb_image


def load_bev_map(
    data_root: str | Path,
    sequence_id: str,
    grid_type: str,
    frame_idx: int,
) -> npt.NDArray[np.int32]:
    """Load BEV segmentation map from .npy file.

    Args:
        bev_path: Path to .npy file

    Returns:
        BEV segmentation map (H, W) with label IDs
    """
    data_root = Path(data_root) if isinstance(data_root, str) else data_root
    bev_path = (
        data_root
        / sequence_id
        / f"bev_seg_{grid_type}"
        / f"{frame_idx:06d}.npy"
    )
    if not bev_path.exists():
        raise FileNotFoundError(f"BEV map not found: {bev_path}")

    return np.load(bev_path).astype(np.int32)


def load_bev_seg_config(
    data_root: str | Path,
    sequence_id: str,
    grid_type: str,
) -> dict:
    """Load BEV segmentation config from json file.

    Args:
        data_root: Root directory containing sequences
        sequence_id: Sequence ID
        lidar_type: Type of lidar to use, either "os1" or "vel"
    """
    data_root = Path(data_root) if isinstance(data_root, str) else data_root
    config_path = (
        data_root / sequence_id / f"bev_seg_{grid_type}" / "bev_seg_config.json"
    )
    if not config_path.exists():
        raise FileNotFoundError(
            f"BEV segmentation config not found: {config_path}"
        )

    return json.load(open(config_path, "r", encoding="utf-8"))
