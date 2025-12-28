"""BEV Segmentation Visualization Utilities.

Additional visualization functions for BEV segmentation maps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.axes as mpl_axes
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .rellis_classes import (
    RELLIS_CLASS_INDEX_MAP,
    RELLIS_CLASS_INDEX_MAP_INV,
    RELLIS_CLASSES,
    RELLIS_COLOR_LIST_SORTED,
    RELLIS_COLOR_MAP,
)


def visualize_point_cloud_bev(
    points: npt.NDArray[np.float32],
    labels: Optional[npt.NDArray[np.int32]] = None,
    create_new_figure: bool = True,
    point_size: float = 0.5,
    x_range: Tuple[float, float] = (-25.0, 25.0),
    y_range: Tuple[float, float] = (-25.0, 25.0),
    show: bool = False,
):
    if create_new_figure:
        plt.figure()

    if labels is not None:
        colors = (
            np.array(
                [
                    RELLIS_COLOR_MAP[RELLIS_CLASS_INDEX_MAP_INV[label]]
                    for label in labels
                ]
            )
            / 255.0
        )
    else:
        colors = points[:, 2]  # use height instead

    plt.scatter(
        points[:, 0],
        points[:, 1],
        c=colors,
        s=point_size,
        alpha=0.5,
    )
    # invert y axis to match the BEV map
    plt.title("Point Cloud (Top View)")
    plt.xlabel("X (forward)")
    plt.ylabel("Y (left)")
    plt.xlim(x_range[0], x_range[1])
    plt.ylim(y_range[0], y_range[1])
    plt.gca().set_aspect("equal")
    plt.grid(True, alpha=0.3)

    if show:
        plt.show()


def visualize_bev_cartesian(
    bev_map: npt.NDArray[np.int32],
    create_new_figure: bool = True,
    show: bool = True,
) -> None:
    """Visualize BEV map created with cartesian grid."""
    if create_new_figure:
        plt.figure()
    color_image = np.zeros((*bev_map.shape, 3), dtype=np.uint8)
    for label_id, color_bgr in RELLIS_COLOR_MAP.items():
        mask = bev_map == RELLIS_CLASS_INDEX_MAP_INV[label_id]
        if np.any(mask):
            print("found label", label_id)
        color_image[mask] = color_bgr

    plt.imshow(color_image)
    plt.gca().set_aspect("equal")
    plt.title("BEV Segmentation Map")
    plt.xlabel("X (forward)")
    plt.ylabel("Y (left)")
    plt.axis("off")
    plt.gca().invert_yaxis()

    if show:
        plt.show()


def rotate_points(
    points: npt.NDArray[np.float32], rotation: float
) -> npt.NDArray[np.float32]:
    """Rotate points around the origin."""
    points_copy = points.copy()
    points_copy[:, 0] = points[:, 0] * np.cos(rotation) - points[:, 1] * np.sin(
        rotation
    )
    points_copy[:, 1] = points[:, 0] * np.sin(rotation) + points[:, 1] * np.cos(
        rotation
    )
    return points_copy


def get_polar_contour_points(
    r1: float,
    r2: float,
    angle1: float,
    angle2: float,
    n_points_arc: int = 20,
    rotation: float | None = None,
) -> list[tuple[float, float]]:
    """Get the contour points of a polar grid cell."""
    x1, y1 = r1 * np.cos(angle1), r1 * np.sin(angle1)
    x2, y2 = r2 * np.cos(angle1), r2 * np.sin(angle1)
    x3, y3 = r2 * np.cos(angle2), r2 * np.sin(angle2)
    x4, y4 = r1 * np.cos(angle2), r1 * np.sin(angle2)
    points = []
    # line segment 1 from (x1, y1) to (x2, y2)
    points.append((x1, y1))
    points.append((x2, y2))

    # now the arc from (x2, y2) to (x3, y3)
    angles = np.linspace(angle1, angle2, n_points_arc)
    x_arc = r2 * np.cos(angles)
    y_arc = r2 * np.sin(angles)
    points.extend(zip(x_arc.tolist(), y_arc.tolist()))

    # line segment 2 from (x3, y3) to (x4, y4)
    points.append((x3, y3))
    points.append((x4, y4))

    # arc from (x4, y4) to (x1, y1)
    x_arc = r1 * np.cos(angles[::-1])
    y_arc = r1 * np.sin(angles[::-1])
    points.extend(zip(x_arc.tolist(), y_arc.tolist()))

    if rotation is not None:
        for i in range(len(points)):
            x, y = points[i]
            x_new = x * np.cos(rotation) - y * np.sin(rotation)
            y_new = x * np.sin(rotation) + y * np.cos(rotation)
            points[i] = (x_new, y_new)
    return points


def create_polar_cell_patch(
    r1: float,
    r2: float,
    angle1: float,
    angle2: float,
    facecolor: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    input_angle_unit: str = "rad",
    n_points_arc: int = 20,
    rotation: float | None = None,
) -> patches.Polygon:
    """Create a patch that approximates the shape of a polar grid cell."""
    if input_angle_unit == "deg":
        angle1 = np.deg2rad(angle1)
        angle2 = np.deg2rad(angle2)
    elif input_angle_unit != "rad":
        raise ValueError(f"Unsupported angle unit '{input_angle_unit}'")

    points = get_polar_contour_points(
        r1, r2, angle1, angle2, n_points_arc, rotation
    )

    # now create the matplotlib patch
    patch = patches.Polygon(
        points, closed=True, fill=True, facecolor=facecolor, edgecolor="none"
    )
    return patch


def visualize_bev_polar(
    bev_map: npt.NDArray[np.int32 | np.float32],
    bev_config: dict,
    create_new_figure: bool = True,
    show: bool = True,
    rotation: float | None = None,
) -> None:
    """Visualize BEV map created with polar grid.

    Args:
        bev_map: BEV segmentation map (num_angle_bins, num_radial_bins) with label IDs
        bev_config: Configuration dictionary for the BEV map. The keys include the following:
            - angle_edges: (num_angle_bins,) array of angle edges
            - radial_edges: (num_radial_bins,) array of radial edges
        create_new_figure: Whether to create a new figure
        show: Whether to display the figure
    """
    if create_new_figure:
        plt.figure()
    ax = plt.gca()
    assert "angle_edges" in bev_config, "angle_edges must be in the bev_config"
    assert "radial_edges" in bev_config, (
        "radial_edges must be in the bev_config"
    )
    angle_edges = bev_config["angle_edges"]
    radial_edges = bev_config["radial_edges"]

    # for each grid cell, we need to create a patch that approximate the shape of the
    # grid cell.

    # keep track of the range of x and y values for the xlim and ylim of the
    # plot
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf
    for i_angle in range(bev_map.shape[0]):
        for i_radial in range(bev_map.shape[1]):
            angle_min = angle_edges[i_angle]
            angle_max = angle_edges[i_angle + 1]
            radial_min = radial_edges[i_radial]
            radial_max = radial_edges[i_radial + 1]
            label = bev_map[i_angle, i_radial]
            facecolor = (
                np.array(RELLIS_COLOR_MAP[RELLIS_CLASS_INDEX_MAP_INV[label]])
                / 255.0
            ).tolist()
            patch = create_polar_cell_patch(
                radial_min,
                radial_max,
                angle_min,
                angle_max,
                facecolor=facecolor,
                input_angle_unit="rad",
                rotation=rotation,
            )
            ax.add_patch(patch)

            xy = patch.get_xy()
            x_min = np.minimum(x_min, np.min(xy[:, 0]))
            x_max = np.maximum(x_max, np.max(xy[:, 0]))
            y_min = np.minimum(y_min, np.min(xy[:, 1]))
            y_max = np.maximum(y_max, np.max(xy[:, 1]))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_title("Polar BEV Segmentation Map")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # turn off the top and right axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show:
        plt.show()


def plot_point_cloud_bev(
    points: npt.NDArray[np.float32],
    labels: Optional[npt.NDArray[np.int32]] = None,
    x_range: Tuple[float, float] = (-25.0, 25.0),
    y_range: Tuple[float, float] = (-25.0, 25.0),
    point_size: float = 0.5,
    axes: mpl_axes.Axes | None = None,
    rotation: float | None = None,
    title: str | None = None,
):
    if axes is None:
        axes = plt.gca()
    if rotation is not None:
        points = rotate_points(points, rotation)
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Filter points in range
    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] < y_max)
    )
    points_filtered = points[mask]

    if labels is not None:
        labels_filtered = labels[mask]
        # Color points by label
        scatter_colors = np.zeros((len(points_filtered), 3), dtype=np.float32)
        for label_id, color_bgr in RELLIS_COLOR_MAP.items():
            mask_label = labels_filtered == RELLIS_CLASS_INDEX_MAP[label_id]
            scatter_colors[mask_label] = (
                np.array(color_bgr) / 255.0
            )  # normalize
        axes.scatter(
            points_filtered[:, 0],
            points_filtered[:, 1],
            c=scatter_colors,
            s=point_size,
            alpha=0.5,
        )
    else:
        axes.scatter(
            points_filtered[:, 0],
            points_filtered[:, 1],
            c=points_filtered[:, 2],
            cmap="viridis",
            s=point_size,
            alpha=0.5,
        )
    axes.set_xlim(-9, 9)
    axes.set_xlabel("X (forward)")
    axes.set_ylabel("Y (left)")
    axes.set_title("Point Cloud (Top View)" if title is None else title)
    axes.set_aspect("equal")
    axes.grid(True, alpha=0.3)

    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)


def visualize_bev_with_point_cloud(
    bev_map: npt.NDArray[np.int32],
    points: npt.NDArray[np.float32],
    labels: Optional[npt.NDArray[np.int32]] = None,
    x_range: Tuple[float, float] = (-25.0, 25.0),
    y_range: Tuple[float, float] = (-25.0, 25.0),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (20, 10),
    point_size: float = 0.5,
) -> None:
    """Visualize BEV map alongside point cloud projection.

    Args:
        bev_map: BEV segmentation map (H, W) with label IDs
        points: Nx3 array of points (x, y, z)
        labels: Optional N array of point labels for coloring
        x_range: (x_min, x_max) in meters
        y_range: (y_min, y_max) in meters
        save_path: Optional path to save visualization
        show: Whether to display using matplotlib
        figsize: Figure size for display
        point_size: Size of points to draw
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: BEV map
    color_image = np.zeros((*bev_map.shape, 3), dtype=np.uint8)
    for label_id, color_bgr in RELLIS_COLOR_MAP.items():
        mask = bev_map == RELLIS_CLASS_INDEX_MAP_INV[label_id]
        color_image[mask] = color_bgr
    rgb_bev = color_image[:, :, ::-1]
    axes[0].imshow(rgb_bev)
    axes[0].set_title("BEV Segmentation Map")
    axes[0].set_xlabel("X (forward)")
    axes[0].set_ylabel("Y (left)")
    axes[0].set_aspect("equal")
    axes[0].invert_yaxis()
    axes[0].axis("off")

    # Right: Point cloud projection
    plot_point_cloud_bev(points, labels, x_range, y_range, point_size, axes[1])

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def create_label_legend(
    draw_on_axes: mpl_axes.Axes | None = None,
    color_map: Optional[dict | list] = RELLIS_COLOR_LIST_SORTED,
    class_names: Optional[dict] = RELLIS_CLASSES,
    figsize: Tuple[int, int] = (6, 12),
) -> None:
    """Create a legend showing all RELLIS classes with their colors.

    Args:
        color_map: Dictionary mapping label IDs to RGB colors
        class_names: Dictionary mapping label IDs to class names
        figsize: Figure size
    """
    if color_map is None:
        color_map = RELLIS_COLOR_LIST_SORTED  # tuple of sorted (id, color) by class abundance
    if class_names is None:
        class_names = RELLIS_CLASSES

    if draw_on_axes is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
    else:
        ax = draw_on_axes
        fig = plt.gcf()

    y_pos = 0
    for label_id, color_rgb in color_map[::-1]:
        # Draw color patch
        rect = patches.Rectangle(
            (0, y_pos), 0.3, 0.8, facecolor=np.array(color_rgb) / 255.0
        )
        ax.add_patch(rect)

        # Add label text
        class_name = class_names.get(label_id, f"Class {label_id}")
        ax.text(
            0.35,
            y_pos + 0.4,
            f"{class_name}",
            verticalalignment="center",
            fontsize=10,
        )

        y_pos += 1.0

    ax.set_xlim(0, 5)
    ax.set_ylim(0, y_pos)
    # ax.set_title("RELLIS-3D Class Legend", fontsize=12, fontweight="bold")
    ax.axis("off")

    fig.tight_layout()
    if draw_on_axes is None:
        plt.show()


def compare_bev_maps(
    bev_map1: npt.NDArray[np.int32],
    bev_map2: npt.NDArray[np.int32],
    title1: str = "Map 1",
    title2: str = "Map 2",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (20, 10),
) -> None:
    """Compare two BEV maps side by side.

    Args:
        bev_map1: First BEV map (H, W)
        bev_map2: Second BEV map (H, W)
        title1: Title for first map
        title2: Title for second map
        save_path: Optional path to save visualization
        show: Whether to display using matplotlib
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, bev_map, title in zip(axes, [bev_map1, bev_map2], [title1, title2]):
        color_image = np.zeros((*bev_map.shape, 3), dtype=np.uint8)
        for label_id, color_bgr in RELLIS_COLOR_MAP.items():
            mask = bev_map == RELLIS_CLASS_INDEX_MAP_INV[label_id]
            color_image[mask] = color_bgr
        rgb_image = color_image[:, :, ::-1]

        ax.imshow(rgb_image)
        ax.set_title(title)
        ax.set_xlabel("X (forward)")
        ax.set_ylabel("Y (left)")
        ax.axis("off")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
