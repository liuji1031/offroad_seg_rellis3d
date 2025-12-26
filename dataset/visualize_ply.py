"""Open3D-based utilities to visualize PLY point cloud files.

This module provides simple helpers to load and view PLY files using Open3D.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence
import importlib


def _ensure_paths(paths: Sequence[str | Path]) -> List[Path]:
    validated_paths: List[Path] = []
    for p in paths:
        pp = Path(p)
        if not pp.exists():
            raise FileNotFoundError(f"Path does not exist: {pp}")
        if pp.is_dir():
            raise IsADirectoryError(f"Expected a file, got directory: {pp}")
        if pp.suffix.lower() != ".ply":
            raise ValueError(f"Expected a .ply file, got: {pp}")
        validated_paths.append(pp)
    return validated_paths


def load_ply(
    ply_path: str | Path,
    voxel_size: Optional[float] = None,
    estimate_normals: bool = False,
):
    """Load a .ply file as an Open3D point cloud.

    - voxel_size: if provided (>0), applies voxel downsampling
    - estimate_normals: if True, computes normals for the cloud
    """
    # Import open3d at runtime to avoid hard dependency for static analysis
    try:  # pragma: no cover - visualization dependency
        o3d = importlib.import_module("open3d")  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "open3d is required for visualization. Install with `pip install open3d`."
        ) from exc

    path = _ensure_paths([ply_path])[0]
    pcd = o3d.io.read_point_cloud(str(path))
    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    if estimate_normals:
        pcd.estimate_normals()
    return pcd


def visualize_ply(
    ply_path: str | Path,
    voxel_size: Optional[float] = None,
    estimate_normals: bool = False,
    bg_color: Optional[Sequence[float]] = (0.0, 0.0, 0.0),
    window_name: str = "PLY Viewer",
) -> None:
    """Visualize a single .ply file using Open3D.

    - voxel_size: if provided (>0), applies voxel downsampling
    - estimate_normals: if True, computes normals for the cloud
    - bg_color: background color as (r,g,b) in [0,1]
    - window_name: title of the Open3D window
    """
    pcd = load_ply(
        ply_path, voxel_size=voxel_size, estimate_normals=estimate_normals
    )
    # Import open3d at runtime
    try:  # pragma: no cover - visualization dependency
        o3d = importlib.import_module("open3d")  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "open3d is required for visualization. Install with `pip install open3d`."
        ) from exc

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    try:
        vis.add_geometry(pcd)
        if bg_color is not None:
            opt = vis.get_render_option()
            opt.background_color = bg_color  # type: ignore[assignment]
        vis.run()
    finally:
        vis.destroy_window()


def visualize_multiple_ply(
    ply_paths: Iterable[str | Path],
    voxel_size: Optional[float] = None,
    estimate_normals: bool = False,
    bg_color: Optional[Sequence[float]] = (0.0, 0.0, 0.0),
    window_name: str = "PLY Viewer (Multiple)",
) -> None:
    """Visualize multiple .ply files in a single Open3D window."""
    # Import open3d at runtime
    try:  # pragma: no cover - visualization dependency
        o3d = importlib.import_module("open3d")  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "open3d is required for visualization. Install with `pip install open3d`."
        ) from exc

    paths = _ensure_paths(list(ply_paths))
    clouds = [
        load_ply(p, voxel_size=voxel_size, estimate_normals=estimate_normals)
        for p in paths
    ]
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    try:
        for c in clouds:
            vis.add_geometry(c)
        if bg_color is not None:
            opt = vis.get_render_option()
            opt.background_color = bg_color  # type: ignore[assignment]
        vis.run()
    finally:
        vis.destroy_window()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize one or more .ply point cloud files using Open3D.",
    )
    parser.add_argument("ply", nargs="+", help="Path(s) to .ply file(s)")
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Optional voxel downsample size (e.g., 0.05)",
    )
    parser.add_argument(
        "--normals",
        action="store_true",
        help="Estimate normals for the loaded clouds",
    )
    parser.add_argument(
        "--bg",
        nargs=3,
        type=float,
        metavar=("R", "G", "B"),
        default=(0.0, 0.0, 0.0),
        help="Background color in [0,1] (default: 0 0 0)",
    )

    args = parser.parse_args()
    if len(args.ply) == 1:
        visualize_ply(
            args.ply[0],
            voxel_size=args.voxel_size,
            estimate_normals=bool(args.normals),
            bg_color=tuple(args.bg),
        )
    else:
        visualize_multiple_ply(
            args.ply,
            voxel_size=args.voxel_size,
            estimate_normals=bool(args.normals),
            bg_color=tuple(args.bg),
        )
