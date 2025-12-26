import json
import sys
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import matplotlib.gridspec as gridspec
from torch import nn

sys.path.append(str(Path(__file__).parent.parent))

from offroad_det_seg_rellis.dataset.bev_visualization import (
    get_polar_contour_points,
    plot_point_cloud_bev,
)
from offroad_det_seg_rellis.dataset import (
    RellisSequence,
    visualize_bev_polar,
    create_label_legend,
)

from offroad_det_seg_rellis.dataset.rellis_classes import (
    RELLIS_COLOR_MAP,
    RELLIS_CLASS_INDEX_MAP_INV,
)

from offroad_det_seg_rellis.dataset.rellis_sequence import (
    get_rellis_sequence_by_id,
)
from offroad_det_seg_rellis.inference import RellisPolarBevFusionInference

_path = Path(__file__).parent
BEV_SEG_CONFIG_PATH = _path / "bev_seg_config.json"
GROUND_Z = -1.0  # rough estimate of the ground height in lidar frame (os1)


def load_bev_seg_config(bev_seg_config_path: str = str(BEV_SEG_CONFIG_PATH)):
    """Load BEV segmentation configuration."""
    with open(bev_seg_config_path, "r", encoding="utf-8") as f:
        bev_seg_config = json.load(f)
    return bev_seg_config


def _apply_transform_to_points(
    points: np.ndarray, cam_intrins: np.ndarray, RT_lidar2cam: np.ndarray
):
    """Apply transformation to points."""
    points = np.concatenate(
        [points, np.ones((points.shape[0], 1))], axis=1
    )  # append 1 to form homogeneous coordinates
    points = np.linalg.matmul(
        points, RT_lidar2cam.T
    )  # each row is now in cam frame
    points = np.linalg.matmul(points[:, :3], cam_intrins.T)
    # divide by the scale
    points = points[:, :2] / points[:, [2]]  # now we get the pixel coordinates
    points = points.astype(np.int32)  # convert to int
    return points


def plot_bev_seg_polar_on_image(
    image: np.ndarray,
    bev_seg: np.ndarray,
    cam_intrins: np.ndarray,
    RT_lidar2cam: np.ndarray,
    bev_config: dict = load_bev_seg_config(),
    alpha: float = 0.5,
    plot_on_blank: bool = False,
):
    """Plot BEV segmentation (polar) on image.

    Args:
        image: np.ndarray, the image to plot on
        bev_seg: np.ndarray, the BEV segmentation map. an array with the label ids of the polar grid cells. shape: (n_angle_bins, n_radial_bins)
        cam_intrins: np.ndarray, the camera intrinsic matrix
        RT_lidar2cam: np.ndarray, the transformation from lidar to camera
        bev_config: dict, the BEV segmentation configuration

    Returns:
    """
    n_radial_bins = len(bev_config["radial_edges"]) - 1
    n_angle_bins = len(bev_config["angle_edges"]) - 1
    points_by_patch = [
        [None for _ in range(n_radial_bins)] for _ in range(n_angle_bins)
    ]
    for i_angle in range(len(bev_config["angle_edges"]) - 1):
        for i_radial in range(len(bev_config["radial_edges"]) - 1):
            r1 = bev_config["radial_edges"][i_radial]
            r2 = bev_config["radial_edges"][i_radial + 1]
            angle1 = bev_config["angle_edges"][i_angle]
            angle2 = bev_config["angle_edges"][i_angle + 1]
            points = get_polar_contour_points(r1, r2, angle1, angle2)
            points = np.array(points)
            # append a z coordinate determined from e.g., 5 percentile of a sample data of point clouds
            points = np.concatenate(
                [points, np.full((points.shape[0], 1), GROUND_Z)], axis=1
            )
            points = _apply_transform_to_points(
                points, cam_intrins, RT_lidar2cam
            )
            points_by_patch[i_angle][i_radial] = points  # type: ignore
    if plot_on_blank:
        img_plot = np.ones_like(image, dtype=np.uint8) * 255
    else:
        img_plot = image.copy()
    for angle_bin in range(bev_seg.shape[0]):
        for radial_bin in range(bev_seg.shape[1]):
            label = bev_seg[angle_bin, radial_bin].item()
            if label == 0:
                continue
            color = RELLIS_COLOR_MAP[RELLIS_CLASS_INDEX_MAP_INV[label]]
            color = [color[2], color[1], color[0]]  # flip from rgb to bgr
            points = points_by_patch[angle_bin][radial_bin]

            # Filter out invalid points (outside image bounds or invalid coordinates)
            # h, w = image.shape[:2]
            # valid_mask = (
            #     (points[:, 0] >= 0) & (points[:, 0] < w) &
            #     (points[:, 1] >= 0) & (points[:, 1] < h)
            # )
            # points = points[valid_mask]

            # Skip if no valid points remain
            if points is None or points.size == 0:
                continue

            # Ensure int32 dtype
            points = points.astype(np.int32)

            img_plot_0 = img_plot.copy()
            # plot the patch on the image with the color and alpha
            cv2.fillPoly(img_plot, [points[:, np.newaxis, :]], color)  # type: ignore

            # blend with alpha
            img_plot = cv2.addWeighted(
                img_plot_0, 1 - alpha, img_plot, alpha, 0
            )

    return img_plot


def plot_prediction(
    inference: RellisPolarBevFusionInference,
    seq_id: str,
    frame_idx: int,
    new_figure: bool = True,
    fig_size: tuple[int, int] = (10, 8),
    show_fig: bool = True,
):
    sequence = get_rellis_sequence_by_id(seq_id)

    bev_seg_config = sequence.load_bev_seg_config()
    image = sequence.get_image(frame_idx)
    points = sequence.get_lidar_points(frame_idx)
    points_in_fov, mask = sequence.filter_points_in_fov(points)
    labels = sequence.get_lidar_label(frame_idx)
    labels_in_fov = labels[mask]
    bev_seg_gt = sequence.load_bev_seg_gt(frame_idx)
    bev_seg_pred = inference.predict(points, image).T

    if new_figure:
        fig = plt.figure(figsize=fig_size)
    else:
        fig = plt.gcf()
        # clear the figure
        fig.clear()
    gs = gridspec.GridSpec(nrows=2, ncols=3, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_point_cloud_bev(
        points=points_in_fov,
        labels=labels_in_fov,
        x_range=(-25.0, 25.0),
        y_range=(0.0, 25.0),
        axes=ax1,
        rotation=np.deg2rad(-90),
    )

    ax2 = fig.add_subplot(gs[0, 1:3])
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Camera")

    ax3 = fig.add_subplot(gs[1, 0])
    visualize_bev_polar(
        bev_map=bev_seg_gt,
        bev_config=bev_seg_config,
        create_new_figure=False,
        show=False,
        rotation=np.deg2rad(-90),
    )
    plt.title("Ground Truth")

    ax4 = fig.add_subplot(gs[1, 1])
    visualize_bev_polar(
        bev_map=bev_seg_pred,
        bev_config=bev_seg_config,
        create_new_figure=False,
        show=False,
        rotation=np.deg2rad(-90),
    )
    plt.title("Prediction")

    ax5 = fig.add_subplot(gs[1, 2])
    create_label_legend(
        draw_on_axes=plt.gca(),
    )
    fig.tight_layout()

    if show_fig:
        fig.show()

    return fig


def make_prediction_movie(
    inference: RellisPolarBevFusionInference,
    seq_id: str,
    frame_idx_list: list[int],
    out_dir: str,
    fig_size: tuple[int, int] = (10, 8),
    frame_rate: int = 30,
):
    out_path = Path(out_dir)
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)
    out_file_path = (
        out_path / f"{seq_id}_{frame_idx_list[0]}-{frame_idx_list[-1]}.mp4"
    )

    fig = plt.figure(figsize=fig_size)
    canvas = FigureCanvas(fig)
    out_vid: None | cv2.VideoWriter = None
    total_frames = len(frame_idx_list)
    for i, frame_idx in enumerate(frame_idx_list):
        progress = (i + 1) / total_frames * 100
        print(f"Processing frame {i+1}/{total_frames} ({progress:.1f}%): frame_idx={frame_idx}")
        plot_prediction(
            inference,
            seq_id,
            frame_idx,
            new_figure=False,
            fig_size=fig_size,
            show_fig=False,
        )
        canvas.draw()
        width, height = map(int, canvas.get_width_height())

        if out_vid is None:
            out_vid = cv2.VideoWriter(
                str(out_file_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                frame_rate,
                (width, height),
            )
            if not out_vid.isOpened():
                raise ValueError(f"Failed to open video writer: {out_file_path}")
        
        frame = np.asarray(canvas.buffer_rgba())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        out_vid.write(frame)
    if out_vid is not None:
        out_vid.release()
    
    plt.close(fig)

    print(f"Saved video to {out_file_path}")
    return out_file_path


def change_video_frame_rate(
    folder_path: str,
    new_frame_rate: float,
    overwrite: bool = False,
    output_suffix: str = "_fps_changed",
):
    """Change the frame rate of all .mp4 files in a folder.

    Args:
        folder_path: Path to the folder containing .mp4 files
        new_frame_rate: The new frame rate (frames per second) to apply
        overwrite: If True, overwrite the original files. If False, create new files
                  with the output_suffix appended to the filename (before .mp4)
        output_suffix: Suffix to append to output filenames when overwrite=False

    Returns:
        List of paths to the processed video files
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")

    # Find all .mp4 files
    mp4_files = sorted(folder.glob("*.mp4"))
    if not mp4_files:
        print(f"No .mp4 files found in {folder_path}")
        return []

    processed_files = []
    total_files = len(mp4_files)

    for idx, video_path in enumerate(mp4_files, 1):
        print(f"Processing [{idx}/{total_files}]: {video_path.name}")

        # Open input video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  Warning: Could not open {video_path.name}, skipping...")
            continue

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

        print(f"  Original: {width}x{height} @ {original_fps:.2f} fps")
        print(f"  New: {width}x{height} @ {new_frame_rate:.2f} fps")

        # Determine output path
        if overwrite:
            output_path = video_path
            # Create temporary file first to avoid overwriting while reading
            temp_path = video_path.with_suffix(".tmp.mp4")
            final_output_path = temp_path
        else:
            output_path = video_path.with_name(
                f"{video_path.stem}{output_suffix}.mp4"
            )
            final_output_path = output_path

        # Create output video writer
        out = cv2.VideoWriter(
            str(final_output_path),
            fourcc,
            new_frame_rate,
            (width, height),
        )

        if not out.isOpened():
            print(f"  Warning: Could not create output video {final_output_path}, skipping...")
            cap.release()
            continue

        # Read and write frames
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_count += 1

        # Release resources
        cap.release()
        out.release()

        # If overwriting, replace original with temporary file
        if overwrite:
            video_path.unlink()  # Delete original
            temp_path.rename(video_path)  # Rename temp to original
            processed_files.append(video_path)
            print(f"  ✓ Overwritten: {video_path.name} ({frame_count} frames)")
        else:
            processed_files.append(output_path)
            print(f"  ✓ Created: {output_path.name} ({frame_count} frames)")

    print(f"\nProcessed {len(processed_files)} video file(s)")
    return processed_files
