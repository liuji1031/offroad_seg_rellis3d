"""PyTorch Dataset for RELLIS-3D BEV Segmentation Training.

Loads multi-modal data (images + point clouds) and BEV segmentation targets
for training the RellisPolarBevFusion model.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
import torch
from einops import rearrange
from torch.utils.data import DataLoader, Dataset

from .bev_segmentation import load_bev_map
from .rellis_bev_cache import DataCache
from .rellis_classes import RELLIS_CLASS_INDEX_MAP
from .rellis_sequence import RellisSequence, get_rellis_sequence

__all__ = [
    "RellisBEVDataset",
    "rellis_bev_collate_fn",
    "create_rellis_dataloader",
]


def preprocess_image(
    image: np.ndarray,
    target_image_size: Tuple[int, int] = (640, 400),
    input_image_bgr: bool = True,
) -> npt.NDArray[np.float32]:
    # Downsample (original: 1920x1200 -> target: 640x400)
    image = cv2.resize(
        image,
        target_image_size,  # (width, height)
        interpolation=cv2.INTER_LINEAR,
    )

    # BGR to RGB
    if input_image_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    # put channel dimension first and add batch dimension
    image = rearrange(image, "h w c -> 1 c h w")

    return image


def load_sample_list(file_path: str):
    list_file_path = Path(file_path)
    if not list_file_path.exists():
        raise FileNotFoundError(f"List file not found: {list_file_path}")

    samples = []
    with open(list_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid line format: {line}. Expected 'sequence_id frame_idx'"
                )
            seq_id, frame_idx = parts[0], int(parts[1])
            samples.append((seq_id, frame_idx))

    return samples


class RellisBEVDataset(Dataset):
    """Dataset for RELLIS-3D BEV segmentation training.

    Loads images, point clouds, camera parameters, and BEV targets from
    sequences 00000-00004.

    Args:
        data_root: Root directory containing RELLIS-3D sequences
        list_file_path: Path to file with format "sequence_id frame_idx" per line
        mode: Dataset mode ('train', 'val', or 'test')
        target_image_size: Target size for downsampled images (width, height)
        grid_type: BEV grid type ('polar' or 'cartesian')
        num_classes: Number of segmentation classes (default: 20)
        lidar_type: LiDAR type ('os1' or 'vel')
        filter_fov: Whether to filter point clouds to camera FOV
        max_samples: Maximum number of samples to use (None = use all).
                     Useful for overfitting tests on small datasets.
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        list_file_path: Union[str, Path],
        mode: str = "train",
        target_image_size: Tuple[int, int] = (640, 400),
        grid_type: str = "polar",
        num_classes: int = 20,
        lidar_type: str = "os1",
        filter_fov: bool = True,
        use_cache: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
        overwrite_cache: bool = False,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.mode = mode
        self.target_image_size = target_image_size  # (width, height)
        self.grid_type = grid_type
        self.num_classes = num_classes
        self.lidar_type = lidar_type
        self.filter_fov = filter_fov
        assert mode in ["train", "val", "test"], f"Invalid mode: {mode}"
        assert grid_type in ["polar", "cartesian"], (
            f"Invalid grid_type: {grid_type}"
        )

        # Load sample list
        self.samples = self._load_sample_list(list_file_path)

        # Limit number of samples if specified (useful for overfitting tests)
        if max_samples is not None and max_samples > 0:
            original_len = len(self.samples)
            self.samples = self.samples[:max_samples]
            print(
                f"[Dataset] Limited samples from {original_len} to {len(self.samples)} (max_samples={max_samples})"
            )

        # Initialize sequences (cache them for efficiency)
        self.sequences: Dict[str, RellisSequence] = {}
        sequence_ids = set(seq_id for seq_id, _ in self.samples)
        for seq_id in sequence_ids:
            self.sequences[seq_id] = get_rellis_sequence(
                self.data_root, seq_id, load_vel2os1=False
            )

        # Compute image scaling factors for camera intrinsics
        # Original image size: 1920x1200
        self.scale_x = self.target_image_size[0] / 1920.0  # width scale
        self.scale_y = self.target_image_size[1] / 1200.0  # height scale

        # Initialize cache
        if cache_dir is None:
            cache_dir = (
                self.data_root
                / "cache"
                / f"{mode}_{grid_type}_{target_image_size[0]}x{target_image_size[1]}"
            )
        self.cache = DataCache(
            cache_dir=cache_dir,
            enabled=use_cache,
            overwrite=overwrite_cache,
        )

        # Save cache configuration
        if use_cache:
            cache_config = {
                "mode": mode,
                "target_image_size": target_image_size,
                "grid_type": grid_type,
                "num_classes": num_classes,
                "lidar_type": lidar_type,
                "filter_fov": filter_fov,
            }
            self.cache.save_config(cache_config)

    def _load_sample_list(
        self, list_file_path: Union[str, Path]
    ) -> List[Tuple[str, int]]:
        """Load sample list from file.

        Expected format: "sequence_id frame_idx" per line
        Example: "00000 308"

        Args:
            list_file_path: Path to list file

        Returns:
            List of (sequence_id, frame_idx) tuples
        """
        return load_sample_list(str(list_file_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        """Load a single sample.

        Returns:
            Dictionary containing:
                - points: [N, 4] point cloud (x, y, z, intensity)
                - images: [1, 3, H, W] camera image
                - camera_params: dict with camera parameters
                - targets: [num_classes, H_bev, W_bev] one-hot BEV targets
                - metadata: dict with sequence_id and frame_idx
        """
        seq_id, frame_idx = self.samples[idx]
        sequence = self.sequences[seq_id]

        # Load image (with caching)
        image = self._load_image(sequence, frame_idx, seq_id)

        # Load point cloud (with caching)
        points = self._load_points(sequence, frame_idx, seq_id)

        # Load BEV target (with caching)
        target = self._load_bev_target(seq_id, frame_idx)

        # Get camera parameters
        camera_params = self._get_camera_params(sequence)

        return {
            "points": points,
            "images": image,
            "camera_params": camera_params,
            "targets": target,
            "metadata": {
                "sequence_id": seq_id,
                "frame_idx": frame_idx,
            },
        }

    def _load_image(
        self, sequence: RellisSequence, frame_idx: int, seq_id: str
    ) -> torch.Tensor:
        """Load and preprocess image (with caching).

        Pipeline:
        1. Check cache first
        2. If not cached: Load via cv2.imread()
        3. Downsample to target size
        4. Convert BGR to RGB
        5. Normalize to [0, 1]
        6. Convert to torch tensor [1, 3, H, W]
        7. Save to cache

        Args:
            sequence: RellisSequence object
            frame_idx: Frame index
            seq_id: Sequence ID

        Returns:
            Image tensor [1, 3, H, W]
        """
        # Try loading from cache
        cached_image = self.cache.load_image(seq_id, frame_idx)
        if cached_image is not None:
            return cached_image

        # Load and preprocess image
        img_path = sequence.get_image_path(frame_idx)
        image = cv2.imread(str(img_path))

        if image is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")

        image = preprocess_image(image, self.target_image_size)
        image = torch.from_numpy(image)

        # Save to cache
        self.cache.save_image(seq_id, frame_idx, image)

        return image

    def _load_points(
        self, sequence: RellisSequence, frame_idx: int, seq_id: str
    ) -> torch.Tensor:
        """Load and preprocess point cloud (with caching).

        Pipeline:
        1. Check cache first
        2. If not cached: Load points from .bin file
        3. Filter to camera FOV (if enabled)
        4. Convert to torch tensor [N, 4]
        5. Save to cache

        Args:
            sequence: RellisSequence object
            frame_idx: Frame index
            seq_id: Sequence ID

        Returns:
            Point cloud tensor [N, 4] (x, y, z, intensity)
        """
        # Try loading from cache
        cached_points = self.cache.load_points(seq_id, frame_idx)
        if cached_points is not None:
            return cached_points

        # Load and preprocess points
        points = sequence.get_lidar_points(
            frame_idx, lidar_type=self.lidar_type
        )

        # Filter to camera FOV if enabled
        if self.filter_fov:
            points, _ = sequence.filter_points_in_fov(points)

        # Convert to torch tensor
        points = torch.from_numpy(points).float()

        # Save to cache
        self.cache.save_points(seq_id, frame_idx, points)

        return points

    def _load_bev_target(
        self, seq_id: str, frame_idx: int, transpose: bool = True
    ) -> torch.Tensor:
        """Load and preprocess BEV segmentation target (with caching).

        Pipeline:
        1. Check cache first
        2. If not cached: Load BEV map from .npy file [H_bev, W_bev]
        3. Convert class indices to one-hot encoding [num_classes, H_bev, W_bev]
        4. Save to cache

        Args:
            seq_id: Sequence ID
            frame_idx: Frame index

        Returns:
            One-hot encoded BEV target [num_classes, H_bev, W_bev]
        """
        # Try loading from cache
        cached_target = self.cache.load_target(seq_id, frame_idx)
        if cached_target is not None:
            return cached_target

        # Load BEV map (class indices)
        bev_map = load_bev_map(
            self.data_root, seq_id, self.grid_type, frame_idx
        )
        if transpose:
            bev_map = bev_map.T

        # Convert to one-hot encoding
        # bev_map: [H_bev, W_bev] with class indices 0-19
        target = self._to_one_hot(bev_map, self.num_classes)

        # Save to cache
        self.cache.save_target(seq_id, frame_idx, target)

        return target

    def _to_one_hot(self, labels: np.ndarray, num_classes: int) -> torch.Tensor:
        """Convert class indices to one-hot encoding.

        Args:
            labels: [H, W] array with class indices
            num_classes: Number of classes

        Returns:
            One-hot tensor [num_classes, H, W]
        """
        H, W = labels.shape
        one_hot = np.zeros((num_classes, H, W), dtype=np.float32)

        for c in range(num_classes):
            one_hot[c] = (labels == c).astype(np.float32)

        return torch.from_numpy(one_hot)

    def _get_camera_params(
        self, sequence: RellisSequence
    ) -> Dict[str, torch.Tensor]:
        """Extract camera parameters with scaling for downsampled images.

        Args:
            sequence: RellisSequence object

        Returns:
            Dictionary with:
                - camera2lidar_rots: [1, 3, 3] rotation matrix
                - camera2lidar_trans: [1, 3] translation vector
                - camera_intrinsics: [1, 3, 3] scaled intrinsic matrix
        """
        # Get rotation and translation (cam2lidar)
        R_cam2lidar = sequence.R_cam2lidar  # [3, 3]
        t_cam2lidar = sequence.t_cam2lidar  # [3]

        # Get camera intrinsics and scale for downsampled image
        K = sequence.camera_intrinsics.copy()  # [3, 3]

        # Convert to torch tensors and add batch dimension
        camera_params = {
            "camera2lidar_rots": torch.from_numpy(R_cam2lidar)
            .float()
            .unsqueeze(0),  # [1, 3, 3]
            "camera2lidar_trans": torch.from_numpy(t_cam2lidar)
            .float()
            .unsqueeze(0),  # [1, 3]
            "camera_intrinsics": torch.from_numpy(K)
            .float()
            .unsqueeze(0),  # [1, 3, 3]
        }

        return camera_params

    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics:
                - num_images: Number of cached images
                - num_points: Number of cached point clouds
                - num_targets: Number of cached targets
                - total_size_mb: Total cache size in MB
                - cache_enabled: Whether caching is enabled
        """
        stats = self.cache.get_cache_size()
        stats["cache_enabled"] = self.cache.enabled
        return stats  # type: ignore


def rellis_bev_collate_fn(
    batch: List[Dict],
) -> Dict[str, Union[torch.Tensor, List, Dict]]:
    """Custom collate function for batching samples with variable-size point clouds.

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched dictionary with:
            - points: List[torch.Tensor] of [N_i, 4] point clouds
            - images: [B, 1, 3, H, W] stacked images
            - camera_params: dict with batched camera parameters
            - targets: [B, num_classes, H_bev, W_bev] stacked targets
            - metadata: List of metadata dicts
    """
    # Stack images
    images = torch.stack(
        [sample["images"] for sample in batch], dim=0
    )  # [B, 1, 3, H, W]

    # Keep points as list (variable size)
    points = [sample["points"] for sample in batch]

    # Stack targets
    targets = torch.stack(
        [sample["targets"] for sample in batch], dim=0
    )  # [B, C, H, W]

    # Batch camera parameters
    camera_params = {
        "camera2lidar_rots": torch.stack(
            [sample["camera_params"]["camera2lidar_rots"] for sample in batch],
            dim=0,
        ),  # [B, 1, 3, 3]
        "camera2lidar_trans": torch.stack(
            [sample["camera_params"]["camera2lidar_trans"] for sample in batch],
            dim=0,
        ),  # [B, 1, 3]
        "camera_intrinsics": torch.stack(
            [sample["camera_params"]["camera_intrinsics"] for sample in batch],
            dim=0,
        ),  # [B, 1, 3, 3]
    }

    # Collect metadata
    metadata = [sample["metadata"] for sample in batch]

    return {
        "points": points,
        "images": images,
        "camera_params": camera_params,
        "targets": targets,
        "metadata": metadata,
    }


def create_rellis_dataloader(
    data_root: Union[str, Path],
    list_file_path: Union[str, Path],
    batch_size: int = 4,
    mode: str = "train",
    shuffle: bool = True,
    num_workers: int = 4,
    target_image_size: Tuple[int, int] = (640, 400),
    grid_type: str = "polar",
    num_classes: int = 20,
    lidar_type: str = "os1",
    filter_fov: bool = True,
    pin_memory: bool = True,
    use_cache: bool = True,
    cache_dir: Optional[Union[str, Path]] = None,
    overwrite_cache: bool = False,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """Create DataLoader for RELLIS-3D BEV segmentation.

    Args:
        data_root: Root directory containing RELLIS-3D sequences
        list_file_path: Path to file with "sequence_id frame_idx" per line
        batch_size: Batch size
        mode: Dataset mode ('train', 'val', or 'test')
        shuffle: Whether to shuffle the dataset
        num_workers: Number of data loading workers
        target_image_size: Target size for downsampled images (width, height)
        grid_type: BEV grid type ('polar' or 'cartesian')
        num_classes: Number of segmentation classes
        lidar_type: LiDAR type ('os1' or 'vel')
        filter_fov: Whether to filter point clouds to camera FOV
        pin_memory: Whether to pin memory for faster GPU transfer
        use_cache: Whether to enable disk caching (default: True)
        cache_dir: Custom cache directory (default: data_root/cache/...)
        overwrite_cache: Whether to overwrite existing cache
        max_samples: Maximum number of samples to use (None = use all). Useful for overfitting tests.

    Returns:
        Configured DataLoader
    """
    dataset = RellisBEVDataset(
        data_root=data_root,
        list_file_path=list_file_path,
        mode=mode,
        target_image_size=target_image_size,
        grid_type=grid_type,
        num_classes=num_classes,
        lidar_type=lidar_type,
        filter_fov=filter_fov,
        use_cache=use_cache,
        cache_dir=cache_dir,
        overwrite_cache=overwrite_cache,
        max_samples=max_samples,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=rellis_bev_collate_fn,
        pin_memory=pin_memory,
        drop_last=(mode == "train"),  # Drop last incomplete batch in training
    )

    return dataloader
