"""Caching utilities for RELLIS-3D BEV dataset.

Implements disk-based caching for preprocessed data to speed up training.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

__all__ = ["DataCache"]


class DataCache:
    """Disk-based cache for preprocessed dataset samples.

    Caches preprocessed images, point clouds, and BEV targets to disk
    to avoid redundant preprocessing in each epoch.

    Args:
        cache_dir: Directory to store cached data
        enabled: Whether caching is enabled
        overwrite: Whether to overwrite existing cache

    Cache Structure:
        cache_dir/
        ├── cache_config.json       # Cache configuration
        ├── images/
        │   └── {seq_id}_{frame_idx}.pt
        ├── points/
        │   └── {seq_id}_{frame_idx}.pt
        └── targets/
            └── {seq_id}_{frame_idx}.pt
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        enabled: bool = True,
        overwrite: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.overwrite = overwrite

        if self.enabled:
            self._initialize_cache()

    def _initialize_cache(self):
        """Initialize cache directory structure."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.images_dir = self.cache_dir / "images"
        self.points_dir = self.cache_dir / "points"
        self.targets_dir = self.cache_dir / "targets"

        self.images_dir.mkdir(exist_ok=True)
        self.points_dir.mkdir(exist_ok=True)
        self.targets_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, seq_id: str, frame_idx: int) -> str:
        """Generate cache key for a sample."""
        return f"{seq_id}_{frame_idx:06d}"

    def _get_image_path(self, seq_id: str, frame_idx: int) -> Path:
        """Get cache path for image."""
        key = self._get_cache_key(seq_id, frame_idx)
        return self.images_dir / f"{key}.pt"

    def _get_points_path(self, seq_id: str, frame_idx: int) -> Path:
        """Get cache path for point cloud."""
        key = self._get_cache_key(seq_id, frame_idx)
        return self.points_dir / f"{key}.pt"

    def _get_target_path(self, seq_id: str, frame_idx: int) -> Path:
        """Get cache path for target."""
        key = self._get_cache_key(seq_id, frame_idx)
        return self.targets_dir / f"{key}.pt"

    def has_image(self, seq_id: str, frame_idx: int) -> bool:
        """Check if preprocessed image exists in cache."""
        if not self.enabled:
            return False
        return self._get_image_path(seq_id, frame_idx).exists()

    def has_points(self, seq_id: str, frame_idx: int) -> bool:
        """Check if preprocessed point cloud exists in cache."""
        if not self.enabled:
            return False
        return self._get_points_path(seq_id, frame_idx).exists()

    def has_target(self, seq_id: str, frame_idx: int) -> bool:
        """Check if preprocessed target exists in cache."""
        if not self.enabled:
            return False
        return self._get_target_path(seq_id, frame_idx).exists()

    def save_image(self, seq_id: str, frame_idx: int, image: torch.Tensor):
        """Save preprocessed image to cache.

        Args:
            seq_id: Sequence ID
            frame_idx: Frame index
            image: Preprocessed image tensor [1, 3, H, W]
        """
        if not self.enabled:
            return

        cache_path = self._get_image_path(seq_id, frame_idx)
        if self.overwrite or not cache_path.exists():
            torch.save(image, cache_path)

    def save_points(self, seq_id: str, frame_idx: int, points: torch.Tensor):
        """Save preprocessed point cloud to cache.

        Args:
            seq_id: Sequence ID
            frame_idx: Frame index
            points: Preprocessed point cloud [N, 4]
        """
        if not self.enabled:
            return

        cache_path = self._get_points_path(seq_id, frame_idx)
        if self.overwrite or not cache_path.exists():
            torch.save(points, cache_path)

    def save_target(self, seq_id: str, frame_idx: int, target: torch.Tensor):
        """Save preprocessed target to cache.

        Args:
            seq_id: Sequence ID
            frame_idx: Frame index
            target: One-hot encoded target [num_classes, H, W]
        """
        if not self.enabled:
            return

        cache_path = self._get_target_path(seq_id, frame_idx)
        if self.overwrite or not cache_path.exists():
            torch.save(target, cache_path)

    def load_image(self, seq_id: str, frame_idx: int) -> Optional[torch.Tensor]:
        """Load preprocessed image from cache.

        Args:
            seq_id: Sequence ID
            frame_idx: Frame index

        Returns:
            Cached image tensor [1, 3, H, W] or None if not cached
        """
        if not self.enabled:
            return None

        cache_path = self._get_image_path(seq_id, frame_idx)
        if cache_path.exists():
            try:
                return torch.load(cache_path)
            except Exception as e:
                print(f"Warning: Failed to load cached image {cache_path}: {e}")
                return None
        return None

    def load_points(
        self, seq_id: str, frame_idx: int
    ) -> Optional[torch.Tensor]:
        """Load preprocessed point cloud from cache.

        Args:
            seq_id: Sequence ID
            frame_idx: Frame index

        Returns:
            Cached point cloud [N, 4] or None if not cached
        """
        if not self.enabled:
            return None

        cache_path = self._get_points_path(seq_id, frame_idx)
        if cache_path.exists():
            try:
                return torch.load(cache_path)
            except Exception as e:
                print(
                    f"Warning: Failed to load cached points {cache_path}: {e}"
                )
                return None
        return None

    def load_target(
        self, seq_id: str, frame_idx: int
    ) -> Optional[torch.Tensor]:
        """Load preprocessed target from cache.

        Args:
            seq_id: Sequence ID
            frame_idx: Frame index

        Returns:
            Cached target [num_classes, H, W] or None if not cached
        """
        if not self.enabled:
            return None

        cache_path = self._get_target_path(seq_id, frame_idx)
        if cache_path.exists():
            try:
                return torch.load(cache_path)
            except Exception as e:
                print(
                    f"Warning: Failed to load cached target {cache_path}: {e}"
                )
                return None
        return None

    def clear_cache(self):
        """Clear all cached data."""
        if not self.enabled:
            return

        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self._initialize_cache()

    def get_cache_size(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache sizes:
                - num_images: Number of cached images
                - num_points: Number of cached point clouds
                - num_targets: Number of cached targets
                - total_size_mb: Total cache size in MB
        """
        if not self.enabled:
            return {
                "num_images": 0,
                "num_points": 0,
                "num_targets": 0,
                "total_size_mb": 0.0,
            }

        num_images = len(list(self.images_dir.glob("*.pt")))
        num_points = len(list(self.points_dir.glob("*.pt")))
        num_targets = len(list(self.targets_dir.glob("*.pt")))

        # Calculate total size
        total_size = 0
        for subdir in [self.images_dir, self.points_dir, self.targets_dir]:
            for file in subdir.glob("*.pt"):
                total_size += file.stat().st_size

        return {
            "num_images": num_images,
            "num_points": num_points,
            "num_targets": num_targets,
            "total_size_mb": total_size / (1024 * 1024),
        }

    def save_config(self, config: dict):
        """Save cache configuration."""
        if not self.enabled:
            return

        config_path = self.cache_dir / "cache_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def load_config(self) -> Optional[dict]:
        """Load cache configuration."""
        if not self.enabled:
            return None

        config_path = self.cache_dir / "cache_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        return None
