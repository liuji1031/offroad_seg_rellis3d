"""Generate balanced train/val/test splits for RELLIS-3D BEV segmentation.

Scans each sequence for available BEV maps and creates balanced splits
using chunk-based assignment to avoid temporal leakage.

This addresses the issue of the original RELLIS-3D split where some
sequences appear only in training and others only in validation.
Sequences are split into contiguous chunks, and entire chunks are
assigned to train/val/test sets to preserve temporal structure.
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple


def get_available_frames(
    data_root: Path,
    sequence_id: str,
    grid_type: str = "polar",
) -> List[int]:
    """Get list of available frame indices for a sequence.
    
    Scans the BEV segmentation directory for .npy files.
    
    Args:
        data_root: Root directory of RELLIS-3D dataset
        sequence_id: Sequence ID (e.g., "00000")
        grid_type: BEV grid type ("polar" or "cartesian")
        
    Returns:
        Sorted list of frame indices
    """
    data_root = data_root.resolve()
    print(f"Data root: {str(data_root)}")
    bev_dir = data_root / sequence_id / f"bev_seg_{grid_type}"
    
    if not bev_dir.exists():
        print(f"Warning: BEV directory not found: {bev_dir}")
        return []
    
    frames = []
    for npy_file in bev_dir.glob("*.npy"):
        # Extract frame index from filename (e.g., "000308.npy" -> 308)
        try:
            frame_idx = int(npy_file.stem)
            frames.append(frame_idx)
        except ValueError:
            continue
    
    return sorted(frames)


def split_into_chunks(
    frames: List[int],
    chunk_size: int,
    min_chunk_size: int = None,
) -> List[List[int]]:
    """Split frames into contiguous chunks.
    
    Args:
        frames: Sorted list of frame indices
        chunk_size: Target size for each chunk
        min_chunk_size: Minimum size for the last chunk (default: chunk_size // 2)
        
    Returns:
        List of chunks, each containing a list of frame indices
    """
    if min_chunk_size is None:
        min_chunk_size = chunk_size // 2
    
    chunks = []
    for i in range(0, len(frames), chunk_size):
        chunk = frames[i:i + chunk_size]
        # Only add chunk if it meets minimum size requirement
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)
        # Otherwise, merge with previous chunk if it exists
        elif chunks:
            chunks[-1].extend(chunk)
    
    return chunks


def split_chunks(
    chunks: List[List[int]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """Split chunks into train/val/test sets by assigning each chunk individually.
    
    Each chunk is independently assigned to train/val/test based on the specified
    ratios. This preserves temporal structure within chunks while ensuring better
    balance across sequences.
    
    Args:
        chunks: List of chunks (each chunk is a list of frame indices)
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_chunks, val_chunks, test_chunks)
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    # Initialize split lists
    train_chunks = []
    val_chunks = []
    test_chunks = []
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Assign each chunk individually to train/val/test
    for chunk in chunks:
        # Generate random number to determine assignment
        r = random.random()
        
        if r < train_ratio:
            train_chunks.append(chunk)
        elif r < train_ratio + val_ratio:
            val_chunks.append(chunk)
        else:
            test_chunks.append(chunk)
    
    return train_chunks, val_chunks, test_chunks


def write_list_file(
    filepath: Path,
    samples: List[Tuple[str, int]],
):
    """Write samples to a .lst file.
    
    Args:
        filepath: Output file path
        samples: List of (sequence_id, frame_idx) tuples
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        for seq_id, frame_idx in samples:
            f.write(f"{seq_id} {frame_idx}\n")
    
    print(f"Wrote {len(samples)} samples to {filepath}")


def generate_splits(
    data_root: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    grid_type: str = "polar",
    sequences: List[str] = None,
    chunk_size: int = 50,
    min_chunk_size: int = None,
):
    """Generate train/val/test splits using chunk-based assignment.
    
    Sequences are split into contiguous chunks, and entire chunks are
    assigned to train/val/test sets. This avoids temporal leakage from
    randomly shuffling individual frames.
    
    Args:
        data_root: Root directory of RELLIS-3D dataset
        output_dir: Directory to save .lst files
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
        grid_type: BEV grid type ("polar" or "cartesian")
        sequences: List of sequence IDs to include (default: auto-detect)
        chunk_size: Target number of frames per chunk
        min_chunk_size: Minimum size for last chunk (default: chunk_size // 2)
    """
    # Auto-detect sequences if not provided
    if sequences is None:
        sequences = []
        for seq_dir in sorted(data_root.iterdir()):
            if seq_dir.is_dir() and seq_dir.name.isdigit():
                sequences.append(seq_dir.name)
    
    print(f"Found sequences: {sequences}")
    print(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    print(f"Random seed: {seed}")
    print(f"Grid type: {grid_type}")
    print(f"Chunk size: {chunk_size}")
    print()
    
    # Collect samples for each split
    train_samples = []
    val_samples = []
    test_samples = []
    
    for seq_id in sequences:
        frames = get_available_frames(data_root, seq_id, grid_type)
        
        if not frames:
            print(f"Skipping sequence {seq_id}: no frames found")
            continue
        
        # Split into chunks
        chunks = split_into_chunks(frames, chunk_size, min_chunk_size)
        
        if not chunks:
            print(f"Skipping sequence {seq_id}: no valid chunks created")
            continue
        
        # Split chunks into train/val/test
        train_chunks, val_chunks, test_chunks = split_chunks(
            chunks, train_ratio, val_ratio, test_ratio, seed
        )
        
        # Flatten chunks into frame lists
        train_frames = [f for chunk in train_chunks for f in chunk]
        val_frames = [f for chunk in val_chunks for f in chunk]
        test_frames = [f for chunk in test_chunks for f in chunk]
        
        # Add to combined lists
        train_samples.extend([(seq_id, f) for f in train_frames])
        val_samples.extend([(seq_id, f) for f in val_frames])
        test_samples.extend([(seq_id, f) for f in test_frames])
        
        print(f"Sequence {seq_id}: {len(frames)} total frames -> "
              f"{len(chunks)} chunks -> "
              f"train={len(train_frames)} ({len(train_chunks)} chunks), "
              f"val={len(val_frames)} ({len(val_chunks)} chunks), "
              f"test={len(test_frames)} ({len(test_chunks)} chunks)")
    
    print()
    
    # Write output files
    write_list_file(output_dir / "split_list_train.lst", train_samples)
    write_list_file(output_dir / "split_list_val.lst", val_samples)
    write_list_file(output_dir / "split_list_test.lst", test_samples)
    
    # Summary
    print()
    print("=" * 60)
    print("Summary:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val:   {len(val_samples)} samples")
    print(f"  Test:  {len(test_samples)} samples")
    print(f"  Total: {len(train_samples) + len(val_samples) + len(test_samples)} samples")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate balanced train/val/test splits for RELLIS-3D using chunk-based assignment"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of RELLIS-3D dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save .lst files (default: same as script location)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Ratio for training set (default: 0.8)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Ratio for validation set (default: 0.1)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Ratio for test set (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--grid_type",
        type=str,
        default="polar",
        choices=["polar", "cartesian"],
        help="BEV grid type (default: polar)"
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=None,
        help="Specific sequences to include (default: auto-detect)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=50,
        help="Target number of frames per chunk (default: 50)"
    )
    parser.add_argument(
        "--min_chunk_size",
        type=int,
        default=None,
        help="Minimum size for last chunk (default: chunk_size // 2)"
    )
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    
    generate_splits(
        data_root=data_root,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        grid_type=args.grid_type,
        sequences=args.sequences,
        chunk_size=args.chunk_size,
        min_chunk_size=args.min_chunk_size,
    )


if __name__ == "__main__":
    main()

