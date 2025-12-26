"""Test script for RELLIS-3D BEV Segmentation.

Evaluates a trained model on the test set and reports IoU metrics.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import yaml
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from offroad_det_seg_rellis.dataset import (
    create_rellis_dataloader,
    load_model_config,
)
from offroad_det_seg_rellis.model import RellisPolarBevFusion
from offroad_det_seg_rellis.train.metrics import (
    compute_iou,
    aggregate_iou_metrics,
    format_iou_metrics,
)
from offroad_det_seg_rellis.train.train import (
    fix_state_dict,
    TRAIN_DIR,
    CONFIG_DIR,
)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def test(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    class_names: list,
) -> Dict[str, Any]:
    """Evaluate model on test set.

    Returns:
        Dictionary with test metrics
    """
    model.eval()

    all_iou_metrics = []

    print(f"Evaluating on {len(test_loader)} test samples...")

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch in pbar:
            # Move data to device
            images = batch["images"].to(device)
            points = [p.to(device) for p in batch["points"]]
            targets = batch["targets"].to(device)
            camera_params = {
                k: v.to(device) for k, v in batch["camera_params"].items()
            }

            # Forward pass
            output = model(
                points=points,
                images=images,
                camera_params=camera_params,
                targets=None,  # No need for loss computation
            )

            # Get predictions
            predictions = output["predictions"]  # [B, num_classes, H, W]

            # Compute IoU
            iou_metrics = compute_iou(
                predictions, targets, num_classes=config["num_classes"]
            )
            all_iou_metrics.append(iou_metrics)

            # Update progress bar
            pbar.set_postfix({"mIoU": iou_metrics["mean_iou"]})

    # Aggregate IoU metrics
    aggregated_iou = aggregate_iou_metrics(
        all_iou_metrics, num_classes=config["num_classes"]
    )

    return aggregated_iou


def main(args):
    """Main test function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Load checkpoint
    checkpoint_dir = TRAIN_DIR / "checkpoints"
    checkpoint_name = args.checkpoint
    checkpoint_path = checkpoint_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    assert "config" in checkpoint, "Config not found in checkpoint"
    config = checkpoint["config"]

    # Load model config
    model_config_path = CONFIG_DIR / config["model_config"]
    model_config = load_model_config(str(model_config_path))

    # Create model
    print("Creating model...")
    model = RellisPolarBevFusion(
        point_cloud_encoder_config=model_config["point_cloud_encoder"],
        image_encoder_config=model_config["image_encoder"],
        bev_transform_config=model_config["bev_transform"],
        fusion_config=model_config["fusion"],
        segmentation_head_config=model_config["segmentation_head"],
        loss_config=config["loss"],
        use_camera=True,
        use_lidar=True,
    ).to(device)

    model.load_state_dict(fix_state_dict(checkpoint["model_state_dict"]))

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if "val_mean_iou" in checkpoint:
        print(f"Validation mIoU: {checkpoint['val_mean_iou']:.4f}")

    # Create test dataloader
    print("Creating test dataloader...")
    print(f"Using list: {CONFIG_DIR / config['test_list']}")
    test_loader = create_rellis_dataloader(
        data_root=config["data_root"],
        list_file_path=CONFIG_DIR / config["test_list"],
        batch_size=1,
        mode="test",
        shuffle=False,
        num_workers=config["num_workers"],
        target_image_size=tuple(config["target_image_size"]),
        grid_type=config["grid_type"],
        num_classes=config["num_classes"],
        lidar_type=config["lidar_type"],
        filter_fov=config["filter_fov"],
        use_cache=config["use_cache"],
    )

    print(f"Test samples: {len(test_loader)}")

    # Get class names
    class_names = config["loss"]["class_names"]

    # Run test
    print("\n" + "=" * 80)
    print("Starting Test Evaluation")
    print("=" * 80 + "\n")

    test_metrics = test(model, test_loader, device, config, class_names)

    # Print results
    print("\n" + "=" * 80)
    print("Test Results")
    print("=" * 80)

    formatted_metrics = format_iou_metrics(test_metrics, class_names)
    print(formatted_metrics)

    # Save results to file
    results_path = (
        checkpoint_path.parent / f"test_results_{checkpoint_name}.txt"
    )
    with open(results_path, "w") as f:
        f.write("Test Results\n")
        f.write("=" * 80 + "\n")
        f.write(formatted_metrics)
        f.write("\n")

    print(f"\nResults saved to: {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test RELLIS-3D BEV Segmentation Model"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )

    args = parser.parse_args()
    main(args)
