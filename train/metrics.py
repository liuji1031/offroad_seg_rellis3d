"""Metrics computation for BEV segmentation.

Provides IoU (Intersection over Union) computation for multi-class segmentation.
"""

import torch
from typing import Dict, List, Any


def compute_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> Dict[str, Any]:
    """Compute per-class IoU and mean IoU.
    
    Args:
        predictions: [B, num_classes, H, W] probabilities (after sigmoid)
        targets: [B, num_classes, H, W] one-hot encoded ground truth
        num_classes: Number of classes
    
    Returns:
        Dictionary with:
            - 'mean_iou': float, mean IoU across all classes (excluding NaN)
            - 'per_class_iou': List[float], IoU for each class (may contain NaN)
    """
    # Convert probabilities to class indices
    pred_classes = torch.argmax(predictions, dim=1)  # [B, H, W]
    target_classes = torch.argmax(targets, dim=1)  # [B, H, W]
    
    ious = []
    for c in range(num_classes):
        # Create binary masks for class c
        pred_mask = (pred_classes == c)
        target_mask = (target_classes == c)
        
        # Compute intersection and union
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        if union > 0:
            iou = (intersection / union).item()
        else:
            # No ground truth and no predictions for this class
            iou = float('nan')
        
        ious.append(iou)
    
    # Compute mean IoU (excluding NaN classes)
    valid_ious = [iou for iou in ious if not torch.isnan(torch.tensor(iou))]
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    
    return {
        'mean_iou': mean_iou,
        'per_class_iou': ious,
    }


def aggregate_iou_metrics(
    iou_list: List[Dict[str, Any]],
    num_classes: int,
) -> Dict[str, Any]:
    """Aggregate IoU metrics across multiple batches.
    
    Args:
        iou_list: List of IoU dicts from compute_iou()
        num_classes: Number of classes
        
    Returns:
        Dictionary with aggregated mean_iou and per_class_iou
    """
    if not iou_list:
        return {'mean_iou': 0.0, 'per_class_iou': [0.0] * num_classes}
    
    # Average mean IoU
    mean_iou = sum([m['mean_iou'] for m in iou_list]) / len(iou_list)
    
    # Average per-class IoU (excluding NaN)
    per_class_iou = []
    for c in range(num_classes):
        class_ious = [m['per_class_iou'][c] for m in iou_list]
        # Filter out NaN values
        valid_class_ious = [iou for iou in class_ious if not torch.isnan(torch.tensor(iou))]
        if valid_class_ious:
            avg_iou = sum(valid_class_ious) / len(valid_class_ious)
        else:
            avg_iou = float('nan')
        per_class_iou.append(avg_iou)
    
    return {
        'mean_iou': mean_iou,
        'per_class_iou': per_class_iou,
    }


def format_iou_metrics(
    metrics: Dict[str, Any],
    class_names: List[str],
) -> str:
    """Format IoU metrics as a readable string.
    
    Args:
        metrics: IoU metrics dict from compute_iou() or aggregate_iou_metrics()
        class_names: List of class names
        
    Returns:
        Formatted string with metrics
    """
    lines = []
    lines.append(f"Mean IoU: {metrics['mean_iou']:.4f}")
    lines.append("Per-class IoU:")
    
    for i, (class_name, iou) in enumerate(zip(class_names, metrics['per_class_iou'])):
        if torch.isnan(torch.tensor(iou)):
            lines.append(f"  {class_name:15s}: N/A")
        else:
            lines.append(f"  {class_name:15s}: {iou:.4f}")
    
    return '\n'.join(lines)