"""Loss functions for semantic segmentation.

Includes focal loss, dice loss, Lovász-Softmax, and combined losses
for handling class imbalance common in off-road datasets.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['focal_loss', 'dice_loss', 'lovasz_softmax', 'combined_loss']


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Focal Loss for addressing class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        inputs: Logits [B, C, H, W] or [B, H, W]
        targets: Binary targets [B, C, H, W] or [B, H, W]
        alpha: Weighting factor in [0, 1] for class balance
        gamma: Focusing parameter γ ≥ 0 (higher = more focus on hard examples)
        reduction: 'none' | 'mean' | 'sum'
        
    Returns:
        Focal loss value
        
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    inputs = inputs.float()
    targets = targets.float()
    
    # Compute BCE loss
    bce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction='none'
    )
    
    # Compute p_t
    p = torch.sigmoid(inputs)
    p_t = p * targets + (1 - p) * (1 - targets)
    
    # Compute focal term: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma
    
    # Compute focal loss
    loss = focal_weight * bce_loss
    
    # Apply alpha weighting
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    
    # Reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Dice Loss for segmentation.
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - Dice
    
    Args:
        inputs: Logits [B, C, H, W]
        targets: Binary targets [B, C, H, W]
        smooth: Smoothing factor to avoid division by zero
        reduction: 'none' | 'mean' | 'sum'
        
    Returns:
        Dice loss value
        
    Note:
        Dice loss is particularly good for small classes and
        directly optimizes the Dice coefficient (similar to IoU).
    """
    inputs = torch.sigmoid(inputs)
    targets = targets.float()
    
    # Flatten spatial dimensions
    inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # [B, C, H*W]
    targets = targets.view(targets.size(0), targets.size(1), -1)
    
    # Compute intersection and union
    intersection = (inputs * targets).sum(dim=2)  # [B, C]
    union = inputs.sum(dim=2) + targets.sum(dim=2)  # [B, C]
    
    # Compute Dice coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    # Dice loss = 1 - Dice
    loss = 1 - dice
    
    # Reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def dice_loss_stable(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
    eps: float = 1e-6,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Numerically stable dice loss."""
    inputs = torch.sigmoid(inputs)
    targets = targets.float()
    
    # Flatten
    inputs = inputs.view(inputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    
    # Compute with better stability
    intersection = (inputs * targets).sum(dim=1)
    cardinality = inputs.sum(dim=1) + targets.sum(dim=1)
    
    # Only compute dice for samples with non-zero targets
    # This prevents instability from empty classes
    valid_mask = cardinality > eps
    
    dice = torch.zeros_like(intersection)
    dice[valid_mask] = (2.0 * intersection[valid_mask] + smooth) / (cardinality[valid_mask] + smooth)
    
    # For invalid samples (no targets), set dice to 1 (loss = 0)
    dice[~valid_mask] = 1.0
    
    loss = 1 - dice
    
    if reduction == 'mean':
        return loss.mean()
    return loss


def lovasz_softmax(
    probas: torch.Tensor,
    labels: torch.Tensor,
    classes: str = 'present',
    per_image: bool = False,
) -> torch.Tensor:
    """Lovász-Softmax loss for multi-class segmentation.
    
    This loss directly optimizes the IoU metric through its convex
    surrogate (Lovász extension).
    
    Args:
        probas: Class probabilities [B, C, H, W] (after softmax/sigmoid)
        labels: Ground truth labels [B, C, H, W] (binary)
        classes: 'all' for all classes, 'present' for classes in GT
        per_image: Compute loss per image then average
        
    Returns:
        Lovász-Softmax loss
        
    Reference:
        Berman et al. "The Lovász-Softmax loss" (CVPR 2018)
    """
    if per_image:
        loss = torch.mean(torch.stack([
            lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0)))
            for prob, lab in zip(probas, labels)
        ]))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels))
    return loss


def flatten_probas(probas: torch.Tensor, labels: torch.Tensor):
    """Flatten probas and labels for Lovász loss."""
    B, C, H, W = probas.shape
    probas = probas.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
    labels = labels.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
    return probas, labels


def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor):
    """Lovász loss on flattened tensors."""
    C = probas.size(1)
    losses = []
    
    for c in range(C):
        # Foreground class c
        fg = labels[:, c].float()
        
        if fg.sum() == 0:
            continue
        
        # Class c probabilities
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        
        # Lovász extension
        grad = lovasz_grad(fg_sorted)
        loss = torch.dot(errors_sorted, grad)
        losses.append(loss)
    
    if len(losses) == 0:
        return probas.sum() * 0.0  # Return zero loss with gradient
    
    return torch.mean(torch.stack(losses))


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute gradient of Lovász extension."""
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    
    if len(jaccard) > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    
    return jaccard


def combined_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    focal_weight: float = 1.0,
    dice_weight: float = 1.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """Combined Focal + Dice loss.
    
    Combines the benefits of both:
    - Focal: Handles class imbalance, focuses on hard examples
    - Dice: Directly optimizes overlap metric, good for small objects
    
    Args:
        inputs: Logits [B, C, H, W]
        targets: Binary targets [B, C, H, W]
        focal_weight: Weight for focal loss term
        dice_weight: Weight for dice loss term
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        
    Returns:
        Combined loss = focal_weight * FL + dice_weight * DL
    """
    loss = 0.0
    
    if focal_weight > 0:
        loss += focal_weight * focal_loss(
            inputs, targets, alpha=focal_alpha, gamma=focal_gamma
        )
    
    if dice_weight > 0:
        loss += dice_weight * dice_loss(inputs, targets)
    
    return loss


class FocalLoss(nn.Module):
    """Focal Loss module."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return focal_loss(inputs, targets, self.alpha, self.gamma)


class DiceLoss(nn.Module):
    """Dice Loss module."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return dice_loss(inputs, targets, self.smooth)


class CombinedLoss(nn.Module):
    """Combined Focal + Dice Loss module."""
    
    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return combined_loss(
            inputs, targets,
            self.focal_weight, self.dice_weight,
            self.focal_alpha, self.focal_gamma,
        )
