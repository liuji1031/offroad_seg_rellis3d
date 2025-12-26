"""Segmentation loss computation module.

Handles per-class loss computation with class weights and loss type selection.
Separated from segmentation heads for better modularity and reusability.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn

from .losses import focal_loss, dice_loss_stable


__all__ = ['SegmentationLoss']


class SegmentationLoss(nn.Module):
    """Per-class loss computation for multi-class segmentation.
    
    Computes losses for each class independently and applies optional class
    weights. Returns a dictionary of losses keyed by class name and loss type.
    
    Args:
        num_classes: Number of segmentation classes
        loss_type: 'focal' | 'dice' | 'combined'
        class_names: Optional list of class names for logging
        class_weights: Optional per-class weights [num_classes]
        focal_alpha: Alpha parameter for focal loss (default: 0.25)
        focal_gamma: Gamma parameter for focal loss (default: 2.0)
        dice_smooth: Smoothing factor for dice loss (default: 1.0)
        focal_weight: Weight for focal term in combined loss (default: 1.0)
        dice_weight: Weight for dice term in combined loss (default: 1.0)
    """
    
    def __init__(
        self,
        num_classes: int,
        loss_type: str = 'focal',
        class_names: Optional[List[str]] = None,
        class_weights: Optional[torch.Tensor] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_smooth: float = 1.0,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        assert len(self.class_names) == num_classes, \
            f"Number of class names ({len(self.class_names)}) must match num_classes ({num_classes})"
        
        # Loss hyperparameters
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_smooth = dice_smooth
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        assert len(self.class_names) == num_classes, \
            f"Number of class names ({len(self.class_names)}) must match num_classes ({num_classes})"
        
        assert loss_type in ['focal', 'dice', 'combined'], \
            f"Unknown loss type: {loss_type}. Must be 'focal', 'dice', or 'combined'"
        
        # Store class weights as buffer (moved to device automatically)
        if class_weights is not None:
            assert len(class_weights) == num_classes, \
                f"Number of class weights ({len(class_weights)}) must match num_classes ({num_classes})"
            self.register_buffer('class_weights', torch.tensor(class_weights))
        else:
            self.class_weights = None
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-class losses.
        
        Args:
            logits: Predicted logits [B, num_classes, H, W]
            targets: Ground truth labels [B, num_classes, H, W] (binary)
            
        Returns:
            Dictionary of losses per class: {class_name/loss_type: loss_value}
            
        Note:
            - Each loss is a scalar tensor that retains gradients
            - To get total loss: total_loss = sum(losses.values())
        """  
        assert logits.shape == targets.shape, \
            f"Logits shape {logits.shape} must match targets shape {targets.shape}"
        
        assert logits.shape[1] == self.num_classes, \
            f"Logits has {logits.shape[1]} channels but expected {self.num_classes}"
        
        losses = {}
        
        for i, class_name in enumerate(self.class_names):
            # Extract per-class logits and targets
            class_logits = logits[:, i]  # [B, H, W]
            class_targets = targets[:, i]  # [B, H, W]
            
            # Compute loss based on type
            if self.loss_type == 'focal':
                f_loss = focal_loss(
                    class_logits,
                    class_targets,
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                )
                d_loss = None
            elif self.loss_type == 'dice':
                d_loss = dice_loss_stable(
                    class_logits,
                    class_targets,
                    smooth=self.dice_smooth,
                )
                f_loss = None
            elif self.loss_type == 'combined':

                f_loss = focal_loss(
                    class_logits,
                    class_targets,
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                )

                d_loss = dice_loss_stable(
                    class_logits,
                    class_targets,
                    smooth=self.dice_smooth,
                )

            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            # Apply class weight if provided
            if self.class_weights is not None:
                f_loss = f_loss * self.class_weights[i] # only apply to focal loss
            
            # Store losses with descriptive keys
            if self.loss_type == 'focal':
                losses[f'focal/{class_name}'] = f_loss
            elif self.loss_type == 'dice':
                losses[f'dice/{class_name}'] = d_loss
            elif self.loss_type == 'combined':
                # Store focal and dice separately for better logging
                losses[f'focal/{class_name}'] = f_loss
                losses[f'dice/{class_name}'] = d_loss * self.dice_weight
        
        return losses
    
    def compute_total_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total loss (sum of all per-class losses).
        
        Args:
            logits: Predicted logits [B, num_classes, H, W]
            targets: Ground truth labels [B, num_classes, H, W]
            
        Returns:
            Total loss as a single scalar tensor
        """
        losses = self.forward(logits, targets)
        return torch.sum(list(losses.values()))

