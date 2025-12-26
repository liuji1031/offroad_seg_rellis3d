"""Training script for RELLIS-3D BEV Segmentation.

Trains the RellisPolarBevFusion model with:
- Train/validation dataloaders
- Adam optimizer + ReduceLROnPlateau scheduler
- Checkpoint management (last + best models)
- TensorBoard logging
- IoU metrics computation
- torch.compile for optimized JIT compilation (enabled by default)
"""

import argparse
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
from pprint import pprint

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from offroad_det_seg_rellis.dataset import create_rellis_dataloader, load_model_config
from offroad_det_seg_rellis.model import RellisPolarBevFusion
from offroad_det_seg_rellis.train.metrics import compute_iou, aggregate_iou_metrics

TRAIN_DIR = Path(__file__).parent
CONFIG_DIR = TRAIN_DIR.parent / "config"

def fix_state_dict(state_dict: dict, prefix: str = "_orig_mod.") -> dict:
    keys = list(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            state_dict[key.replace(prefix, "")] = state_dict[key]
            del state_dict[key]
    return state_dict

def compute_gradient_stats(model: nn.Module) -> Dict[str, float]:
    """Compute gradient statistics for all model parameters.
    
    Returns dictionary with module/layer names as keys and average gradient
    magnitudes as values. Useful for debugging gradient flow issues.
    
    Args:
        model: PyTorch model with computed gradients
        
    Returns:
        Dictionary mapping parameter names to gradient L2 norms
    """
    grad_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Compute L2 norm of gradients for this parameter
            grad_norm = param.grad.data.norm(2).item()
            
            # Simplify name: remove 'module.' prefix if using DataParallel
            clean_name = name.replace('module.', '')
            
            # Store the gradient norm
            grad_stats[clean_name] = grad_norm
        else:
            # Parameter has no gradient (might be frozen or not used)
            clean_name = name.replace('module.', '')
            grad_stats[clean_name] = 0.0
    
    return grad_stats


def aggregate_gradient_stats_by_module(grad_stats: Dict[str, float]) -> Dict[str, float]:
    """Aggregate gradient statistics by module (e.g., all conv layers in resnet.layer1).
    
    Args:
        grad_stats: Dictionary from compute_gradient_stats()
        
    Returns:
        Dictionary with aggregated stats per module
    """
    module_stats = defaultdict(list)
    
    for param_name, grad_norm in grad_stats.items():
        # Extract module name (everything before the last dot)
        if '.' in param_name:
            module_name = '.'.join(param_name.split('.')[:-1])
        else:
            module_name = 'root'
        
        module_stats[module_name].append(grad_norm)
    
    # Compute average gradient norm per module
    aggregated = {}
    for module_name, grad_norms in module_stats.items():
        if grad_norms:
            aggregated[module_name] = sum(grad_norms) / len(grad_norms)
        else:
            aggregated[module_name] = 0.0
    
    return aggregated


def set_random_seed(seed: int, deterministic: bool = False):
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (slower but reproducible).
                      If False, allow cudnn.benchmark for faster training.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    if deterministic:
        # Fully reproducible but slower
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Faster training with cudnn auto-tuning
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: Path,
    timestamp: str,
):
    """Save checkpoint (always last, optionally best).
    
    Args:
        state: Dictionary with model_state_dict, optimizer_state_dict, etc.
        is_best: Whether this is the best model so far
        checkpoint_dir: Directory to save checkpoints
        timestamp: Timestamp string to append to checkpoint filenames
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Always save last checkpoint with timestamp
    last_path = checkpoint_dir / f'checkpoint_last_{timestamp}.pth'
    torch.save(state, last_path)
    print(f"Saved checkpoint to {last_path}")
    
    # Save best checkpoint with same timestamp
    if is_best:
        best_path = checkpoint_dir / f'checkpoint_best_{timestamp}.pth'
        shutil.copy(last_path, best_path)
        print(f"✓ New best model saved to {best_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
) -> tuple[int, float]:
    """Load checkpoint and resume training.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: LR scheduler to load state into
        device: Device to load checkpoint on
        
    Returns:
        Tuple of (start_epoch, best_val_loss)
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(fix_state_dict(checkpoint['model_state_dict']))
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    
    print(f"Resumed from epoch {checkpoint['epoch']}, best val loss: {best_val_loss:.4f}")
    
    return start_epoch, best_val_loss


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    writer: SummaryWriter,
    global_step: int,
    class_names: list,
    scaler: torch.amp.GradScaler = None,
    use_amp: bool = False,
) -> tuple[Dict[str, float], int, int]:
    """Train for one epoch.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        config: Training configuration
        writer: TensorBoard writer
        global_step: Global step counter
        class_names: List of class names
        scaler: GradScaler for mixed precision training (optional)
        use_amp: Whether to use automatic mixed precision
    
    Returns:
        Tuple of (loss_dict, global_step, nan_count)
    """
    model.train()
    
    epoch_losses = defaultdict(float)
    all_iou_metrics = []
    num_batches = len(train_loader)
    nan_count = 0  # Track batches with NaN loss
    
    # Set up autocast context
    amp_dtype = torch.float16 if use_amp else torch.float32
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        images = batch['images'].to(device)
        points = [p.to(device) for p in batch['points']]
        targets = batch['targets'].to(device)
        camera_params = {
            k: v.to(device) for k, v in batch['camera_params'].items()
        }
        
        # Forward pass with optional mixed precision
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            output = model(
                points=points,
                images=images,
                camera_params=camera_params,
                targets=targets,
            )
            
            # Compute loss
            losses = output['loss']
            mean_loss = torch.mean(torch.stack([v for v in losses.values()]))
        
        # Check for NaN/Inf or abnormally large loss values
        loss_threshold = 1000.0  # Loss values above this indicate instability
        is_bad_loss = (
            torch.isnan(mean_loss) or 
            torch.isinf(mean_loss) or 
            mean_loss.item() > loss_threshold
        )
        
        if is_bad_loss:
            loss_val = mean_loss.item() if not (torch.isnan(mean_loss) or torch.isinf(mean_loss)) else float('nan')
            print(f"\n⚠️ Warning: Bad loss detected at batch {batch_idx}/{num_batches} (loss={loss_val:.4f}). Skipping batch.")
            nan_count += 1
            
            # Clean up to prevent memory leak
            del output, losses, mean_loss, images, points, targets, camera_params
            torch.cuda.empty_cache()
            
            # Check if model weights are corrupted (consecutive bad batches indicate this)
            if nan_count >= 10:
                # Check a sample of model parameters for NaN
                has_nan_weights = False
                for name, param in model.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        has_nan_weights = True
                        print(f"❌ NaN/Inf detected in model weights: {name}")
                        break
                
                if has_nan_weights:
                    raise RuntimeError(
                        f"Model weights have become NaN/Inf after {nan_count} consecutive bad batches. "
                        "Training cannot continue. Consider: reducing learning rate, increasing gradient clip, "
                        "or resuming from an earlier checkpoint."
                    )
            
            global_step += 1
            continue
        
        # Backward pass (with optional AMP scaling)
        if use_amp and scaler is not None:
            scaler.scale(mean_loss).backward()
        else:
            mean_loss.backward()
        
        # Unscale gradients for gradient clipping and stats (AMP only)
        if use_amp and scaler is not None:
            scaler.unscale_(optimizer)
        
        # Compute gradient statistics (before clipping)
        if batch_idx % config.get('grad_check_interval', 100) == 0:
            grad_stats = compute_gradient_stats(model)
            module_grad_stats = aggregate_gradient_stats_by_module(grad_stats)
            
            # Log to TensorBoard
            for module_name, avg_grad in module_grad_stats.items():
                writer.add_scalar(f'gradients/{module_name}', avg_grad, global_step)
            
            # Compute total gradient norm
            total_grad_norm = sum(grad_stats.values()) ** 0.5
            writer.add_scalar('gradients/total_norm', total_grad_norm, global_step)
            
            # Print summary for key modules
            if batch_idx == 0:
                print(f"\n[Gradient Check - Epoch {epoch}]")
                key_modules = ['lidar_encoder', 'image_encoder', 'bev_transform', 'fuser', 'seg_head']
                for key_mod in key_modules:
                    matching = [f"{k}: {v:.6f}" for k, v in module_grad_stats.items() if key_mod in k]
                    if matching:
                        print(f"  {key_mod}: {matching[0]}")
                print(f"  Total gradient norm: {total_grad_norm:.6f}")
        
        # Check for NaN/Inf gradients before applying (prevents weight corruption)
        has_bad_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_bad_gradients = True
                    break
        
        if has_bad_gradients:
            print(f"\n⚠️ Warning: NaN/Inf gradients detected at batch {batch_idx}/{num_batches}. Skipping optimizer step.")
            nan_count += 1
            optimizer.zero_grad()  # Clear the bad gradients
            if use_amp and scaler is not None:
                scaler.update()  # Still need to update scaler state
            global_step += 1
            continue
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config['gradient_clip']
        )
        
        # Optimizer step (with optional AMP scaling)
        if use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        # Accumulate losses
        epoch_losses['total'] += mean_loss.item()
        for name, loss in losses.items():
            epoch_losses[name] += loss.item()
        
        # Compute IoU (no gradient needed for metrics)
        with torch.no_grad():
            predictions = output['predictions']  # [B, num_classes, H, W]
            iou_metrics = compute_iou(
                predictions,
                targets,
                num_classes=config['num_classes']
            )
            all_iou_metrics.append(iou_metrics)
        
        # Update progress bar
        pbar.set_postfix({'loss': mean_loss.item()})
        
        # Log to TensorBoard
        if batch_idx % config['log_interval'] == 0:
            writer.add_scalar('train/total_loss', mean_loss.item(), global_step)
            for name, loss in losses.items():
                writer.add_scalar(f'train/{name}', loss.item(), global_step)
        
        global_step += 1
    
    # Average losses (account for skipped NaN batches)
    valid_batches = num_batches - nan_count
    if valid_batches > 0:
        avg_losses = {k: v / valid_batches for k, v in epoch_losses.items()}
    else:
        avg_losses = {k: float('nan') for k in epoch_losses.keys()}
    
    # Log NaN batch count if any occurred
    if nan_count > 0:
        print(f"\n⚠️ Epoch {epoch}: {nan_count}/{num_batches} batches had NaN loss and were skipped.")
    writer.add_scalar('train/nan_batch_count', nan_count, epoch)
    
    # Aggregate IoU metrics
    aggregated_iou = aggregate_iou_metrics(
        all_iou_metrics,
        num_classes=config['num_classes']
    )
    
    # Log training IoU to TensorBoard
    writer.add_scalar('train/mean_iou', aggregated_iou['mean_iou'], epoch)
    for i, (class_name, iou) in enumerate(zip(class_names, aggregated_iou['per_class_iou'])):
        if not torch.isnan(torch.tensor(iou)):
            writer.add_scalar(f'train/iou/{class_name}', iou, epoch)
    
    return avg_losses, global_step, nan_count


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    writer: SummaryWriter,
    class_names: list,
) -> Dict[str, float]:
    """Run validation and compute metrics.
    
    Returns:
        Dictionary with validation metrics. Keys:
        - 'total_loss': float, total loss
        - 'mean_iou': float, mean IoU
        - 'per_class_iou': list, IoU for each class
    """
    model.eval()
    
    val_losses = defaultdict(float)
    all_iou_metrics = []
    num_batches = len(val_loader)
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for batch in pbar:
            # Move data to device
            images = batch['images'].to(device)
            points = [p.to(device) for p in batch['points']]
            targets = batch['targets'].to(device)
            camera_params = {
                k: v.to(device) for k, v in batch['camera_params'].items()
            }
            
            # Forward pass
            output = model(
                points=points,
                images=images,
                camera_params=camera_params,
                targets=targets,
            )
            
            # Accumulate losses
            losses = output['loss']
            mean_loss = torch.mean(torch.stack([v for v in losses.values()]))
            val_losses['total'] += mean_loss.item()
            for name, loss in losses.items():
                val_losses[name] += loss.item()
            
            # Compute IoU
            predictions = output['predictions']  # [B, num_classes, H, W]
            iou_metrics = compute_iou(
                predictions,
                targets,
                num_classes=config['num_classes']
            )
            all_iou_metrics.append(iou_metrics)
            
            # Update progress bar
            pbar.set_postfix({'loss': mean_loss.item()})
    
    # Average losses
    avg_losses = {k: v / num_batches for k, v in val_losses.items()}
    
    # Aggregate IoU metrics
    aggregated_iou = aggregate_iou_metrics(
        all_iou_metrics,
        num_classes=config['num_classes']
    )
    
    # Log to TensorBoard
    writer.add_scalar('val/total_loss', avg_losses['total'], epoch)
    writer.add_scalar('val/mean_iou', aggregated_iou['mean_iou'], epoch)
    
    for name, loss in avg_losses.items():
        if name != 'total':
            writer.add_scalar(f'val/{name}', loss, epoch)
    
    for i, (class_name, iou) in enumerate(zip(class_names, aggregated_iou['per_class_iou'])):
        if not torch.isnan(torch.tensor(iou)):
            writer.add_scalar(f'val/iou/{class_name}', iou, epoch)
    
    # Print validation results
    print(f"\nValidation Results (Epoch {epoch}):")
    print(f"  Total Loss: {avg_losses['total']:.4f}")
    print(f"  Mean IoU: {aggregated_iou['mean_iou']:.4f}")
    
    return {
        'total_loss': avg_losses['total'],
        'mean_iou': aggregated_iou['mean_iou'],
        'per_class_iou': aggregated_iou['per_class_iou'],
    }


def train(train_cfg: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Perform training using train config and model config.
    
    Returns:
        Dictionary containing:
        - 'best_val_iou': Best validation mIoU
        - 'best_checkpoint_path': Path to the best checkpoint file
        - 'run_timestamp': Timestamp of this training run
    """
    # Set up directories
    checkpoint_dir = TRAIN_DIR / train_cfg['checkpoint_dir']
    log_dir = TRAIN_DIR / train_cfg['log_dir']
    
    # Generate timestamp for this training run (used for checkpoints)
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Set random seed (with cudnn.benchmark enabled for speed unless deterministic requested)
    deterministic = train_cfg.get('deterministic', False)
    set_random_seed(train_cfg['seed'], deterministic=deterministic)
    if not deterministic:
        print("cudnn.benchmark enabled for faster training")
    
    # Set up device
    device = torch.device(
        f"cuda:{train_cfg.get('gpu_id', 0)}" if torch.cuda.is_available() and train_cfg['device'] == 'cuda'
        else 'cpu'
    )
    print(f"Using device: {device}")
    
    # Set up mixed precision training (AMP)
    use_amp = train_cfg.get('use_amp', False)
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
        print("Mixed precision training (AMP) enabled - using float16")
    else:
        scaler = None

    
    # Create model
    print("Creating model...")
    model = RellisPolarBevFusion(
        point_cloud_encoder_config=model_cfg['point_cloud_encoder'],
        image_encoder_config=model_cfg['image_encoder'],
        bev_transform_config=model_cfg['bev_transform'],
        fusion_config=model_cfg['fusion'],
        segmentation_head_config=model_cfg['segmentation_head'],
        loss_config=train_cfg['loss'],
        use_camera=True,
        use_lidar=True,
    ).to(device)
    
    # Print model size
    n_params = model.get_n_params()
    print(f"Model parameters: {n_params}")
    print(f"  Total: {n_params['total']:,}")
    
    # Create optimizer
    optimizer = Adam(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
    )
    
    # Create scheduler with optional warmup
    warmup_epochs = train_cfg.get('warmup_epochs', 0)
    
    # Main scheduler (ReduceLROnPlateau)
    main_scheduler = ReduceLROnPlateau(
        optimizer,
        mode=train_cfg['scheduler']['mode'],
        factor=train_cfg['scheduler']['factor'],
        patience=train_cfg['scheduler']['patience'],
        min_lr=train_cfg['scheduler']['min_lr'],
        threshold=train_cfg['scheduler']['threshold'],
        threshold_mode=train_cfg['scheduler']['threshold_mode'],
    )
    
    # Warmup parameters
    if warmup_epochs > 0:
        warmup_start_factor = train_cfg.get('warmup_start_factor', 0.1)  # Start at 10% of target LR
        print(f"Using learning rate warmup for {warmup_epochs} epochs (start_factor={warmup_start_factor})")
    else:
        warmup_start_factor = 1.0
    
    # Store scheduler reference (we'll manually handle warmup)
    scheduler = main_scheduler
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_iou = 0.0
    will_resume = bool(train_cfg.get('resume_from', None))
    
    if will_resume:
        resume_path = TRAIN_DIR / train_cfg['resume_from']
        start_epoch, best_val_loss = load_checkpoint(
            resume_path, model, optimizer, scheduler, device  # type: ignore
        )
    
    # Compile model with torch.compile for faster training
    # Do this after checkpoint loading to ensure we compile the model with loaded weights
    if train_cfg.get('use_torch_compile', True):
        compile_mode = train_cfg.get('torch_compile_mode', 'default')
        print(f"Compiling model with torch.compile (mode: {compile_mode}, backend: dynamo)...")
        model = torch.compile(model, mode=compile_mode)
        print("✓ Model compiled successfully with dynamo backend")
    else:
        print("torch.compile disabled (use_torch_compile=False)")
    
    # Create dataloaders
    print("Creating dataloaders...")
    max_samples = train_cfg.get('max_samples', None)
    if max_samples is not None:
        print(f"⚠️  Limiting training samples to {max_samples} (overfitting test mode)")
    
    train_loader = create_rellis_dataloader(
        data_root=train_cfg['data_root'],
        list_file_path=CONFIG_DIR / train_cfg['train_list'],
        batch_size=train_cfg['batch_size'],
        mode='train',
        shuffle=True,
        num_workers=train_cfg['num_workers'],
        target_image_size=tuple(train_cfg['target_image_size']),
        grid_type=train_cfg['grid_type'],
        num_classes=train_cfg['num_classes'],
        lidar_type=train_cfg['lidar_type'],
        filter_fov=train_cfg['filter_fov'],
        use_cache=train_cfg['use_cache'],
        max_samples=max_samples,
        prefetch_factor=train_cfg['prefetch_factor'],
    )
    
    val_loader = create_rellis_dataloader(
        data_root=train_cfg['data_root'],
        list_file_path=CONFIG_DIR / train_cfg['val_list'],
        batch_size=train_cfg['batch_size'],
        mode='val',
        shuffle=False,
        num_workers=train_cfg['num_workers'],
        target_image_size=tuple(train_cfg['target_image_size']),
        grid_type=train_cfg['grid_type'],
        num_classes=train_cfg['num_classes'],
        lidar_type=train_cfg['lidar_type'],
        filter_fov=train_cfg['filter_fov'],
        use_cache=train_cfg['use_cache'],
        max_samples=None,  # Don't limit validation samples
        prefetch_factor=train_cfg['prefetch_factor'],
    )
    
    print(f"Training samples: {len(train_loader)}")
    print(f"Validation samples: {len(val_loader)}")
    
    # Get class names
    class_names = train_cfg['loss']['class_names']
    
    # Set up TensorBoard
    experiment_name = f"{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir / experiment_name)
    
    # Save config to log dir
    config_save_path = log_dir / experiment_name / 'config.yaml'
    config_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_save_path, 'w') as f:
        yaml.dump(train_cfg, f)
    
    print(f"\nTensorBoard logs: {log_dir / experiment_name}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"\nStarting training from epoch {start_epoch}...\n")
    
    # Training loop
    global_step = 0
    
    for epoch in range(start_epoch, train_cfg['num_epochs']):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{train_cfg['num_epochs']-1}")
        print(f"{'='*80}")
        
        # Apply learning rate warmup
        if epoch <= warmup_epochs:
            # Linear warmup: LR increases from warmup_start_factor * base_lr to base_lr
            warmup_factor = warmup_start_factor + (1.0 - warmup_start_factor) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = train_cfg['learning_rate'] * warmup_factor
            print(f"[Warmup] Epoch {epoch}/{warmup_epochs}, LR factor: {warmup_factor:.3f}")
        
        print(f'Current learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Train
        train_losses, global_step, nan_count = train_one_epoch(
            model, train_loader, optimizer, device,
            epoch, train_cfg, writer, global_step, class_names,
            scaler=scaler, use_amp=use_amp
        )
        
        print(f"\nTraining Loss: {train_losses['total']:.4f}")
        if nan_count > 0:
            print(f"NaN batches skipped: {nan_count}")
        
        # Validate
        val_metrics = validate(
            model, val_loader, device, epoch, train_cfg, writer, class_names
        )
        
        # Update learning rate (only after warmup)
        if epoch >= warmup_epochs:
            scheduler.step(val_metrics['total_loss'])
        
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, epoch)
        print(f"Learning rate: {current_lr:.6f}")
        
        # Save checkpoint
        is_best = val_metrics['total_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['total_loss']

        is_best_iou = val_metrics['mean_iou'] > best_val_iou
        if is_best_iou:
            best_val_iou = val_metrics['mean_iou']
        
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_loss': train_losses['total'],
            'val_loss': val_metrics['total_loss'],
            'val_mean_iou': val_metrics['mean_iou'],
            'config': train_cfg,
        }
        
        save_checkpoint(checkpoint_state, is_best, checkpoint_dir, run_timestamp)
    
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"TensorBoard logs: {log_dir / experiment_name}")
    print("="*80)
    
    writer.close()

    best_checkpoint_path = checkpoint_dir / f'checkpoint_best_{run_timestamp}.pth'
    return {
        'best_val_iou': best_val_iou,
        'best_checkpoint_path': str(best_checkpoint_path),
        'run_timestamp': run_timestamp,
    }



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train RELLIS-3D BEV Segmentation Model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='train_config.yaml',
        help='Path to training config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU ID to use (overrides config)'
    )
    
    args = parser.parse_args()

    # Load config
    config_path = CONFIG_DIR / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    train_config = load_config(config_path)
    pprint(train_config)
    # Load model config
    model_config_path = CONFIG_DIR / train_config['model_config']
    model_config = load_model_config(str(model_config_path))

    train(train_config, model_config)

