"""Example usage of PolarBEVTransform with RELLIS-3D dataset.

This script demonstrates how to use the polar BEV transformation
with actual RELLIS-3D data.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from bev_former import PolarBEVTransform


def load_rellis_camera_info(sequence_dir: Path) -> dict:
    """Load camera calibration from RELLIS-3D sequence.
    
    Args:
        sequence_dir: Path to sequence (e.g., RELLIS/Rellis-3D/00000/)
        
    Returns:
        dict with camera intrinsics and image size
    """
    camera_info_file = sequence_dir / "camera_info.txt"
    
    # Parse camera_info.txt
    # Format:
    # image_width: 1920
    # image_height: 1200
    # camera_name: camera
    # camera_matrix:
    #   rows: 3
    #   cols: 3
    #   data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    
    info = {}
    with open(camera_info_file, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if 'image_width:' in line:
            info['width'] = int(line.split(':')[1].strip())
        elif 'image_height:' in line:
            info['height'] = int(line.split(':')[1].strip())
        elif 'data:' in line:
            # Extract camera matrix
            data_str = line.split('[')[1].split(']')[0]
            K_flat = [float(x) for x in data_str.split(',')]
            K = np.array(K_flat).reshape(3, 3)
            info['K'] = K
            
    return info


def create_simple_transform_example():
    """Simple example with dummy data to verify transform works."""
    print("=" * 80)
    print("EXAMPLE 1: Simple Transform with Dummy Data")
    print("=" * 80)
    
    # Configuration
    B, N = 2, 1  # Batch size=2, 1 camera
    C_in, C_out = 256, 128
    img_size = (900, 1600)
    feat_size = (225, 400)
    
    # Create transform
    # Assuming bev_seg_config.json is in the RELLIS data directory
    config_path = Path(__file__).parent.parent.parent / "RELLIS/Rellis-3D/00000/bev_seg_polar/bev_seg_config.json"
    
    if not config_path.exists():
        print(f"Warning: Config not found at {config_path}")
        print("Creating dummy config...")
        config = {
            "angle_resolution": 0.0595,
            "angle_range": [2.844, 3.439],
            "radial_resolution": 0.25,
            "radial_mode": "linear",
            "radial_growth_rate": 0.05,
            "radial_max": 25.0,
            "angle_edges": np.linspace(2.844, 3.439, 11).tolist(),
            "radial_edges": [0.0, 0.25, 0.51, 0.79, 1.08, 1.38, 1.69, 2.01, 
                            2.35, 2.70, 3.06, 3.44, 3.84, 4.25, 4.68, 5.13, 
                            5.59, 6.07, 6.57, 7.09, 7.63, 8.19, 8.77, 9.37, 
                            10.0, 10.65, 11.32, 12.02, 12.75, 13.50, 14.28, 
                            15.09, 15.93, 16.80, 17.70, 18.64, 19.61, 20.62, 
                            21.67, 22.76, 23.89, 25.0]
        }
    else:
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    print(f"\nPolar grid configuration:")
    print(f"  Angular bins: {len(config['angle_edges']) - 1}")
    print(f"  Radial bins: {len(config['radial_edges']) - 1}")
    print(f"  Angle range: [{config['angle_range'][0]:.3f}, {config['angle_range'][1]:.3f}] rad")
    print(f"  Radial range: [0, {config['radial_max']:.1f}] m")
    
    transform = PolarBEVTransform(
        in_channels=C_in,
        out_channels=C_out,
        image_size=img_size,
        feature_size=feat_size,
        polar_grid_config=config,
        dbound=(1.0, 25.0, 0.25),
        downsample=1,
    )
    
    print(f"\nTransform created:")
    print(f"  Input: [{B}, {N}, {C_in}, {feat_size[0]}, {feat_size[1]}]")
    print(f"  Output: [{B}, {C_out}, {transform.n_r}, {transform.n_theta}]")
    print(f"  Depth bins: {transform.D}")
    
    # Create dummy inputs
    img_feats = torch.randn(B, N, C_in, *feat_size)
    
    # Simple camera calibration (identity transform for demo)
    camera2lidar_rots = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
    camera2lidar_trans = torch.zeros(B, N, 3)
    
    # Dummy intrinsics (roughly RELLIS-3D camera)
    K = torch.tensor([
        [[1266.4, 0, 816.3],
         [0, 1266.4, 491.5],
         [0, 0, 1]]
    ], dtype=torch.float32).unsqueeze(0).expand(B, N, -1, -1)
    
    # No augmentation
    img_aug = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
    
    print("\nRunning forward pass...")
    with torch.no_grad():
        bev_feats = transform(
            img=img_feats,
            camera2lidar_rots=camera2lidar_rots,
            camera2lidar_trans=camera2lidar_trans,
            camera_intrinsics=K,
            img_aug_matrix=img_aug,
        )
    
    print(f"\n✓ Success! Output shape: {bev_feats.shape}")
    print(f"  Memory: {bev_feats.numel() * 4 / 1e6:.2f} MB")
    
    # Visualize one sample
    visualize_bev_features(bev_feats[0], config)
    
    return transform, bev_feats


def visualize_bev_features(bev_feat: torch.Tensor, config: dict):
    """Visualize polar BEV features.
    
    Args:
        bev_feat: [C, n_r, n_theta] BEV features
        config: Polar grid configuration
    """
    print("\n" + "=" * 80)
    print("Visualizing BEV Features")
    print("=" * 80)
    
    # Take mean across channels for visualization
    bev_viz = bev_feat.mean(0).cpu().numpy()  # [n_r, n_theta]
    
    n_r, n_theta = bev_viz.shape
    radial_edges = np.array(config['radial_edges'])
    angle_edges = np.array(config['angle_edges'])
    
    # Create polar plot
    fig = plt.figure(figsize=(15, 5))
    
    # Subplot 1: Polar BEV (actual polar coordinates)
    ax1 = plt.subplot(131, projection='polar')
    
    # Create meshgrid for polar plot
    r_centers = (radial_edges[:-1] + radial_edges[1:]) / 2
    theta_centers = (angle_edges[:-1] + angle_edges[1:]) / 2
    R, Theta = np.meshgrid(r_centers, theta_centers)
    
    # Plot
    c1 = ax1.pcolormesh(Theta.T, R.T, bev_viz, cmap='viridis', shading='auto')
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_title('Polar BEV Features (Mean across channels)', fontsize=10)
    plt.colorbar(c1, ax=ax1, label='Feature magnitude')
    
    # Subplot 2: Rectangular view of polar grid
    ax2 = plt.subplot(132)
    im2 = ax2.imshow(bev_viz, aspect='auto', cmap='viridis', origin='lower')
    ax2.set_xlabel('Angular bin (θ)')
    ax2.set_ylabel('Radial bin (r)')
    ax2.set_title('Polar Grid (Rectangular view)', fontsize=10)
    plt.colorbar(im2, ax=ax2, label='Feature magnitude')
    
    # Subplot 3: Cartesian projection
    ax3 = plt.subplot(133)
    
    # Convert polar grid to Cartesian for visualization
    x_cart = R.T * np.cos(Theta.T)
    y_cart = R.T * np.sin(Theta.T)
    
    c3 = ax3.scatter(x_cart.ravel(), y_cart.ravel(), c=bev_viz.ravel(), 
                     cmap='viridis', s=10, alpha=0.6)
    ax3.set_xlabel('X (m) [forward]')
    ax3.set_ylabel('Y (m) [left]')
    ax3.set_title('Cartesian Projection', fontsize=10)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(c3, ax=ax3, label='Feature magnitude')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent / 'bev_features_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Don't show plot in headless environments
    # plt.show()
    plt.close()


def analyze_grid_properties(config: dict):
    """Analyze properties of the non-uniform polar grid."""
    print("\n" + "=" * 80)
    print("POLAR GRID ANALYSIS")
    print("=" * 80)
    
    radial_edges = np.array(config['radial_edges'])
    angle_edges = np.array(config['angle_edges'])
    
    # Radial bin widths
    radial_widths = np.diff(radial_edges)
    
    print(f"\nRadial bins:")
    print(f"  Total bins: {len(radial_edges) - 1}")
    print(f"  Range: [0, {radial_edges[-1]:.2f}] m")
    print(f"  Min width: {radial_widths.min():.3f} m (near sensor)")
    print(f"  Max width: {radial_widths.max():.3f} m (far from sensor)")
    print(f"  Mean width: {radial_widths.mean():.3f} m")
    print(f"  Growth factor: {(radial_widths[-1] / radial_widths[0]):.2f}x")
    
    # Angular bins
    angle_widths = np.diff(angle_edges)
    angle_range_deg = (angle_edges[-1] - angle_edges[0]) * 180 / np.pi
    
    print(f"\nAngular bins:")
    print(f"  Total bins: {len(angle_edges) - 1}")
    print(f"  Range: [{angle_edges[0]:.3f}, {angle_edges[-1]:.3f}] rad")
    print(f"  Range (deg): {angle_range_deg:.1f}°")
    print(f"  Angular resolution: {np.mean(angle_widths) * 180 / np.pi:.2f}°")
    
    # Total cells
    n_cells = (len(radial_edges) - 1) * (len(angle_edges) - 1)
    print(f"\nGrid statistics:")
    print(f"  Total cells: {n_cells:,}")
    
    # Compare with uniform Cartesian
    uniform_res = radial_widths[0]  # Use finest resolution
    uniform_side = int(np.ceil(radial_edges[-1] * 2 / uniform_res))
    uniform_cells = uniform_side ** 2
    
    print(f"\nComparison with uniform Cartesian grid:")
    print(f"  Uniform @ {uniform_res}m: {uniform_side}×{uniform_side} = {uniform_cells:,} cells")
    print(f"  Polar (non-uniform): {n_cells:,} cells")
    print(f"  Memory savings: {(1 - n_cells/uniform_cells) * 100:.1f}%")
    
    # Visualize bin sizes
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Radial bin widths
    ax = axes[0]
    bin_centers = (radial_edges[:-1] + radial_edges[1:]) / 2
    ax.plot(bin_centers, radial_widths, 'o-', markersize=4)
    ax.set_xlabel('Radial distance (m)')
    ax.set_ylabel('Bin width (m)')
    ax.set_title('Non-uniform Radial Bin Widths')
    ax.grid(True, alpha=0.3)
    
    # Cumulative cells
    ax = axes[1]
    cumulative_bins = np.arange(1, len(radial_edges))
    cumulative_cells = cumulative_bins * len(angle_edges) - 1
    uniform_cumulative = (2 * radial_edges[1:] / uniform_res) ** 2
    
    ax.plot(radial_edges[1:], cumulative_cells, 'o-', label='Polar (non-uniform)', markersize=4)
    ax.plot(radial_edges[1:], uniform_cumulative, 's--', label='Cartesian (uniform)', markersize=4, alpha=0.7)
    ax.set_xlabel('Maximum range (m)')
    ax.set_ylabel('Total grid cells')
    ax.set_title('Cumulative Cell Count vs Range')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(__file__).parent / 'grid_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Grid analysis saved to: {output_path}")
    plt.close()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("POLAR BEV TRANSFORM - EXAMPLE USAGE")
    print("=" * 80)
    
    # Example 1: Simple transform with dummy data
    transform, bev_feats = create_simple_transform_example()
    
    # Example 2: Analyze grid properties
    config_path = Path(__file__).parent.parent.parent / "RELLIS/Rellis-3D/00000/bev_seg_polar/bev_seg_config.json"
    if config_path.exists():
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        analyze_grid_properties(config)
    
    print("\n" + "=" * 80)
    print("✓ All examples completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Integrate into full BEVFusion pipeline")
    print("  2. Add data loading for RELLIS-3D dataset")
    print("  3. Train with BEV segmentation targets")
    print("  4. Evaluate on RELLIS-3D test set")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()





