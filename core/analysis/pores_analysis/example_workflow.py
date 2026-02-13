"""
Example: Complete PSD Calculation Workflow
===========================================

This script demonstrates the full pipeline for computing PSD from a 3D micro-CT scan.

Workflow:
    1. Load segmented volume (or create synthetic test data)
    2. Configure pipeline parameters
    3. Compute PSD with checkpointing
    4. Export results
    5. Visualize PSD curves
"""


import numpy as np
from pathlib import Path

from .config_loader import load_config
from .psd_calculator import compute_psd
from .psd_output import psd_to_dataframe, save_psd_dataframe, plot_psd

CONFIG = load_config()
PATH_CONFIG = CONFIG.get("paths", {})
IMAGE_CONFIG = CONFIG.get("image_params", {})
PROCESSING_CONFIG = CONFIG.get("processing_thresholds", {})
OUTPUT_CONFIG = CONFIG.get("output_settings", {})

EXAMPLE_OUTPUT_DIR = Path(PATH_CONFIG.get("output_dir", "results")) / "examples"
EXAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def example_synthetic_data():
    """
    Example 1: Synthetic test volume
    
    Creates a volume with multiple spherical pores of varying sizes.
    This demonstrates the pipeline without requiring actual μCT data.
    """
    print("=" * 70)
    print("EXAMPLE 1: Synthetic Test Volume")
    print("=" * 70)
    
    # Generate synthetic volume (128³ with random spheres)
    size = 128
    volume = np.zeros((size, size, size), dtype=bool)
    
    # Add several spheres of different sizes
    spheres = [
        ((40, 64, 64), 20),   # Large sphere
        ((88, 64, 64), 12),   # Medium sphere
        ((64, 40, 40), 8),    # Small sphere
        ((64, 88, 88), 15),   # Another medium sphere
    ]
    
    for center, radius in spheres:
        cz, cy, cx = center
        for z in range(size):
            for y in range(size):
                for x in range(size):
                    dist = np.sqrt(
                        (z - cz)**2 + 
                        (y - cy)**2 + 
                        (x - cx)**2
                    )
                    if dist <= radius:
                        volume[z, y, x] = True
    
    print(f"Volume shape: {volume.shape}")
    print(f"Porosity: {volume.mean():.4f}")
    print(f"Expected pore diameters: 40, 24, 16, 30 voxels")
    
    voxel_spacing = tuple(IMAGE_CONFIG.get("voxel_spacing", (1.0, 1.0, 1.0)))

    # Compute PSD (CPU mode for compatibility)
    print("\nComputing PSD...")
    psd = compute_psd(
        volume,
        voxel_spacing=voxel_spacing,
        use_gpu=False,                  # Keep CPU-friendly for this synthetic example
        use_chunking=False              # No chunking for small volume
    )
    
    # Convert to DataFrame
    df = psd_to_dataframe(psd)
    
    # Display results
    print("\nPSD Results:")
    print(df.head(10))
    print(f"\nTotal bins: {len(df)}")
    print(f"Reliable bins: {df['is_reliable'].sum()}/{len(df)}")
    
    # Save results
    synthetic_path = EXAMPLE_OUTPUT_DIR / "synthetic_psd.csv"
    save_psd_dataframe(df, synthetic_path)
    print(f"\nResults saved to {synthetic_path}")
    
    # Plot (requires matplotlib)
    try:
        plot_psd(df, save_path=EXAMPLE_OUTPUT_DIR / "synthetic_psd_plot.png")
    except ImportError:
        print("Matplotlib not available, skipping plot")


def example_real_data_with_chunking():
    """
    Example 2: Large volume with block processing and checkpointing
    
    Demonstrates the recommended workflow for processing large μCT scans
    in Google Colab with GPU acceleration and checkpoint resilience.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Large Volume with Chunking & Checkpointing")
    print("=" * 70)
    print("\n⚠️  This example honors the centralized config for paths and thresholds.")

    volume_path = PATH_CONFIG.get("input_volume_path")
    if not volume_path:
        print("No input volume defined in config. Set paths.input_volume_relative_path before running this example.")
        return

    volume_file = Path(volume_path)
    if not volume_file.exists():
        print(f"Segmented volume not found: {volume_file}")
        print("Update config.yaml or provide the file before re-running this example.")
        return

    chunk_size = tuple(PROCESSING_CONFIG.get("chunk_size", (128, 128, 128)))
    halo_width = PROCESSING_CONFIG.get("halo_width", 50)
    voxel_spacing = tuple(IMAGE_CONFIG.get("voxel_spacing", (1.0, 1.0, 1.0)))
    use_gpu = PROCESSING_CONFIG.get("use_gpu", True)
    use_chunking = PROCESSING_CONFIG.get("use_chunking", True)
    checkpoint_dir = PATH_CONFIG.get("checkpoint_dir")
    run_id = PATH_CONFIG.get("default_run_id")

    print(f"Loading volume from: {volume_file}")
    volume = np.load(volume_file)

    psd = compute_psd(
        volume,
        voxel_spacing=voxel_spacing,
        use_gpu=use_gpu,
        use_chunking=use_chunking,
        chunk_size=chunk_size,
        halo_width=halo_width,
        checkpoint_dir=checkpoint_dir,
        run_id=run_id,
        resume=PROCESSING_CONFIG.get("resume_from_checkpoint", False)
    )

    df = psd_to_dataframe(psd)
    real_results_path = EXAMPLE_OUTPUT_DIR / "real_volume_psd.csv"
    save_psd_dataframe(
        df,
        real_results_path,
        metadata=OUTPUT_CONFIG.get("metadata", {})
    )

    reliable_df = df[df['is_reliable']]
    print(f"Reliable bins: {len(reliable_df)}/{len(df)}")

    try:
        plot_psd(df, save_path=EXAMPLE_OUTPUT_DIR / "real_volume_plot.png")
    except ImportError:
        print("Matplotlib not available, skipping plot")


def example_anisotropic_voxels():
    """
    Example 3: Anisotropic voxel spacing
    
    Demonstrates handling of non-cubic voxels (common in μCT).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Anisotropic Voxel Spacing")
    print("=" * 70)
    
    # Create synthetic volume
    size = 64
    volume = np.zeros((size, size, size), dtype=bool)
    
    # Single sphere
    center = (32, 32, 32)
    radius = 15
    
    for z in range(size):
        for y in range(size):
            for x in range(size):
                dist = np.sqrt(
                    (z - center[0])**2 + 
                    (y - center[1])**2 + 
                    (x - center[2])**2
                )
                if dist <= radius:
                    volume[z, y, x] = True
    
    # Anisotropic voxel spacing (Z spacing 2× larger than X/Y)
    voxel_spacing = (2.0, 1.0, 1.0)  # μm
    
    print(f"Volume shape: {volume.shape}")
    print(f"Voxel spacing: {voxel_spacing} μm")
    print(f"Physical dimensions: {size*2:.1f} × {size*1:.1f} × {size*1:.1f} μm³")
    
    # Compute PSD
    psd = compute_psd(
        volume,
        voxel_spacing=voxel_spacing,
        use_gpu=False,
        use_chunking=False
    )
    
    df = psd_to_dataframe(psd)
    
    print("\nPeak diameter:")
    peak_idx = df['Volume_Count'].argmax()
    print(f"  Voxel units: {df.loc[peak_idx, 'Diameter_px']:.2f} voxels")
    print(f"  Physical units: {df.loc[peak_idx, 'Diameter_um']:.2f} μm")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("PSD CALCULATOR - EXAMPLE WORKFLOWS")
    print("=" * 70)
    
    # Example 1: Synthetic data (executable)
    example_synthetic_data()
    
    # Example 2: Large volume workflow (template)
    example_real_data_with_chunking()
    
    # Example 3: Anisotropic voxels
    example_anisotropic_voxels()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Adapt Example 2 for your actual data")
    print("  2. Run validation tests: python test_psd_synthetic.py")
    print("  3. Review output DataFrames for reliability flags")
    print("\nDocumentation: See pores_analysis/README.md")


if __name__ == "__main__":
    main()
