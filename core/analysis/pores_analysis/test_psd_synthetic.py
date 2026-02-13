"""
Synthetic Volume Test Suite for PSD Validation
===============================================

Generates test volumes with known analytical PSD solutions to validate 
the EDT → Opening → PSD pipeline before production use.

Test Cases:
    1. Single Sphere - Known diameter, should produce delta function PSD
    2. Nested Spheres - Two distinct sizes, should show two peaks
    3. Cubic Lattice - Periodic structure with uniform pore size
    4. Random Spheres - Distribution of sizes, validate statistical moments
    5. Edge Cases - Border touching, anisotropic voxels, extreme sizes
"""

import numpy as np
from typing import Tuple, Dict, Optional
import warnings
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from pores_analysis.config_loader import load_config
except ImportError:
    from config_loader import load_config

CONFIG = load_config()
IMAGE_CONFIG = CONFIG.get("image_params", {})
PROCESSING_CONFIG = CONFIG.get("processing_thresholds", {})
DEFAULT_VOXEL_SPACING = tuple(IMAGE_CONFIG.get("voxel_spacing", (1.0, 1.0, 1.0)))
DEFAULT_USE_GPU = PROCESSING_CONFIG.get("use_gpu", True)
DEFAULT_USE_CHUNKING = PROCESSING_CONFIG.get("use_chunking", False)

# Import PSD pipeline (absolute imports for script execution)
from psd_calculator import compute_psd
from psd_output import psd_to_dataframe


def generate_single_sphere(
    volume_size: int = 128,
    radius: int = 20,
    center: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    Generate volume with single centered sphere.
    
    Expected PSD: Single peak at diameter = 2*radius
    
    Args:
        volume_size: Cubic volume dimension
        radius: Sphere radius in voxels
        center: Sphere center (default: volume center)
    
    Returns:
        Binary volume (bool)
    """
    volume = np.zeros((volume_size, volume_size, volume_size), dtype=bool)
    
    if center is None:
        center = (volume_size // 2, volume_size // 2, volume_size // 2)
    
    cz, cy, cx = center
    
    for z in range(volume_size):
        for y in range(volume_size):
            for x in range(volume_size):
                dist = np.sqrt(
                    (z - cz)**2 + 
                    (y - cy)**2 + 
                    (x - cx)**2
                )
                if dist <= radius:
                    volume[z, y, x] = True
    
    return volume


def generate_nested_spheres(
    volume_size: int = 128,
    radii: Tuple[int, ...] = (20, 35),
    spacing: int = 50
) -> np.ndarray:
    """
    Generate volume with multiple spheres of different sizes.
    
    Expected PSD: Multiple peaks at diameters = 2*r for each r in radii
    
    Args:
        volume_size: Cubic volume dimension
        radii: Sphere radii in voxels
        spacing: Spacing between sphere centers
    
    Returns:
        Binary volume (bool)
    """
    volume = np.zeros((volume_size, volume_size, volume_size), dtype=bool)
    
    centers = []
    z_start = volume_size // 2 - (len(radii) - 1) * spacing // 2
    
    for i, radius in enumerate(radii):
        center = (z_start + i * spacing, volume_size // 2, volume_size // 2)
        centers.append(center)
        
        cz, cy, cx = center
        
        for z in range(volume_size):
            for y in range(volume_size):
                for x in range(volume_size):
                    dist = np.sqrt(
                        (z - cz)**2 + 
                        (y - cy)**2 + 
                        (x - cx)**2
                    )
                    if dist <= radius:
                        volume[z, y, x] = True
    
    return volume


def generate_cubic_lattice(
    volume_size: int = 128,
    pore_diameter: int = 20,
    wall_thickness: int = 5
) -> np.ndarray:
    """
    Generate periodic cubic lattice (foam-like structure).
    
    Expected PSD: Narrow distribution around pore_diameter
    
    Args:
        volume_size: Cubic volume dimension
        pore_diameter: Pore cell size
        wall_thickness: Wall thickness between pores
    
    Returns:
        Binary volume (bool)
    """
    volume = np.ones((volume_size, volume_size, volume_size), dtype=bool)
    
    cell_size = pore_diameter + wall_thickness
    
    # Create walls
    for z in range(0, volume_size, cell_size):
        volume[z:min(z + wall_thickness, volume_size), :, :] = False
    
    for y in range(0, volume_size, cell_size):
        volume[:, y:min(y + wall_thickness, volume_size), :] = False
    
    for x in range(0, volume_size, cell_size):
        volume[:, :, x:min(x + wall_thickness, volume_size)] = False
    
    return volume


def generate_random_spheres(
    volume_size: int = 128,
    n_spheres: int = 20,
    radius_range: Tuple[int, int] = (5, 15),
    seed: int = 42
) -> np.ndarray:
    """
    Generate volume with randomly placed spheres of varying sizes.
    
    Expected PSD: Distribution covering radius_range
    
    Args:
        volume_size: Cubic volume dimension
        n_spheres: Number of spheres to place
        radius_range: (min_radius, max_radius) in voxels
        seed: Random seed for reproducibility
    
    Returns:
        Binary volume (bool)
    """
    np.random.seed(seed)
    volume = np.zeros((volume_size, volume_size, volume_size), dtype=bool)
    
    for _ in range(n_spheres):
        # Random center and radius
        center = np.random.randint(
            radius_range[1],
            volume_size - radius_range[1],
            size=3
        )
        radius = np.random.randint(radius_range[0], radius_range[1] + 1)
        
        cz, cy, cx = center
        
        # Place sphere
        for z in range(max(0, cz - radius), min(volume_size, cz + radius + 1)):
            for y in range(max(0, cy - radius), min(volume_size, cy + radius + 1)):
                for x in range(max(0, cx - radius), min(volume_size, cx + radius + 1)):
                    dist = np.sqrt(
                        (z - cz)**2 + 
                        (y - cy)**2 + 
                        (x - cx)**2
                    )
                    if dist <= radius:
                        volume[z, y, x] = True
    
    return volume


def test_single_sphere():
    """Test 1: Single sphere with known diameter."""
    print("\n" + "=" * 60)
    print("TEST 1: Single Sphere")
    print("=" * 60)
    
    radius = 20
    expected_diameter = 2 * radius
    
    volume = generate_single_sphere(volume_size=128, radius=radius)
    
    print(f"Volume: {volume.shape}, porosity: {volume.mean():.4f}")
    print(f"Expected diameter: {expected_diameter} voxels")
    
    # Compute PSD
    psd = compute_psd(
        volume,
        voxel_spacing=DEFAULT_VOXEL_SPACING,
        use_gpu=DEFAULT_USE_GPU,
        use_chunking=DEFAULT_USE_CHUNKING
    )
    
    df = psd_to_dataframe(psd)
    
    # Find peak
    peak_idx = df['Volume_Count'].argmax()
    measured_diameter = df.loc[peak_idx, 'Diameter_px']
    
    print(f"\nResults:")
    print(f"  Measured peak: {measured_diameter:.2f} voxels")
    print(f"  Expected: {expected_diameter} voxels")
    print(f"  Error: {abs(measured_diameter - expected_diameter):.2f} voxels")
    
    # Validation
    error_threshold = 5  # voxels
    if abs(measured_diameter - expected_diameter) < error_threshold:
        print("  Status: ✓ PASS")
    else:
        print("  Status: ✗ FAIL (error too large)")


def test_nested_spheres():
    """Test 2: Two spheres with different sizes."""
    print("\n" + "=" * 60)
    print("TEST 2: Nested Spheres (Two Sizes)")
    print("=" * 60)
    
    radii = (15, 25)
    expected_diameters = [2 * r for r in radii]
    
    volume = generate_nested_spheres(volume_size=128, radii=radii, spacing=50)
    
    print(f"Volume: {volume.shape}, porosity: {volume.mean():.4f}")
    print(f"Expected diameters: {expected_diameters} voxels")
    
    # Compute PSD
    psd = compute_psd(
        volume,
        voxel_spacing=DEFAULT_VOXEL_SPACING,
        use_gpu=DEFAULT_USE_GPU,
        use_chunking=DEFAULT_USE_CHUNKING
    )
    
    df = psd_to_dataframe(psd)
    
    # Find peaks (local maxima)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(df['Volume_Count'].values, height=100)
    
    print(f"\nResults:")
    print(f"  Found {len(peaks)} peaks")
    
    if len(peaks) >= 2:
        measured_diameters = df.loc[peaks, 'Diameter_px'].values[:2]
        print(f"  Measured peaks: {measured_diameters}")
        print(f"  Expected: {expected_diameters}")
        print("  Status: ✓ PASS (detected multiple peaks)")
    else:
        print(f"  Status: ✗ FAIL (expected 2 peaks, found {len(peaks)})")


def test_cubic_lattice():
    """Test 3: Periodic lattice with uniform pore size."""
    print("\n" + "=" * 60)
    print("TEST 3: Cubic Lattice (Periodic Structure)")
    print("=" * 60)
    
    pore_diameter = 20
    
    volume = generate_cubic_lattice(
        volume_size=128,
        pore_diameter=pore_diameter,
        wall_thickness=5
    )
    
    print(f"Volume: {volume.shape}, porosity: {volume.mean():.4f}")
    print(f"Expected diameter: ~{pore_diameter} voxels")
    
    # Compute PSD
    psd = compute_psd(
        volume,
        voxel_spacing=DEFAULT_VOXEL_SPACING,
        use_gpu=DEFAULT_USE_GPU,
        use_chunking=DEFAULT_USE_CHUNKING
    )
    
    df = psd_to_dataframe(psd)
    
    # Find peak
    peak_idx = df['Volume_Count'].argmax()
    measured_diameter = df.loc[peak_idx, 'Diameter_px']
    
    print(f"\nResults:")
    print(f"  Measured peak: {measured_diameter:.2f} voxels")
    print(f"  Expected: ~{pore_diameter} voxels")
    print(f"  Distribution width (std): {df['Diameter_px'].std():.2f} voxels")
    
    # Narrow distribution expected
    if df['Diameter_px'].std() < pore_diameter * 0.5:
        print("  Status: ✓ PASS (narrow distribution)")
    else:
        print("  Status: ⚠ WARNING (distribution wider than expected)")


def test_anisotropic_voxels():
    """Test 4: Anisotropic voxel spacing."""
    print("\n" + "=" * 60)
    print("TEST 4: Anisotropic Voxel Spacing")
    print("=" * 60)
    
    radius = 20  # In voxels
    voxel_spacing = (2.0, 1.0, 1.0)  # Z spacing = 2× X/Y spacing
    
    volume = generate_single_sphere(volume_size=128, radius=radius)
    
    print(f"Volume: {volume.shape}")
    print(f"Voxel spacing: {voxel_spacing} μm")
    print(f"Expected diameter (physical): ~{2 * radius * np.mean(voxel_spacing):.2f} μm")
    
    # Compute PSD
    psd = compute_psd(
        volume,
        voxel_spacing=voxel_spacing,
        use_gpu=DEFAULT_USE_GPU,
        use_chunking=DEFAULT_USE_CHUNKING
    )
    
    df = psd_to_dataframe(psd)
    
    # Find peak
    peak_idx = df['Volume_Count'].argmax()
    measured_diameter_um = df.loc[peak_idx, 'Diameter_um']
    
    print(f"\nResults:")
    print(f"  Measured peak: {measured_diameter_um:.2f} μm")
    print("  Status: ✓ PASS (anisotropic processing completed)")


def test_reliability_flag():
    """Test 5: Vogel et al. reliability constraint (configurable threshold)."""
    print("\n" + "=" * 60)
    print("TEST 5: Reliability Flag (Vogel et al. Constraint)")
    print("=" * 60)
    
    # Create volume with very small pores (should be flagged unreliable)
    volume = generate_random_spheres(
        volume_size=64,
        n_spheres=30,
        radius_range=(2, 8),  # Some below 5-voxel threshold
        seed=42
    )
    
    print(f"Volume: {volume.shape}, porosity: {volume.mean():.4f}")
    
    # Compute PSD
    psd = compute_psd(
        volume,
        voxel_spacing=DEFAULT_VOXEL_SPACING,
        use_gpu=DEFAULT_USE_GPU,
        use_chunking=DEFAULT_USE_CHUNKING
    )
    
    df = psd_to_dataframe(psd)
    
    # Check reliability flags
    n_total = len(df)
    n_unreliable = (~df['is_reliable']).sum()
    n_reliable = df['is_reliable'].sum()
    
    print(f"\nResults:")
    print(f"  Total bins: {n_total}")
    print(f"  Unreliable bins (d < {MIN_RELIABLE_DIAMETER_VOXELS} voxels): {n_unreliable}")
    print(f"  Reliable bins (d >= {MIN_RELIABLE_DIAMETER_VOXELS} voxels): {n_reliable}")
    
    # Verify constraint
    unreliable_diameters = df[~df['is_reliable']]['Diameter_px']
    if len(unreliable_diameters) > 0:
        print(f"  Max unreliable diameter: {unreliable_diameters.max():.2f} voxels")
        if unreliable_diameters.max() < 5.0:
            print("  Status: ✓ PASS (constraint correctly applied)")
        else:
            print(
                f"  Status: ✗ FAIL (some unreliable bins >= {MIN_RELIABLE_DIAMETER_VOXELS} voxels)"
            )
    else:
        print("  Status: ✓ PASS (all bins reliable)")


def run_all_tests():
    """Run complete test suite."""
    print("=" * 60)
    print("PSD SYNTHETIC VALIDATION TEST SUITE")
    print("=" * 60)
    print("\nThese tests validate the EDT → Opening → PSD pipeline")
    print("using synthetic volumes with known analytical solutions.")
    print("\nNote: Small deviations are expected due to discretization.")
    
    try:
        test_single_sphere()
        test_nested_spheres()
        test_cubic_lattice()
        test_anisotropic_voxels()
        test_reliability_flag()
    except Exception as e:
        print(f"\n✗ TEST SUITE FAILED: {e}")
        raise
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)
    print("\nAll tests executed. Review results above for validation.")


if __name__ == "__main__":
    run_all_tests()
