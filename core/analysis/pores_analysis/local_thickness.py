"""
Local Thickness via Morphological Opening
==========================================

Implements Hildebrand & Ruegsegger (1997) morphological reconstruction method 
for computing local thickness (opening map) from EDT.

Methodology (Vogel et al., 2010):
    - Opening Map O(x) = Reconstruction(Marker=EDT, Mask=EDT)
    - Local Thickness = 2 × O(x)  [diameter of maximal inscribed sphere]
    
References:
    Hildebrand, T., & Rüegsegger, P. (1997). A new method for the model-
    independent assessment of thickness in three-dimensional images.
    Journal of Microscopy, 185(1), 67-75.
"""

import numpy as np
from typing import Optional
import warnings


def _check_gpu_available() -> bool:
    """Check if CuPy is available for GPU morphology."""
    try:
        import cupy as cp
        _ = cp.cuda.Device(0).compute_capability
        return True
    except (ImportError, RuntimeError):
        return False


def compute_opening_map(
    edt_map: np.ndarray,
    use_gpu: bool = True
) -> np.ndarray:
    """
    Compute opening map (local thickness) via iterative granulometry.
    
    Algorithm (Vogel et al., 2010 / Hildebrand & Ruegsegger, 1997):
        For each radius r from max(EDT) down to 1:
            1. Identify centers where EDT >= r (potential sphere centers)
            2. Dilate these centers with spherical structuring element of radius r
            3. Assign diameter (2×r) to all voxels covered by these spheres
            4. Keep maximum diameter value at each voxel (largest fitting sphere)
    
    This iterative approach correctly identifies the maximal inscribed sphere
    diameter at each voxel, following the morphological opening methodology.
    
    Args:
        edt_map: Euclidean distance transform (Z, Y, X), float32
                Values represent distance to nearest solid voxel
        use_gpu: If True, attempt GPU acceleration via CuPy
    
    Returns:
        opening_map: Local thickness map (Z, Y, X), float32.
                    Values represent pore diameter (2 × maximal inscribed radius)
                    Already converted to diameter (no need to multiply by 2)
    
    Notes:
        - This is the computational bottleneck of the pipeline (O(max_r × volume))
        - GPU acceleration significantly speeds up dilation operations
        - For large volumes with large pores, consider block processing
        - Output values are in same physical units as EDT input
        - Progress messages printed for radii > 50
    
    Raises:
        ValueError: If input is not 3D or wrong dtype
        RuntimeError: If both GPU and CPU methods fail
    
    Examples:
        >>> from distance_transform import compute_edt
        >>> volume = np.random.rand(100, 100, 100) > 0.7
        >>> edt = compute_edt(volume)
        >>> opening = compute_opening_map(edt)
        >>> # opening already contains diameters, not radii
        >>> print(f"Max pore diameter: {opening.max():.2f}")
    """
    # Validation
    if edt_map.ndim != 3:
        raise ValueError(
            f"Expected 3D EDT map, got shape {edt_map.shape}"
        )
    
    if edt_map.dtype not in [np.float32, np.float64]:
        warnings.warn(
            f"EDT map dtype {edt_map.dtype} not float32/64, converting",
            UserWarning
        )
        edt_map = edt_map.astype(np.float32)
    
    # Attempt GPU computation
    if use_gpu and _check_gpu_available():
        try:
            return _compute_opening_gpu(edt_map)
        except Exception as e:
            warnings.warn(
                f"GPU opening failed ({e}), falling back to CPU",
                RuntimeWarning
            )
    
    # CPU fallback
    return _compute_opening_cpu(edt_map)


def _compute_opening_gpu(edt_map: np.ndarray) -> np.ndarray:
    """
    GPU implementation using iterative granulometry (Vogel et al., 2010).
    
    Algorithm:
        For each radius r from max_edt down to 1:
            1. Find centers where EDT >= r
            2. Dilate these centers with a sphere of radius r
            3. Assign diameter (2*r) to dilated region (keeping max value)
    """
    import cupy as cp
    from cupyx.scipy.ndimage import binary_dilation
    
    # Transfer to GPU
    edt_gpu = cp.asarray(edt_map, dtype=cp.float32)
    opening_map = cp.zeros_like(edt_gpu)
    max_r = int(edt_gpu.max())
    
    print(f"  Computing Opening Map (GPU Iterative, Max Radius: {max_r})...")
    
    # Iterative granulometry: process each radius
    for r in range(1, max_r + 1):
        # 1. Identify centers where spheres of radius r can fit
        centers = (edt_gpu >= r)
        if not cp.any(centers):
            continue
        
        # 2. Create spherical structuring element of radius r
        z, y, x = cp.ogrid[-r:r+1, -r:r+1, -r:r+1]
        sphere = (z**2 + y**2 + x**2) <= r**2
        
        # 3. Dilate centers to reconstruct spheres
        spheres_mask = binary_dilation(centers, structure=sphere)
        
        # 4. Update opening map: assign diameter (2*r) keeping maximum
        mask_indices = spheres_mask > 0
        opening_map[mask_indices] = cp.maximum(
            opening_map[mask_indices], 
            2.0 * r
        )
        
        # Progress for large volumes
        if max_r > 50 and r % 10 == 0:
            print(f"    Progress: {r}/{max_r} radii processed")
    
    # Transfer back to CPU
    opening_cpu = cp.asnumpy(opening_map).astype(np.float32)
    
    # Free GPU memory
    del edt_gpu, opening_map, centers, sphere, spheres_mask
    cp.get_default_memory_pool().free_all_blocks()
    
    return opening_cpu


def _compute_opening_cpu(edt_map: np.ndarray) -> np.ndarray:
    """
    CPU implementation using iterative granulometry (Vogel et al., 2010).
    
    Algorithm:
        For each radius r from max_edt down to 1:
            1. Find centers where EDT >= r
            2. Dilate these centers with a sphere of radius r
            3. Assign diameter (2*r) to dilated region (keeping max value)
    """
    from scipy.ndimage import binary_dilation
    
    opening_map = np.zeros_like(edt_map, dtype=np.float32)
    max_r = int(edt_map.max())
    
    print(f"  Computing Opening Map (CPU Iterative, Max Radius: {max_r})...")
    
    # Iterative granulometry: process each radius
    for r in range(1, max_r + 1):
        # 1. Identify centers where spheres of radius r can fit
        centers = (edt_map >= r)
        if not np.any(centers):
            continue
        
        # 2. Create spherical structuring element of radius r
        z, y, x = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
        sphere = (z**2 + y**2 + x**2) <= r**2
        
        # 3. Dilate centers to reconstruct spheres
        spheres_mask = binary_dilation(centers, structure=sphere)
        
        # 4. Update opening map: assign diameter (2*r) keeping maximum
        mask_indices = spheres_mask > 0
        opening_map[mask_indices] = np.maximum(
            opening_map[mask_indices], 
            2.0 * r
        )
        
        # Progress for large volumes
        if max_r > 50 and r % 10 == 0:
            print(f"    Progress: {r}/{max_r} radii processed")
    
    return opening_map


def opening_to_diameter(opening_map: np.ndarray) -> np.ndarray:
    """
    Convert opening map to pore diameter map.
    
    Note: After the iterative granulometry fix, the opening map already
    contains diameters (2×r), so this function now returns the input unchanged.
    Kept for API compatibility.
    
    Args:
        opening_map: Local thickness map (already contains diameters)
    
    Returns:
        diameter_map: Pore diameters (same as input)
    """
    # Opening map now contains diameters directly (after granulometry fix)
    # No multiplication needed
    return opening_map


def compute_local_thickness(
    binary_volume: np.ndarray,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    use_gpu: bool = True
) -> np.ndarray:
    """
    End-to-end: Binary volume → EDT → Opening → Diameter.
    
    Convenience function combining distance_transform + opening_map.
    Note: Opening map now directly contains diameters after granulometry fix.
    
    Args:
        binary_volume: 3D boolean volume (True = pore)
        voxel_spacing: Physical voxel dimensions (dz, dy, dx)
        use_gpu: GPU acceleration flag
    
    Returns:
        diameter_map: Local pore diameter at each voxel (float32).
                     Values represent diameter of maximal inscribed sphere.
    
    Example:
        >>> volume = np.random.rand(100, 100, 100) > 0.7
        >>> diameters = compute_local_thickness(volume, (2, 1, 1))
        >>> print(f"Mean pore diameter: {diameters[volume].mean():.2f} μm")
    """
    try:
        from .distance_transform import compute_edt
    except ImportError:
        # Fallback for standalone execution
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from distance_transform import compute_edt
    
    # Step 1: EDT
    edt = compute_edt(binary_volume, voxel_spacing, use_gpu)
    
    # Step 2: Opening map
    opening = compute_opening_map(edt, use_gpu)
    
    # Step 3: Convert to diameter
    diameter = opening_to_diameter(opening)
    
    return diameter


if __name__ == "__main__":
    # Validation test
    print("Local Thickness Module - Validation Test")
    print("=" * 60)
    
    # Create synthetic volume: single sphere
    size = 64
    volume = np.zeros((size, size, size), dtype=bool)
    center = size // 2
    radius = 15
    
    for z in range(size):
        for y in range(size):
            for x in range(size):
                dist = np.sqrt(
                    (z - center)**2 + 
                    (y - center)**2 + 
                    (x - center)**2
                )
                if dist <= radius:
                    volume[z, y, x] = True
    
    print(f"Test volume: {volume.shape}, pore fraction: {volume.mean():.3f}")
    print(f"Expected max diameter: ~{2*radius}")
    
    # Compute local thickness
    try:
        from .distance_transform import compute_edt
    except ImportError:
        # Fallback for standalone testing
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from distance_transform import compute_edt
    
    print("\n[Computing EDT...]")
    edt = compute_edt(volume, use_gpu=False)  # Use CPU for test
    print(f"  EDT max: {edt.max():.2f}")
    
    print("\n[Computing Opening Map...]")
    opening = compute_opening_map(edt, use_gpu=False)
    print(f"  Opening max: {opening.max():.2f}")
    
    print("\n[Diameter Map...]")
    diameter = opening_to_diameter(opening)
    print(f"  Diameter max: {diameter.max():.2f}")
    
    # Check center voxel (should be close to full diameter)
    center_diameter = diameter[center, center, center]
    print(f"\nCenter voxel diameter: {center_diameter:.2f}")
    print(f"Expected: ~{2*radius}")
    
    print("\n" + "=" * 60)
    print("Module validation complete.")
    print("\n⚠️  Note: Opening map now directly contains diameters (2×r)")
    print("    from iterative granulometry. No multiplication needed.")
