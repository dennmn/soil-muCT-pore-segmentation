"""
Euclidean Distance Transform Module
====================================

Computes anisotropic EDT on 3D binary volumes with GPU acceleration (CuPy) 
and CPU fallback (SciPy). Handles arbitrary voxel spacing for μCT data.

Hardware Strategy:
    - Priority: CuPy (CUDA) for GPU acceleration
    - Fallback: SciPy (CPU) if GPU unavailable or OOM
    - Automatic detection and graceful degradation
"""

import numpy as np
from typing import Tuple, Optional
import warnings


def _check_gpu_available() -> bool:
    """Check if CuPy and CUDA are available."""
    try:
        import cupy as cp
        # Test GPU access
        _ = cp.cuda.Device(0).compute_capability
        return True
    except (ImportError, RuntimeError):
        return False


def compute_edt(
    binary_volume: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    use_gpu: bool = True,
    return_indices: bool = False
) -> np.ndarray:
    """
    Compute Euclidean Distance Transform on a 3D binary volume.
    
    The EDT assigns each background voxel the distance to the nearest 
    foreground voxel. For pore analysis, background = pore space (True/1),
    foreground = solid phase (False/0).
    
    Args:
        binary_volume: 3D boolean or binary array (Z, Y, X).
                      True/1 = pore space, False/0 = solid phase.
        voxel_spacing: Physical spacing (dz, dy, dx) in micrometers.
                      Default (1,1,1) for isotropic unit voxels.
        use_gpu: If True, attempt GPU computation via CuPy.
                If False or GPU unavailable, fall back to SciPy.
        return_indices: If True, also return the indices of nearest 
                       solid voxels (not implemented yet).
    
    Returns:
        edt_map: Float32 array (Z, Y, X) with distances in physical units.
                 Each pore voxel contains distance to nearest solid voxel.
    
    Notes:
        - Anisotropic spacing is critical for μCT data (often Z ≠ X,Y)
        - Memory: Requires ~4 bytes/voxel for output (float32)
        - GPU memory limits: Falls back to CPU if GPU OOM
    
    Raises:
        ValueError: If input is not 3D or voxel_spacing has wrong shape.
        RuntimeError: If both GPU and CPU methods fail.
    
    Examples:
        >>> volume = np.random.rand(100, 100, 100) > 0.7  # 30% pore space
        >>> edt = compute_edt(volume, voxel_spacing=(2.0, 1.0, 1.0))
        >>> print(f"Max pore radius: {edt.max() / 2:.2f} μm")
    """
    # Validation
    if binary_volume.ndim != 3:
        raise ValueError(
            f"Expected 3D volume, got shape {binary_volume.shape}"
        )
    
    if len(voxel_spacing) != 3:
        raise ValueError(
            f"voxel_spacing must have 3 elements, got {len(voxel_spacing)}"
        )
    
    if return_indices:
        raise NotImplementedError(
            "return_indices feature not yet implemented"
        )
    
    # Attempt GPU computation
    if use_gpu and _check_gpu_available():
        try:
            return _compute_edt_gpu(binary_volume, voxel_spacing)
        except Exception as e:
            warnings.warn(
                f"GPU EDT failed ({e}), falling back to CPU",
                RuntimeWarning
            )
    
    # CPU fallback
    return _compute_edt_cpu(binary_volume, voxel_spacing)


def _compute_edt_gpu(
    binary_volume: np.ndarray,
    voxel_spacing: Tuple[float, float, float]
) -> np.ndarray:
    """GPU implementation using CuPy."""
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt
    
    # Transfer to GPU
    volume_gpu = cp.asarray(binary_volume, dtype=cp.bool_)
    
    # Compute EDT with anisotropic spacing
    edt_gpu = distance_transform_edt(
        volume_gpu,
        sampling=voxel_spacing
    )
    
    # Transfer back to CPU as float32
    edt_cpu = cp.asnumpy(edt_gpu).astype(np.float32)
    
    # Free GPU memory explicitly
    del volume_gpu, edt_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    return edt_cpu


def _compute_edt_cpu(
    binary_volume: np.ndarray,
    voxel_spacing: Tuple[float, float, float]
) -> np.ndarray:
    """CPU implementation using SciPy."""
    from scipy.ndimage import distance_transform_edt
    
    # SciPy EDT (returns float64 by default)
    edt = distance_transform_edt(
        binary_volume,
        sampling=voxel_spacing
    )
    
    # Convert to float32 to save memory
    return edt.astype(np.float32)


def compute_edt_chunked(
    binary_volume: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    chunk_size: Tuple[int, int, int] = (128, 128, 128),
    halo_width: int = 50,
    use_gpu: bool = True
) -> np.ndarray:
    """
    Compute EDT using block processing with halo overlap.
    
    This function is a placeholder for integration with block_processor.py.
    The actual implementation will be delegated to BlockProcessor class.
    
    Args:
        binary_volume: 3D boolean volume
        voxel_spacing: Physical voxel dimensions
        chunk_size: Size of each processing block (before halo)
        halo_width: Overlap padding width (must be >= max pore diameter)
        use_gpu: GPU acceleration flag
    
    Returns:
        Full EDT map assembled from processed chunks
    
    See Also:
        block_processor.BlockProcessor - Full chunked processing implementation
    """
    # For now, delegate to monolithic compute
    # Will be replaced by BlockProcessor integration
    return compute_edt(binary_volume, voxel_spacing, use_gpu)


if __name__ == "__main__":
    # Quick validation test
    print("EDT Module - Quick Test")
    print("=" * 60)
    
    # Create synthetic volume: single sphere in center
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
    
    # Test GPU
    if _check_gpu_available():
        print("\n[GPU Test]")
        edt_gpu = compute_edt(volume, voxel_spacing=(1, 1, 1), use_gpu=True)
        print(f"  Max distance: {edt_gpu.max():.2f} (expected ~{radius})")
        print(f"  Memory: {edt_gpu.nbytes / 1024**2:.2f} MB")
    else:
        print("\n[GPU] Not available")
    
    # Test CPU
    print("\n[CPU Test]")
    edt_cpu = compute_edt(volume, voxel_spacing=(1, 1, 1), use_gpu=False)
    print(f"  Max distance: {edt_cpu.max():.2f} (expected ~{radius})")
    print(f"  Memory: {edt_cpu.nbytes / 1024**2:.2f} MB")
    
    print("\n" + "=" * 60)
    print("Module validation complete.")
