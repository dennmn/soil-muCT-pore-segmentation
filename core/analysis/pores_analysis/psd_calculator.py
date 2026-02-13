"""
Pore Size Distribution Calculator
==================================

Main pipeline for computing PSD via Minkowski Functionals and Morphological Opening.
Strictly follows methodology of Vogel et al. (2010).

Scientific Constraints (Vogel et al., 2010):
    1. Reliability Flag: Diameters below the configured `min_reliable_diameter_voxels` threshold are flagged as unreliable
    2. Border Exclusion: Pores touching volume edges must be excluded
    3. Shape Assumption: Spherical structuring elements (Hydraulic Diameter)
    4. Connectivity: 26-connectivity (face+edge+corner) for 3D pore space

References:
    Vogel, H. J., Weller, U., & Schlüter, S. (2010). Quantification of soil 
    structure based on Minkowski functions. Computers & Geosciences, 36(10), 1236-1245.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import warnings
import sys

# Handle imports for both package and script execution
try:
    from .distance_transform import compute_edt
    from .local_thickness import compute_opening_map, opening_to_diameter
    from .block_processor import BlockProcessor
    from .checkpoint_manager import CheckpointManager
except ImportError:
    # Fallback for direct script execution
    sys.path.insert(0, str(Path(__file__).parent))
    from distance_transform import compute_edt
    from local_thickness import compute_opening_map, opening_to_diameter
    from block_processor import BlockProcessor
    from checkpoint_manager import CheckpointManager

try:
    from .config_loader import load_config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from config_loader import load_config

CONFIG = load_config()
PATH_CONFIG = CONFIG.get("paths", {})
IMAGE_CONFIG = CONFIG.get("image_params", {})
PROCESSING_CONFIG = CONFIG.get("processing_thresholds", {})
OUTPUT_CONFIG = CONFIG.get("output_settings", {})

DEFAULT_PHYSICAL_VOXEL_SPACING = tuple(IMAGE_CONFIG.get("voxel_spacing", (1.0, 1.0, 1.0)))
INTERNAL_VOXEL_SPACING: Tuple[float, float, float] = (1.0, 1.0, 1.0)
DEFAULT_USE_GPU = PROCESSING_CONFIG.get("use_gpu", True)
DEFAULT_USE_CHUNKING = PROCESSING_CONFIG.get("use_chunking", False)
_CHUNK_SIZE_CFG = PROCESSING_CONFIG.get("chunk_size", (128, 128, 128))
DEFAULT_CHUNK_SIZE = tuple(_CHUNK_SIZE_CFG)
DEFAULT_HALO_WIDTH = PROCESSING_CONFIG.get("halo_width", 50)
DEFAULT_BIN_EDGES = PROCESSING_CONFIG.get("bin_edges_um")
DEFAULT_EXCLUDE_BORDERS = PROCESSING_CONFIG.get("exclude_border_pores", True)
DEFAULT_RESUME = PROCESSING_CONFIG.get("resume_from_checkpoint", False)
DEFAULT_CHECKPOINT_DIR = PATH_CONFIG.get("checkpoint_dir")
DEFAULT_RUN_ID = PATH_CONFIG.get("default_run_id")
DEFAULT_BORDER_WIDTH = PROCESSING_CONFIG.get("border_width", 1)
MIN_RELIABLE_DIAMETER_VOXELS = PROCESSING_CONFIG.get("min_reliable_diameter_voxels", 5)



def mask_border_voxels(
    binary_volume: np.ndarray,
    border_width: Optional[int] = None
) -> np.ndarray:
    """
    Mask voxels touching volume boundaries.
    
    Pores intersecting boundaries cannot be accurately sized, as their 
    full extent is unknown. These must be excluded from PSD analysis.
    
    Args:
        binary_volume: 3D boolean volume (True = pore)
        border_width: Width of border region to mask (default defined in config)
    
    Returns:
        masked_volume: Volume with borders set to False (solid)
    
    Example:
        >>> volume = np.ones((100, 100, 100), dtype=bool)
        >>> masked = mask_border_voxels(volume)
        >>> assert not masked[0, :, :].any()  # Z=0 border masked
    """
    if border_width is None:
        border_width = DEFAULT_BORDER_WIDTH

    masked = binary_volume.copy()
    
    # Mask all six faces
    masked[:border_width, :, :] = False  # Z min
    masked[-border_width:, :, :] = False  # Z max
    masked[:, :border_width, :] = False  # Y min
    masked[:, -border_width:, :] = False  # Y max
    masked[:, :, :border_width] = False  # X min
    masked[:, :, -border_width:] = False  # X max
    
    return masked


def compute_psd_from_opening_map(
    opening_map: np.ndarray,
    binary_volume: np.ndarray,
    voxel_spacing: Tuple[float, float, float],
    bin_edges: Optional[np.ndarray] = None,
    exclude_borders: Optional[bool] = None
) -> Dict[str, np.ndarray]:
    """
    Extract PSD from opening map.
    
    Pipeline:
        1. Convert opening map (radius) to diameter map
        2. Mask border voxels if requested
        3. Bin diameters into histogram
        4. Compute cumulative and differential PSD
        5. Apply reliability flags (diameter < configured reliability threshold)
    
    Args:
        opening_map: Local thickness map (radii) from morphological opening
        binary_volume: Original binary volume (for masking)
        voxel_spacing: Physical voxel dimensions (dz, dy, dx) in μm
        bin_edges: Diameter bin edges in physical units. If None, auto-generate.
        exclude_borders: If True, exclude border-touching voxels
    
    Returns:
        Dictionary with:
            - 'bin_centers_px': Bin centers in voxels
            - 'bin_centers_um': Bin centers in micrometers
            - 'bin_edges_um': Bin edges in micrometers
            - 'volume_counts': Number of voxels per bin
            - 'cumulative_volume': Cumulative volume fraction
            - 'differential_volume': Differential PSD (dV/dd)
            - 'reliability_flag': Boolean array (True = reliable, diameter >= configured reliability threshold)
            - 'total_pore_voxels': Total pore voxels analyzed
    
    Notes:
        - Bin edges are in physical units (μm)
        - Reliability flag follows Vogel et al. constraint
        - Excluded border voxels not counted in statistics
    """
    # Step 1: Convert to diameter
    diameter_map = opening_to_diameter(opening_map)
    
    if exclude_borders is None:
        exclude_borders = DEFAULT_EXCLUDE_BORDERS

    # Step 2: Mask borders if requested
    if exclude_borders:
        masked_volume = mask_border_voxels(binary_volume)
    else:
        masked_volume = binary_volume
    
    # Only analyze pore voxels (not border-masked)
    pore_mask = masked_volume & (diameter_map > 0)
    pore_diameters = diameter_map[pore_mask]
    
    if len(pore_diameters) == 0:
        warnings.warn("No pore voxels found after masking", UserWarning)
        return _empty_psd_result()
    
    # Step 3: Determine bin edges
    voxel_scale_um = float(np.mean(voxel_spacing))
    if bin_edges is None:
        # Auto-generate logarithmic bins in voxel units
        min_diameter = pore_diameters.min()
        max_diameter = pore_diameters.max()
        n_bins = 50
        bin_edges_px = np.logspace(
            np.log10(max(min_diameter, 0.1)),
            np.log10(max_diameter * 1.1),
            n_bins + 1
        )
    else:
        bin_edges_px = np.array(bin_edges, dtype=np.float32) / voxel_scale_um
    bin_edges_um = bin_edges_px * voxel_scale_um
    
    # Step 4: Histogram (volume counts per bin)
    volume_counts, _ = np.histogram(pore_diameters, bins=bin_edges_px)
    bin_centers_px = (bin_edges_px[:-1] + bin_edges_px[1:]) / 2
    bin_centers_um = bin_centers_px * voxel_scale_um
    
    # Step 5: Cumulative PSD (M0 volume density)
    total_pore_voxels = len(pore_diameters)
    cumulative_volume = np.cumsum(volume_counts) / total_pore_voxels
    
    # Step 6: Differential PSD (dV/dd)
    bin_widths = np.diff(bin_edges_um)
    differential_volume = volume_counts / (total_pore_voxels * bin_widths)
    
    # Step 7: Reliability flag (Vogel et al., 2010 constraint)
    reliability_flag = bin_centers_px >= MIN_RELIABLE_DIAMETER_VOXELS
    
    return {
        'bin_centers_px': bin_centers_px,
        'bin_centers_um': bin_centers_um,
        'bin_edges_um': bin_edges_um,
        'volume_counts': volume_counts,
        'cumulative_volume': cumulative_volume,
        'differential_volume': differential_volume,
        'reliability_flag': reliability_flag,
        'total_pore_voxels': total_pore_voxels,
        'voxel_spacing': voxel_spacing
    }


def _empty_psd_result() -> Dict[str, np.ndarray]:
    """Return empty PSD result for error cases."""
    return {
        'bin_centers_px': np.array([]),
        'bin_centers_um': np.array([]),
        'bin_edges_um': np.array([]),
        'volume_counts': np.array([]),
        'cumulative_volume': np.array([]),
        'differential_volume': np.array([]),
        'reliability_flag': np.array([], dtype=bool),
        'total_pore_voxels': 0,
        'voxel_spacing': DEFAULT_PHYSICAL_VOXEL_SPACING
    }


def compute_psd(
    binary_volume: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = DEFAULT_PHYSICAL_VOXEL_SPACING,
    bin_edges: Optional[np.ndarray] = None,
    use_gpu: bool = DEFAULT_USE_GPU,
    use_chunking: bool = DEFAULT_USE_CHUNKING,
    chunk_size: Tuple[int, int, int] = DEFAULT_CHUNK_SIZE,
    halo_width: int = DEFAULT_HALO_WIDTH,
    checkpoint_dir: Optional[str] = DEFAULT_CHECKPOINT_DIR,
    run_id: Optional[str] = DEFAULT_RUN_ID,
    resume: bool = DEFAULT_RESUME
) -> Dict[str, np.ndarray]:
    """
    Complete PSD calculation pipeline.
    
    Pipeline Steps:
        1. Mask border voxels (exclude boundary-touching pores)
        2. Compute EDT with optional chunking
        3. Compute opening map (morphological reconstruction)
        4. Extract PSD with binning and statistics
        5. Apply Vogel et al. reliability constraints
    
    Args:
        binary_volume: 3D boolean array (Z, Y, X). True = pore, False = solid.
        voxel_spacing: Physical voxel dimensions (dz, dy, dx) in micrometers.
        bin_edges: Optional custom bin edges for PSD histogram (in μm).
        use_gpu: GPU acceleration flag (CuPy if available).
        use_chunking: If True, use block processing for large volumes.
        chunk_size: Block size for chunked processing.
        halo_width: Overlap width for chunked processing (>= max pore diameter).
        checkpoint_dir: Directory for checkpoints (Colab resilience).
        run_id: Unique run identifier for checkpointing.
        resume: If True, attempt to resume from checkpoint.
    
    Returns:
        PSD dictionary (see compute_psd_from_opening_map docstring)
    
    Raises:
        ValueError: If input validation fails
        RuntimeError: If processing fails
    
    Example:
        >>> # Load pre-segmented volume
        >>> volume = np.load('segmented_scan.npy')  # Boolean array
        >>> 
        >>> # Compute PSD
        >>> psd = compute_psd(
        ...     volume,
        ...     voxel_spacing=(2.0, 1.0, 1.0),  # Anisotropic
        ...     use_gpu=True,
        ...     use_chunking=True,
        ...     checkpoint_dir='/content/drive/MyDrive/checkpoints'
        ... )
        >>> 
        >>> # Access results
        >>> diameters = psd['bin_centers_um']
        >>> cumulative = psd['cumulative_volume']
        >>> reliable = psd['reliability_flag']
    """
    # Validation
    if binary_volume.ndim != 3:
        raise ValueError(
            f"Expected 3D volume, got shape {binary_volume.shape}"
        )
    
    if binary_volume.dtype != bool:
        warnings.warn(
            f"Converting volume from {binary_volume.dtype} to bool",
            UserWarning
        )
        binary_volume = binary_volume.astype(bool)
    
    print("=" * 60)
    print("PSD Calculator - Pipeline Execution")
    print("=" * 60)
    physical_voxel_spacing = tuple(voxel_spacing)
    internal_voxel_spacing = INTERNAL_VOXEL_SPACING
    print(f"Volume shape: {binary_volume.shape}")
    print(f"Voxel spacing (physical): {physical_voxel_spacing} μm")
    print(f"Porosity: {binary_volume.mean():.4f}")
    print(f"GPU: {use_gpu}, Chunking: {use_chunking}")
    
    # Step 1: EDT computation
    print("\n[1/3] Computing Euclidean Distance Transform...")
    
    if use_chunking:
        # Setup checkpoint manager
        checkpoint_mgr = None
        if checkpoint_dir is not None:
            checkpoint_mgr = CheckpointManager(checkpoint_dir, run_id)
        
        # Define EDT processing function for blocks
        def edt_func(block):
            return compute_edt(block, internal_voxel_spacing, use_gpu=use_gpu)
        
        chunk_size = tuple(chunk_size)

        # Block processing
        processor = BlockProcessor(
            volume_shape=binary_volume.shape,
            chunk_size=chunk_size,
            halo_width=halo_width
        )
        
        mem = processor.get_memory_estimate()
        print(f"  Memory per block: {mem['total_per_block_mb']:.2f} MB")
        print(f"  Total blocks: {len(processor.blocks)}")
        
        edt_map = processor.process_volume(
            binary_volume,
            edt_func,
            checkpoint_manager=checkpoint_mgr,
            resume=resume
        )
    else:
        # Monolithic EDT
        edt_map = compute_edt(binary_volume, internal_voxel_spacing, use_gpu=use_gpu)
    
    print(f"  EDT complete. Max distance: {edt_map.max():.2f} voxels")
    
    # Step 2: Opening map (local thickness)
    print("\n[2/3] Computing Opening Map (Morphological Reconstruction)...")
    opening_map = compute_opening_map(edt_map, use_gpu=use_gpu)
    print(f"  Opening complete. Max diameter: {opening_map.max():.2f} voxels")
    
    # Step 3: Extract PSD
    print("\n[3/3] Extracting Pore Size Distribution...")
    if bin_edges is None and DEFAULT_BIN_EDGES is not None:
        bin_edges = np.array(DEFAULT_BIN_EDGES, dtype=np.float32)
    elif bin_edges is not None:
        bin_edges = np.array(bin_edges, dtype=np.float32)

    psd = compute_psd_from_opening_map(
        opening_map,
        binary_volume,
        physical_voxel_spacing,
        bin_edges=bin_edges
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("PSD Calculation Complete")
    print("=" * 60)
    print(f"Total pore voxels analyzed: {psd['total_pore_voxels']:,}")
    if len(psd['bin_centers_um']) > 0:
        print(f"Diameter range: {psd['bin_centers_um'].min():.2f} - {psd['bin_centers_um'].max():.2f} μm")
    else:
        print("Diameter range: N/A")
    n_reliable = psd['reliability_flag'].sum()
    n_total = len(psd['reliability_flag'])
    print(f"Reliable bins (d >= {MIN_RELIABLE_DIAMETER_VOXELS} voxels): {n_reliable}/{n_total}")
    
    return psd


if __name__ == "__main__":
    # Quick validation test
    print("PSD Calculator - Quick Test")
    print("=" * 60)
    
    # Create synthetic volume: two spheres of different sizes
    size = 128
    volume = np.zeros((size, size, size), dtype=bool)
    
    # Large sphere
    center1 = (40, 64, 64)
    radius1 = 20
    for z in range(size):
        for y in range(size):
            for x in range(size):
                dist = np.sqrt(
                    (z - center1[0])**2 + 
                    (y - center1[1])**2 + 
                    (x - center1[2])**2
                )
                if dist <= radius1:
                    volume[z, y, x] = True
    
    # Small sphere
    center2 = (88, 64, 64)
    radius2 = 10
    for z in range(size):
        for y in range(size):
            for x in range(size):
                dist = np.sqrt(
                    (z - center2[0])**2 + 
                    (y - center2[1])**2 + 
                    (x - center2[2])**2
                )
                if dist <= radius2:
                    volume[z, y, x] = True
    
    print(f"Test volume: {volume.shape}, porosity: {volume.mean():.4f}")
    print(f"Expected peaks: ~{2*radius1} μm and ~{2*radius2} μm")
    
    # Compute PSD (CPU only, no chunking for test)
    psd = compute_psd(
        volume,
        voxel_spacing=(1.0, 1.0, 1.0),
        use_gpu=False,
        use_chunking=False
    )
    
    # Show results
    print("\nPSD Results:")
    if len(psd['bin_centers_um']) > 0:
        print(f"  Bins: {len(psd['bin_centers_um'])}")
        print(f"  Diameter range: {psd['bin_centers_um'].min():.2f} - {psd['bin_centers_um'].max():.2f} μm")
        print(f"  Peak at: {psd['bin_centers_um'][psd['volume_counts'].argmax()]:.2f} μm")
    else:
        print("  No bins were generated (empty result)")
    
    print("\n" + "=" * 60)
    print("Quick test complete.")
