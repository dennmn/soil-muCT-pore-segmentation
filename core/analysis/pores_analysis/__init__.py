"""
Pore Size Distribution Calculator Package
==========================================

Robust pipeline for computing PSD from 3D micro-CT volumes using 
Minkowski Functionals and Morphological Opening (Vogel et al., 2010).

Core Modules:
    - distance_transform: Anisotropic EDT with GPU/CPU support
    - local_thickness: Morphological opening (Hildebrand & Ruegsegger)
    - psd_calculator: Main pipeline with Vogel constraints
    - psd_output: DataFrame formatter and export
    - block_processor: Chunked processing with halo overlap
    - checkpoint_manager: Colab timeout resilience

Quick Start:
    >>> from pores_analysis import compute_psd, psd_to_dataframe, save_psd_dataframe
    >>> 
    >>> # Load your segmented volume (boolean array, True = pore)
    >>> volume = np.load('segmented_scan.npy')
    >>> 
    >>> # Compute PSD
    >>> psd = compute_psd(
    ...     volume,
    ...     voxel_spacing=(2.0, 1.0, 1.0),  # μm
    ...     use_gpu=True,
    ...     use_chunking=True
    ... )
    >>> 
    >>> # Export results
    >>> df = psd_to_dataframe(psd)
    >>> save_psd_dataframe(df, 'results/psd.csv')

Scientific Constraints (Vogel et al., 2010):
    - Reliability flag: diameters below the configured `min_reliable_diameter_voxels` threshold are marked unreliable
    - Border exclusion: Edge-touching pores excluded
    - 26-connectivity: Standard 3D connectivity
    - Spherical assumption: Hydraulic diameter metric
"""

__version__ = "1.0.0"
__author__ = "PSD Calculator Team"

from .distance_transform import compute_edt
from .local_thickness import compute_opening_map, compute_local_thickness
from .psd_calculator import compute_psd, mask_border_voxels
from .psd_output import (
    psd_to_dataframe,
    save_psd_dataframe,
    load_psd_dataframe,
    plot_psd
)
from .block_processor import BlockProcessor
from .checkpoint_manager import CheckpointManager

__all__ = [
    # Main API
    'compute_psd',
    'psd_to_dataframe',
    'save_psd_dataframe',
    'load_psd_dataframe',
    'plot_psd',
    
    # Advanced components
    'compute_edt',
    'compute_opening_map',
    'compute_local_thickness',
    'BlockProcessor',
    'CheckpointManager',
    'mask_border_voxels',
]
