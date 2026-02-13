# Pore Size Distribution Calculator

Scientific Python pipeline for computing Pore Size Distribution (PSD) from 3D soil micro-CT volumes using Minkowski Functionals and Morphological Opening.

## Features

✅ **Anisotropic EDT** - CuPy (GPU) with SciPy (CPU) fallback  
✅ **Block Processing** - Halo overlap for boundary-correct EDT on large volumes  
✅ **Checkpointing** - Google Drive persistence for Colab timeout resilience  
✅ **Vogel et al. (2010) Constraints** - Reliability flags, border masking, 26-connectivity  
✅ **Multiple Output Formats** - CSV, JSON, HDF5, Excel with metadata preservation  

## Installation

```bash
# Core dependencies
pip install numpy scipy pandas

# Optional: GPU acceleration
pip install cupy-cuda11x  # or cupy-cuda12x

# Optional: Advanced features
pip install scikit-image matplotlib openpyxl tables h5py
```

## Quick Start

```python
import numpy as np
from pores_analysis import compute_psd, psd_to_dataframe, save_psd_dataframe

# Load your segmented 3D volume (boolean array: True = pore, False = solid)
volume = np.load('segmented_scan.npy')

# Compute PSD with Vogel et al. constraints
psd = compute_psd(
    volume,
    voxel_spacing=(2.0, 1.0, 1.0),  # Anisotropic spacing in μm (Z, Y, X)
    use_gpu=True,                   # CuPy acceleration
    use_chunking=True,              # Block processing for large volumes
    checkpoint_dir='/content/drive/MyDrive/checkpoints',  # Colab resilience
    run_id='scan_001'
)

# Convert to DataFrame
df = psd_to_dataframe(psd)

# Filter reliable data (d >= 5 voxels per Vogel et al.)
reliable_df = df[df['is_reliable']]

# Export results
save_psd_dataframe(df, 'results/scan_001_psd.csv')
```

## Module Structure

- **`distance_transform.py`** - Anisotropic EDT with GPU/CPU support
- **`local_thickness.py`** - Hildebrand & Ruegsegger morphological reconstruction
- **`block_processor.py`** - Chunked processing with halo overlap (padding→compute→crop)
- **`checkpoint_manager.py`** - Drive persistence for Colab timeout recovery
- **`psd_calculator.py`** - Main pipeline orchestrator with Vogel constraints
- **`psd_output.py`** - DataFrame formatter, export, and plotting
- **`test_psd_synthetic.py`** - Validation suite with synthetic volumes

## Scientific Constraints (Vogel et al., 2010)

1. **Reliability Flag**: Diameters < 5 voxels marked unreliable (column `is_reliable`)
2. **Border Exclusion**: Pores touching volume edges excluded from statistics
3. **26-Connectivity**: Face+edge+corner connectivity for 3D pore space
4. **Spherical Assumption**: Hydraulic diameter via maximal inscribed sphere

## Output DataFrame Columns

| Column | Description |
|--------|-------------|
| `Diameter_px` | Pore diameter in voxel units |
| `Diameter_um` | Pore diameter in micrometers |
| `Volume_Count` | Number of voxels per bin |
| `Cumulative_Porosity` | Cumulative volume fraction [0-1] |
| `Differential_PSD` | dV/dd (μm⁻¹) |
| `is_reliable` | `True` if d ≥ 5 voxels (Vogel constraint) |

## Block Processing (Large Volumes)

For volumes exceeding GPU memory:

```python
psd = compute_psd(
    volume,
    voxel_spacing=(2.0, 1.0, 1.0),
    use_chunking=True,
    chunk_size=(128, 128, 128),     # Core block size
    halo_width=50,                  # >= max expected pore diameter
    checkpoint_dir='/content/drive/MyDrive/checkpoints',
    resume=True                     # Resume from last checkpoint
)
```

**Halo Logic**: Each block is padded with overlap → EDT computed → halo cropped. This ensures boundary-correct distance measurements.

## Testing

Run validation suite with synthetic volumes:

```bash
cd pores_analysis
python run_tests.py
```

Or in Google Colab:
```python
%cd /content/drive/MyDrive/path/to/pores_analysis
!python run_tests.py
```

Tests include:
- Single sphere (known diameter)
- Nested spheres (multiple peaks)
- Cubic lattice (periodic structure)
- Anisotropic voxels
- Reliability flag validation

## References

- **Vogel, H. J., Weller, U., & Schlüter, S. (2010)**. Quantification of soil structure based on Minkowski functions. *Computers & Geosciences*, 36(10), 1236-1245.
- **Hildebrand, T., & Rüegsegger, P. (1997)**. A new method for the model-independent assessment of thickness in three-dimensional images. *Journal of Microscopy*, 185(1), 67-75.

## License

MIT License - See LICENSE file for details.

## Authors

PSD Calculator Team - February 2026
