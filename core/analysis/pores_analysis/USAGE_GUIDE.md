# PSD Calculator - Complete Usage Guide
## From Installation to Results

This guide walks you through the complete workflow for computing Pore Size Distribution (PSD) from 3D micro-CT volumes.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Data Preparation](#2-data-preparation)
3. [Basic Usage](#3-basic-usage)
4. [Advanced Features](#4-advanced-features)
5. [Validation](#5-validation)
6. [Troubleshooting](#6-troubleshooting)
7. [Scientific Background](#7-scientific-background)

---

## 1. Installation

### Option A: Google Colab (Recommended)

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install numpy scipy pandas scikit-image matplotlib

# Install CuPy for GPU acceleration
!pip install cupy-cuda11x  # Use cuda12x for CUDA 12
```

### Option B: Local Installation

```bash
# Clone or download the pores_analysis folder
cd express_annotation/pores_analysis

# Install requirements
pip install -r requirements.txt

# Optional: GPU support
pip install cupy-cuda11x
```

---

## 2. Data Preparation

### Input Requirements

Your input volume **must** be:
- ✅ **3D binary array** (Z, Y, X)
- ✅ **Boolean dtype**: `True` = pore space, `False` = solid phase
- ✅ **Pre-segmented**: Already thresholded and labeled

### Loading Data

```python
import numpy as np

# From numpy file
volume = np.load('segmented_scan.npy')

# From TIFF stack
import tifffile
volume = tifffile.imread('scan_stack/*.tif')
volume = volume > threshold  # Threshold if needed
volume = volume.astype(bool)

# Verify
print(f"Shape: {volume.shape}")
print(f"Dtype: {volume.dtype}")
print(f"Porosity: {volume.mean():.4f}")
```

---

## 3. Basic Usage

### Minimal Example

```python
from pores_analysis import compute_psd, psd_to_dataframe, save_psd_dataframe

# Compute PSD
psd = compute_psd(
    volume,
    voxel_spacing=(2.0, 1.0, 1.0),  # μm (Z, Y, X)
    use_gpu=True
)

# Convert to DataFrame
df = psd_to_dataframe(psd)

# Save results
save_psd_dataframe(df, 'results/psd.csv')
```

### Understanding the Output

```python
print(df.head())
```

Output columns:
- **`Diameter_px`**: Pore diameter in voxel units
- **`Diameter_um`**: Pore diameter in micrometers
- **`Volume_Count`**: Number of voxels in this diameter bin
- **`Cumulative_Porosity`**: Cumulative volume fraction [0-1]
- **`Differential_PSD`**: dV/dd (differential distribution)
- **`is_reliable`**: **Vogel et al. constraint** - `True` if d ≥ 5 voxels

### Filtering Reliable Data

```python
# Only use bins with d >= 5 voxels (Vogel et al., 2010)
reliable_df = df[df['is_reliable']]

print(f"Reliable bins: {len(reliable_df)}/{len(df)}")
```

---

## 4. Advanced Features

### 4.1 Large Volumes (Block Processing)

For volumes > 512³ voxels or limited GPU memory:

```python
psd = compute_psd(
    volume,
    voxel_spacing=(2.0, 1.0, 1.0),
    use_chunking=True,
    chunk_size=(128, 128, 128),  # Adjust for your GPU
    halo_width=50,               # >= max pore diameter
)
```

**Halo Width Guide**:
- Small pores (< 20 voxels): `halo_width = 40`
- Medium pores (20-50 voxels): `halo_width = 100`
- Large pores (> 50 voxels): `halo_width = 2 × max_diameter`

### 4.2 Checkpointing (Colab Resilience)

Enable automatic checkpointing to survive disconnections:

```python
psd = compute_psd(
    volume,
    voxel_spacing=(2.0, 1.0, 1.0),
    use_chunking=True,
    checkpoint_dir='/content/drive/MyDrive/checkpoints',
    run_id='scan_001',
    resume=True  # Resume from last checkpoint
)
```

### 4.3 Custom Bin Edges

Define custom diameter bins:

```python
import numpy as np

# Logarithmic bins from 1 to 100 μm
custom_bins = np.logspace(0, 2, 50)

psd = compute_psd(
    volume,
    voxel_spacing=(2.0, 1.0, 1.0),
    bin_edges=custom_bins
)
```

### 4.4 Visualization

```python
from pores_analysis import plot_psd

# Generate plot
plot_psd(
    df,
    show_unreliable=False,  # Grey out d < 5 voxels
    log_scale=True,
    save_path='results/psd_plot.png'
)
```

---

## 5. Validation

### Run Synthetic Tests

Before processing real data, validate the pipeline:

```bash
cd pores_analysis
python run_tests.py
```

Or in Google Colab:
```python
%cd /content/drive/MyDrive/path/to/pores_analysis
!python run_tests.py
```

This runs 5 validation tests:
1. Single sphere (known diameter)
2. Nested spheres (multiple peaks)
3. Cubic lattice (periodic structure)
4. Anisotropic voxels
5. Reliability flag validation

**Expected output**: All tests should show `✓ PASS`

### Quick Example

```python
# In Google Colab or Python environment
import sys
sys.path.insert(0, '/content/drive/MyDrive/path/to/pores_analysis')

from test_psd_synthetic import example_synthetic_data
example_synthetic_data()
```

Or use the example workflow runner:
```bash
python example_workflow.py
```

---

## 6. Troubleshooting

### Problem: "ImportError: attempted relative import"

**Cause**: Running modules directly that use relative imports

**Solution 1** (Recommended): Use the provided test runners
```bash
python run_tests.py        # For validation tests
python example_workflow.py  # For examples
```

**Solution 2**: Add to Python path before importing
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path('/path/to/pores_analysis').absolute()))

# Now imports work
from psd_calculator import compute_psd
```

**Solution 3**: In Google Colab
```python
# Navigate to directory
%cd /content/drive/MyDrive/path/to/pores_analysis

# Run tests
!python run_tests.py
```

### Problem: GPU Out of Memory

**Solution**: Reduce chunk size or use CPU

```python
psd = compute_psd(
    volume,
    use_gpu=False,  # Force CPU
    # OR
    chunk_size=(64, 64, 64)  # Smaller chunks
)
```

### Problem: "Module not found" errors

**Solution**: Install missing dependencies

```bash
pip install scikit-image  # For morphological reconstruction
pip install matplotlib    # For plotting
```

### Problem: Unrealistic PSD results

**Checklist**:
1. ✅ Is `voxel_spacing` correct? (Check μCT metadata)
2. ✅ Is volume pre-segmented? (Boolean, not raw greyscale)
3. ✅ Is `True` = pore space? (Not inverted)
4. ✅ Is `halo_width` large enough? (>= max pore diameter)

### Problem: Very slow computation

**Optimization**:
```python
# Enable GPU
psd = compute_psd(volume, use_gpu=True)

# Use chunking even for moderate volumes
psd = compute_psd(volume, use_chunking=True, chunk_size=(128,128,128))
```

---

## 7. Scientific Background

### Vogel et al. (2010) Constraints

This implementation strictly follows:

1. **Reliability Threshold**: d < 5 voxels marked unreliable
   - *Reason*: Discretization errors dominate at small scales
   - *Action*: Always filter with `df[df['is_reliable']]`

2. **Border Exclusion**: Pores touching edges excluded
   - *Reason*: Unknown extent outside volume
   - *Action*: Automatic (can disable with `exclude_borders=False`)

3. **26-Connectivity**: Face+edge+corner connectivity
   - *Standard*: Used for 3D pore space continuity

4. **Spherical Assumption**: Maximal inscribed sphere
   - *Metric*: Hydraulic diameter (volume-equivalent sphere)

### Pipeline Steps

```
Binary Volume
    ↓ [1. Border Masking]
Masked Volume
    ↓ [2. Euclidean Distance Transform]
EDT Map (distances to solid phase)
    ↓ [3. Morphological Opening (Hildebrand & Ruegsegger)]
Opening Map (local thickness = 2×radius)
    ↓ [4. Histogram Binning]
PSD DataFrame
    ↓ [5. Apply Vogel Constraints]
Validated Results
```

### Key References

- **Vogel, H. J., et al. (2010)**. *Computers & Geosciences*, 36(10), 1236-1245.
- **Hildebrand, T., & Rüegsegger, P. (1997)**. *Journal of Microscopy*, 185(1), 67-75.

---

## Quick Reference Card

### Essential Commands

```python
# Minimal workflow
from pores_analysis import compute_psd, psd_to_dataframe, save_psd_dataframe

volume = np.load('scan.npy')
psd = compute_psd(volume, voxel_spacing=(2,1,1), use_gpu=True)
df = psd_to_dataframe(psd)
save_psd_dataframe(df, 'results/psd.csv')

# Large volume workflow (Colab + GPU)
psd = compute_psd(
    volume,
    voxel_spacing=(2,1,1),
    use_gpu=True,
    use_chunking=True,
    chunk_size=(128,128,128),
    checkpoint_dir='/content/drive/MyDrive/checkpoints',
    resume=True
)

# Filter reliable data
reliable = df[df['is_reliable']]
```

---

## Need Help?

1. **Check imports first**: `python check_imports.py`
2. **Run validation tests**: `python run_tests.py`
3. **Review examples**: `python example_workflow.py`
4. Check [README.md](README.md) for module documentation
5. Inspect configuration template: `config_template.py`

### Common Issues

**Import errors when running scripts**:
```python
# Solution: Add pores_analysis to Python path
import sys
sys.path.insert(0, '/path/to/pores_analysis')
```

**In Google Colab**:
```python
%cd /content/drive/MyDrive/path/to/pores_analysis
!python run_tests.py
```

---

**Last Updated**: February 2026  
**Version**: 1.0.0
