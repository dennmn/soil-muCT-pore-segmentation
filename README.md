# Soil Micro-CT Analysis Pipeline

A modular Python pipeline for processing and analyzing soil micro-computed tomography (μCT) images. The system performs image preprocessing, semantic segmentation, temporal stability correction, and pore size distribution (PSD) analysis.

---

## 1. Project Overview

### Purpose

This pipeline enables quantitative analysis of soil pore structure from 3D μCT scan data. The primary analytical capabilities include:

- **Image Preprocessing**: CLAHE histogram equalization and median filtering for noise reduction
- **Segmentation**: Multi-Otsu thresholding for multi-class semantic labeling (pores, ambiguous, solids)
- **Z-Stability Analysis**: Temporal consistency correction to eliminate slice-to-slice classification noise
- **Binary Collapse**: Conversion of multi-class masks to binary pore/solid representation
- **Pore Size Distribution (PSD)**: Morphological analysis using local thickness and Euclidean distance transforms

### Design Philosophy

The refactored architecture prioritizes:

1. **Modularity**: Each analysis stage operates as an independent module with its own CLI entry point
2. **Reproducibility**: Configuration-driven execution with explicit parameter logging
3. **Flexibility**: Stages can run independently or as part of an integrated pipeline
4. **Validation**: Schema validation and I/O contract enforcement at every stage

---

## 2. Folder Structure

```
project_refactor_v1/
├── run_pipeline.py              # Main CLI entry point (forwards to cli/pipeline.py)
├── run_pipeline.ipynb           # Colab-compatible notebook pipeline runner
├── README.md                    # This documentation
│
├── cli/
│   └── pipeline.py              # Integrated pipeline orchestrator
│
├── config/
│   ├── pipeline.yaml            # Default 3-class pipeline configuration
│   ├── 2_class_pipeline.yaml    # Alternative 2-class configuration
│   ├── pores_analysis/
│   │   └── config.yaml          # PSD module standalone configuration
│   ├── z_stability/
│   │   └── pipeline.yaml        # Z-stability standalone configuration
│   ├── templates/               # Configuration templates
│   └── requirements/            # Module-specific requirements files
│
├── core/
│   ├── preprocessing/
│   │   └── preprocess_ct_images.py   # CLAHE + median filter preprocessing
│   │
│   ├── segmentation/
│   │   └── multiotsu.py              # Multi-Otsu thresholding segmentation
│   │
│   ├── stability/
│   │   ├── run_pipeline.py           # Z-stability CLI entry point
│   │   ├── src/
│   │   │   ├── io_utils.py           # Mask I/O and label normalization
│   │   │   ├── metrics_engine.py     # Z-axis stability metrics (CPU/GPU)
│   │   │   └── correction_logic.py   # Conservative/aggressive correction
│   │   └── postprocessing/
│   │       └── binary_builder/
│   │           └── run_binary_builder.py  # Multi-class → binary collapse
│   │
│   └── analysis/
│       └── pores_analysis/
│           ├── psd_entrypoint.py     # PSD CLI entry point
│           ├── psd_calculator.py     # Main PSD orchestrator
│           ├── distance_transform.py # Anisotropic EDT (CPU/GPU)
│           ├── local_thickness.py    # Morphological opening map
│           ├── block_processor.py    # Chunked processing for large volumes
│           ├── checkpoint_manager.py # Colab timeout resilience
│           ├── config_loader.py      # YAML/JSON config loader
│           ├── psd_output.py         # DataFrame export (CSV/JSON/HDF5)
│           ├── README.md             # PSD module documentation
│           └── USAGE_GUIDE.md        # Detailed usage instructions
│
├── data/
│   ├── dummy_scan/              # Test data for validation runs
│   └── raw/                     # Raw scan data location
│
├── outputs/
│   ├── psd_results/             # PSD analysis outputs
│   └── z_stability/             # Z-stability correction outputs
│
├── experiments/
│   └── notebooks/               # Exploratory analysis notebooks
│
└── archive/
    ├── colab_bridge/            # Legacy Colab integration
    └── napari_loader/           # Legacy napari visualization
```

---

## 3. Module Independence

### Design Principle

Each processing module is designed to operate **independently**. The integrated pipeline (`cli/pipeline.py`) serves as a convenience wrapper that chains stages together, but **is not a strict dependency**.

### Module Summary

| Module | Standalone | CLI Entry Point | Required Inputs |
|--------|------------|-----------------|-----------------|
| **Preprocessing** | ✅ Yes | `python -m core.preprocessing.preprocess_ct_images` | Raw TIFF/PNG images |
| **Segmentation** | ✅ Yes | `python -m core.segmentation.multiotsu` | Preprocessed images |
| **Z-Stability** | ✅ Yes | `python core/stability/run_pipeline.py` | Segmentation masks |
| **Binary Builder** | ✅ Yes | `python core/stability/postprocessing/binary_builder/run_binary_builder.py` | Z-stability masks OR segmentation masks |
| **PSD Analysis** | ✅ Yes | `python -m core.analysis.pores_analysis.psd_entrypoint` | Binary volume (.npy) |

### Dependency Graph

```
Raw Images
    │
    ▼
┌──────────────────┐
│  Preprocessing   │  (Independent)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Segmentation    │  (Requires: preprocessed images)
└────────┬─────────┘
         │
         ├──────────────────────────────┐
         │                              │
         ▼                              ▼
┌──────────────────┐          ┌──────────────────┐
│   Z-Stability    │          │  Binary Builder  │  (Can bypass z-stability
└────────┬─────────┘          │   (direct mode)  │   with direct segmentation)
         │                    └────────┬─────────┘
         │                              │
         ▼                              │
┌──────────────────┐                    │
│  Binary Builder  │◄───────────────────┘
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   PSD Analysis   │  (Requires: binary_pores.npy)
└──────────────────┘
```

### Bypassing Stages

- **Skip Z-Stability**: Set `z_stability.enabled: false` in config and use `binary_builder.mask_variant: direct` to process segmentation output directly
- **Skip Preprocessing**: Provide pre-processed images directly to segmentation via `stages.segmentation.input_dirs`
- **Standalone PSD**: Run PSD on any boolean 3D NumPy array (True=pore, False=solid)

---

## 4. Current Execution Status

### Integrated Pipeline Status

The integrated pipeline (`cli/pipeline.py`) orchestrates stages sequentially with subprocess invocation for Z-stability and PSD modules.

#### Operational Stages (Direct Function Calls)

| Stage | Status | Invocation Method |
|-------|--------|-------------------|
| Preprocessing | ✅ Functional | Direct function call to `run_preprocessing_stage()` |
| Segmentation | ✅ Functional | Direct function call to `run_multiotsu()` |
| Binary Builder | ✅ Functional | Direct function call to `collapse_to_binary()` |

#### Subprocess-Invoked Stages

| Stage | Status | Issue |
|-------|--------|-------|
| Z-Stability | ⚠️ Config-dependent | Invoked via `subprocess.run()` with config passed through `Z_STABILITY_CONFIG_JSON` environment variable |
| PSD | ⚠️ Config-dependent | Invoked via `subprocess.run()` with config passed through `PORES_ANALYSIS_CONFIG_JSON` environment variable |

### Known Integration Constraints

1. **Environment Variable Configuration**: Z-stability and PSD receive their configuration via JSON-encoded environment variables. This works but can fail silently if JSON serialization encounters non-serializable values.

2. **Path Resolution**: The integrated pipeline constructs output directories under `<scan_dir>/pipeline_outputs/<stage>`. Subprocess modules must correctly resolve these paths from the environment-passed configuration.

3. **DRY_RUN Mode**: Set `PIPELINE_DRY_RUN=1` (default) to validate wiring without executing compute-heavy stages. Set to `0` for actual execution.

### Standalone Module Status

All modules function correctly when executed independently with proper configuration:

- **Preprocessing**: ✅ Fully functional standalone
- **Segmentation**: ✅ Fully functional standalone
- **Z-Stability**: ✅ Fully functional with YAML config file
- **Binary Builder**: ✅ Fully functional with `--input_folder` argument
- **PSD Analysis**: ✅ Fully functional with YAML or JSON config

---

## 5. How to Run Each Module

### Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install numpy pyyaml tifffile opencv-python scikit-image scipy pandas pillow matplotlib
```

### 5.1 Preprocessing

Applies CLAHE histogram equalization (kernel=61, clip=0.01) and median filter (radius=5).

```bash
python -m core.preprocessing.preprocess_ct_images \
    --input_dir data/raw/scan_001/images \
    --output_dir outputs/preprocessed/scan_001 \
    --overwrite
```

**Inputs**: TIFF or PNG image stack (2D slices)  
**Outputs**: Preprocessed PNG images (`*_clahe61_med5.png`)

### 5.2 Segmentation (Multi-Otsu)

Performs multi-class thresholding to separate pores from solids.

```bash
python -m core.segmentation.multiotsu --config config/pipeline.yaml
```

Or via integrated pipeline:
```bash
python run_pipeline.py --scan data/dummy_scan/images --scan-id dummy
```

**Inputs**: Preprocessed grayscale images  
**Outputs**: 
- `classmap/` - Integer class labels (0=darkest, 1=mid, 2=brightest)
- `pores_class0/` - Binary mask of class 0 (darkest pores)
- `pores_class0_plus_1/` - Combined mask of classes 0+1

**Config Keys**: `method` (multiotsu), `n_classes` (2 or 3), `extensions`, `save_overlay`

### 5.3 Z-Stability Analysis

Corrects temporal inconsistencies in segmentation masks using sliding window metrics.

```bash
python core/stability/run_pipeline.py --config config/z_stability/pipeline.yaml
```

Or with inline JSON config:
```bash
set Z_STABILITY_CONFIG_JSON={"data":{"scan_folder":"path/to/masks"},"target_class":1,"windows":{"short":3,"long":7},"thresholds":{"conservative":{"min_short_frequency":0.6,"max_flip_rate":2,"min_long_frequency":0.3,"min_component_size":10},"aggressive":{"min_short_frequency":0.4,"max_flip_rate":3,"min_long_frequency":0.2,"min_component_size":5}},"output":{"output_folder":"outputs/z_stability"}}
python core/stability/run_pipeline.py --config config/z_stability/pipeline.yaml
```

**Inputs**: Segmentation mask folder (PNG stack with values {0, 1, 2})  
**Outputs**:
- `mask_conservative/` - Minimally corrected masks
- `mask_aggressive/` - Aggressively corrected masks
- `conf_conservative/` - Confidence diagnostic maps
- `conf_aggressive/` - Confidence diagnostic maps

**GPU Acceleration**: Set `compute.use_gpu_metrics: true` in config (requires PyTorch with CUDA)

### 5.4 Binary Builder

Collapses multi-class masks to binary pore/solid representation.

```bash
python core/stability/postprocessing/binary_builder/run_binary_builder.py \
    --input_folder outputs/z_stability/scan_001/mask_aggressive
```

**Inputs**: Corrected mask folder from Z-stability (or segmentation classmap)  
**Semantic Rule**: `pore = (mask == 0)`, `solid = (mask != 0)`  
**Outputs**:
- `binary_pores/` - PNG stack (255=pore, 0=solid)
- `binary_solids/` - PNG stack (255=solid, 0=pore)
- `binary_pores.npy` - 3D NumPy array for PSD analysis

### 5.5 PSD Analysis (Pore Size Distribution)

Computes pore size distribution using morphological opening (Vogel et al., 2010).

```bash
python -m core.analysis.pores_analysis.psd_entrypoint
```

Or with explicit JSON config:
```bash
set PORES_ANALYSIS_CONFIG_JSON={"paths":{"input_volume_path":"outputs/binary/binary_pores.npy","output_dir":"outputs/psd","checkpoint_dir":"outputs/psd/checkpoints","default_run_id":"scan_001_psd"},"image_params":{"voxel_spacing":[15.0,15.0,15.0]},"processing_thresholds":{"use_gpu":false},"output_settings":{"export_formats":["csv","json"]}}
python -m core.analysis.pores_analysis.psd_entrypoint
```

**Inputs**: Binary 3D volume as `.npy` file (dtype: bool or uint8, True/255=pore)  
**Outputs**:
- `psd.csv` / `psd.json` - PSD results with diameters and volume fractions
- Checkpoints (if enabled)

**Key Parameters**:
- `voxel_spacing`: Physical size in μm (Z, Y, X)
- `use_gpu`: Enable CuPy acceleration (requires `cupy-cuda11x` or `cupy-cuda12x`)
- `use_chunking`: Block processing for large volumes
- `chunk_size` / `halo_width`: Block dimensions and overlap

### 5.6 Full Integrated Pipeline

Run all stages sequentially:

```bash
# Dry run (validation only)
set PIPELINE_DRY_RUN=1
python run_pipeline.py --scan data/dummy_scan/images --scan-id dummy

# Full execution
set PIPELINE_DRY_RUN=0
python run_pipeline.py --config config/pipeline.yaml --scan data/raw/scan_001 --scan-id scan_001
```

---

## 6. Configuration Reference

### Main Pipeline Config (`config/pipeline.yaml`)

```yaml
scan_dir: data/dummy_scan/images
scan_id: dummy_scan

stages:
  preprocessing:
    enabled: true
    overwrite: false

  segmentation:
    enabled: true
    method: multiotsu
    n_classes: 2          # 2 or 3
    save_overlay: false
    extensions: [".png"]

  z_stability:
    enabled: true
    config_path: config/z_stability/pipeline.yaml
    definition:
      windows:
        short: 3
        long: 7
      target_class: 1
      thresholds:
        conservative:
          min_short_frequency: 0.6
          max_flip_rate: 2
          min_long_frequency: 0.3
          min_component_size: 10
        aggressive:
          min_short_frequency: 0.4
          max_flip_rate: 3
          min_long_frequency: 0.2
          min_component_size: 5
      compute:
        use_gpu_metrics: false

  binary_builder:
    enabled: true
    mask_variant: aggressive   # conservative | aggressive | direct

  psd:
    enabled: true
    entrypoint: core.analysis.pores_analysis.psd_entrypoint
    image_params:
      voxel_spacing: [15.0, 15.0, 15.0]
    processing_thresholds:
      use_gpu: false
      use_chunking: false
    output_settings:
      export_formats: [csv, json]
```

---

## 7. Known Limitations and TODO

### Current Limitations

1. **Subprocess Configuration Passing**: Z-stability and PSD stages receive configuration as JSON through environment variables. Complex nested structures or non-JSON-serializable values may cause failures without clear error messages.

2. **GPU Detection**: GPU availability is checked at runtime. If `use_gpu: true` is set but CUDA/CuPy is unavailable, the pipeline falls back to CPU without explicit warning.

3. **Label Convention Dependency**: The pipeline assumes a specific label encoding ({0, 1, 2} or {0, 127, 254}). Non-conforming masks will fail validation.

4. **Single-Scan Processing**: The integrated pipeline processes one scan at a time. Batch processing requires multiple invocations.

### Planned Improvements

- [ ] Direct Python imports for Z-stability and PSD (eliminate subprocess overhead)
- [ ] Unified configuration validation across all modules
- [ ] Parallel stage execution where dependencies allow
- [ ] Comprehensive logging with structured output (JSON logs)
- [ ] Progress reporting with estimated time remaining

### Verified Bug Fixes

- **Opening Map Computation** (Fixed 2026-02-03): The morphological opening algorithm was corrected from identity reconstruction to proper iterative granulometry per Vogel et al. (2010). See `core/analysis/pores_analysis/BUGFIX_CHANGELOG.md` for details.

---

## 8. Additional Resources

### Documentation

- [PSD Module README](core/analysis/pores_analysis/README.md) - Detailed PSD algorithm documentation
- [PSD Usage Guide](core/analysis/pores_analysis/USAGE_GUIDE.md) - Step-by-step PSD tutorial
- [Bugfix Changelog](core/analysis/pores_analysis/BUGFIX_CHANGELOG.md) - Critical fix documentation

### Legacy/Experimental

- `experiments/notebooks/` - Exploratory analysis and development notebooks
- `archive/` - Legacy code preserved for reference

### Configuration Templates

- `config/templates/config_template.py` - Programmatic config generation
- `config/2_class_pipeline.yaml` - Alternative 2-class segmentation configuration

---

## References

- Vogel, H.-J., Weller, U., & Schlüter, S. (2010). Quantification of soil structure based on Minkowski functions. *Computers & Geosciences*, 36(10), 1236-1245.
