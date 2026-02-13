"""
PSD Output Formatter
====================

Converts PSD results to Pandas DataFrames with proper formatting and export.

Output Format:
    - Diameter_px: Bin center in voxel units
    - Diameter_um: Bin center in micrometers
    - Volume_Count: Number of voxels in bin
    - Cumulative_Porosity: Cumulative volume fraction [0-1]
    - Differential_PSD: dV/dd (differential distribution)
    - is_reliable: Boolean flag (True if diameter >= configured reliability threshold)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from pathlib import Path
import json
import warnings

try:
    from .config_loader import load_config
except ImportError:
    from config_loader import load_config

_CONFIG = load_config()
MIN_RELIABLE_DIAMETER_VOXELS = _CONFIG.get("processing_thresholds", {}).get("min_reliable_diameter_voxels", 5)


def psd_to_dataframe(
    psd_dict: Dict[str, np.ndarray],
    include_metadata: bool = True
) -> pd.DataFrame:
    """
    Convert PSD dictionary to Pandas DataFrame.
    
    Args:
        psd_dict: PSD results from compute_psd()
        include_metadata: If True, store metadata as DataFrame attrs
    
    Returns:
        DataFrame with columns:
            - Diameter_px: Pore diameter in voxel units
            - Diameter_um: Pore diameter in micrometers
            - Volume_Count: Number of voxels per bin
            - Cumulative_Porosity: Cumulative volume fraction
            - Differential_PSD: Differential pore size distribution (dV/dd)
            - is_reliable: Reliability flag (Vogel et al. constraint)
    
    Example:
        >>> psd = compute_psd(volume, voxel_spacing=(2, 1, 1))
        >>> df = psd_to_dataframe(psd)
        >>> print(df[df['is_reliable']])  # Show only reliable data
    """
    # Build DataFrame
    df = pd.DataFrame({
        'Diameter_px': psd_dict['bin_centers_px'],
        'Diameter_um': psd_dict['bin_centers_um'],
        'Volume_Count': psd_dict['volume_counts'],
        'Cumulative_Porosity': psd_dict['cumulative_volume'],
        'Differential_PSD': psd_dict['differential_volume'],
        'is_reliable': psd_dict['reliability_flag']
    })
    
    # Add metadata as attributes
    if include_metadata:
        df.attrs['total_pore_voxels'] = int(psd_dict['total_pore_voxels'])
        df.attrs['voxel_spacing_um'] = tuple(psd_dict['voxel_spacing'])
        df.attrs['min_reliable_diameter_voxels'] = MIN_RELIABLE_DIAMETER_VOXELS  # Vogel et al. constraint
    
    return df


def save_psd_dataframe(
    df: pd.DataFrame,
    output_path: str,
    format: str = 'csv',
    metadata: Optional[Dict] = None
) -> None:
    """
    Save PSD DataFrame to disk with optional metadata.
    
    Args:
        df: PSD DataFrame from psd_to_dataframe()
        output_path: Output file path (extension auto-added if missing)
        format: Output format: 'csv', 'hdf5', 'excel', 'json'
        metadata: Optional additional metadata to save alongside data
    
    Supported Formats:
        - csv: Human-readable, portable, no metadata preservation
        - hdf5: Compressed, preserves metadata, requires pytables
        - excel: Human-readable, requires openpyxl
        - json: Portable, includes metadata
    
    Example:
        >>> save_psd_dataframe(
        ...     df,
        ...     'results/scan_001_psd',
        ...     format='csv',
        ...     metadata={'scan_id': '001', 'date': '2026-02-03'}
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Merge metadata
    full_metadata = dict(df.attrs) if hasattr(df, 'attrs') else {}
    if metadata is not None:
        full_metadata.update(metadata)
    
    if format == 'csv':
        # CSV: Save data + metadata in separate file
        if output_path.suffix != '.csv':
            output_path = output_path.with_suffix('.csv')
        
        df.to_csv(output_path, index=False, float_format='%.6f')
        print(f"Saved CSV: {output_path}")
        
        # Save metadata separately
        if full_metadata:
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f, indent=2)
            print(f"Saved metadata: {metadata_path}")
    
    elif format == 'hdf5':
        # HDF5: Compact, preserves metadata
        if output_path.suffix not in ['.h5', '.hdf5']:
            output_path = output_path.with_suffix('.h5')
        
        try:
            df.to_hdf(
                output_path,
                key='psd',
                mode='w',
                complevel=9,
                complib='zlib'
            )
            
            # Store metadata as attributes
            import h5py
            with h5py.File(output_path, 'a') as f:
                for key, value in full_metadata.items():
                    f['psd'].attrs[key] = value
            
            print(f"Saved HDF5: {output_path}")
        except ImportError:
            raise ImportError(
                "HDF5 format requires 'tables' (pytables). "
                "Install with: pip install tables"
            )
    
    elif format == 'excel':
        # Excel: Human-readable spreadsheet
        if output_path.suffix not in ['.xlsx', '.xls']:
            output_path = output_path.with_suffix('.xlsx')
        
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='PSD', index=False)
                
                # Add metadata sheet
                if full_metadata:
                    metadata_df = pd.DataFrame([full_metadata]).T
                    metadata_df.columns = ['Value']
                    metadata_df.to_excel(writer, sheet_name='Metadata')
            
            print(f"Saved Excel: {output_path}")
        except ImportError:
            raise ImportError(
                "Excel format requires 'openpyxl'. "
                "Install with: pip install openpyxl"
            )
    
    elif format == 'json':
        # JSON: Portable with embedded metadata
        if output_path.suffix != '.json':
            output_path = output_path.with_suffix('.json')
        
        output_dict = {
            'metadata': full_metadata,
            'data': df.to_dict(orient='records')
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_dict, f, indent=2)
        
        print(f"Saved JSON: {output_path}")
    
    else:
        raise ValueError(
            f"Unknown format '{format}'. "
            f"Supported: csv, hdf5, excel, json"
        )


def load_psd_dataframe(
    input_path: str,
    format: Optional[str] = None
) -> pd.DataFrame:
    """
    Load PSD DataFrame from disk.
    
    Args:
        input_path: Path to saved PSD file
        format: File format (auto-detected from extension if None)
    
    Returns:
        DataFrame with metadata restored (if available)
    
    Example:
        >>> df = load_psd_dataframe('results/scan_001_psd.csv')
        >>> print(df.attrs)  # Access metadata
    """
    input_path = Path(input_path)
    
    # Auto-detect format
    if format is None:
        ext = input_path.suffix.lower()
        format_map = {
            '.csv': 'csv',
            '.h5': 'hdf5',
            '.hdf5': 'hdf5',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.json': 'json'
        }
        format = format_map.get(ext)
        if format is None:
            raise ValueError(
                f"Cannot auto-detect format from extension '{ext}'"
            )
    
    if format == 'csv':
        df = pd.read_csv(input_path)
        
        # Try to load metadata
        metadata_path = input_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                df.attrs = json.load(f)
    
    elif format == 'hdf5':
        df = pd.read_hdf(input_path, key='psd')
        
        # Load metadata from HDF5 attributes
        try:
            import h5py
            with h5py.File(input_path, 'r') as f:
                metadata = {k: v for k, v in f['psd'].attrs.items()}
                df.attrs = metadata
        except ImportError:
            warnings.warn(
                "HDF5 metadata requires 'h5py', metadata not loaded",
                UserWarning
            )
    
    elif format == 'excel':
        df = pd.read_excel(input_path, sheet_name='PSD')
        
        # Try to load metadata sheet
        try:
            metadata_df = pd.read_excel(input_path, sheet_name='Metadata', index_col=0)
            df.attrs = metadata_df['Value'].to_dict()
        except Exception:
            pass  # Metadata sheet not found
    
    elif format == 'json':
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data['data'])
        df.attrs = data.get('metadata', {})
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return df


def plot_psd(
    df: pd.DataFrame,
    show_unreliable: bool = False,
    log_scale: bool = True,
    save_path: Optional[str] = None
) -> None:
    """
    Plot PSD curve from DataFrame.
    
    Args:
        df: PSD DataFrame
        show_unreliable: If False, grey out unreliable region (diameter < configured reliability threshold)
        log_scale: Use log scale for x-axis (diameter)
        save_path: If provided, save plot to this path
    
    Example:
        >>> df = load_psd_dataframe('results/scan_001_psd.csv')
        >>> plot_psd(df, save_path='results/scan_001_plot.png')
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Plotting requires 'matplotlib'. "
            "Install with: pip install matplotlib"
        )
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    unreliable_label = (
        f"Unreliable (d < {MIN_RELIABLE_DIAMETER_VOXELS} voxels)"
    )
    
    # Cumulative PSD
    ax1.plot(
        df['Diameter_um'],
        df['Cumulative_Porosity'],
        'b-',
        linewidth=2,
        label='Cumulative'
    )
    
    if not show_unreliable:
        # Highlight unreliable region
        unreliable = ~df['is_reliable']
        if unreliable.any():
            ax1.fill_between(
                df['Diameter_um'],
                0,
                df['Cumulative_Porosity'],
                where=unreliable,
                alpha=0.3,
                color='red',
                label=unreliable_label
            )
    
    ax1.set_ylabel('Cumulative Volume Fraction', fontsize=12)
    ax1.set_title('Pore Size Distribution - Cumulative', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Differential PSD
    ax2.plot(
        df['Diameter_um'],
        df['Differential_PSD'],
        'g-',
        linewidth=2,
        label='Differential'
    )
    
    if not show_unreliable:
        unreliable = ~df['is_reliable']
        if unreliable.any():
            ax2.fill_between(
                df['Diameter_um'],
                0,
                df['Differential_PSD'],
                where=unreliable,
                alpha=0.3,
                color='red',
                label=unreliable_label
            )
    
    ax2.set_xlabel('Pore Diameter (μm)', fontsize=12)
    ax2.set_ylabel('dV/dd (μm⁻¹)', fontsize=12)
    ax2.set_title('Pore Size Distribution - Differential', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Log scale
    if log_scale:
        ax1.set_xscale('log')
        ax2.set_xscale('log')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Validation test
    import tempfile
    
    print("PSD Output Module - Validation Test")
    print("=" * 60)
    
    # Create synthetic PSD data
    n_bins = 20
    psd_dict = {
        'bin_centers_px': np.linspace(1, 50, n_bins),
        'bin_centers_um': np.linspace(2, 100, n_bins),
        'bin_edges_um': np.linspace(2, 100, n_bins + 1),
        'volume_counts': np.random.randint(100, 1000, n_bins),
        'cumulative_volume': np.linspace(0.1, 1.0, n_bins),
        'differential_volume': np.random.rand(n_bins) * 0.1,
        'reliability_flag': np.array([False]*5 + [True]*15),  # First 5 unreliable
        'total_pore_voxels': 10000,
        'voxel_spacing': (2.0, 1.0, 1.0)
    }
    
    # Convert to DataFrame
    print("\n[Test 1: DataFrame conversion]")
    df = psd_to_dataframe(psd_dict)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Reliable bins: {df['is_reliable'].sum()}/{len(df)}")
    print(f"  Metadata: {df.attrs}")
    
    # Test save/load for each format
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        for fmt in ['csv', 'json']:  # Skip hdf5/excel (require extra deps)
            print(f"\n[Test 2: {fmt.upper()} save/load]")
            
            save_path = tmpdir / f"test_psd.{fmt}"
            save_psd_dataframe(df, str(save_path), format=fmt)
            
            loaded_df = load_psd_dataframe(str(save_path))
            print(f"  Loaded shape: {loaded_df.shape}")
            
            # Verify data integrity
            assert np.allclose(loaded_df['Diameter_um'], df['Diameter_um'])
            print(f"  Data integrity: ✓")
    
    print("\n" + "=" * 60)
    print("Module validation complete.")
