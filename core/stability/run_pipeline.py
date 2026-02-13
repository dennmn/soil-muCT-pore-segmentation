
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_PATH))

from io_utils import load_mask_stack, save_mask, save_diagnostic_map, get_scan_name
from metrics_engine import (compute_z_metrics, compute_2d_metrics_stack, 
                             compute_adjacent_slice_consistency, compute_z_run_length_stability,
                             PORE_CLASS, SOLID_CLASS)
from correction_logic import apply_conservative_correction, apply_aggressive_correction

def main(config_path: Path):
    print("Starting Pipeline...")
    env_config = os.environ.get("Z_STABILITY_CONFIG_JSON")
    if env_config:
        config = json.loads(env_config)
    else:
        if not config_path.exists():
            raise FileNotFoundError(f"Z-stability config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    
    scan_folder = config['data']['scan_folder']
    target_class = config['target_class']
    
    print(f"Processing: {scan_folder}")
    
    # Load
    mask_volume = load_mask_stack(str(Path(scan_folder)/"masks"))
    total_voxels = mask_volume.size
    unique_labels = np.unique(mask_volume)
    expected_labels = {PORE_CLASS, target_class, SOLID_CLASS}
    if not set(unique_labels).issubset(expected_labels):
        raise AssertionError(f"Internal masks must use labels {expected_labels}, found {sorted(unique_labels)}")
    class_0_count = np.sum(mask_volume == PORE_CLASS)
    target_count = np.sum(mask_volume == target_class)
    class_2_count = np.sum(mask_volume == SOLID_CLASS)
    target_fraction = (target_count / total_voxels) if total_voxels else 0.0
    print("\n=== MASK VOLUME COUNTS ===")
    print(f"Value 0 voxels (pore): {class_0_count}")
    print(f"Value {target_class} voxels (target): {target_count}")
    print(f"Value 2 voxels (solid): {class_2_count}")
    print(f"Total voxels: {total_voxels}")
    print(f"Target class {target_class}: {target_count} ({target_fraction:.4f} fraction)")
    
    compute_config = config.get('compute', {})
    use_gpu = bool(compute_config.get('use_gpu_metrics', False))
    validate_gpu = bool(compute_config.get('validate_gpu_metrics', False))

    # Metrics
    m_short = compute_z_metrics(
        mask_volume, target_class, config['windows']['short'], use_gpu=use_gpu, validate_gpu=validate_gpu
    )
    m_long = compute_z_metrics(
        mask_volume, target_class, config['windows']['long'], use_gpu=use_gpu, validate_gpu=validate_gpu
    )
    m_2d = compute_2d_metrics_stack(mask_volume, target_class)
    
    target_voxels = (mask_volume == target_class)
    target_total = np.sum(target_voxels)
    if target_total == 0:
        print("\n=== Z-METRIC DIAGNOSTICS (target class) ===")
        print("No target_class voxels found; skipping Z-metric diagnostics.")
    else:
        flip_rate = m_short['flip_rate'][target_voxels]
        freq_short = m_short['frequency'][target_voxels]
        freq_long = m_long['frequency'][target_voxels]
        flip_gt_zero = np.sum(flip_rate > 0)
        flip_gt_zero_pct = 100 * flip_gt_zero / target_total
        print("\n=== Z-METRIC DIAGNOSTICS (target class) ===")
        print(f"Flip rate: min={np.min(flip_rate):.6f}, max={np.max(flip_rate):.6f}, mean={np.mean(flip_rate):.6f}, >0={flip_gt_zero} ({flip_gt_zero_pct:.2f}%)")
        print(
            "Frequency (short): "
            f"min={np.min(freq_short):.6f}, mean={np.mean(freq_short):.6f}, median={np.median(freq_short):.6f}, "
            f"p5={np.percentile(freq_short, 5):.6f}, p95={np.percentile(freq_short, 95):.6f}"
        )
        print(
            "Frequency (long):  "
            f"min={np.min(freq_long):.6f}, mean={np.mean(freq_long):.6f}, median={np.median(freq_long):.6f}, "
            f"p5={np.percentile(freq_long, 5):.6f}, p95={np.percentile(freq_long, 95):.6f}"
        )
        cons_thresholds = config['thresholds']['conservative']
        unstable_short = m_short['frequency'] < cons_thresholds['min_short_frequency']
        unstable_long = m_long['frequency'] < cons_thresholds['min_long_frequency']
        short_only = np.sum(target_voxels & unstable_short & ~unstable_long)
        long_only = np.sum(target_voxels & ~unstable_short & unstable_long)
        print(
            "Unstable frequency (conservative thresholds): "
            f"short_only={short_only} ({100*short_only/target_total:.2f}%), "
            f"long_only={long_only} ({100*long_only/target_total:.2f}%)"
        )
    
    # Correction
    out_path = Path(config['output']['output_folder']) / get_scan_name(scan_folder)
    
    mask_cons, conf_cons = apply_conservative_correction(
        mask_volume, m_short, m_long, m_2d, config['thresholds']['conservative'], target_class=target_class
    )
    mask_agg, conf_agg = apply_aggressive_correction(
        mask_volume, m_short, m_long, m_2d, config['thresholds']['aggressive'], target_class=target_class
    )
    
    # Validation: ensure masks are different
    assert not np.array_equal(mask_cons, mask_agg), "ERROR: Conservative and aggressive masks are identical!"
    
    # Log statistics
    print(f"\n=== MASK STATISTICS ===")
    print(f"Conservative: unique={np.unique(mask_cons)}, sum={np.sum(mask_cons)}, voxels={mask_cons.size}")
    print(f"Aggressive:   unique={np.unique(mask_agg)}, sum={np.sum(mask_agg)}, voxels={mask_agg.size}")
    diff_count = np.sum(mask_cons != mask_agg)
    print(f"Difference:   {diff_count} voxels differ ({100*diff_count/mask_cons.size:.2f}%)")
    
    # Save
    save_mask(mask_cons, str(out_path / "mask_conservative"))
    save_diagnostic_map(conf_cons, str(out_path / "conf_conservative"), colormap='hot', vmin=0, vmax=1)
    save_mask(mask_agg, str(out_path / "mask_aggressive"))
    save_diagnostic_map(conf_agg, str(out_path / "conf_aggressive"), colormap='hot', vmin=0, vmax=1)
    
    # Check if masks are identical
    masks_identical = np.array_equal(mask_cons, mask_agg)
    if masks_identical:
        print("\nWARNING: Conservative and Aggressive masks are IDENTICAL!")
        print("No Z-stability analysis performed.\n")
    else:
        print("\nComputing Z-stability metrics...")
        
        # Conservative mask metrics
        dice_cons = compute_adjacent_slice_consistency(
            mask_cons, target_class, use_gpu=use_gpu, validate_gpu=validate_gpu
        )
        run_cons = compute_z_run_length_stability(mask_cons, target_class)
        
        # Aggressive mask metrics
        dice_agg = compute_adjacent_slice_consistency(
            mask_agg, target_class, use_gpu=use_gpu, validate_gpu=validate_gpu
        )
        run_agg = compute_z_run_length_stability(mask_agg, target_class)
        
        # Print minimal summary
        print("\nConservative:")
        print(f"  Mean adjacent Dice: {np.mean(dice_cons):.4f}")
        print(f"  Median run length: {run_cons['median_run_length']:.2f}")
        
        print("\nAggressive:")
        print(f"  Mean adjacent Dice: {np.mean(dice_agg):.4f}")
        print(f"  Median run length: {run_agg['median_run_length']:.2f}")
    
    print("\nDone!")

def _parse_args() -> Path:
    parser = argparse.ArgumentParser(description="Z-stability diagnostics")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "z_stability" / "pipeline.yaml",
    )
    args = parser.parse_args()
    return args.config if args.config.is_absolute() else PROJECT_ROOT / args.config


if __name__ == "__main__":
    main(_parse_args())
