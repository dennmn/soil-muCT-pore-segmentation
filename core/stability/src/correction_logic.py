import numpy as np
from typing import Dict, Tuple
import sys
sys.path.insert(0, 'src')
from metrics_engine import PORE_CLASS, SOLID_CLASS

def apply_conservative_correction(original_mask, metrics_short, metrics_long, metrics_2d, thresholds, target_class=1):
    corrected = original_mask.copy()
    target_voxels = (original_mask == target_class)
    
    print(f"\nConservative thresholds: freq_short={thresholds['min_short_frequency']}, flip={thresholds['max_flip_rate']}, freq_long={thresholds['min_long_frequency']}, size={thresholds['min_component_size']}")
    
    unstable_short = metrics_short['frequency'] < thresholds['min_short_frequency']
    high_flip = metrics_short['flip_rate'] > thresholds['max_flip_rate']
    unstable_long = metrics_long['frequency'] < thresholds['min_long_frequency']
    small_comp = metrics_2d['component_size'] < thresholds['min_component_size']
    target_total = np.sum(target_voxels)
    denom = target_total if target_total else 1
    print("Conservative diagnostics (target class):")
    print(f"  unstable_short: {np.sum(target_voxels & unstable_short)} ({100*np.sum(target_voxels & unstable_short)/denom:.2f}%)")
    print(f"  high_flip:      {np.sum(target_voxels & high_flip)} ({100*np.sum(target_voxels & high_flip)/denom:.2f}%)")
    print(f"  unstable_long:  {np.sum(target_voxels & unstable_long)} ({100*np.sum(target_voxels & unstable_long)/denom:.2f}%)")
    print(f"  small_comp:     {np.sum(target_voxels & small_comp)} ({100*np.sum(target_voxels & small_comp)/denom:.2f}%)")
    
    # Conservative: Require ALL bad signs
    noise_mask = target_voxels & unstable_short & high_flip & unstable_long & small_comp
    print(f"  noise_mask:     {np.sum(noise_mask)} ({100*np.sum(noise_mask)/denom:.2f}%)")
    
    relabel_pore = noise_mask & (metrics_short['class_0_fraction'] > metrics_short['class_2_fraction'])
    relabel_solid = noise_mask & (metrics_short['class_0_fraction'] <= metrics_short['class_2_fraction'])
    
    corrected[relabel_pore] = PORE_CLASS
    corrected[relabel_solid] = SOLID_CLASS
    
    # Simple confidence
    conf = np.zeros_like(original_mask, dtype=np.float32)
    if np.any(noise_mask):
         c1 = np.clip(1 - metrics_short['frequency']/thresholds['min_short_frequency'],0,1)
         c2 = np.clip((metrics_short['flip_rate']-thresholds['max_flip_rate'])/thresholds['max_flip_rate'],0,1)
         conf[noise_mask] = (c1+c2)[noise_mask] / 2
         
    print(f"Conservative: Corrected {np.sum(noise_mask)} voxels (to_pore={np.sum(relabel_pore)}, to_solid={np.sum(relabel_solid)})")
    return corrected, conf

def apply_aggressive_correction(original_mask, metrics_short, metrics_long, metrics_2d, thresholds, target_class=1):
    corrected = original_mask.copy()
    target_voxels = (original_mask == target_class)
    
    print(f"\nAggressive thresholds: freq_short={thresholds['min_short_frequency']}, flip={thresholds['max_flip_rate']}, freq_long={thresholds['min_long_frequency']}, size={thresholds['min_component_size']}")
    
    unstable_short = metrics_short['frequency'] < thresholds['min_short_frequency']
    high_flip = metrics_short['flip_rate'] > thresholds['max_flip_rate']
    unstable_long = metrics_long['frequency'] < thresholds['min_long_frequency']
    small_comp = metrics_2d['component_size'] < thresholds['min_component_size']
    target_total = np.sum(target_voxels)
    denom = target_total if target_total else 1
    print("Aggressive diagnostics (target class):")
    print(f"  unstable_short: {np.sum(target_voxels & unstable_short)} ({100*np.sum(target_voxels & unstable_short)/denom:.2f}%)")
    print(f"  high_flip:      {np.sum(target_voxels & high_flip)} ({100*np.sum(target_voxels & high_flip)/denom:.2f}%)")
    print(f"  unstable_long:  {np.sum(target_voxels & unstable_long)} ({100*np.sum(target_voxels & unstable_long)/denom:.2f}%)")
    print(f"  small_comp:     {np.sum(target_voxels & small_comp)} ({100*np.sum(target_voxels & small_comp)/denom:.2f}%)")
    
    # Aggressive: Require ANY bad sign
    noise_mask = target_voxels & (unstable_short | high_flip | unstable_long | small_comp)
    print(f"  noise_mask:     {np.sum(noise_mask)} ({100*np.sum(noise_mask)/denom:.2f}%)")
    
    relabel_pore = noise_mask & (metrics_short['class_0_fraction'] > metrics_short['class_2_fraction'])
    relabel_solid = noise_mask & (metrics_short['class_0_fraction'] <= metrics_short['class_2_fraction'])
    
    corrected[relabel_pore] = PORE_CLASS
    corrected[relabel_solid] = SOLID_CLASS
    
    conf = np.zeros_like(original_mask, dtype=np.float32)
    print(f"Aggressive: Corrected {np.sum(noise_mask)} voxels (to_pore={np.sum(relabel_pore)}, to_solid={np.sum(relabel_solid)})")
    return corrected, conf
