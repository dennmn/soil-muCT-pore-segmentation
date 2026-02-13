#!/usr/bin/env python3
"""
Simple test runner for PSD validation suite
============================================

This script ensures proper module imports and runs the validation tests.
Use this instead of running test_psd_synthetic.py directly.
"""


import sys
from pathlib import Path

# Add pores_analysis to Python path
pores_analysis_dir = Path(__file__).parent
if str(pores_analysis_dir) not in sys.path:
    sys.path.insert(0, str(pores_analysis_dir))

try:
    from pores_analysis.config_loader import load_config
except ImportError:
    from config_loader import load_config

CONFIG = load_config()
PATH_CONFIG = CONFIG.get("paths", {})

# Now import and run tests
if __name__ == "__main__":
    print("=" * 70)
    print("PSD VALIDATION TEST RUNNER")
    print("=" * 70)
    print(f"\nModule path: {pores_analysis_dir}")
    print("Importing test suite...\n")
    print("Centralized config:")
    print(f"  input volume: {PATH_CONFIG.get('input_volume_path')}")
    print(f"  checkpoint dir: {PATH_CONFIG.get('checkpoint_dir')}")
    
    try:
        from test_psd_synthetic import run_all_tests
        run_all_tests()
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✅ Test runner completed successfully")
    sys.exit(0)
