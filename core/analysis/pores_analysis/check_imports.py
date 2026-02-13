#!/usr/bin/env python3
"""
Quick import validation test
=============================

Verifies all modules can be imported correctly.
Run this first to check if the package is properly set up.
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("IMPORT VALIDATION TEST")
print("=" * 60)

modules_to_test = [
    'distance_transform',
    'block_processor',
    'local_thickness',
    'checkpoint_manager',
    'psd_calculator',
    'psd_output',
]

failed = []

for module_name in modules_to_test:
    try:
        print(f"\n[✓] Testing: {module_name}...", end=" ")
        exec(f"import {module_name}")
        print("OK")
    except Exception as e:
        print(f"FAILED")
        print(f"    Error: {e}")
        failed.append((module_name, e))

print("\n" + "=" * 60)
if not failed:
    print("✅ ALL IMPORTS SUCCESSFUL")
    print("=" * 60)
    print("\nYou can now run:")
    print("  python run_tests.py")
    print("  python example_workflow.py")
else:
    print(f"❌ {len(failed)} IMPORTS FAILED")
    print("=" * 60)
    for module_name, error in failed:
        print(f"\n{module_name}: {error}")
    sys.exit(1)
