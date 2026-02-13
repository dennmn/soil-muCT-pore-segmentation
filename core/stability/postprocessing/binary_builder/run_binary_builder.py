"""
Binary Builder CLI
==================

Purpose
-------
This script is a **semantic adapter** between the output of the z_stability
pipeline and downstream pore-space analysis modules (e.g. PSD / EDT).

It converts a stack of *corrected multi-class masks* into **binary pore/solid
representations**, both as PNG slices (for inspection) and as a NumPy volume
(for direct consumption by analysis notebooks).

---

Input
-----
- A folder containing PNG slices produced by z_stability
- Each slice is a corrected semantic mask with integer class labels

Supported input conventions include (but are not limited to):
- {0, 1, 2}
- {0, 127, 254}

The script makes **no assumptions about exact numeric values**, only about
their *semantic meaning*.

---

Semantic Assumptions
--------------------
This script assumes the following contract with z_stability outputs:

- Class 0  → pore
- Class >0 → solid

In other words:
- **All non-zero classes (including class 1 / ambiguous)** are treated as solid
- This reflects a *conservative solid definition*, appropriate for PSD analysis

This behavior is intentional and explicit.

---

Binary Collapse Rule
--------------------
The semantic collapse is defined as:

    solid = (mask != 0)
    pore  = (mask == 0)

Resulting binary masks use:
- uint8 encoding
- 255 = True
- 0   = False

---

Output
------
Given an input folder:

    <input_folder>/

The script produces the following outputs in the parent directory:

1. binary_pores/
   - PNG stack of pore mask slices (255 = pore, 0 = solid)

2. binary_solids/
   - PNG stack of solid mask slices (255 = solid, 0 = pore)

3. binary_pores.npy
   - 3D NumPy array (uint8, shape = [Z, Y, X])
   - Intended for direct loading via:
       volume = np.load("binary_pores.npy")

---

Design Notes
------------
- This script performs **no spatial operations**
- It does **not** modify topology or connectivity
- It only enforces a clear semantic contract:
    z_stability → binary pore/solid → analysis

Any change to class semantics must be made explicitly here.

---

Usage
-----
    python run_binary_builder.py --input_folder <path_to_corrected_masks>

Example:
    python run_binary_builder.py \
        --input_folder z_stability/outputs/test_scan/mask_aggressive
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def load_image_stack(folder_path: Path) -> np.ndarray:
    folder = folder_path
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    image_files = sorted(folder.glob("*.png"))
    if not image_files:
        raise ValueError(f"No PNG files found in {folder_path}")

    first_img = np.array(Image.open(image_files[0]))
    height, width = first_img.shape[:2]
    n_slices = len(image_files)

    volume = np.zeros((n_slices, height, width), dtype=np.uint8)

    for i, img_path in enumerate(image_files):
        img = np.array(Image.open(img_path))
        if img.ndim == 3:
            img = img[:, :, 0]
        volume[i] = img

    print(f"Loaded {n_slices} slices from {folder_path} -> shape {volume.shape}")
    return volume


def save_mask(array: np.ndarray, output_path: Path) -> None:
    output_folder = output_path
    output_folder.mkdir(parents=True, exist_ok=True)

    n_slices = array.shape[0]
    for i in range(n_slices):
        slice_img = Image.fromarray(array[i].astype(np.uint8))
        slice_img.save(output_folder / f"slice_{i:04d}.png")

    print(f"Saved {n_slices} slices to {output_path}")


def collapse_to_binary(
    corrected_mask: np.ndarray,
    unique_labels: Optional[Tuple[int, ...]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    if unique_labels is None:
        unique_labels = tuple(np.unique(corrected_mask).tolist())

    label_set = {int(label) for label in unique_labels}

    if label_set == {0, 1}:
        binary_solids = (corrected_mask == 1).astype(np.uint8) * 255
        binary_pores = (corrected_mask == 0).astype(np.uint8) * 255
    elif label_set == {0, 1, 2}:
        binary_solids = (corrected_mask != 0).astype(np.uint8) * 255
        binary_pores = (corrected_mask == 0).astype(np.uint8) * 255
    else:
        allowed = "[{0, 1}, {0, 1, 2}]"
        raise ValueError(
            f"Binary builder accepts label sets {allowed}, but found {sorted(label_set)}"
        )

    assert binary_pores.shape == corrected_mask.shape, "Pores mask shape mismatch"
    assert binary_solids.shape == corrected_mask.shape, "Solids mask shape mismatch"
    assert (binary_pores & binary_solids).sum() == 0, "Voxels assigned to both pore and solid"
    assert ((binary_pores | binary_solids) == 255).all(), "Not all voxels assigned"

    return binary_pores, binary_solids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert z_stability corrected masks to binary pore/solid outputs"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to folder containing corrected mask PNG slices (z_stability output)"
    )

    args = parser.parse_args()
    input_folder = Path(args.input_folder)

    if not input_folder.exists():
        print(f"ERROR: Input folder not found: {input_folder}")
        sys.exit(1)

    print("=" * 70)
    print("BINARY BUILDER - Semantic Collapse Module")
    print("=" * 70)
    print(f"Input folder: {input_folder}")

    print("\n[1/3] Loading corrected mask stack...")
    corrected_mask = load_image_stack(input_folder)
    print(f"      Loaded volume shape: {corrected_mask.shape}")

    detected_labels = tuple(sorted(set(np.unique(corrected_mask).tolist())))
    print("\n[2/3] Applying semantic collapse rule...")
    print(f"      Detected labels: {detected_labels}")
    binary_pores, binary_solids = collapse_to_binary(corrected_mask, unique_labels=detected_labels)

    pore_count = (binary_pores == 255).sum()
    solid_count = (binary_solids == 255).sum()
    total_voxels = binary_pores.size

    print(f"      Pore voxels:  {pore_count:,} ({100 * pore_count / total_voxels:.2f}%)")
    print(f"      Solid voxels: {solid_count:,} ({100 * solid_count / total_voxels:.2f}%)")

    print("\n[3/3] Saving binary outputs...")
    parent_folder = input_folder.parent
    pores_output = parent_folder / "binary_pores"
    solids_output = parent_folder / "binary_solids"
    np.save(parent_folder / "binary_pores.npy", binary_pores)

    save_mask(binary_pores, pores_output)
    save_mask(binary_solids, solids_output)

    print("\n" + "=" * 70)
    print("BINARY BUILDER - Complete")
    print("=" * 70)
    print(f"Binary pores:  {pores_output}")
    print(f"Binary solids: {solids_output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
