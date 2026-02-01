"""
preprocess_ct_images.py

Batch preprocessing for μCT images
EXACT match to napari_preproc_playground.py (mode = clahe+median)

ASSUMPTIONS:
- Input images are already normalized to [0, 1]
- No thresholding, no masks, no segmentation

Pipeline:
1. Load image as float32 [0,1]
2. CLAHE (kernel=61, clip=0.01)
3. Median filter (scipy.ndimage, size=ensure_odd(radius))
4. Save as 8-bit PNG
"""

import argparse
from pathlib import Path
import numpy as np
import tifffile

from skimage import img_as_float32
from skimage.exposure import equalize_adapthist
from skimage.util import img_as_ubyte
from scipy.ndimage import median_filter


# ==========================
# EXACT napari parameters
# ==========================
CLAHE_KERNEL = 61
CLAHE_CLIP = 0.01
MEDIAN_RADIUS = 5   # 

def ensure_odd(val: int) -> int:
    return val if val % 2 == 1 else val + 1


# ==========================
# Core preprocessing
# ==========================
def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Apply preprocessing IDENTICAL to napari (clahe + median).
    """

    # --- EXACT napari behavior ---
    img = img_as_float32(img)
    img = np.clip(img, 0.0, 1.0)

    img = equalize_adapthist(
        img,
        kernel_size=CLAHE_KERNEL,
        clip_limit=CLAHE_CLIP,
    )

    img = median_filter(
        img,
        size=ensure_odd(MEDIAN_RADIUS),
    )

    return img_as_ubyte(img)


# ==========================
# CLI
# ==========================
def main():
    parser = argparse.ArgumentParser(
        description="μCT preprocessing (EXACT napari clahe+median)"
    )
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        list(args.input_dir.glob("*.tif")) +
        list(args.input_dir.glob("*.tiff")) +
        list(args.input_dir.glob("*.png"))
    )

    if not images:
        raise RuntimeError("No images found")

    for img_path in images:
        out = args.output_dir / f"{img_path.stem}_clahe61_med{MEDIAN_RADIUS}.png"

        if out.exists() and not args.overwrite:
            continue

        img = tifffile.imread(img_path)
        if img.ndim != 2:
            raise ValueError(f"{img_path.name} is not 2D")

        img8 = preprocess_image(img)
        tifffile.imwrite(out, img8)

        print(f"Saved: {out.name}")

    print("Done.")


if __name__ == "__main__":
    main()
