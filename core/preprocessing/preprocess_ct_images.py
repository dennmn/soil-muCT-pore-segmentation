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

NORM_METHOD = "percentile"
P_LOW = 0.5
P_HIGH = 99.5
MODE_LOW = 100
MODE_HIGH = 254
TARGET_MODE = 200.0


def ensure_odd(val: int) -> int:
    return val if val % 2 == 1 else val + 1


# ==========================
# Core preprocessing
# ==========================

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Apply preprocessing IDENTICAL to napari (clahe + median).
    """

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


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert intensity range to uint8 using the notebook's percentile rules."""

    if img.dtype == np.uint8:
        return img

    x = img.astype(np.float32, copy=False)
    if NORM_METHOD == "divide256":
        x = np.clip(x / 256.0, 0, 255)
        return x.astype(np.uint8)

    if NORM_METHOD == "minmax":
        mn = float(np.min(x))
        mx = float(np.max(x))
    elif NORM_METHOD == "percentile":
        mn = float(np.percentile(x, P_LOW))
        mx = float(np.percentile(x, P_HIGH))
    else:
        raise ValueError("Unknown normalization method: %r" % NORM_METHOD)

    if mx <= mn:
        return np.zeros_like(x, dtype=np.uint8)

    scaled = (x - mn) * (255.0 / (mx - mn))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def detect_mode_above_threshold(img: np.ndarray, low: int = MODE_LOW, high: int = MODE_HIGH) -> int:
    """Return mode in [low, high] range for uint8 data."""

    if img.dtype != np.uint8:
        raise ValueError("Expected uint8 image for mode detection")

    flat = img.ravel()
    mask = (flat >= low) & (flat <= high)
    if not np.any(mask):
        return 0

    counts = np.bincount(flat[mask], minlength=256)
    return int(np.argmax(counts[low:high + 1]) + low)


def normalize_stack_to_mode_200(img: np.ndarray, mode: int, target: float = TARGET_MODE) -> np.ndarray:
    """Scale a uint8 array so that its mode moves close to `target`."""

    if mode <= 0:
        return img

    factor = target / float(mode)
    out = img.astype(np.float32) * factor
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def apply_norm_200_logic(img: np.ndarray) -> np.ndarray:
    """Run the notebook's normalization pipeline and keep float32 [0, 1]."""

    img255 = np.clip(img.astype(np.float32, copy=False) * 255.0, 0, 255)
    img8 = to_uint8(img255)
    mode = detect_mode_above_threshold(img8)
    normed = normalize_stack_to_mode_200(img8, mode)
    return normed.astype(np.float32) / 255.0


def normalize_loaded_image(img: np.ndarray) -> np.ndarray:
    """Convert 16-bit inputs to normalized float32 and validate incoming floats."""

    if np.issubdtype(img.dtype, np.uint16):
        float_img = img.astype(np.float32, copy=False) / 65535.0
    elif np.issubdtype(img.dtype, np.uint8):
        float_img = img.astype(np.float32, copy=False) / 255.0
    elif np.issubdtype(img.dtype, np.floating):
        float_img = img.astype(np.float32, copy=False)
    else:
        raise TypeError(f"Unsupported image dtype: {img.dtype}")

    if float_img.min() < 0.0 or float_img.max() > 1.0:
        raise ValueError("Float image values must already be within [0, 1]")

    float_img = np.clip(float_img, 0.0, 1.0)
    return apply_norm_200_logic(float_img)


def run_preprocessing_stage(input_dir: Path, output_dir: Path, overwrite: bool = False) -> None:
    """Run preprocessing using the provided stage configuration."""

    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        list(input_dir.glob("*.tif")) +
        list(input_dir.glob("*.tiff")) +
        list(input_dir.glob("*.png"))
    )

    if not images:
        raise RuntimeError("No images found")

    for img_path in images:
        out = output_dir / f"{img_path.stem}_clahe61_med{MEDIAN_RADIUS}.png"

        if out.exists() and not overwrite:
            continue

        img = normalize_loaded_image(tifffile.imread(img_path))
        if img.ndim != 2:
            raise ValueError(f"{img_path.name} is not 2D")

        img8 = preprocess_image(img)
        tifffile.imwrite(out, img8)

        print(f"Saved: {out.name}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="μCT preprocessing (EXACT napari clahe+median)"
    )
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_preprocessing_stage(args.input_dir, args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()
