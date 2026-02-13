import os
os.environ["SKIMAGE_DISABLE_PLUGIN_SCAN"] = "1"

"""
napari_preproc_playground.py

Interactive preprocessing playground for 2D μCT soil slices using napari + magicgui.

PURPOSE:
- Choose preprocessing pipeline for ANNOTATION ONLY
- NO thresholding
- NO segmentation
- NO binary masks

Saved images:
- output/
- PNG
- filename encodes preprocessing parameters
"""

import sys
import argparse
import numpy as np
from skimage import io, img_as_float32
from skimage.util import img_as_ubyte
from skimage.exposure import equalize_adapthist
from scipy.ndimage import gaussian_filter, median_filter
from magicgui import magicgui
import napari


# -------------------------
# Utilities
# -------------------------
def ensure_odd(val):
    val = int(round(val))
    return val if val % 2 == 1 else val + 1


def load_image(path):
    if not os.path.isfile(path):
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    img = io.imread(path)

    if img.ndim == 3:
        img = img[..., 0]

    img = img_as_float32(img)
    img = np.nan_to_num(img)
    img = np.clip(img, 0, 1)

    return img


def ensure_output_dir():
    outdir = os.path.join(os.getcwd(), "output")
    os.makedirs(outdir, exist_ok=True)
    return outdir


# -------------------------
# Preprocessing ops
# -------------------------
def apply_gaussian(img, sigma):
    return gaussian_filter(img, sigma=sigma) if sigma > 0 else img


def apply_median(img, radius):
    return median_filter(img, size=ensure_odd(radius)) if radius > 0 else img


def apply_clahe(img, kernel, clip):
    return equalize_adapthist(
        img,
        kernel_size=ensure_odd(kernel),
        clip_limit=clip
    )


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True)
    args = parser.parse_args()

    img = load_image(args.img)
    output_dir = ensure_output_dir()

    state = {
        "mode": "raw",
        "gaussian_sigma": 1.0,
        "median_radius": 2,
        "clahe_kernel": 61,
        "clahe_clip": 0.01,
    }

    cache = {}

    def update_layers():
        cache["raw"] = img
        cache["gaussian"] = apply_gaussian(img, state["gaussian_sigma"])
        cache["median"] = apply_median(img, state["median_radius"])
        cache["clahe"] = apply_clahe(img, state["clahe_kernel"], state["clahe_clip"])
        cache["clahe_median"] = apply_median(cache["clahe"], state["median_radius"])

        cache["proc"] = {
            "raw": cache["raw"],
            "gaussian": cache["gaussian"],
            "median": cache["median"],
            "clahe": cache["clahe"],
            "clahe+median": cache["clahe_median"],
        }[state["mode"]]

    viewer = napari.Viewer(title="μCT Preprocessing Playground (NO MASKS)")
    update_layers()

    layers = {}
    for name in ["raw", "gaussian", "median", "clahe", "clahe_median"]:
        layers[name] = viewer.add_image(
            cache[name],
            name=name,
            visible=(name == "raw"),
        )

    @magicgui(
        auto_call=True,
        mode={"choices": ["raw", "gaussian", "median", "clahe", "clahe+median"]},
        gaussian_sigma={"min": 0, "max": 5, "step": 0.1},
        median_radius={"min": 1, "max": 10, "step": 1},
        clahe_kernel={"min": 3, "max": 255, "step": 2},
        clahe_clip={"min": 0.001, "max": 0.1, "step": 0.001},
    )
    def controls(
        mode="raw",
        gaussian_sigma=1.0,
        median_radius=2,
        clahe_kernel=61,
        clahe_clip=0.01,
    ):
        state.update(
            mode=mode,
            gaussian_sigma=gaussian_sigma,
            median_radius=median_radius,
            clahe_kernel=clahe_kernel,
            clahe_clip=clahe_clip,
        )

        update_layers()

        for layer in layers.values():
            layer.visible = False

        key = "clahe_median" if mode == "clahe+median" else mode
        layers[key].data = cache[key]
        layers[key].visible = True

    viewer.window.add_dock_widget(controls, area="right")

    @magicgui(call_button="Save processed image")
    def save_proc():
        base = os.path.splitext(os.path.basename(args.img))[0]

        label = (
            f"mode-{state['mode']}"
            f"__g-{state['gaussian_sigma']}"
            f"__m-{state['median_radius']}"
            f"__ck-{state['clahe_kernel']}"
            f"__cc-{state['clahe_clip']}"
        )

        filename = f"{base}__{label}.png"
        outpath = os.path.join(output_dir, filename)

        io.imsave(outpath, img_as_ubyte(cache["proc"]))
        print(f"Saved: {outpath}")

    viewer.window.add_dock_widget(save_proc, area="right")

    napari.run()


if __name__ == "__main__":
    main()
