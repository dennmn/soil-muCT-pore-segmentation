import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
from skimage.filters import threshold_multiotsu
import yaml

logger = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
OVERLAY_ALPHA = 0.35


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _list_images(folder: Path, extensions: Iterable[str]) -> list[Path]:
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions]
    return sorted(files)


def _make_overlay(gray: np.ndarray, mask: np.ndarray, alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = bgr.copy()
    overlay[mask > 0] = (0, 0, 255)
    return cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)


def run_multiotsu(
    input_dirs: Sequence[Path],
    output_root: Path,
    n_classes: int,
    extensions: Sequence[str] | None = None,
    save_overlay: bool = False,
    scan_name: Optional[str] = None,
) -> None:
    """Translated from multiotsu.ipynb cell 'COLAB: DARK PORES SEGMENTATION - PURE MULTI-OTSU'."""
    if extensions is None:
        extensions_set = DEFAULT_EXTENSIONS
    else:
        extensions_set = {ext.lower() for ext in extensions}

    seen = []
    for value in input_dirs:
        path = Path(value)
        if path not in seen:
            seen.append(path)
    resolved_inputs = seen
    logger.info("Running Multi-Otsu with %d classes", n_classes)
    logger.info("Segmentation inputs: %s", resolved_inputs)
    logger.info("Segmentation outputs: %s", output_root)

    stats: dict[str, dict[str, float | None]] = {}
    for input_dir in resolved_inputs:
        images = _list_images(input_dir, extensions_set)
        if not images:
            logger.warning("No images found under %s", input_dir)
            continue

        folder_name = scan_name or input_dir.name
        out_base = output_root / folder_name
        class_out = out_base / "classmap"
        mask0_out = out_base / "pores_class0"
        mask01_out = out_base / "pores_class0_plus_1"
        overlay0 = out_base / "overlays_class0"
        overlay01 = out_base / "overlays_class0_plus_1"
        for path in [class_out, mask0_out, mask01_out]:
            _ensure_dir(path)
        if save_overlay:
            _ensure_dir(overlay0)
            _ensure_dir(overlay01)

        threshold_records = []
        for img_path in images:
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                logger.warning("Failed to read %s", img_path)
                continue

            try:
                thresholds = threshold_multiotsu(gray, classes=n_classes)
                classmap = np.digitize(gray, bins=thresholds).astype(np.uint8)
            except Exception as error:
                logger.warning("Multi-Otsu failed for %s: %s", img_path, error)
                continue

            threshold_records.append([float(t) for t in thresholds])
            pores0 = (classmap == 0).astype(np.uint8) * 255
            pores01 = (classmap <= 1).astype(np.uint8) * 255

            base_name = img_path.stem
            cv2.imwrite(str(class_out / f"{base_name}_classmap.png"), classmap)
            cv2.imwrite(str(mask0_out / f"{base_name}_mask0.png"), pores0)
            cv2.imwrite(str(mask01_out / f"{base_name}_mask01.png"), pores01)

            if save_overlay:
                cv2.imwrite(str(overlay0 / f"{base_name}_ov0.png"), _make_overlay(gray, pores0))
                cv2.imwrite(str(overlay01 / f"{base_name}_ov01.png"), _make_overlay(gray, pores01))

        if threshold_records:
            arr = np.array(threshold_records)
            stats[folder_name] = {
                "mean_T1": float(arr[:, 0].mean()),
                "mean_T2": float(arr[:, 1].mean()) if n_classes > 2 else None,
            }
    logger.info("Multi-Otsu stats: %s", stats)


def _resolve_path(value: Path | str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path


def _load_pipeline_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the automated Multi-Otsu segmentation stage")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "config" / "pipeline.yaml",
    )
    args = parser.parse_args()
    pipeline = _load_pipeline_config(args.config)
    stage = pipeline.get("stages", {}).get("segmentation", {})
    if stage.get("enabled") is not True:
        logger.info("Segmentation stage is disabled in %s", args.config)
        return

    input_dirs = stage.get("input_dirs", [])
    output_root = stage.get("output_root")
    if not input_dirs or not output_root:
        raise ValueError("Segmentation stage requires 'input_dirs' and 'output_root'.")

    run_multiotsu(
        input_dirs=[_resolve_path(p) for p in input_dirs],
        output_root=_resolve_path(output_root),
        n_classes=int(stage.get("n_classes", 3)),
        extensions=stage.get("extensions", list(DEFAULT_EXTENSIONS)),
        save_overlay=bool(stage.get("save_overlay", False)),
    )


if __name__ == "__main__":
    main()
