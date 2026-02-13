"""Entrypoint for computing PSD on a real binary volume."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from .config_loader import load_config
from .psd_calculator import compute_psd
from .psd_output import psd_to_dataframe, save_psd_dataframe

logger = logging.getLogger(__name__)


def _normalize_export_formats(value: Any) -> List[str]:
    if not value:
        return ["csv"]
    if isinstance(value, str):
        value = [value]
    return [str(item).lower() for item in value]


def _build_compute_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    image_params = config.get("image_params") or {}
    processing = config.get("processing_thresholds") or {}
    paths = config.get("paths") or {}
    kwargs: Dict[str, Any] = {}

    voxel_spacing = image_params.get("voxel_spacing")
    if voxel_spacing:
        kwargs["voxel_spacing"] = tuple(float(v) for v in voxel_spacing)

    bin_edges = processing.get("bin_edges_um")
    if bin_edges is not None:
        kwargs["bin_edges"] = np.array(bin_edges, dtype=np.float32)

    if "use_gpu" in processing:
        kwargs["use_gpu"] = bool(processing["use_gpu"])
    if "use_chunking" in processing:
        kwargs["use_chunking"] = bool(processing["use_chunking"])
    if "chunk_size" in processing and processing["chunk_size"]:
        kwargs["chunk_size"] = tuple(int(value) for value in processing["chunk_size"])
    if "halo_width" in processing:
        kwargs["halo_width"] = int(processing["halo_width"])
    if "resume_from_checkpoint" in processing:
        kwargs["resume"] = bool(processing["resume_from_checkpoint"])

    checkpoint_dir = paths.get("checkpoint_dir")
    if checkpoint_dir:
        kwargs["checkpoint_dir"] = str(checkpoint_dir)
    run_id = paths.get("default_run_id")
    if run_id:
        kwargs["run_id"] = str(run_id)

    return kwargs


def _export_results(
    df, output_dir: Path, base_name: str, formats: Iterable[str], metadata: Dict[str, Any]
) -> None:
    base_path = output_dir / base_name
    for fmt in formats:
        save_psd_dataframe(df, str(base_path), format=fmt, metadata=metadata)
        logger.info("Exported PSD results as %s at %s", fmt, base_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Starting PSD entrypoint")

    try:
        config = load_config()
        paths = config.get("paths") or {}
        input_volume = paths.get("input_volume_path")
        if not input_volume:
            raise ValueError("PSD config missing paths.input_volume_path")
        input_path = Path(input_volume)
        if not input_path.exists():
            logger.error("Binary volume missing: %s", input_path)
            sys.exit(1)

        output_dir = Path(paths.get("output_dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Loading binary volume from %s", input_path)
        volume = np.load(input_path)

        compute_kwargs = _build_compute_kwargs(config)
        logger.info("Computing PSD with configuration: %s", compute_kwargs)
        psd = compute_psd(volume, **compute_kwargs)

        df = psd_to_dataframe(psd)
        output_settings = config.get("output_settings") or {}
        formats = _normalize_export_formats(output_settings.get("export_formats"))
        metadata = dict(output_settings.get("metadata") or {})
        metadata.setdefault("input_volume_path", str(input_path))
        metadata.setdefault("output_dir", str(output_dir))
        metadata.setdefault("export_formats", list(formats))
        if paths.get("default_run_id"):
            metadata.setdefault("run_id", paths["default_run_id"])
        base_name = output_settings.get("filename", "psd")
        _export_results(df, output_dir, base_name, formats, metadata)

        logger.info("PSD entrypoint completed")
    except Exception:
        logger.exception("PSD entrypoint failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
