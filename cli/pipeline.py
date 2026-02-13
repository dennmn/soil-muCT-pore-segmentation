import argparse
import json
import logging
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from core.preprocessing.preprocess_ct_images import run_preprocessing_stage
from core.segmentation.multiotsu import run_multiotsu
from core.stability.postprocessing.binary_builder.run_binary_builder import (
    collapse_to_binary,
    load_image_stack,
    save_mask,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger(__name__)
# This temporary flag avoids GPU-heavy work during local validation runs.
DRY_RUN = os.environ.get("PIPELINE_DRY_RUN", "1").lower() in {"1", "true", "yes"}
STAGE_ORDER = ("preprocessing", "segmentation", "z_stability", "binary", "psd")
CONFIG_STAGE_NAMES = ("preprocessing", "segmentation", "z_stability", "binary_builder", "psd")
STAGE_VALIDATION_RULES: Dict[str, tuple[str, ...]] = {
    "preprocessing": (),
    "segmentation": ("method", "n_classes"),
    "z_stability": ("definition",),
    "binary_builder": (),
    "psd": ("entrypoint",),
}
PSD_ENTRYPOINT_MODULE = "core.analysis.pores_analysis.psd_entrypoint"


class PipelineError(Exception):
    pass


def _load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise PipelineError(f"Pipeline config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _get_stage_config(config: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
    return config.get("stages", {}).get(stage_name, {}) or {}


def _assert_stage_dir_has_contents(path: Path, description: str) -> None:
    if DRY_RUN:
        return
    if not path.exists():
        raise PipelineError(f"{description} missing at {path}")
    for _ in path.iterdir():
        return
    raise PipelineError(f"{description} is empty: {path}")


def _validate_stage_definition(stage_name: str, stage_def: Dict[str, Any]) -> None:
    if not isinstance(stage_def, dict):
        raise PipelineError(f"Stage configuration for {stage_name} must be a mapping")
    required_keys = STAGE_VALIDATION_RULES.get(stage_name, ())
    enabled = stage_def.get("enabled", True)
    if not enabled:
        return
    missing = [key for key in required_keys if key not in stage_def]
    if missing:
        raise PipelineError(
            f"Stage '{stage_name}' requires keys {', '.join(missing)} when enabled"
        )
    if stage_name == "z_stability":
        definition = stage_def.get("definition")
        if not isinstance(definition, dict):
            raise PipelineError("z_stability.definition must be an object")
        required = {"windows", "thresholds", "output", "target_class"}
        missing_def = required - definition.keys()
        if missing_def:
            raise PipelineError(
                f"z_stability.definition is missing keys: {', '.join(sorted(missing_def))}"
            )
        windows = definition.get("windows", {})
        if "short" not in windows or "long" not in windows:
            raise PipelineError("z_stability.definition.windows must include short and long values")
        thresholds = definition.get("thresholds", {})
        if "conservative" not in thresholds or "aggressive" not in thresholds:
            raise PipelineError("z_stability.definition.thresholds must define conservative and aggressive configs")


def _validate_config_schema(config: Dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise PipelineError("Pipeline configuration must be a mapping")
    if "scan_dir" not in config:
        raise PipelineError("Pipeline configuration requires 'scan_dir'")
    stages = config.get("stages")
    if not isinstance(stages, dict):
        raise PipelineError("Pipeline configuration requires a 'stages' block")
    missing = [stage for stage in CONFIG_STAGE_NAMES if stage not in stages]
    if missing:
        raise PipelineError(
            f"Pipeline configuration must include the following stages: {', '.join(missing)}"
        )
    for stage_name in CONFIG_STAGE_NAMES:
        _validate_stage_definition(stage_name, stages.get(stage_name, {}))


def _log_stage_keys(stage_name: str, stage_def: Dict[str, Any]) -> None:
    keys = sorted(stage_def.keys())
    key_display = ", ".join(keys) if keys else "<none>"
    logger.info("Stage %s resolved config keys: %s", stage_name, key_display)


def _run_subprocess(command: list[str], env: Dict[str, str] | None = None) -> None:
    logger.info("Running subprocess: %s", " ".join(str(p) for p in command))

    result = subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True,
    )

    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )


def _resolve_scan_dir(config: Dict[str, Any], override: Path | None) -> Path:
    source = override or config.get("scan_dir")
    if not source:
        raise PipelineError("Pipeline requires a scan directory. Provide --scan or set scan_dir in the config.")

    scan_dir = _resolve_path(source)
    if not scan_dir.exists():
        raise PipelineError(f"Scan directory not found: {scan_dir}")
    return scan_dir


def _resolve_scan_id(config: Dict[str, Any], override: str | None, scan_dir: Path) -> str:
    candidate = override or config.get("scan_id")
    return str(candidate or scan_dir.name)


def _build_stage_dirs(scan_dir: Path) -> Dict[str, Path]:
    base = scan_dir / "pipeline_outputs"
    stage_dirs: Dict[str, Path] = {}
    for stage in STAGE_ORDER:
        stage_path = base / stage
        stage_path.mkdir(parents=True, exist_ok=True)
        stage_dirs[stage] = stage_path
    if DRY_RUN:
        logger.info("DRY RUN stage directories: %s", {name: str(path) for name, path in stage_dirs.items()})
    return stage_dirs


def run_preprocessing(config: Dict[str, Any], stage_dirs: Dict[str, Path], scan_dir: Path) -> None:
    """Run the preprocessing stage, which populates the preprocessing directory with extracted slices."""
    stage = _get_stage_config(config, "preprocessing")
    if not stage or stage.get("enabled", True) is False:
        logger.info("Skipping preprocessing (not enabled)")
        return

    _log_stage_keys("preprocessing", stage)
    if not any(scan_dir.iterdir()):
        raise PipelineError(f"Scan directory is empty: {scan_dir}")
    if DRY_RUN:
        logger.info("DRY RUN: would preprocess %s into %s (overwrite=%s)", scan_dir, stage_dirs["preprocessing"], stage.get("overwrite", False))
        return
    output_dir = stage_dirs["preprocessing"]
    run_preprocessing_stage(
        input_dir=scan_dir,
        output_dir=output_dir,
        overwrite=bool(stage.get("overwrite", False)),
    )
    _assert_stage_dir_has_contents(output_dir, "Preprocessing outputs")


def run_segmentation(config: Dict[str, Any], stage_dirs: Dict[str, Path], scan_id: str) -> None:
    """Run the segmentation stage by invoking the multi-otsu helper and ensure the segmentation output exists."""
    stage = _get_stage_config(config, "segmentation")
    if not stage or stage.get("enabled", False) is False:
        logger.info("Segmentation stage disabled; run experiments manually.")
        return

    _log_stage_keys("segmentation", stage)
    method = stage.get("method")
    if method != "multiotsu":
        raise PipelineError("Only multiotsu segmentation is supported in this pipeline")

    input_dir = stage_dirs["preprocessing"]
    if not DRY_RUN:
        _assert_stage_dir_has_contents(input_dir, "Preprocessing outputs consumed by segmentation")
    output_root = stage_dirs["segmentation"]
    if DRY_RUN:
        logger.info(
            "DRY RUN: would run segmentation with input %s into %s (scan_id=%s)",
            input_dir,
            output_root,
            scan_id,
        )
        return

    run_multiotsu(
        input_dirs=[input_dir],
        output_root=output_root,
        n_classes=int(stage.get("n_classes", 3)),
        extensions=stage.get("extensions"),
        save_overlay=bool(stage.get("save_overlay", False)),
        scan_name=scan_id,
    )
    scan_output = output_root / scan_id
    _assert_stage_dir_has_contents(scan_output, f"Segmentation outputs for {scan_id}")


def run_z_stability(config: Dict[str, Any], stage_dirs: Dict[str, Path], scan_id: str) -> None:
    """Run z-stability diagnostics/corrections and enforce the required mask folder contract."""
    stage = _get_stage_config(config, "z_stability")
    if not stage or stage.get("enabled", False) is False:
        logger.info("Z-stability stage skipped (disabled by config).")
        return

    _log_stage_keys("z_stability", stage)
    definition = deepcopy(stage.get("definition", {}))
    seg_output = stage_dirs["segmentation"] / scan_id
    if not DRY_RUN:
        _assert_stage_dir_has_contents(seg_output, f"Segmentation outputs consumed by z_stability for {scan_id}")
    definition.setdefault("data", {})
    definition["data"]["scan_folder"] = str(seg_output)
    definition.setdefault("output", {})
    definition["output"]["output_folder"] = str(stage_dirs["z_stability"])
    if not seg_output.exists():
        logger.warning(
            "Segmentation output folder missing (expected after segmentation): %s",
            seg_output,
        )
    logger.info("Z-stability definition resolved: %s", json.dumps(definition))

    if DRY_RUN:
        logger.info(
            "DRY RUN: would invoke z_stability pipeline with config %s and definition above",
            stage.get("config_path", "config/z_stability/pipeline.yaml"),
        )
        return

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "core" / "stability" / "run_pipeline.py"),
        "--config",
        str(_resolve_path(stage.get("config_path", "config/z_stability/pipeline.yaml"))),
    ]
    env = os.environ.copy()
    env["Z_STABILITY_CONFIG_JSON"] = json.dumps(definition)
    _run_subprocess(cmd, env=env)
    scan_output = stage_dirs["z_stability"] / scan_id
    _assert_stage_dir_has_contents(scan_output, f"Z-stability outputs for {scan_id}")


def run_binary_builder(config: Dict[str, Any], stage_dirs: Dict[str, Path], scan_id: str) -> None:
    """Collapse z_stability mask slices into binary pore/solid artifacts for downstream analysis."""
    stage = _get_stage_config(config, "binary_builder")
    if not stage or stage.get("enabled", True) is False:
        logger.info("Binary builder stage skipped (disabled).")
        return

    _log_stage_keys("binary_builder", stage)
    mask_variant = stage.get("mask_variant", "aggressive")
    mask_folder = stage_dirs["z_stability"] / scan_id / f"mask_{mask_variant}"
    if DRY_RUN:
        logger.info("DRY RUN: would build binary masks from %s into %s", mask_folder, stage_dirs["binary"])
        return
    source_folder = mask_folder
    if not mask_folder.exists():
        seg_folder = stage_dirs["segmentation"] / scan_id
        classmap_folder = seg_folder / "classmap"
        if classmap_folder.exists():
            logger.warning(
                "Z-stability mask folder missing (%s); falling back to segmentation classmap %s",
                mask_folder,
                classmap_folder,
            )
            source_folder = classmap_folder
        elif seg_folder.exists():
            logger.warning(
                "Z-stability mask folder missing (%s); falling back to segmentation output %s",
                mask_folder,
                seg_folder,
            )
            source_folder = seg_folder
        else:
            raise PipelineError(f"Z-stability mask folder not found: {mask_folder}")

    logger.info("Building binary masks from %s", source_folder)
    volume = load_image_stack(source_folder)
    binary_pores, binary_solids = collapse_to_binary(volume)
    binary_root = stage_dirs["binary"]
    binary_root.mkdir(parents=True, exist_ok=True)
    np.save(binary_root / "binary_pores.npy", binary_pores)
    save_mask(binary_pores, binary_root / "binary_pores")
    save_mask(binary_solids, binary_root / "binary_solids")
    binary_file = binary_root / "binary_pores.npy"
    if not binary_file.exists():
        raise PipelineError(f"Binary pores array missing: {binary_file}")
    _assert_stage_dir_has_contents(binary_root, "Binary builder outputs")


def run_psd(config: Dict[str, Any], stage_dirs: Dict[str, Path], scan_id: str) -> None:
    """Construct the PSD config payload and run the pores_analysis entrypoint while verifying binary inputs exist."""
    stage = _get_stage_config(config, "psd")
    if not stage or stage.get("enabled", True) is False:
        logger.info("PSD stage skipped (disabled by config).")
        return

    _log_stage_keys("psd", stage)
    binary_volume = stage_dirs["binary"] / "binary_pores.npy"
    psd_root = stage_dirs["psd"]
    checkpoint_dir = psd_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _json_safe(v) for k, v in value.items()}
        if isinstance(value, set):
            return [_json_safe(v) for v in value]
        if isinstance(value, list):
            return [_json_safe(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_json_safe(v) for v in value)
        return value

    definition: Dict[str, Any] = {
        "paths": {
            "input_volume_path": str(binary_volume),
            "output_dir": str(psd_root),
            "checkpoint_dir": str(checkpoint_dir),
            "default_run_id": stage.get("default_run_id", f"{scan_id}_psd"),
        },
        "image_params": stage.get("image_params", {}),
        "processing_thresholds": stage.get("processing_thresholds", {}),
        "output_settings": stage.get("output_settings", {}),
    }

    safe_definition = _json_safe(definition)
    logger.info("PSD payload:")
    logger.info(json.dumps(safe_definition, indent=2))

    logger.info("PSD definition resolved: %s", json.dumps(definition))
    if DRY_RUN:
        logger.info(
            "DRY RUN: skipping PSD execution (entrypoint=%s, binary=%s)",
            PSD_ENTRYPOINT_MODULE,
            binary_volume,
        )
        return

    if not binary_volume.exists():
        raise PipelineError(f"Binary volume missing: {binary_volume}")

    env = os.environ.copy()
    try:
        env["PORES_ANALYSIS_CONFIG_JSON"] = json.dumps(safe_definition)
    except TypeError as exc:
        logger.error("Failed to serialize PSD payload: %s", exc)
        raise
    cmd = [
        sys.executable,
        "-m",
        PSD_ENTRYPOINT_MODULE,
    ]
    _run_subprocess(cmd, env=env)


def main(config_path: Path, scan_arg: Path | None, scan_id_arg: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger.info("Pipeline root: %s", PROJECT_ROOT)

    config = _load_config(config_path)
    _validate_config_schema(config)
    logger.info("Pipeline configuration validated")
    scan_dir = _resolve_scan_dir(config, scan_arg)
    scan_id = _resolve_scan_id(config, scan_id_arg, scan_dir)
    stage_dirs = _build_stage_dirs(scan_dir)

    logger.info("Scan directory: %s", scan_dir)
    logger.info("Scan identifier: %s", scan_id)
    for stage_name, stage_path in stage_dirs.items():
        logger.info("Stage '%s' folder: %s", stage_name, stage_path)

    run_preprocessing(config, stage_dirs, scan_dir)
    run_segmentation(config, stage_dirs, scan_id)
    run_z_stability(config, stage_dirs, scan_id)
    run_binary_builder(config, stage_dirs, scan_id)
    run_psd(config, stage_dirs, scan_id)


