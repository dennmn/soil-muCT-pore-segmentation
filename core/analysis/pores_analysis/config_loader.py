import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = PROJECT_ROOT / "config" / "pores_analysis" / "config.yaml"
_ENV_CONFIG_KEY = "PORES_ANALYSIS_CONFIG_JSON"


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _resolve_paths(config: Dict[str, Any]) -> None:
    project_root = PROJECT_ROOT
    paths = config.setdefault("paths", {})

    base_drive = paths.get("base_drive_path")
    if base_drive:
        base_drive = Path(base_drive)

    input_rel = paths.get("input_volume_relative_path")
    if input_rel and base_drive:
        paths["input_volume_path"] = str(base_drive / input_rel)
    elif input_rel:
        paths["input_volume_path"] = str(project_root / input_rel)

    output_dir = Path(paths.get("output_dir", "results"))
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    paths["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(paths.get("checkpoint_dir", "checkpoints"))
    if not checkpoint_dir.is_absolute():
        if base_drive:
            checkpoint_dir = base_drive / checkpoint_dir
        else:
            checkpoint_dir = project_root / checkpoint_dir
    paths["checkpoint_dir"] = str(checkpoint_dir)


def _load_env_overrides() -> Optional[Dict[str, Any]]:
    raw = os.environ.get(_ENV_CONFIG_KEY)
    if not raw:
        return None
    return json.loads(raw)


def load_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    env_overrides = _load_env_overrides()
    if env_overrides:
        config = deepcopy(env_overrides)
    else:
        if not _CONFIG_PATH.exists():
            raise FileNotFoundError(f"Config file not found: {_CONFIG_PATH}")
        with _CONFIG_PATH.open("r", encoding="utf-8") as handle:
            config = deepcopy(yaml.safe_load(handle) or {})

    if overrides:
        _deep_update(config, overrides)

    _resolve_paths(config)
    return config
