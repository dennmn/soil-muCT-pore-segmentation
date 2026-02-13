"""
Configuration Template (Reference Only)
======================================

`config.yaml` now hosts the single source of truth for all runtime constants.
Use `pores_analysis.config_loader.load_config()` to load this YAML and get
`paths`, `image_params`, `processing_thresholds`, and `output_settings`.

This module remains only to remind you where `config.yaml` lives.
"""

from pathlib import Path


def describe_config() -> None:
    """Print the expected location of the YAML config file."""
    config_path = Path(__file__).with_name("config.yaml")
    print("Centralized configuration:", config_path)


if __name__ == "__main__":
    describe_config()
