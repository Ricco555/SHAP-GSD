"""YAML config loader with recursive default-override merge."""

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base; override wins on scalar conflicts."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def load_config(experiment_path: str | Path,
                default_path: str | Path | None = None) -> dict[str, Any]:
    """Load and merge default + experiment YAML configs.

    If default_path is None, looks for configs/default.yaml relative to
    experiment_path's parent directory.
    """
    experiment_path = Path(experiment_path)
    if default_path is None:
        default_path = experiment_path.parent / "default.yaml"

    with open(default_path) as f:
        cfg = yaml.safe_load(f)

    with open(experiment_path) as f:
        override = yaml.safe_load(f)

    if override:
        cfg = _deep_merge(cfg, override)

    return cfg
