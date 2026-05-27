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


def _apply_run_dir_prefix(cfg: dict) -> dict:
    """Prefix all relative output/graph paths with ``run.dir`` when it is set.

    When ``run.dir`` is empty (the default) the config is returned unchanged,
    preserving backwards-compatibility with existing on-disk artifacts.
    Absolute paths are never modified.
    """
    run_dir = cfg.get("run", {}).get("dir", "") or ""
    if not run_dir:
        return cfg
    prefix = Path(run_dir)
    for section in ("output", "graph"):
        if section not in cfg:
            continue
        for k, v in cfg[section].items():
            if isinstance(v, str) and v and not Path(v).is_absolute():
                cfg[section][k] = str(prefix / v)
    return cfg


def load_config(experiment_path: str | Path,
                default_path: str | Path | None = None) -> dict[str, Any]:
    """Load and merge default + experiment YAML configs.

    If default_path is None, looks for configs/default.yaml relative to
    experiment_path's parent directory.

    After merging, applies ``_apply_run_dir_prefix`` so that a non-empty
    ``run.dir`` transparently redirects all output and graph paths without
    requiring callers to be aware of the prefix.
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

    cfg = _apply_run_dir_prefix(cfg)

    return cfg
