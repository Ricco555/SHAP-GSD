"""Shared test fixture resolving config-driven paths.

Import ``resolve_cfg`` and ``REPO_ROOT`` rather than hardcoding artifact paths.
The config is selected via the ``SHAP_GSD_CONFIG`` environment variable
(default: ``configs/experiment_unsw.yaml``), so any run directory configured
via ``run.dir`` is picked up automatically.
"""

import os
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so ``src`` is importable.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config  # noqa: E402  (after sys.path adjustment)


def resolve_cfg() -> dict:
    """Load the merged config honoured by the current test run.

    Reads the config path from the ``SHAP_GSD_CONFIG`` environment variable.
    When unset, falls back to ``configs/experiment_unsw.yaml`` so that the
    existing UNSW artifacts remain discoverable without any extra setup.
    """
    cfg_rel = os.environ.get("SHAP_GSD_CONFIG", "configs/experiment_unsw.yaml")
    cfg_path = REPO_ROOT / cfg_rel
    return load_config(cfg_path)
