"""Shared path resolver for exploration scripts.

Reads SHAP_GSD_CONFIG env var (default: configs/experiment_unsw.yaml) and
returns a dict of resolved paths that respect cfg["run"]["dir"].
"""

import os
import sys
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config  # noqa: E402


def resolve_cfg() -> dict:
    """Load the merged config honoured by the current exploration run.

    Reads the config path from the SHAP_GSD_CONFIG environment variable.
    When unset, falls back to configs/experiment_unsw.yaml so that the
    existing UNSW artifacts remain discoverable without any extra setup.
    """
    cfg_path = os.environ.get("SHAP_GSD_CONFIG", "configs/experiment_unsw.yaml")
    return load_config(REPO_ROOT / cfg_path)


def paths() -> dict:
    """Return a dict of resolved absolute paths for the current dataset run.

    All paths respect cfg["run"]["dir"]: when set (e.g. "runs/dataset02"),
    outputs/figures/metrics/etc. resolve under that prefix.  When unset
    (default UNSW run), bare paths like outputs/figures/ are returned.
    """
    cfg = resolve_cfg()
    outputs = REPO_ROOT / cfg["output"]["outputs_dir"]
    return {
        "cfg":          cfg,
        "outputs":      outputs,
        "metrics":      outputs / "metrics",
        "explanations": outputs / "explanations",
        "baselines":    outputs / "baselines",
        "w_ablation":   outputs / "w_ablation",
        "figures":      outputs / "figures",
        "fs":           REPO_ROOT / cfg["output"]["feature_store_dir"],
        "graphs":       REPO_ROOT / cfg["graph"]["dir"],
        "label_map":    REPO_ROOT / cfg["output"]["artifacts_dir"] / "label_map.json",
        "dataset_name": Path(cfg["data"]["csv_path"]).stem,
    }
