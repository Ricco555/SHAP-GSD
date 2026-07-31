"""
Phase 3: Hyperparameter tuning — 108 configurations (configs/tuning_grid.yaml
search_space: 3x3x4x3).

Per-trial budget (max_epochs, patience) is read from that file's trial: block —
it is NOT hardcoded here. Walltime scales with 108 x trial.max_epochs; at the
shipped 40 epochs the worst case is 4,320 epochs (~108 h on A100 by the
pre-searchsorted-sampler UNSW estimate, against a 168 h HPC queue cap —
specs/33 §II.2.3).

Prerequisites:
  - Phase 1 complete: feature_store/ exists
  - Phase 2 complete: graphs/ and node_state_snapshots/ exist
  - All gate tests pass: pytest tests/ -v (excluding test_shap_axioms.py)

Usage:
  python scripts/03_tune.py --config configs/experiment_unsw.yaml

Outputs:
  artifacts/tuning/tuning_results.json   (all 108 trial results)
  artifacts/tuning/best_params.json      (best hyperparameters)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.feature_store import FeatureStore
from src.model.node_state import NodeStateManager
from src.model.tuner import HyperparameterTuner, resolve_trial_settings
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _assert_early_stopping_metric_declared(config_path: Path) -> None:
    """Fail closed unless the experiment config itself sets ``model.early_stopping_metric``.

    ``configs/default.yaml`` always supplies ``early_stopping_metric: "macro_f1"``,
    so a merged config can never be missing the key — checking the merged dict
    would never fire. This checks the experiment YAML directly, before the
    merge, so a dataset config that forgot to declare its own selection metric
    is caught rather than silently tuning 108 trials against the wrong
    objective. Phase 3 shard jobs (specs/34) call this script directly and
    bypass ``run_dataset.py``'s bare-stub guard entirely, so this is the only
    remaining gate. See specs/34 §0.1.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    if "early_stopping_metric" not in (raw.get("model") or {}):
        raise KeyError(
            f"{config_path} has no model.early_stopping_metric of its own — "
            "refusing to silently inherit configs/default.yaml's 'macro_f1' "
            "default for a 108-trial grid search. Add a model: block with "
            "early_stopping_metric to the experiment config. See specs/34 §0.1."
        )


def main(cfg: dict, config_path: Path) -> None:
    repo_root = REPO_ROOT
    _assert_early_stopping_metric_declared(config_path)
    device = torch.device(
        cfg["compute"]["device"] if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Device: {device}")

    # ── 1. Load split graphs ───────────────────────────────────────────────────
    import dgl
    graph_dir = repo_root / cfg["graph"]["dir"]
    g_train_list, _ = dgl.load_graphs(str(graph_dir / "train.bin"))
    g_val_list,   _ = dgl.load_graphs(str(graph_dir / "val.bin"))
    g_train = g_train_list[0]
    g_val   = g_val_list[0]
    logger.info(f"Graphs loaded: train={g_train.num_edges():,}, val={g_val.num_edges():,}")

    # ── 2. Load feature stores ─────────────────────────────────────────────────
    fs_dir  = repo_root / cfg["output"]["feature_store_dir"]
    fs_train = FeatureStore(fs_dir / "train")
    fs_val   = FeatureStore(fs_dir / "val")

    # ── 3. Load NodeStateManager ───────────────────────────────────────────────
    nsm = NodeStateManager.load(repo_root / cfg["graph"]["node_state_dir"])

    # ── 4. Load balanced EIDs and class weights ────────────────────────────────
    balanced_eids = np.load(
        repo_root / cfg["output"]["balanced_train_indices_path"]
    )
    class_weights = torch.from_numpy(
        np.load(repo_root / cfg["output"]["class_weights_path"])
    ).float()
    train_labels = np.load(
        repo_root / cfg["output"]["feature_store_dir"] / "train" / "labels.npy"
    )

    # Derive num_classes from label_map.json (produced by 01_preprocess.py).
    label_map_path = repo_root / cfg["output"]["artifacts_dir"] / "label_map.json"
    with open(label_map_path) as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    cfg["model"]["num_classes"] = num_classes
    logger.info(f"num_classes={num_classes} (loaded from {label_map_path})")

    train_label_counts = np.bincount(
        train_labels, minlength=num_classes
    ).astype(np.int64)

    logger.info(
        f"Balanced train: {len(balanced_eids):,} EIDs, "
        f"class_weights: {class_weights.tolist()}"
    )

    # ── 5. Build tuner from configs/tuning_grid.yaml ───────────────────────────
    grid_cfg = load_config(
        repo_root / "configs" / "tuning_grid.yaml",
        default_path=repo_root / "configs" / "default.yaml",
    )
    if "search_space" not in grid_cfg:
        raise KeyError(
            "configs/tuning_grid.yaml is missing the required 'search_space:' "
            "block. Refusing to guess a hyperparameter grid — see specs/33."
        )
    tuning_ss  = grid_cfg["search_space"]
    tuning_fix = grid_cfg.get("fixed", {})
    tuning_fix.update({
        "num_layers": cfg["model"]["num_layers"],
        "aggregator": cfg["model"]["aggregator"],
        "learning_rate": cfg["model"]["learning_rate"],
        "temporal_window_seconds": cfg["model"]["temporal_window_seconds"],
        "node_state_dim": cfg["model"]["node_state_dim"],
    })

    trial_settings = resolve_trial_settings(grid_cfg)
    tuner = HyperparameterTuner(
        search_space=tuning_ss,
        fixed_params=tuning_fix,
        **trial_settings,
    )

    # ── 6. Run tuning ──────────────────────────────────────────────────────────
    logger.info(
        "Effective Phase 3 selection settings: early_stopping_metric=%r, "
        "minority_class_threshold=%r, composite_minority_weight=%r",
        cfg["model"]["early_stopping_metric"],
        cfg["model"].get("minority_class_threshold"),
        cfg["model"].get("composite_minority_weight"),
    )
    output_dir = repo_root / cfg["output"]["artifacts_dir"] / "tuning"
    best_params = tuner.run(
        g_train=g_train,
        g_val=g_val,
        fs_train=fs_train,
        fs_val=fs_val,
        nsm=nsm,
        balanced_train_eids=balanced_eids,
        class_weights=class_weights,
        train_label_counts=train_label_counts,
        node_in_dim=cfg["model"]["node_state_dim"],
        edge_in_dim=fs_train.d_e,
        num_classes=cfg["model"]["num_classes"],
        base_cfg=cfg,
        device=device,
        output_dir=output_dir,
        seed=cfg["reproducibility"]["model_seed"],
    )

    logger.info(f"Best params: {best_params}")
    logger.info("Tuning complete. Run scripts/04_train.py to train with best params.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_unsw.yaml")
    args = parser.parse_args()
    config_path = REPO_ROOT / args.config
    main(load_config(config_path), config_path)
