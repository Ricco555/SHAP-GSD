"""
Phase 3: Hyperparameter tuning — 108 configurations, ~54h on A100.

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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.feature_store import FeatureStore
from src.model.node_state import NodeStateManager
from src.model.tuner import HyperparameterTuner
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(cfg: dict) -> None:
    repo_root = REPO_ROOT
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

    logger.info(
        f"Balanced train: {len(balanced_eids):,} EIDs, "
        f"class_weights: {class_weights.tolist()}"
    )

    # ── 5. Build tuner from configs/tuning_grid.yaml ───────────────────────────
    grid_cfg = load_config(
        repo_root / "configs" / "tuning_grid.yaml",
        default_path=repo_root / "configs" / "default.yaml",
    )
    tuning_ss  = grid_cfg.get("search_space", cfg["tuning"]["search_space"])
    tuning_fix = grid_cfg.get("fixed", {})
    tuning_fix.update({
        "num_layers": cfg["model"]["num_layers"],
        "aggregator": cfg["model"]["aggregator"],
        "learning_rate": cfg["model"]["learning_rate"],
        "temporal_window_seconds": cfg["model"]["temporal_window_seconds"],
        "node_state_dim": cfg["model"]["node_state_dim"],
    })

    tuner = HyperparameterTuner(
        search_space=tuning_ss,
        fixed_params=tuning_fix,
        max_epochs_per_trial=cfg["tuning"]["max_epochs_per_trial"],
        patience=cfg["tuning"]["patience_per_trial"],
        selection_metric=cfg["tuning"]["selection_metric"],
    )

    # ── 6. Run tuning ──────────────────────────────────────────────────────────
    output_dir = repo_root / cfg["output"]["artifacts_dir"] / "tuning"
    best_params = tuner.run(
        g_train=g_train,
        g_val=g_val,
        fs_train=fs_train,
        fs_val=fs_val,
        nsm=nsm,
        balanced_train_eids=balanced_eids,
        class_weights=class_weights,
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
    cfg  = load_config(REPO_ROOT / args.config)
    # Inject tuning config defaults if not present in experiment yaml
    cfg.setdefault("tuning", {
        "max_epochs_per_trial": 20,
        "patience_per_trial": 5,
        "selection_metric": "val_macro_f1",
        "search_space": {
            "fanouts":     [[15, 10], [25, 15], [35, 25]],
            "hidden_size": [64, 128, 256],
            "dropout":     [0.1, 0.2, 0.3, 0.4],
            "batch_size":  [512, 1024, 2048],
        },
    })
    main(cfg)
