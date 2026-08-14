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
  python scripts/03_tune.py --config <yaml> --shard-axes fanouts,hidden_size --shard K/9

Outputs:
  artifacts/tuning/tuning_results.json   (all 108 trial results)
  artifacts/tuning/best_params.json      (best hyperparameters)
  artifacts/tuning/tuning_results_shard_*.json  (shard mode: this shard's
                                          results only — no tuning_results.json
                                          or best_params.json is written)
  Shard mode requires a final scripts/promote_best.py run to merge shard
  files and emit tuning_results.json + best_params.json (specs/34, specs/35).
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
from src.model.selection import compute_grid_fingerprint, resolve_shard
from src.model.tuner import (
    HyperparameterTuner,
    resolve_selection_settings,
    resolve_trial_settings,
)
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


def main(
    cfg: dict,
    config_path: Path,
    shard_axes: str | None = None,
    shard: str | None = None,
) -> None:
    """Run Phase-3 grid tuning — the full grid, or one shard of it.

    Args:
        cfg:         merged experiment config (load_config output).
        config_path: path to the experiment YAML (for the pre-merge guard).
        shard_axes:  raw ``--shard-axes`` CLI string, or None (single job).
        shard:       raw ``--shard`` CLI string ``K/N``, or None (single job).
    """
    repo_root = REPO_ROOT
    _assert_early_stopping_metric_declared(config_path)
    if (shard_axes is None) != (shard is None):
        raise ValueError(
            "--shard-axes and --shard must be given together "
            "(or neither, for a full single-job run)."
        )
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
    nsm = NodeStateManager.load(
        repo_root / cfg["graph"]["node_state_dir"],
        expected_novelty_mode=cfg["model"].get("novelty_mode", "recent_window"),
    )

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

    trial_settings     = resolve_trial_settings(grid_cfg)
    selection_settings = resolve_selection_settings(grid_cfg)

    # ── 5b. Resolve shard ownership + grid fingerprint (shard mode only) ───────
    # Placed after the search_space fail-fast, the fixed: overwrite and the
    # num_classes injection, so the fingerprint sees the fully effective
    # model: block. All shard-selector assertions (unknown axis, batch_size
    # exclusion, K/N validation) fire inside resolve_shard, before any GPU work.
    shard_spec = None
    if shard_axes is not None:
        shard_spec = resolve_shard(tuning_ss, shard_axes, shard)
        shard_spec["grid_fingerprint"] = compute_grid_fingerprint(
            model_block=cfg["model"],
            search_space=tuning_ss,
            trial_block={
                "max_epochs": trial_settings["max_epochs_per_trial"],
                "patience":   trial_settings["patience"],
            },
        )
        shard_spec["tie_band_pp"]    = selection_settings["tie_band_pp"]
        shard_spec["tie_break_axes"] = selection_settings["tie_break_axes"]
        shard_spec["search_space"]   = tuning_ss

    tuner = HyperparameterTuner(
        search_space=tuning_ss,
        fixed_params=tuning_fix,
        **trial_settings,
        **selection_settings,
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
    if shard_spec is not None:
        logger.info(
            "Shard mode: scheme=axis, axes=%s, shard %d/%d owns %s -> "
            "%d/%d trials, global indices=%s",
            shard_spec["shard_axes"], shard_spec["shard_index"],
            shard_spec["num_shards"], shard_spec["owned_values"],
            len(shard_spec["owned_indices"]), shard_spec["n_total"],
            shard_spec["owned_indices"],
        )
        logger.info("Shard results file: %s",
                    output_dir / shard_spec["filename"])
        logger.info("Grid fingerprint: %s", shard_spec["grid_fingerprint"])
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
        shard=shard_spec,
    )

    if shard_spec is not None:
        logger.info(f"Shard-local best params (NOT promoted): {best_params}")
        logger.info(
            "Shard complete. Run scripts/promote_best.py after ALL shards "
            "finish to merge shard files and write best_params.json."
        )
    else:
        logger.info(f"Best params: {best_params}")
        logger.info("Tuning complete. Run scripts/04_train.py to train with best params.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_unsw.yaml")
    parser.add_argument(
        "--shard-axes", default=None, metavar="AXIS[,AXIS...]",
        help="Comma-separated search_space axis names to partition the grid "
             "by (e.g. fanouts,hidden_size). Requires --shard. batch_size "
             "is rejected (dominant-cost axis).",
    )
    parser.add_argument(
        "--shard", default=None, metavar="K/N",
        help="Own shard K of N under --shard-axes (e.g. 4/9). N must equal "
             "the product of the chosen axes' cardinalities.",
    )
    args = parser.parse_args()
    config_path = REPO_ROOT / args.config
    main(load_config(config_path), config_path,
         shard_axes=args.shard_axes, shard=args.shard)
