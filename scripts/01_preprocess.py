"""
Phase 1: Data pipeline.

Usage:
  python scripts/01_preprocess.py --config configs/experiment_unsw.yaml

Outputs:
  feature_store/train/  (features.dat, edge_indices.npy, timestamps.npy, labels.npy)
  feature_store/val/
  feature_store/test/
  split_indices.json
  feature_groups.json
  balanced_train_indices.npy
  class_weights.npy
  artifacts/transformers/  (scaler.pkl, ohe.pkl, spearman_mask.npy, meta.json)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# ── repo root on sys.path so `src.*` imports resolve ──────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.balancer import TemporalBalancer
from src.data.feature_groups import FeatureGrouping
from src.data.feature_store import write_edges_meta, write_feature_store
from src.data.loader import CATEGORICAL_COLS, load_raw
from src.data.preprocessor import Preprocessor
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(cfg: dict) -> None:
    repo_root = REPO_ROOT

    # ── 1. Load raw data ───────────────────────────────────────────────────────
    csv_path = repo_root / cfg["data"]["csv_path"]
    df = load_raw(csv_path)

    # ── 2. Preprocess: split + feature engineering ────────────────────────────
    pre = Preprocessor(
        train_frac=cfg["data"]["train_frac"],
        val_frac=cfg["data"]["val_frac"],
        spearman_threshold=cfg["preprocessing"]["spearman_threshold"],
        ohe_min_frequency=cfg["preprocessing"]["ohe_min_frequency"],
        ohe_handle_unknown=cfg["preprocessing"]["ohe_handle_unknown"],
    )
    train_data, val_data, test_data = pre.fit_transform(df)
    del df  # free memory — 2.4M rows × 55 cols no longer needed

    # ── 3. Save label map ─────────────────────────────────────────────────────
    artifacts_dir = repo_root / cfg["output"]["artifacts_dir"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    label_map_path = artifacts_dir / "label_map.json"
    pre.save_label_map(label_map_path)

    # Log per-class counts in the training split for sanity checking
    train_labels = train_data["labels"]
    int_to_name = {v: k for k, v in pre.label_map.items()}
    logger.info("Training split class distribution:")
    unique, counts = np.unique(train_labels, return_counts=True)
    for cls_int, cnt in zip(unique.tolist(), counts.tolist()):
        logger.info(f"  {int_to_name.get(cls_int, str(cls_int)):<20s} (int={cls_int})  n={cnt:,}")

    # ── 4. Save transformers ───────────────────────────────────────────────────
    transformers_dir = repo_root / cfg["output"]["transformers_dir"]
    pre.save_transformers(transformers_dir)

    # ── 5. Write feature stores ───────────────────────────────────────────────
    fs_root = repo_root / cfg["output"]["feature_store_dir"]
    for split_name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        write_feature_store(
            split_dir=fs_root / split_name,
            features=data["features"],
            edge_indices=data["edge_indices"],
            timestamps=data["timestamps"],
            labels=data["labels"],
        )
        # Persist the raw columns Phase 2 needs (IPs, bytes, dst_port), aligned
        # row-for-row with edge_indices — lets Phase 2 skip the CSV reload+sort.
        write_edges_meta(
            fs_root / split_name,
            data["edges_meta"],
            n_expected=len(data["edge_indices"]),
        )

    # ── 6. Save split indices ─────────────────────────────────────────────────
    split_indices = pre.split_indices_dict()
    si_path = repo_root / cfg["output"]["split_indices_path"]
    with open(si_path, "w") as f:
        json.dump(split_indices, f, indent=2)
    logger.info(f"split_indices.json → {si_path}")
    logger.info(
        f"  train: {split_indices['splits']['train']['n_edges']:,}  "
        f"val: {split_indices['splits']['val']['n_edges']:,}  "
        f"test: {split_indices['splits']['test']['n_edges']:,}"
    )

    # ── 7. Build and save feature groups ──────────────────────────────────────
    ohe_names = list(pre.ohe.get_feature_names_out(CATEGORICAL_COLS))
    fg = FeatureGrouping.build(
        numeric_cols_kept=pre.numeric_cols_kept,
        ohe_feature_names=ohe_names,
        categorical_cols=CATEGORICAL_COLS,
    )
    fg_path = repo_root / cfg["output"]["feature_groups_path"]
    fg.save(fg_path)
    logger.info(f"feature_groups.json → {fg_path}  (K={fg.K}, d_e={fg.d_e})")

    # ── 8. Balance training split ─────────────────────────────────────────────
    bal_cfg = cfg["balancer"]
    balancer = TemporalBalancer(
        strategy=bal_cfg["strategy"],
        min_class_ratio=bal_cfg["min_class_ratio"],
        max_majority_ratio=bal_cfg["max_majority_ratio"],
        seed=bal_cfg["seed"],
    )
    balanced_eids = balancer.balance(
        edge_ids=train_data["edge_indices"],
        timestamps=train_data["timestamps"],
        labels=train_data["labels"],
    )
    bal_path = repo_root / cfg["output"]["balanced_train_indices_path"]
    np.save(bal_path, balanced_eids)
    logger.info(f"balanced_train_indices.npy → {bal_path}  ({len(balanced_eids):,} EIDs)")

    # ── 9. Class weights from ORIGINAL unbalanced distribution ────────────────
    # This source is intentionally hardcoded (train_data["labels"], never the
    # balanced/resampled EIDs) per CLAUDE.md CRITICAL INVARIANT #4 — it is not
    # a configurable choice, so no config key exposes it as one.
    class_weights = balancer.get_class_weights(
        train_data["labels"],
        num_classes=pre.num_classes,
        method=bal_cfg.get("class_weight_method", "effective_num"),
        beta=bal_cfg.get("effective_num_beta", 0.9999),
        max_clamp=bal_cfg.get("class_weight_max_clamp", None),
        log_weights=bal_cfg.get("log_class_weights", True),
    )
    cw_path = repo_root / cfg["output"]["class_weights_path"]
    np.save(cw_path, class_weights.numpy())
    logger.info(f"class_weights.npy → {cw_path}  (shape {class_weights.shape})")

    # ── 10. Sanity summary ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 1 complete. Output summary:")
    logger.info(f"  d_e = {pre.d_e}")
    logger.info(f"  K   = {fg.K}  semantic groups")
    logger.info(f"  Numeric features after pruning: {len(pre.numeric_cols_kept)}")
    logger.info(f"  OHE columns: {len(ohe_names)}")
    for split_name in ("train", "val", "test"):
        p = fs_root / split_name / "features.dat"
        size_mb = p.stat().st_size / 1e6
        logger.info(f"  feature_store/{split_name}/features.dat  {size_mb:.1f} MB")
    logger.info("=" * 60)

    # Final invariant: d_e matches across all outputs
    assert pre.d_e == fg.d_e, "d_e mismatch between Preprocessor and FeatureGrouping"
    assert pre.d_e == train_data["features"].shape[1]
    logger.info("All invariant checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment_unsw.yaml",
        help="Path to experiment YAML config (relative to repo root)",
    )
    args = parser.parse_args()

    config_path = REPO_ROOT / args.config
    cfg = load_config(config_path)
    main(cfg)
