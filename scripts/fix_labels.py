"""
Fix binary labels in feature_store to multi-class Attack labels.

Phase 1 incorrectly stored df["Label"] (binary 0/1) instead of multi-class
integer codes derived from df["Attack"]. This script patches all three splits
without re-running the full Phase 1 pipeline.

Changes:
  feature_store/{train,val,test}/labels.npy  — overwritten with 10-class ints
  class_weights.npy                          — recomputed from 10-class train dist
  artifacts/label_map.json                   — written (Attack string → int)

Usage:
  python scripts/fix_labels.py --config configs/experiment_unsw.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.balancer import TemporalBalancer
from src.data.loader import load_raw
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Canonical ordering must match evaluator.DEFAULT_CLASS_NAMES exactly.
ATTACK_TO_INT: dict[str, int] = {
    "Benign":          0,
    "Generic":         1,
    "Exploits":        2,
    "Fuzzers":         3,
    "DoS":             4,
    "Reconnaissance":  5,   # dataset uses full name; display name is "Recon"
    "Analysis":        6,
    "Backdoor":        7,
    "Shellcode":       8,
    "Worms":           9,
}
N_CLASSES = len(ATTACK_TO_INT)


def encode_attack(attack_series) -> np.ndarray:
    """Map Attack strings → integer codes using ATTACK_TO_INT.

    Raises KeyError on any value not in the canonical mapping so we catch
    dataset surprises immediately rather than silently producing wrong labels.
    """
    codes = np.empty(len(attack_series), dtype=np.int64)
    for i, val in enumerate(attack_series):
        try:
            codes[i] = ATTACK_TO_INT[str(val)]
        except KeyError as exc:
            raise KeyError(
                f"Unknown Attack value '{val}' at row {i}. "
                f"Known values: {sorted(ATTACK_TO_INT)}"
            ) from exc
    return codes


def main(cfg: dict) -> None:
    repo_root = REPO_ROOT

    # ── 1. Load sorted CSV ─────────────────────────────────────────────────────
    csv_path = repo_root / cfg["data"]["csv_path"]
    df = load_raw(csv_path)
    df = df.sort_values("FLOW_START_MILLISECONDS", kind="mergesort").reset_index(drop=True)
    logger.info(f"Dataset sorted: {len(df):,} rows")

    # ── 2. Sanity-check Attack values ──────────────────────────────────────────
    unique_attack = sorted(df["Attack"].unique())
    logger.info(f"Unique Attack values ({len(unique_attack)}): {unique_attack}")
    unknown = [v for v in unique_attack if str(v) not in ATTACK_TO_INT]
    if unknown:
        raise ValueError(
            f"Attack column contains unmapped values: {unknown}. "
            "Update ATTACK_TO_INT in this script."
        )

    # ── 3. Encode all rows ─────────────────────────────────────────────────────
    logger.info("Encoding Attack → int ...")
    all_labels = encode_attack(df["Attack"])
    logger.info(f"Label distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")

    # ── 4. Load split boundaries ───────────────────────────────────────────────
    si_path = repo_root / cfg["output"]["split_indices_path"]
    with open(si_path) as f:
        si = json.load(f)
    tau_train_ms = si["tau_train_ms"]
    tau_val_ms   = si["tau_val_ms"]
    n_total      = si["n_total"]

    assert len(df) == n_total, (
        f"CSV row count {len(df):,} != split_indices n_total {n_total:,}. "
        "Re-run Phase 1 if the CSV was changed."
    )

    ts_all = df["FLOW_START_MILLISECONDS"].values.astype(np.int64)
    train_mask = ts_all <= tau_train_ms
    val_mask   = (ts_all > tau_train_ms) & (ts_all <= tau_val_ms)
    test_mask  = ts_all > tau_val_ms

    logger.info(
        f"Split sizes — train: {train_mask.sum():,}  "
        f"val: {val_mask.sum():,}  test: {test_mask.sum():,}"
    )

    # ── 5. Overwrite labels.npy for each split ─────────────────────────────────
    fs_root = repo_root / cfg["output"]["feature_store_dir"]
    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        labels_path = fs_root / split_name / "labels.npy"
        assert labels_path.exists(), f"Expected {labels_path} to exist (Phase 1 must be done first)"

        old = np.load(labels_path)
        new = all_labels[mask]

        assert len(old) == len(new), (
            f"{split_name}: old labels len {len(old)} != new labels len {len(new)}"
        )

        np.save(labels_path, new)
        uniq, cnts = np.unique(new, return_counts=True)
        logger.info(
            f"  {split_name}: {len(new):,} labels written  "
            f"classes present: {dict(zip(uniq.tolist(), cnts.tolist()))}"
        )

    # ── 6. Recompute class_weights from ORIGINAL (unbalanced) train labels ─────
    train_labels = all_labels[train_mask]
    bal_cfg = cfg["balancer"]
    balancer = TemporalBalancer()
    weights_tensor = balancer.get_class_weights(
        train_labels,
        method=bal_cfg.get("class_weight_method", "effective_num"),
        beta=bal_cfg.get("effective_num_beta", 0.9999),
        max_clamp=bal_cfg.get("class_weight_max_clamp", None),
        log_weights=bal_cfg.get("log_class_weights", True),
    )

    cw_path = repo_root / cfg["output"]["class_weights_path"]
    np.save(cw_path, weights_tensor.numpy())
    logger.info(f"class_weights.npy written → {cw_path}  shape={weights_tensor.shape}")

    # ── 7. Save label_map.json ─────────────────────────────────────────────────
    # Maps display names → int (used by evaluator.py class_names lookup).
    # "Reconnaissance" in the CSV becomes "Recon" in plots/tables.
    display_label_map = {
        "Benign":    0,
        "Generic":   1,
        "Exploits":  2,
        "Fuzzers":   3,
        "DoS":       4,
        "Recon":     5,
        "Analysis":  6,
        "Backdoor":  7,
        "Shellcode": 8,
        "Worms":     9,
    }
    artifacts_dir = repo_root / cfg["output"]["artifacts_dir"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    lmap_path = artifacts_dir / "label_map.json"
    with open(lmap_path, "w") as f:
        json.dump(display_label_map, f, indent=2)
    logger.info(f"label_map.json written → {lmap_path}")

    logger.info("Label fix complete. Run scripts/02_build_graph.py next.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_unsw.yaml")
    args = parser.parse_args()
    cfg = load_config(REPO_ROOT / args.config)
    main(cfg)
