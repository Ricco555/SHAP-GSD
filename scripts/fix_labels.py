"""
Fix binary labels in feature_store to multi-class Attack labels.

Phase 1 incorrectly stored df["Label"] (binary 0/1) instead of multi-class
integer codes derived from df["Attack"]. This script patches all three splits
without re-running the full Phase 1 pipeline.

Changes:
  feature_store/{train,val,test}/labels.npy  — overwritten with N-class ints
  class_weights.npy                          — recomputed from N-class train dist
  artifacts/label_map.json                   — written (Attack string → int)

Usage:
  python scripts/fix_labels.py --config configs/experiment_unsw.yaml
  python scripts/fix_labels.py --config configs/experiment_unsw.yaml \\
      --label-map artifacts/label_map.json
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


def encode_attack(attack_series, attack_to_int: dict[str, int]) -> np.ndarray:
    """Map Attack strings → integer codes using the provided mapping.

    Raises KeyError on any value not in the mapping so we catch dataset
    surprises immediately rather than silently producing wrong labels.

    Args:
        attack_series: iterable of Attack string values.
        attack_to_int: mapping of Attack string → integer label.

    Returns:
        int64 array of integer label codes, same length as attack_series.
    """
    codes = np.empty(len(attack_series), dtype=np.int64)
    for i, val in enumerate(attack_series):
        try:
            codes[i] = attack_to_int[str(val)]
        except KeyError as exc:
            raise KeyError(
                f"Unknown Attack value '{val}' at row {i}. "
                f"Known values: {sorted(attack_to_int)}"
            ) from exc
    return codes


def main(cfg: dict, label_map_path: Path) -> None:
    repo_root = REPO_ROOT

    # ── 0. Load label map ──────────────────────────────────────────────────────
    if label_map_path.exists():
        with open(label_map_path) as f:
            attack_to_int: dict[str, int] = json.load(f)
        logger.info(f"Loaded label map from {label_map_path}: {attack_to_int}")
    else:
        # Bootstrap: derive the map from the CSV and write it
        logger.warning(
            f"{label_map_path} not found — deriving label map from CSV Attack column. "
            "Run 01_preprocess.py first to generate a persistent label_map.json."
        )
        from src.data.preprocessor import Preprocessor
        csv_path_tmp = repo_root / cfg["data"]["csv_path"]
        df_tmp = load_raw(csv_path_tmp)
        attack_to_int = Preprocessor.build_label_map(df_tmp)
        del df_tmp
        label_map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_map_path, "w") as f:
            json.dump(attack_to_int, f, indent=2)
        logger.info(f"Derived label map saved → {label_map_path}: {attack_to_int}")

    n_classes = len(attack_to_int)

    # ── 1. Load sorted CSV ─────────────────────────────────────────────────────
    csv_path = repo_root / cfg["data"]["csv_path"]
    df = load_raw(csv_path)
    df = df.sort_values("FLOW_START_MILLISECONDS", kind="mergesort").reset_index(drop=True)
    logger.info(f"Dataset sorted: {len(df):,} rows")

    # ── 2. Sanity-check Attack values ──────────────────────────────────────────
    unique_attack = sorted(df["Attack"].unique())
    logger.info(f"Unique Attack values ({len(unique_attack)}): {unique_attack}")
    unknown = [v for v in unique_attack if str(v) not in attack_to_int]
    if unknown:
        raise ValueError(
            f"Attack column contains unmapped values: {unknown}. "
            "Update the label map or re-run 01_preprocess.py."
        )

    # ── 3. Encode all rows ─────────────────────────────────────────────────────
    logger.info("Encoding Attack → int ...")
    all_labels = encode_attack(df["Attack"], attack_to_int)
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
    # Re-write the loaded label_map so the file reflects what was actually used.
    artifacts_dir = repo_root / cfg["output"]["artifacts_dir"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    lmap_path = artifacts_dir / "label_map.json"
    with open(lmap_path, "w") as f:
        json.dump(attack_to_int, f, indent=2)
    logger.info(f"label_map.json written → {lmap_path}  ({n_classes} classes)")

    logger.info("Label fix complete. Run scripts/02_build_graph.py next.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_unsw.yaml")
    parser.add_argument(
        "--label-map",
        default="artifacts/label_map.json",
        help=(
            "Path to label_map.json (Attack string → int). "
            "If the file does not exist, the map is derived from the CSV "
            "Attack column and written to this path. "
            "Default: artifacts/label_map.json"
        ),
    )
    args = parser.parse_args()
    cfg = load_config(REPO_ROOT / args.config)
    lmap = REPO_ROOT / args.label_map
    main(cfg, lmap)
