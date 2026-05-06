"""
Phase 2: Build DGL graphs and node state snapshots.

Usage:
  python scripts/02_build_graph.py --config configs/experiment_unsw.yaml

Outputs:
  graphs/train.bin, graphs/val.bin, graphs/test.bin
  graphs/node_id_map.json
  node_state_snapshots/  (baselines.pkl, histories.pkl, snapshots.pkl, ...)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.feature_store import FeatureStore
from src.data.graph_builder import GraphBuilder
from src.data.loader import load_raw
from src.model.node_state import NodeStateManager
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(cfg: dict) -> None:
    repo_root = REPO_ROOT

    # ── 1. Load and sort raw data (same order as Phase 1) ─────────────────────
    csv_path = repo_root / cfg["data"]["csv_path"]
    logger.info(f"Loading {csv_path}")
    df = load_raw(csv_path)
    df = df.sort_values("FLOW_START_MILLISECONDS", kind="mergesort").reset_index(drop=True)
    logger.info(f"Sorted dataset: {len(df):,} rows")

    # ── 2. Load split boundaries from Phase 1 ─────────────────────────────────
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

    # Assign global EIDs (position in sorted dataset, 0-indexed)
    global_eids = np.arange(n_total, dtype=np.int64)

    train_mask = ts_all <= tau_train_ms
    val_mask   = (ts_all > tau_train_ms) & (ts_all <= tau_val_ms)
    test_mask  = ts_all > tau_val_ms

    logger.info(
        f"Split sizes — train: {train_mask.sum():,}  "
        f"val: {val_mask.sum():,}  test: {test_mask.sum():,}"
    )

    src_ips_all = df["IPV4_SRC_ADDR"].values
    dst_ips_all = df["IPV4_DST_ADDR"].values

    # Multi-class labels from Attack column (matches evaluator.DEFAULT_CLASS_NAMES order)
    _attack_to_int = {
        "Benign": 0, "Generic": 1, "Exploits": 2, "Fuzzers": 3, "DoS": 4,
        "Reconnaissance": 5, "Analysis": 6, "Backdoor": 7, "Shellcode": 8, "Worms": 9,
    }
    labels_all = np.array([_attack_to_int[str(v)] for v in df["Attack"].values], dtype=np.int64)

    # ── 3. Build global node map ───────────────────────────────────────────────
    graph_dir = repo_root / cfg["graph"]["dir"]
    builder   = GraphBuilder(graph_dir=graph_dir)
    builder.build_global_node_map(src_ips_all, dst_ips_all)
    builder.save_node_id_map()

    is_internal_arr = GraphBuilder.compute_is_internal_array(builder.node_id_map)

    # ── 4. Build and save split graphs ─────────────────────────────────────────
    splits = {
        "train": train_mask,
        "val":   val_mask,
        "test":  test_mask,
    }
    graphs = {}
    for split_name, mask in splits.items():
        fs = FeatureStore(repo_root / cfg["output"]["feature_store_dir"] / split_name)

        g = builder.build_split_graph(
            split_name=split_name,
            src_ips=src_ips_all[mask],
            dst_ips=dst_ips_all[mask],
            global_eids=global_eids[mask],
            timestamps=ts_all[mask],
            labels=labels_all[mask],
        )
        builder.validate_eid_alignment(
            g,
            repo_root / cfg["output"]["feature_store_dir"] / split_name,
        )
        builder.save_split_graph(split_name, g)
        graphs[split_name] = g

    # ── 5. Build NodeStateManager ──────────────────────────────────────────────
    window_s    = cfg["model"]["temporal_window_seconds"]
    snap_interval = cfg["model"]["snapshot_interval"]

    nsm = NodeStateManager(
        window_seconds=window_s,
        snapshot_interval=snap_interval,
    )
    nsm.set_is_internal(is_internal_arr)

    # Baselines from training data only (EIDs 0..n_train-1)
    train_src = src_ips_all[train_mask]
    train_dst = dst_ips_all[train_mask]

    # Convert IP strings to node IDs
    train_src_ids = np.array(
        [builder.node_id_map[str(ip)] for ip in train_src], dtype=np.int64
    )
    train_dst_ids = np.array(
        [builder.node_id_map[str(ip)] for ip in train_dst], dtype=np.int64
    )

    train_ts = ts_all[train_mask]

    # IN_BYTES and OUT_BYTES for baseline and snapshot computation
    train_in_bytes  = df["IN_BYTES"].values[train_mask].astype(np.float32)
    train_out_bytes = df["OUT_BYTES"].values[train_mask].astype(np.float32)

    logger.info("Building hourly baselines from training data...")
    nsm.build_hourly_baselines(
        src_node_ids=train_src_ids,
        dst_node_ids=train_dst_ids,
        timestamps_ms=train_ts,
        in_bytes=train_in_bytes,
        out_bytes=train_out_bytes,
    )

    # Snapshots from ALL edges (preserves correct node state for val/test)
    all_src_ids = np.array(
        [builder.node_id_map[str(ip)] for ip in src_ips_all], dtype=np.int64
    )
    all_dst_ids = np.array(
        [builder.node_id_map[str(ip)] for ip in dst_ips_all], dtype=np.int64
    )
    all_in_bytes  = df["IN_BYTES"].values.astype(np.float32)
    all_out_bytes = df["OUT_BYTES"].values.astype(np.float32)
    all_dst_ports = df["L4_DST_PORT"].values.astype(np.int32)

    logger.info("Building node state snapshots (all edges)...")
    nsm.build_snapshots(
        src_node_ids=all_src_ids,
        dst_node_ids=all_dst_ids,
        timestamps_ms=ts_all,
        in_bytes=all_in_bytes,
        out_bytes=all_out_bytes,
        dst_ports=all_dst_ports,
        snapshot_interval=snap_interval,
    )

    snap_dir = repo_root / cfg["graph"]["node_state_dir"]
    nsm.save(snap_dir)

    # ── 6. Summary ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 2 complete:")
    logger.info(f"  Graphs:     {graph_dir}/  (train/val/test.bin)")
    logger.info(f"  NodeState:  {snap_dir}/")
    logger.info(f"  Nodes:      {builder._num_nodes:,}")
    for split_name, g in graphs.items():
        logger.info(f"  {split_name}: {g.num_edges():,} edges")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_unsw.yaml")
    args   = parser.parse_args()
    cfg    = load_config(REPO_ROOT / args.config)
    main(cfg)
