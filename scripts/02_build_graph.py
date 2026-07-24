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
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.feature_store import read_edges_meta
from src.data.graph_builder import GraphBuilder
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

    # ── 1. Read Phase 1's per-split feature-store artifacts (no CSV reload) ────
    # Phase 2 is a consumer of Phase 1's global EID ordering. Rather than
    # re-load + re-sort the raw CSV (two independent sorts agreeing only by
    # coincidence), we read the per-split artifacts Phase 1 already wrote. The
    # raw columns Phase 2 still needs (IPs, bytes, dst_port) come from the new
    # edges_meta.parquet, aligned row-for-row with edge_indices.npy.
    fs_root = repo_root / cfg["output"]["feature_store_dir"]
    SPLITS: tuple[str, ...] = ("train", "val", "test")
    per_split: dict[str, dict[str, np.ndarray]] = {}
    for s in SPLITS:
        d = fs_root / s
        m = read_edges_meta(d)
        per_split[s] = {
            "edge_indices": np.load(d / "edge_indices.npy"),
            "timestamps":   np.load(d / "timestamps.npy"),
            "labels":       np.load(d / "labels.npy"),
            "src_ip":       m["src_ip"],
            "dst_ip":       m["dst_ip"],
            "in_bytes":     m["in_bytes"],    # already float32
            "out_bytes":    m["out_bytes"],   # already float32
            "dst_port":     m["dst_port"],    # already int32
        }
        logger.info(f"Read split '{s}': {len(per_split[s]['edge_indices']):,} edges")

    def _cat(key: str) -> np.ndarray:
        """Concatenate a per-split array across train→val→test in EID order."""
        return np.concatenate([per_split[s][key] for s in SPLITS])

    # All-edges arrays in global temporal order. Concatenating the contiguous
    # per-split EID ranges (train [0,n_train), val [n_train,…), test remainder)
    # reproduces Phase 1's exact arange-ordered global sequence — no second sort.
    src_ips_all   = _cat("src_ip")
    dst_ips_all   = _cat("dst_ip")
    ts_all        = _cat("timestamps")
    global_eids   = _cat("edge_indices")
    all_in_bytes  = _cat("in_bytes")     # already float32
    all_out_bytes = _cat("out_bytes")    # already float32
    all_dst_ports = _cat("dst_port")     # already int32

    # Tripwire (replaces the deleted CSV-length check): the concatenated per-split
    # EIDs must be the contiguous global arange (CRITICAL INVARIANT 2). A failure
    # means Phase 1's splits are no longer contiguous ranges — re-run Phase 1.
    assert np.array_equal(global_eids, np.arange(len(global_eids), dtype=np.int64)), (
        "Concatenated per-split EIDs are not the contiguous global arange — "
        "splits diverged; re-run Phase 1."
    )
    logger.info(
        f"Split sizes — train: {len(per_split['train']['edge_indices']):,}  "
        f"val: {len(per_split['val']['edge_indices']):,}  "
        f"test: {len(per_split['test']['edge_indices']):,}"
    )

    # ── 2. Build global node map ───────────────────────────────────────────────
    graph_dir = repo_root / cfg["graph"]["dir"]
    builder   = GraphBuilder(graph_dir=graph_dir)
    builder.build_global_node_map(src_ips_all, dst_ips_all)
    builder.save_node_id_map()

    is_internal_arr = GraphBuilder.compute_is_internal_array(builder.node_id_map)

    # ── 3. Build and save split graphs (directly from per-split arrays) ────────
    graphs = {}
    for s in SPLITS:
        ps = per_split[s]
        g = builder.build_split_graph(
            split_name=s,
            src_ips=ps["src_ip"],
            dst_ips=ps["dst_ip"],
            global_eids=ps["edge_indices"],
            timestamps=ps["timestamps"],
            labels=ps["labels"],
        )
        builder.validate_eid_alignment(g, fs_root / s)
        builder.save_split_graph(s, g)
        graphs[s] = g

    # ── 4. Build NodeStateManager ──────────────────────────────────────────────
    window_s    = cfg["model"]["temporal_window_seconds"]
    snap_interval = cfg["model"]["snapshot_interval"]

    nsm = NodeStateManager(
        window_seconds=window_s,
        snapshot_interval=snap_interval,
    )
    nsm.set_is_internal(is_internal_arr)

    # Baselines from training data only (the first per-split array, EIDs 0..n_train-1)
    train_ps = per_split["train"]
    train_src = train_ps["src_ip"]
    train_dst = train_ps["dst_ip"]

    # Convert IP strings to node IDs
    train_src_ids = np.array(
        [builder.node_id_map[str(ip)] for ip in train_src], dtype=np.int64
    )
    train_dst_ids = np.array(
        [builder.node_id_map[str(ip)] for ip in train_dst], dtype=np.int64
    )

    train_ts = train_ps["timestamps"]

    # IN_BYTES and OUT_BYTES for baseline computation (already float32)
    train_in_bytes  = train_ps["in_bytes"]
    train_out_bytes = train_ps["out_bytes"]

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
    # Concatenated all-edges byte/port arrays (already float32 / int32).
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

    # ── 5. Summary ─────────────────────────────────────────────────────────────
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
