"""
Phase 2: Build DGL graphs and node state snapshots.

Usage:
  python scripts/02_build_graph.py --config configs/experiment_unsw.yaml

Outputs:
  graphs/train.bin, graphs/val.bin, graphs/test.bin
  graphs/node_id_map.json
  node_state_snapshots/  (baselines.pkl, histories.pkl, snapshots.pkl, ...)
                         snapshots.pkl is empty ({"times": [], "states": []})
                         by default — model.snapshot_interval=0 disables the
                         Pass 2 snapshot cache (see src/model/node_state.py).
"""

import argparse
import gc
import logging
import resource
import sys
import time
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


def _maxrss_mb() -> float:
    """Peak resident set size so far, in MiB (kilobytes on Linux, this project's target)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _log_stage(stage: str, t_start: float) -> None:
    """Log elapsed wall time and peak memory for a completed Phase 2 stage."""
    logger.info(
        f"[stage] {stage}: elapsed={time.monotonic() - t_start:.1f}s, "
        f"maxrss={_maxrss_mb():.0f} MiB"
    )


def main(cfg: dict) -> None:
    repo_root = REPO_ROOT
    t_total = time.monotonic()

    # ── 1. Read Phase 1's per-split feature-store artifacts (no CSV reload) ────
    # Phase 2 is a consumer of Phase 1's global EID ordering. Rather than
    # re-load + re-sort the raw CSV (two independent sorts agreeing only by
    # coincidence), we read the per-split artifacts Phase 1 already wrote. The
    # raw columns Phase 2 still needs (IPs, bytes, dst_port) come from the new
    # edges_meta.parquet, aligned row-for-row with edge_indices.npy.
    t0 = time.monotonic()
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
    _log_stage("1. feature-store read", t0)

    # ── 2. Build global node map ───────────────────────────────────────────────
    t0 = time.monotonic()
    graph_dir = repo_root / cfg["graph"]["dir"]
    builder   = GraphBuilder(graph_dir=graph_dir)
    builder.build_global_node_map(src_ips_all, dst_ips_all)
    builder.save_node_id_map()

    is_internal_arr = GraphBuilder.compute_is_internal_array(builder.node_id_map)
    _log_stage("2. global node map", t0)

    # ── 3. Build and save split graphs (directly from per-split arrays) ────────
    # Edge counts are captured per-split for the final summary log; the
    # DGLGraph objects themselves are not retained beyond validate+save — for
    # large datasets (Paper-3: up to 19.5M edges) keeping all three split
    # graphs alive simultaneously is pure dead weight once each is persisted.
    t0 = time.monotonic()
    edge_counts: dict[str, int] = {}
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
        edge_counts[s] = g.num_edges()
        del g  # not needed beyond validate+save; see comment above the loop

    # per_split["val"]/["test"] are not read again below (only "train" is, for
    # baseline computation) — drop them now rather than holding full-size
    # in_bytes/out_bytes/dst_port/src_ip/dst_ip arrays alive uselessly.
    per_split.pop("val", None)
    per_split.pop("test", None)
    gc.collect()
    _log_stage("3. split graphs (train/val/test)", t0)

    # ── 4. Build NodeStateManager ──────────────────────────────────────────────
    t0 = time.monotonic()
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
    _log_stage("4a. hourly baselines", t0)

    # Snapshots from ALL edges (preserves correct node state for val/test)
    t0 = time.monotonic()
    all_src_ids = np.array(
        [builder.node_id_map[str(ip)] for ip in src_ips_all], dtype=np.int64
    )
    all_dst_ids = np.array(
        [builder.node_id_map[str(ip)] for ip in dst_ips_all], dtype=np.int64
    )
    # src_ips_all/dst_ips_all (raw IP strings, full dataset length) are only
    # needed to build build_global_node_map's input and the two node-id
    # arrays above; nothing downstream reads them again.
    del src_ips_all, dst_ips_all
    gc.collect()

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
    _log_stage("4b. node-state snapshots (Pass 1 histories + optional Pass 2)", t0)

    t0 = time.monotonic()
    snap_dir = repo_root / cfg["graph"]["node_state_dir"]
    nsm.save(snap_dir)
    _log_stage("4c. node-state save", t0)

    # ── 5. Summary ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 2 complete:")
    logger.info(f"  Graphs:     {graph_dir}/  (train/val/test.bin)")
    logger.info(f"  NodeState:  {snap_dir}/")
    logger.info(f"  Nodes:      {builder._num_nodes:,}")
    for split_name, n_edges in edge_counts.items():
        logger.info(f"  {split_name}: {n_edges:,} edges")
    logger.info(f"  Total wall time: {time.monotonic() - t_total:.1f}s")
    logger.info(f"  Peak maxrss: {_maxrss_mb():.0f} MiB")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_unsw.yaml")
    args   = parser.parse_args()
    cfg    = load_config(REPO_ROOT / args.config)
    main(cfg)
