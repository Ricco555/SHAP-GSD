"""Dry-run diagnostic: malicious-subgraph component sizes vs. proposed diameter caps.

Standalone, read-only research script — NOT part of the numbered pipeline
(01-13) or the phase-14 diagnostic itself. It exists to gather real evidence
for `specs/10_phase14_diameter_optimization.md` (DRAFT, unauthorized) before
that fix is built: does the malicious-only training subgraph on each Paper-3
dataset actually blow past the proposed caps (2000 nodes / 5000 edges per
connected component), and by how much?

It NEVER calls `nx.diameter` / `usebounds` / any O(V*E) algorithm — only
O(V+E) connected-component sizing via `scipy.sparse.csgraph`. This is the
whole point: get the size distribution cheaply, without risking the hang
the diameter fix is meant to prevent.

Reads each raw CSV directly (only 4 of 54 columns) and replicates the
project's real train split (`configs/default.yaml: data.train_frac`,
chronological on FLOW_START_MILLISECONDS) and malicious filter (Label == 1).
Does not depend on Phase 1/2 pipeline artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

USECOLS = ["FLOW_START_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label"]


def load_malicious_train_edges(csv_path: Path, train_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Read a raw NF-* CSV and return (src_ip, dst_ip) string arrays for
    malicious (Label==1) edges within the chronological train split.
    """
    t0 = time.time()
    df = pd.read_csv(csv_path, usecols=USECOLS)
    logger.info("%s: read %d rows in %.1fs", csv_path.name, len(df), time.time() - t0)

    df = df.sort_values("FLOW_START_MILLISECONDS", kind="mergesort")
    n_train = int(len(df) * train_frac)
    train_df = df.iloc[:n_train]

    mal = train_df[train_df["Label"] == 1]
    logger.info(
        "%s: train split = %d rows, malicious = %d rows (%.2f%%)",
        csv_path.name, n_train, len(mal), 100.0 * len(mal) / max(n_train, 1),
    )
    return mal["IPV4_SRC_ADDR"].to_numpy(), mal["IPV4_DST_ADDR"].to_numpy()


def component_sizes(src_ip: np.ndarray, dst_ip: np.ndarray) -> list[dict]:
    """Connected-component node/edge counts over the undirected malicious
    subgraph. Edge count per component = unique undirected (src, dst) pairs,
    matching nx.Graph's parallel-edge collapse semantics used by the shipped
    compute_diameter.
    """
    if src_ip.size == 0:
        return []

    all_ips, node_ids = np.unique(np.concatenate([src_ip, dst_ip]), return_inverse=True)
    n_nodes = all_ips.size
    src_id = node_ids[: src_ip.size]
    dst_id = node_ids[src_ip.size :]

    # Dedup parallel/reverse-duplicate edges (undirected simple graph, as nx.Graph builds it).
    lo = np.minimum(src_id, dst_id)
    hi = np.maximum(src_id, dst_id)
    edge_pairs = np.unique(np.stack([lo, hi], axis=1), axis=0)

    adj = coo_matrix(
        (np.ones(edge_pairs.shape[0], dtype=np.int8), (edge_pairs[:, 0], edge_pairs[:, 1])),
        shape=(n_nodes, n_nodes),
    )
    n_components, labels = connected_components(adj, directed=False)

    node_counts = np.bincount(labels, minlength=n_components)
    edge_component = labels[edge_pairs[:, 0]]  # both endpoints share a component id
    edge_counts = np.bincount(edge_component, minlength=n_components)

    return [
        {"n_nodes": int(node_counts[c]), "n_edges": int(edge_counts[c])}
        for c in range(n_components)
        if node_counts[c] > 0
    ]


def summarize(components: list[dict], node_cap: int, edge_cap: int) -> dict:
    if not components:
        return {"n_components": 0}
    nodes = [c["n_nodes"] for c in components]
    edges = [c["n_edges"] for c in components]
    over_cap = [c for c in components if c["n_nodes"] > node_cap or c["n_edges"] > edge_cap]
    return {
        "n_components": len(components),
        "max_nodes": max(nodes),
        "max_edges": max(edges),
        "n_components_over_cap": len(over_cap),
        "largest_over_cap": max(over_cap, key=lambda c: c["n_nodes"] * c["n_edges"]) if over_cap else None,
        "would_hit_fallback": bool(over_cap),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="*", default=None, help="CSV filenames; default = all data/*.csv")
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--node-cap", type=int, default=2000)
    parser.add_argument("--edge-cap", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=Path("outputs/dryrun_diameter_caps.json"))
    args = parser.parse_args()

    csv_paths = (
        [args.data_dir / name for name in args.datasets]
        if args.datasets
        else sorted(args.data_dir.glob("*.csv"))
    )

    results = {}
    for csv_path in csv_paths:
        if not csv_path.exists():
            logger.warning("skip missing file: %s", csv_path)
            continue
        t0 = time.time()
        src_ip, dst_ip = load_malicious_train_edges(csv_path, args.train_frac)
        comps = component_sizes(src_ip, dst_ip)
        summary = summarize(comps, args.node_cap, args.edge_cap)
        summary["elapsed_s"] = round(time.time() - t0, 1)
        results[csv_path.name] = summary
        logger.info("%s: %s", csv_path.name, json.dumps(summary, default=str))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=str))
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
