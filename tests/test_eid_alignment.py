"""
EID alignment test — requires graphs/*.bin AND feature_store/*.

For 1000 random edges per split:
  eid = g.edata[dgl.EID][local_idx]
  feature_from_store = feature_store[eid]
  timestamps must also match between graph and feature store

Skipped automatically if graphs/*.bin do not exist yet
(run scripts/02_build_graph.py first).

Graph and feature-store paths are resolved from the config selected by the
``SHAP_GSD_CONFIG`` environment variable (default: configs/experiment_unsw.yaml),
so the tests automatically respect ``run.dir`` for multi-dataset runs.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tests._paths import REPO_ROOT, resolve_cfg


def _resolve_paths() -> tuple[Path, Path]:
    """Return (graph_dir, feature_store_root) from the active config."""
    cfg = resolve_cfg()
    graph_dir = REPO_ROOT / cfg["graph"]["dir"]
    fs_root = REPO_ROOT / cfg["output"]["feature_store_dir"]
    return graph_dir, fs_root


def _graphs_available() -> bool:
    graph_dir, _ = _resolve_paths()
    return (
        (graph_dir / "train.bin").exists()
        and (graph_dir / "val.bin").exists()
        and (graph_dir / "test.bin").exists()
    )


# ---------------------------------------------------------------------------
# Parametrized over splits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_eid_alignment(split: str) -> None:
    """g.edata[dgl.EID][i] must equal feature_store/edge_indices.npy[i] for all i."""
    if not _graphs_available():
        pytest.skip("graphs/*.bin not found — run scripts/02_build_graph.py first")

    import dgl
    from src.data.feature_store import FeatureStore

    graph_dir, fs_root = _resolve_paths()

    g_list, _ = dgl.load_graphs(str(graph_dir / f"{split}.bin"))
    g = g_list[0]

    fs = FeatureStore(fs_root / split)

    rng = np.random.default_rng(42)
    n_edges = g.num_edges()
    n_sample = min(1000, n_edges)
    local_indices = rng.choice(n_edges, size=n_sample, replace=False)

    graph_eids = g.edata[dgl.EID].numpy()[local_indices]
    store_eids = fs.edge_indices[local_indices]

    mismatches = int((graph_eids != store_eids).sum())
    assert mismatches == 0, (
        f"{split}: {mismatches}/{n_sample} EID mismatches between graph and feature store"
    )


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_timestamp_alignment(split: str) -> None:
    """Graph edge timestamps must match feature_store/timestamps.npy."""
    if not _graphs_available():
        pytest.skip("graphs/*.bin not found — run scripts/02_build_graph.py first")

    import dgl
    from src.data.feature_store import FeatureStore

    graph_dir, fs_root = _resolve_paths()

    g_list, _ = dgl.load_graphs(str(graph_dir / f"{split}.bin"))
    g = g_list[0]

    fs = FeatureStore(fs_root / split)

    rng = np.random.default_rng(99)
    n_edges = g.num_edges()
    n_sample = min(1000, n_edges)
    local_indices = rng.choice(n_edges, size=n_sample, replace=False)

    graph_ts = g.edata["timestamp"].numpy()[local_indices]
    store_ts = fs.timestamps[local_indices]

    mismatches = int((graph_ts != store_ts).sum())
    assert mismatches == 0, (
        f"{split}: {mismatches}/{n_sample} timestamp mismatches "
        f"between graph and feature store"
    )


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_edges_meta_alignment(split: str) -> None:
    """edges_meta.parquet must be row-aligned with edge_indices.npy and the graph.

    The Phase-1 ``edges_meta.parquet`` artifact carries the raw columns (IPs,
    bytes, dst_port) that Phase 2 reads instead of re-loading the CSV. Its row
    ``i`` must correspond to ``edge_indices[i]`` (and hence graph edge ``i``) by
    construction. This test proves the alignment on the real regenerated
    artifacts, not just synthetic round-trips (see tests/test_edges_meta.py):

    1. Length agreement: ``len(edges_meta) == len(edge_indices) == g.num_edges()``.
    2. Row-for-row IP agreement: mapping graph edge ``i``'s source/destination
       node ID back to an IP string via ``node_id_map`` must equal
       ``edges_meta["src_ip"/"dst_ip"][i]``. This is the load-bearing check —
       length equality alone does not prove positional alignment.
    """
    if not _graphs_available():
        pytest.skip("graphs/*.bin not found — run scripts/02_build_graph.py first")

    pytest.importorskip("pyarrow", reason="pyarrow required for edges_meta.parquet")

    import dgl

    from src.data.feature_store import read_edges_meta

    graph_dir, fs_root = _resolve_paths()

    meta_path = fs_root / split / "edges_meta.parquet"
    if not meta_path.exists():
        pytest.skip(
            f"{meta_path} not found — feature store predates edges_meta "
            "(re-run scripts/01_preprocess.py)"
        )

    meta = read_edges_meta(fs_root / split)
    store_eids = np.load(fs_root / split / "edge_indices.npy")

    g_list, _ = dgl.load_graphs(str(graph_dir / f"{split}.bin"))
    g = g_list[0]

    n = g.num_edges()
    # (1) length agreement across all three artifacts
    assert len(meta["src_ip"]) == len(store_eids) == n, (
        f"{split}: length mismatch — edges_meta={len(meta['src_ip']):,}, "
        f"edge_indices={len(store_eids):,}, graph_edges={n:,}"
    )
    for col in ("dst_ip", "in_bytes", "out_bytes", "dst_port"):
        assert len(meta[col]) == n, f"{split}: edges_meta['{col}'] length != {n:,}"

    # (2) row-for-row IP agreement via node_id_map (proves positional alignment)
    with open(graph_dir / "node_id_map.json") as f:
        node_id_map = json.load(f)
    id_to_ip = {int(v): k for k, v in node_id_map.items()}

    src_nodes, dst_nodes = g.edges(order="eid")
    src_nodes = src_nodes.numpy()
    dst_nodes = dst_nodes.numpy()

    rng = np.random.default_rng(2024)
    n_sample = min(1000, n)
    idx = rng.choice(n, size=n_sample, replace=False)

    src_ip_from_graph = np.array([id_to_ip[int(src_nodes[i])] for i in idx], dtype=object)
    dst_ip_from_graph = np.array([id_to_ip[int(dst_nodes[i])] for i in idx], dtype=object)

    src_mismatch = int((src_ip_from_graph != meta["src_ip"][idx]).sum())
    dst_mismatch = int((dst_ip_from_graph != meta["dst_ip"][idx]).sum())
    assert src_mismatch == 0, (
        f"{split}: {src_mismatch}/{n_sample} src_ip mismatches between graph edges "
        f"and edges_meta — rows are not aligned with edge_indices"
    )
    assert dst_mismatch == 0, (
        f"{split}: {dst_mismatch}/{n_sample} dst_ip mismatches between graph edges "
        f"and edges_meta — rows are not aligned with edge_indices"
    )


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_global_node_count_consistent(split: str) -> None:
    """All split graphs must have the same num_nodes (global IP count)."""
    if not _graphs_available():
        pytest.skip("graphs/*.bin not found — run scripts/02_build_graph.py first")

    import dgl

    graph_dir, _ = _resolve_paths()

    node_counts: dict[str, int] = {}
    for s in ("train", "val", "test"):
        g_list, _ = dgl.load_graphs(str(graph_dir / f"{s}.bin"))
        node_counts[s] = g_list[0].num_nodes()

    assert node_counts["train"] == node_counts["val"] == node_counts["test"], (
        f"Node count mismatch across splits: {node_counts}"
    )
