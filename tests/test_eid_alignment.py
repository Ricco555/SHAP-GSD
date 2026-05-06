"""
EID alignment test — requires graphs/*.bin AND feature_store/*.

For 1000 random edges per split:
  eid = g.edata[dgl.EID][local_idx]
  feature_from_store = feature_store[eid]
  timestamps must also match between graph and feature store

Skipped automatically if graphs/*.bin do not exist yet
(run scripts/02_build_graph.py first).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = REPO_ROOT / "graphs"


def _graphs_available() -> bool:
    return (
        (GRAPH_DIR / "train.bin").exists()
        and (GRAPH_DIR / "val.bin").exists()
        and (GRAPH_DIR / "test.bin").exists()
    )


# ---------------------------------------------------------------------------
# Parametrized over splits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_eid_alignment(split: str):
    """g.edata[dgl.EID][i] must equal feature_store/edge_indices.npy[i] for all i."""
    if not _graphs_available():
        pytest.skip("graphs/*.bin not found — run scripts/02_build_graph.py first")

    import dgl
    from src.data.feature_store import FeatureStore

    g_list, _ = dgl.load_graphs(str(GRAPH_DIR / f"{split}.bin"))
    g = g_list[0]

    fs = FeatureStore(REPO_ROOT / "feature_store" / split)

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
def test_timestamp_alignment(split: str):
    """Graph edge timestamps must match feature_store/timestamps.npy."""
    if not _graphs_available():
        pytest.skip("graphs/*.bin not found — run scripts/02_build_graph.py first")

    import dgl
    from src.data.feature_store import FeatureStore

    g_list, _ = dgl.load_graphs(str(GRAPH_DIR / f"{split}.bin"))
    g = g_list[0]

    fs = FeatureStore(REPO_ROOT / "feature_store" / split)

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
def test_global_node_count_consistent(split: str):
    """All split graphs must have the same num_nodes (global IP count)."""
    if not _graphs_available():
        pytest.skip("graphs/*.bin not found — run scripts/02_build_graph.py first")

    import dgl

    node_counts = {}
    for s in ("train", "val", "test"):
        g_list, _ = dgl.load_graphs(str(GRAPH_DIR / f"{s}.bin"))
        node_counts[s] = g_list[0].num_nodes()

    assert node_counts["train"] == node_counts["val"] == node_counts["test"], (
        f"Node count mismatch across splits: {node_counts}"
    )
