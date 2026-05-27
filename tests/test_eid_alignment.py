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
