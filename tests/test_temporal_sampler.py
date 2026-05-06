"""
Tests for TemporalNeighborSampler — HARD GATE (zero violations required).

All tests use a synthetic graph to avoid dependency on graphs/*.bin.

Tests:
  1 — No future edges: for 1000 random target edges, every sampled neighbor
      edge has timestamp <= target edge timestamp. ZERO violations allowed.
  2 — Completeness: all edges with ts <= t_e are candidates before fan-out.
  3 — Batch consistency: same seed → identical blocks.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dgl
from src.model.temporal_sampler import TemporalNeighborSampler


# ---------------------------------------------------------------------------
# Synthetic graph fixture
# ---------------------------------------------------------------------------

def _make_synthetic_graph(n_nodes: int = 50, n_edges: int = 2000,
                           seed: int = 0) -> dgl.DGLGraph:
    """Build a directed graph with monotonically increasing timestamps.

    Edges are added in chronological order so that
    g.edata['timestamp'][i] <= g.edata['timestamp'][j] for i < j.
    This satisfies the shuffle=False precondition.
    """
    rng = np.random.default_rng(seed)

    # Chronological timestamps
    ts = np.sort(rng.integers(1000, 1_000_000, size=n_edges)).astype(np.int64)

    src = rng.integers(0, n_nodes, size=n_edges).astype(np.int64)
    dst = rng.integers(0, n_nodes, size=n_edges).astype(np.int64)
    # Avoid self-loops so neighbor sampling is non-trivial
    same = src == dst
    dst[same] = (dst[same] + 1) % n_nodes

    g = dgl.graph((src, dst), num_nodes=n_nodes)
    g.edata["timestamp"] = torch.tensor(ts, dtype=torch.int64)
    return g


# ---------------------------------------------------------------------------
# Test 1: No future edges — zero violations
# ---------------------------------------------------------------------------

def test_no_future_edges():
    """Every sampled neighbor edge timestamp must be <= target edge timestamp."""
    rng = np.random.default_rng(42)
    g = _make_synthetic_graph(n_nodes=80, n_edges=3000, seed=0)

    sampler = TemporalNeighborSampler(fanouts=[10, 5])

    ts_all = g.edata["timestamp"].numpy()
    n_edges = g.num_edges()

    # Sample 1000 individual target edges as singleton batches so each edge
    # gets its own exact cutoff (avoids batch-max approximation).
    n_trials = min(1000, n_edges)
    chosen = rng.choice(n_edges, size=n_trials, replace=False)
    chosen.sort()  # preserve non-decreasing order

    violations = 0
    for eid in chosen:
        seed_eids = torch.tensor([eid], dtype=torch.long)
        t_e = int(ts_all[eid])

        _, _, blocks = sampler.sample_blocks(g, seed_eids)

        for block in blocks:
            if block.num_edges() == 0:
                continue
            block_ts = block.edata["timestamp"].numpy()
            v = int((block_ts > t_e).sum())
            violations += v

    assert violations == 0, (
        f"TemporalNeighborSampler: {violations} future-edge violations "
        f"across {n_trials} target edges. ZERO allowed."
    )


# ---------------------------------------------------------------------------
# Test 2: Completeness — all eligible edges are candidates before fan-out
# ---------------------------------------------------------------------------

def test_completeness():
    """All edges with ts <= t_e must be eligible candidates for a given node."""
    # Build a simple star graph: node 0 is the hub with many outgoing edges,
    # and one leaf node with known incoming history.
    n_nodes = 10
    # Hub sends edges to leaf 1 at times [100, 200, 300, 400, 500]
    hub, leaf = 0, 1
    hub_ts   = [100, 200, 300, 400, 500]
    other_ts = [600, 700]
    src = torch.tensor([hub]*5 + [2, 3], dtype=torch.long)
    dst = torch.tensor([leaf]*5 + [leaf, leaf], dtype=torch.long)
    ts  = torch.tensor(hub_ts + other_ts, dtype=torch.int64)

    g = dgl.graph((src, dst), num_nodes=n_nodes)
    g.edata["timestamp"] = ts

    # Target: edge from hub→leaf at t=500 (EID=4, the last hub edge)
    sampler = TemporalNeighborSampler(fanouts=[10, 10])  # large enough to not subsample

    seed_eids = torch.tensor([4], dtype=torch.long)  # hub→leaf at t=500
    _, _, blocks = sampler.sample_blocks(g, seed_eids)

    # In the 1-hop block, all in-edges to leaf with ts <= 500 should be present.
    # Those are the 5 hub→leaf edges (ts=100..500).
    # Collect edges from the 1-hop block (blocks[-1]):
    block_last = blocks[-1]
    if block_last.num_edges() > 0:
        eligible_ts = [100, 200, 300, 400, 500]
        sampled_ts = sorted(block_last.edata["timestamp"].tolist())
        # Every sampled timestamp must be <= 500
        for t in sampled_ts:
            assert t <= 500, f"Future edge t={t} sampled for target at t=500"
        # With fanout=10 (>= 5 available), all 5 should be sampled
        assert len(sampled_ts) == 5, (
            f"Expected 5 eligible edges, got {len(sampled_ts)}: {sampled_ts}"
        )
        for t in eligible_ts:
            assert t in sampled_ts, (
                f"Eligible edge t={t} missing from sampled set {sampled_ts}"
            )


# ---------------------------------------------------------------------------
# Test 3: Batch consistency — same seed → identical blocks
# ---------------------------------------------------------------------------

def test_batch_consistency():
    """Identical seed and same graph must yield identical sampled blocks."""
    g = _make_synthetic_graph(n_nodes=50, n_edges=1000, seed=7)
    sampler = TemporalNeighborSampler(fanouts=[5, 3])

    seed_eids = torch.tensor([10, 11, 12], dtype=torch.long)

    _, _, blocks_a = sampler.sample_blocks(g, seed_eids.clone())
    _, _, blocks_b = sampler.sample_blocks(g, seed_eids.clone())

    assert len(blocks_a) == len(blocks_b)
    for ba, bb in zip(blocks_a, blocks_b):
        assert ba.num_edges() == bb.num_edges(), "Block edge counts differ"
        # Edge timestamps should be identical
        if ba.num_edges() > 0:
            ta = ba.edata["timestamp"].sort().values
            tb = bb.edata["timestamp"].sort().values
            assert torch.equal(ta, tb), "Sampled timestamps differ between runs"
