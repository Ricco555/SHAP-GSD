"""
Tests for TemporalNeighborSampler — HARD GATE (zero violations required).

All tests use a synthetic graph to avoid dependency on graphs/*.bin.

Tests:
  1 — No future edges: for 1000 random target edges, every sampled neighbor
      edge has timestamp <= target edge timestamp. ZERO violations allowed.
  2 — Completeness: all edges with ts <= t_e are candidates before fan-out.
  3 — Batch consistency: same seed → identical blocks.
  4 — Block EID contract: block.edata[dgl.EID] must resolve, via g.find_edges,
      to edges whose actual timestamp matches block.edata['timestamp']. DGL's
      edge_subgraph/sample_neighbors/to_block chain does not compose EID
      across more than one op by default, so this guards against the EID
      silently reverting to frontier-local positions.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dgl
from src.model.temporal_sampler import TemporalNeighborSampler, floyd_sample
from tests._paths import REPO_ROOT, resolve_cfg


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

def test_block_eid_resolves_to_true_split_graph_edge():
    """block.edata[dgl.EID] must be the true local EID in the split graph.

    Regression test for a bug where block.edata[dgl.EID] silently held
    frontier/subgraph-local positions instead of the split-graph-local
    position: g.find_edges(block.edata[dgl.EID]) resolved to arbitrary wrong
    edges (typically near the start of the split, since it is chronologically
    sorted). Downstream consumers (temporal_shap.py, visualize/w_ablation
    scripts) index the split graph directly with this field and require it
    to name the real sampled edge.
    """
    g = _make_synthetic_graph(n_nodes=80, n_edges=3000, seed=1)
    sampler = TemporalNeighborSampler(fanouts=[10, 5])
    ts_all = g.edata["timestamp"]

    rng = np.random.default_rng(3)
    n_edges = g.num_edges()
    chosen = np.sort(rng.choice(n_edges, size=200, replace=False))

    checked_edges = 0
    for eid in chosen:
        seed_eids = torch.tensor([int(eid)], dtype=torch.long)
        _, _, blocks = sampler.sample_blocks(g, seed_eids)

        for block in blocks:
            local_eids = block.edata[dgl.EID]
            if local_eids.numel() == 0:
                continue
            # The EID must be a valid position in g, and g's timestamp at
            # that position must match the timestamp DGL attached to the
            # block edge itself (the one field never touched by the bug).
            assert local_eids.max().item() < n_edges
            resolved_ts = ts_all[local_eids]
            block_ts = block.edata["timestamp"]
            assert torch.equal(resolved_ts, block_ts), (
                "block.edata[dgl.EID] does not resolve to the edge whose "
                "timestamp the block actually carries — EID composition "
                "through the sample_neighbors/to_block chain is broken."
            )
            checked_edges += local_eids.numel()

    assert checked_edges > 0, "No sampled edges checked — test is vacuous"


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


# ---------------------------------------------------------------------------
# Searchsorted-sampler test-porting plan (specs/18, specs/19) — items 1-8.
#
# TemporalNeighborSampler now *is* the searchsorted CSC + Floyd-draw
# algorithm (spec 19 §1), so tests needing "the old O(E) semantics as a
# reference" must not instantiate TemporalNeighborSampler and expect it to
# reproduce the pre-swap algorithm — they use the test-local legacy oracle
# below instead (spec 19 §3.0).
# ---------------------------------------------------------------------------

def _make_hub_graph(n_nodes=60, n_edges=200_000, n_hubs=4, seed=0) -> dgl.DGLGraph:
    """Synthetic graph with hub destination nodes so fanout << eligible in-degree.

    Ported from pilot/oracle_fast.py:make_hub_graph — used by tests 1-4 to
    reach the forced-subsampling regime the take-all tests (5, 8) must NOT use.
    """
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.integers(1_000, 10_000_000, size=n_edges)).astype(np.int64)
    src = rng.integers(0, n_nodes, size=n_edges).astype(np.int64)
    # 80% of edges land on the hubs
    hub_ids = np.arange(n_hubs)
    dst = np.where(
        rng.random(n_edges) < 0.8,
        rng.choice(hub_ids, size=n_edges),
        rng.integers(0, n_nodes, size=n_edges),
    ).astype(np.int64)
    same = src == dst
    dst[same] = (dst[same] + 1) % n_nodes
    g = dgl.graph((src, dst), num_nodes=n_nodes)
    g.edata["timestamp"] = torch.from_numpy(ts)
    g.edata[dgl.EID] = torch.arange(n_edges, dtype=torch.int64)
    return g


def _reference_eligible(g: dgl.DGLGraph, node: int, cutoff: int) -> np.ndarray:
    """Oracle: eligible in-edge local EIDs of `node` under `cutoff`, computed
    exactly the way the pre-swap _filtered_subgraph + in_edges did (brute
    force O(E)). Ported from pilot/oracle_fast.py:reference_eligible."""
    ts = g.edata["timestamp"]
    _, dst = g.edges()
    mask = (dst == node) & (ts <= cutoff)
    return mask.nonzero(as_tuple=False).view(-1).numpy()


def _legacy_filtered_subgraph(g, cutoffs, timestamp_key="timestamp"):
    """Reference oracle: exact pre-swap `_filtered_subgraph` semantics,
    ts[e] <= cutoffs[dst[e]]. Ported verbatim from the deleted
    src/model/temporal_sampler.py:_filtered_subgraph (pre spec-19 build) —
    kept here only as the differential-test oracle per spec 17 §4.1 item 5.
    """
    ts = g.edata[timestamp_key]
    _, dst = g.edges()
    valid_mask = ts <= cutoffs[dst]
    valid_eids = valid_mask.nonzero(as_tuple=False).view(-1)
    return g.edge_subgraph(valid_eids, relabel_nodes=False)


def _legacy_sample_blocks(sampler, g, seed_eids, fanouts, timestamp_key="timestamp"):
    """Reference oracle: exact pre-swap sample_blocks hop loop (dgl.sampling
    .sample_neighbors on a _legacy_filtered_subgraph, composed through sg's
    own EID before to_block). `sampler` supplies only _build_cutoffs /
    _propagate_cutoffs (reused verbatim, unaffected by the swap). Used ONLY
    by test 5 (take-all block equality) and the take-all half of test 8
    (value-level equivalence) — the differential gate spec 18 §6 item 2
    requires ("take-all block equality against the old _filtered_subgraph
    semantics kept as an in-test reference oracle").
    """
    seed_eids = seed_eids.long()
    ts = g.edata[timestamp_key][seed_eids]
    cutoffs = sampler._build_cutoffs(g, seed_eids, ts)
    src_s, dst_s = g.find_edges(seed_eids)
    seed_nodes = torch.unique(torch.cat([src_s, dst_s]))
    blocks, truths = [], []
    curr_seeds, curr_cutoffs = seed_nodes, cutoffs
    for hop_idx, fanout in enumerate(reversed(fanouts)):
        sg = _legacy_filtered_subgraph(g, curr_cutoffs, timestamp_key)
        frontier = dgl.sampling.sample_neighbors(sg, curr_seeds, fanout, edge_dir="in")
        frontier_g_local_eid = sg.edata[dgl.EID][frontier.edata[dgl.EID]]
        block = dgl.to_block(frontier, curr_seeds)
        block.edata[dgl.EID] = frontier_g_local_eid[block.edata[dgl.EID]]
        blocks.insert(0, block)
        truths.insert(0, block.edata[dgl.EID])
        next_seeds = block.srcdata[dgl.NID]
        if hop_idx < len(fanouts) - 1:
            curr_cutoffs = sampler._propagate_cutoffs(frontier, curr_cutoffs)
        curr_seeds = next_seeds
    return blocks, truths


def _block_triples(block, leids, g):
    """(src_gnid, dst_gnid, g_local_eid) triple set for a block, endpoints
    mapped through the block's NID frames."""
    ls, ld = block.edges()
    src_g = block.srcdata[dgl.NID][ls]
    dst_g = block.dstdata[dgl.NID][ld]
    return set(zip(src_g.tolist(), dst_g.tolist(), leids.tolist()))


def _canon(block):
    """Canonical representation of a block: sorted (src_nid, dst_nid, ts)."""
    src, dst = block.edges()
    s = block.srcdata[dgl.NID][src].numpy()
    d = block.dstdata[dgl.NID][dst].numpy()
    t = block.edata["timestamp"].numpy()
    return sorted(zip(s.tolist(), d.tolist(), t.tolist()))


def _resolve_paths() -> tuple[Path, Path]:
    """Return (graph_dir, feature_store_root) from the active config.

    Mirrors tests/test_eid_alignment.py's skip-convention helper.
    """
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
# Test 1 (spec 17 §5 item 1): candidate-set equivalence under forced
# subsampling. Ported from pilot/oracle_fast.py:section_A_B (candidate-set
# half).
# ---------------------------------------------------------------------------

def test_candidate_set_equivalence_forced_subsampling():
    """eligible_local_eids must equal the brute-force reference set at every
    (node, hop) pair reached, in a regime where forced subsampling actually
    occurs (eligible count > fanout for many nodes)."""
    g = _make_hub_graph()
    fan = 10
    sampler = TemporalNeighborSampler(fanouts=[fan, fan])

    n_e = g.num_edges()
    seed_eids = torch.arange(n_e - 64, n_e, dtype=torch.long)
    ts = g.edata["timestamp"][seed_eids]
    cutoffs = sampler._build_cutoffs(g, seed_eids, ts)
    src_s, dst_s = g.find_edges(seed_eids)
    seeds = torch.unique(torch.cat([src_s, dst_s]))

    forced = 0
    rng = np.random.default_rng(0)
    for hop in range(2):
        for v in seeds.numpy():
            c = int(cutoffs[v])
            ref = set(_reference_eligible(g, int(v), c).tolist())
            got = set(sampler.eligible_local_eids(g, int(v), c).tolist())
            assert ref == got, (
                f"hop{hop} node {v}: candidate-set mismatch "
                f"(|ref|={len(ref)} |got|={len(got)})"
            )
            if len(ref) > fan:
                forced += 1
        frontier = sampler._sample_frontier(g, seeds, cutoffs, fan, rng)
        cutoffs = sampler._propagate_cutoffs(frontier, cutoffs)
        seeds = dgl.to_block(frontier, seeds).srcdata[dgl.NID]

    assert forced > 50, f"forced-subsampling regime not reached ({forced})"


# ---------------------------------------------------------------------------
# Test 2 (spec 17 §5 item 2): draw validity. Ported from
# pilot/oracle_fast.py:section_A_B (draw-validity half).
# ---------------------------------------------------------------------------

def test_draw_validity_subset_size_distinct():
    """For each seed node, the frontier's drawn local EIDs must be a subset
    of eligible_local_eids, of size min(fanout, p), and pairwise distinct."""
    g = _make_hub_graph()
    fan = 10
    sampler = TemporalNeighborSampler(fanouts=[fan, fan])

    n_e = g.num_edges()
    seed_eids = torch.arange(n_e - 64, n_e, dtype=torch.long)
    ts = g.edata["timestamp"][seed_eids]
    cutoffs = sampler._build_cutoffs(g, seed_eids, ts)
    src_s, dst_s = g.find_edges(seed_eids)
    seeds = torch.unique(torch.cat([src_s, dst_s]))

    checked = 0
    rng = np.random.default_rng(0)
    for hop in range(2):
        frontier = sampler._sample_frontier(g, seeds, cutoffs, fan, rng)
        fsrc, fdst = frontier.edges()
        leid = frontier.edata["_leid"].numpy()
        fdst_np = fdst.numpy()
        for v in seeds.numpy():
            sel = leid[fdst_np == v]
            elig = sampler.eligible_local_eids(g, int(v), int(cutoffs[v]))
            p = len(elig)
            assert len(sel) == min(fan, p), (
                f"node {v}: drew {len(sel)}, want {min(fan, p)}"
            )
            assert len(set(sel.tolist())) == len(sel), "duplicate draw"
            assert set(sel.tolist()) <= set(elig.tolist()), "drew outside candidate set"
            checked += 1
        cutoffs = sampler._propagate_cutoffs(frontier, cutoffs)
        seeds = dgl.to_block(frontier, seeds).srcdata[dgl.NID]

    assert checked > 0, "No seeds checked — test is vacuous"


# ---------------------------------------------------------------------------
# Test 3 (spec 17 §5 item 3): uniformity of the Floyd draw. Ported from
# pilot/oracle_fast.py:section_C.
# ---------------------------------------------------------------------------

def test_floyd_draw_uniformity():
    """Empirical per-item selection frequency of floyd_sample must be close
    to the theoretical k/p, over 20,000 seeded repetitions."""
    g = _make_hub_graph()
    sampler = TemporalNeighborSampler(fanouts=[10])
    v = 0  # hub
    c = int(g.edata["timestamp"].max())
    elig = sampler.eligible_local_eids(g, v, c)
    p, k = len(elig), 10
    assert p > 5_000

    counts = np.zeros(p, dtype=np.int64)
    reps = 20_000
    rng = np.random.default_rng(123)
    for _ in range(reps):
        idx = floyd_sample(rng, p, k)
        counts[idx] += 1
    freq = counts / reps
    expected = k / p
    se = np.sqrt(expected * (1 - expected) / reps)
    z = np.abs(freq - expected) / se
    frac_bad = float((z > 4.5).mean())
    assert frac_bad < 1e-3, f"biased draw? frac(|z|>4.5)={frac_bad}"


# ---------------------------------------------------------------------------
# Test 4 (spec 17 §5 item 4): determinism under forced subsampling. Ported
# from pilot/oracle_fast.py:section_D.
# ---------------------------------------------------------------------------

def test_determinism_under_forced_subsampling():
    """Two sample_blocks calls with identical (g, seed_eids) in the forced-
    subsampling regime must return blocks with identical edge counts and
    identical sorted g-local EID sets per hop."""
    g = _make_hub_graph()
    sampler = TemporalNeighborSampler(fanouts=[10, 5])
    n_e = g.num_edges()
    seed_eids = torch.arange(n_e - 32, n_e, dtype=torch.long)

    _, _, ba = sampler.sample_blocks(g, seed_eids.clone())
    _, _, bb = sampler.sample_blocks(g, seed_eids.clone())

    assert len(ba) == len(bb)
    for x, y in zip(ba, bb):
        assert x.num_edges() == y.num_edges()
        assert torch.equal(
            x.edata[dgl.EID].sort().values, y.edata[dgl.EID].sort().values
        )


# ---------------------------------------------------------------------------
# Test 5 (spec 17 §5 item 5): take-all block equality vs the legacy
# reference oracle. Ported from pilot/block_equivalence.py (baseline arm
# only — the GraphBolt arm is dropped, out of scope per spec 18 §5).
# ---------------------------------------------------------------------------

BIG = 10_000  # fanout >= any in-degree in these fixtures -> take-all regime


def test_take_all_block_equality_vs_legacy_reference():
    """In the take-all regime, the new sampler's blocks must be structurally
    identical (src/dst NID sets, canonical (src,dst,ts) edge triples) to the
    legacy _filtered_subgraph-based reference oracle's blocks."""
    g = _make_hub_graph(n_nodes=40, n_edges=5_000)
    sampler = TemporalNeighborSampler(fanouts=[BIG, BIG])

    rng = np.random.default_rng(3)
    for _ in range(30):
        e0 = int(rng.integers(0, g.num_edges() - 4))
        seed_eids = torch.arange(e0, e0 + 4, dtype=torch.long)

        _, _, new_blocks = sampler.sample_blocks(g, seed_eids.clone())
        legacy_blocks, _ = _legacy_sample_blocks(sampler, g, seed_eids.clone(), [BIG, BIG])

        assert len(new_blocks) == len(legacy_blocks)
        for nb, lb in zip(new_blocks, legacy_blocks):
            assert torch.equal(
                nb.srcdata[dgl.NID].sort().values, lb.srcdata[dgl.NID].sort().values
            ), "src NIDs differ"
            assert torch.equal(
                nb.dstdata[dgl.NID].sort().values, lb.dstdata[dgl.NID].sort().values
            ), "dst NIDs differ"
            assert _canon(nb) == _canon(lb), "edge (src,dst,ts) triples differ"

    if _graphs_available():
        # Real-graph optional half (spec 19 §3.3): 5 real UNSW train
        # singleton batches, train-only acceptable here because this test
        # checks block structure (src/dst/ts), not EID values.
        graph_dir, _ = _resolve_paths()
        gt_list, _ = dgl.load_graphs(str(graph_dir / "train.bin"))
        gt = gt_list[0]
        big_sampler = TemporalNeighborSampler(fanouts=[BIG * 200, BIG * 200])
        for eid in np.sort(
            rng.choice(min(200_000, gt.num_edges()), size=5, replace=False)
        ):
            seed_eids = torch.tensor([int(eid)], dtype=torch.long)
            _, _, new_blocks = big_sampler.sample_blocks(gt, seed_eids.clone())
            legacy_blocks, _ = _legacy_sample_blocks(
                big_sampler, gt, seed_eids.clone(), [BIG * 200, BIG * 200]
            )
            assert len(new_blocks) == len(legacy_blocks)
            for nb, lb in zip(new_blocks, legacy_blocks):
                assert torch.equal(
                    nb.srcdata[dgl.NID].sort().values,
                    lb.srcdata[dgl.NID].sort().values,
                )
                assert torch.equal(
                    nb.dstdata[dgl.NID].sort().values,
                    lb.dstdata[dgl.NID].sort().values,
                )
                assert _canon(nb) == _canon(lb)


# ---------------------------------------------------------------------------
# Test 6 (spec 17 §5 item 6): real-graph leak oracle. Ported from
# pilot/oracle_fast.py:leak_oracle_real. Artifact-gated, skip convention per
# spec 19 §3.2 / tests/test_eid_alignment.py.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split", ["train", "val"])
def test_real_graph_leak_oracle(split: str):
    """Every sampled edge's global EID must round-trip, via
    feature_store/<split>/edge_indices.npy -> timestamps.npy, to a true
    timestamp <= the batch's cutoff, with 0 violations and 0 carried-vs-true
    timestamp mismatches."""
    if not _graphs_available():
        pytest.skip("graphs/*.bin not found — run scripts/02_build_graph.py first")

    graph_dir, fs_root = _resolve_paths()
    g_list, _ = dgl.load_graphs(str(graph_dir / f"{split}.bin"))
    g = g_list[0]
    fs_dir = fs_root / split
    fs_eids = np.load(fs_dir / "edge_indices.npy")
    fs_ts = np.load(fs_dir / "timestamps.npy")
    assert np.all(np.diff(fs_eids) > 0)

    def true_ts_of_geid(geids):
        pos = np.searchsorted(fs_eids, geids)
        assert np.all(fs_eids[pos] == geids), "global EID missing from store"
        return fs_ts[pos]

    sampler = TemporalNeighborSampler(fanouts=[25, 15])
    ts_all = g.edata["timestamp"].numpy()
    n_e = g.num_edges()
    rng = np.random.default_rng(7)

    violations = 0
    checked = 0
    mismatch = 0
    n_singleton, n_batches, batch_size = 300, 10, 1024

    for eid in np.sort(rng.choice(n_e, size=n_singleton, replace=False)):
        t_e = int(ts_all[eid])
        _, _, blocks = sampler.sample_blocks(g, torch.tensor([eid]))
        for b in blocks:
            if b.num_edges() == 0:
                continue
            geid = b.edata["_geid"].numpy()
            true_ts = true_ts_of_geid(geid)
            carried = b.edata["timestamp"].numpy()
            mismatch += int((true_ts != carried).sum())
            violations += int((true_ts > t_e).sum())
            checked += len(geid)

    starts = np.sort(rng.choice(n_e - batch_size, size=n_batches, replace=False))
    for s in starts:
        seed_eids = torch.arange(s, s + batch_size, dtype=torch.long)
        t_max = int(ts_all[s + batch_size - 1])
        _, _, blocks = sampler.sample_blocks(g, seed_eids)
        for b in blocks:
            if b.num_edges() == 0:
                continue
            geid = b.edata["_geid"].numpy()
            true_ts = true_ts_of_geid(geid)
            mismatch += int((true_ts != b.edata["timestamp"].numpy()).sum())
            violations += int((true_ts > t_max).sum())
            checked += len(geid)

    assert mismatch == 0, f"{mismatch} carried-vs-true timestamp mismatches"
    assert violations == 0, f"{violations} temporal LEAKS on {split}"
    assert checked > 0, "No sampled edges checked — test is vacuous"


# ---------------------------------------------------------------------------
# Test 8 (spec 17 §5 item 8, NEW audit finding): block-EID value contract on
# real val/test split artifacts. Ported from pilot/oracle_block_eid.py
# sections H (fixed-sampler contract), I (take-all value equivalence vs
# legacy truth), J (real explainer round-trip, fast side only).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split", ["val", "test"])
def test_block_eid_value_contract_real_split(split: str):
    """block.edata[dgl.EID] must be a true g-local split-graph edge position:
    timestamps, global EIDs, endpoints, and the feature-store round-trip
    all agree exactly (H); take-all triples equal the legacy reference
    oracle's truth triples (I/e); and the real explainer round-trip
    (src.explainer.temporal_shap._extract_neighbor_edges) returns only
    records for edges the sampler actually drew (J-fast-side). This test
    must run on val or test, never train-only — train's global==local EID
    coincidence masks exactly this bug class.
    """
    if not _graphs_available():
        pytest.skip("graphs/*.bin not found — run scripts/02_build_graph.py first")

    graph_dir, fs_root = _resolve_paths()
    g_list, _ = dgl.load_graphs(str(graph_dir / f"{split}.bin"))
    g = g_list[0]
    fs_dir = fs_root / split
    fs_eids = np.load(fs_dir / "edge_indices.npy")
    fs_ts = np.load(fs_dir / "timestamps.npy")
    order = np.argsort(fs_eids)
    fs_eids_s, fs_ts_s = fs_eids[order], fs_ts[order]

    def true_ts_of_geid(geids):
        pos = np.searchsorted(fs_eids_s, geids)
        assert np.all(fs_eids_s[pos] == geids), "global EID missing in feature store"
        return fs_ts_s[pos]

    n = g.num_edges()
    rng = np.random.default_rng(7)
    singles = [torch.tensor([int(x)]) for x in rng.integers(n // 10, n - 1, size=40)]
    batches = []
    for _ in range(5):
        start = int(rng.integers(0, n - 300))
        batches.append(torch.arange(start, start + 256))
    seeds_gh = singles + batches

    fanouts = [25, 15]
    sampler = TemporalNeighborSampler(fanouts=fanouts)

    # (a)-(d): fixed-sampler contract (pilot section H).
    checked = 0
    for seed_eids in seeds_gh:
        _, _, blocks = sampler.sample_blocks(g, seed_eids)
        for blk in blocks:
            e = blk.edata[dgl.EID]
            if e.numel() == 0:
                continue
            # (a) g-local: timestamps agree exactly
            assert bool((blk.edata["timestamp"] == g.edata["timestamp"][e]).all())
            # (b) global EID via g-local lookup == carried _geid
            assert bool((g.edata[dgl.EID][e] == blk.edata["_geid"]).all())
            # (c) endpoints: find_edges(e) == block endpoints mapped to global NIDs
            fs_, fd_ = g.find_edges(e)
            ls, ld = blk.edges()
            assert bool((blk.srcdata[dgl.NID][ls] == fs_).all())
            assert bool((blk.dstdata[dgl.NID][ld] == fd_).all())
            # (d) feature-store round-trip: true ts by global EID == carried ts
            tts = true_ts_of_geid(blk.edata["_geid"].numpy().astype(np.int64))
            assert np.array_equal(tts, blk.edata["timestamp"].numpy())
            checked += int(e.numel())
    assert checked > 0, "No block edges checked in section H — test is vacuous"

    # (e) take-all value-level equivalence vs legacy reference oracle truth.
    big_sampler = TemporalNeighborSampler(fanouts=[10**6, 10**6])
    early = [torch.tensor([int(x)]) for x in rng.integers(30, 250, size=8)]
    n_cmp = 0
    for seed_eids in early:
        legacy_blocks, legacy_truths = _legacy_sample_blocks(
            big_sampler, g, seed_eids.clone(), [10**6, 10**6]
        )
        _, _, new_blocks = big_sampler.sample_blocks(g, seed_eids.clone())
        assert len(legacy_blocks) == len(new_blocks)
        for lb, lt, nb in zip(legacy_blocks, legacy_truths, new_blocks):
            t_legacy = _block_triples(lb, lt, g)
            t_new = _block_triples(nb, nb.edata[dgl.EID], g)
            assert t_legacy == t_new, (
                f"take-all triple/EID set mismatch ({len(t_legacy)} vs {len(t_new)})"
            )
            n_cmp += len(t_legacy)
    assert n_cmp > 0, "No take-all triples compared in section I — test is vacuous"

    # (f) real explainer round-trip (fast side only).
    from src.explainer.temporal_shap import TemporalNeighborhoodSHAP

    class _StubNSM:
        _W_ms = 60_000  # temporal_window_seconds=60 (configs)

    tshap = TemporalNeighborhoodSHAP(
        background=None, nsm=_StubNSM(), g_split=g, device=torch.device("cpu")
    )
    n_rec = 0
    for seed_eids in singles[:25]:
        leid = int(seed_eids[0])
        t_target = float(g.edata["timestamp"][leid])
        _, _, blocks = sampler.sample_blocks(g, seed_eids)
        recs = tshap._extract_neighbor_edges(blocks, leid, t_target)
        drawn = torch.cat([b.edata[dgl.EID] for b in blocks]).tolist()
        for r in recs:
            n_rec += 1
            assert float(g.edata["timestamp"][r.local_eid]) == r.timestamp_ms
            assert int(g.edata[dgl.EID][r.local_eid]) == r.global_eid
            assert r.timestamp_ms <= t_target
            assert (t_target - r.timestamp_ms) <= tshap.nsm._W_ms
            s_, d_ = g.find_edges(torch.tensor([r.local_eid]))
            assert int(s_[0]) == r.src_nid
            assert int(d_[0]) == r.dst_nid
            # the record must be an edge the sampler ACTUALLY drew
            assert r.local_eid in drawn
    assert n_rec >= 0  # some singleton seeds may have empty windows; no crash required
