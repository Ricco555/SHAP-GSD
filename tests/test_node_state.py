"""
Tests for NodeStateManager (15-dim temporal node state).

All tests use a synthetic NodeStateManager built from known edges so
exact expected values can be computed analytically.

Tests:
  1 — Known state:     verify all 15 dims for a node with 5 known edges.
  2 — Rollback latest: removing the most recent edge updates recency correctly.
  3 — Rollback first:  removing the first_seen edge sets novelty=0.
  4 — Multi-rollback:  three successive single-edge rollbacks match expectations.
  5 — Seasonal:        time_sin/cos computed from known timestamp; volume_deviation=0
                       when current volume equals hourly baseline.
  6 — Snapshots disabled by default: build_snapshots() with snapshot_interval=0/None
                       (the new default) leaves _snap_times/_snap_states empty, and
                       an empty snapshots.pkl round-trips through save()/load().
  7 — Snapshots still work when explicitly enabled: same edges, interval>0,
                       produces non-empty snapshots exactly as before this change.
  8 — Query-path parity: get_state_at_time/rollback_edge results are identical
                       whether snapshot generation is enabled or disabled — proving
                       snapshots were always dead weight for the real query path.
"""

import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import entropy as scipy_entropy
from src.model.node_state import (
    NOVELTY_MODE_RECENT_WINDOW,
    NOVELTY_MODE_UNSEEN_IN_TRAINING,
    NodeStateManager,
)
from tests._paths import REPO_ROOT, resolve_cfg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_W_MS = 1_000.0   # 1-second window for test clarity


def _build_nsm(edges: list[dict], W_ms: float = _W_MS,
               snapshot_interval: int = 5,
               novelty_mode: str = NOVELTY_MODE_RECENT_WINDOW,
               ) -> NodeStateManager:
    """Build a NodeStateManager from a list of edge dicts.

    Each dict: {'ts': int, 'src': int, 'dst': int, 'in_bytes': float,
                'out_bytes': float, 'dst_port': int}

    The training-node set is established UNCONDITIONALLY (in both modes) from
    the same first-half edge slice used for the hourly baselines, and in the
    same position relative to build_snapshots as scripts/02_build_graph.py
    (specs/60 §3, §6.2) — so this helper exercises the production ordering.

    Args:
        edges:             edge dicts, ascending by 'ts'.
        W_ms:              rolling window width in milliseconds.
        snapshot_interval: Pass-2 snapshot interval (0/None disables Pass 2).
        novelty_mode:      dim-1 semantics; see src.model.node_state.

    Returns:
        A ready-to-query NodeStateManager.
    """
    nsm = NodeStateManager(window_seconds=W_ms / 1000.0,
                           snapshot_interval=snapshot_interval,
                           novelty_mode=novelty_mode)

    n = len(edges)
    src  = np.array([e["src"]       for e in edges], dtype=np.int64)
    dst  = np.array([e["dst"]       for e in edges], dtype=np.int64)
    ts   = np.array([e["ts"]        for e in edges], dtype=np.int64)
    ib   = np.array([e["in_bytes"]  for e in edges], dtype=np.float32)
    ob   = np.array([e["out_bytes"] for e in edges], dtype=np.float32)
    dp   = np.array([e["dst_port"]  for e in edges], dtype=np.int32)

    # Baselines: build from first half of edges (simulates train-only)
    half = max(1, n // 2)
    nsm.build_hourly_baselines(src[:half], dst[:half], ts[:half], ib[:half], ob[:half])
    # Same ordering rule as phase 2: AFTER build_hourly_baselines, BEFORE
    # build_snapshots (Pass 2 calls _compute_state, which reads this set).
    nsm.set_train_nodes(src[:half], dst[:half])
    nsm.build_snapshots(src, dst, ts, ib, ob, dp, snapshot_interval=snapshot_interval)

    # Minimal is_internal: all 0
    max_node = max(int(src.max()), int(dst.max())) + 1
    nsm.set_is_internal(np.zeros(max_node, dtype=np.float32))

    return nsm


# ---------------------------------------------------------------------------
# Test 1: Known state
# ---------------------------------------------------------------------------

def test_known_state():
    """Verify all 15 dims for node 0 with known edge history."""
    # Window W=1000ms, query at t=600ms → window [−400, 600]
    # Node 0 edges:
    #   t=100  outgoing to node 1  bytes=100  dst_port=80  (HTTP → bin 0)
    #   t=200  outgoing to node 2  bytes=200  dst_port=443 (HTTPS → bin 1)
    #   t=300  outgoing to node 1  bytes=150  dst_port=80  (HTTP → bin 0)
    #   t=400  incoming from node 3 bytes=500
    #   t=500  outgoing to node 2  bytes=300  dst_port=22  (SSH → bin 3)
    edges = [
        {"ts": 100, "src": 0, "dst": 1, "in_bytes": 0,   "out_bytes": 100, "dst_port": 80},
        {"ts": 200, "src": 0, "dst": 2, "in_bytes": 0,   "out_bytes": 200, "dst_port": 443},
        {"ts": 300, "src": 0, "dst": 1, "in_bytes": 0,   "out_bytes": 150, "dst_port": 80},
        {"ts": 400, "src": 3, "dst": 0, "in_bytes": 500, "out_bytes": 0,   "dst_port": 9999},
        {"ts": 500, "src": 0, "dst": 2, "in_bytes": 0,   "out_bytes": 300, "dst_port": 22},
    ]
    nsm = _build_nsm(edges, W_ms=1000.0)
    state = nsm.get_state_at_time(node_id=0, time_ms=600.0)

    assert state.shape == (15,), f"Expected shape (15,), got {state.shape}"
    assert state.dtype == np.float32

    # [0] is_internal = 0 (all nodes have is_internal=0)
    assert state[0] == 0.0

    # [1] novelty: first_seen=100ms, w_start=−400ms → 100 in [−400, 600] → 1
    assert state[1] == 1.0, f"novelty={state[1]}"

    # [2] recency: (600 − 500) / 1000 = 0.1
    assert abs(state[2] - 0.1) < 1e-5, f"recency={state[2]}"

    # [3] rolling_in_degree: 1 (edge at t=400)
    assert state[3] == 1.0

    # [4] rolling_out_degree: 4
    assert state[4] == 4.0

    # [5] unique_dst_ip_count: 2 (nodes 1 and 2)
    assert state[5] == 2.0

    # [6] unique_src_ip_count: 1 (node 3)
    assert state[6] == 1.0

    # [7] unique_dst_port_count: 3 bins used (0=HTTP, 1=HTTPS, 3=SSH)
    assert state[7] == 3.0

    # [8] dst_port_entropy: port_bins=[0,1,0,3] → counts HTTP:2, HTTPS:1, SSH:1
    expected_entropy = float(scipy_entropy([2, 1, 1]))
    assert abs(state[8] - expected_entropy) < 1e-4, (
        f"dst_port_entropy={state[8]}, expected={expected_entropy}"
    )

    # [9] rolling_in_bytes: log1p(500) ≈ 6.2146
    assert abs(state[9] - math.log1p(500)) < 1e-4, f"rolling_in_bytes={state[9]}"

    # [10] rolling_out_bytes: log1p(100+200+150+300)=log1p(750) ≈ 6.6213
    assert abs(state[10] - math.log1p(750)) < 1e-4, f"rolling_out_bytes={state[10]}"

    # [11][12] time_sin/cos: hour from t=600ms
    hour_frac = (600.0 / 3_600_000.0) % 24.0
    exp_sin = math.sin(2.0 * math.pi * hour_frac / 24.0)
    exp_cos = math.cos(2.0 * math.pi * hour_frac / 24.0)
    assert abs(state[11] - exp_sin) < 1e-5, f"time_sin={state[11]}"
    assert abs(state[12] - exp_cos) < 1e-5, f"time_cos={state[12]}"

    # [14] iat_regularity: all 5 edges in window
    all_ts = np.array([100, 200, 300, 400, 500], dtype=np.float64)
    iats = np.diff(all_ts)  # [100, 100, 100, 100]
    expected_cv = 0.0       # std=0, mean=100 → CV=0
    assert abs(state[14] - expected_cv) < 1e-5, f"iat_regularity={state[14]}"


# ---------------------------------------------------------------------------
# Test 2: Rollback most recent edge
# ---------------------------------------------------------------------------

def test_rollback_most_recent():
    """Removing the most recent edge updates recency to the next-most-recent."""
    edges = [
        {"ts": 100, "src": 0, "dst": 1, "in_bytes": 0,   "out_bytes": 100, "dst_port": 80},
        {"ts": 200, "src": 0, "dst": 2, "in_bytes": 0,   "out_bytes": 200, "dst_port": 443},
        {"ts": 300, "src": 0, "dst": 1, "in_bytes": 0,   "out_bytes": 150, "dst_port": 80},
        {"ts": 400, "src": 3, "dst": 0, "in_bytes": 500, "out_bytes": 0,   "dst_port": 9999},
        {"ts": 500, "src": 0, "dst": 2, "in_bytes": 0,   "out_bytes": 300, "dst_port": 22},
    ]
    nsm = _build_nsm(edges, W_ms=1000.0)

    state_after = nsm.rollback_edge(
        node_id=0,
        edge_timestamp_ms=500.0,
        edge_direction="outgoing",
        edge_features={"peer_id": 2},
        query_time_ms=600.0,
    )

    # After removing t=500 outgoing edge:
    # recency: (600 − 400) / 1000 = 0.2
    assert abs(state_after[2] - 0.2) < 1e-5, f"recency after rollback={state_after[2]}"

    # rolling_out_degree should drop from 4 to 3
    assert state_after[4] == 3.0, f"out_degree after rollback={state_after[4]}"

    # unique_dst_ip_count: now outgoing to nodes 1,1,1 → still {1} ... wait,
    # edges: [dst=1@100, dst=2@200, dst=1@300, dst=2@500 removed] → {1,2} still
    # Actually: t=100→1, t=200→2, t=300→1 remain → unique={1,2} → count=2
    assert state_after[5] == 2.0, f"unique_dst_ip={state_after[5]}"

    # novelty unchanged (first_seen=100ms still in window)
    assert state_after[1] == 1.0


# ---------------------------------------------------------------------------
# Test 3: Rollback first edge → novelty changes 1→0
# ---------------------------------------------------------------------------

def test_rollback_first_edge_novelty():
    """Removing the global first-seen edge sets novelty to 0.

    DEFAULT MODE ONLY (``novelty_mode="recent_window"``). Under
    ``"unseen_in_training"`` dim 1 is a property of the node rather than of its
    edges and is rollback-invariant — see
    ``test_unseen_mode_rollback_invariance`` (specs/59 D3).
    """
    # Node 0 has exactly ONE edge (at t=500ms).
    # Window W=1000ms, query at t=600ms → window=[−400, 600].
    # first_seen=500ms is within window → novelty=1.
    # After rollback of t=500ms: no edges remain → novelty=0.
    edges = [
        {"ts": 500, "src": 0, "dst": 1, "in_bytes": 0, "out_bytes": 100, "dst_port": 80},
    ]
    nsm = _build_nsm(edges, W_ms=1000.0)

    state_before = nsm.get_state_at_time(node_id=0, time_ms=600.0)
    assert state_before[1] == 1.0, f"novelty before rollback={state_before[1]}"

    state_after = nsm.rollback_edge(
        node_id=0,
        edge_timestamp_ms=500.0,
        edge_direction="outgoing",
        edge_features={"peer_id": 1},
        query_time_ms=600.0,
    )
    assert state_after[1] == 0.0, (
        f"novelty after rollback of first_seen edge={state_after[1]}, expected 0"
    )


# ---------------------------------------------------------------------------
# Test 4: Multi-edge rollback
# ---------------------------------------------------------------------------

def test_multi_rollback():
    """Three sequential single-edge rollbacks give correct cumulative state.

    NOTE: rollback_edge is non-mutating; we test three separate calls each
    masking a different edge, then verify the combined expected values would
    match a manual calculation.  We do NOT chain rollbacks (that's a Phase 6
    concern); here we only verify that each individual rollback is independent.
    """
    edges = [
        {"ts": 100, "src": 0, "dst": 1, "in_bytes": 0,   "out_bytes": 100, "dst_port": 80},
        {"ts": 200, "src": 0, "dst": 2, "in_bytes": 0,   "out_bytes": 200, "dst_port": 443},
        {"ts": 300, "src": 3, "dst": 0, "in_bytes": 500, "out_bytes": 0,   "dst_port": 9999},
    ]
    nsm = _build_nsm(edges, W_ms=1000.0)

    # Rollback edge at t=100 (outgoing to node 1)
    s1 = nsm.rollback_edge(0, 100.0, "outgoing", {"peer_id": 1}, 400.0)
    # After: out_degree=1 (only t=200 remains outgoing), in_degree=1 (t=300)
    assert s1[4] == 1.0, f"out_degree after rollback t=100: {s1[4]}"
    assert s1[3] == 1.0, f"in_degree after rollback t=100: {s1[3]}"

    # Rollback edge at t=200 (outgoing to node 2)
    s2 = nsm.rollback_edge(0, 200.0, "outgoing", {"peer_id": 2}, 400.0)
    # After: out_degree=1 (only t=100 remains outgoing)
    assert s2[4] == 1.0, f"out_degree after rollback t=200: {s2[4]}"

    # Rollback edge at t=300 (incoming from node 3)
    s3 = nsm.rollback_edge(0, 300.0, "incoming", {"peer_id": 3}, 400.0)
    # After: in_degree=0
    assert s3[3] == 0.0, f"in_degree after rollback t=300: {s3[3]}"
    # out_degree unchanged (we only rolled back an incoming edge)
    assert s3[4] == 2.0, f"out_degree after incoming rollback: {s3[4]}"


# ---------------------------------------------------------------------------
# Test 5: Seasonal features
# ---------------------------------------------------------------------------

def test_seasonal_features():
    """time_sin/cos correct; volume_deviation=0 when current matches baseline."""
    # Use t=43_200_000ms = 12 hours → hour_frac=12
    # sin(2π*12/24) = sin(π) ≈ 0,  cos(π) = −1
    T = 43_200_000
    edges = [
        {"ts": T, "src": 0, "dst": 1, "in_bytes": 100, "out_bytes": 200, "dst_port": 80},
    ]
    nsm = _build_nsm(edges, W_ms=60_000.0)  # 60s window

    state = nsm.get_state_at_time(node_id=0, time_ms=float(T))

    # time_sin at hour=12: sin(2π*12/24) = sin(π) ≈ 0
    assert abs(state[11]) < 1e-5, f"time_sin at noon={state[11]}"
    # time_cos at hour=12: cos(π) = −1
    assert abs(state[12] - (-1.0)) < 1e-5, f"time_cos at noon={state[12]}"

    # Volume deviation: the hourly baseline is built from the same edge (T is
    # in training half). rolling_bytes = log1p(200) + log1p(100) ≈ 3.499 + 4.615.
    # The baseline is the mean of the same value → deviation ≈ 0.
    # (Exact equality isn't guaranteed due to float averaging, so use atol.)
    assert abs(state[13]) < 0.5, (
        f"volume_deviation={state[13]} too large when current matches baseline"
    )


# ---------------------------------------------------------------------------
# Shared fixture edges for the snapshot on/off tests below
# ---------------------------------------------------------------------------

_SNAPSHOT_TEST_EDGES = [
    {"ts": 100, "src": 0, "dst": 1, "in_bytes": 0,   "out_bytes": 100, "dst_port": 80},
    {"ts": 200, "src": 0, "dst": 2, "in_bytes": 0,   "out_bytes": 200, "dst_port": 443},
    {"ts": 300, "src": 0, "dst": 1, "in_bytes": 0,   "out_bytes": 150, "dst_port": 80},
    {"ts": 400, "src": 3, "dst": 0, "in_bytes": 500, "out_bytes": 0,   "dst_port": 9999},
    {"ts": 500, "src": 0, "dst": 2, "in_bytes": 0,   "out_bytes": 300, "dst_port": 22},
    {"ts": 600, "src": 2, "dst": 3, "in_bytes": 50,  "out_bytes": 0,   "dst_port": 8080},
]


def _arrays_from_edges(edges: list[dict]) -> dict[str, np.ndarray]:
    return {
        "src": np.array([e["src"] for e in edges], dtype=np.int64),
        "dst": np.array([e["dst"] for e in edges], dtype=np.int64),
        "ts":  np.array([e["ts"] for e in edges], dtype=np.int64),
        "ib":  np.array([e["in_bytes"] for e in edges], dtype=np.float32),
        "ob":  np.array([e["out_bytes"] for e in edges], dtype=np.float32),
        "dp":  np.array([e["dst_port"] for e in edges], dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# Test 6: Snapshot generation disabled by default
# ---------------------------------------------------------------------------

def test_snapshots_disabled_by_default():
    """Default snapshot_interval (0) skips Pass 2: empty lists, valid empty pkl."""
    arr = _arrays_from_edges(_SNAPSHOT_TEST_EDGES)
    half = len(_SNAPSHOT_TEST_EDGES) // 2

    nsm = NodeStateManager(window_seconds=1.0)   # default snapshot_interval=0
    assert nsm.snapshot_interval == 0

    nsm.build_hourly_baselines(
        arr["src"][:half], arr["dst"][:half], arr["ts"][:half],
        arr["ib"][:half], arr["ob"][:half],
    )
    nsm.build_snapshots(arr["src"], arr["dst"], arr["ts"], arr["ib"], arr["ob"], arr["dp"])

    assert nsm._snap_times == []
    assert nsm._snap_states == []
    # Pass 1 (histories) must still have run — the real query path depends on it.
    assert len(nsm._histories) > 0


def test_snapshots_disabled_explicit_none():
    """snapshot_interval=None is also treated as disabled (YAML null)."""
    arr = _arrays_from_edges(_SNAPSHOT_TEST_EDGES)
    half = len(_SNAPSHOT_TEST_EDGES) // 2

    nsm = NodeStateManager(window_seconds=1.0, snapshot_interval=None)
    nsm.build_hourly_baselines(
        arr["src"][:half], arr["dst"][:half], arr["ts"][:half],
        arr["ib"][:half], arr["ob"][:half],
    )
    nsm.build_snapshots(arr["src"], arr["dst"], arr["ts"], arr["ib"], arr["ob"], arr["dp"])

    assert nsm._snap_times == []
    assert nsm._snap_states == []


def test_disabled_snapshots_save_load_roundtrip(tmp_path):
    """save()/load() round-trip a valid empty snapshots.pkl when disabled."""
    arr = _arrays_from_edges(_SNAPSHOT_TEST_EDGES)
    half = len(_SNAPSHOT_TEST_EDGES) // 2

    nsm = NodeStateManager(window_seconds=1.0, snapshot_interval=0)
    nsm.build_hourly_baselines(
        arr["src"][:half], arr["dst"][:half], arr["ts"][:half],
        arr["ib"][:half], arr["ob"][:half],
    )
    nsm.build_snapshots(arr["src"], arr["dst"], arr["ts"], arr["ib"], arr["ob"], arr["dp"])
    max_node = max(int(arr["src"].max()), int(arr["dst"].max())) + 1
    nsm.set_is_internal(np.zeros(max_node, dtype=np.float32))

    out_dir = tmp_path / "node_state"
    nsm.save(out_dir)

    with open(out_dir / "snapshots.pkl", "rb") as f:
        snaps = pickle.load(f)
    assert snaps == {"times": [], "states": []}

    loaded = NodeStateManager.load(out_dir)
    assert loaded._snap_times == []
    assert loaded._snap_states == []
    assert loaded.snapshot_interval == 0

    # get_state_at_time still works normally after a disabled-snapshot round-trip.
    state = loaded.get_state_at_time(node_id=0, time_ms=600.0)
    assert state.shape == (15,)


# ---------------------------------------------------------------------------
# Test 7: Snapshot generation still works when explicitly enabled
# ---------------------------------------------------------------------------

def test_snapshots_enabled_explicitly():
    """snapshot_interval>0 still builds non-empty snapshots exactly as before."""
    arr = _arrays_from_edges(_SNAPSHOT_TEST_EDGES)
    half = len(_SNAPSHOT_TEST_EDGES) // 2

    nsm = NodeStateManager(window_seconds=1.0, snapshot_interval=2)
    nsm.build_hourly_baselines(
        arr["src"][:half], arr["dst"][:half], arr["ts"][:half],
        arr["ib"][:half], arr["ob"][:half],
    )
    nsm.build_snapshots(arr["src"], arr["dst"], arr["ts"], arr["ib"], arr["ob"], arr["dp"])

    n = len(_SNAPSHOT_TEST_EDGES)
    expected_n_snaps = len(range(2 - 1, n, 2))
    assert len(nsm._snap_times) == expected_n_snaps
    assert len(nsm._snap_states) == expected_n_snaps
    assert expected_n_snaps > 0
    # Every snapshot dict maps node_id -> 15-dim vector.
    for snap in nsm._snap_states:
        for vec in snap.values():
            assert vec.shape == (15,)


def test_snapshots_enabled_save_load_roundtrip(tmp_path):
    """Enabled snapshots survive a save()/load() round-trip with real content."""
    arr = _arrays_from_edges(_SNAPSHOT_TEST_EDGES)
    half = len(_SNAPSHOT_TEST_EDGES) // 2

    nsm = NodeStateManager(window_seconds=1.0, snapshot_interval=2)
    nsm.build_hourly_baselines(
        arr["src"][:half], arr["dst"][:half], arr["ts"][:half],
        arr["ib"][:half], arr["ob"][:half],
    )
    nsm.build_snapshots(arr["src"], arr["dst"], arr["ts"], arr["ib"], arr["ob"], arr["dp"])
    max_node = max(int(arr["src"].max()), int(arr["dst"].max())) + 1
    nsm.set_is_internal(np.zeros(max_node, dtype=np.float32))

    out_dir = tmp_path / "node_state_enabled"
    nsm.save(out_dir)
    loaded = NodeStateManager.load(out_dir)

    assert len(loaded._snap_times) == len(nsm._snap_times)
    assert len(loaded._snap_states) == len(nsm._snap_states)
    assert loaded.snapshot_interval == 2


# ---------------------------------------------------------------------------
# Test 8: Query-path parity — snapshots on vs off must not change real queries
# ---------------------------------------------------------------------------

def test_query_results_identical_with_snapshots_on_or_off():
    """get_state_at_time / rollback_edge give identical results regardless of
    whether snapshot generation (Pass 2) ran — proving the snapshot cache is
    never consulted by the real query path."""
    arr = _arrays_from_edges(_SNAPSHOT_TEST_EDGES)
    half = len(_SNAPSHOT_TEST_EDGES) // 2

    def _make(interval):
        nsm = NodeStateManager(window_seconds=1.0, snapshot_interval=interval)
        nsm.build_hourly_baselines(
            arr["src"][:half], arr["dst"][:half], arr["ts"][:half],
            arr["ib"][:half], arr["ob"][:half],
        )
        nsm.build_snapshots(arr["src"], arr["dst"], arr["ts"], arr["ib"], arr["ob"], arr["dp"])
        max_node = max(int(arr["src"].max()), int(arr["dst"].max())) + 1
        nsm.set_is_internal(np.zeros(max_node, dtype=np.float32))
        return nsm

    nsm_off = _make(0)
    nsm_on  = _make(2)

    for node_id in range(4):
        s_off = nsm_off.get_state_at_time(node_id=node_id, time_ms=600.0)
        s_on  = nsm_on.get_state_at_time(node_id=node_id, time_ms=600.0)
        np.testing.assert_array_equal(s_off, s_on)

    rb_off = nsm_off.rollback_edge(
        node_id=0, edge_timestamp_ms=500.0, edge_direction="outgoing",
        edge_features={"peer_id": 2}, query_time_ms=600.0,
    )
    rb_on = nsm_on.rollback_edge(
        node_id=0, edge_timestamp_ms=500.0, edge_direction="outgoing",
        edge_features={"peer_id": 2}, query_time_ms=600.0,
    )
    np.testing.assert_array_equal(rb_off, rb_on)


# ===========================================================================
# Vectorized get_batch_states — parity with the retained _compute_state oracle
# (spec 21 §4). Parity split per spec 21 §3.9:
#   bit-exact dims: 0,1,2,3,4,5,6,7,11,12
#   allclose (atol 1e-5) dims: 8,9,10,13,14
# ===========================================================================

_EXACT_DIMS = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]
_CLOSE_DIMS = [8, 9, 10, 13, 14]


def _assert_batch_parity(nsm: NodeStateManager, node_ids, t: float) -> None:
    """Assert get_batch_states matches the stacked _compute_state oracle."""
    node_ids = np.asarray(node_ids, dtype=np.int64)
    oracle = np.stack([nsm._compute_state(int(v), t) for v in node_ids])
    got = nsm.get_batch_states(node_ids, t)
    assert got.shape == (len(node_ids), 15)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got[:, _EXACT_DIMS], oracle[:, _EXACT_DIMS])
    np.testing.assert_allclose(
        got[:, _CLOSE_DIMS], oracle[:, _CLOSE_DIMS], atol=1e-5, rtol=0
    )


_PARITY_EDGES = [
    {"ts": 100, "src": 0, "dst": 1, "in_bytes": 0,   "out_bytes": 100, "dst_port": 80},
    {"ts": 200, "src": 0, "dst": 2, "in_bytes": 0,   "out_bytes": 200, "dst_port": 443},
    {"ts": 300, "src": 0, "dst": 1, "in_bytes": 0,   "out_bytes": 150, "dst_port": 80},
    {"ts": 400, "src": 3, "dst": 0, "in_bytes": 500, "out_bytes": 0,   "dst_port": 9999},
    {"ts": 500, "src": 0, "dst": 2, "in_bytes": 0,   "out_bytes": 300, "dst_port": 22},
]


# ---------------------------------------------------------------------------
# Category 1 — oracle parity
# ---------------------------------------------------------------------------

_BOTH_MODES = [NOVELTY_MODE_RECENT_WINDOW, NOVELTY_MODE_UNSEEN_IN_TRAINING]


@pytest.mark.parametrize("novelty_mode", _BOTH_MODES)
@pytest.mark.parametrize("t", [-500.0, 50.0, 350.0, 600.0, 5000.0])
def test_batch_states_oracle_parity(t, novelty_mode):
    """Vectorized batch equals the stacked scalar oracle across query times.

    T2 — parametrized over both novelty modes (specs/60 §6.2). dim 1 is in
    ``_EXACT_DIMS``, so this asserts bit-parity of the two dim-1
    implementations under the new mode as well as the default.
    """
    nsm = _build_nsm(_PARITY_EDGES, W_ms=1000.0, novelty_mode=novelty_mode)
    node_ids = np.array([0, 1, 2, 3], dtype=np.int64)
    _assert_batch_parity(nsm, node_ids, t)


# ---------------------------------------------------------------------------
# Category 2 — edge cases (each asserted against the _compute_state oracle)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("novelty_mode", _BOTH_MODES)
def test_batch_states_edge_cases(novelty_mode):
    """Every edge case in spec 20 §1.9 item 2, vs the oracle under §3.9 split.

    T2 — parametrized over both novelty modes (specs/60 §6.2): out-of-domain
    ids, negative ids, history-less nodes, duplicates and unsorted batches all
    have to agree on dim 1 between the vectorized and scalar paths.
    """
    nsm = _build_nsm(_PARITY_EDGES, W_ms=1000.0, novelty_mode=novelty_mode)
    # Domain: nodes 0-3 have history; is_internal length = 4.

    # Node absent from _histories but in domain: use an id with no edges by
    # extending is_internal so num_nodes > max history id.
    nsm2 = _build_nsm(_PARITY_EDGES, W_ms=1000.0, novelty_mode=novelty_mode)
    nsm2.set_is_internal(np.zeros(6, dtype=np.float32))  # domain now 0..5
    # node 5 in domain, no history
    _assert_batch_parity(nsm2, [5], 600.0)

    # Node with history but empty window (query far after last edge).
    _assert_batch_parity(nsm, [0], 100000.0)

    # Exactly 1 in-window edge → dim 14 = 0. Node 3 has one edge at t=400.
    _assert_batch_parity(nsm, [3], 600.0)

    # All-outgoing node (node 0 mostly out) and all-incoming (node 1 receives).
    _assert_batch_parity(nsm, [0, 1, 2, 3], 600.0)

    # Boundary: first_seen == lo and first_seen == t, ts.max() == t.
    _assert_batch_parity(nsm, [0], 100.0)     # t == first_seen edge
    _assert_batch_parity(nsm, [0], 1100.0)    # w_start=100=first_seen boundary

    # nid >= len(is_internal) but WITH history: shrink is_internal to len 1,
    # node 0 keeps history, dim 0 must be 0 (independent guard).
    nsm3 = _build_nsm(_PARITY_EDGES, W_ms=1000.0, novelty_mode=novelty_mode)
    nsm3.set_is_internal(np.array([1.0], dtype=np.float32))  # only node 0 covered
    # node 2 has history but is beyond is_internal length -> dim0 = 0
    _assert_batch_parity(nsm3, [0, 1, 2, 3], 600.0)

    # nid >= num_nodes (fully out of domain).
    _assert_batch_parity(nsm, [9999], 600.0)
    _assert_batch_parity(nsm, [-1], 600.0)

    # Hour-rollover t (hour_int wraps toward 0): t ~ 23:59 then 00:01.
    t_2359 = (23 * 3600 + 59 * 60) * 1000.0
    t_0001 = (24 * 3600 + 60) * 1000.0
    _assert_batch_parity(nsm, [0, 1, 2, 3], t_2359)
    _assert_batch_parity(nsm, [0, 1, 2, 3], t_0001)

    # Duplicate node_ids → identical rows.
    dup = np.array([0, 0, 2, 2], dtype=np.int64)
    got = nsm.get_batch_states(dup, 600.0)
    np.testing.assert_array_equal(got[0], got[1])
    np.testing.assert_array_equal(got[2], got[3])
    _assert_batch_parity(nsm, dup, 600.0)

    # Unsorted node_ids → row order preserved.
    _assert_batch_parity(nsm, [3, 1, 0, 2], 600.0)


# ---------------------------------------------------------------------------
# Category 4 — contract checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("novelty_mode", _BOTH_MODES)
def test_batch_states_contract(novelty_mode):
    """Shape/dtype/row-order/empty-batch contract (T2 — both modes)."""
    nsm = _build_nsm(_PARITY_EDGES, W_ms=1000.0, novelty_mode=novelty_mode)

    got = nsm.get_batch_states(np.array([0, 2], dtype=np.int64), 600.0)
    assert got.shape == (2, 15)
    assert got.dtype == np.float32
    # Row order matches node_ids.
    o0 = nsm._compute_state(0, 600.0)
    o2 = nsm._compute_state(2, 600.0)
    np.testing.assert_array_equal(got[0][_EXACT_DIMS], o0[_EXACT_DIMS])
    np.testing.assert_array_equal(got[1][_EXACT_DIMS], o2[_EXACT_DIMS])

    # Empty batch.
    empty = nsm.get_batch_states(np.empty(0, dtype=np.int64), 600.0)
    assert empty.shape == (0, 15)
    assert empty.dtype == np.float32


# ---------------------------------------------------------------------------
# Category 3 / 5 — real-data parity + micro-benchmark (artifact-gated)
# ---------------------------------------------------------------------------

def _artifacts_available() -> bool:
    cfg = resolve_cfg()
    gdir = REPO_ROOT / cfg["graph"]["dir"]
    nsd = REPO_ROOT / cfg["graph"]["node_state_dir"]
    return (gdir / "train.bin").exists() and (nsd / "histories.pkl").exists()


def _real_input_batches(split: str, n_batches: int = 2, batch_size: int = 512):
    """Load nsm and yield (input_nodes np.ndarray, batch_ts float) mini-batches.

    Mirrors trainer.py:268-276 sampler-driven input_nodes reconstruction.
    """
    import dgl
    import torch

    from src.model.temporal_sampler import TemporalNeighborSampler

    cfg = resolve_cfg()
    nsm = NodeStateManager.load(REPO_ROOT / cfg["graph"]["node_state_dir"])

    g_list, _ = dgl.load_graphs(str(REPO_ROOT / cfg["graph"]["dir"] / f"{split}.bin"))
    g = g_list[0]

    sampler = TemporalNeighborSampler(fanouts=cfg["model"]["fanouts"])

    n_edges = g.num_edges()
    batches = []
    for b in range(n_batches):
        lo = b * batch_size
        if lo >= n_edges:
            break
        hi = min(lo + batch_size, n_edges)
        batch_local = torch.arange(lo, hi, dtype=torch.long)
        input_nodes, _seed_local, _blocks = sampler.sample_blocks(g, batch_local)
        batch_ts = float(g.edata["timestamp"][batch_local].max().item())
        batches.append((input_nodes.numpy(), batch_ts))
    return nsm, batches


@pytest.mark.parametrize("split", ["train", "val"])
def test_batch_states_real_data_parity(split):
    """Real-data parity: get_batch_states vs stacked _compute_state oracle."""
    if not _artifacts_available():
        pytest.skip(
            "graphs/*.bin or node_state artifacts not found — run "
            "scripts/02_build_graph.py / node-state build first"
        )
    nsm, batches = _real_input_batches(split, n_batches=2)
    assert len(batches) > 0, "no mini-batches produced"

    saw_nonempty = False
    for input_nodes, batch_ts in batches:
        if input_nodes.size > 0:
            saw_nonempty = True
        got = nsm.get_batch_states(input_nodes, batch_ts)
        oracle = np.stack(
            [nsm._compute_state(int(v), batch_ts) for v in input_nodes]
        ) if input_nodes.size > 0 else np.zeros((0, 15), dtype=np.float32)
        assert got.shape == (input_nodes.size, 15)
        if input_nodes.size > 0:
            np.testing.assert_array_equal(
                got[:, _EXACT_DIMS], oracle[:, _EXACT_DIMS]
            )
            np.testing.assert_allclose(
                got[:, _CLOSE_DIMS], oracle[:, _CLOSE_DIMS], atol=1e-5, rtol=0
            )
    assert saw_nonempty, "all real mini-batches had empty input_nodes"


def test_batch_states_microbenchmark(capsys):
    """Timed old-loop vs vectorized on real input_nodes; assert parity only."""
    if not _artifacts_available():
        pytest.skip(
            "graphs/*.bin or node_state artifacts not found — run "
            "scripts/02_build_graph.py / node-state build first"
        )
    import time as _time

    # NF-UNSW-NB15-v3 is IP-level (~44 nodes), so input_nodes saturates near
    # the full node set even for a modest batch — N is dataset-bounded here.
    nsm, batches = _real_input_batches("train", n_batches=3, batch_size=4096)
    input_nodes, batch_ts = max(batches, key=lambda b: b[0].size)
    assert input_nodes.size > 0

    # Warm the flat index so the build cost isn't charged to the timed call.
    nsm.get_batch_states(input_nodes, batch_ts)

    t0 = _time.perf_counter()
    old = np.stack([nsm._compute_state(int(v), batch_ts) for v in input_nodes])
    t_old = _time.perf_counter() - t0

    t1 = _time.perf_counter()
    new = nsm.get_batch_states(input_nodes, batch_ts)
    t_new = _time.perf_counter() - t1

    # Evidence only — no speedup threshold asserted.
    with capsys.disabled():
        speedup = (t_old / t_new) if t_new > 0 else float("inf")
        print(
            f"\n[microbenchmark] N={input_nodes.size} nodes | "
            f"old loop={t_old * 1e3:.2f} ms | vectorized={t_new * 1e3:.2f} ms | "
            f"speedup={speedup:.1f}x"
        )

    np.testing.assert_array_equal(new[:, _EXACT_DIMS], old[:, _EXACT_DIMS])
    np.testing.assert_allclose(
        new[:, _CLOSE_DIMS], old[:, _CLOSE_DIMS], atol=1e-5, rtol=0
    )


# ===========================================================================
# Spec 23 — code-review fix tests (Findings 1, 3, 5)
# ===========================================================================


@pytest.mark.parametrize("novelty_mode", _BOTH_MODES)
def test_batch_states_negative_id_parity(novelty_mode):
    """Finding 1 — negative node_id parity with a non-degenerate is_internal.

    T2 — parametrized over both novelty modes: under ``"unseen_in_training"``
    the vectorized path looks membership up on the RAW ids, so a negative id
    must not alias node 0's membership.
    """
    nsm = _build_nsm(_PARITY_EDGES, W_ms=1000.0, novelty_mode=novelty_mode)
    # Non-degenerate is_internal: last element (node 3) = 1.0 so is_internal[-1] != 0.
    nsm.set_is_internal(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    nsm._flat_index = None            # force a rebuild with the new is_internal

    # Valid internal node still parity-correct (dim 0 == 1.0 on both paths).
    _assert_batch_parity(nsm, [3], 600.0)

    # Negative id in [-len, -1]: WITHOUT the §2.2 guard the scalar oracle reads
    # is_internal[-1] == 1.0 while the vectorized path gives 0.0 -> mismatch.
    # WITH the guard both return the clean out-of-domain vector (dim 0 == 0).
    _assert_batch_parity(nsm, [-1], 600.0)
    assert nsm.get_batch_states(np.array([-1]), 600.0)[0, 0] == 0.0

    # Large-negative id (< -len(is_internal)): WITHOUT the guard the scalar
    # oracle raises IndexError; WITH it both return the clean vector.
    _assert_batch_parity(nsm, [-100], 600.0)

    # Mixed batch with a valid internal node, a negative id, and a duplicate.
    _assert_batch_parity(nsm, [3, -1, 3, -100], 600.0)


def test_batch_states_window_bounds_composite_equivalence():
    """Finding 3 — composite-key window bounds equal the per-segment oracle."""
    nsm = _build_nsm(_PARITY_EDGES, W_ms=1000.0)
    nsm.get_batch_states(np.array([0], dtype=np.int64), 0.0)   # trigger _ensure_flat_index
    fi = nsm._flat_index
    composite   = fi["composite"]
    big         = fi["big"]
    node_indptr = fi["node_indptr"]
    flat_ts     = fi["flat_ts"]
    num_nodes   = fi["num_nodes"]
    W_ms        = nsm._W_ms
    max_ts      = int(flat_ts.max())

    def _ref_and_bat(v, t):
        lo = int(t - W_ms); hi = int(t)
        seg_lo, seg_hi = int(node_indptr[v]), int(node_indptr[v + 1])
        seg_ts = flat_ts[seg_lo:seg_hi]
        left  = int(np.searchsorted(seg_ts, lo, side="left"))
        right = int(np.searchsorted(seg_ts, hi, side="right"))
        win_lo_ref, win_len_ref = seg_lo + left, right - left
        lo_c = min(max(lo, 0), big)
        hi_c = min(max(hi, -1), big - 1)
        win_lo_bat  = int(np.searchsorted(composite, lo_c + v * big, side="left"))
        win_hi_bat  = int(np.searchsorted(composite, hi_c + v * big, side="right"))
        return (win_lo_ref, win_len_ref), (win_lo_bat, win_hi_bat - win_lo_bat)

    # Randomized sweep across all nodes and query times spanning below/in/after window.
    rng = np.random.default_rng(0)
    for _ in range(1000):
        v = int(rng.integers(0, num_nodes))
        t = float(rng.integers(-3000, 6000))
        ref, bat = _ref_and_bat(v, t)
        assert bat == ref, f"v={v} t={t}: ref={ref} bat={bat}"

    # Explicit boundary sub-cases (the traps §3.3 calls out):
    for v in range(num_nodes):
        # hi == max_ts (tightest in-domain upper bound).
        assert _ref_and_bat(v, float(max_ts))[0] == _ref_and_bat(v, float(max_ts))[1]
        # lo > big on a segment containing ts == max_ts (the big-vs-big-1 trap):
        # pick t so lo = int(t - W_ms) > big, i.e. t > big + W_ms.
        t_hi = float(big + W_ms + 10_000)
        assert _ref_and_bat(v, t_hi)[0] == _ref_and_bat(v, t_hi)[1]
        # lo and hi both negative (fully-before-window).
        assert _ref_and_bat(v, -5000.0)[0] == _ref_and_bat(v, -5000.0)[1]


def test_node_state_uses_preprocessor_port_bin_constant():
    """Finding 5 — the module uses the shared preprocessor port-bin constant."""
    import src.model.node_state as ns
    from src.data.preprocessor import N_DST_PORT_BINS as PP_BINS
    assert ns.N_DST_PORT_BINS is PP_BINS       # module-level import, same object
    assert ns.N_DST_PORT_BINS == 16


# ===========================================================================
# specs/59 + specs/60 — model.novelty_mode ("recent_window" vs
# "unseen_in_training"). T1-T11 per specs/60 §6.
#
# Training-node set for _build_nsm(_PARITY_EDGES): half = 5 // 2 = 2, so the
# training slice is the edges at ts 100 (0->1) and ts 200 (0->2)
# => train nodes {0, 1, 2}. Node 3 first appears at ts 400 and is UNSEEN.
# ===========================================================================

_GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" \
    / "node_state_golden_recent_window.npz"

_GOLDEN_KEYS = ("scalar", "batch_all", "batch_odd", "rollback_first")

_GOLDEN_FAILURE_NOTE = (
    "\n\nDO NOT REGENERATE THE FIXTURE TO MAKE THIS PASS. "
    "tests/fixtures/node_state_golden_recent_window.npz was generated from "
    "src/model/node_state.py as it stood at the pre-change commit (its "
    "source_sha256 pins that file's exact bytes). A mismatch here means the "
    "default 'recent_window' path — the published, peer-reviewed dim-1 "
    "behaviour — has CHANGED, which specs/59 §6 forbids. Fix the product code."
)

# Cross-machine comparison policy for the golden fixture.
#
# The fixture is generated on one machine and asserted on another (developer
# laptop vs. the HPC container), so dims computed through libm transcendentals
# are NOT bit-portable: np.sin/np.cos (dims 11/12), np.log/np.log1p (8, 9, 10)
# and everything derived from them (13, 14) can differ by a float32 ULP between
# numpy/BLAS builds and CPU architectures. Observed for real on supek
# 2026-08-14: 2 of 480 elements differed by 1.1920929e-07 — exactly one ULP at
# that magnitude — which failed the pipeline's pytest gate and killed a
# 38-job chain.
#
# Relaxing those dims to a float32 tolerance does not weaken what this test
# exists to pin. Dim 1 (novelty) is pure boolean logic and stays EXACT, as do
# the flag/count dims, which carry no transcendental. A real change to the
# published recent_window dim-1 behaviour still fails loudly.
_GOLDEN_EXACT_DIMS = [0, 1, 3, 4, 5, 6, 7]   # flags and integer-valued counts
_GOLDEN_TOL = 1e-6                            # float32-appropriate; ~8x one ULP


def _assert_golden_match(actual: np.ndarray, expected: np.ndarray,
                         key: str, context: str) -> None:
    """Compare one golden array: exact on flag/count dims, tolerant elsewhere.

    Args:
        actual:   freshly computed array, shape (..., 15).
        expected: the stored fixture array, same shape.
        key:      fixture key name, for the failure message.
        context:  how the manager was constructed, for the failure message.
    """
    assert actual.shape == expected.shape, (
        f"golden shape mismatch on '{key}' ({context}): "
        f"{actual.shape} != {expected.shape}" + _GOLDEN_FAILURE_NOTE
    )
    # Exact — dim 1 is the published phi_N behaviour this test exists to pin.
    np.testing.assert_array_equal(
        actual[..., _GOLDEN_EXACT_DIMS], expected[..., _GOLDEN_EXACT_DIMS],
        err_msg=f"golden mismatch on '{key}' ({context}), "
                f"EXACT dims {_GOLDEN_EXACT_DIMS}" + _GOLDEN_FAILURE_NOTE,
    )
    # Tolerant — libm-dependent dims (see the policy note above).
    tol_dims = [d for d in range(actual.shape[-1]) if d not in _GOLDEN_EXACT_DIMS]
    np.testing.assert_allclose(
        actual[..., tol_dims], expected[..., tol_dims],
        rtol=_GOLDEN_TOL, atol=_GOLDEN_TOL,
        err_msg=f"golden mismatch on '{key}' ({context}), "
                f"tolerant dims {tol_dims} beyond {_GOLDEN_TOL}"
                + _GOLDEN_FAILURE_NOTE,
    )


def _compute_golden_arrays(nsm: NodeStateManager) -> dict:
    """Recompute the golden arrays from a freshly built manager.

    Args:
        nsm: manager built exactly as the generator builds it.

    Returns:
        Mapping of fixture key -> freshly computed array.
    """
    from tests.gen_node_state_golden import compute_golden
    return compute_golden(nsm)


# ---------------------------------------------------------------------------
# T1 — default / absent config reproduces the published behaviour BIT-EXACTLY
# ---------------------------------------------------------------------------

def test_default_mode_matches_golden():
    """T1 — the default path is bit-identical to the pre-change code.

    The golden fixture was generated BEFORE model.novelty_mode existed, so this
    is the only assertion in the suite that pins today's behaviour to the code
    that produced the published phi_N result, rather than to the post-change
    code comparing against itself.
    """
    from tests.gen_node_state_golden import build_reference_manager

    assert _GOLDEN_PATH.exists(), f"missing golden fixture: {_GOLDEN_PATH}"
    golden = np.load(_GOLDEN_PATH, allow_pickle=False)

    # (a) constructed with NO novelty_mode argument at all.
    fresh = _compute_golden_arrays(build_reference_manager(novelty_mode=None))
    for key in _GOLDEN_KEYS:
        _assert_golden_match(fresh[key], golden[key], key,
                             "no novelty_mode argument")

    # (b) constructed with an EXPLICIT "recent_window" — must be identical.
    explicit = _compute_golden_arrays(
        build_reference_manager(novelty_mode=NOVELTY_MODE_RECENT_WINDOW)
    )
    for key in _GOLDEN_KEYS:
        _assert_golden_match(explicit[key], golden[key], key,
                             "explicit novelty_mode='recent_window'")

    # (c) the two constructions must agree with each other BIT-EXACTLY on every
    # dim — same machine, same libm, so no tolerance is warranted here. This
    # keeps a full-precision equality assertion in the test despite (a)/(b)
    # being cross-machine tolerant.
    for key in _GOLDEN_KEYS:
        np.testing.assert_array_equal(
            fresh[key], explicit[key],
            err_msg=f"'{key}': omitting novelty_mode and passing "
                    f"'recent_window' explicitly disagree on the SAME machine"
                    + _GOLDEN_FAILURE_NOTE,
        )


def test_golden_fixture_provenance_is_recorded():
    """T1 support — the fixture carries the provenance stamp stage 5 verifies.

    Checks only that the stamp EXISTS and is well-formed; the cross-check
    against ``git show <git_head>:src/model/node_state.py | sha256sum`` is a
    stage-5 review step (specs/60 §6.1, §8).
    """
    golden = np.load(_GOLDEN_PATH, allow_pickle=False)
    src_sha = str(golden["source_sha256"])
    head = str(golden["git_head"])
    assert len(src_sha) == 64 and all(c in "0123456789abcdef" for c in src_sha)
    assert head != "unknown" and len(head) == 40


# ---------------------------------------------------------------------------
# T3 — semantics of the new mode
# ---------------------------------------------------------------------------

def test_unseen_mode_semantics():
    """T3 — dim 1 marks nodes absent from the training split, time-invariantly."""
    nsm = _build_nsm(_PARITY_EDGES, W_ms=1000.0,
                     novelty_mode=NOVELTY_MODE_UNSEEN_IN_TRAINING)
    ref = _build_nsm(_PARITY_EDGES, W_ms=1000.0)   # default mode

    np.testing.assert_array_equal(
        nsm._train_node_ids, np.array([0, 1, 2], dtype=np.int64)
    )

    # Node 3 is absent from the training split -> novel, at ANY query time
    # at or after its first appearance (D2 time-invariance).
    assert nsm.get_state_at_time(3, 600.0)[1] == 1.0
    assert nsm.get_state_at_time(3, 5000.0)[1] == 1.0

    # Node 0 IS a training node -> never novel, even at a time where the
    # window predicate would fire. Proves the predicate was REPLACED, not OR-ed.
    assert ref.get_state_at_time(0, 600.0)[1] == 1.0, (
        "fixture assumption broken: recent_window must fire for node 0 at t=600"
    )
    assert nsm.get_state_at_time(0, 600.0)[1] == 0.0

    # D4 — a domain-resident node with no history, an out-of-domain id and a
    # negative id all read 0.0 on both surfaces.
    nsm5 = _build_nsm(_PARITY_EDGES, W_ms=1000.0,
                      novelty_mode=NOVELTY_MODE_UNSEEN_IN_TRAINING)
    nsm5.set_is_internal(np.zeros(6, dtype=np.float32))   # domain 0..5
    assert nsm5.get_state_at_time(5, 600.0)[1] == 0.0
    assert nsm5.get_batch_states(np.array([5], dtype=np.int64), 600.0)[0, 1] == 0.0
    for bad in (9999, -1):
        assert nsm.get_state_at_time(bad, 600.0)[1] == 0.0
        assert nsm.get_batch_states(
            np.array([bad], dtype=np.int64), 600.0
        )[0, 1] == 0.0


# ---------------------------------------------------------------------------
# T4 — causality: a node is not "novel" before it has appeared at all
# ---------------------------------------------------------------------------

def test_unseen_mode_causality():
    """T4 — dim 1 is 0 for an unseen node queried BEFORE its first edge (D1)."""
    nsm = _build_nsm(_PARITY_EDGES, W_ms=1000.0,
                     novelty_mode=NOVELTY_MODE_UNSEEN_IN_TRAINING)
    # Node 3's only edge is at ts 400; query at 300 precedes it.
    assert nsm.get_state_at_time(3, 300.0)[1] == 0.0
    assert nsm.get_batch_states(np.array([3], dtype=np.int64), 300.0)[0, 1] == 0.0
    # ... and 1.0 once it has appeared, so the 0.0 above is causality, not a
    # dead code path.
    assert nsm.get_state_at_time(3, 400.0)[1] == 1.0


# ---------------------------------------------------------------------------
# T5 — rollback invariance under the new mode (D3)
# ---------------------------------------------------------------------------

def test_unseen_mode_rollback_invariance():
    """T5 — masking the first-seen edge does NOT change dim 1 in the new mode.

    Mirror of ``test_rollback_first_edge_novelty`` (default mode), which stays
    unchanged. Training membership is a property of the node, so a masked
    neighbour edge cannot alter it (specs/59 D3).
    """
    nsm = _build_nsm(_PARITY_EDGES, W_ms=1000.0,
                     novelty_mode=NOVELTY_MODE_UNSEEN_IN_TRAINING)

    # Unseen node (3): first-seen edge at ts 400, incoming from node 0.
    assert nsm.get_state_at_time(3, 600.0)[1] == 1.0
    after = nsm.rollback_edge(
        node_id=3, edge_timestamp_ms=400.0, edge_direction="outgoing",
        edge_features={"peer_id": 0}, query_time_ms=600.0,
    )
    assert after[1] == 1.0, "unseen node must stay novel across a rollback"

    # Training node (0): first-seen edge at ts 100, outgoing to node 1.
    assert nsm.get_state_at_time(0, 600.0)[1] == 0.0
    after0 = nsm.rollback_edge(
        node_id=0, edge_timestamp_ms=100.0, edge_direction="outgoing",
        edge_features={"peer_id": 1}, query_time_ms=600.0,
    )
    assert after0[1] == 0.0

    # Under the DEFAULT mode the same rollback DOES move dim 1 — proving the
    # invariance above is mode-specific and not an inert assertion.
    ref = _build_nsm(_PARITY_EDGES, W_ms=1000.0)
    assert ref.get_state_at_time(3, 600.0)[1] == 1.0
    assert ref.rollback_edge(
        node_id=3, edge_timestamp_ms=400.0, edge_direction="outgoing",
        edge_features={"peer_id": 0}, query_time_ms=600.0,
    )[1] == 0.0


# ---------------------------------------------------------------------------
# T6 — persistence round-trip and legacy-directory degradation
# ---------------------------------------------------------------------------

def _rewrite_legacy_meta(out_dir: Path) -> None:
    """Strip a saved directory back to the pre-specs/60 on-disk shape.

    Args:
        out_dir: directory previously written by ``NodeStateManager.save``.
    """
    tn = out_dir / "train_node_ids.npy"
    if tn.exists():
        tn.unlink()
    with open(out_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump({"window_seconds": meta["window_seconds"],
                     "snapshot_interval": meta["snapshot_interval"]}, f)


def test_novelty_mode_persistence_roundtrip(tmp_path):
    """T6 — save/load round-trips the mode and the training-node set."""
    nsm = _build_nsm(_PARITY_EDGES, W_ms=1000.0,
                     novelty_mode=NOVELTY_MODE_UNSEEN_IN_TRAINING)
    out_dir = tmp_path / "ns_unseen"
    nsm.save(out_dir)

    assert (out_dir / "train_node_ids.npy").exists()
    loaded = NodeStateManager.load(out_dir)
    assert loaded.novelty_mode == NOVELTY_MODE_UNSEEN_IN_TRAINING
    np.testing.assert_array_equal(loaded._train_node_ids, nsm._train_node_ids)
    assert loaded.get_state_at_time(3, 600.0)[1] == 1.0
    assert loaded.get_state_at_time(0, 600.0)[1] == 0.0


def test_legacy_snapshot_dir_degrades_to_default_mode(tmp_path):
    """T6 — a directory written before specs/60 loads as 'recent_window'.

    Simulated by saving, then deleting ``train_node_ids.npy`` and rewriting
    ``meta.pkl`` with only the two keys the old ``save()`` wrote.
    """
    nsm = _build_nsm(_PARITY_EDGES, W_ms=1000.0)
    out_dir = tmp_path / "ns_legacy"
    nsm.save(out_dir)
    _rewrite_legacy_meta(out_dir)

    loaded = NodeStateManager.load(out_dir)
    assert loaded.novelty_mode == NOVELTY_MODE_RECENT_WINDOW
    assert loaded._train_node_ids is None

    # Behaviour matches a never-saved default-mode manager, bit for bit.
    ref = _build_nsm(_PARITY_EDGES, W_ms=1000.0)
    for t in (-500.0, 350.0, 600.0, 5000.0):
        for v in (0, 1, 2, 3):
            np.testing.assert_array_equal(
                loaded.get_state_at_time(v, t)[_EXACT_DIMS],
                ref.get_state_at_time(v, t)[_EXACT_DIMS],
            )

    # A caller CLAIMING the new mode against a legacy directory must be refused.
    with pytest.raises(ValueError, match="novelty_mode mismatch"):
        NodeStateManager.load(
            out_dir, expected_novelty_mode=NOVELTY_MODE_UNSEEN_IN_TRAINING
        )


def _real_node_state_dirs() -> list[Path]:
    """Every genuine on-disk node-state directory in this checkout.

    Superset of specs/60 §6.2's ``_artifacts_available()`` gate, which also
    requires ``graphs/train.bin`` — irrelevant to a load-only backward-compat
    check, and absent in checkouts that still carry usable legacy node-state
    directories.

    Returns:
        Directories containing a ``histories.pkl`` and a ``meta.pkl``.
    """
    cfg = resolve_cfg()
    candidates = [REPO_ROOT / cfg["graph"]["node_state_dir"],
                  REPO_ROOT / "node_state_snapshots"]
    candidates += sorted((REPO_ROOT / "runs").glob("*/node_state_snapshots"))
    seen: list[Path] = []
    for d in candidates:
        if d in seen:
            continue
        if (d / "histories.pkl").exists() and (d / "meta.pkl").exists():
            seen.append(d)
    return seen


def test_legacy_on_disk_snapshot_dir_loads_as_default_mode():
    """T6b — REAL on-disk node-state directories still load (artifact-gated).

    T6's delete-and-rewrite simulation proves the code handles a directory the
    test itself built; this proves it handles the directories that actually
    exist (including the HPC-mirror shape) — the backward-compat claim that
    matters most (specs/60 §6.2).
    """
    dirs = _real_node_state_dirs()
    if not dirs:
        pytest.skip(
            "no on-disk node_state directory found — run "
            "scripts/02_build_graph.py first"
        )
    for d in dirs:
        nsm = NodeStateManager.load(d)
        assert nsm.novelty_mode == NOVELTY_MODE_RECENT_WINDOW, d
        assert nsm._train_node_ids is None, d
        # A pre-specs/60 directory must also satisfy an explicit default claim.
        NodeStateManager.load(d, expected_novelty_mode=NOVELTY_MODE_RECENT_WINDOW)


# ---------------------------------------------------------------------------
# T7 — validation of the mode string and the config/artifact agreement guard
# ---------------------------------------------------------------------------

def test_novelty_mode_validation(tmp_path):
    """T7 — every illegal or inconsistent mode combination raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model.novelty_mode"):
        NodeStateManager(novelty_mode="bogus")

    good = _build_nsm(_PARITY_EDGES, W_ms=1000.0)
    good_dir = tmp_path / "ns_default"
    good.save(good_dir)

    with pytest.raises(ValueError, match="Unknown expected_novelty_mode"):
        NodeStateManager.load(good_dir, expected_novelty_mode="bogus")

    unseen = _build_nsm(_PARITY_EDGES, W_ms=1000.0,
                        novelty_mode=NOVELTY_MODE_UNSEEN_IN_TRAINING)
    unseen_dir = tmp_path / "ns_unseen"
    unseen.save(unseen_dir)

    with pytest.raises(ValueError, match="novelty_mode mismatch"):
        NodeStateManager.load(
            unseen_dir, expected_novelty_mode=NOVELTY_MODE_RECENT_WINDOW
        )

    # "No claim" (expected_novelty_mode=None) accepts a consistent directory.
    loaded = NodeStateManager.load(unseen_dir)
    assert loaded.novelty_mode == NOVELTY_MODE_UNSEEN_IN_TRAINING

    # An INTERNALLY inconsistent directory is refused even with no claim.
    (unseen_dir / "train_node_ids.npy").unlink()
    with pytest.raises(ValueError, match="no train_node_ids.npy"):
        NodeStateManager.load(unseen_dir)


# ---------------------------------------------------------------------------
# T8 — the new mode refuses to guess when the training-node set is unavailable
# ---------------------------------------------------------------------------

def test_unseen_mode_membership_unavailable_raises():
    """T8 — querying without an established training-node set is a hard error.

    Asymmetry by design (specs/60 §6.2): ``_compute_state``'s no-history early
    return fires before dim 1, so the scalar surfaces only reach the membership
    lookup for a node that HAS history (node 3). ``get_batch_states`` reaches it
    unconditionally.
    """
    arr = _arrays_from_edges(_PARITY_EDGES)
    nsm = NodeStateManager(window_seconds=1.0, snapshot_interval=0,
                           novelty_mode=NOVELTY_MODE_UNSEEN_IN_TRAINING)
    nsm.build_hourly_baselines(
        arr["src"][:2], arr["dst"][:2], arr["ts"][:2],
        arr["ib"][:2], arr["ob"][:2],
    )
    # NOTE: no set_train_nodes() call.
    nsm.build_snapshots(arr["src"], arr["dst"], arr["ts"],
                        arr["ib"], arr["ob"], arr["dp"], snapshot_interval=0)
    nsm.set_is_internal(np.zeros(4, dtype=np.float32))
    assert nsm._train_node_ids is None

    with pytest.raises(RuntimeError, match="requires a training-node set"):
        nsm.get_state_at_time(3, 600.0)
    with pytest.raises(RuntimeError, match="requires a training-node set"):
        nsm.rollback_edges(3, 600.0, [])
    with pytest.raises(RuntimeError, match="requires a training-node set"):
        nsm.get_batch_states(np.array([0, 1, 2, 3], dtype=np.int64), 600.0)
    with pytest.raises(RuntimeError, match="requires a training-node set"):
        nsm.get_batch_states(np.array([-1, 9999], dtype=np.int64), 600.0)


# ---------------------------------------------------------------------------
# T9 — every scripts/ load site passes expected_novelty_mode (source pinning)
# ---------------------------------------------------------------------------

def _extract_call_texts(source: str, needle: str) -> list[str]:
    """Return the full text of every ``needle(...)`` call, paren-balanced.

    A naive line scan misses the multi-line call form the load sites use, so
    walk parenthesis depth from the opening paren until it returns to zero.

    Args:
        source: full file source text.
        needle: call prefix INCLUDING the opening paren, e.g. ``"foo.load("``.

    Returns:
        One string per call, from ``needle`` through its matching ``)``.
    """
    calls: list[str] = []
    start = source.find(needle)
    while start != -1:
        i = start + len(needle) - 1        # index of the opening paren
        depth = 0
        for j in range(i, len(source)):
            if source[j] == "(":
                depth += 1
            elif source[j] == ")":
                depth -= 1
                if depth == 0:
                    calls.append(source[start:j + 1])
                    break
        start = source.find(needle, start + len(needle))
    return calls


def test_all_load_sites_pass_expected_novelty_mode():
    """T9 — no scripts/ file may load a NodeStateManager without the guard.

    Globbed, never a hardcoded list, so a future script cannot silently skip
    the config/artifact agreement check (specs/60 §4, §8). explore/ is out of
    scope by owner constraint 6; tests/ deliberately load with no claim.
    """
    offenders: list[str] = []
    checked = 0
    for path in sorted((REPO_ROOT / "scripts").glob("*.py")):
        source = path.read_text()
        for call in _extract_call_texts(source, "NodeStateManager.load("):
            checked += 1
            if "expected_novelty_mode" not in call:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {call}")
    assert checked > 0, "no NodeStateManager.load( call sites found in scripts/"
    assert not offenders, (
        "these scripts/ call sites load a NodeStateManager without the "
        "config/artifact agreement guard. Add "
        "expected_novelty_mode=cfg['model'].get('novelty_mode', "
        "'recent_window') to each:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# (d) — the training-node set is derived from TRAINING-split edges only
# ---------------------------------------------------------------------------

def test_phase2_sets_train_nodes_from_train_split_only():
    """Phase 2 passes the train-only node arrays, in the required position.

    Source-text pinned (house precedent: tests/test_tuning_config.py). Two
    claims, both of which a refactor could silently break:

      1. ``set_train_nodes`` receives ``train_src_ids``/``train_dst_ids`` — the
         SAME arrays ``build_hourly_baselines`` receives, built from
         ``per_split["train"]`` only (invariant 2/3, no val/test leakage).
      2. The call sits AFTER ``build_hourly_baselines`` and BEFORE
         ``build_snapshots`` — Pass 2 calls ``_compute_state``, which reads the
         set under ``"unseen_in_training"`` (specs/60 §3, §8).
    """
    source = (REPO_ROOT / "scripts" / "02_build_graph.py").read_text()

    assert "nsm.set_train_nodes(train_src_ids, train_dst_ids)" in source, (
        "scripts/02_build_graph.py must establish the training-node set from "
        "the train-only node-id arrays (invariant 2/3)."
    )
    # The arrays really are the ones the hourly baselines use.
    assert "train_src_ids" in source and "train_dst_ids" in source
    baselines_at = source.index("nsm.build_hourly_baselines(")
    set_train_at = source.index("nsm.set_train_nodes(")
    snapshots_at = source.index("nsm.build_snapshots(")
    assert baselines_at < set_train_at < snapshots_at, (
        "set_train_nodes must be called AFTER build_hourly_baselines and "
        "BEFORE build_snapshots (specs/60 §3)."
    )
    # And no val/test array is passed to it.
    call = _extract_call_texts(source, "nsm.set_train_nodes(")[0]
    for forbidden in ("val", "test"):
        assert forbidden not in call, (
            f"set_train_nodes call mentions {forbidden!r}: {call}"
        )


# ---------------------------------------------------------------------------
# T10 — positive control: the mode fires through all three query surfaces
# ---------------------------------------------------------------------------

def test_unseen_mode_fires_on_all_three_surfaces():
    """T10 — dim 1 == 1.0 for node 3 via batch, scalar and rollback surfaces.

    A mode that failed to propagate through any single surface cannot pass.
    This is the unit-level half of specs/60 §8's anti-false-negative control:
    an all-zero dim 1 is bit-identical to the expected UNSW finding, so the
    plumbing has to be proven independently of any dataset number.
    """
    nsm = _build_nsm(_PARITY_EDGES, W_ms=1000.0,
                     novelty_mode=NOVELTY_MODE_UNSEEN_IN_TRAINING)

    batch = nsm.get_batch_states(np.array([0, 1, 2, 3], dtype=np.int64), 600.0)
    np.testing.assert_array_equal(
        batch[:, 1], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )
    assert nsm.get_state_at_time(3, 600.0)[1] == 1.0
    assert nsm.rollback_edges(3, 600.0, [])[1] == 1.0
    assert nsm.rollback_edges(
        3, 600.0, [(400.0, "outgoing", 0)]
    )[1] == 1.0
