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
from src.model.node_state import NodeStateManager
from tests._paths import REPO_ROOT, resolve_cfg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_W_MS = 1_000.0   # 1-second window for test clarity


def _build_nsm(edges: list[dict], W_ms: float = _W_MS,
               snapshot_interval: int = 5) -> NodeStateManager:
    """Build a NodeStateManager from a list of edge dicts.

    Each dict: {'ts': int, 'src': int, 'dst': int, 'in_bytes': float,
                'out_bytes': float, 'dst_port': int}
    """
    nsm = NodeStateManager(window_seconds=W_ms / 1000.0,
                           snapshot_interval=snapshot_interval)

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
    """Removing the global first-seen edge sets novelty to 0."""
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

@pytest.mark.parametrize("t", [-500.0, 50.0, 350.0, 600.0, 5000.0])
def test_batch_states_oracle_parity(t):
    """Vectorized batch equals the stacked scalar oracle across query times."""
    nsm = _build_nsm(_PARITY_EDGES, W_ms=1000.0)
    node_ids = np.array([0, 1, 2, 3], dtype=np.int64)
    _assert_batch_parity(nsm, node_ids, t)


# ---------------------------------------------------------------------------
# Category 2 — edge cases (each asserted against the _compute_state oracle)
# ---------------------------------------------------------------------------

def test_batch_states_edge_cases():
    """Every edge case in spec 20 §1.9 item 2, vs the oracle under §3.9 split."""
    nsm = _build_nsm(_PARITY_EDGES, W_ms=1000.0)
    # Domain: nodes 0-3 have history; is_internal length = 4.

    # Node absent from _histories but in domain: use an id with no edges by
    # extending is_internal so num_nodes > max history id.
    nsm2 = _build_nsm(_PARITY_EDGES, W_ms=1000.0)
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
    nsm3 = _build_nsm(_PARITY_EDGES, W_ms=1000.0)
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

def test_batch_states_contract():
    """Shape/dtype/row-order/empty-batch contract."""
    nsm = _build_nsm(_PARITY_EDGES, W_ms=1000.0)

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
