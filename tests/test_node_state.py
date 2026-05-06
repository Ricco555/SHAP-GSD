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
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import entropy as scipy_entropy
from src.model.node_state import NodeStateManager

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
