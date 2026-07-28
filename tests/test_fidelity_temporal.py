"""
Tests for the temporal-neighborhood (φ_T) fidelity metric.

Follows the toy-model philosophy of ``tests/test_shap_axioms.py`` (the real GNN
needs a GPU and loaded graphs) combined with the synthetic-NodeStateManager
fixture pattern of ``tests/test_node_state.py``: a stub model consumes only the
node-state matrix and ignores ``blocks`` entirely, so no DGL block construction
is required, while the masking path exercises the real
``NodeStateManager.rollback_edges``.

Tests (spec 24 §9):
  T1 — masking nothing gives Fidelity+ == 0 exactly.
  T2 — keeping everything (k >= N) gives Fidelity− == 0 exactly.
  T3 — build_masked_node_feats matches a direct rollback_edges call, row for row.
  T4 — empty neighborhood returns the None row and makes zero model calls.
  T5 — top-k selection ranks by |φ|, descending.
  T6 — align_phi_to_records reorders on EID, and returns None + warns on mismatch.
  T7 — two absent edges sharing an endpoint produce ONE grouped rollback call.
  T8 — the production row-alignment assertion fires on a mismatched matrix.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explainer.temporal_fidelity import (
    align_phi_to_records,
    temporal_fidelity_for_flow,
)
from src.explainer.temporal_shap import TemporalNeighborhoodSHAP, _NeighborEdge
from src.model.node_state import NodeStateManager

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

_W_MS = 1_000.0          # 1-second node-state window, as in test_node_state.py
_TARGET_TS = 600.0       # query time; window is [-400, 600] so all edges below
                         # are in-window
_NUM_CLASSES = 3
_DEVICE = torch.device("cpu")

# Every neighbor record used below is derived from THIS list, so a rollback of
# any of them genuinely removes an edge the NodeStateManager knows about. If the
# records described edges the NSM had never seen, rollback_edges would return
# the unmodified state and the faithfulness tests would pass vacuously.
_EDGES: list[dict] = [
    {"ts": 100, "src": 0, "dst": 1, "in_bytes": 0, "out_bytes": 100, "dst_port": 80},
    {"ts": 200, "src": 0, "dst": 2, "in_bytes": 0, "out_bytes": 200, "dst_port": 443},
    {"ts": 300, "src": 0, "dst": 1, "in_bytes": 0, "out_bytes": 150, "dst_port": 80},
    {"ts": 400, "src": 3, "dst": 0, "in_bytes": 500, "out_bytes": 0, "dst_port": 53},
    {"ts": 500, "src": 0, "dst": 2, "in_bytes": 0, "out_bytes": 300, "dst_port": 22},
    {"ts": 550, "src": 1, "dst": 2, "in_bytes": 0, "out_bytes": 50, "dst_port": 80},
]

# Global EIDs are deliberately not equal to the list index, so any code that
# confuses "index into neighbor_records" with "global EID" fails loudly.
_GEID_BASE = 1_000


def _build_nsm(edges: list[dict] = _EDGES, W_ms: float = _W_MS) -> NodeStateManager:
    """Build a NodeStateManager over synthetic edges (test_node_state.py pattern)."""
    nsm = NodeStateManager(window_seconds=W_ms / 1000.0, snapshot_interval=0)

    src = np.array([e["src"] for e in edges], dtype=np.int64)
    dst = np.array([e["dst"] for e in edges], dtype=np.int64)
    ts = np.array([e["ts"] for e in edges], dtype=np.int64)
    ib = np.array([e["in_bytes"] for e in edges], dtype=np.float32)
    ob = np.array([e["out_bytes"] for e in edges], dtype=np.float32)
    dp = np.array([e["dst_port"] for e in edges], dtype=np.int32)

    half = max(1, len(edges) // 2)
    nsm.build_hourly_baselines(src[:half], dst[:half], ts[:half], ib[:half], ob[:half])
    nsm.build_snapshots(src, dst, ts, ib, ob, dp, snapshot_interval=0)

    max_node = max(int(src.max()), int(dst.max())) + 1
    nsm.set_is_internal(np.zeros(max_node, dtype=np.float32))
    return nsm


def _record(edge_idx: int) -> _NeighborEdge:
    """Build the _NeighborEdge for ``_EDGES[edge_idx]``."""
    e = _EDGES[edge_idx]
    return _NeighborEdge(
        local_eid=edge_idx,
        global_eid=_GEID_BASE + edge_idx,
        timestamp_ms=float(e["ts"]),
        src_nid=int(e["src"]),
        dst_nid=int(e["dst"]),
    )


def _base_feats(nsm: NodeStateManager, input_node_ids: np.ndarray) -> np.ndarray:
    """Node-state matrix with every neighbor present, in block row order."""
    return np.stack(
        [nsm.get_state_at_time(int(nid), _TARGET_TS) for nid in input_node_ids]
    ).astype(np.float32)


class _StubModel:
    """Deterministic stand-in for EdgeAwareGraphSAGE.

    Ignores ``blocks`` and the edge features entirely and projects the summed
    node states to ``_NUM_CLASSES`` logits with a fixed matrix, so a changed
    node state necessarily changes the predicted probabilities. Counts forward
    calls so tests can assert that the N == 0 path makes none.
    """

    def __init__(self, node_state_dim: int = 15,
                 num_classes: int = _NUM_CLASSES) -> None:
        rng = np.random.default_rng(0)
        self.W = torch.tensor(
            rng.normal(size=(node_state_dim, num_classes)), dtype=torch.float32
        )
        self.n_forward = 0

    def eval(self) -> "_StubModel":
        """No-op; mirrors torch.nn.Module.eval()."""
        return self

    def encode(self, blocks: list, node_feats: torch.Tensor) -> torch.Tensor:
        """Fake node embeddings — the summed state, broadcast to one row."""
        return node_feats.sum(dim=0, keepdim=True)

    def classify(
        self,
        h: torch.Tensor,
        edge_feats: torch.Tensor,
        src_pos: torch.Tensor,
        dst_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Project a pooled embedding to logits."""
        return h @ self.W

    def __call__(
        self,
        blocks: list,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        src_pos: torch.Tensor,
        dst_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward — returns (1, num_classes) logits."""
        self.n_forward += 1
        return self.classify(
            self.encode(blocks, node_feats), edge_feats, src_pos, dst_pos
        )

    forward = __call__


class _SpyNSM:
    """Wraps a NodeStateManager, recording every rollback_edges call."""

    def __init__(self, nsm: NodeStateManager) -> None:
        self._nsm = nsm
        self.calls: list[tuple[int, float, list]] = []

    def rollback_edges(
        self, node_id: int, query_time_ms: float, excluded: list
    ) -> np.ndarray:
        """Delegate to the real NSM after recording the arguments."""
        self.calls.append((int(node_id), float(query_time_ms), list(excluded)))
        return self._nsm.rollback_edges(node_id, query_time_ms, excluded)

    def __getattr__(self, name: str) -> object:
        return getattr(self._nsm, name)


def _make_ctx(
    record_idxs: list[int],
) -> tuple[TemporalNeighborhoodSHAP, _StubModel, list[_NeighborEdge],
           np.ndarray, np.ndarray, dict]:
    """Assemble (temp_shap, model, records, input_ids, base_feats, kwargs)."""
    nsm = _build_nsm()
    # g_split is only touched by _extract_neighbor_edges, which these tests
    # bypass by constructing _NeighborEdge instances directly.
    temp_shap = TemporalNeighborhoodSHAP(
        background=None, nsm=nsm, g_split=None, device=_DEVICE
    )
    model = _StubModel()
    records = [_record(i) for i in record_idxs]
    input_node_ids = np.array([0, 1, 2, 3], dtype=np.int64)
    base = _base_feats(nsm, input_node_ids)
    fwd_kwargs = dict(
        x_e_t=torch.zeros((1, 4), dtype=torch.float32),
        src_pos=torch.tensor([0], dtype=torch.long),
        dst_pos=torch.tensor([1], dtype=torch.long),
    )
    return temp_shap, model, records, input_node_ids, base, fwd_kwargs


def _p_full(model: _StubModel, base: np.ndarray, fwd_kwargs: dict,
            true_label: int) -> float:
    """P(true_label) with every neighbor present, via the same forward path."""
    nf_t = torch.tensor(base, dtype=torch.float32, device=_DEVICE)
    with torch.no_grad():
        logits = model(
            [], nf_t, fwd_kwargs["x_e_t"], fwd_kwargs["src_pos"],
            fwd_kwargs["dst_pos"],
        )
        proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return float(proba[true_label])


# ---------------------------------------------------------------------------
# T1 — masking nothing gives Fidelity+ == 0
# ---------------------------------------------------------------------------

def test_mask_nothing_gives_zero_fidelity_plus():
    """absent_idx = [] must return base_node_feats itself → Fidelity+ == 0.0."""
    temp_shap, model, records, input_ids, base, kw = _make_ctx([0, 2, 4])

    unmasked = temp_shap.build_masked_node_feats(
        records, [], base, input_ids, _TARGET_TS
    )
    assert np.array_equal(unmasked, base)

    true_label = 0
    p_full = _p_full(model, base, kw, true_label)

    res = temporal_fidelity_for_flow(
        model=model, temp_shap=temp_shap, blocks=[], neighbor_records=records,
        phi_temporal=np.array([0.5, -0.2, 0.1]), base_node_feats=base,
        input_node_ids=input_ids, target_ts_ms=_TARGET_TS, true_label=true_label,
        p_full=p_full, top_k=0, device=_DEVICE, **kw,
    )

    assert res["effective_k"] == 0
    assert res["p_masked"] == p_full
    assert res["fidelity_plus"] == 0.0
    assert res["top_k_neighbor_eids"] == []
    # Non-vacuity: with k = 0 the Fidelity− pass masks EVERY neighbor, so the
    # kept-probability must actually move — otherwise the stub is insensitive
    # to node state and T1's exactness would be meaningless.
    assert res["p_kept"] != p_full


# ---------------------------------------------------------------------------
# T2 — keeping everything gives Fidelity− == 0
# ---------------------------------------------------------------------------

def test_keep_all_gives_zero_fidelity_minus():
    """top_k >= N leaves rest_idx empty → p_kept == p_full → Fidelity− == 0.0."""
    temp_shap, model, records, input_ids, base, kw = _make_ctx([0, 2, 4])
    n = len(records)
    true_label = 1
    p_full = _p_full(model, base, kw, true_label)

    res = temporal_fidelity_for_flow(
        model=model, temp_shap=temp_shap, blocks=[], neighbor_records=records,
        phi_temporal=np.array([0.5, -0.2, 0.1]), base_node_feats=base,
        input_node_ids=input_ids, target_ts_ms=_TARGET_TS, true_label=true_label,
        p_full=p_full, top_k=n + 2, device=_DEVICE, **kw,
    )

    assert res["n_neighbors"] == n
    assert res["effective_k"] == n
    assert res["p_kept"] == p_full
    assert res["fidelity_minus"] == 0.0
    # Non-vacuity: masking all N neighbors (the Fidelity+ pass) must move p.
    assert res["p_masked"] != p_full


# ---------------------------------------------------------------------------
# T3 — masking matches a direct rollback (anti-drift guard)
# ---------------------------------------------------------------------------

def test_masking_matches_direct_rollback():
    """build_masked_node_feats must equal explicit rollback_edges, row for row."""
    # Absent record: edge index 2 → ts=300, src=0, dst=1. Nodes 2 and 3 are
    # untouched but are still real endpoints in the synthetic NSM.
    temp_shap, _model, records, input_ids, base, _kw = _make_ctx([2])
    nsm = temp_shap.nsm
    rec = records[0]

    masked = temp_shap.build_masked_node_feats(
        records, [0], base, input_ids, _TARGET_TS
    )

    expected_src = nsm.rollback_edges(
        rec.src_nid, _TARGET_TS, [(rec.timestamp_ms, "outgoing", rec.dst_nid)]
    )
    expected_dst = nsm.rollback_edges(
        rec.dst_nid, _TARGET_TS, [(rec.timestamp_ms, "incoming", rec.src_nid)]
    )

    row_of = {int(nid): j for j, nid in enumerate(input_ids)}
    np.testing.assert_allclose(masked[row_of[rec.src_nid]], expected_src)
    np.testing.assert_allclose(masked[row_of[rec.dst_nid]], expected_dst)
    for nid in (2, 3):
        np.testing.assert_allclose(masked[row_of[nid]], base[row_of[nid]])

    # Non-vacuity: the rolled-back endpoints must genuinely differ from base,
    # otherwise the assertions above compare base against base.
    assert not np.array_equal(masked[row_of[rec.src_nid]], base[row_of[rec.src_nid]])
    assert not np.array_equal(masked[row_of[rec.dst_nid]], base[row_of[rec.dst_nid]])
    # The base matrix must not be mutated in place.
    assert masked is not base


# ---------------------------------------------------------------------------
# T4 — empty neighborhood
# ---------------------------------------------------------------------------

def test_empty_neighborhood_row():
    """N == 0 emits the None row and makes no model call at all."""
    temp_shap, model, _records, input_ids, base, kw = _make_ctx([])

    res = temporal_fidelity_for_flow(
        model=model, temp_shap=temp_shap, blocks=[], neighbor_records=[],
        phi_temporal=np.array([]), base_node_feats=base,
        input_node_ids=input_ids, target_ts_ms=_TARGET_TS, true_label=0,
        p_full=0.7, top_k=3, device=_DEVICE, **kw,
    )

    assert res["n_neighbors"] == 0
    assert res["effective_k"] == 0
    assert res["p_masked"] is None
    assert res["p_kept"] is None
    assert res["fidelity_plus"] is None
    assert res["fidelity_minus"] is None
    assert res["top_k_neighbor_eids"] == []
    # p_full is passed through, never recomputed inside the function (spec §6.1).
    assert res["p_full"] == 0.7
    assert model.n_forward == 0


# ---------------------------------------------------------------------------
# T5 — top-k selection by |φ|
# ---------------------------------------------------------------------------

def test_top_k_selection_by_abs_phi():
    """Selection ranks by |φ| descending — the largest magnitude wins."""
    temp_shap, model, records, input_ids, base, kw = _make_ctx([0, 2, 4])
    true_label = 2
    p_full = _p_full(model, base, kw, true_label)

    res = temporal_fidelity_for_flow(
        model=model, temp_shap=temp_shap, blocks=[], neighbor_records=records,
        phi_temporal=np.array([0.1, -0.9, 0.3]), base_node_feats=base,
        input_node_ids=input_ids, target_ts_ms=_TARGET_TS, true_label=true_label,
        p_full=p_full, top_k=1, device=_DEVICE, **kw,
    )

    assert res["effective_k"] == 1
    assert res["top_k_neighbor_eids"] == [records[1].global_eid]


# ---------------------------------------------------------------------------
# T6 — φ alignment onto the re-extracted record order
# ---------------------------------------------------------------------------

def test_align_phi_to_records(caplog):
    """(a) shuffled stored EIDs reorder correctly; (b) mismatch → None + warning."""
    records = [_record(i) for i in (0, 2, 4)]

    # (a) stored order shuffled relative to the re-extracted record order
    stored_eids = [records[2].global_eid, records[0].global_eid, records[1].global_eid]
    stored_phi = [0.3, 0.1, -0.9]
    phi = align_phi_to_records(records, stored_eids, stored_phi)
    np.testing.assert_allclose(phi, np.array([0.1, -0.9, 0.3]))

    # Sanity: the helper still rejects duplicate stored EIDs.
    with pytest.raises(AssertionError):
        align_phi_to_records(records[:2], [1, 1], [0.0, 0.0])

    # (b) same LENGTH, different membership → exercises the set-comparison
    # branch rather than the length short-circuit.
    bad_eids = [records[0].global_eid, records[1].global_eid, 999_999]
    with caplog.at_level(logging.WARNING, logger="src.explainer.temporal_fidelity"):
        result = align_phi_to_records(records, bad_eids, [0.1, 0.2, 0.3])
    assert result is None
    assert any("mismatch" in r.getMessage().lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# T7 — absent edges sharing an endpoint are grouped into one rollback
# ---------------------------------------------------------------------------

def test_multi_absent_grouped_per_node():
    """Two absent edges sharing node 0 → ONE rollback_edges call with both."""
    # Edge 2: 0 → 1 at ts=300. Edge 4: 0 → 2 at ts=500. Shared endpoint: node 0.
    temp_shap, _model, records, input_ids, base, _kw = _make_ctx([2, 4])
    spy = _SpyNSM(temp_shap.nsm)
    temp_shap.nsm = spy

    masked = temp_shap.build_masked_node_feats(
        records, [0, 1], base, input_ids, _TARGET_TS
    )

    by_node = {}
    for node_id, _qt, excl in spy.calls:
        assert node_id not in by_node, f"node {node_id} rolled back more than once"
        by_node[node_id] = excl

    assert set(by_node) == {0, 1, 2}
    assert len(by_node[0]) == 2, "node 0's two absent edges must be one grouped call"
    assert set(by_node[0]) == {(300.0, "outgoing", 1), (500.0, "outgoing", 2)}
    assert by_node[1] == [(300.0, "incoming", 0)]
    assert by_node[2] == [(500.0, "incoming", 0)]

    # Non-vacuity: the grouped rollback must actually change node 0's state.
    row_of = {int(nid): j for j, nid in enumerate(input_ids)}
    assert not np.array_equal(masked[row_of[0]], base[row_of[0]])


# ---------------------------------------------------------------------------
# T8 — the production row-alignment assertion
# ---------------------------------------------------------------------------

def test_build_masked_node_feats_row_alignment_assert():
    """A base matrix whose row count disagrees with input_node_ids must assert."""
    temp_shap, _model, records, input_ids, base, _kw = _make_ctx([2])

    with pytest.raises(AssertionError):
        temp_shap.build_masked_node_feats(
            records, [0], base[:-1], input_ids, _TARGET_TS
        )
