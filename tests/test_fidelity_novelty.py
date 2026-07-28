"""
Tests for the node-novelty fidelity core (src/explainer/node_fidelity.py) and
the coalition-masking helpers extracted from src/explainer/node_shap.py.

Follows tests/test_shap_axioms.py's toy-model philosophy: what is exercised is
the masking/coalition machinery, not the full GNN (which needs a GPU and loaded
graphs). The model is a deterministic stub that ignores `blocks` entirely, so no
DGL block construction is required and the tests stay gate-safe — they depend on
no runtime artifacts (graphs/, artifacts/, feature_store/, outputs/).

Fixture conventions used throughout:
  * node_state_dim = 15, num_classes = 3.
  * background_node_state[c] == c + 100.0 in every dim, so a background-replaced
    row is trivially recognisable and always differs from the actual state
    (including at the never-masked time dims 11/12, which makes the
    "copied back from the actual state" assertions falsifiable).
  * base_node_feats values are distinct per (row, dim), so a row- or dim-index
    bug surfaces instead of silently comparing equal values.
  * src_nid != dst_nid in every fixture. The self-loop case (defect D1, where
    the if/elif dispatch makes the dst-novelty player dead) is deliberately NOT
    tested: the extraction must preserve that behavior, but asserting it here
    would cement a defect that is flagged for future repair.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explainer import node_fidelity
from src.explainer.background import BackgroundDistributions
from src.explainer.node_fidelity import align_phi_to_players, novelty_fidelity_for_flow
from src.explainer.node_shap import (
    _NEVER_MASK_DIMS,
    _NOVELTY_DIM,
    NodeNoveltySHAP,
    build_non_target_ids,
    build_non_target_pos,
)

NODE_STATE_DIM = 15
NUM_CLASSES = 3
D_E = 4
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _StubModel:
    """Deterministic stand-in for EdgeAwareGraphSAGE.

    Ignores `blocks`, `edge_feats`, `src_pos` and `dst_pos` and maps the summed
    node states through a fixed projection to `(1, num_classes)` logits — the
    shape `node_fidelity._proba` requires (it calls `softmax(logits, dim=1)`).
    Counts forward calls so tests can pin that every masked evaluation runs a
    FULL forward (no cached-embedding shortcut, which would zero both metrics).
    """

    def __init__(self) -> None:
        rng = np.random.default_rng(7)
        self.W = torch.tensor(
            rng.uniform(-0.02, 0.02, size=(NODE_STATE_DIM, NUM_CLASSES)),
            dtype=torch.float32,
        )
        self.n_forward = 0

    def eval(self) -> "_StubModel":
        """No-op; present so the stub matches the model interface."""
        return self

    def __call__(
        self,
        blocks: list,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        src_pos: torch.Tensor,
        dst_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Return (1, num_classes) logits from the node states alone."""
        self.n_forward += 1
        return (node_feats.sum(dim=0) @ self.W).unsqueeze(0)


def _stub_background() -> BackgroundDistributions:
    """Real BackgroundDistributions built from two small arrays.

    `.compute()` is deliberately not called — it needs a feature store and a
    training graph. `background_node_state[c] == c + 100.0` everywhere.
    """
    bg_feats = np.zeros((NUM_CLASSES, D_E), dtype=np.float32)
    bg_ns = np.stack(
        [np.full(NODE_STATE_DIM, float(c) + 100.0, dtype=np.float32)
         for c in range(NUM_CLASSES)]
    )
    return BackgroundDistributions(bg_feats, bg_ns)


def _base_feats(n_rows: int) -> np.ndarray:
    """(n_rows, 15) float32 actual node states, distinct per (row, dim).

    Dim `_NOVELTY_DIM` is non-zero in every row so that "the novelty dim was
    zeroed" is a falsifiable assertion.
    """
    feats = np.array(
        [[1.0 + j * 0.5 + d * 0.01 for d in range(NODE_STATE_DIM)]
         for j in range(n_rows)],
        dtype=np.float32,
    )
    assert np.all(feats[:, _NOVELTY_DIM] != 0.0)
    return feats


class _Flow:
    """A toy flow: input nodes, endpoints, states and the derived player set."""

    def __init__(self, input_node_ids: list[int], src_nid: int, dst_nid: int) -> None:
        self.input_node_ids = np.array(input_node_ids, dtype=np.int64)
        self.src_nid = src_nid
        self.dst_nid = dst_nid
        self.base_node_feats = _base_feats(len(input_node_ids))
        self.non_target_ids = build_non_target_ids(self.input_node_ids, src_nid, dst_nid)
        self.non_target_pos = build_non_target_pos(self.non_target_ids)
        self.n_players = 2 + len(self.non_target_ids)
        self.row_of = {int(nid): j for j, nid in enumerate(self.input_node_ids)}

    def masked(self, ns: NodeNoveltySHAP, row: np.ndarray, true_class: int) -> np.ndarray:
        """Convenience wrapper around build_masked_node_states."""
        return ns.build_masked_node_states(
            row, self.base_node_feats, self.input_node_ids,
            self.src_nid, self.dst_nid, self.non_target_pos, true_class,
        )


@pytest.fixture
def background() -> BackgroundDistributions:
    return _stub_background()


@pytest.fixture
def node_shap(background: BackgroundDistributions) -> NodeNoveltySHAP:
    return NodeNoveltySHAP(background, DEVICE)


@pytest.fixture
def model() -> _StubModel:
    return _StubModel()


@pytest.fixture
def flow() -> _Flow:
    """5 input nodes → src=10, dst=20, M=3 non-target nodes, P=5."""
    return _Flow([10, 20, 30, 40, 50], src_nid=10, dst_nid=20)


def _edge_args() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(x_e_t, src_pos, dst_pos) — ignored by the stub model but required."""
    return (
        torch.zeros((1, D_E), dtype=torch.float32),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
    )


def _p_full(model: _StubModel, flow: _Flow, true_label: int) -> float:
    """P(true_label | everything present).

    Uses node_fidelity._proba — the same private helper the metric itself uses.
    Calling it (rather than reimplementing softmax here) is what makes the
    EXACT-equality assertions in T2/T3 well-founded: `novelty_fidelity_for_flow`
    never recomputes p_full, so any numerically-different path would make
    `fidelity_plus == 0.0` unachievable for reasons unrelated to the code
    under test.
    """
    x_e_t, src_pos, dst_pos = _edge_args()
    return node_fidelity._proba(
        model, [], flow.base_node_feats, x_e_t, src_pos, dst_pos, true_label, DEVICE
    )


def _run(
    model: _StubModel,
    node_shap: NodeNoveltySHAP,
    flow: _Flow,
    phi: np.ndarray,
    top_k: int,
    true_label: int = 1,
) -> dict:
    """Drive novelty_fidelity_for_flow on a toy flow with a real p_full."""
    p_full = _p_full(model, flow, true_label)
    model.n_forward = 0
    x_e_t, src_pos, dst_pos = _edge_args()
    return novelty_fidelity_for_flow(
        model=model,
        node_shap=node_shap,
        blocks=[],
        phi_novelty=phi,
        base_node_feats=flow.base_node_feats,
        input_node_ids=flow.input_node_ids,
        src_nid=flow.src_nid,
        dst_nid=flow.dst_nid,
        non_target_pos=flow.non_target_pos,
        x_e_t=x_e_t,
        src_pos=src_pos,
        dst_pos=dst_pos,
        true_label=true_label,
        p_full=p_full,
        top_k=top_k,
        device=DEVICE,
    )


def _assert_background_replaced(
    result: np.ndarray,
    base: np.ndarray,
    j: int,
    background: BackgroundDistributions,
    true_class: int,
) -> None:
    """Row j is background_node_state[true_class] except at the time dims."""
    bg = background.background_node_state[true_class]
    for d in range(NODE_STATE_DIM):
        if d in _NEVER_MASK_DIMS:
            np.testing.assert_allclose(result[j, d], base[j, d])
        else:
            np.testing.assert_allclose(result[j, d], bg[d])


# ---------------------------------------------------------------------------
# T1 — orientation guard
# ---------------------------------------------------------------------------


def test_all_present_row_returns_equal_states(node_shap, flow):
    """An all-ones coalition leaves every node state untouched."""
    row = np.ones(flow.n_players, dtype=np.float32)
    out = flow.masked(node_shap, row, true_class=1)

    # Equality, NOT identity: the masking helper does an unconditional
    # base_node_feats.copy(), so the result is equal-but-distinct. There is
    # deliberately no all-present fast path.
    assert np.array_equal(out, flow.base_node_feats)
    assert out is not flow.base_node_feats


# ---------------------------------------------------------------------------
# T2 / T3 — core metric orientation
# ---------------------------------------------------------------------------


def test_mask_nothing_gives_zero_fidelity_plus(model, node_shap, flow):
    """top_k=0 masks no player, so Fidelity+ is exactly zero."""
    phi = np.array([0.4, -0.3, 0.9, 0.2, 0.1])
    res = _run(model, node_shap, flow, phi, top_k=0)

    assert res["effective_k"] == 0
    assert res["p_masked"] == res["p_full"]
    assert res["fidelity_plus"] == 0.0
    assert res["top_k_players"] == []
    assert res["n_novelty_in_top_k"] == 0
    # Two full forwards, one per pass — no cached-embedding shortcut.
    assert model.n_forward == 2


def test_keep_all_gives_zero_fidelity_minus(model, node_shap, flow):
    """top_k >= P keeps every player, so Fidelity- is exactly zero."""
    phi = np.array([0.4, -0.3, 0.9, 0.2, 0.1])
    res = _run(model, node_shap, flow, phi, top_k=flow.n_players + 3)

    assert res["effective_k"] == flow.n_players
    assert res["n_players"] == flow.n_players
    assert res["p_kept"] == res["p_full"]
    assert res["fidelity_minus"] == 0.0
    assert model.n_forward == 2


# ---------------------------------------------------------------------------
# T4 — faithfulness of the background-replacement mechanism
# ---------------------------------------------------------------------------


def test_background_replacement_matches_background_object(node_shap, flow, background):
    """An absent non-target node gets the real background row, time dims kept."""
    true_class = 2
    row = np.ones(flow.n_players, dtype=np.float32)
    absent_nid = 30
    row[flow.non_target_pos[absent_nid]] = 0.0

    out = flow.masked(node_shap, row, true_class)
    j_absent = flow.row_of[absent_nid]

    _assert_background_replaced(out, flow.base_node_feats, j_absent, background, true_class)

    for j in range(len(flow.input_node_ids)):
        if j == j_absent:
            continue
        np.testing.assert_allclose(out[j], flow.base_node_feats[j])


# ---------------------------------------------------------------------------
# T5 — the dispatch pin
# ---------------------------------------------------------------------------


def test_two_mechanisms_are_not_conflated(node_shap, flow, background):
    """Novelty-dim zeroing and full background replacement must not be merged."""
    true_class = 1
    row = np.ones(flow.n_players, dtype=np.float32)
    row[0] = 0.0                                   # target src novelty absent
    absent_nid = 40
    row[flow.non_target_pos[absent_nid]] = 0.0     # one non-target node absent

    out = flow.masked(node_shap, row, true_class)
    base = flow.base_node_feats
    j_src = flow.row_of[flow.src_nid]
    j_dst = flow.row_of[flow.dst_nid]
    j_node = flow.row_of[absent_nid]

    # (a) the target-src row differs from base ONLY at the novelty dim —
    #     dims 13/14 in particular are untouched.
    for d in range(NODE_STATE_DIM):
        if d == _NOVELTY_DIM:
            continue
        np.testing.assert_allclose(out[j_src, d], base[j_src, d])

    # (b) and that dim is exactly zero (base is non-zero there).
    assert out[j_src, _NOVELTY_DIM] == 0.0
    assert base[j_src, _NOVELTY_DIM] != 0.0

    # (c) the absent non-target row is FULLY background-replaced (13/14 too).
    _assert_background_replaced(out, base, j_node, background, true_class)
    assert out[j_node, 13] != base[j_node, 13]
    assert out[j_node, 14] != base[j_node, 14]

    # (d) the target-dst row is completely unchanged.
    np.testing.assert_allclose(out[j_dst], base[j_dst])


# ---------------------------------------------------------------------------
# T6 — Fidelity- masks BOTH player types
# ---------------------------------------------------------------------------


def test_fidelity_minus_masks_both_player_types(model, node_shap, flow, monkeypatch):
    """Non-selected novelty flags are masked too, not only the node players."""
    calls: list[tuple[np.ndarray, np.ndarray]] = []
    real = node_shap.build_masked_node_states

    def spy(row, *args, **kwargs):
        out = real(row, *args, **kwargs)
        calls.append((np.asarray(row).copy(), out.copy()))
        return out

    monkeypatch.setattr(node_shap, "build_masked_node_states", spy)

    # Both novelty players have strictly the smallest |phi|, so the top-2
    # selection is unambiguously node-only.
    phi = np.array([0.001, -0.002, 0.5, 0.4, 0.3])
    res = _run(model, node_shap, flow, phi, top_k=2, true_label=1)

    assert res["n_novelty_in_top_k"] == 0
    assert all(p.startswith("node:") for p in res["top_k_players"])

    # Call 0 is the Fidelity+ row, call 1 is the Fidelity- row.
    assert len(calls) == 2
    row_kept, states_kept = calls[1]
    assert row_kept[0] == 0.0
    assert row_kept[1] == 0.0

    j_src = flow.row_of[flow.src_nid]
    j_dst = flow.row_of[flow.dst_nid]
    assert states_kept[j_src, _NOVELTY_DIM] == 0.0
    assert states_kept[j_dst, _NOVELTY_DIM] == 0.0


# ---------------------------------------------------------------------------
# T7 — top-k selection and deterministic tie-breaking
# ---------------------------------------------------------------------------


def test_top_k_selection_by_abs_phi_stable_ties(model, node_shap):
    """Selection is by |phi| descending, with ties broken deterministically."""
    # (a) largest |phi| is negative and at column 1 → guards against a missing
    #     np.abs or a missing reversal.
    flow4 = _Flow([10, 20, 30, 40], src_nid=10, dst_nid=20)
    assert flow4.n_players == 4
    res = _run(model, node_shap, flow4, np.array([0.1, -0.9, 0.3, 0.05]), top_k=1)
    assert res["top_k_players"] == ["dst_novelty"]
    assert res["n_novelty_in_top_k"] == 1

    # (b) exact-zero ties: argsort(|phi|, kind="stable")[::-1] resolves ties by
    #     DESCENDING coalition index, so with columns 0-3 all exactly zero and
    #     column 4 large, k=3 selects columns 4, 3, 2.
    flow5 = _Flow([10, 20, 30, 40, 50], src_nid=10, dst_nid=20)
    phi_tied = np.array([0.0, 0.0, 0.0, 0.0, 0.9])
    res_a = _run(model, node_shap, flow5, phi_tied, top_k=3)
    res_b = _run(model, node_shap, flow5, phi_tied, top_k=3)

    assert res_a["top_k_players"] == ["node:50", "node:40", "node:30"]
    assert res_a["top_k_players"] == res_b["top_k_players"]
    assert res_a["p_masked"] == res_b["p_masked"]
    assert res_a["p_kept"] == res_b["p_kept"]


# ---------------------------------------------------------------------------
# T8 — the M == 0 flow
# ---------------------------------------------------------------------------


def test_m_zero_flow_has_two_players(model, node_shap):
    """A flow whose block holds only the two endpoints still has two players."""
    flow2 = _Flow([10, 20], src_nid=10, dst_nid=20)
    assert flow2.non_target_ids == []
    assert flow2.non_target_pos == {}
    assert flow2.n_players == 2

    res = _run(model, node_shap, flow2, np.array([0.3, -0.2]), top_k=1)

    assert res["n_players"] == 2
    assert res["n_non_target"] == 0
    for key in ("p_masked", "p_kept", "fidelity_plus", "fidelity_minus"):
        assert res[key] is not None
        assert isinstance(res[key], float)
        assert np.isfinite(res[key])
    assert res["top_k_players"] == ["src_novelty"]


# ---------------------------------------------------------------------------
# T9 — stored-phi alignment onto the re-derived layout
# ---------------------------------------------------------------------------


def test_align_phi_to_players(caplog):
    """Stored phi is reordered onto the fresh layout, or the flow is skipped."""
    non_target_ids = [30, 40, 50]

    # (a) stored order shuffled relative to the fresh coalition order.
    phi = align_phi_to_players(
        non_target_ids,
        stored_node_ids=[50, 30, 40],
        stored_node_phi=[0.5, 0.3, 0.4],
        src_novelty_phi=0.11,
        dst_novelty_phi=-0.22,
    )
    assert phi is not None
    np.testing.assert_allclose(phi, np.array([0.11, -0.22, 0.3, 0.4, 0.5]))

    # (b) a stored node absent from the re-derived set → warning + None.
    caplog.clear()
    with caplog.at_level("WARNING", logger="src.explainer.node_fidelity"):
        out = align_phi_to_players(
            non_target_ids,
            stored_node_ids=[30, 40, 99],
            stored_node_phi=[0.3, 0.4, 0.9],
            src_novelty_phi=0.1,
            dst_novelty_phi=0.2,
        )
    assert out is None
    assert "Player-set mismatch" in caplog.text

    # (c) length mismatch between the stored and the re-derived player set.
    #     (A stored_node_ids/stored_node_phi length mismatch is a different
    #     case: that trips the production assert rather than returning None.)
    caplog.clear()
    with caplog.at_level("WARNING", logger="src.explainer.node_fidelity"):
        out = align_phi_to_players(
            non_target_ids,
            stored_node_ids=[30, 40],
            stored_node_phi=[0.3, 0.4],
            src_novelty_phi=0.1,
            dst_novelty_phi=0.2,
        )
    assert out is None
    assert "Player-set mismatch" in caplog.text


# ---------------------------------------------------------------------------
# T10 — the production assertions fire
# ---------------------------------------------------------------------------


def test_build_masked_node_states_assertions(node_shap, flow):
    """CODING STANDARDS §7: the invariants are asserted in production code."""
    # (a) base_node_feats row count disagrees with len(input_node_ids).
    short_feats = flow.base_node_feats[:-1]
    with pytest.raises(AssertionError):
        node_shap.build_masked_node_states(
            np.ones(flow.n_players, dtype=np.float32),
            short_feats, flow.input_node_ids,
            flow.src_nid, flow.dst_nid, flow.non_target_pos, 1,
        )

    # (b) correctly aligned states, but a coalition row of the wrong width.
    with pytest.raises(AssertionError):
        node_shap.build_masked_node_states(
            np.ones(flow.n_players - 1, dtype=np.float32),
            flow.base_node_feats, flow.input_node_ids,
            flow.src_nid, flow.dst_nid, flow.non_target_pos, 1,
        )


# ---------------------------------------------------------------------------
# T11 — the player layout has one source of truth
# ---------------------------------------------------------------------------


def test_non_target_ids_ordering():
    """Endpoints are removed, block row order is preserved, columns start at 2."""
    input_node_ids = np.array([70, 10, 30, 20, 50, 40], dtype=np.int64)
    non_target_ids = build_non_target_ids(input_node_ids, src_nid=10, dst_nid=20)

    assert non_target_ids == [70, 30, 50, 40]
    assert all(isinstance(n, int) for n in non_target_ids)

    pos = build_non_target_pos(non_target_ids)
    assert pos == {70: 2, 30: 3, 50: 4, 40: 5}
    assert sorted(pos.values()) == list(range(2, 2 + len(non_target_ids)))
