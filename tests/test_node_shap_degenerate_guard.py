"""
Tests for the two dummy-player guards in
``src.explainer.node_shap.NodeNoveltySHAP`` — the input-degeneracy guard
(specs/63) and the output-dummy guard (specs/65), both now applied by
``screen_novelty_players`` before the KernelSHAP regression is fit.

The specs/63 tests are grouped as T1-T7 below; the specs/65 tests as D1-D8
after them.

specs/62 sec 3b/3c found that a target endpoint whose native node-state dim
1 ("novelty" dim, ``_NOVELTY_DIM``) is already 0.0 has a coalition "absent"
state that is bit-identical to its "present" state -- the TRUE Shapley
value for that scalar is exactly 0 by construction, not solver noise near
0. specs/63's fix drops that player's column from the KernelSHAP coalition
matrix entirely and reports an exact ``0.0`` for it, reassembling the
full-width phi vector afterward.

These tests drive ``NodeNoveltySHAP.explain`` end to end through a
lightweight linear/additive fake model -- mirroring how
tests/test_shap_axioms.py's toy predict_fns avoid needing a real GNN/GPU --
so the coalition construction, degenerate guard, ``_expand_row``/
``full_to_reduced`` bookkeeping, and phi reassembly all run for real.

Also covers explore/node_shap_convergence.py's ``M_effective`` tracking
(``collect_player_counts``/``is_exhaustive``), which specs/63 sec 6 brings
in scope: after the guard, the solver's actual working width is
``P - n_degenerate_novelty_players``, not the conceptual player count P,
so the exactness determination must use ``M_effective``.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.explainer.background import BackgroundDistributions
from src.explainer.node_shap import NodeNoveltySHAP, _NOVELTY_DIM
from src.explainer.shap_gsd import ExplanationResult
from explore.node_shap_convergence import collect_player_counts, is_exhaustive

EFF_TOL = 1e-5  # coalitions below are small (M <= 2) -> KernelSHAP enumerates
                # exhaustively, so the analytic Shapley value should match tightly


def _load_phase6_module():
    """Import ``scripts/06_explain.py`` (module name is not a valid identifier),
    mirroring tests/test_evaluate_label_map.py's ``_load_phase5`` convention."""
    spec = importlib.util.spec_from_file_location(
        "phase06_explain", REPO_ROOT / "scripts" / "06_explain.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fake model: logit[0, true_class] = bias + sum_j weight[j] * nf_t[j, dim[j]]
#
# A purely linear/additive function of the (masked) node-state matrix, so
# the ground-truth Shapley value for each coalition player is known
# analytically: phi_i = c_i(present) - c_i(absent), independent of ordering
# (same pattern TestNodeNoveltySHAPAxioms/TestTemporalSHAPAxioms in
# test_shap_axioms.py rely on for their toy predict_fns).
# ---------------------------------------------------------------------------

class _LinearNoveltyModel(torch.nn.Module):
    def __init__(self, weights, dims, num_classes, true_class, bias=0.0):
        super().__init__()
        self.weights = list(weights)
        self.dims = list(dims)
        self.num_classes = num_classes
        self.true_class = true_class
        self.bias = bias

    def forward(self, blocks, nf_t, x_e_t, src_pos, dst_pos):
        val = self.bias
        for j, (w, d) in enumerate(zip(self.weights, self.dims)):
            val = val + w * float(nf_t[j, d])
        out = torch.zeros((1, self.num_classes), dtype=torch.float32)
        out[0, self.true_class] = val
        return out


def _run_scenario(
    src_dim1: float,
    dst_dim1: float,
    non_target_dim0_vals=(5.0, -3.0),
    non_target_weights=(2.0, 1.0),
    src_weight: float = 3.0,
    dst_weight: float = -2.5,
    bg_dim0: float = 0.2,
    bias: float = 0.7,
    true_class: int = 0,
    num_classes: int = 2,
    node_state_dim: int = 15,
    nsamples: int = 512,
) -> dict:
    """Build and run one NodeNoveltySHAP.explain() call against the fake model."""
    src_nid, dst_nid = 100, 200
    non_target_ids = [300, 301, 302][: len(non_target_dim0_vals)]
    input_node_ids = np.array([src_nid, dst_nid, *non_target_ids], dtype=np.int64)

    base_node_feats = np.zeros((len(input_node_ids), node_state_dim), dtype=np.float32)
    base_node_feats[0, _NOVELTY_DIM] = src_dim1
    base_node_feats[1, _NOVELTY_DIM] = dst_dim1
    for i, v in enumerate(non_target_dim0_vals):
        base_node_feats[2 + i, 0] = v
    # Never-masked time-context dims -- arbitrary constants, irrelevant here.
    base_node_feats[:, 11] = 0.11
    base_node_feats[:, 12] = 0.22

    weights = [src_weight, dst_weight] + list(non_target_weights)
    dims = [_NOVELTY_DIM, _NOVELTY_DIM] + [0] * len(non_target_dim0_vals)
    model = _LinearNoveltyModel(weights, dims, num_classes, true_class, bias=bias)

    bg_node_state = np.zeros((num_classes, node_state_dim), dtype=np.float32)
    bg_node_state[:, 0] = bg_dim0
    background = BackgroundDistributions(
        background_features=np.zeros((num_classes, 4), dtype=np.float32),
        background_node_state=bg_node_state,
    )

    explainer = NodeNoveltySHAP(background=background, device=torch.device("cpu"))

    result = explainer.explain(
        true_class=true_class,
        src_nid=src_nid,
        dst_nid=dst_nid,
        model=model,
        blocks=[],
        input_nodes=torch.tensor(input_node_ids, dtype=torch.int64),
        base_node_feats=base_node_feats,
        src_pos=torch.tensor([0]),
        dst_pos=torch.tensor([1]),
        x_e=np.zeros(4, dtype=np.float32),
        nsamples=nsamples,
    )
    result["_non_target_ids"] = non_target_ids
    result["_expected_nt_phi"] = {
        str(nid): non_target_weights[i] * (non_target_dim0_vals[i] - bg_dim0)
        for i, nid in enumerate(non_target_ids)
    }
    return result


# ---------------------------------------------------------------------------
# T1 -- native=0 (degenerate) case: guard fires, phi reported as exact 0.0
# ---------------------------------------------------------------------------

def test_t1_degenerate_src_reports_exact_zero_not_estimated_noise():
    """src native dim1=0.0: present/absent coalition inputs for that column
    are bit-identical, so the true Shapley value is exactly 0. Assert with
    ==, not a tolerance -- the whole point of the fix is "exact
    short-circuit", not "solver happened to land near zero this run"."""
    result = _run_scenario(src_dim1=0.0, dst_dim1=1.0)
    assert result["src_novelty_shap"] == 0.0
    assert result["n_degenerate_novelty_players"] == 1


def test_t1_degenerate_dst_reports_exact_zero_not_estimated_noise():
    result = _run_scenario(src_dim1=1.0, dst_dim1=0.0)
    assert result["dst_novelty_shap"] == 0.0
    assert result["n_degenerate_novelty_players"] == 1


# ---------------------------------------------------------------------------
# T2 -- native=1 (non-degenerate) case: normal estimation path unchanged
# ---------------------------------------------------------------------------

def test_t2_non_degenerate_matches_analytic_shapley_value():
    """Both endpoints native=1.0 (non-degenerate): confirms the
    reduced-width/full-width bookkeeping (full_to_reduced, _expand_row, the
    post-hoc phi reassembly) introduces no numerical drift for a genuinely
    varying player -- golden-fixture regression protection."""
    src_weight, dst_weight = 3.0, -2.5
    result = _run_scenario(src_dim1=1.0, dst_dim1=1.0, src_weight=src_weight, dst_weight=dst_weight)

    # Additive game: phi_i = c_i(present) - c_i(absent), absent target state
    # is always forced to 0.0 (node_shap.py build_masked_node_states).
    expected_src = src_weight * (1.0 - 0.0)
    expected_dst = dst_weight * (1.0 - 0.0)

    assert abs(result["src_novelty_shap"] - expected_src) < EFF_TOL
    assert abs(result["dst_novelty_shap"] - expected_dst) < EFF_TOL
    assert result["n_degenerate_novelty_players"] == 0

    for nid, expected in result["_expected_nt_phi"].items():
        assert abs(result["node_shap"][nid] - expected) < EFF_TOL


# ---------------------------------------------------------------------------
# T3 -- mixed: src degenerate, dst not (and the symmetric case)
# ---------------------------------------------------------------------------

def test_t3_mixed_src_degenerate_dst_normal():
    dst_weight = -2.5
    result = _run_scenario(src_dim1=0.0, dst_dim1=1.0, dst_weight=dst_weight)
    assert result["src_novelty_shap"] == 0.0
    assert abs(result["dst_novelty_shap"] - dst_weight * 1.0) < EFF_TOL
    assert result["n_degenerate_novelty_players"] == 1
    # Non-target players' fit is materially unaffected by which endpoint
    # was dropped.
    for nid, expected in result["_expected_nt_phi"].items():
        assert abs(result["node_shap"][nid] - expected) < EFF_TOL


def test_t3_mixed_dst_degenerate_src_normal():
    src_weight = 3.0
    result = _run_scenario(src_dim1=1.0, dst_dim1=0.0, src_weight=src_weight)
    assert result["dst_novelty_shap"] == 0.0
    assert abs(result["src_novelty_shap"] - src_weight * 1.0) < EFF_TOL
    assert result["n_degenerate_novelty_players"] == 1
    for nid, expected in result["_expected_nt_phi"].items():
        assert abs(result["node_shap"][nid] - expected) < EFF_TOL


# ---------------------------------------------------------------------------
# T4 -- both src and dst degenerate, including the M == 0 boundary case
# ---------------------------------------------------------------------------

def test_t4_both_degenerate_with_non_target_nodes():
    result = _run_scenario(src_dim1=0.0, dst_dim1=0.0)
    assert result["src_novelty_shap"] == 0.0
    assert result["dst_novelty_shap"] == 0.0
    assert result["n_degenerate_novelty_players"] == 2
    for nid, expected in result["_expected_nt_phi"].items():
        assert abs(result["node_shap"][nid] - expected) < EFF_TOL


def test_t4_both_degenerate_and_zero_non_target_nodes_reduced_size_zero():
    """M == 0 and both targets degenerate -> reduced_size == 0: the single
    achievable coalition state is the only one there is, so KernelExplainer
    must never be constructed with a zero-width background, and f_baseline
    must equal f_logit exactly."""
    result = _run_scenario(
        src_dim1=0.0, dst_dim1=0.0,
        non_target_dim0_vals=(), non_target_weights=(),
    )
    assert result["src_novelty_shap"] == 0.0
    assert result["dst_novelty_shap"] == 0.0
    assert result["node_shap"] == {}
    assert result["n_degenerate_novelty_players"] == 2
    assert result["f_baseline"] == result["f_logit"]


# ---------------------------------------------------------------------------
# T5 -- n_degenerate_novelty_players field is correct
# (folded into the assertions above: 0, 1, 1, 2, 2 respectively -- restated
# explicitly here as a single dedicated check per specs/63 sec 8 T5.)
# ---------------------------------------------------------------------------

def test_t5_n_degenerate_novelty_players_counts_are_correct():
    counts = {
        (1.0, 1.0): 0,
        (0.0, 1.0): 1,
        (1.0, 0.0): 1,
        (0.0, 0.0): 2,
    }
    for (src_dim1, dst_dim1), expected in counts.items():
        result = _run_scenario(src_dim1=src_dim1, dst_dim1=dst_dim1)
        assert result["n_degenerate_novelty_players"] == expected, (
            f"src_dim1={src_dim1}, dst_dim1={dst_dim1}: "
            f"expected {expected}, got {result['n_degenerate_novelty_players']}"
        )


# ---------------------------------------------------------------------------
# T6 -- coalition_size keeps its full-width (2 + M) meaning, unaffected by
# the guard -- regression guard for explore/node_shap_convergence.py's
# P = len(node_shap) + 2 formula's other operand.
# ---------------------------------------------------------------------------

def test_t6_coalition_size_stays_full_width_when_degenerate():
    result = _run_scenario(src_dim1=0.0, dst_dim1=0.0)  # 2 non-target nodes, both targets degenerate
    M = len(result["_non_target_ids"])
    assert result["coalition_size"] == 2 + M
    assert result["n_degenerate_novelty_players"] == 2
    # coalition_size must NOT be silently narrowed to the reduced width.
    assert result["coalition_size"] != (2 + M) - result["n_degenerate_novelty_players"]


# ---------------------------------------------------------------------------
# T7 -- efficiency axiom re-verified after a degenerate player is dropped
# and its exact-zero column reinserted post hoc.
# ---------------------------------------------------------------------------

def test_t7_efficiency_axiom_holds_with_a_degenerate_player_dropped():
    """Sigma phi (full-width, with the exact-zero degenerate slot inserted)
    == f_logit - f_baseline. Inserting an exact zero for a column whose
    marginal contribution is provably always zero cannot change the sum --
    asserted explicitly, not merely assumed, since it is the direct test
    that dropping-and-reinserting is a sum-preserving transformation."""
    result = _run_scenario(src_dim1=0.0, dst_dim1=1.0, dst_weight=-2.5)
    phi_sum = (
        result["src_novelty_shap"]
        + result["dst_novelty_shap"]
        + sum(result["node_shap"].values())
    )
    gap = result["f_logit"] - result["f_baseline"]
    assert abs(phi_sum - gap) < EFF_TOL, (
        f"efficiency failed: sum(phi)={phi_sum:.6f}, f_logit-f_baseline={gap:.6f}"
    )


def test_t7_efficiency_axiom_holds_with_no_degenerate_players():
    """Same check on the fully non-degenerate path, as a contrast baseline."""
    result = _run_scenario(src_dim1=1.0, dst_dim1=1.0)
    phi_sum = (
        result["src_novelty_shap"]
        + result["dst_novelty_shap"]
        + sum(result["node_shap"].values())
    )
    gap = result["f_logit"] - result["f_baseline"]
    assert abs(phi_sum - gap) < EFF_TOL, (
        f"efficiency failed: sum(phi)={phi_sum:.6f}, f_logit-f_baseline={gap:.6f}"
    )


# ---------------------------------------------------------------------------
# explore/node_shap_convergence.py -- M_effective tracking (specs/63 sec 6)
# ---------------------------------------------------------------------------

def test_collect_player_counts_tracks_p_and_m_effective_separately(tmp_path):
    """collect_player_counts must return BOTH the conceptual P
    (len(node_shap) + 2, unchanged) and M_effective (P minus
    n_degenerate_novelty_players) per flow, not conflate the two."""
    expl_dir = tmp_path / "explanations"
    cls_dir = expl_dir / "attack"
    cls_dir.mkdir(parents=True)

    # Flow 1: 3 non-target node_shap entries + 1 degenerate target
    # -> P = 3 + 2 = 5, M_effective = 5 - 1 = 4.
    (cls_dir / "1.json").write_text(json.dumps({
        "edge_id": 1,
        "node_shap": {"10": 0.1, "11": 0.2, "12": 0.3},
        "node_ids": [10, 11, 12],
        "n_degenerate_novelty_players": 1,
    }))

    # Flow 2: 2 non-target node_shap entries, no degenerate players
    # -> P = 2 + 2 = 4, M_effective = 4.
    (cls_dir / "2.json").write_text(json.dumps({
        "edge_id": 2,
        "node_shap": {"20": 0.5, "21": -0.1},
        "node_ids": [20, 21],
        "n_degenerate_novelty_players": 0,
    }))

    P, M_eff, per_class, by_eid = collect_player_counts(expl_dir)

    assert sorted(P.tolist()) == [4, 5]
    assert sorted(M_eff.tolist()) == [4, 4]
    assert by_eid == {1: 5, 2: 4}
    assert per_class == {"attack": 2}


def test_collect_player_counts_defaults_missing_degenerate_field_to_zero(tmp_path):
    """A JSON written before specs/63 (no n_degenerate_novelty_players key)
    must be treated as 0 degenerate players -- M_effective == P."""
    expl_dir = tmp_path / "explanations"
    cls_dir = expl_dir / "benign"
    cls_dir.mkdir(parents=True)
    (cls_dir / "1.json").write_text(json.dumps({
        "edge_id": 1,
        "node_shap": {"10": 0.1},
        "node_ids": [10],
    }))
    P, M_eff, _, _ = collect_player_counts(expl_dir)
    assert P.tolist() == [3]
    assert M_eff.tolist() == [3]


def test_is_exhaustive_exactness_determination_uses_m_effective_not_p():
    """A flow with P=5 but M_effective=4 must be judged exact against
    M_effective (2**4-2=14 <= nsamples), never against the conceptual P
    (2**5-2=30 > nsamples) -- the solver only ever fits the reduced,
    M_effective-wide coalition matrix after specs/63's guard drops
    degenerate columns."""
    nsamples = 14
    assert not is_exhaustive(5, nsamples), (
        "P=5 at nsamples=14 is NOT exhaustive by the P-based (wrong) formula"
    )
    assert is_exhaustive(4, nsamples), (
        "M_effective=4 at nsamples=14 IS exhaustive -- this is the count "
        "that must back the exactness determination"
    )


# ---------------------------------------------------------------------------
# Production wiring pin: n_degenerate_novelty_players must actually reach
# the on-disk explanation JSON, not just NodeNoveltySHAP.explain()'s return
# dict. Found empirically this session: shap_gsd.py's ExplanationResult and
# scripts/06_explain.py's _result_to_dict both dropped the field on the
# floor between explain() and the JSON writer, so every on-disk explanation
# JSON silently carried n_degenerate_novelty_players == 0 regardless of the
# guard's real determination -- collect_player_counts' M_effective silently
# degraded to P for every flow. Fixed in this same change; pinned here so a
# future edit cannot reintroduce the drop without failing a test, mirroring
# TestL1RegProductionWiring's convention in test_shap_axioms.py.
# ---------------------------------------------------------------------------

def test_explanation_result_carries_n_degenerate_novelty_players():
    """ExplanationResult must expose the field NodeNoveltySHAP.explain()
    computes, not silently drop it."""
    result = ExplanationResult(
        edge_id=1,
        true_label=0,
        predicted_label=0,
        predicted_proba=np.zeros(2, dtype=np.float32),
        feature_group_names=[],
        feature_group_shap=np.zeros(0),
        neighbor_edge_ids=[],
        neighbor_timestamps=[],
        neighbor_shap=np.zeros(0),
        node_ids=[],
        node_shap=np.zeros(0),
        src_novelty_shap=0.0,
        dst_novelty_shap=0.0,
        n_degenerate_novelty_players=2,
        subgraph_edge_ids=[],
        subgraph_shap_weights=[],
    )
    assert result.n_degenerate_novelty_players == 2


def test_phase6_result_to_dict_serializes_n_degenerate_novelty_players():
    """scripts/06_explain.py's JSON serializer (the function
    explore/node_shap_convergence.py's collect_player_counts ultimately
    reads output from) must include n_degenerate_novelty_players -- not
    just accept it on the dataclass."""
    phase6 = _load_phase6_module()
    result = ExplanationResult(
        edge_id=42,
        true_label=1,
        predicted_label=1,
        predicted_proba=np.array([0.1, 0.9], dtype=np.float32),
        feature_group_names=["g0"],
        feature_group_shap=np.array([0.5]),
        neighbor_edge_ids=[7],
        neighbor_timestamps=[123.0],
        neighbor_shap=np.array([0.2]),
        node_ids=[300],
        node_shap=np.array([0.1]),
        src_novelty_shap=0.0,
        dst_novelty_shap=1.3,
        n_degenerate_novelty_players=1,
        subgraph_edge_ids=[7],
        subgraph_shap_weights=[0.2],
    )
    d = phase6._result_to_dict(result)
    assert "n_degenerate_novelty_players" in d, (
        "n_degenerate_novelty_players missing from the serialized explanation "
        "JSON -- explore/node_shap_convergence.py's collect_player_counts "
        "would silently read the d.get(..., 0) default for every flow"
    )
    assert d["n_degenerate_novelty_players"] == 1


# ===========================================================================
# specs/65 -- OUTPUT-DUMMY guard
#
# The specs/63 guard above is an INPUT-side check: it drops a target novelty
# player only when its masked and unmasked coalition inputs are bit-identical.
# specs/65 sec 1 then records a strictly larger failure class, measured on
# all four trained checkpoints: a player whose input genuinely varies (the endpoint
# really is novel, dim 1 really flips 1 -> 0) but whose weight column has been
# driven to 1e-24..1e-40 by Adam's weight_decay, leaving the model bit-
# invariant to it. Its true Shapley value is 0 by the dummy-player axiom, yet
# KernelSHAP fits noise onto its column whenever 2**P > nsamples -- which is
# the whole of CICIDS2018's retired "35/600 phi_N fires" result.
#
# These tests use the same additive fake-model harness as the specs/63 tests
# above; a zero weight on a novelty player reproduces exactly the dead-weight-
# column condition, deterministically and without a real checkpoint.
# ===========================================================================


class _MultiplicativeNoveltyModel(torch.nn.Module):
    """logit = coef * nf[0, _NOVELTY_DIM] * gate(nf[2, 0]).

    Deliberately NON-additive, so a novelty player's marginal effect depends
    on the rest of the coalition. Used to pin the two-anchor probe rule: a
    player that is inert at ONE anchor but live at the other must not be
    dropped.
    """

    def __init__(self, coef, num_classes, true_class, gate):
        super().__init__()
        self.coef = coef
        self.num_classes = num_classes
        self.true_class = true_class
        self.gate = gate

    def forward(self, blocks, nf_t, x_e_t, src_pos, dst_pos):
        val = self.coef * float(nf_t[0, _NOVELTY_DIM]) * self.gate(float(nf_t[2, 0]))
        out = torch.zeros((1, self.num_classes), dtype=torch.float32)
        out[0, self.true_class] = val
        return out


def _run_multiplicative(gate, nsamples: int = 512) -> dict:
    """One NodeNoveltySHAP.explain() against _MultiplicativeNoveltyModel.

    src novelty is native 1.0 (NOT input-degenerate); the single non-target
    node's dim 0 is 5.0 natively and 0.0 in the background, so the all-present
    and all-absent anchors see different gate values.
    """
    src_nid, dst_nid, nt_nid = 100, 200, 300
    input_node_ids = np.array([src_nid, dst_nid, nt_nid], dtype=np.int64)
    base_node_feats = np.zeros((3, 15), dtype=np.float32)
    base_node_feats[0, _NOVELTY_DIM] = 1.0   # src novel -> input varies
    base_node_feats[1, _NOVELTY_DIM] = 0.0   # dst degenerate (specs/63 drops it)
    base_node_feats[2, 0] = 5.0

    model = _MultiplicativeNoveltyModel(3.0, 2, 0, gate)
    bg_node_state = np.zeros((2, 15), dtype=np.float32)  # background dim0 = 0.0
    background = BackgroundDistributions(
        background_features=np.zeros((2, 4), dtype=np.float32),
        background_node_state=bg_node_state,
    )
    return NodeNoveltySHAP(background, torch.device("cpu")).explain(
        true_class=0, src_nid=src_nid, dst_nid=dst_nid, model=model, blocks=[],
        input_nodes=torch.tensor(input_node_ids, dtype=torch.int64),
        base_node_feats=base_node_feats,
        src_pos=torch.tensor([0]), dst_pos=torch.tensor([1]),
        x_e=np.zeros(4, dtype=np.float32), nsamples=nsamples,
    )


# ---------------------------------------------------------------------------
# D1 -- the new case: input VARIES, output does NOT respond -> dropped
# ---------------------------------------------------------------------------

def test_d1_dead_weight_column_player_is_dropped_as_output_dummy():
    """src is genuinely novel (native dim1 = 1.0, so the specs/63 guard does
    NOT fire) but the model's weight on that input is 0 -- exactly the
    trained-away column specs/62 sec 1 measured. phi must be an exact 0.0
    from the dummy-player axiom, not an estimated near-zero."""
    result = _run_scenario(src_dim1=1.0, dst_dim1=1.0, src_weight=0.0)
    assert result["src_novelty_shap"] == 0.0
    assert result["n_degenerate_novelty_players"] == 0, (
        "input is NOT degenerate here -- specs/63's guard must not claim this one"
    )
    assert result["n_dummy_novelty_players"] == 1
    # The genuinely live dst player is untouched.
    assert result["dst_novelty_shap"] != 0.0


def test_d2_both_novelty_players_output_dummy():
    result = _run_scenario(
        src_dim1=1.0, dst_dim1=1.0, src_weight=0.0, dst_weight=0.0,
    )
    assert result["src_novelty_shap"] == 0.0
    assert result["dst_novelty_shap"] == 0.0
    assert result["n_degenerate_novelty_players"] == 0
    assert result["n_dummy_novelty_players"] == 2
    # Non-target node players still solved correctly around the dropped columns.
    for nid, expected in result["_expected_nt_phi"].items():
        assert abs(result["node_shap"][nid] - expected) < EFF_TOL


# ---------------------------------------------------------------------------
# D3 -- the over-broadening guard: a genuinely LIVE player must survive
# ---------------------------------------------------------------------------

def test_d3_live_player_is_not_dropped():
    """Input varies AND output responds: neither guard may fire. This is the
    test that proves the new check has not been broadened into an
    always-zero, which would silently null out every phi_N."""
    src_weight = 3.0
    result = _run_scenario(src_dim1=1.0, dst_dim1=1.0, src_weight=src_weight)
    assert result["n_degenerate_novelty_players"] == 0
    assert result["n_dummy_novelty_players"] == 0
    assert abs(result["src_novelty_shap"] - src_weight) < EFF_TOL


def test_d3_small_but_live_response_is_not_dropped_on_a_tolerance():
    """A tiny -- but genuinely non-zero -- output response must NOT be
    dropped. The guard tests exact equality on two passes of the SAME model,
    deliberately: dropping on a tolerance would discard real attribution."""
    result = _run_scenario(
        src_dim1=1.0, dst_dim1=1.0, src_weight=1e-5, bias=0.0,
        non_target_dim0_vals=(), non_target_weights=(),
    )
    assert result["n_dummy_novelty_players"] == 0
    assert result["src_novelty_shap"] != 0.0


# ---------------------------------------------------------------------------
# D4 -- disjointness: an input-degenerate player is NOT also counted as dummy
# ---------------------------------------------------------------------------

def test_d4_degenerate_and_dummy_counts_are_disjoint():
    """A player caught by specs/63 is trivially output-dummy too, but must be
    reported only once, under n_degenerate_novelty_players. Folding it into
    the dummy count would destroy `2 - n_degenerate_novelty_players` as the
    flow's novel-endpoint count, which downstream analyses read."""
    result = _run_scenario(src_dim1=0.0, dst_dim1=0.0)
    assert result["n_degenerate_novelty_players"] == 2
    assert result["n_dummy_novelty_players"] == 0


def test_d4_mixed_one_degenerate_one_dummy():
    """The real CICIDS2018 shape: dst input-degenerate, src novel but dead in
    the model. Both counts are 1, and both phis are exact zeros."""
    result = _run_scenario(src_dim1=1.0, dst_dim1=0.0, src_weight=0.0)
    assert result["n_degenerate_novelty_players"] == 1
    assert result["n_dummy_novelty_players"] == 1
    assert result["src_novelty_shap"] == 0.0
    assert result["dst_novelty_shap"] == 0.0


# ---------------------------------------------------------------------------
# D5 -- two-anchor probe rule (a single-anchor screen would over-drop)
# ---------------------------------------------------------------------------

def test_d5_player_inert_at_all_absent_but_live_at_all_present_survives():
    """gate(x) = x: at the all-absent anchor the non-target node is at its
    background (dim0 = 0), so the gate is 0 and toggling src novelty changes
    nothing. At the all-present anchor the gate is 5 and it changes plenty.
    A single-anchor screen at all-absent would wrongly drop this live
    player; the two-anchor rule must keep it."""
    result = _run_multiplicative(gate=lambda x: x)
    assert result["n_dummy_novelty_players"] == 0
    assert result["src_novelty_shap"] != 0.0


def test_d5_player_inert_at_all_present_but_live_at_all_absent_survives():
    """The mirror case: gate(x) = 5 - x is 0 at the all-present anchor and 5
    at the all-absent one, so a single-anchor screen at all-present would
    wrongly drop it."""
    result = _run_multiplicative(gate=lambda x: 5.0 - x)
    assert result["n_dummy_novelty_players"] == 0
    assert result["src_novelty_shap"] != 0.0


def test_d5_player_inert_at_both_anchors_is_dropped():
    """gate(x) = 0 everywhere -> inert at both anchors -> dropped, confirming
    the two-anchor rule still fires when it should."""
    result = _run_multiplicative(gate=lambda x: 0.0)
    assert result["n_dummy_novelty_players"] == 1
    assert result["src_novelty_shap"] == 0.0


# ---------------------------------------------------------------------------
# D6 -- structural invariants preserved alongside the new guard
# ---------------------------------------------------------------------------

def test_d6_coalition_size_stays_full_width_when_dummy():
    """coalition_size keeps its conceptual 2 + M meaning, exactly as under
    specs/63 -- explore/node_shap_convergence.py's P formula depends on it."""
    result = _run_scenario(src_dim1=1.0, dst_dim1=1.0, src_weight=0.0, dst_weight=0.0)
    M = len(result["_non_target_ids"])
    assert result["coalition_size"] == 2 + M
    assert result["n_dummy_novelty_players"] == 2


def test_d6_efficiency_axiom_holds_with_a_dummy_player_dropped():
    """Sigma phi (full width, with the exact-zero dummy slot reinserted) ==
    f_logit - f_baseline: dropping and reinserting an exact zero for a
    player that cannot move the output is sum-preserving."""
    result = _run_scenario(src_dim1=1.0, dst_dim1=1.0, src_weight=0.0)
    phi_sum = (
        result["src_novelty_shap"]
        + result["dst_novelty_shap"]
        + sum(result["node_shap"].values())
    )
    gap = result["f_logit"] - result["f_baseline"]
    assert abs(phi_sum - gap) < EFF_TOL, (
        f"efficiency failed: sum(phi)={phi_sum:.6f}, f_logit-f_baseline={gap:.6f}"
    )


def test_d6_screen_novelty_players_is_directly_callable_and_disjoint():
    """The screen is a public method so the guard can be replayed against a
    real checkpoint without re-running the solver (specs/65 sec 5). Pin its
    contract: *_dummy is never True when the matching *_degenerate is."""
    background = BackgroundDistributions(
        background_features=np.zeros((2, 4), dtype=np.float32),
        background_node_state=np.zeros((2, 15), dtype=np.float32),
    )
    screener = NodeNoveltySHAP(background, torch.device("cpu"))
    base = np.zeros((3, 15), dtype=np.float32)
    base[0, _NOVELTY_DIM] = 1.0   # src novel
    base[1, _NOVELTY_DIM] = 0.0   # dst degenerate
    calls: list[np.ndarray] = []

    def _forward(full_row):
        calls.append(full_row.copy())
        return 0.0  # a model that responds to nothing

    src_deg, dst_deg, src_dum, dst_dum = screener.screen_novelty_players(
        forward_full=_forward,
        base_node_feats=base,
        input_node_ids=np.array([100, 200, 300], dtype=np.int64),
        src_nid=100, dst_nid=200, coalition_size=3,
    )
    assert (src_deg, dst_deg) == (False, True)
    assert (src_dum, dst_dum) == (True, False), "dst is already degenerate"
    # Exactly one player is screened here (dst is already input-degenerate),
    # so the bound is 2 anchors + up to 2 probes. Do NOT relax this to 6 --
    # 6 is the bound only when BOTH players are screened, which is
    # test_d6_guard_cost_bound_with_both_players_screened below.
    assert len(calls) <= 4, f"guard cost regression: {len(calls)} forward passes"


def test_d6_guard_cost_bound_with_both_players_screened():
    """Worst case is 2 anchors + 2 probes PER screened player = 6, not 4.
    Pins the performance contract at the widest point so the spec's cost
    figure cannot silently drift back to a single-player bound."""
    background = BackgroundDistributions(
        background_features=np.zeros((2, 4), dtype=np.float32),
        background_node_state=np.zeros((2, 15), dtype=np.float32),
    )
    screener = NodeNoveltySHAP(background, torch.device("cpu"))
    base = np.zeros((3, 15), dtype=np.float32)
    base[0, _NOVELTY_DIM] = 1.0   # src novel  -> screened
    base[1, _NOVELTY_DIM] = 1.0   # dst novel  -> screened
    calls: list[int] = []

    def _forward(full_row):
        calls.append(1)
        return 0.0  # responds to nothing -> no early break, worst case

    src_deg, dst_deg, src_dum, dst_dum = screener.screen_novelty_players(
        forward_full=_forward,
        base_node_feats=base,
        input_node_ids=np.array([100, 200, 300], dtype=np.int64),
        src_nid=100, dst_nid=200, coalition_size=3,
    )
    assert (src_deg, dst_deg) == (False, False)
    assert (src_dum, dst_dum) == (True, True)
    assert len(calls) == 6, f"expected the 6-pass worst case, got {len(calls)}"


def test_d6_reduced_size_zero_is_reachable_via_the_dummy_guard():
    """specs/63's reduced_size == 0 branch was reachable only when BOTH
    targets were input-degenerate. Guard 2 opens two more ways in (both
    dummy, or one of each), and that branch asserts f_baseline == f_logit
    without evaluating the all-absent row. At M == 0 the coalition space is
    only 4 points and the guard's anchors+probes cover all of them, so the
    assertion still holds -- pin it rather than trust the comment."""
    both_dummy = _run_scenario(
        src_dim1=1.0, dst_dim1=1.0, src_weight=0.0, dst_weight=0.0,
        non_target_dim0_vals=(), non_target_weights=(),
    )
    assert both_dummy["n_degenerate_novelty_players"] == 0
    assert both_dummy["n_dummy_novelty_players"] == 2
    assert both_dummy["node_shap"] == {}
    assert both_dummy["f_baseline"] == both_dummy["f_logit"]
    assert both_dummy["src_novelty_shap"] == 0.0
    assert both_dummy["dst_novelty_shap"] == 0.0

    one_each = _run_scenario(
        src_dim1=1.0, dst_dim1=0.0, src_weight=0.0,
        non_target_dim0_vals=(), non_target_weights=(),
    )
    assert one_each["n_degenerate_novelty_players"] == 1
    assert one_each["n_dummy_novelty_players"] == 1
    assert one_each["f_baseline"] == one_each["f_logit"]


def test_d6_self_loop_live_player_survives_the_two_anchor_screen():
    """src_nid == dst_nid. build_masked_node_states dispatches on if/elif,
    so the src branch shadows the dst branch on the shared node row: at the
    ALL-ABSENT anchor, flipping either novelty column is a no-op on the
    input, and a single-anchor screen there would null out both players on
    every self-loop flow. The all-present anchor distinguishes them, so the
    two-anchor rule must keep this genuinely live player."""
    nid = 100
    base = np.zeros((1, 15), dtype=np.float32)
    base[0, _NOVELTY_DIM] = 1.0  # genuinely novel endpoint

    model = _LinearNoveltyModel([4.0], [_NOVELTY_DIM], 2, 0, bias=0.5)
    background = BackgroundDistributions(
        background_features=np.zeros((2, 4), dtype=np.float32),
        background_node_state=np.zeros((2, 15), dtype=np.float32),
    )
    result = NodeNoveltySHAP(background, torch.device("cpu")).explain(
        true_class=0, src_nid=nid, dst_nid=nid, model=model, blocks=[],
        input_nodes=torch.tensor([nid], dtype=torch.int64),
        base_node_feats=base,
        src_pos=torch.tensor([0]), dst_pos=torch.tensor([0]),
        x_e=np.zeros(4, dtype=np.float32), nsamples=512,
    )
    assert result["n_degenerate_novelty_players"] == 0
    assert result["n_dummy_novelty_players"] == 0, (
        "a live self-loop novelty player was screened out -- the all-present "
        "anchor is the only thing standing between this and a silent null"
    )
    # Symmetric AND game: f(1,1)=4.5, every other coalition 0.5 -> phi = 2.0 each.
    assert abs(result["src_novelty_shap"] - 2.0) < EFF_TOL
    assert abs(result["dst_novelty_shap"] - 2.0) < EFF_TOL
    gap = result["f_logit"] - result["f_baseline"]
    phi_sum = result["src_novelty_shap"] + result["dst_novelty_shap"]
    assert abs(phi_sum - gap) < EFF_TOL


def test_d6_guard_costs_zero_forward_passes_when_both_input_degenerate():
    """The specs/63 path must stay free: when neither player survives guard
    1 there is nothing to probe, so guard 2 must not touch the model at
    all. This is the 556/600 CICIDS2018 case and the whole of UNSW."""
    background = BackgroundDistributions(
        background_features=np.zeros((2, 4), dtype=np.float32),
        background_node_state=np.zeros((2, 15), dtype=np.float32),
    )
    screener = NodeNoveltySHAP(background, torch.device("cpu"))
    base = np.zeros((3, 15), dtype=np.float32)  # both dim 1 already 0.0
    calls: list[int] = []

    def _forward(full_row):
        calls.append(1)
        return 0.0

    result = screener.screen_novelty_players(
        forward_full=_forward,
        base_node_feats=base,
        input_node_ids=np.array([100, 200, 300], dtype=np.int64),
        src_nid=100, dst_nid=200, coalition_size=3,
    )
    assert result == (True, True, False, False)
    assert calls == [], f"guard 2 ran {len(calls)} forward passes for nothing"


# ---------------------------------------------------------------------------
# D7 -- explore/node_shap_convergence.py: M_effective subtracts BOTH counts
# ---------------------------------------------------------------------------

def test_d7_collect_player_counts_subtracts_both_guard_counts(tmp_path):
    expl_dir = tmp_path / "explanations"
    cls_dir = expl_dir / "attack"
    cls_dir.mkdir(parents=True)
    # P = 3 + 2 = 5; 1 degenerate + 1 dummy -> M_effective = 3.
    (cls_dir / "1.json").write_text(json.dumps({
        "edge_id": 1,
        "node_shap": {"10": 0.1, "11": 0.2, "12": 0.3},
        "node_ids": [10, 11, 12],
        "n_degenerate_novelty_players": 1,
        "n_dummy_novelty_players": 1,
    }))
    P, M_eff, _, _ = collect_player_counts(expl_dir)
    assert P.tolist() == [5]
    assert M_eff.tolist() == [3]


def test_d7_collect_player_counts_defaults_missing_dummy_field_to_zero(tmp_path):
    """A JSON written before specs/65 (specs/63-era, so it carries the
    degenerate field but not the dummy one) must still subtract only what it
    knows about -- never KeyError, never over-subtract."""
    expl_dir = tmp_path / "explanations"
    cls_dir = expl_dir / "benign"
    cls_dir.mkdir(parents=True)
    (cls_dir / "1.json").write_text(json.dumps({
        "edge_id": 1,
        "node_shap": {"10": 0.1, "11": 0.2},
        "node_ids": [10, 11],
        "n_degenerate_novelty_players": 1,
    }))
    P, M_eff, _, _ = collect_player_counts(expl_dir)
    assert P.tolist() == [4]
    assert M_eff.tolist() == [3]


# ---------------------------------------------------------------------------
# D8 -- production wiring pin, END TO END to a file on disk.
#
# specs/63 computed n_degenerate_novelty_players correctly but dropped it
# between explain() and the JSON writer; that bug was caught only at the TEST
# stage. The dict-level assertion alone did not exist then. Here the field is
# asserted on the bytes actually written to disk by the same json.dump call
# scripts/06_explain.py uses, so the identical mistake cannot recur.
# ---------------------------------------------------------------------------

def _dummy_result(n_degenerate: int, n_dummy: int) -> ExplanationResult:
    return ExplanationResult(
        edge_id=99,
        true_label=1,
        predicted_label=1,
        predicted_proba=np.array([0.2, 0.8], dtype=np.float32),
        feature_group_names=["g0"],
        feature_group_shap=np.array([0.5]),
        neighbor_edge_ids=[7],
        neighbor_timestamps=[123.0],
        neighbor_shap=np.array([0.2]),
        node_ids=[300],
        node_shap=np.array([0.1]),
        src_novelty_shap=0.0,
        dst_novelty_shap=0.0,
        n_degenerate_novelty_players=n_degenerate,
        n_dummy_novelty_players=n_dummy,
        subgraph_edge_ids=[7],
        subgraph_shap_weights=[0.2],
    )


def test_d8_explanation_result_carries_n_dummy_novelty_players():
    assert _dummy_result(1, 1).n_dummy_novelty_players == 1


def test_d8_n_dummy_novelty_players_reaches_the_written_json(tmp_path):
    """The field must survive ExplanationResult -> _result_to_dict ->
    json.dump -> disk -> json.load. Round-tripped through a real file, not
    just asserted on the intermediate dict -- that is where the specs/63
    wiring bug lived. (The explain() -> ExplanationResult hop upstream is
    not re-tested here: shap_gsd.py reads node_result["..."] by key, so a
    dropped field raises KeyError rather than defaulting silently.)"""
    phase6 = _load_phase6_module()
    out_path = tmp_path / "99.json"
    with open(out_path, "w") as f:
        json.dump(phase6._result_to_dict(_dummy_result(1, 1)), f)

    on_disk = json.loads(out_path.read_text())
    assert "n_dummy_novelty_players" in on_disk, (
        "n_dummy_novelty_players never reached the on-disk explanation JSON "
        "-- exactly the specs/63 wiring bug, repeated"
    )
    assert on_disk["n_dummy_novelty_players"] == 1
    # Both counts must land, and stay distinguishable from each other.
    assert on_disk["n_degenerate_novelty_players"] == 1
    assert "n_degenerate_novelty_players" in on_disk


def test_d8_written_json_is_readable_by_collect_player_counts(tmp_path):
    """Closes the loop: the JSON scripts/06_explain.py writes is exactly what
    explore/node_shap_convergence.py's M_effective consumer reads."""
    phase6 = _load_phase6_module()
    cls_dir = tmp_path / "explanations" / "attack"
    cls_dir.mkdir(parents=True)
    with open(cls_dir / "99.json", "w") as f:
        json.dump(phase6._result_to_dict(_dummy_result(1, 1)), f)

    P, M_eff, _, _ = collect_player_counts(tmp_path / "explanations")
    assert P.tolist() == [3]        # 1 non-target node + 2 novelty flags
    assert M_eff.tolist() == [1]    # minus 1 degenerate, minus 1 dummy
