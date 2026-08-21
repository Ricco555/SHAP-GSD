"""
Tests for the mode-agnostic degenerate-toggle guard in
``src.explainer.node_shap.NodeNoveltySHAP.explain`` (specs/63).

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
