"""
Tests for the input-degeneracy guard in
``src.explainer.feature_shap.FeatureGroupSHAP`` (specs/67).

This mirrors ``tests/test_node_shap_degenerate_guard.py``'s specs/63 section
(T1-T7) in structure, driving the REAL ``FeatureGroupSHAP.explain`` end to end
through a lightweight additive fake model (no real GNN, no GPU, no dataset),
so every group's true Shapley value is known analytically -- exactly the
``_LinearNoveltyModel`` pattern that file uses for phi_N.

specs/67 sec 1.1: a feature group ``g`` is an input-degenerate coalition
player for a given explained flow whenever
``x_e[idxs_g] == bg_feat[idxs_g]`` elementwise, exactly -- because then
masking group ``g`` to the background is a bit-level no-op, so its TRUE
Shapley value is exactly 0 by construction. ``FeatureGroupSHAP.explain``
drops such groups from the KernelSHAP coalition matrix entirely (never
zeroing a fitted coefficient after the fact) and reinserts an exact 0.0 for
each in the reassembled full-width phi vector.

Because feature-group index sets are pairwise disjoint (feature_groups.py's
``FeatureGrouping.validate``), the underlying game here is additive across
ALL d_e raw feature dimensions -- so, exactly like the node-novelty fake
model, the true Shapley value of a group is simply
``sum_i weight[i] * (x_e[i] - bg_feat[i])`` over the group's own indices,
independent of coalition ordering and unaffected by which OTHER groups are
dropped.
"""

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.explainer.background import BackgroundDistributions
from src.explainer.feature_shap import FeatureGroupSHAP
from src.explainer.shap_gsd import ExplanationResult, SHAPGSDExplainer

EFF_TOL = 1e-4  # small coalition spaces (K <= 4) -> KernelSHAP enumerates
                # exhaustively, so the analytic value should match tightly


def _load_module(name: str, rel_path: str):
    """Import a scripts/*.py module whose filename is not a valid identifier,
    mirroring tests/test_node_shap_degenerate_guard.py's
    ``_load_phase6_module`` convention."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_phase6_module():
    return _load_module("phase06_explain", "scripts/06_explain.py")


def _load_phase8_module():
    return _load_module("phase08_metrics", "scripts/08_metrics.py")


# ---------------------------------------------------------------------------
# Fake model: logit[0, true_class] = bias + sum_i weight[i] * masked_t[0, i]
#
# A purely linear/additive function of the (masked) FULL d_e-dim edge feature
# vector, so the ground-truth Shapley value for each GROUP player is known
# analytically -- phi_g = sum_{i in group g} weight[i] * (x_e[i] - bg[i]),
# independent of ordering and of which other groups are dropped (same
# reasoning _LinearNoveltyModel relies on in test_node_shap_degenerate_guard.py).
# ---------------------------------------------------------------------------

class _LinearFeatureModel(torch.nn.Module):
    def __init__(self, weights, num_classes, true_class, bias=0.0, record_calls=False):
        super().__init__()
        self.weights = np.asarray(weights, dtype=np.float64)
        self.num_classes = num_classes
        self.true_class = true_class
        self.bias = bias
        self.record_calls = record_calls
        self.calls: list[torch.Tensor] = []

    def encode(self, blocks, node_feats):
        return None  # unused by classify below; kept for interface parity

    def classify(self, h_fixed, masked_t, src_pos, dst_pos):
        if self.record_calls:
            self.calls.append(masked_t.detach().clone())
        row = masked_t[0].detach().cpu().numpy().astype(np.float64)
        val = self.bias + float(np.dot(self.weights, row))
        out = torch.zeros((1, self.num_classes), dtype=torch.float32)
        out[0, self.true_class] = val
        return out


# ---------------------------------------------------------------------------
# Shared 4-group / 6-dim fixture used by T1-T7 and T5.
#
#   gA: indices [0, 1]      weights [3.0, -1.0]
#   gB: indices [2]         weights [2.0]
#   gC: indices [3, 4]      weights [1.0, 1.5]
#   gD: indices [5]         weights [-2.0]
# ---------------------------------------------------------------------------

_D_E = 6
_WEIGHTS = np.array([3.0, -1.0, 2.0, 1.0, 1.5, -2.0])
_GROUPS = {
    "gA": {"indices": [0, 1], "type": "numeric"},
    "gB": {"indices": [2], "type": "numeric"},
    "gC": {"indices": [3, 4], "type": "categorical"},
    "gD": {"indices": [5], "type": "numeric"},
}
_GROUP_NAMES = list(_GROUPS.keys())
_K = len(_GROUP_NAMES)
_NUM_CLASSES = 2
_TRUE_CLASS = 0
_BIAS = 0.5


def _feature_groups_dict() -> dict:
    return {
        "d_e": _D_E,
        "K": _K,
        "feature_names": [f"f{i}" for i in range(_D_E)],
        "groups": _GROUPS,
    }


def _make_background(bg_row: np.ndarray) -> BackgroundDistributions:
    bg_features = np.tile(bg_row.astype(np.float32), (_NUM_CLASSES, 1))
    return BackgroundDistributions(
        background_features=bg_features,
        background_node_state=np.zeros((_NUM_CLASSES, 15), dtype=np.float32),
    )


def _analytic_phi(x_e: np.ndarray, bg_row: np.ndarray) -> dict[str, float]:
    return {
        name: float(sum(
            _WEIGHTS[i] * (x_e[i] - bg_row[i]) for i in _GROUPS[name]["indices"]
        ))
        for name in _GROUP_NAMES
    }


def _run(
    x_e: np.ndarray,
    bg_row: np.ndarray,
    weights=None,
    bias: float = _BIAS,
    true_class: int = _TRUE_CLASS,
    nsamples: int = 256,
    record_calls: bool = False,
) -> tuple[dict, float, float, list[str], "_LinearFeatureModel"]:
    """Build a FeatureGroupSHAP + fake model pair and run explain() once."""
    background = _make_background(bg_row)
    explainer = FeatureGroupSHAP(_feature_groups_dict(), background, torch.device("cpu"))
    model = _LinearFeatureModel(
        weights if weights is not None else _WEIGHTS,
        _NUM_CLASSES, true_class, bias=bias, record_calls=record_calls,
    )
    phi_dict, f_baseline, f_logit, degenerate = explainer.explain(
        true_class=true_class,
        model=model,
        blocks=[],
        node_feats=torch.zeros((1, 15), dtype=torch.float32),
        x_e=x_e.astype(np.float32),
        src_pos=torch.tensor([0]),
        dst_pos=torch.tensor([0]),
        nsamples=nsamples,
    )
    return phi_dict, f_baseline, f_logit, degenerate, model


# ---------------------------------------------------------------------------
# T1 -- one group input-degenerate: dropped, reports exact 0.0
# ---------------------------------------------------------------------------

def test_t1_degenerate_group_reports_exact_zero_not_estimated_noise():
    """gA's raw dims equal the background exactly for this flow; every other
    group varies. gA's phi must be EXACT 0.0 -- assert with ==, never
    abs(phi) < tol. That distinction ("exact short-circuit" vs "the solver
    happened to fit near zero this run") is the whole point of the fix."""
    bg_row = np.array([1.0, 2.0, -3.0, 0.0, 0.0, 0.0])
    x_e = np.array([1.0, 2.0, 7.0, 4.0, -2.0, 9.0])  # gA == bg; others differ

    phi, f_baseline, f_logit, degenerate, _ = _run(x_e, bg_row)

    assert phi["gA"] == 0.0
    assert degenerate == ["gA"]
    assert len(degenerate) == 1


# ---------------------------------------------------------------------------
# T2 -- fully non-degenerate flow: normal estimation path, unchanged
# ---------------------------------------------------------------------------

def test_t2_fully_non_degenerate_matches_analytic_shapley_values():
    """No group's raw dims equal the background anywhere: confirms the
    full_to_reduced / _expand_row bookkeeping introduces no drift on the
    normal (nothing dropped) path."""
    bg_row = np.array([1.0, 2.0, -3.0, 0.0, 0.0, 0.0])
    x_e = np.array([4.0, -5.0, 7.0, 4.0, -2.0, 9.0])  # every group differs

    phi, f_baseline, f_logit, degenerate, _ = _run(x_e, bg_row)
    expected = _analytic_phi(x_e, bg_row)

    assert degenerate == []
    for name in _GROUP_NAMES:
        assert abs(phi[name] - expected[name]) < EFF_TOL, (
            f"{name}: got {phi[name]}, expected {expected[name]}"
        )


# ---------------------------------------------------------------------------
# T3 -- mixed: some groups degenerate, others live
# ---------------------------------------------------------------------------

def test_t3_mixed_two_degenerate_two_live():
    """gA and gC input-degenerate; gB and gD live. Live groups' phi must
    still match their analytic values -- dropping columns must not perturb
    the survivors' fit."""
    bg_row = np.array([1.0, 2.0, -3.0, 5.0, 6.0, 0.0])
    x_e = np.array([1.0, 2.0, 9.0, 5.0, 6.0, -8.0])  # gA, gC == bg; gB, gD differ

    phi, f_baseline, f_logit, degenerate, _ = _run(x_e, bg_row)
    expected = _analytic_phi(x_e, bg_row)

    assert sorted(degenerate) == ["gA", "gC"]
    assert phi["gA"] == 0.0
    assert phi["gC"] == 0.0
    assert abs(phi["gB"] - expected["gB"]) < EFF_TOL
    assert abs(phi["gD"] - expected["gD"]) < EFF_TOL


def test_t3b_tiny_but_genuine_difference_is_not_dropped():
    """gB differs from its background by exactly 1e-5 in its single index --
    genuinely, if minutely, live. It must NOT be dropped. Ports specs/65's
    D3b/specs/63's tolerance-free philosophy to the feature granularity: no
    tolerance is ever applied, only exact equality."""
    bg_row = np.array([1.0, 2.0, -3.0, 5.0, 6.0, 0.0])
    x_e = bg_row.copy()
    x_e[2] += 1e-5  # gB's single index, tiny but nonzero difference

    phi, f_baseline, f_logit, degenerate, _ = _run(x_e, bg_row)

    assert "gB" not in degenerate
    assert phi["gB"] != 0.0


# ---------------------------------------------------------------------------
# T4 -- efficiency axiom, on both the reduced and the reassembled full vector
# ---------------------------------------------------------------------------

def test_t4_efficiency_axiom_holds_reduced_and_full():
    """sum(phi) == f_logit - f_baseline must hold BOTH on the reduced vector
    the solver actually fit and on the full reassembled vector with exact
    zeros reinserted for the dropped groups -- asserted separately, not
    merely assumed, since this is the direct test that dropping-and-
    reinserting is sum-preserving (specs/67 sec 3.1)."""
    bg_row = np.array([1.0, 2.0, -3.0, 5.0, 6.0, 0.0])
    x_e = np.array([1.0, 2.0, 9.0, 5.0, 6.0, -8.0])  # gA, gC degenerate

    background = _make_background(bg_row)
    explainer = FeatureGroupSHAP(_feature_groups_dict(), background, torch.device("cpu"))
    model = _LinearFeatureModel(_WEIGHTS, _NUM_CLASSES, _TRUE_CLASS, bias=_BIAS)

    phi_dict, f_baseline, f_logit, degenerate = explainer.explain(
        true_class=_TRUE_CLASS, model=model, blocks=[],
        node_feats=torch.zeros((1, 15), dtype=torch.float32),
        x_e=x_e.astype(np.float32),
        src_pos=torch.tensor([0]), dst_pos=torch.tensor([0]),
        nsamples=256,
    )
    phi_full = np.array([phi_dict[n] for n in _GROUP_NAMES])
    gap = f_logit - f_baseline

    # Full-width sum.
    assert abs(phi_full.sum() - gap) < EFF_TOL, (
        f"full-width efficiency failed: sum={phi_full.sum():.6f}, gap={gap:.6f}"
    )
    # Reduced-width sum: the live groups' phis alone (dropped groups
    # contribute exactly 0.0 to either sum, so this must equal the same gap).
    live_names = [n for n in _GROUP_NAMES if n not in degenerate]
    phi_reduced_sum = sum(phi_dict[n] for n in live_names)
    assert abs(phi_reduced_sum - gap) < EFF_TOL, (
        f"reduced-width efficiency failed: sum={phi_reduced_sum:.6f}, gap={gap:.6f}"
    )


# ---------------------------------------------------------------------------
# T5 -- n_degenerate_feature_players / degenerate_feature_groups agreement,
# across several scenarios, in group_names order.
# ---------------------------------------------------------------------------

def test_t5_degenerate_field_values_are_consistent_and_ordered():
    bg_row = np.array([1.0, 2.0, -3.0, 5.0, 6.0, 0.0])
    scenarios = {
        # (x_e) -> expected degenerate group names, in group_names order
        "none": (np.array([4.0, -5.0, 7.0, 9.0, -2.0, 3.0]), []),
        "gA_only": (np.array([1.0, 2.0, 7.0, 9.0, -2.0, 3.0]), ["gA"]),
        "gB_gD": (np.array([4.0, -5.0, -3.0, 9.0, -2.0, 0.0]), ["gB", "gD"]),
        "all": (bg_row.copy(), ["gA", "gB", "gC", "gD"]),
    }
    for label, (x_e, expected) in scenarios.items():
        phi, f_baseline, f_logit, degenerate, _ = _run(x_e, bg_row)
        n_degenerate = len(degenerate)
        assert degenerate == expected, f"{label}: {degenerate} != {expected}"
        assert n_degenerate == len(expected), label
        # group_names order is _GROUP_NAMES's order -- assert directly.
        assert degenerate == [n for n in _GROUP_NAMES if n in expected], label


# ---------------------------------------------------------------------------
# T6a / T6b -- reduced_size boundary cases
# ---------------------------------------------------------------------------

def test_t6a_reduced_size_one_boundary_does_not_raise():
    """3 of 4 groups degenerate -> reduced_size == 1. Pre-fix,
    np.array(phi_raw).squeeze() would yield a 0-d array and the reassembly's
    phi_reduced[reduced_idx] indexing would raise. Must return a correctly
    shaped K-wide dict without raising."""
    bg_row = np.array([1.0, 2.0, -3.0, 5.0, 6.0, 0.0])
    x_e = bg_row.copy()
    x_e[3] = 999.0  # only gC's first index differs -> gC live, rest degenerate
    x_e[4] = bg_row[4]  # keep gC's second index equal so gC is still "live"
    # gC = indices [3, 4]; index 3 differs -> gC as a whole is NOT degenerate
    # (array_equal over both indices fails). gA, gB, gD remain exactly bg.

    phi, f_baseline, f_logit, degenerate, _ = _run(x_e, bg_row)

    assert sorted(degenerate) == ["gA", "gB", "gD"]
    assert set(phi.keys()) == set(_GROUP_NAMES)
    assert phi["gA"] == 0.0
    assert phi["gB"] == 0.0
    assert phi["gD"] == 0.0
    expected_gC = _analytic_phi(x_e, bg_row)["gC"]
    assert abs(phi["gC"] - expected_gC) < EFF_TOL


def test_t6b_reduced_size_zero_boundary_all_degenerate():
    """Every group degenerate -> reduced_size == 0. shap.KernelExplainer must
    never be constructed with a zero-width background; all K phi exactly
    0.0; f_baseline == f_logit exactly (the single achievable coalition
    state is the only one there is)."""
    bg_row = np.array([1.0, 2.0, -3.0, 5.0, 6.0, 0.0])
    x_e = bg_row.copy()

    phi, f_baseline, f_logit, degenerate, _ = _run(x_e, bg_row)

    assert sorted(degenerate) == sorted(_GROUP_NAMES)
    for name in _GROUP_NAMES:
        assert phi[name] == 0.0
    assert f_baseline == f_logit


# ---------------------------------------------------------------------------
# T7 -- simultaneous drop of >=2 degenerate groups leaves `masked`
# bit-identical -- the direct test of specs/67 sec 3.1's disjointness proof.
#
# `masked` is a local inside `_predict_fn`'s closure and is never returned,
# so the fake model's `classify` records every masked_t tensor it receives;
# the assertion is made on those CAPTURED tensors, never on model outputs
# (identical outputs cannot distinguish "input was bit-identical" from
# "input differed and the output happened to coincide").
# ---------------------------------------------------------------------------

def test_t7_simultaneous_drop_leaves_masked_bit_identical_across_coalitions():
    bg_row = np.array([1.0, 2.0, -3.0, 5.0, 6.0, 0.0])
    x_e = np.array([1.0, 2.0, 9.0, 5.0, 6.0, -8.0])  # gA, gC degenerate; gB, gD live

    background = _make_background(bg_row)
    explainer = FeatureGroupSHAP(_feature_groups_dict(), background, torch.device("cpu"))
    model = _LinearFeatureModel(
        _WEIGHTS, _NUM_CLASSES, _TRUE_CLASS, bias=_BIAS, record_calls=True,
    )

    explainer.explain(
        true_class=_TRUE_CLASS, model=model, blocks=[],
        node_feats=torch.zeros((1, 15), dtype=torch.float32),
        x_e=x_e.astype(np.float32),
        src_pos=torch.tensor([0]), dst_pos=torch.tensor([0]),
        nsamples=256,
    )

    assert len(model.calls) > 0, "the fake model recorded no masked_t calls at all"

    # Every captured masked_t must agree on indices 0, 1 (gA) and 3, 4 (gC) --
    # the degenerate groups' indices -- regardless of what coalition (gB, gD
    # present/absent) produced that call, because gA/gC's masked value is
    # bit-identical to x_e there no matter what the coalition row says.
    degenerate_idxs = _GROUPS["gA"]["indices"] + _GROUPS["gC"]["indices"]
    reference = model.calls[0][0].numpy()[degenerate_idxs]
    for call in model.calls[1:]:
        this_row = call[0].numpy()[degenerate_idxs]
        assert np.array_equal(this_row, reference), (
            "a degenerate group's masked columns varied across coalitions -- "
            "the disjointness-composition proof (specs/67 sec 3.1) is violated"
        )
    # And the reference itself must equal x_e's own values at those indices
    # (i.e. "present" and "absent" are bit-identical there).
    assert np.array_equal(reference, x_e[degenerate_idxs].astype(np.float32))


# ---------------------------------------------------------------------------
# T8 -- production wiring, end to end, on disk (real file, not the dict).
# ---------------------------------------------------------------------------

def _dummy_result(n_degenerate: int, degenerate_names: list[str]) -> ExplanationResult:
    return ExplanationResult(
        edge_id=99,
        true_label=1,
        predicted_label=1,
        predicted_proba=np.array([0.2, 0.8], dtype=np.float32),
        feature_group_names=_GROUP_NAMES,
        feature_group_shap=np.array([0.0 if n in degenerate_names else 0.5 for n in _GROUP_NAMES]),
        neighbor_edge_ids=[7],
        neighbor_timestamps=[123.0],
        neighbor_shap=np.array([0.2]),
        node_ids=[300],
        node_shap=np.array([0.1]),
        src_novelty_shap=0.0,
        dst_novelty_shap=0.0,
        n_degenerate_feature_players=n_degenerate,
        degenerate_feature_groups=degenerate_names,
        subgraph_edge_ids=[7],
        subgraph_shap_weights=[0.2],
    )


def test_t8_result_reaches_disk_json_bytes_not_the_intermediate_dict(tmp_path):
    """explain() -> ExplanationResult -> _result_to_dict -> json.dump to a
    REAL file -> json.load -> both fields present with the right values.
    specs/63 shipped a computed-but-unwired field caught only at this stage;
    the assertion here is on bytes read back from a file, not the
    intermediate dict."""
    phase6 = _load_phase6_module()
    result = _dummy_result(2, ["gA", "gC"])

    out_path = tmp_path / "99.json"
    with open(out_path, "w") as f:
        json.dump(phase6._result_to_dict(result), f)

    on_disk = json.loads(out_path.read_text())
    assert "n_degenerate_feature_players" in on_disk, (
        "n_degenerate_feature_players never reached the on-disk explanation "
        "JSON -- the specs/63 wiring bug, repeated"
    )
    assert "degenerate_feature_groups" in on_disk
    assert on_disk["n_degenerate_feature_players"] == 2
    assert on_disk["degenerate_feature_groups"] == ["gA", "gC"]


def test_t8_summary_csv_row_carries_n_degenerate_feature_players():
    """scripts/06_explain.py's summary.csv writer must carry
    n_degenerate_feature_players per row (specs/67 sec 8.3). There is no
    standalone function to call for a single row (the dict literal lives
    inline in the stratified-explain loop), so this is a targeted static
    check on the source text of that literal, in the spirit of
    test_shap_axioms.py's TestL1RegProductionWiring source-text checks."""
    phase6 = _load_phase6_module()
    src_text = inspect.getsource(phase6)
    # Find the summary_rows.append({...}) block specifically, not just
    # anywhere in the file, so a match elsewhere cannot produce a false pass.
    start = src_text.index("summary_rows.append({")
    end = src_text.index("})", start)
    block = src_text[start:end]
    assert '"n_degenerate_feature_players": result.n_degenerate_feature_players' in block, (
        "summary.csv's row-construction block no longer carries "
        "n_degenerate_feature_players"
    )


# ---------------------------------------------------------------------------
# T10 -- dtype-aware predicate: a unit test of screen_feature_players'
# CONTRACT, not of a live bug in the current pipeline (both sides are
# float32 today -- see docstring below).
# ---------------------------------------------------------------------------

def test_t10_screen_feature_players_follows_the_post_assignment_cast_contract():
    """Not a bug in current production code: BackgroundDistributions.__init__
    always casts background_features to float32 (background.py:39-41), so no
    float64-vs-float32 disagreement is reachable through the real pipeline.
    This pins the METHOD's contract directly -- constructed by hand, bypassing
    BackgroundDistributions -- so the predicate stays correct if either side's
    dtype ever changes in the future (specs/67 sec 6.2 item 1).

    0.1 is not exactly representable in float32; casting the float64 0.1 down
    to float32 and comparing against a hand-picked float32 value that IS the
    float32 rounding of 0.1 must be judged degenerate, because that is what
    the masking assignment actually stores (`masked[idxs] = bg_feat[idxs]`,
    which implicitly casts to x_e's dtype). A naive cross-dtype `==` would
    disagree, since np.float64(0.1) != np.float32(0.1) bit-for-bit.
    """
    background = BackgroundDistributions(
        background_features=np.zeros((_NUM_CLASSES, _D_E), dtype=np.float32),
        background_node_state=np.zeros((_NUM_CLASSES, 15), dtype=np.float32),
    )
    explainer = FeatureGroupSHAP(_feature_groups_dict(), background, torch.device("cpu"))

    bg_feat_f64 = np.zeros(_D_E, dtype=np.float64)
    bg_feat_f64[2] = 0.1  # gB's single index; not exactly representable in f32

    x_e_f32 = np.zeros(_D_E, dtype=np.float32)
    x_e_f32[2] = np.float32(0.1)  # the float32 rounding of the same value

    # A naive, non-cast comparison would disagree here.
    assert not np.array_equal(x_e_f32[2:3], bg_feat_f64[2:3].astype(np.float64)), (
        "test setup invalid: chosen values already agree without a cast"
    )

    degenerate = explainer.screen_feature_players(x_e_f32, bg_feat_f64)
    assert degenerate[_GROUP_NAMES.index("gB")] is True, (
        "screen_feature_players did not follow the post-assignment stored "
        "value (masked[idxs] = bg_feat[idxs] cast to x_e's dtype)"
    )
    # The other groups (still all-zero on both sides) remain degenerate too.
    for i, name in enumerate(_GROUP_NAMES):
        if name != "gB":
            assert degenerate[i] is True, name


# ---------------------------------------------------------------------------
# T11 -- fidelity-ranking effect: a pure array-level test of the exact
# expression scripts/08_metrics.py:_compute_fidelity uses.
# ---------------------------------------------------------------------------

def test_t11_exact_zero_never_enters_the_top_5_ranking():
    """A degenerate group's phi is exactly 0.0; every other (live) group has
    a nonzero magnitude. np.argsort(np.abs(phi))[::-1][:5] -- the exact
    ranking expression scripts/08_metrics.py's _compute_fidelity uses -- must
    never select the exact-zero index into a top-5 slot when at least 5
    nonzero competitors exist. No model, no pipeline run: pure array logic."""
    K = 10
    degenerate_idx = 3
    phi = np.array([0.9, -0.8, 0.7, 0.0, 0.6, -0.5, 0.4, -0.3, 0.2, -0.1])
    assert phi[degenerate_idx] == 0.0

    top5_idx = np.argsort(np.abs(phi))[::-1][:5]
    assert degenerate_idx not in top5_idx, (
        "the exact structural zero was selected into the fidelity top-5 -- "
        "the ranking correction specs/67 sec 4 depends on is broken"
    )
    # Sanity: the top-5 really are the 5 largest-magnitude entries.
    assert set(top5_idx.tolist()) == {0, 1, 2, 4, 5}


# ---------------------------------------------------------------------------
# T12 -- real-case regression, fast: reconstruct the CICIDS2018 /
# UNSW-NB15 degeneracy shapes synthetically (no checkpoint, no dataset).
# ---------------------------------------------------------------------------

def test_t12_cicids2018_single_column_categorical_constant_one_is_dropped():
    """FTP_COMMAND_RET_CODE shape: a single-index group whose background
    value is exactly 1.0 and whose flow value is exactly 1.0 (constant
    across nearly all training rows and this flow's class)."""
    groups = {"FTP_COMMAND_RET_CODE": {"indices": [0], "type": "categorical"}}
    feature_groups = {"d_e": 1, "K": 1, "feature_names": ["f0"], "groups": groups}
    background = BackgroundDistributions(
        background_features=np.array([[1.0]], dtype=np.float32),
        background_node_state=np.zeros((1, 15), dtype=np.float32),
    )
    explainer = FeatureGroupSHAP(feature_groups, background, torch.device("cpu"))
    model = _LinearFeatureModel([5.0], 1, 0, bias=0.3)

    phi, f_baseline, f_logit, degenerate = explainer.explain(
        true_class=0, model=model, blocks=[],
        node_feats=torch.zeros((1, 15), dtype=torch.float32),
        x_e=np.array([1.0], dtype=np.float32),
        src_pos=torch.tensor([0]), dst_pos=torch.tensor([0]),
        nsamples=64,
    )
    assert degenerate == ["FTP_COMMAND_RET_CODE"]
    assert phi["FTP_COMMAND_RET_CODE"] == 0.0
    assert f_baseline == f_logit


def test_t12_unsw_multi_index_one_hot_group_is_dropped():
    """UNSW shape: a multi-index one-hot group (e.g. PROTOCOL) whose
    background exactly equals the flow's one-hot encoding for this class."""
    groups = {
        "PROTOCOL": {"indices": [0, 1, 2], "type": "categorical"},
        "OTHER": {"indices": [3], "type": "numeric"},
    }
    feature_groups = {"d_e": 4, "K": 2, "feature_names": [f"f{i}" for i in range(4)], "groups": groups}
    bg_row = np.array([0.0, 1.0, 0.0, 5.0], dtype=np.float32)
    x_e = np.array([0.0, 1.0, 0.0, 9.0], dtype=np.float32)  # PROTOCOL == bg; OTHER differs
    background = BackgroundDistributions(
        background_features=np.tile(bg_row, (1, 1)),
        background_node_state=np.zeros((1, 15), dtype=np.float32),
    )
    explainer = FeatureGroupSHAP(feature_groups, background, torch.device("cpu"))
    model = _LinearFeatureModel([1.0, 1.0, 1.0, 2.0], 1, 0, bias=0.0)

    phi, f_baseline, f_logit, degenerate = explainer.explain(
        true_class=0, model=model, blocks=[],
        node_feats=torch.zeros((1, 15), dtype=torch.float32),
        x_e=x_e,
        src_pos=torch.tensor([0]), dst_pos=torch.tensor([0]),
        nsamples=64,
    )
    assert degenerate == ["PROTOCOL"]
    assert phi["PROTOCOL"] == 0.0
    assert abs(phi["OTHER"] - 2.0 * (9.0 - 5.0)) < EFF_TOL


def test_t12_unsw_numeric_single_column_group_is_dropped():
    """UNSW shape (the 'not categorical-only' requirement, specs/67 sec 6.2):
    a NUMERIC single-index group (e.g. TCP_WIN_MAX_IN) at 100% exact-match
    degeneracy for a class must be dropped exactly like a categorical
    one-hot -- the predicate must not branch on groups[name]['type']."""
    groups = {
        "TCP_WIN_MAX_IN": {"indices": [0], "type": "numeric"},
        "OTHER": {"indices": [1], "type": "numeric"},
    }
    feature_groups = {"d_e": 2, "K": 2, "feature_names": ["f0", "f1"], "groups": groups}
    background = BackgroundDistributions(
        background_features=np.array([[65535.0, 3.0]], dtype=np.float32),
        background_node_state=np.zeros((1, 15), dtype=np.float32),
    )
    explainer = FeatureGroupSHAP(feature_groups, background, torch.device("cpu"))
    model = _LinearFeatureModel([0.5, 4.0], 1, 0, bias=0.0)

    phi, f_baseline, f_logit, degenerate = explainer.explain(
        true_class=0, model=model, blocks=[],
        node_feats=torch.zeros((1, 15), dtype=torch.float32),
        x_e=np.array([65535.0, 7.0], dtype=np.float32),  # TCP_WIN_MAX_IN == bg
        src_pos=torch.tensor([0]), dst_pos=torch.tensor([0]),
        nsamples=64,
    )
    assert degenerate == ["TCP_WIN_MAX_IN"]
    assert phi["TCP_WIN_MAX_IN"] == 0.0
    assert abs(phi["OTHER"] - 4.0 * (7.0 - 3.0)) < EFF_TOL


# ---------------------------------------------------------------------------
# T13 -- both call sites unpack four values (static guard).
#
# Without this, a stale 3-target unpack of feat_shap.explain(...) at either
# site raises a tuple-unpacking ValueError at runtime -- the Stability loop
# in particular is not otherwise covered by any test in tests/, so a broken
# unpack there would surface only on a real Phase-8 run (specs/67 sec 7.5).
# ---------------------------------------------------------------------------

def test_t13_shap_gsd_explain_edge_uses_a_four_target_unpack():
    src_text = inspect.getsource(SHAPGSDExplainer.explain_edge)
    assert "= self.feat_shap.explain(" in src_text
    # Locate the assignment line feeding that call and count its targets.
    call_line_start = src_text.index("self.feat_shap.explain(")
    assign_segment = src_text[:call_line_start]
    assign_line = assign_segment.splitlines()[-1]
    targets = assign_line.split("=")[0].strip()
    n_targets = len([t for t in targets.split(",") if t.strip()])
    assert n_targets == 4, (
        f"src/explainer/shap_gsd.py's explain_edge unpacks "
        f"self.feat_shap.explain(...) into {n_targets} targets, expected 4 "
        f"-- got assignment line: {assign_line!r}"
    )
    # And the historical stale 3-target form must not be present anywhere.
    assert "feat_phi_dict, f_baseline_feat, f_logit_feat = self.feat_shap.explain(" not in src_text


def test_t13_phase8_stability_loop_uses_a_four_target_unpack():
    phase8 = _load_phase8_module()
    src_text = inspect.getsource(phase8.compute_stability)
    assert "= feat_shap.explain(" in src_text
    call_line_start = src_text.index("feat_shap.explain(")
    assign_segment = src_text[:call_line_start]
    assign_line = assign_segment.splitlines()[-1]
    targets = assign_line.split("=")[0].strip()
    n_targets = len([t for t in targets.split(",") if t.strip()])
    assert n_targets == 4, (
        f"scripts/08_metrics.py's compute_stability unpacks "
        f"feat_shap.explain(...) into {n_targets} targets, expected 4 -- "
        f"got assignment line: {assign_line!r}"
    )
    # The historical stale 3-target form must not be present anywhere.
    assert "phi_dict, _f_baseline, _f_logit = feat_shap.explain(" not in src_text
