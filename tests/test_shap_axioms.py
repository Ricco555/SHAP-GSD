"""
SHAP axiom gate tests — must pass before running Phase 6 explanations.

Tests verify the three core SHAP axioms on toy predict functions that mirror
the coalition structure of each SHAP-GSD granularity:

  1. Efficiency  (local accuracy): sum(φ) ≈ f(x) - f(background)
  2. Symmetry:   swapping identical coalition members gives equal φ
  3. Dummy:      a group/edge with zero marginal contribution gets φ ≈ 0

Each test uses a lightweight linear/additive model so the ground-truth
SHAP values are known analytically. The implementations under test are the
predict_fn patterns used in feature_shap.py, temporal_shap.py, and
node_shap.py — not the full GNN (which requires GPU + loaded graphs).
"""

import inspect

import numpy as np
import pytest
import shap


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
EFFICIENCY_TOL = 0.05   # 5% of |f(x) - f(bg)| tolerance for KernelSHAP
DUMMY_TOL = 0.05        # absolute tolerance for dummy φ


def _kernel_shap(
    predict_fn,
    n_coalition: int,
    nsamples: int = 4096,
    l1_reg: "bool | str" = False,
) -> np.ndarray:
    """Run KernelSHAP with all-zeros background, all-ones foreground.

    l1_reg defaults to False to match production (specs/45, specs/46 sec
    0.3/1-3): shap>=0.47's own default, "num_features(10)", silently
    L1-truncates attributions to at most 10 nonzero players, which none of
    this file's existing <=10-player toy coalitions ever exercised. Callers
    that specifically want to reproduce the pre-fix truncated behavior (see
    TestNoL1RegTruncation) pass l1_reg="num_features(10)" explicitly.
    """
    bg = np.zeros((1, n_coalition), dtype=np.float64)
    explainer = shap.KernelExplainer(predict_fn, bg)
    phi = explainer.shap_values(
        np.ones((1, n_coalition), dtype=np.float64),
        nsamples=nsamples,
        l1_reg=l1_reg,
        silent=True,
    )
    return np.array(phi).squeeze()


# ---------------------------------------------------------------------------
# Granularity 1: Feature-group SHAP axioms
# ---------------------------------------------------------------------------

class TestFeatureGroupSHAPAxioms:
    """Axiom tests for the feature-group coalition pattern."""

    # Toy setup: 6 groups, 2 features each, linear model
    K = 6
    D = 12
    np.random.seed(0)
    _weights = RNG.standard_normal(D).astype(np.float64)
    _x_e = RNG.standard_normal(D).astype(np.float64)
    _bg = np.zeros(D, dtype=np.float64)  # background replaces absent groups

    _groups = {f"g{i}": list(range(i * 2, i * 2 + 2)) for i in range(K)}
    _group_names = [f"g{i}" for i in range(K)]

    def _predict_fn(self, coalition_matrix: np.ndarray) -> np.ndarray:
        results = []
        for row in coalition_matrix:
            masked = self._x_e.copy()
            for i, present in enumerate(row):
                if not present:
                    masked[self._groups[self._group_names[i]]] = self._bg[
                        self._groups[self._group_names[i]]
                    ]
            results.append(float(np.dot(masked, self._weights)))
        return np.array(results)

    def test_efficiency(self):
        """sum(φ) ≈ f(x) - f(background) within tolerance."""
        phi = _kernel_shap(self._predict_fn, self.K)
        f_x  = self._predict_fn(np.ones((1, self.K)))[0]
        f_bg = self._predict_fn(np.zeros((1, self.K)))[0]
        gap = f_x - f_bg
        err = abs(phi.sum() - gap)
        tol = max(EFFICIENCY_TOL * abs(gap), 1e-6)
        assert err < tol, (
            f"Feature SHAP efficiency failed: sum(φ)={phi.sum():.5f}, "
            f"f(x)-f(bg)={gap:.5f}, err={err:.5f}, tol={tol:.5f}"
        )

    def test_dummy(self):
        """A group with zero weight gets φ ≈ 0."""
        # Make group g0 use zero weights → zero marginal contribution
        weights_zero = self._weights.copy()
        weights_zero[0] = 0.0
        weights_zero[1] = 0.0
        x_e_diff = self._x_e.copy()
        x_e_diff[0] = 5.0   # non-zero feature, but weight is zero → no impact

        def predict_fn_zero(coalition_matrix: np.ndarray) -> np.ndarray:
            results = []
            for row in coalition_matrix:
                masked = x_e_diff.copy()
                for i, present in enumerate(row):
                    if not present:
                        masked[self._groups[self._group_names[i]]] = 0.0
                results.append(float(np.dot(masked, weights_zero)))
            return np.array(results)

        phi = _kernel_shap(predict_fn_zero, self.K)
        assert abs(phi[0]) < DUMMY_TOL, (
            f"Feature SHAP dummy failed: φ[g0]={phi[0]:.5f}, expected ≈ 0"
        )

    def test_symmetry(self):
        """Two groups with identical feature values get equal φ."""
        # Make g0 and g1 have the same values and same weight magnitude
        x_sym = self._x_e.copy()
        x_sym[0:2] = 1.0   # g0 features
        x_sym[2:4] = 1.0   # g1 features (identical to g0)
        w_sym = self._weights.copy()
        w_sym[0:2] = 0.5
        w_sym[2:4] = 0.5   # same contribution per unit as g0

        def predict_fn_sym(coalition_matrix: np.ndarray) -> np.ndarray:
            results = []
            for row in coalition_matrix:
                masked = x_sym.copy()
                for i, present in enumerate(row):
                    if not present:
                        masked[self._groups[self._group_names[i]]] = 0.0
                results.append(float(np.dot(masked, w_sym)))
            return np.array(results)

        phi = _kernel_shap(predict_fn_sym, self.K)
        assert abs(phi[0] - phi[1]) < DUMMY_TOL, (
            f"Feature SHAP symmetry failed: φ[g0]={phi[0]:.4f}, φ[g1]={phi[1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Granularity 2: Temporal neighborhood SHAP axioms
# ---------------------------------------------------------------------------

class TestTemporalSHAPAxioms:
    """Axiom tests for the temporal-neighborhood coalition pattern.

    Models the case where each neighbor edge contributes independently to
    the final score (additive surrogate), mirroring the predict_fn structure
    in temporal_shap.py but without requiring the actual GNN.
    """

    N = 5  # toy number of neighbor edges
    _edge_contrib = RNG.standard_normal(N).astype(np.float64)  # ground-truth contributions
    _base_score = 0.0  # f when all neighbors absent

    def _predict_fn(self, coalition_matrix: np.ndarray) -> np.ndarray:
        """Each present edge adds its fixed contribution."""
        return coalition_matrix @ self._edge_contrib + self._base_score

    def test_efficiency(self):
        phi = _kernel_shap(self._predict_fn, self.N)
        f_x  = self._predict_fn(np.ones((1, self.N)))[0]
        f_bg = self._predict_fn(np.zeros((1, self.N)))[0]
        gap = f_x - f_bg
        err = abs(phi.sum() - gap)
        tol = max(EFFICIENCY_TOL * abs(gap), 1e-6)
        assert err < tol, (
            f"Temporal SHAP efficiency failed: sum(φ)={phi.sum():.5f}, "
            f"f(x)-f(bg)={gap:.5f}, err={err:.5f}"
        )

    def test_dummy(self):
        """An edge with zero contribution gets φ ≈ 0."""
        contrib = self._edge_contrib.copy()
        contrib[2] = 0.0  # edge 2 has no effect

        def predict_fn_zero(coalition_matrix: np.ndarray) -> np.ndarray:
            return coalition_matrix @ contrib

        phi = _kernel_shap(predict_fn_zero, self.N)
        assert abs(phi[2]) < DUMMY_TOL, (
            f"Temporal SHAP dummy failed: φ[2]={phi[2]:.5f}"
        )

    def test_symmetry(self):
        """Two edges with identical contribution get equal φ."""
        contrib = self._edge_contrib.copy()
        contrib[0] = 1.0
        contrib[1] = 1.0  # identical to edge 0

        def predict_fn_sym(coalition_matrix: np.ndarray) -> np.ndarray:
            return coalition_matrix @ contrib

        phi = _kernel_shap(predict_fn_sym, self.N)
        assert abs(phi[0] - phi[1]) < DUMMY_TOL, (
            f"Temporal SHAP symmetry failed: φ[0]={phi[0]:.4f}, φ[1]={phi[1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Granularity 3: Node novelty SHAP axioms
# ---------------------------------------------------------------------------

class TestNodeNoveltySHAPAxioms:
    """Axiom tests for the node-novelty coalition pattern."""

    # Coalition: [src_novelty, dst_novelty, node_0, node_1, node_2]
    COALITION_SIZE = 5
    _node_contrib = RNG.standard_normal(COALITION_SIZE).astype(np.float64)

    def _predict_fn(self, coalition_matrix: np.ndarray) -> np.ndarray:
        """Each present coalition member adds its fixed contribution."""
        return coalition_matrix @ self._node_contrib

    def test_efficiency(self):
        phi = _kernel_shap(self._predict_fn, self.COALITION_SIZE)
        f_x  = self._predict_fn(np.ones((1, self.COALITION_SIZE)))[0]
        f_bg = self._predict_fn(np.zeros((1, self.COALITION_SIZE)))[0]
        gap = f_x - f_bg
        err = abs(phi.sum() - gap)
        tol = max(EFFICIENCY_TOL * abs(gap), 1e-6)
        assert err < tol, (
            f"Node SHAP efficiency failed: sum(φ)={phi.sum():.5f}, "
            f"f(x)-f(bg)={gap:.5f}, err={err:.5f}"
        )

    def test_dummy(self):
        """A node with zero contribution gets φ ≈ 0."""
        contrib = self._node_contrib.copy()
        contrib[3] = 0.0  # node_1 has no effect

        def predict_fn_zero(coalition_matrix: np.ndarray) -> np.ndarray:
            return coalition_matrix @ contrib

        phi = _kernel_shap(predict_fn_zero, self.COALITION_SIZE)
        assert abs(phi[3]) < DUMMY_TOL, (
            f"Node SHAP dummy failed: φ[3]={phi[3]:.5f}"
        )

    def test_symmetry(self):
        """Two nodes with identical contribution get equal φ."""
        contrib = self._node_contrib.copy()
        contrib[2] = 0.8
        contrib[3] = 0.8  # identical to node_0

        def predict_fn_sym(coalition_matrix: np.ndarray) -> np.ndarray:
            return coalition_matrix @ contrib

        phi = _kernel_shap(predict_fn_sym, self.COALITION_SIZE)
        assert abs(phi[2] - phi[3]) < DUMMY_TOL, (
            f"Node SHAP symmetry failed: φ[2]={phi[2]:.4f}, φ[3]={phi[3]:.4f}"
        )


# ---------------------------------------------------------------------------
# Regression pin: l1_reg=False must not artificially cap nonzero players
# at 10 when the true signal is spread across more than 10 (specs/45,
# specs/46). None of the classes above ever exercises a coalition size
# > 10, so none of them would have failed against the pre-fix code (shap's
# default l1_reg="num_features(10)" silently zeros all but 10 players
# regardless of true signal spread) -- this class is the dedicated check.
# ---------------------------------------------------------------------------

class TestNoL1RegTruncation:
    """l1_reg=False must recover genuine attribution spread beyond 10 players."""

    # 15 players, each with a distinct nonzero weight -> every player has
    # genuine nonzero marginal contribution. A correct, unregularized fit
    # must report all 15 nonzero; "num_features(10)" would cap at 10.
    #
    # |contrib[i]| >= 1.0 for every i by construction (not by two
    # independent draws that could cancel near zero): sign(z) * (1 + |z|)
    # is bounded away from zero regardless of how small |z| happens to
    # land, so no player's true contribution can coincide with ordinary
    # KernelSHAP Monte Carlo noise at nsamples=4096 -- this test targets
    # artificial capping at exactly 10, not estimator noise near zero.
    P = 15
    _z = RNG.standard_normal(P)
    _contrib = np.sign(_z) * (1.0 + np.abs(_z))

    def _predict_fn(self, coalition_matrix: np.ndarray) -> np.ndarray:
        return coalition_matrix @ self._contrib

    def test_l1_reg_false_recovers_all_nonzero_players(self):
        """With l1_reg=False, all 15 genuinely-contributing players are
        nonzero -- not capped at 10."""
        phi = _kernel_shap(self._predict_fn, self.P, nsamples=4096, l1_reg=False)
        nonzero = np.sum(np.abs(phi) > DUMMY_TOL)
        assert nonzero > 10, (
            f"Expected > 10 nonzero of {self.P} players under l1_reg=False, "
            f"got {nonzero} -- looks truncated. phi={phi}"
        )
        assert nonzero == self.P, (
            f"Expected all {self.P} players nonzero (every player has "
            f"genuine nonzero contribution), got {nonzero}. phi={phi}"
        )

    def test_l1_reg_num_features_10_does_truncate(self):
        """Contrast case, pins the pre-fix defect's actual shape: shap's
        own 'num_features(10)' default caps nonzero players at 10 on this
        exact toy setup, confirming the truncation this fix removes is
        real and reproducible, not a misreading of the shap docstring."""
        phi = _kernel_shap(
            self._predict_fn, self.P, nsamples=4096, l1_reg="num_features(10)"
        )
        nonzero = np.sum(np.abs(phi) > DUMMY_TOL)
        assert nonzero <= 10, (
            f"Expected <= 10 nonzero of {self.P} players under "
            f"l1_reg='num_features(10)' (this shap version's actual "
            f"default), got {nonzero} -- if this now fails, shap's default "
            f"truncation behavior may have changed upstream; re-verify "
            f"specs/45's root-cause citation against the installed shap "
            f"version before treating this as a real regression. phi={phi}"
        )

    def test_efficiency_holds_under_l1_reg_false_at_p_gt_10(self):
        """Efficiency axiom is not affected by disabling l1_reg, including
        at coalition sizes above the truncation threshold (specs/45 sec 2:
        'efficiency axiom ... holds to float precision ... same as today')."""
        phi = _kernel_shap(self._predict_fn, self.P, nsamples=4096, l1_reg=False)
        f_x = self._predict_fn(np.ones((1, self.P)))[0]
        f_bg = self._predict_fn(np.zeros((1, self.P)))[0]
        gap = f_x - f_bg
        err = abs(phi.sum() - gap)
        tol = max(EFFICIENCY_TOL * abs(gap), 1e-6)
        assert err < tol, (
            f"Efficiency failed at P=15 under l1_reg=False: "
            f"sum(φ)={phi.sum():.5f}, f(x)-f(bg)={gap:.5f}, err={err:.5f}"
        )


# ---------------------------------------------------------------------------
# Production wiring pin: TestNoL1RegTruncation above only exercises this
# file's own _kernel_shap helper -- it never imports feature_shap.py /
# temporal_shap.py / node_shap.py, so it would keep passing even if a
# future edit dropped `l1_reg=_L1_REG` from one of those three
# `explainer.shap_values(...)` call sites or flipped its `_L1_REG` constant
# back to a truthy value (verified empirically this session: deleting the
# `l1_reg=_L1_REG` line from feature_shap.py left all other tests green).
# This class pins the actual production wiring, mirroring
# test_wiring_fixes.py's source/config-pinning convention (its
# not-referenced-anywhere balancer-key guard,
# test_tune_script_never_mentions_selection_metric).
# ---------------------------------------------------------------------------

class TestL1RegProductionWiring:
    """Each granularity module must define _L1_REG=False and actually pass
    it (by name) into its explain()'s shap_values(...) call."""

    def test_feature_shap_l1_reg_constant_is_false(self):
        import src.explainer.feature_shap as mod
        assert mod._L1_REG is False

    def test_temporal_shap_l1_reg_constant_is_false(self):
        import src.explainer.temporal_shap as mod
        assert mod._L1_REG is False

    def test_node_shap_l1_reg_constant_is_false(self):
        import src.explainer.node_shap as mod
        assert mod._L1_REG is False

    def test_feature_shap_explain_passes_l1_reg_by_name(self):
        import src.explainer.feature_shap as mod
        src_text = inspect.getsource(mod.FeatureGroupSHAP.explain)
        assert "l1_reg=_L1_REG" in src_text, (
            "feature_shap.py's explain() no longer passes l1_reg=_L1_REG "
            "into shap_values(...) -- this would silently re-enable shap's "
            "default 'num_features(10)' truncation."
        )

    def test_temporal_shap_explain_passes_l1_reg_by_name(self):
        import src.explainer.temporal_shap as mod
        src_text = inspect.getsource(mod.TemporalNeighborhoodSHAP.explain)
        assert "l1_reg=_L1_REG" in src_text, (
            "temporal_shap.py's explain() no longer passes l1_reg=_L1_REG "
            "into shap_values(...) -- this would silently re-enable shap's "
            "default 'num_features(10)' truncation."
        )

    def test_node_shap_explain_passes_l1_reg_by_name(self):
        import src.explainer.node_shap as mod
        src_text = inspect.getsource(mod.NodeNoveltySHAP.explain)
        assert "l1_reg=_L1_REG" in src_text, (
            "node_shap.py's explain() no longer passes l1_reg=_L1_REG "
            "into shap_values(...) -- this would silently re-enable shap's "
            "default 'num_features(10)' truncation."
        )
