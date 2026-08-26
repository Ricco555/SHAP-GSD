"""
Feature-group KernelSHAP. First SHAP-GSD granularity.

Answers: "Which semantic feature groups drove this classification?"

Coalition space: K feature groups from feature_groups.json.
Node embeddings are pre-encoded once; only the edge MLP is re-run per
coalition sample, keeping inference cost low.

Output: dict {group_name: signed_φ} (K entries).
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import dgl
    from src.model.sage_model import EdgeAwareGraphSAGE
    from src.explainer.background import BackgroundDistributions

logger = logging.getLogger(__name__)

# shap>=0.47 defaults l1_reg to "num_features(10)", silently L1-truncating
# attributions to at most 10 nonzero coalition players. SHAP-GSD needs the
# genuine, unregularized attribution spread for its concentration
# statistics (specs/45 sec 1-2) -- must stay False, not a config key
# (specs/45 sec 3.2).
_L1_REG: bool = False


class FeatureGroupSHAP:
    """KernelSHAP over semantic feature groups for a single target edge."""

    def __init__(
        self,
        feature_groups: dict,
        background: "BackgroundDistributions",
        device: torch.device,
    ) -> None:
        """
        Args:
            feature_groups: parsed feature_groups.json dict with keys
                            'd_e', 'K', 'feature_names', 'groups'.
            background:     BackgroundDistributions (training split).
            device:         torch device.
        """
        self.group_names: list[str] = list(feature_groups["groups"].keys())
        self.groups: dict = feature_groups["groups"]
        self.K: int = len(self.group_names)
        self.background = background
        self.device = device
        assert self.K == feature_groups["K"], (
            f"group count mismatch: expected {feature_groups['K']}, got {self.K}"
        )

    def screen_feature_players(
        self,
        x_e: np.ndarray,
        bg_feat: np.ndarray,
    ) -> list[bool]:
        """Per-group input-degeneracy screen (specs/67), in group_names order.

        Returns a length-K list: entry i is True iff masking group i is a
        bit-level no-op for this flow, i.e. its absent and present coalition
        inputs are identical, so its TRUE Shapley value is exactly 0.

        Exact equality, deliberately -- never a tolerance (specs/63's
        ``== 0.0`` philosophy). A group that differs from the background in
        the last bit is genuinely, if minutely, live and stays in the
        solver. ``np.array_equal``'s NaN behavior (NaN != NaN) is the
        conservative direction and is intended: a group containing NaN on
        either side is judged non-degenerate and kept, never dropped, so it
        can only ever return this flow's pre-fix behavior for that group.
        Do not "fix" this into ``np.allclose`` or an ``equal_nan=True``
        variant.

        Not branched on ``groups[name]["type"]``: numeric groups can be
        class-constant in training just as categoricals can (specs/67
        sec 1.2), so the predicate must be a general value comparison.
        """
        # The masking statement is `masked[idxs] = bg_feat[idxs]` where
        # `masked = x_e.copy()`, so the value actually stored is
        # `bg_feat[idxs]` cast to x_e's dtype -- compare the assignment's
        # result, not a naive cross-dtype `==`. Both sides are float32
        # today, so this cast is a no-op in practice; it is written
        # explicitly so the predicate stays correct if either dtype ever
        # changes, and `copy=False` makes it free in the common case.
        bg_cast = bg_feat.astype(x_e.dtype, copy=False)
        return [
            bool(np.array_equal(
                x_e[self.groups[name]["indices"]],
                bg_cast[self.groups[name]["indices"]],
            ))
            for name in self.group_names
        ]

    def explain(
        self,
        true_class: int,
        model: "EdgeAwareGraphSAGE",
        blocks: list,
        node_feats: torch.Tensor,
        x_e: np.ndarray,
        src_pos: torch.Tensor,
        dst_pos: torch.Tensor,
        nsamples: int = 512,
    ) -> tuple[dict[str, float], float, float, list[str]]:
        """Run KernelSHAP over feature groups for one target edge.

        Node embeddings are encoded once from the actual subgraph and reused
        for all M coalition evaluations; only the edge MLP is re-invoked.

        Args:
            true_class:  integer class label of the target edge.
            model:       trained EdgeAwareGraphSAGE (eval mode expected).
            blocks:      DGL computation blocks from TemporalNeighborSampler.
            node_feats:  float32 tensor (num_input_nodes, 15) on device.
            x_e:         float32 numpy array (d_e,) — target edge features.
            src_pos:     int64 tensor — seed edge source positions into h.
            dst_pos:     int64 tensor — seed edge destination positions into h.
            nsamples:    KernelSHAP coalition samples (default 512).

        Returns:
            4-tuple ``(dict mapping group_name -> signed phi value,
            f_baseline, f_logit, degenerate_group_names)``, where
            ``degenerate_group_names`` are the groups dropped from the
            KernelSHAP coalition matrix because their absent and present
            coalition inputs are bit-identical for this flow (specs/67);
            their reported phi is an exact structural 0.0, never estimated.
        """
        import shap

        model.eval()
        bg_feat = self.background.background_features[true_class]  # (d_e,)

        # Input-degeneracy guard (specs/67): drop groups whose masked
        # ("absent") and native ("present") inputs are bit-identical for
        # this flow before the coalition matrix is built at all. Because
        # group index sets are pairwise disjoint (feature_groups.py
        # validate()), dropping any subset simultaneously leaves `masked`
        # bit-identical at every coalition -- this is a proof, not a
        # necessary condition (specs/67 sec 3.1).
        degenerate = self.screen_feature_players(x_e, bg_feat)
        degenerate_group_names = [
            name for name, d in zip(self.group_names, degenerate) if d
        ]

        # Full (conceptual, width K) -> reduced (solver-facing) column
        # index mapping. Degenerate columns are simply absent from this
        # mapping.
        full_to_reduced: dict[int, int] = {}
        reduced_col = 0
        for i in range(self.K):
            if not degenerate[i]:
                full_to_reduced[i] = reduced_col
                reduced_col += 1
        reduced_size = reduced_col   # == K - len(degenerate_group_names)

        # Pre-encode node embeddings once — fixed for all coalition evaluations
        with torch.no_grad():
            h_fixed = model.encode(blocks, node_feats)  # (n_seed_nodes, hidden)

        def _expand_row(reduced_row: np.ndarray) -> np.ndarray:
            """Reduced-width solver row -> full-width (K,) coalition row.

            Dropped columns are filled with a constant 1 ("present"), which
            is a no-op for a degenerate group by definition: its absent and
            present masked inputs are bit-identical (specs/67 sec 1.1), and
            group index sets are pairwise disjoint (feature_groups.py
            validate()), so this holds jointly for any number of dropped
            groups, at every coalition. The fill also keeps f_logit and
            f_baseline the exact same two quantities they were pre-fix --
            see the reassembly comment below.
            """
            full_row = np.ones(self.K, dtype=np.float32)
            for full_idx, reduced_idx in full_to_reduced.items():
                full_row[full_idx] = reduced_row[reduced_idx]
            return full_row

        def _predict_fn(coalition_matrix: np.ndarray) -> np.ndarray:
            results: list[float] = []
            for reduced_row in coalition_matrix:
                row = _expand_row(reduced_row)
                masked = x_e.copy()
                for i, (name, present) in enumerate(zip(self.group_names, row)):
                    if not present:
                        idxs = self.groups[name]["indices"]
                        masked[idxs] = bg_feat[idxs]
                masked_t = torch.tensor(
                    masked, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                with torch.no_grad():
                    logit = model.classify(h_fixed, masked_t, src_pos, dst_pos)
                results.append(logit[0, true_class].item())
            return np.array(results, dtype=np.float64)

        if reduced_size == 0:
            # Every group degenerate. shap.KernelExplainer cannot be
            # constructed with a zero-width background. The single
            # achievable coalition state is the only one there is, so
            # baseline and foreground logits coincide by construction.
            # Reachable only if x_e == bg_feat on every one of d_e indices,
            # astronomically unlikely on real data but must not raise.
            phi_reduced = np.array([])
            f_baseline = f_logit = float(
                _predict_fn(np.ones((1, 0), dtype=np.float32))[0]
            )
        else:
            background_data = np.zeros((1, reduced_size), dtype=np.float32)
            foreground_data = np.ones((1, reduced_size), dtype=np.float32)
            explainer = shap.KernelExplainer(_predict_fn, background_data)
            # f_baseline = E[f(background)] — evaluated at init time by KernelExplainer
            f_baseline = float(np.squeeze(explainer.expected_value))
            # f_logit = f(all groups present) — one extra forward pass
            f_logit = float(_predict_fn(foreground_data)[0])
            phi_raw = explainer.shap_values(
                foreground_data,
                nsamples=nsamples,
                l1_reg=_L1_REG,
                silent=True,
            )
            if reduced_size == 1:
                # np.array(phi_raw).squeeze() would yield a 0-d array and
                # the reassembly's phi_reduced[reduced_idx] indexing would
                # raise.
                phi_reduced = np.array([float(np.squeeze(phi_raw))])
            else:
                phi_reduced = np.array(phi_raw).squeeze()  # (reduced_size,)

        # Reassemble the full-width phi vector. Degenerate columns are left
        # at their np.zeros initialization -- an exact 0.0, never estimated
        # (specs/67 sec 1.1, sec 3.1).
        phi = np.zeros(self.K, dtype=np.float64)
        for full_idx, reduced_idx in full_to_reduced.items():
            phi[full_idx] = phi_reduced[reduced_idx]

        logger.debug(
            f"Feature SHAP: class={true_class}, "
            f"n_degenerate_feature_players={len(degenerate_group_names)}, "
            f"sum(φ)={phi.sum():.4f}, f_logit={f_logit:.4f}, f_baseline={f_baseline:.4f}, "
            f"efficiency_err={abs(phi.sum() - (f_logit - f_baseline)):.4f}"
        )
        return (
            {name: float(phi[i]) for i, name in enumerate(self.group_names)},
            f_baseline,
            f_logit,
            degenerate_group_names,
        )
