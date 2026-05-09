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
    ) -> dict[str, float]:
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
            dict mapping group_name → signed φ value.
        """
        import shap

        model.eval()
        bg_feat = self.background.background_features[true_class]  # (d_e,)

        # Pre-encode node embeddings once — fixed for all coalition evaluations
        with torch.no_grad():
            h_fixed = model.encode(blocks, node_feats)  # (n_seed_nodes, hidden)

        def _predict_fn(coalition_matrix: np.ndarray) -> np.ndarray:
            results: list[float] = []
            for row in coalition_matrix:
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

        background_data = np.zeros((1, self.K), dtype=np.float32)
        explainer = shap.KernelExplainer(_predict_fn, background_data)
        phi_raw = explainer.shap_values(
            np.ones((1, self.K), dtype=np.float32),
            nsamples=nsamples,
            silent=True,
        )
        phi = np.array(phi_raw).squeeze()  # (K,)

        logger.debug(
            f"Feature SHAP: class={true_class}, "
            f"sum(φ)={phi.sum():.4f}, top group={self.group_names[int(np.abs(phi).argmax())]}"
        )
        return {name: float(phi[i]) for i, name in enumerate(self.group_names)}
