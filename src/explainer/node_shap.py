"""
Node novelty coalition SHAP. Third SHAP-GSD granularity.

Answers: "Did endpoint history/novelty drive the classification?"

Coalition space: z_N ∈ {0,1}^(|V_sub| + 2) where:
  z_N[0]: source novelty flag  (1=keep actual, 0=zero out state dim 1)
  z_N[1]: dest novelty flag    (1=keep actual, 0=zero out state dim 1)
  z_N[2:]: non-target node presence (0=replace full 15-dim state with background)

Masking rules:
  Non-target node absent: replace full 15-dim state with background_node_state[c]
  Target novelty absent (z_N[0/1]=0): zero state[1] (novelty dim), keep rest
  time_sin (dim 11), time_cos (dim 12): never masked — time context not identity
  volume_deviation (dim 13), iat_regularity (dim 14): masked with non-target nodes

Output: dict with 'src_novelty', 'dst_novelty', and per-node φ values.
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

_NOVELTY_DIM = 1
_NEVER_MASK_DIMS = (11, 12)  # time_sin, time_cos


class NodeNoveltySHAP:
    """KernelSHAP over node novelty and non-target node states."""

    def __init__(
        self,
        background: "BackgroundDistributions",
        device: torch.device,
    ) -> None:
        """
        Args:
            background: BackgroundDistributions (training split).
            device:     torch device.
        """
        self.background = background
        self.device = device

    def explain(
        self,
        true_class: int,
        src_nid: int,
        dst_nid: int,
        model: "EdgeAwareGraphSAGE",
        blocks: list,
        input_nodes: torch.Tensor,
        base_node_feats: np.ndarray,
        src_pos: torch.Tensor,
        dst_pos: torch.Tensor,
        x_e: np.ndarray,
        nsamples: int = 512,
    ) -> dict:
        """Run KernelSHAP over node novelty and non-target node states.

        Args:
            true_class:      integer class label of the target edge.
            src_nid:         global node ID of the target edge's source.
            dst_nid:         global node ID of the target edge's destination.
            model:           trained EdgeAwareGraphSAGE (eval mode expected).
            blocks:          DGL computation blocks from the sampler.
            input_nodes:     global node IDs for blocks[0], shape (N_in,).
            base_node_feats: float32 array (N_in, 15) — actual node states.
            src_pos:         int64 tensor — seed edge source positions into h.
            dst_pos:         int64 tensor — seed edge destination positions into h.
            x_e:             float32 array (d_e,) — target edge features.
            nsamples:        KernelSHAP coalition samples (default 512).

        Returns:
            dict with keys:
              'src_novelty_shap': float
              'dst_novelty_shap': float
              'node_shap': dict {node_id_str: φ} for non-target nodes
              'coalition_size': int (|V_sub| + 2)
        """
        import shap

        model.eval()
        input_node_ids = input_nodes.cpu().numpy()
        bg_ns = self.background.background_node_state[true_class]  # (15,)
        x_e_t = torch.tensor(x_e, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Coalition layout:
        #   index 0 → src novelty (target src)
        #   index 1 → dst novelty (target dst)
        #   index 2..2+M-1 → non-target node states
        non_target_ids: list[int] = [
            int(nid) for nid in input_node_ids
            if int(nid) != src_nid and int(nid) != dst_nid
        ]
        M = len(non_target_ids)
        coalition_size = 2 + M

        non_target_pos: dict[int, int] = {nid: 2 + i for i, nid in enumerate(non_target_ids)}

        def _build_masked_states(row: np.ndarray) -> np.ndarray:
            """Build modified node feature matrix from binary coalition row."""
            modified = base_node_feats.copy()
            src_novelty_present = bool(row[0])
            dst_novelty_present = bool(row[1])

            for j, nid in enumerate(input_node_ids):
                nid_int = int(nid)
                if nid_int == src_nid and not src_novelty_present:
                    # Zero out novelty dimension; preserve all other dims
                    modified[j, _NOVELTY_DIM] = 0.0
                elif nid_int == dst_nid and not dst_novelty_present:
                    modified[j, _NOVELTY_DIM] = 0.0
                elif nid_int in non_target_pos:
                    if not row[non_target_pos[nid_int]]:
                        # Replace full state with class-conditional background
                        bg = bg_ns.copy()
                        # Preserve time context (never masked)
                        for d in _NEVER_MASK_DIMS:
                            bg[d] = base_node_feats[j, d]
                        modified[j] = bg
            return modified

        def _predict_fn(coalition_matrix: np.ndarray) -> np.ndarray:
            results: list[float] = []
            for row in coalition_matrix:
                modified = _build_masked_states(row)
                nf_t = torch.tensor(modified, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    logit = model(blocks, nf_t, x_e_t, src_pos, dst_pos)
                results.append(logit[0, true_class].item())
            return np.array(results, dtype=np.float64)

        background_data = np.zeros((1, coalition_size), dtype=np.float32)
        explainer = shap.KernelExplainer(_predict_fn, background_data)
        phi_raw = explainer.shap_values(
            np.ones((1, coalition_size), dtype=np.float32),
            nsamples=nsamples,
            silent=True,
        )
        phi = np.array(phi_raw).squeeze()
        if coalition_size == 1:
            phi = np.array([float(phi)])

        src_novelty_phi = float(phi[0])
        dst_novelty_phi = float(phi[1])
        node_phi = {str(nid): float(phi[2 + i]) for i, nid in enumerate(non_target_ids)}

        logger.debug(
            f"Node SHAP: class={true_class}, coalition_size={coalition_size}, "
            f"src_novelty_φ={src_novelty_phi:.4f}, dst_novelty_φ={dst_novelty_phi:.4f}"
        )

        return {
            "src_novelty_shap": src_novelty_phi,
            "dst_novelty_shap": dst_novelty_phi,
            "node_shap": node_phi,
            "coalition_size": coalition_size,
        }
