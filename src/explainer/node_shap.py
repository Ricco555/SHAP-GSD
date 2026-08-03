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
# See feature_shap.py's _L1_REG for the shared rationale (specs/45 sec 1-2,
# 3.2).
_L1_REG: bool = False


def build_non_target_ids(
    input_node_ids: np.ndarray,
    src_nid: int,
    dst_nid: int,
) -> list[int]:
    """Non-target node IDs in coalition-column order (indices 2..2+M-1).

    Single source of truth for the node-novelty player layout: both
    ``NodeNoveltySHAP.explain`` and the fidelity pass in
    ``src/explainer/node_fidelity.py`` derive the coalition columns from
    this function, so a stored per-node phi can never be attached to the
    wrong player.

    Args:
        input_node_ids: int array (N_in,) of global node IDs for blocks[0]
                        input, in block row order.
        src_nid:        global node ID of the target edge's source.
        dst_nid:        global node ID of the target edge's destination.

    Returns:
        The input node IDs with both target endpoints removed, in the
        original block row order.
    """
    return [
        int(nid) for nid in input_node_ids
        if int(nid) != src_nid and int(nid) != dst_nid
    ]


def build_non_target_pos(non_target_ids: list[int]) -> dict[int, int]:
    """{global node id -> coalition column}, columns starting at 2.

    Args:
        non_target_ids: output of :func:`build_non_target_ids`.

    Returns:
        Mapping from global node ID to its coalition column, where column
        0 is the target source novelty flag and column 1 the target
        destination novelty flag.
    """
    return {nid: 2 + i for i, nid in enumerate(non_target_ids)}


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

    def build_masked_node_states(
        self,
        row: np.ndarray,
        base_node_feats: np.ndarray,
        input_node_ids: np.ndarray,
        src_nid: int,
        dst_nid: int,
        non_target_pos: dict[int, int],
        true_class: int,
    ) -> np.ndarray:
        """Node-state matrix for one node-novelty coalition (Option C masking).

        Two masking mechanisms, dispatched per input-node row — never
        interchangeable:

          * target endpoint (src_nid / dst_nid) with its novelty bit absent
            (row[0] / row[1] == 0): ONLY dim _NOVELTY_DIM is zeroed; every
            other dim of the node's actual state is preserved.
          * non-target node with its own bit absent: the full state row is
            replaced by background_node_state[true_class], with dims
            _NEVER_MASK_DIMS copied back from the node's actual state.

        Nothing is rebuilt in DGL: the caller's `blocks` object is reused
        unchanged (CLAUDE.md Option C).

        Unlike the temporal granularity's masking helper, this takes the FULL
        binary coalition row rather than an absent-index list, so that the
        per-player mechanism dispatch lives here and cannot be duplicated or
        conflated by callers.

        Args:
            row:             (2 + M,) binary coalition vector. Index 0 = target
                             src novelty, index 1 = target dst novelty,
                             indices 2.. = non-target nodes in non_target_pos
                             order. 1 = present (keep actual), 0 = absent.
            base_node_feats: float32 (N_in, node_state_dim) actual node states;
                             row j corresponds to input_node_ids[j].
            input_node_ids:  int array (N_in,) of global node IDs for blocks[0]
                             input, in block row order.
            src_nid:         global node ID of the target edge's source.
            dst_nid:         global node ID of the target edge's destination.
            non_target_pos:  {global node id -> coalition column}, as built by
                             build_non_target_pos().
            true_class:      class index selecting the background node state.

        Returns:
            float32 array (N_in, node_state_dim) — always a fresh copy, never
            `base_node_feats` itself.
        """
        assert base_node_feats.shape[0] == len(input_node_ids), (
            f"row misalignment: base_node_feats has {base_node_feats.shape[0]} rows "
            f"but input_node_ids has {len(input_node_ids)}"
        )
        assert len(row) == 2 + len(non_target_pos), (
            f"coalition width {len(row)} != 2 + {len(non_target_pos)} players"
        )

        bg_ns = self.background.background_node_state[true_class]  # (15,)
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
        x_e_t = torch.tensor(x_e, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Coalition layout:
        #   index 0 → src novelty (target src)
        #   index 1 → dst novelty (target dst)
        #   index 2..2+M-1 → non-target node states
        non_target_ids: list[int] = build_non_target_ids(input_node_ids, src_nid, dst_nid)
        M = len(non_target_ids)
        coalition_size = 2 + M

        non_target_pos: dict[int, int] = build_non_target_pos(non_target_ids)

        def _predict_fn(coalition_matrix: np.ndarray) -> np.ndarray:
            results: list[float] = []
            for row in coalition_matrix:
                modified = self.build_masked_node_states(
                    row, base_node_feats, input_node_ids,
                    src_nid, dst_nid, non_target_pos, true_class,
                )
                nf_t = torch.tensor(modified, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    logit = model(blocks, nf_t, x_e_t, src_pos, dst_pos)
                results.append(logit[0, true_class].item())
            return np.array(results, dtype=np.float64)

        background_data = np.zeros((1, coalition_size), dtype=np.float32)
        foreground_data = np.ones((1, coalition_size), dtype=np.float32)
        explainer = shap.KernelExplainer(_predict_fn, background_data)
        f_baseline = float(np.squeeze(explainer.expected_value))
        f_logit = float(_predict_fn(foreground_data)[0])
        phi_raw = explainer.shap_values(
            foreground_data,
            nsamples=nsamples,
            l1_reg=_L1_REG,
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
            f"src_novelty_φ={src_novelty_phi:.4f}, dst_novelty_φ={dst_novelty_phi:.4f}, "
            f"efficiency_err={abs(phi.sum() - (f_logit - f_baseline)):.4f}"
        )

        return {
            "src_novelty_shap": src_novelty_phi,
            "dst_novelty_shap": dst_novelty_phi,
            "node_shap": node_phi,
            "coalition_size": coalition_size,
            "f_baseline": f_baseline,
            "f_logit": f_logit,
        }
