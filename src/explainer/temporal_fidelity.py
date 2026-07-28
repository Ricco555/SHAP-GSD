"""
Fidelity+/Fidelity− for the temporal-neighborhood granularity (φ_T).

Metric definition (mirrors the feature-group metric in scripts/08_metrics.py):

  Fidelity+ (necessity):  p_full − p_masked
      p_masked = P(true_class | top-k neighbor edges REMOVED from node state)
  Fidelity− (sufficiency): p_full − p_kept
      p_kept   = P(true_class | only the top-k neighbor edges KEPT, the rest
                 removed from node state)

Masking is Option C, exactly as in the φ_T coalition game: an absent neighbor
edge is removed by recomputing its two endpoints' 15-dim node states via
NodeStateManager.rollback_edges. There is NO DGL graph surgery, and there is NO
background substitution — for φ_T, "absent" means state rollback, so
BackgroundDistributions plays no role in this module.

IMPORTANT — the cached-embedding shortcut used by the φ_F fidelity pass
(`model.classify(h_fixed, …)`) is INVALID here. Rolling back a node state
changes `model.encode()`'s output, so a cached `h_fixed` is stale and would make
Fidelity+ ≡ Fidelity− ≡ 0 for every flow. Every masked evaluation in this module
therefore runs the FULL `model.forward()`.

Space convention: φ_T attributions themselves live in logit space
(temporal_shap.py scores `logit[0, true_class]`), but φ_T *fidelity* is computed
in softmax-probability space so the resulting row is directly comparable to the
feature-group fidelity row.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from src.model.sage_model import EdgeAwareGraphSAGE
    from src.explainer.temporal_shap import TemporalNeighborhoodSHAP

logger = logging.getLogger(__name__)


def _proba(
    model: "EdgeAwareGraphSAGE",
    blocks: list,
    node_feats: np.ndarray,
    x_e_t: torch.Tensor,
    src_pos: torch.Tensor,
    dst_pos: torch.Tensor,
    true_label: int,
    device: torch.device,
) -> float:
    """P(true_label) from a FULL model.forward() on the given node states.

    Args:
        model:      trained EdgeAwareGraphSAGE (eval mode expected).
        blocks:     DGL computation blocks, reused unchanged (Option C).
        node_feats: float32 (N_in, node_state_dim) node-state matrix.
        x_e_t:      float32 (1, d_e) target edge features, already on device.
        src_pos:    int64 tensor — seed edge source positions into h.
        dst_pos:    int64 tensor — seed edge destination positions into h.
        true_label: integer class label of the target edge.
        device:     torch device.

    Returns:
        Softmax probability of `true_label`.
    """
    nf_t = torch.tensor(node_feats, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(blocks, nf_t, x_e_t, src_pos, dst_pos)
        proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return float(proba[true_label])


def align_phi_to_records(
    neighbor_records: list,
    stored_eids: list[int],
    stored_phi: list[float],
) -> "np.ndarray | None":
    """Reorder stored φ_T onto the freshly re-extracted record order.

    The φ_T values on disk were written sorted by |φ| descending, while the
    fidelity pass re-samples the neighborhood and gets records in sampler order.
    This joins the two on global EID.

    Args:
        neighbor_records: freshly extracted _NeighborEdge records.
        stored_eids:      global EIDs from the Phase 6 explanation record.
        stored_phi:       signed φ_T values, parallel to stored_eids.

    Returns:
        A (N,) float array aligned to neighbor_records, or None if the
        re-sampled neighbor set does not match the stored one (caller skips
        the flow). Per the sampler-determinism invariant this should never
        happen on the current tree; it is a counted, logged skip rather than a
        crash so that a future sampler change surfaces instead of silently
        misattributing φ values.
    """
    eid_to_phi = dict(zip(stored_eids, stored_phi))
    assert len(eid_to_phi) == len(stored_eids), (
        f"duplicate global EIDs in stored explanation: {len(eid_to_phi)} unique "
        f"of {len(stored_eids)}"
    )

    record_eids = {r.global_eid for r in neighbor_records}
    if len(neighbor_records) != len(stored_eids) or record_eids != set(stored_eids):
        logger.warning(
            f"Neighborhood mismatch: re-sampled {len(neighbor_records)} neighbors "
            f"but explanation stored {len(stored_eids)} "
            f"({len(record_eids - set(stored_eids))} EIDs not in the stored set); "
            f"skipping flow"
        )
        return None

    return np.array([eid_to_phi[r.global_eid] for r in neighbor_records], dtype=np.float64)


def temporal_fidelity_for_flow(
    model: "EdgeAwareGraphSAGE",
    temp_shap: "TemporalNeighborhoodSHAP",
    blocks: list,
    neighbor_records: list,
    phi_temporal: np.ndarray,
    base_node_feats: np.ndarray,
    input_node_ids: np.ndarray,
    target_ts_ms: float,
    x_e_t: torch.Tensor,
    src_pos: torch.Tensor,
    dst_pos: torch.Tensor,
    true_label: int,
    p_full: float,
    top_k: int,
    device: torch.device,
) -> dict:
    """Fidelity+/− for the temporal-neighborhood granularity of one flow.

    Args:
        model:            trained EdgeAwareGraphSAGE (eval mode expected).
        temp_shap:        TemporalNeighborhoodSHAP providing the SAME masking
                          routine the φ_T attributions were computed with.
        blocks:           DGL computation blocks, reused unchanged (Option C).
        neighbor_records: in-window _NeighborEdge records for this flow.
        phi_temporal:     (N,) signed φ_T aligned to neighbor_records.
        base_node_feats:  float32 (N_in, node_state_dim), all neighbors present.
        input_node_ids:   int (N_in,) global node IDs in block row order.
        target_ts_ms:     target edge timestamp in milliseconds.
        x_e_t:            float32 (1, d_e) target edge features, on device.
        src_pos:          int64 tensor — seed edge source positions into h.
        dst_pos:          int64 tensor — seed edge destination positions into h.
        true_label:       integer class label of the target edge.
        p_full:           P(true_label) with all neighbors present, computed by
                          the caller and passed through unchanged.
        top_k:            number of top-|φ| neighbors to mask / keep.
        device:           torch device.

    Returns:
        Dict with keys n_neighbors, effective_k, p_full, p_masked, p_kept,
        fidelity_plus, fidelity_minus, top_k_neighbor_eids. The four masked
        quantities are None iff the flow has no in-window neighbors (N == 0),
        in which case no model call is made at all.
    """
    N = len(neighbor_records)
    assert len(phi_temporal) == N, (
        f"phi/record length mismatch: {len(phi_temporal)} φ values for {N} records"
    )

    if N == 0:
        return {
            "n_neighbors":         0,
            "effective_k":         0,
            "p_full":              p_full,
            "p_masked":            None,
            "p_kept":              None,
            "fidelity_plus":       None,
            "fidelity_minus":      None,
            "top_k_neighbor_eids": [],
        }

    k = min(top_k, N)
    order = np.argsort(np.abs(phi_temporal))[::-1]
    top_idx = order[:k]

    # Fidelity+ — the top-k neighbors are ABSENT
    nf_plus = temp_shap.build_masked_node_feats(
        neighbor_records, top_idx, base_node_feats, input_node_ids, target_ts_ms
    )
    p_masked = _proba(
        model, blocks, nf_plus, x_e_t, src_pos, dst_pos, true_label, device
    )

    # Fidelity− — everything EXCEPT the top-k is ABSENT. When k == N this set is
    # empty, so p_kept == p_full exactly and fidelity_minus == 0.0 by
    # construction (keeping the top-k is keeping everything).
    rest_idx = np.setdiff1d(np.arange(N), top_idx)
    nf_kept = temp_shap.build_masked_node_feats(
        neighbor_records, rest_idx, base_node_feats, input_node_ids, target_ts_ms
    )
    p_kept = _proba(
        model, blocks, nf_kept, x_e_t, src_pos, dst_pos, true_label, device
    )

    return {
        "n_neighbors":         N,
        "effective_k":         int(k),
        "p_full":              p_full,
        "p_masked":            p_masked,
        "p_kept":              p_kept,
        "fidelity_plus":       p_full - p_masked,
        "fidelity_minus":      p_full - p_kept,
        "top_k_neighbor_eids": [int(neighbor_records[i].global_eid) for i in top_idx],
    }
