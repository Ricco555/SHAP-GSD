"""
Fidelity+/Fidelity− for the node-novelty granularity (φ_N).

Metric definitions (matching scripts/08_metrics.py's feature-group pass
term-for-term, so the resulting paper row is directly comparable):

  Fidelity+ = P(true_class | all players present)
              − P(true_class | top-k players masked)
  Fidelity− = P(true_class | all players present)
              − P(true_class | ONLY top-k players present)

Player set: 2 + M, where index 0 is the target edge's source novelty flag,
index 1 the target destination novelty flag, and indices 2..2+M-1 the M
non-target nodes of the sampled block, in `build_non_target_ids()` order.
P = 2 + M >= 2 always, so — unlike the temporal granularity — there is no
zero-player case and no flow is excluded.

TWO MASKING MECHANISMS, never interchangeable (see
`NodeNoveltySHAP.build_masked_node_states`, which this module calls rather
than re-implementing):

  * an absent novelty flag zeroes ONLY the novelty state dim of that endpoint;
  * an absent non-target node has its FULL state row replaced by
    background_node_state[true_class], except the never-masked time dims.

Both fire in the same Fidelity− pass: "keep only the top-k" masks every
non-selected player by its own mechanism. Masking only one player type
silently computes a different metric.

Option C is preserved: no DGL graph surgery — `blocks` is reused unchanged
and only the node-state array varies.

Logit vs probability: φ_N itself is attributed in logit space
(`node_shap.py`'s scoring), but fidelity here is computed in softmax
PROBABILITY space, matching the feature-group fidelity definitions.

WARNING — the cached-embedding shortcut is INVALID here. The feature-group
pass may reuse a cached `h_fixed = model.encode(...)` because its masking only
touches edge features. φ_N masking mutates NODE STATES, which are the input to
`encode()`, so a cached embedding is stale: using `model.classify(h_fixed, …)`
would yield Fidelity+ ≡ Fidelity− ≡ 0 for every flow. Every masked evaluation
below therefore runs the FULL `model.forward()`.
"""

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from src.explainer.node_shap import NodeNoveltySHAP
    from src.model.sage_model import EdgeAwareGraphSAGE

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
        node_feats: (N_in, node_state_dim) masked node-state matrix.
        x_e_t:      (1, d_e) float32 target edge features, already on device.
        src_pos:    seed edge source positions into h.
        dst_pos:    seed edge destination positions into h.
        true_label: class index whose probability is returned.
        device:     torch device.

    Returns:
        Softmax probability of `true_label`.
    """
    nf_t = torch.tensor(node_feats, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(blocks, nf_t, x_e_t, src_pos, dst_pos)
        proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return float(proba[true_label])


def align_phi_to_players(
    non_target_ids: list[int],
    stored_node_ids: list[int],
    stored_node_phi: list[float],
    src_novelty_phi: float,
    dst_novelty_phi: float,
) -> "Optional[np.ndarray]":
    """Reorder stored φ_N onto the freshly re-derived coalition layout.

    The stored `node_ids` / `node_shap` lists come from Phase 6 in
    `non_target_ids` insertion order and are NOT sorted by |φ|, while the
    fresh `non_target_ids` order comes from a freshly re-sampled block. The
    node sets are expected to be equal (the temporal neighbour sampler is
    deterministic for a given single-element seed), but order equality is not
    guaranteed a priori — hence a genuine reorder, not just a length check.

    Args:
        non_target_ids:  freshly re-derived non-target node IDs, coalition
                         column order (columns 2..2+M-1).
        stored_node_ids: `node_ids` from the Phase-6 explanation JSON.
        stored_node_phi: `node_shap` from the Phase-6 explanation JSON,
                         parallel to `stored_node_ids`.
        src_novelty_phi: stored φ for coalition column 0.
        dst_novelty_phi: stored φ for coalition column 1.

    Returns:
        (2 + M,) float array in coalition order
        [src_novelty, dst_novelty, *non_target_ids order], or None if the
        re-derived non-target node set does not match the stored one — in
        which case the caller skips the flow rather than misattributing φ.
    """
    assert len(stored_node_ids) == len(stored_node_phi), (
        f"stored node_ids/node_shap length mismatch: "
        f"{len(stored_node_ids)} != {len(stored_node_phi)}"
    )

    # Coerce to int: JSON round-trips have historically produced string node
    # keys (node_shap's dict is keyed by str(nid)), and a str/int mismatch
    # would make every set comparison below fail silently.
    stored_ids = [int(n) for n in stored_node_ids]
    fresh_ids = [int(n) for n in non_target_ids]

    nid_to_phi = dict(zip(stored_ids, stored_node_phi))
    assert len(nid_to_phi) == len(stored_ids), (
        f"duplicate node IDs in stored node_ids ({len(stored_ids)} entries, "
        f"{len(nid_to_phi)} unique) — φ mapping would be corrupt"
    )

    if len(stored_ids) != len(fresh_ids) or set(stored_ids) != set(fresh_ids):
        n_diff = len(set(stored_ids) ^ set(fresh_ids))
        logger.warning(
            f"Player-set mismatch: {len(stored_ids)} stored non-target nodes vs "
            f"{len(fresh_ids)} re-derived, symmetric difference {n_diff}; "
            f"skipping flow"
        )
        return None

    return np.array(
        [src_novelty_phi, dst_novelty_phi] + [nid_to_phi[n] for n in fresh_ids],
        dtype=np.float64,
    )


def novelty_fidelity_for_flow(
    model: "EdgeAwareGraphSAGE",
    node_shap: "NodeNoveltySHAP",
    blocks: list,
    phi_novelty: np.ndarray,
    base_node_feats: np.ndarray,
    input_node_ids: np.ndarray,
    src_nid: int,
    dst_nid: int,
    non_target_pos: dict[int, int],
    x_e_t: torch.Tensor,
    src_pos: torch.Tensor,
    dst_pos: torch.Tensor,
    true_label: int,
    p_full: float,
    top_k: int,
    device: torch.device,
) -> dict:
    """Fidelity+/− for the node-novelty granularity of one flow.

    Both masked coalitions are built by
    `NodeNoveltySHAP.build_masked_node_states`, i.e. by the identical code
    path φ_N's own KernelSHAP uses, so the metric measures exactly the game
    that was attributed. This function only decides WHICH players are present;
    it never decides which masking mechanism applies.

    Args:
        model:           trained EdgeAwareGraphSAGE (eval mode expected).
        node_shap:       NodeNoveltySHAP holding the training-fit background.
        blocks:          DGL computation blocks, reused unchanged (Option C).
        phi_novelty:     (2 + M,) signed φ_N in coalition column order.
        base_node_feats: (N_in, node_state_dim) float32 actual node states.
        input_node_ids:  (N_in,) int global node IDs, block row order.
        src_nid:         target edge source, global node ID.
        dst_nid:         target edge destination, global node ID.
        non_target_pos:  {global node id -> coalition column}.
        x_e_t:           (1, d_e) float32 target edge features, on device.
        src_pos:         seed edge source positions into h.
        dst_pos:         seed edge destination positions into h.
        true_label:      class index of the target edge.
        p_full:          P(true_label | everything present), precomputed by
                         the caller and passed through unchanged.
        top_k:           number of top-|φ| players to select.
        device:          torch device.

    Returns:
        dict with keys n_players, n_non_target, effective_k, p_full,
        p_masked, p_kept, fidelity_plus, fidelity_minus, top_k_players,
        n_novelty_in_top_k. n_novelty_in_top_k only counts a novelty flag
        whose own |phi| is nonzero — a zero-phi flag that landed in top_idx
        merely because P was small does not count as "reaching top-k".
        Rounding is left to the row-building layer.
    """
    P = len(phi_novelty)
    assert P == 2 + len(non_target_pos), (
        f"φ width {P} != 2 + {len(non_target_pos)} players"
    )
    assert P >= 2, f"node-novelty player count must be >= 2, got {P}"

    k = min(top_k, P)

    # kind="stable" is mandatory: shap's default l1_reg L1-truncates φ_N, so
    # exact-zero ties are common and an unstable sort would make the published
    # top-k non-reproducible across NumPy versions. Ascending-stable-then-
    # reverse breaks ties by DESCENDING coalition index; do not rewrite this as
    # argsort(-|φ|), which would break them the other way and change the
    # selection. (The temporal granularity uses a plain argsort, where its
    # median player count of 1 made ties near-irrelevant.)
    order = np.argsort(np.abs(phi_novelty), kind="stable")[::-1]
    top_idx = order[:k]

    # Fidelity+ : start all-present, turn the top-k off.
    row_plus = np.ones(P, dtype=np.float32)
    row_plus[top_idx] = 0.0
    nf_plus = node_shap.build_masked_node_states(
        row_plus, base_node_feats, input_node_ids,
        src_nid, dst_nid, non_target_pos, true_label,
    )
    p_masked = _proba(
        model, blocks, nf_plus, x_e_t, src_pos, dst_pos, true_label, device
    )

    # Fidelity− : start all-absent, turn ONLY the top-k on. Every non-top-k
    # player is masked by its own mechanism — novelty flags zeroed AND
    # non-target nodes background-replaced.
    row_kept = np.zeros(P, dtype=np.float32)
    row_kept[top_idx] = 1.0
    nf_kept = node_shap.build_masked_node_states(
        row_kept, base_node_feats, input_node_ids,
        src_nid, dst_nid, non_target_pos, true_label,
    )
    p_kept = _proba(
        model, blocks, nf_kept, x_e_t, src_pos, dst_pos, true_label, device
    )

    idx_to_nid = {col: nid for nid, col in non_target_pos.items()}
    top_k_players: list[str] = []
    for col in top_idx:
        col = int(col)
        if col == 0:
            top_k_players.append("src_novelty")
        elif col == 1:
            top_k_players.append("dst_novelty")
        else:
            top_k_players.append(f"node:{idx_to_nid[col]}")

    # n_novelty_in_top_k must NOT count a novelty flag that only landed in
    # top_idx because P was small (k = min(top_k, P) degenerates to "keep
    # everyone" for low-player flows). shap's l1_reg produces exact-zero phi
    # ties (see the argsort comment above), so a zero-phi novelty flag can sit
    # inside top_idx without carrying any genuine attribution. Require the
    # novelty column's own |phi| to be nonzero before it counts as "reaching
    # top-k" — this is what scripts/08_metrics.py's table2_novelty.txt reports
    # as "novelty reaches top-k in N/1846", so a trivial zero-phi inclusion
    # would silently inflate that count.
    novelty_top_idx = top_idx[top_idx < 2]
    n_novelty_in_top_k = int(np.sum(np.abs(phi_novelty[novelty_top_idx]) > 0))

    return {
        "n_players":          P,
        "n_non_target":       P - 2,
        "effective_k":        k,
        "p_full":             p_full,
        "p_masked":           p_masked,
        "p_kept":             p_kept,
        "fidelity_plus":      p_full - p_masked,
        "fidelity_minus":     p_full - p_kept,
        "top_k_players":      top_k_players,
        "n_novelty_in_top_k": n_novelty_in_top_k,
    }
