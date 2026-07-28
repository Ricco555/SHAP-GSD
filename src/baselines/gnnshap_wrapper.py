"""
C3 — GNNShap baseline wrapper (journal version).

GNNShap (Xypolopoulos et al. 2023) estimates edge-level Shapley values in a
k-hop computational subgraph using kernel-weighted WLS regression over sampled
edge coalitions.

Attribution space: EDGES in the k-hop subgraph → aggregated to N_input nodes.
Fidelity is computed via fidelity_from_node_mask (top-k nodes).

Import note: GNNShap's CUDA extension requires os.chdir to GNNShap's root.
The wrapper handles this internally.

GNNSHAP_DIR defaults to <repo_root>/external/gnnshap; override with the
SHAP_GSD_GNNSHAP_DIR environment variable.
"""

from __future__ import annotations

import copy
import logging
import os
import sys
import time
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
import torch.nn as nn

from src.baselines._paths import resolve_baseline_dir

if TYPE_CHECKING:
    from src.baselines.adapter import FlowContext
    from src.explainer.background import BackgroundDistributions

logger = logging.getLogger(__name__)

GNNSHAP_DIR = resolve_baseline_dir("SHAP_GSD_GNNSHAP_DIR", "gnnshap")


# ── Dummy MessagePassing layer for GNNShap compatibility ─────────────────────

def _get_dummy_mp_class():
    """Return a DummyMP class using PyG MessagePassing (lazy import)."""
    from torch_geometric.nn import MessagePassing
    import torch.nn as _nn

    class DummyMP(MessagePassing):
        """Dummy layer so GNNShap's has_add_self_loops() works correctly."""
        def __init__(self):
            super().__init__(aggr="add")
            self.add_self_loops = False

        def forward(self, x, edge_index):
            return x

        def message(self, x_j):
            return x_j

    return DummyMP


# ── GNNShap-compatible wrapper ─────────────────────────────────────────────────

class GNNShapWrapper(nn.Module):
    """Wrapper for GNNShap with pre-computed DGL node embeddings.

    Includes a dummy MessagePassing layer so GNNShap's has_add_self_loops()
    utility does not raise AssertionError.  Actual predictions use the frozen
    DGL embeddings with one-hop masked aggregation controlled by GNNShap's
    edge coalitions.
    """

    def __init__(
        self,
        h_full: torch.Tensor,       # (N_local, hidden) — pre-computed DGL embeddings
        edge_mlp: nn.Module,
        x_e_t: torch.Tensor,        # (1, d_e)
        src_relabeled: int,          # relabeled position of target edge's src node
        dst_relabeled: int,          # relabeled position of target edge's dst node
    ) -> None:
        super().__init__()
        DummyMP = _get_dummy_mp_class()
        self._dummy_mp = DummyMP()
        self.register_buffer("_h_full", h_full.detach())
        self.edge_mlp = edge_mlp
        self.register_buffer("_x_e_t", x_e_t.detach())
        self._src_relabeled = src_relabeled
        self._dst_relabeled = dst_relabeled


# ── Forward function factory ───────────────────────────────────────────────────

def _make_forward_fn(edge_mlp_cpu: nn.Module, x_e_t_cpu: torch.Tensor,
                     src_relabeled: int, dst_relabeled: int,
                     return_logits: bool = False) -> Callable[..., torch.Tensor]:
    """Return a GNNShap-compatible forward_fn (no-batch mode).

    GNNShap calls:
        y_hat = forward_fn(model, node_features, masked_edge_index, node_idx)
    where node_features = data.x[subset] (re-indexed subset of h_full).
    Returns shape (num_classes,) for no-batch mode.

    Args:
        edge_mlp_cpu:  CPU deep-copy of the model's edge MLP.
        x_e_t_cpu:     (1, d_e) target edge feature vector on CPU.
        src_relabeled: readout position of the target edge's source node after
                       GNNShap's k-hop pruning/relabeling.
        dst_relabeled: readout position of the target edge's destination node.
        return_logits: if True the closure returns the RAW, pre-softmax class
                       logits instead of softmax probabilities.  The flag lives
                       on the FACTORY, never on the returned closure: GNNShap
                       calls ``forward_fn`` positionally, so an extra parameter
                       on the closure itself would be a collision hazard.  Build
                       a second closure for logit-space measurement instead —
                       both share this one body, so they cannot drift.

    Returns:
        A ``forward_fn(model, node_features, edge_index, node_idx, edge_weight)``
        closure returning a 1-D ``(num_classes,)`` tensor.
    """
    def forward_fn(
        model,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_idx: int,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        h = node_features.clone()

        if edge_index.size(1) > 0:
            h_agg = torch.zeros_like(h)
            src_idx, dst_idx = edge_index[0], edge_index[1]
            if edge_weight is not None:
                msgs = edge_weight.view(-1, 1) * h[src_idx]
            else:
                msgs = h[src_idx]
            h_agg.index_add_(0, dst_idx, msgs)
            h = h + h_agg       # residual: self + weighted neighbours

        h_src = h[src_relabeled].unsqueeze(0)   # (1, hidden)
        h_dst = h[dst_relabeled].unsqueeze(0)   # (1, hidden)
        combined = torch.cat([h_src, h_dst, x_e_t_cpu], dim=1)
        logit = edge_mlp_cpu(combined)           # (1, num_classes)
        if return_logits:
            return logit.squeeze(0)                    # (num_classes,) raw
        # PROBABILITIES by default, unchanged: this is what GNNShap's
        # WLS coalition solver is handed as its value function.
        return torch.softmax(logit, dim=1).squeeze(0)   # (num_classes,)

    return forward_fn


# ── helpers ────────────────────────────────────────────────────────────────────


def _compute_relabeled_positions(
    src_local_idx: int,
    dst_local_idx: int,
    edge_index: torch.Tensor,
    N_local: int,
    nhops: int = 2,
) -> tuple[int, int]:
    """Pre-compute where src and dst end up after GNNShap's k-hop relabeling.

    GNNShap calls pruned_comp_graph(src_local_idx, nhops, edge_index)
    internally. We replicate that call here to find the relabeled positions
    of src and dst before passing them to the forward_fn.
    """
    from torch_geometric.utils import k_hop_subgraph
    subset, _, mapping, _ = k_hop_subgraph(
        node_idx=src_local_idx,
        num_hops=nhops,
        edge_index=edge_index,
        relabel_nodes=True,
        num_nodes=N_local,
    )
    src_relabeled = mapping[0].item() if mapping.numel() > 0 else 0
    subset_list = subset.tolist()
    if dst_local_idx in subset_list:
        dst_relabeled = subset_list.index(dst_local_idx)
    else:
        dst_relabeled = src_relabeled   # fallback: use same position as src
    return src_relabeled, dst_relabeled


def _edge_shap_to_node_scores(
    shap_vals: np.ndarray,
    sub_edge_index: torch.Tensor,
    N_local: int,
) -> np.ndarray:
    """Aggregate per-edge Shapley values to per-node scores (sum of |φ|)."""
    scores = np.zeros(N_local, dtype=np.float64)
    src_idx = sub_edge_index[0].cpu().numpy()
    dst_idx = sub_edge_index[1].cpu().numpy()
    for e_i, (s, d) in enumerate(zip(src_idx, dst_idx)):
        if s < N_local and d < N_local:
            scores[s] += abs(float(shap_vals[e_i]))
            scores[d] += abs(float(shap_vals[e_i]))
    return scores


def _map_local_to_input(
    local_scores: np.ndarray,
    blocks: list,
    gnid_to_local: dict,
) -> np.ndarray:
    import dgl as _dgl
    input_gnids = blocks[0].srcdata[_dgl.NID].cpu().tolist()
    input_scores = np.zeros(len(input_gnids), dtype=np.float64)
    for i, gnid in enumerate(input_gnids):
        local_idx = gnid_to_local.get(gnid)
        if local_idx is not None:
            input_scores[i] = local_scores[local_idx]
    return input_scores


def _move_to_cpu(module: nn.Module) -> nn.Module:
    return copy.deepcopy(module).cpu()


# ── Main inference function ────────────────────────────────────────────────────

def run_gnnshap_with_model(
    ctx: "FlowContext",
    model: nn.Module,
    feature_groups: dict,
    background: "BackgroundDistributions",
    g_test,
    nsamples: int = 256,
    nhops: int = 2,
    top_k: int = 3,
) -> dict:
    """Run GNNShap on one test flow and return fidelity scores.

    Args:
        ctx:           FlowContext from adapter.build_flow_context().
        model:         EdgeAwareGraphSAGE (for edge_mlp access).
        feature_groups: (unused — kept for consistent API).
        background:    BackgroundDistributions (for background_node_state).
        g_test:        DGL test graph (not directly used; kept for API consistency).
        nsamples:      Number of coalition samples (default 256).
        nhops:         k-hop depth for computational subgraph (must match DGL sampler).
        top_k:         Number of top nodes for fidelity masking.

    Returns:
        dict with: edge_id, true_label, predicted_label, p_full,
                   node_scores (N_input-dim list), fidelity_plus,
                   fidelity_minus, runtime_s.
    """
    from torch_geometric.data import Data
    from src.baselines.adapter import (
        build_h_full,
        classify_degenerate_attribution,
        dgl_subgraph_to_pyg,
        empty_diagnostics,
        fidelity_from_node_mask,
        pack_node_baseline_result,
        surrogate_diagnostics,
        surrogate_delta,
    )

    t0 = time.time()
    # Accumulated wall-clock spent on instrumentation.  Subtracted from
    # runtime_s at every exit so the reported runtime keeps its original
    # meaning; t0 itself is never mutated, so it always means "start".
    # _zero_fallback below READS this name from the enclosing scope at call
    # time (it is not a default argument), so it always sees the current value.
    diag_s = 0.0
    edge_mlp_cpu = _move_to_cpu(model.edge_mlp)

    pyg_data, gnid_to_local = dgl_subgraph_to_pyg(
        ctx.blocks, None, ctx.base_node_feats
    )
    N_local = len(gnid_to_local)

    src_local = gnid_to_local.get(ctx.target_src_nid, 0)
    dst_local = gnid_to_local.get(ctx.target_dst_nid, 0)
    x_e_t_cpu = ctx.x_e_t.detach().cpu()
    bg_node = background.background_node_state[ctx.true_label]   # (15,)

    def _zero_fallback(reason: str, diag_block: dict):
        scores = np.zeros(len(ctx.input_node_ids))
        fp, fm = fidelity_from_node_mask(scores, ctx, bg_node, model, top_k=top_k)
        return pack_node_baseline_result(
            ctx, scores, fp, fm, time.time() - t0 - diag_s,
            fallback_reason=reason, diag=diag_block,
        )

    # Isolated flow or too few players for coalition sampling.  Both checked
    # BEFORE build_h_full, whose multi-layer message-passing result these
    # branches would immediately discard.
    if pyg_data.edge_index.size(1) == 0:
        return _zero_fallback(
            "empty_subgraph",
            empty_diagnostics(n_local=N_local, n_subgraph_edges=0),
        )
    if N_local < 3:
        logger.debug(
            f"GNNShap EID={ctx.global_eid}: N_local={N_local} < 3, "
            "insufficient players for coalition sampling — zero fallback"
        )
        return _zero_fallback(
            "insufficient_players",
            empty_diagnostics(
                n_local=N_local,
                n_subgraph_edges=int(pyg_data.edge_index.size(1)),
            ),
        )

    h_full = build_h_full(ctx, model, gnid_to_local, pyg_data.edge_index)

    # Diagnostics (instrumentation only).  Timed separately and excluded from
    # runtime_s so the reported runtime keeps its original meaning.
    _t_diag = time.time()
    diag = surrogate_diagnostics(
        h_full, pyg_data.edge_index, src_local, dst_local
    )
    diag_s += time.time() - _t_diag

    # Pre-compute relabeled positions for forward_fn
    src_relabeled, dst_relabeled = _compute_relabeled_positions(
        src_local, dst_local, pyg_data.edge_index, N_local, nhops
    )

    wrapper = GNNShapWrapper(
        h_full, edge_mlp_cpu, x_e_t_cpu, src_relabeled, dst_relabeled
    ).eval()

    forward_fn = _make_forward_fn(
        edge_mlp_cpu, x_e_t_cpu, src_relabeled, dst_relabeled
    )

    # surrogate_delta must be measured on the SAME graph GNNShap plays on:
    # the pruned, relabeled k-hop comp graph around src_local, with
    # node_features = h_full[subset] and readout index = src_relabeled.
    # Measuring it on the full unrelabeled subgraph would overstate the
    # surrogate's sensitivity to GNNShap's actual player set.  Only the output
    # SPACE changes below — the scoping (_subset / _sub_ei / src_relabeled) is
    # untouched.
    _t_diag = time.time()
    # A second closure over the SAME body, differing only in output space: the
    # library-facing forward_fn above stays softmax-probability (that is what
    # GNNShap's WLS solver consumes), while the diagnostic is measured on raw
    # logits, where the flatness predicate is actually sound (see
    # adapter.surrogate_delta).  Built inside the diagnostic timing window
    # because nothing but the diagnostic uses it.
    forward_fn_logits = _make_forward_fn(
        edge_mlp_cpu, x_e_t_cpu, src_relabeled, dst_relabeled,
        return_logits=True,
    )
    from torch_geometric.utils import k_hop_subgraph as _khs
    _subset, _sub_ei, _, _ = _khs(
        node_idx=src_local, num_hops=nhops, edge_index=pyg_data.edge_index,
        relabel_nodes=True, num_nodes=N_local,
    )
    _h_sub = h_full[_subset]
    diag["surrogate_delta_logit"] = surrogate_delta(
        lambda ei: forward_fn_logits(wrapper, _h_sub, ei, src_relabeled), _sub_ei
    )
    diag_s += time.time() - _t_diag

    # Build PyG Data object for GNNShap
    data = Data(
        x=h_full,
        edge_index=pyg_data.edge_index,
        y=torch.full((N_local,), ctx.true_label, dtype=torch.long),
    )

    # Import GNNShap (must chdir to its root for CUDA extension)
    _orig_dir = os.getcwd()
    _orig_path = sys.path[:]
    explanation = None
    fallback_reason: str | None = None
    try:
        os.chdir(GNNSHAP_DIR)
        if GNNSHAP_DIR not in sys.path:
            sys.path.insert(0, GNNSHAP_DIR)
        from gnnshap.explainer import GNNShapExplainer

        explainer = GNNShapExplainer(
            model=wrapper,
            data=data,
            nhops=nhops,
            device="cpu",
            forward_fn=forward_fn,
            progress_hide=True,
            verbose=0,
        )

        explanation = explainer.explain(
            node_idx=src_local,
            nsamples=nsamples,
            batch_size=0,               # no batching — forward_fn returns (C,)
            target_class=ctx.true_label,
            sampler_name="GNNShapSampler",
            solver_name="WLSSolver",
        )
    except Exception as exc:
        fallback_reason = f"internal_exception:{type(exc).__name__}"
        logger.debug(
            f"GNNShap EID={ctx.global_eid} N_local={N_local}: {exc} — zero fallback"
        )
    finally:
        os.chdir(_orig_dir)
        sys.path = _orig_path

    if explanation is None:
        return _zero_fallback(
            fallback_reason or "explainer_returned_none", diag
        )

    # GNNShapExplanation stores shap_values (np.array) and sub_edge_index (already numpy)
    shap_vals = np.array(explanation.shap_values)
    sub_edge_index = torch.from_numpy(explanation.sub_edge_index).long()

    local_scores = _edge_shap_to_node_scores(shap_vals, sub_edge_index, N_local)
    node_scores_input = _map_local_to_input(local_scores, ctx.blocks, gnid_to_local)

    # No branch fired, but the estimator itself may still have returned an
    # all-zero / numerically saturated attribution vector.  Recorded so a
    # degenerate row can be told apart from a fallback row.
    fallback_reason = fallback_reason or classify_degenerate_attribution(
        node_scores_input
    )

    # Fidelity computed against actual DGL model
    fid_plus, fid_minus = fidelity_from_node_mask(
        node_scores_input, ctx, bg_node, model, top_k=top_k
    )

    runtime_s = time.time() - t0 - diag_s
    logger.debug(
        f"GNNShap EID={ctx.global_eid}: "
        f"fid+={fid_plus:.4f} fid-={fid_minus:.4f} t={runtime_s:.1f}s"
    )
    return pack_node_baseline_result(
        ctx, node_scores_input, fid_plus, fid_minus, runtime_s,
        fallback_reason=fallback_reason, diag=diag,
    )
