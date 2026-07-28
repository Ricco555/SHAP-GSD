"""
C4 — GraphSVX baseline wrapper (journal version).

GraphSVX (Duval & Malliaros 2021) estimates Shapley values for both node
features and graph structure jointly, using a surrogate weighted linear model
trained on binary coalition samples.

We use structure-only mode (regu=0) so that GraphSVX attributes importance to
k-hop neighbour nodes — the same coalition space as GNNShap and PGExplainer.

Attribution space: N_input NODES (k-hop neighbours of the target edge's src).
Fidelity is computed via fidelity_from_node_mask (top-k nodes).

Import note: GraphSVX must be imported with os.chdir / sys.path pointing to
its own directory (src/ has non-package-style relative imports).

GRAPHSVX_DIR defaults to <repo_root>/external/graphsvx; override with the
SHAP_GSD_GRAPHSVX_DIR environment variable.
"""

from __future__ import annotations

import copy
import logging
import os
import sys
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from src.baselines._paths import resolve_baseline_dir

if TYPE_CHECKING:
    from src.baselines.adapter import FlowContext
    from src.explainer.background import BackgroundDistributions

logger = logging.getLogger(__name__)

GRAPHSVX_DIR = resolve_baseline_dir("SHAP_GSD_GRAPHSVX_DIR", "graphsvx")


# ── GraphSVX-compatible model wrapper ─────────────────────────────────────────

class GraphSVXNodeWrapper(nn.Module):
    """Model wrapper compatible with GraphSVX's node-classification interface.

    GraphSVX calls:
        self.model(x, edge_index).exp()[node_index]

    where it expects log-probabilities (shape N × C).  This wrapper:
      1. Accepts the full node feature matrix x (= h_full) and a coalition-
         masked edge_index from GraphSVX's structure ablation.
      2. Applies one-hop weighted aggregation using the masked edge_index.
      3. Classifies the target edge (fixed src/dst positions) via the frozen
         edge MLP.
      4. Returns log_softmax broadcast to shape (N_local, C).
    """

    def __init__(
        self,
        edge_mlp: nn.Module,
        x_e_t: torch.Tensor,        # (1, d_e)
        src_local_idx: int,
        dst_local_idx: int,
        num_nodes: int,
    ) -> None:
        super().__init__()
        self.edge_mlp = edge_mlp
        self.register_buffer("_x_e_t", x_e_t.detach())
        self._src_local = src_local_idx
        self._dst_local = dst_local_idx
        self._num_nodes = num_nodes

    def forward(
        self,
        x: torch.Tensor,            # (N, hidden) — h_full (or modified by GraphSVX)
        edge_index: torch.Tensor,   # (2, E_masked) — coalition-masked edges
    ) -> torch.Tensor:
        h = x.clone()

        if edge_index.size(1) > 0:
            h_agg = torch.zeros_like(h)
            src_idx, dst_idx = edge_index[0], edge_index[1]
            msgs = h[src_idx]                          # (E, hidden)
            h_agg.index_add_(0, dst_idx, msgs)
            h = h + h_agg                              # residual

        # Clamp indices to avoid out-of-bounds on very small subgraphs
        src_pos = min(self._src_local, h.size(0) - 1)
        dst_pos = min(self._dst_local, h.size(0) - 1)
        h_src = h[src_pos].unsqueeze(0)               # (1, hidden)
        h_dst = h[dst_pos].unsqueeze(0)               # (1, hidden)
        combined = torch.cat([h_src, h_dst, self._x_e_t], dim=1)
        logit = self.edge_mlp(combined)                # (1, C)
        log_prob = torch.log_softmax(logit, dim=1)     # (1, C) — GraphSVX applies .exp()
        return log_prob.expand(self._num_nodes, -1).contiguous()


# ── helpers ────────────────────────────────────────────────────────────────────


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

def run_graphsvx_with_model(
    ctx: "FlowContext",
    model: nn.Module,
    feature_groups: dict,
    background: "BackgroundDistributions",
    g_test,
    num_samples: int = 256,
    nhops: int = 2,
    top_k: int = 3,
) -> dict:
    """Run GraphSVX on one test flow (structure-only mode) and return fidelity.

    Args:
        ctx:           FlowContext from adapter.build_flow_context().
        model:         EdgeAwareGraphSAGE (for edge_mlp access).
        feature_groups: (unused — kept for consistent API).
        background:    BackgroundDistributions (for background_node_state).
        g_test:        DGL test graph (not directly used; kept for API consistency).
        num_samples:   Number of coalition samples for GraphSVX (default 256).
        nhops:         Neighbourhood depth, must match DGL sampler (default 2).
        top_k:         Number of top nodes for fidelity masking.

    Returns:
        dict with: edge_id, true_label, predicted_label, p_full,
                   node_scores (N_input-dim list), fidelity_plus,
                   fidelity_minus, runtime_s.
    """
    from torch_geometric.data import Data
    from src.baselines.adapter import (
        build_h_full,
        dgl_subgraph_to_pyg,
        fidelity_from_node_mask,
        surrogate_diagnostics,
        surrogate_delta,
    )

    t0 = time.time()
    edge_mlp_cpu = _move_to_cpu(model.edge_mlp)

    pyg_data, gnid_to_local = dgl_subgraph_to_pyg(
        ctx.blocks, None, ctx.base_node_feats
    )
    N_local = len(gnid_to_local)
    h_full = build_h_full(ctx, model, gnid_to_local, pyg_data.edge_index)

    src_local = gnid_to_local.get(ctx.target_src_nid, 0)
    dst_local = gnid_to_local.get(ctx.target_dst_nid, 0)
    x_e_t_cpu = ctx.x_e_t.detach().cpu()
    bg_node = background.background_node_state[ctx.true_label]   # (15,)

    # Diagnostics (instrumentation only).  Timed separately and excluded from
    # runtime_s so the reported runtime keeps its original meaning.
    _t_diag = time.time()
    diag = surrogate_diagnostics(
        h_full, pyg_data.edge_index, src_local, dst_local
    )
    t0 += time.time() - _t_diag

    # Isolated flow fallback
    if pyg_data.edge_index.size(1) == 0:
        node_scores_input = np.zeros(len(ctx.input_node_ids))
        fid_plus, fid_minus = fidelity_from_node_mask(
            node_scores_input, ctx, bg_node, model, top_k=top_k
        )
        return _pack_result(ctx, node_scores_input, fid_plus, fid_minus,
                            time.time() - t0,
                            fallback_reason="empty_subgraph", diag=diag)

    wrapper = GraphSVXNodeWrapper(
        edge_mlp_cpu, x_e_t_cpu, src_local, dst_local, N_local
    ).eval()

    _t_diag = time.time()
    diag["surrogate_delta"] = surrogate_delta(
        lambda ei: wrapper(h_full, ei)[0], pyg_data.edge_index
    )
    t0 += time.time() - _t_diag

    # Build PyG Data for GraphSVX (x = h_full; edge_index = subgraph edges)
    data = Data(
        x=h_full,
        edge_index=pyg_data.edge_index,
    )

    # Import GraphSVX.
    # GraphSVX's src/ has no __init__.py, so Python prefers SHAP-GSD's
    # 'src' package (which has __init__.py) even when GRAPHSVX_DIR is
    # earlier in sys.path.  Fix: create an empty __init__.py in GraphSVX's
    # src/ so it becomes a regular package and sys.path ordering is respected.
    _gsvx_init = os.path.join(GRAPHSVX_DIR, "src", "__init__.py")
    if not os.path.exists(_gsvx_init):
        open(_gsvx_init, "w").close()

    _orig_dir = os.getcwd()
    _orig_path = sys.path[:]
    _saved_src_mods = {k: v for k, v in sys.modules.items()
                       if k == "src" or k.startswith("src.")}
    for k in _saved_src_mods:
        del sys.modules[k]

    phi_list = None
    svx = None
    fallback_reason: str | None = None
    try:
        os.chdir(GRAPHSVX_DIR)
        sys.path.insert(0, GRAPHSVX_DIR)
        from src.explainers import GraphSVX

        svx = GraphSVX(data=data, model=wrapper, gpu=False)
        phi_list = svx.explain(
            node_indexes=[src_local],
            hops=nhops,
            num_samples=num_samples,
            info=False,
            multiclass=False,
            fullempty=None,
            S=3,
            args_hv="compute_pred",
            args_feat="Expectation",
            args_coal="Smarter",
            args_g="WLS",
            regu=0,             # structure-only: F=0, phi shape = (D,)
            vizu=False,
        )
    except Exception as exc:
        # NetworkXNoPath (disconnected directed subgraph) or any other internal
        # GraphSVX failure → fall back to zero attribution for this flow.
        #
        # NOTE (diagnostics): GraphSVX's indirect-effect branch calls
        # nx.all_shortest_paths(G, source=node_index, target=incl_nei) on a
        # DiGraph (custom_to_networkx defaults to to_undirected=False), while
        # DGL block edges point neighbour → target.  There is therefore usually
        # no outgoing path from the target node, and NetworkXNoPath is raised.
        # fallback_reason records the concrete exception class so the branch is
        # observable per flow instead of only at DEBUG level.
        fallback_reason = f"internal_exception:{type(exc).__name__}"
        logger.debug(f"GraphSVX internal error EID={ctx.global_eid}: {exc}")
    finally:
        os.chdir(_orig_dir)
        sys.path = _orig_path
        # Remove GraphSVX's src.* from module cache
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            if k not in _saved_src_mods:
                del sys.modules[k]
        # Restore SHAP-GSD's src.* entries
        sys.modules.update(_saved_src_mods)

    if phi_list is None or svx is None:
        node_scores_input = np.zeros(len(ctx.input_node_ids))
        fid_plus, fid_minus = fidelity_from_node_mask(
            node_scores_input, ctx, bg_node, model, top_k=top_k
        )
        return _pack_result(ctx, node_scores_input, fid_plus, fid_minus,
                            time.time() - t0,
                            fallback_reason=fallback_reason
                            or "graphsvx_returned_none", diag=diag)

    phi = phi_list[0]                  # numpy array, shape (D,)
    neighbours = svx.neighbours        # tensor of local node indices, length D

    node_scores_local = np.zeros(N_local)
    if phi is not None and len(phi) > 0 and len(neighbours) > 0:
        for i, n in enumerate(neighbours.tolist()):
            if i < len(phi) and 0 <= n < N_local:
                node_scores_local[n] = abs(float(phi[i]))
    else:
        # GraphSVX ran without raising, but produced no players (D == 0) or an
        # empty phi — silently zero before this instrumentation existed.
        fallback_reason = "empty_phi_or_neighbours"

    node_scores_input = _map_local_to_input(
        node_scores_local, ctx.blocks, gnid_to_local
    )

    # Fidelity computed against actual DGL model
    fid_plus, fid_minus = fidelity_from_node_mask(
        node_scores_input, ctx, bg_node, model, top_k=top_k
    )

    runtime_s = time.time() - t0
    logger.debug(
        f"GraphSVX EID={ctx.global_eid}: "
        f"fid+={fid_plus:.4f} fid-={fid_minus:.4f} t={runtime_s:.1f}s"
    )
    return _pack_result(ctx, node_scores_input, fid_plus, fid_minus, runtime_s,
                        fallback_reason=fallback_reason, diag=diag)


def _pack_result(
    ctx: "FlowContext",
    node_scores: np.ndarray,
    fid_plus: float,
    fid_minus: float,
    runtime_s: float,
    fallback_reason: str | None = None,
    diag: dict | None = None,
) -> dict:
    """Pack one flow's result.

    ``fallback_reason`` and the ``diag`` fields are additive diagnostic metadata
    (new CSV columns); they never affect the attribution or fidelity numbers.
    """
    from src.baselines.adapter import empty_diagnostics

    d = dict(empty_diagnostics())
    d["surrogate_delta"] = float("nan")
    if diag:
        d.update(diag)

    return {
        "edge_id":         ctx.global_eid,
        "true_label":      ctx.true_label,
        "predicted_label": ctx.predicted_label,
        "p_full":          round(ctx.p_full, 6),
        "node_scores":     node_scores.tolist(),
        "fidelity_plus":   round(fid_plus, 6),
        "fidelity_minus":  round(fid_minus, 6),
        "runtime_s":       round(runtime_s, 3),
        "fallback_reason": fallback_reason,
        **d,
    }
