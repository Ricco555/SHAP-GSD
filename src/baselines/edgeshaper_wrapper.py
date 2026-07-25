"""
C5 — EdgeSHAPer baseline wrapper (journal version).

EdgeSHAPer (Mastropietro et al. 2022) estimates per-edge Shapley values via
Monte Carlo marginal-contribution sampling over edge coalitions.  Originally
designed for molecule graph classification; we adapt it for network flow edge
classification by supplying a graph-level model wrapper.

Attribution space: EDGES in the k-hop subgraph → aggregated to N_input nodes.
Fidelity is computed via fidelity_from_node_mask (top-k nodes).

Dependencies: rdkit, rdkit-heatmaps — required at import time (EdgeSHAPer's
own edgeshaper.py imports both unconditionally at module load, regardless of
whether visualization is used); pip install rdkit rdkit-heatmaps

EDGESHAPER_SRC defaults to <repo_root>/external/edgeshaper/src; override the
clone root (not the /src subpath) with the SHAP_GSD_EDGESHAPER_DIR
environment variable.
"""

from __future__ import annotations

import copy
import logging
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

EDGESHAPER_SRC = resolve_baseline_dir("SHAP_GSD_EDGESHAPER_DIR", "edgeshaper", "src")


# ── EdgeSHAPer-compatible model wrapper ───────────────────────────────────────

class EdgeSHAPModelWrapper(nn.Module):
    """Graph-level classifier compatible with EdgeSHAPer's forward interface.

    EdgeSHAPer calls:
        out = model(x, edge_index, batch=batch, edge_weight=edge_weight)
    and expects:
        F.softmax(out, dim=1)[0][target_class]   → scalar probability

    This wrapper returns logits of shape (1, C) by:
      1. Accepting coalition-masked edge_index (edges to include).
      2. Applying one-hop weighted aggregation over included edges.
      3. Classifying the target edge using frozen src/dst embeddings.
    """

    def __init__(
        self,
        edge_mlp: nn.Module,
        x_e_t: torch.Tensor,      # (1, d_e)
        src_local_idx: int,
        dst_local_idx: int,
    ) -> None:
        super().__init__()
        self.edge_mlp = edge_mlp
        self.register_buffer("_x_e_t", x_e_t.detach())
        self._src_local = src_local_idx
        self._dst_local = dst_local_idx

    def forward(
        self,
        x: torch.Tensor,             # (N, hidden) — h_full (node embeddings)
        edge_index: torch.Tensor,    # (2, E_masked) — coalition-masked edges
        batch: torch.Tensor = None,  # ignored (required by EdgeSHAPer API)
        edge_weight: torch.Tensor = None,  # optional
    ) -> torch.Tensor:
        h = x.clone()

        if edge_index.size(1) > 0:
            h_agg = torch.zeros_like(h)
            src_idx, dst_idx = edge_index[0], edge_index[1]
            if edge_weight is not None:
                msgs = edge_weight.view(-1, 1) * h[src_idx]
            else:
                msgs = h[src_idx]
            h_agg.index_add_(0, dst_idx, msgs)
            h = h + h_agg

        src_pos = min(self._src_local, h.size(0) - 1)
        dst_pos = min(self._dst_local, h.size(0) - 1)
        h_src = h[src_pos].unsqueeze(0)    # (1, hidden)
        h_dst = h[dst_pos].unsqueeze(0)    # (1, hidden)
        combined = torch.cat([h_src, h_dst, self._x_e_t], dim=1)
        return self.edge_mlp(combined)     # (1, C) — EdgeSHAPer applies softmax


# ── helpers ────────────────────────────────────────────────────────────────────

def _build_h_full(
    h_fixed: torch.Tensor,
    blocks: list,
    gnid_to_local: dict,
) -> torch.Tensor:
    import dgl as _dgl
    hidden_size = h_fixed.size(1)
    N_local = len(gnid_to_local)
    h_full = torch.zeros(N_local, hidden_size, dtype=torch.float32)
    seed_gnids = blocks[-1].dstdata[_dgl.NID].cpu().tolist()
    for i, gnid in enumerate(seed_gnids):
        local_idx = gnid_to_local.get(gnid)
        if local_idx is not None:
            h_full[local_idx] = h_fixed[i].detach().cpu()
    return h_full


def _edge_shap_to_node_scores(
    phi_edges: list,
    edge_index: torch.Tensor,
    N_local: int,
) -> np.ndarray:
    """Aggregate per-edge EdgeSHAPer values to per-node scores (sum |φ|)."""
    scores = np.zeros(N_local, dtype=np.float64)
    if edge_index.size(1) == 0:
        return scores
    src_idx = edge_index[0].cpu().numpy()
    dst_idx = edge_index[1].cpu().numpy()
    for e_i in range(min(len(phi_edges), edge_index.size(1))):
        s, d = src_idx[e_i], dst_idx[e_i]
        v = abs(float(phi_edges[e_i]))
        if s < N_local:
            scores[s] += v
        if d < N_local:
            scores[d] += v
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

def run_edgeshaper_with_model(
    ctx: "FlowContext",
    model: nn.Module,
    feature_groups: dict,
    background: "BackgroundDistributions",
    g_test,
    M: int = 100,
    top_k: int = 3,
) -> dict:
    """Run EdgeSHAPer on one test flow and return fidelity scores.

    Args:
        ctx:           FlowContext from adapter.build_flow_context().
        model:         EdgeAwareGraphSAGE (for edge_mlp access).
        feature_groups: (unused — kept for consistent API).
        background:    BackgroundDistributions (for background_node_state).
        g_test:        DGL test graph (not used; kept for API consistency).
        M:             Monte Carlo sampling steps per edge (default 100).
        top_k:         Number of top nodes for fidelity masking.

    Returns:
        dict with: edge_id, true_label, predicted_label, p_full,
                   node_scores (N_input-dim list), fidelity_plus,
                   fidelity_minus, runtime_s.
    """
    from src.baselines.adapter import (
        dgl_subgraph_to_pyg,
        fidelity_from_node_mask,
    )

    t0 = time.time()
    edge_mlp_cpu = _move_to_cpu(model.edge_mlp)

    pyg_data, gnid_to_local = dgl_subgraph_to_pyg(
        ctx.blocks, None, ctx.base_node_feats
    )
    N_local = len(gnid_to_local)
    h_full = _build_h_full(ctx.h_fixed, ctx.blocks, gnid_to_local)

    src_local = gnid_to_local.get(ctx.target_src_nid, 0)
    dst_local = gnid_to_local.get(ctx.target_dst_nid, 0)
    x_e_t_cpu = ctx.x_e_t.detach().cpu()
    bg_node = background.background_node_state[ctx.true_label]   # (15,)

    # Isolated flow fallback
    if pyg_data.edge_index.size(1) == 0:
        node_scores_input = np.zeros(len(ctx.input_node_ids))
        fid_plus, fid_minus = fidelity_from_node_mask(
            node_scores_input, ctx, bg_node, model, top_k=top_k
        )
        return _pack_result(ctx, node_scores_input, fid_plus, fid_minus,
                            time.time() - t0)

    wrapper = EdgeSHAPModelWrapper(
        edge_mlp_cpu, x_e_t_cpu, src_local, dst_local
    ).eval()

    # Import EdgeSHAPer (just a single source file — add its dir to sys.path)
    if EDGESHAPER_SRC not in sys.path:
        sys.path.insert(0, EDGESHAPER_SRC)
    from edgeshaper import Edgeshaper

    explainer = Edgeshaper(
        model=wrapper,
        x=h_full,
        edge_index=pyg_data.edge_index,
        device="cpu",
    )

    # Compute graph density clamped to valid binomial probability.
    # Dense subgraphs (many repeated edges) can yield density > 1, which
    # causes numpy.random.binomial to raise ValueError.
    n_e = pyg_data.edge_index.size(1)
    max_edges = max(N_local * (N_local - 1), 1)
    P = float(np.clip(n_e / max_edges, 0.01, 0.99))

    phi_edges = explainer.explain(
        M=M,
        target_class=ctx.true_label,
        P=P,
        deviation=None,
        log_odds=False,
        seed=42,
    )

    local_scores = _edge_shap_to_node_scores(
        phi_edges, pyg_data.edge_index, N_local
    )
    node_scores_input = _map_local_to_input(local_scores, ctx.blocks, gnid_to_local)

    # Fidelity computed against actual DGL model
    fid_plus, fid_minus = fidelity_from_node_mask(
        node_scores_input, ctx, bg_node, model, top_k=top_k
    )

    runtime_s = time.time() - t0
    logger.debug(
        f"EdgeSHAPer EID={ctx.global_eid}: "
        f"fid+={fid_plus:.4f} fid-={fid_minus:.4f} t={runtime_s:.1f}s"
    )
    return _pack_result(ctx, node_scores_input, fid_plus, fid_minus, runtime_s)


def _pack_result(
    ctx: "FlowContext",
    node_scores: np.ndarray,
    fid_plus: float,
    fid_minus: float,
    runtime_s: float,
) -> dict:
    return {
        "edge_id":         ctx.global_eid,
        "true_label":      ctx.true_label,
        "predicted_label": ctx.predicted_label,
        "p_full":          round(ctx.p_full, 6),
        "node_scores":     node_scores.tolist(),
        "fidelity_plus":   round(fid_plus, 6),
        "fidelity_minus":  round(fid_minus, 6),
        "runtime_s":       round(runtime_s, 3),
    }
