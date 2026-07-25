"""
Phase B — Shared adapter: DGL ↔ PyG bridge for all four baseline explainers.

Provides:
  FlowContext        — all pre-computed tensors for one target flow
  build_flow_context — samples blocks, computes node states, pre-encodes h
  FeatureAttributionWrapper
                     — nn.Module with forward(x, edge_index) treating x[0]
                       as the edge feature vector; lets GNNExplainer/PGExplainer
                       run gradient-based attribution over 218 raw dims
  NodeAttributionWrapper
                     — nn.Module where forward(x, edge_index) replaces the node
                       feature matrix, then classifies the target edge;
                       used by GNNShap and GraphSVX
  dgl_subgraph_to_pyg
                     — extracts a PyG Data object from DGL blocks
  scores_to_group_importances
                     — aggregates 218-dim raw masks to 48 semantic groups
  fidelity_from_group_importances
                     — computes Fidelity+/- from a 48-group importance vector
                       using the same forward-pass logic as 08_metrics.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    import dgl
    from torch_geometric.data import Data
    from src.model.sage_model import EdgeAwareGraphSAGE
    from src.model.node_state import NodeStateManager
    from src.data.feature_store import FeatureStore
    from src.explainer.background import BackgroundDistributions
    from src.model.temporal_sampler import TemporalNeighborSampler

logger = logging.getLogger(__name__)


# ── FlowContext ────────────────────────────────────────────────────────────────

@dataclass
class FlowContext:
    """All pre-computed tensors for one target flow (shared by all baselines)."""

    global_eid:       int
    local_eid:        int
    true_label:       int
    predicted_label:  int
    p_full:           float           # P(true_label | all features)

    # DGL blocks — on device
    blocks:           list
    h_fixed:          torch.Tensor    # (n_seed_nodes, hidden) — pre-encoded
    x_e:              np.ndarray      # (d_e,) edge features, float32
    x_e_t:            torch.Tensor    # (1, d_e) on device
    src_pos:          torch.Tensor    # seed edge src position into h
    dst_pos:          torch.Tensor    # seed edge dst position into h

    # Node state tensors — both formats kept for different baselines
    input_node_ids:   np.ndarray      # (N_input,) global node IDs
    base_node_feats:  np.ndarray      # (N_input, 15)
    node_feats_t:     torch.Tensor    # (N_input, 15) on device

    target_ts_ms:     float
    target_src_nid:   int             # global node ID of target edge source
    target_dst_nid:   int             # global node ID of target edge destination


def build_flow_context(
    global_eid: int,
    model: "EdgeAwareGraphSAGE",
    g_test: "dgl.DGLGraph",
    nsm: "NodeStateManager",
    fs: "FeatureStore",
    sampler: "TemporalNeighborSampler",
    device: torch.device,
) -> FlowContext:
    """Sample blocks, compute node states, pre-encode h for one target flow.

    Performs the expensive shared work once so all baselines reuse the result.
    """
    import dgl as _dgl
    from src.model.sage_model import build_src_dst_pos

    # Find local EID
    global_eids_t = g_test.edata[_dgl.EID]
    local_eid = int((global_eids_t == global_eid).nonzero(as_tuple=True)[0][0].item())

    seed_t = torch.tensor([local_eid], dtype=torch.long)
    inp, _, blocks = sampler.sample_blocks(g_test, seed_t)
    blocks_d = [b.to(device) for b in blocks]
    inp_d = inp.to(device)

    target_ts_ms = float(g_test.edata["timestamp"][local_eid].item())
    src_t, dst_t = g_test.find_edges(seed_t)
    target_src_nid = int(src_t[0].item())
    target_dst_nid = int(dst_t[0].item())

    # Node states
    input_node_ids = inp.numpy()
    base_node_feats = np.stack([
        nsm.get_state_at_time(int(nid), target_ts_ms) for nid in input_node_ids
    ])
    node_feats_t = torch.tensor(base_node_feats, dtype=torch.float32, device=device)

    # Edge features
    x_e = fs[global_eid].copy()
    x_e_t = torch.tensor(x_e, dtype=torch.float32, device=device).unsqueeze(0)

    # Seed positions
    seed_nodes_final = blocks_d[-1].dstdata[_dgl.NID]
    src_pos, dst_pos = build_src_dst_pos(g_test, seed_t, seed_nodes_final.cpu())
    src_pos = src_pos.to(device)
    dst_pos = dst_pos.to(device)

    # Full prediction + pre-encode h
    model.eval()
    with torch.no_grad():
        h_fixed = model.encode(blocks_d, node_feats_t)
        logit = model.classify(h_fixed, x_e_t, src_pos, dst_pos)
        proba = torch.softmax(logit, dim=1).cpu().numpy()[0]

    true_label = int(fs.labels[fs._pos_of(global_eid)])
    predicted_label = int(np.argmax(proba))
    p_full = float(proba[true_label])

    return FlowContext(
        global_eid=global_eid,
        local_eid=local_eid,
        true_label=true_label,
        predicted_label=predicted_label,
        p_full=p_full,
        blocks=blocks_d,
        h_fixed=h_fixed,
        x_e=x_e,
        x_e_t=x_e_t,
        src_pos=src_pos,
        dst_pos=dst_pos,
        input_node_ids=input_node_ids,
        base_node_feats=base_node_feats,
        node_feats_t=node_feats_t,
        target_ts_ms=target_ts_ms,
        target_src_nid=target_src_nid,
        target_dst_nid=target_dst_nid,
    )


# ── FeatureAttributionWrapper ──────────────────────────────────────────────────

class FeatureAttributionWrapper(nn.Module):
    """Singleton-graph model for GNNExplainer / PGExplainer feature attribution.

    The "graph" has one node whose 218-dim features are the target edge's feature
    vector.  GNNExplainer optimises a mask over those 218 dims; the result is
    then aggregated to 48 semantic groups.

    forward(x, edge_index=None):
        x          : (1, d_e) — possibly masked edge feature tensor
        edge_index : ignored  (no graph structure for feature-only attribution)
        returns    : (1, num_classes) logits
    """

    def __init__(
        self,
        h_fixed: torch.Tensor,     # (n_seed_nodes, hidden) — frozen
        edge_mlp: nn.Module,
        src_pos: torch.Tensor,     # (1,) int64
        dst_pos: torch.Tensor,     # (1,) int64
        device: torch.device,
    ) -> None:
        super().__init__()
        # Register as buffers so .to(device) works and they are non-trainable
        self.register_buffer("_h_fixed", h_fixed.detach())
        self.register_buffer("_src_pos", src_pos)
        self.register_buffer("_dst_pos", dst_pos)
        self.edge_mlp = edge_mlp
        self._device = device

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        h_src = self._h_fixed[self._src_pos]   # (1, hidden)
        h_dst = self._h_fixed[self._dst_pos]   # (1, hidden)
        # x expected (1, d_e) from GNNExplainer; broadcast if singleton
        if x.dim() == 1:
            x = x.unsqueeze(0)
        combined = torch.cat([h_src, h_dst, x], dim=1)   # (1, 2*hidden+d_e)
        return self.edge_mlp(combined)                     # (1, num_classes)


# ── NodeAttributionWrapper ─────────────────────────────────────────────────────

class NodeAttributionWrapper(nn.Module):
    """Model wrapper for GNNShap / GraphSVX node-coalition attribution.

    forward(x, edge_index=None):
        x          : (N_input, 15) — perturbed node feature matrix
                     (absent coalition nodes replaced with background state)
        edge_index : ignored  (topology comes from frozen DGL blocks)
        returns    : (N_input, num_classes) — each row is the target-edge
                     logit broadcast to all nodes so that indexing [node_idx]
                     returns the edge prediction regardless of which node_idx
                     the baseline queries.
    """

    def __init__(
        self,
        model: "EdgeAwareGraphSAGE",
        blocks: list,
        x_e_t: torch.Tensor,      # (1, d_e)
        src_pos: torch.Tensor,
        dst_pos: torch.Tensor,
        num_nodes: int,            # N_input — needed for output shape
        device: torch.device,
    ) -> None:
        super().__init__()
        self._model = model
        self._blocks = blocks
        self.register_buffer("_x_e_t",   x_e_t.detach())
        self.register_buffer("_src_pos", src_pos)
        self.register_buffer("_dst_pos", dst_pos)
        self._num_nodes = num_nodes
        self._device = device

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            h = self._model.encode(self._blocks, x)
            logit = self._model.classify(h, self._x_e_t, self._src_pos, self._dst_pos)
            proba = torch.softmax(logit, dim=1)   # (1, num_classes)
        # Broadcast to (N_input, num_classes) so node_idx slicing works
        return proba.expand(self._num_nodes, -1)


# ── DGL → PyG conversion ──────────────────────────────────────────────────────

def dgl_subgraph_to_pyg(
    blocks: list,
    g_test: "dgl.DGLGraph",
    node_feats_np: np.ndarray,
) -> "Data":
    """Build a PyG Data object from DGL computation blocks.

    All edges from both blocks are included; nodes are re-indexed locally.

    Args:
        blocks:       DGL computation blocks (on device or CPU — NID edata used).
        g_test:       original test split graph (for global→local edge lookup).
        node_feats_np: (N_input, 15) numpy array of node features.

    Returns:
        torch_geometric.data.Data with:
          x          : (N_local, 15) node features
          edge_index : (2, E_sub) re-indexed edges
    """
    from torch_geometric.data import Data
    import dgl as _dgl

    # Collect all unique global node IDs across both blocks
    all_gnids: list[int] = []
    for block in blocks:
        all_gnids.extend(block.srcdata[_dgl.NID].cpu().tolist())
        all_gnids.extend(block.dstdata[_dgl.NID].cpu().tolist())
    unique_gnids = list(dict.fromkeys(all_gnids))          # preserve insertion order
    gnid_to_local: dict[int, int] = {g: i for i, g in enumerate(unique_gnids)}

    # Build edge_index in local space from block edges
    srcs_local: list[int] = []
    dsts_local: list[int] = []
    for block in blocks:
        b_src_gnids = block.srcdata[_dgl.NID].cpu().tolist()
        b_dst_gnids = block.dstdata[_dgl.NID].cpu().tolist()
        b_srcs, b_dsts = block.edges()
        for s_local, d_local in zip(b_srcs.tolist(), b_dsts.tolist()):
            src_g = b_src_gnids[s_local]
            dst_g = b_dst_gnids[d_local]
            srcs_local.append(gnid_to_local[src_g])
            dsts_local.append(gnid_to_local[dst_g])

    edge_index = torch.tensor([srcs_local, dsts_local], dtype=torch.long)

    # Build node feature matrix in local order
    # node_feats_np rows correspond to input_node_ids (= blocks[0].srcdata[NID])
    # We need features for ALL nodes in unique_gnids
    input_gnids = blocks[0].srcdata[_dgl.NID].cpu().tolist()
    input_pos = {g: i for i, g in enumerate(input_gnids)}

    N_local = len(unique_gnids)
    x = np.zeros((N_local, node_feats_np.shape[1]), dtype=np.float32)
    for local_idx, gnid in enumerate(unique_gnids):
        if gnid in input_pos:
            x[local_idx] = node_feats_np[input_pos[gnid]]
        # nodes not in input (seed nodes that are also dst in deeper blocks) → zero

    x_t = torch.tensor(x, dtype=torch.float32)
    return Data(x=x_t, edge_index=edge_index), gnid_to_local


# ── Feature score aggregation ──────────────────────────────────────────────────

def scores_to_group_importances(
    raw_scores: np.ndarray,      # (d_e,) importance per raw feature
    feature_groups: dict,
) -> np.ndarray:
    """Aggregate per-raw-feature scores to 48 semantic groups (sum of |score|)."""
    groups = feature_groups["groups"]
    group_names = list(groups.keys())
    group_scores = np.zeros(len(group_names), dtype=np.float64)
    for i, name in enumerate(group_names):
        idxs = groups[name]["indices"]
        group_scores[i] = float(np.abs(raw_scores[idxs]).sum())
    return group_scores


# ── Fidelity computation (feature-group space) ─────────────────────────────────

def fidelity_from_group_importances(
    group_scores: np.ndarray,    # (K,) importance scores (higher = more important)
    ctx: FlowContext,
    feature_groups: dict,
    bg_feat: np.ndarray,         # (d_e,) class-conditional background
    top_k: int = 5,
) -> tuple[float, float]:
    """Compute Fidelity+ and Fidelity- from a 48-group importance vector.

    Same masking logic as 08_metrics.py — results are directly comparable.

    Returns:
        (fidelity_plus, fidelity_minus)
        fidelity_plus  = P_full − P(true | top-k masked)   — necessity
        fidelity_minus = P_full − P(true | only top-k kept) — sufficiency
    """
    groups = feature_groups["groups"]
    group_names = list(groups.keys())
    top_k_idx = set(np.argsort(group_scores)[::-1][:top_k].tolist())

    device = ctx._device if hasattr(ctx, "_device") else ctx.x_e_t.device

    def _masked_forward(mask_top_k: bool) -> float:
        masked = ctx.x_e.copy()
        for i, name in enumerate(group_names):
            idxs = groups[name]["indices"]
            is_top = i in top_k_idx
            if mask_top_k and is_top:
                masked[idxs] = bg_feat[idxs]
            elif not mask_top_k and not is_top:
                masked[idxs] = bg_feat[idxs]
        masked_t = torch.tensor(masked, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logit = ctx.h_fixed.__class__  # type probe — unused
            # Reconstruct edge MLP call via stored tensors
            h_src = ctx.h_fixed[ctx.src_pos]
            h_dst = ctx.h_fixed[ctx.dst_pos]
            combined = torch.cat([h_src, h_dst, masked_t], dim=1)
        return combined  # caller must pass through edge_mlp

    # Need edge_mlp reference — stored in the model, passed via ctx
    # We expose a simpler function that receives it explicitly
    raise NotImplementedError(
        "Call fidelity_from_group_importances_with_mlp() which takes edge_mlp as arg."
    )


def fidelity_from_group_importances_with_mlp(
    group_scores: np.ndarray,
    ctx: FlowContext,
    feature_groups: dict,
    bg_feat: np.ndarray,
    edge_mlp: nn.Module,
    top_k: int = 5,
) -> tuple[float, float]:
    """Compute Fidelity+ / Fidelity- given the edge MLP module."""
    groups = feature_groups["groups"]
    group_names = list(groups.keys())
    top_k_idx = set(np.argsort(group_scores)[::-1][:top_k].tolist())
    device = ctx.x_e_t.device

    def _fwd(mask_top_k: bool) -> float:
        masked = ctx.x_e.copy()
        for i, name in enumerate(group_names):
            idxs = groups[name]["indices"]
            is_top = i in top_k_idx
            if mask_top_k and is_top:
                masked[idxs] = bg_feat[idxs]
            elif not mask_top_k and not is_top:
                masked[idxs] = bg_feat[idxs]
        masked_t = torch.tensor(masked, dtype=torch.float32, device=device).unsqueeze(0)
        h_src = ctx.h_fixed[ctx.src_pos]   # (1, hidden)
        h_dst = ctx.h_fixed[ctx.dst_pos]   # (1, hidden)
        combined = torch.cat([h_src, h_dst, masked_t], dim=1)
        with torch.no_grad():
            logit = edge_mlp(combined)
            proba = torch.softmax(logit, dim=1).cpu().numpy()[0]
        return float(proba[ctx.true_label])

    p_masked = _fwd(mask_top_k=True)
    p_kept   = _fwd(mask_top_k=False)
    return (ctx.p_full - p_masked, ctx.p_full - p_kept)


def fidelity_from_node_mask(
    node_scores: np.ndarray,     # (N_input,) importance per node
    ctx: FlowContext,
    bg_node_state: np.ndarray,   # (15,) background node state
    model: "EdgeAwareGraphSAGE",
    top_k: int = 3,
) -> tuple[float, float]:
    """Fidelity for node-level explainers (GNNShap, GraphSVX).

    Masks the top-k most important nodes by replacing their state with
    the class-conditional background node state.

    Returns:
        (fidelity_plus, fidelity_minus)
    """
    top_k_idx = set(np.argsort(np.abs(node_scores))[::-1][:top_k].tolist())
    device = ctx.x_e_t.device

    def _fwd(mask_top_k: bool) -> float:
        nf = ctx.base_node_feats.copy()
        for i in range(len(nf)):
            is_top = i in top_k_idx
            if mask_top_k and is_top:
                nf[i] = bg_node_state
            elif not mask_top_k and not is_top:
                nf[i] = bg_node_state
        nf_t = torch.tensor(nf, dtype=torch.float32, device=device)
        with torch.no_grad():
            h = model.encode(ctx.blocks, nf_t)
            logit = model.classify(h, ctx.x_e_t, ctx.src_pos, ctx.dst_pos)
            proba = torch.softmax(logit, dim=1).cpu().numpy()[0]
        return float(proba[ctx.true_label])

    p_masked = _fwd(mask_top_k=True)
    p_kept   = _fwd(mask_top_k=False)
    return (ctx.p_full - p_masked, ctx.p_full - p_kept)
