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
  build_h_full       — the (N_local, hidden) surrogate node-embedding matrix
                       shared by PGExplainer/GNNShap/GraphSVX/EdgeSHAPer;
                       every subgraph node gets a real model-derived embedding
  scores_to_group_importances
                     — aggregates 218-dim raw masks to 48 semantic groups
  fidelity_from_group_importances
                     — computes Fidelity+/- from a 48-group importance vector
                       using the same forward-pass logic as 08_metrics.py
  classify_degenerate_attribution
                     — shared all-zero / saturated fallback_reason classifier
  pack_node_baseline_result
                     — the one result-row packer for the four node-coalition
                       baselines (PGExplainer/GNNShap/GraphSVX/EdgeSHAPer)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

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


# ── Surrogate node-embedding matrix (h_full) ──────────────────────────────────

def _resolve_compute_device(model: "EdgeAwareGraphSAGE") -> torch.device:
    """Return the device the model's GNN layers actually live on.

    ``build_h_full`` runs ``model.convs`` / ``model.bns`` in place on the shared
    model object, so the model — not the caller, and not the CPU-by-default
    baseline wrappers — is the authority on where those modules can execute.
    The wrappers deep-copy only ``model.edge_mlp`` to CPU (``_move_to_cpu``);
    the conv/BN stack stays wherever ``scripts/10_baselines.py`` put it.

    Args:
        model: the trained EdgeAwareGraphSAGE whose layers will be run.

    Returns:
        The ``torch.device`` of the model's first parameter.

    Raises:
        AssertionError: if the model has no parameters at all, which would make
            the device unresolvable and any downstream device assert vacuous.
    """
    try:
        p = next(model.parameters())
    except StopIteration:                       # pragma: no cover - defensive
        raise AssertionError(
            "build_h_full cannot resolve a compute device: the model exposes "
            "no parameters."
        )
    return p.device


# Versions build_h_full's NUMERICAL OUTPUT SEMANTICS, not the module.
#
# scripts/10_baselines.py caches PGExplainer's mask-MLP, which is fit on this
# function's output; the cache is silently invalid across a semantics change
# (the MLP's input dim does not change, so nothing raises -- see specs/28).
# The cached checkpoint's sidecar records this value and a retrain is forced
# when it differs.
#
#   v1 -- original: only the two seed rows held real embeddings; every other
#         row was zero-filled.
#   v2 -- commit 01acee1: every subgraph node gets a real, model-derived
#         embedding.
#   v3 -- specs/26 + specs/27: compute device resolved from the model's own
#         parameters, and layer-aligned per-block message passing replacing the
#         flattened, hop-mixed single graph.
#
# BUMP RULE: bump this if the change could make build_h_full return a different
# tensor for identical inputs -- including a rewrite that only changes
# floating-point reduction order. Docstring, comment, logging, type-annotation
# and assertion-message edits do NOT bump it. When in doubt, bump: a spurious
# bump costs one retrain, a missed bump costs a silently wrong Table-2 column.
#
# NOT BUMPED by the seed-row tolerance rework below (relative two-tier check
# replacing the absolute atol=1e-4 assert). Deliberate, and not a "when in doubt"
# case: the rework touches only WHETHER the function raises, never what it
# computes -- not one arithmetic op, dtype, device or reduction order changed, so
# every input produces a byte-identical tensor. The one behaviour that does
# change (inputs that used to raise now return) cannot invalidate a cache either:
# a raise produced NO cached PGExplainer MLP at all, so no sidecar on disk was
# ever fit on an input whose h_full now differs. The staleness guard (specs/28)
# is untouched.
H_FULL_SCHEMA_VERSION: int = 3


# ── build_h_full seed-row agreement tolerances ────────────────────────────────
#
# The check these parametrise (see check_seed_row_agreement) replaced an
# absolute `torch.allclose(..., atol=1e-4)` that hard-failed an entire
# multi-hour Phase-10 job on ordinary GPU float32 noise (A100 job 1007104:
# max|Δ|=1.771e-04 at seed node 7). Root cause: the bound was ABSOLUTE, but the
# quantity it bounds scales with the checkpoint's embedding magnitude, while the
# error itself is scale-INVARIANT -- ~2-4e-7 RELATIVE (2-3 fp32 epsilons) from
# DGL's internal reduction order differing between a bipartite-block pass and a
# homogeneous-graph pass. Same mathematics, different accumulation order: a
# float64 control collapsed the divergence from 2.29e-05 to 6.66e-16, tracking
# machine epsilon exactly.
#
# Criterion: max|Δ| <= ABS_FLOOR + rel_tol * ‖h_fixed_row‖∞, i.e. relative to the
# reference row's own scale with a small additive absolute allowance. The floor
# exists for rows that ReLU legitimately drives to (near) zero, where a pure
# ratio is ill-conditioned; it is additive, not a denominator, so a dead row is
# judged by an absolute 1e-5 rather than by dividing a tiny number by a tinier
# one.
#
# Threshold justification -- both sides stated RELATIVELY, since the thresholds
# are relative (the raw literature numbers below are absolute and were measured
# on an O(1)-scale checkpoint, so they are not directly comparable):
#   * float32 noise floor: 2-4e-7 relative, invariant to scale, device, fanout
#     and determinism flags. REL_WARN = 1e-4 -> 250-500x headroom.
#   * The bug class this guard exists for (Bug 2, specs/26/27, commit 5434795:
#     every conv layer applied over the flattened union graph instead of
#     per-block, giving nodes hop-mixed depth the model never computes) measured
#     1.00-3.63 ABSOLUTE on a fixture whose embedding scale is O(1)
#     (tests/test_baseline_h_full.py PER_BLOCK_VS_FLAT_EPS), i.e. ~O(1)
#     RELATIVE. REL_FAIL = 1e-2 -> ~100x margin, and REL_WARN would flag it
#     ~10,000x over.
#
# LOAD-BEARING ASSUMPTION, stated because leaving it implicit is precisely what
# made the old absolute design fail: Bug 2's relative signature is
# checkpoint-invariant. Its over-aggregation error scales WITH the embedding
# magnitude (it is the same layers' output, just at the wrong depth), exactly as
# the fp32 noise does, so the ~100x fail margin does not evaporate on a
# large-embedding checkpoint the way absolute atol=1e-4's margin did. The A100
# checkpoint's scale is recoverable from its own numbers as
# 1.771e-4 / ~2.5e-7 ≈ 700; under this criterion its warn budget is
# 1e-5 + 1e-4*700 ≈ 0.07 and its fail budget ≈ 7 -- while Bug 2 on that same
# checkpoint would land near 700, still ~100x over.
#
#: Additive absolute allowance, for reference rows ReLU drove to ~zero.
H_FULL_SEED_ABS_FLOOR: float = 1e-5
#: Above this relative error the divergence is logged (never raised).
H_FULL_SEED_REL_WARN: float = 1e-4
#: Above this relative error the divergence is a hard failure.
H_FULL_SEED_REL_FAIL: float = 1e-2

#: Warn-tier log budget. The check runs per seed node, per flow, per baseline;
#: a checkpoint sitting just above REL_WARN would otherwise flood the log of the
#: exact multi-hour job this two-tier design exists to keep alive.
H_FULL_SEED_WARN_LOG_BUDGET: int = 5

_h_full_seed_warn_count: int = 0


def reset_seed_agreement_warn_budget() -> None:
    """Reset the warn-tier log budget counter (test/orchestration hook)."""
    global _h_full_seed_warn_count
    _h_full_seed_warn_count = 0


def check_seed_row_agreement(
    h_row: torch.Tensor,
    h_ref: torch.Tensor,
    gnid: int,
    *,
    abs_floor: float = H_FULL_SEED_ABS_FLOOR,
    rel_warn: float = H_FULL_SEED_REL_WARN,
    rel_fail: float = H_FULL_SEED_REL_FAIL,
) -> float:
    """Check one seed row of the layered pass against ``model.encode()``.

    Two tiers over a single criterion,
    ``max|Δ| <= abs_floor + rel_tol * ‖h_ref‖∞``:

    * at or below the ``rel_warn`` budget — silent pass (expected float32
      reduction-order noise);
    * above ``rel_warn`` but at or below ``rel_fail`` — logged at WARNING, does
      NOT raise, so a numerically noisy but structurally correct checkpoint
      cannot kill a multi-hour Phase-10 job;
    * above ``rel_fail`` — ``AssertionError``.

    The budget is the sole gate; the relative error in the message is a derived,
    human-readable field. Note there is no ``rtol`` hiding anywhere: the old
    ``torch.allclose`` carried an undocumented active ``rtol=1e-5`` on top of its
    ``atol``, which meant the reported row-max element need not have been the
    element that actually failed. Here the reported element IS the failing one by
    construction — it is the argmax of ``|Δ|``, and the budget is per row.

    Args:
        h_row:     (hidden,) seed row from the layered per-block pass.
        h_ref:     (hidden,) the same seed's row of ``ctx.h_fixed`` (the model's
                   own block-computed embedding) — the reference whose ∞-norm
                   sets the scale.
        gnid:      global node ID of the seed, for the message.
        abs_floor: additive absolute allowance for near-zero reference rows.
        rel_warn:  relative threshold above which the divergence is logged.
        rel_fail:  relative threshold above which it is an error.

    Returns:
        The relative error ``max|Δ| / max(‖h_ref‖∞, abs_floor)``.

    Raises:
        AssertionError: if the divergence exceeds the ``rel_fail`` budget.
    """
    global _h_full_seed_warn_count

    delta = (h_row - h_ref).abs()
    k = int(torch.argmax(delta))
    max_abs = float(delta[k])
    scale = float(h_ref.abs().max()) if h_ref.numel() else 0.0
    rel = max_abs / max(scale, abs_floor)

    warn_budget = abs_floor + rel_warn * scale
    fail_budget = abs_floor + rel_fail * scale
    if max_abs <= warn_budget:
        return rel

    def _diagnose(tier: str, budget: float, rel_tol: float) -> str:
        return (
            f"layered pass diverged from model.encode() at seed node {gnid} "
            f"({tier}): element {k} is {float(h_row[k]):.9e} vs h_fixed "
            f"{float(h_ref[k]):.9e}, |Δ|={max_abs:.3e}, over a budget of "
            f"{budget:.3e} (= abs_floor {abs_floor:.1e} + rel_tol {rel_tol:.1e} "
            f"* ‖h_fixed[i]‖∞ {scale:.3e}) by {max_abs - budget:.3e}; "
            f"relative error {rel:.3e} "
            f"(|Δ| / max(‖h_fixed[i]‖∞, {abs_floor:.1e}))"
        )

    if max_abs > fail_budget:
        raise AssertionError(
            _diagnose("hard fail", fail_budget, rel_fail)
            + ". A relative error this large is far above float32 "
            "reduction-order noise (~2-4e-7) and is the signature of a "
            "structural error in the per-block pass, e.g. Bug 2's flattened "
            "hop-mixed aggregation (specs/26, specs/27)."
        )

    _h_full_seed_warn_count += 1
    if _h_full_seed_warn_count <= H_FULL_SEED_WARN_LOG_BUDGET:
        logger.warning(
            "%s. Above expected float32 noise but far below the hard-fail "
            "budget (%.3e), so this is reported, not raised.",
            _diagnose("warn", warn_budget, rel_warn), fail_budget,
        )
        if _h_full_seed_warn_count == H_FULL_SEED_WARN_LOG_BUDGET:
            logger.warning(
                "Further build_h_full seed-row agreement warnings suppressed "
                "(log budget %d reached).", H_FULL_SEED_WARN_LOG_BUDGET,
            )
    return rel


def build_h_full(
    ctx: "FlowContext",
    model: "EdgeAwareGraphSAGE",
    gnid_to_local: dict[int, int],
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Build the (N_local, hidden) node-embedding matrix used by all baselines.

    What is computed — layer-aligned, per-block message passing.  For each
    layer ``i`` a homogeneous local graph ``G_i`` is built over all ``N_local``
    nodes carrying **exactly** ``ctx.blocks[i]``'s edges in local index space,
    and the pass is ``h = relu(bn_i(conv_i(G_i, h)))``.  This mirrors
    ``EdgeAwareGraphSAGE.encode()`` (``src/model/sage_model.py``), which zips
    ``self.convs`` / ``self.bns`` with ``blocks``.  Dropout is omitted because
    this function is eval-mode-only (asserted below), where it is the identity.

    What is exact: for a node that is a genuine *destination* in ``blocks[i]``,
    layer ``i``'s output here equals what ``model.encode()`` computes for it —
    identical in-neighbour multiset, identical layer-(i-1) inputs.  The seed
    nodes are the only nodes for which that holds at every layer, so the
    layered pass's seed rows reproduce ``ctx.h_fixed`` up to float32
    reduction-order noise — checked per flow below by
    ``check_seed_row_agreement``, which logs a small divergence and raises only
    on a large one — and they are then overwritten with ``ctx.h_fixed`` verbatim
    so the surrogate's readout equals the real model prediction exactly.

    Approximation 1 — the self-lift.  For a node that appears in ``blocks[i]``
    only as a *source* (never a destination), the convolution degenerates to
    the self term alone: ``relu(bn_i(fc_self(h)))``.  (Verified for DGL
    ``SAGEConv`` with ``aggregator`` in {``mean``, ``pool``, ``lstm``}; the
    ``gcn`` aggregator has no ``fc_self`` and is not selected by any config in
    this repo.)  The real model computes *nothing* for such a node at that
    depth.  The lift invents **no neighbour messages** — it linearly carries a
    node's deepest genuinely-computed embedding into the final hidden space so
    that ``h_full`` can be one uniform ``(N_local, hidden)`` matrix, which the
    wrappers' indexing requires.  This is a documented approximation, not a
    correctness claim.

    Approximation 2 — the fanout residual.  A 1-hop node's layer-1 embedding
    uses the neighbours sampled under the *target edge's* cutoff and fanout
    budget, not the fresh, node-rooted sample a re-rooted ``encode()`` would
    draw.  Not fixed here, and not fixable without re-rooting.

    Why not per-node re-rooting (the literally-exact-looking option): it costs
    ``O(N_local)`` ``TemporalNeighborSampler.sample_blocks`` calls per flow per
    baseline — the sampler call, not the encode, is the cost — across four
    baselines, the whole Phase-10 flow sample and four datasets.  And it does
    not actually purchase exactness: a re-rooted sample draws a *different*
    neighbour set under that node's own cutoff and fanout budget
    (Approximation 2).

    Device contract: all GNN computation runs on ``_resolve_compute_device``
    (the model's own parameter device), because ``model.convs`` / ``model.bns``
    are executed in place on the shared model object.  The return is always a
    float32 **CPU** tensor: every wrapper's coalition machinery runs on CPU
    against a CPU deep-copy of ``model.edge_mlp``.

    Temporal safety: every edge in every ``ctx.blocks[i]`` comes from the
    temporally filtered blocks (all timestamps <= the target edge's), and every
    node feature is the node state at the target timestamp, so no future
    information enters (CRITICAL INVARIANT 2/3).  The per-block restriction
    *narrows* which of those edges each layer may use, so this is a
    strengthening.

    Args:
        ctx:           FlowContext for the target flow.
        model:         the trained EdgeAwareGraphSAGE (must be in eval mode).
        gnid_to_local: global node ID → local index, from dgl_subgraph_to_pyg.
        edge_index:    (2, E) local-space subgraph edges.  NOT the graph used
                       for message passing any more — it is a **checked
                       consistency guard**: the concatenation of the per-block
                       local edge lists, in block order, must equal it exactly
                       (``torch.equal``), which proves h_full and the coalition
                       graph describe the same object.  Parallel edges are NOT
                       deduplicated — NetFlow is a multigraph and each parallel
                       edge is a distinct flow.

    Returns:
        (N_local, hidden) float32 CPU tensor.  All baseline wrappers run on CPU.
    """
    import dgl as _dgl

    assert not model.training, (
        "build_h_full requires model.eval(): train-mode BatchNorm would compute "
        "batch statistics over a handful of subgraph nodes and dropout would "
        "randomise the surrogate's embeddings."
    )

    device = _resolve_compute_device(model)

    n_local = len(gnid_to_local)
    hidden_size = int(ctx.h_fixed.size(1))
    n_layers = len(model.convs)

    assert len(ctx.blocks) == n_layers, (
        f"block/layer mismatch: len(ctx.blocks)={len(ctx.blocks)} but the model "
        f"has {n_layers} conv layers; a fanouts/num_layers disagreement must "
        "fail loudly, not be silently truncated by zip()."
    )
    assert len(model.bns) == n_layers, (
        f"len(model.bns)={len(model.bns)} != len(model.convs)={n_layers}"
    )
    for name, dev in (
        ("ctx.h_fixed",      ctx.h_fixed.device),
        ("ctx.node_feats_t", ctx.node_feats_t.device),
        ("ctx.blocks[0]",    ctx.blocks[0].device),
        ("ctx.blocks[-1]",   ctx.blocks[-1].device),
    ):
        assert dev == device, (
            f"device mismatch: {name} is on {dev} but the model's layers are on "
            f"{device}; build_h_full runs model.convs/model.bns in place, so "
            "every input must already be on the model's device."
        )

    # Local node features, ordered by local index.  dgl_subgraph_to_pyg's local
    # ordering derives from blocks[0].srcdata[NID] (= input_node_ids, which is
    # also node_feats_t's row order), so every local node must have a row —
    # assert, do not assume.
    input_gnids = ctx.blocks[0].srcdata[_dgl.NID].cpu().tolist()
    input_pos = {g: i for i, g in enumerate(input_gnids)}
    rows: list[int] = []
    for gnid, local_idx in sorted(gnid_to_local.items(), key=lambda kv: kv[1]):
        assert gnid in input_pos, (
            f"local node {gnid} has no row in base_node_feats; the local "
            "subgraph is not a subset of the block input nodes."
        )
        rows.append(input_pos[gnid])
    assert len(rows) == n_local, (
        f"row map has {len(rows)} entries for {n_local} local nodes"
    )
    row_idx = torch.tensor(rows, dtype=torch.long, device=device)
    # Index the already-on-device node-state tensor rather than re-materialising
    # ctx.base_node_feats from numpy on the CPU: saves one host→device copy per
    # flow per call site.  node_feats_t is float32 by construction.
    x_local = ctx.node_feats_t.index_select(0, row_idx)   # (n_local, 15), on device

    # One homogeneous local graph PER BLOCK, over all n_local nodes, carrying
    # exactly that block's edges.  Layer i must message-pass over blocks[i]
    # alone — this is what EdgeAwareGraphSAGE.encode() does, and running every
    # layer over the flattened union gave non-seed nodes aggregation depth the
    # trained model never computes for them.
    block_graphs: list = []
    cat_src: list[int] = []
    cat_dst: list[int] = []
    for block in ctx.blocks:
        b_src_gnids = block.srcdata[_dgl.NID].cpu().tolist()
        b_dst_gnids = block.dstdata[_dgl.NID].cpu().tolist()
        b_srcs, b_dsts = block.edges()
        s_local: list[int] = []
        d_local: list[int] = []
        for s_i, d_i in zip(b_srcs.cpu().tolist(), b_dsts.cpu().tolist()):
            s_local.append(gnid_to_local[b_src_gnids[s_i]])
            d_local.append(gnid_to_local[b_dst_gnids[d_i]])
        cat_src.extend(s_local)
        cat_dst.extend(d_local)
        block_graphs.append(_dgl.graph(
            (torch.tensor(s_local, dtype=torch.int64, device=device),
             torch.tensor(d_local, dtype=torch.int64, device=device)),
            num_nodes=n_local,
            device=device,
        ))

    # Consistency guard: the per-block reconstruction must be exactly the
    # edge_index the caller derived from the SAME ctx.blocks via
    # dgl_subgraph_to_pyg.  If that construction order ever changes, or a caller
    # passes a subsetted/reordered edge_index, h_full and the coalition graph
    # would silently describe different objects.
    recon = torch.stack([
        torch.tensor(cat_src, dtype=torch.int64),
        torch.tensor(cat_dst, dtype=torch.int64),
    ])
    assert torch.equal(recon, edge_index.detach().cpu().long()), (
        "edge_index does not match the per-block reconstruction from "
        "ctx.blocks; h_full and the coalition subgraph would describe "
        "different graphs."
    )

    with torch.no_grad():
        h = x_local
        for g_i, conv, bn in zip(block_graphs, model.convs, model.bns):
            h = conv(g_i, h)
            h = bn(h)
            h = torch.relu(h)          # no dropout: eval-mode inference only
        h_full = h.detach().cpu().float()

    assert h_full.shape == (n_local, hidden_size), (
        f"h_full shape {tuple(h_full.shape)} != ({n_local}, {hidden_size})"
    )

    # Correctness proof, per flow: for a node that is a genuine destination in
    # blocks[i] at every layer i, the per-block pass computes exactly what
    # model.encode() computes.  The seed nodes are the only nodes for which that
    # holds at EVERY layer, so comparing the layered pass's seed rows against
    # ctx.h_fixed BEFORE overwriting them is the strongest available proof that
    # this function reproduces the model.  allclose, not equal: mean aggregation
    # over the same in-neighbour multiset in a different edge order is not
    # bit-identical.
    #
    # The tolerance is RELATIVE and two-tiered — see check_seed_row_agreement and
    # the threshold block above it for the criterion, the measured numbers behind
    # each threshold, and why the previous absolute `torch.allclose(atol=1e-4)`
    # (whose undocumented active rtol=1e-5 is now gone) was the wrong shape of
    # bound for a checkpoint-scale-dependent quantity.
    seed_gnids = ctx.blocks[-1].dstdata[_dgl.NID].cpu().tolist()
    h_fixed_cpu = ctx.h_fixed.detach().cpu().float()
    for i, gnid in enumerate(seed_gnids):
        local_idx = gnid_to_local.get(gnid)
        if local_idx is not None:
            check_seed_row_agreement(h_full[local_idx], h_fixed_cpu[i], gnid)

    # Overwrite the seed rows with the model's authoritative block-computed
    # embeddings so the surrogate's readout equals the real model prediction.
    for i, gnid in enumerate(seed_gnids):
        local_idx = gnid_to_local.get(gnid)
        if local_idx is not None:
            h_full[local_idx] = h_fixed_cpu[i]

    for nid in (ctx.target_src_nid, ctx.target_dst_nid):
        assert nid in gnid_to_local, (
            f"target endpoint {nid} missing from the local subgraph; the "
            "wrappers' readout index would silently fall back to node 0."
        )

    return h_full


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


# ── Degeneracy diagnostics (instrumentation only — no effect on attributions) ──

#: Magnitude below which an attribution vector is treated as numerically
#: uninformative ("saturated") even though it is not exactly zero.  Used only
#: for the diagnostic ``fallback_reason`` field, never for any computation.
SATURATION_EPS: float = 1e-12


def classify_degenerate_attribution(scores: np.ndarray) -> str | None:
    """Classify an attribution vector as all-zero, saturated, or informative.

    Instrumentation only — the returned string goes into the ``fallback_reason``
    column and never influences any attribution or fidelity number.  Shared by
    all four node-coalition baselines so a degenerate row means the same thing
    in every wrapper's output.

    Args:
        scores: attribution vector (any shape; only its magnitudes are read).

    Returns:
        ``"estimator_all_zero"``   if the peak |score| is exactly 0.0;
        ``"estimator_saturated"``  if it is non-zero but below SATURATION_EPS;
        ``None``                   if the vector carries usable magnitude, or is
                                   empty (nothing to classify).
    """
    if scores.size == 0:
        return None
    peak = float(np.max(np.abs(scores)))
    if peak == 0.0:
        return "estimator_all_zero"
    if peak < SATURATION_EPS:
        return "estimator_saturated"
    return None


def surrogate_diagnostics(
    h_full: torch.Tensor,
    edge_index: torch.Tensor,
    src_local: int,
    dst_local: int,
) -> dict:
    """Structural diagnostics for a baseline surrogate's coalition subgraph.

    Purely observational: records how many nodes in the local subgraph carry a
    non-zero embedding in ``h_full`` and how many subgraph edges can actually
    carry a non-zero message into the readout positions — only edges whose
    *source* is a non-zero row can change the surrogate's output under an
    edge/node coalition.

    Historical note: the superseded zero-fill ``_build_h_full`` populated only
    the seed nodes (``blocks[-1].dstdata``) and left every other row at zero,
    which is what these counters were added to expose.  With ``build_h_full``
    every row is model-derived, so ``n_h_full_nonzero == n_local`` and
    ``n_live_edges_into_readout`` degenerates to a plain in-edge count for
    almost every flow.  These fields are therefore now a *regression guard*
    (a zero row reappearing would be a bug), not evidence of sensitivity —
    for that, use ``surrogate_delta`` (packed as ``surrogate_delta_logit``).

    Args:
        h_full:     (N_local, hidden) surrogate node-embedding matrix.
        edge_index: (2, E) local-space subgraph edges.
        src_local:  local index of the target edge's source node (readout).
        dst_local:  local index of the target edge's destination node (readout).

    Returns:
        dict with keys ``n_local``, ``n_h_full_nonzero``, ``n_subgraph_edges``,
        ``n_live_edges_into_readout``, ``n_live_edges_into_src``.
    """
    n_local = int(h_full.size(0))
    nonzero_rows = (h_full.abs().sum(dim=1) > 0)
    n_nonzero = int(nonzero_rows.sum().item())

    n_edges = int(edge_index.size(1))
    if n_edges == 0:
        return {
            "n_local":                    n_local,
            "n_h_full_nonzero":           n_nonzero,
            "n_subgraph_edges":           0,
            "n_live_edges_into_readout":  0,
            "n_live_edges_into_src":      0,
        }

    src_rows, dst_rows = edge_index[0], edge_index[1]
    live_src = nonzero_rows[src_rows]
    into_readout = (dst_rows == src_local) | (dst_rows == dst_local)
    into_src = (dst_rows == src_local)

    return {
        "n_local":                    n_local,
        "n_h_full_nonzero":           n_nonzero,
        "n_subgraph_edges":           n_edges,
        "n_live_edges_into_readout":  int((live_src & into_readout).sum().item()),
        "n_live_edges_into_src":      int((live_src & into_src).sum().item()),
    }


def surrogate_delta(
    predict: Callable[[torch.Tensor], torch.Tensor],
    edge_index: torch.Tensor,
) -> float:
    """Max LOGIT change between the full-coalition and empty-coalition surrogate.

    A value of exactly 0.0 means the surrogate's value function is constant over
    every coalition, so *any* Shapley estimator must return all-zero
    attributions regardless of the estimator's own behaviour.

    **Logit space, not probability space.**  ``predict`` must return the
    surrogate's RAW, pre-normalisation class scores.  Every caller passes a
    logits-only entry point for exactly this reason (``forward_logits`` on the
    GraphSVX/PGExplainer wrappers, ``return_logits=True`` on GNNShap's forward
    factory, and EdgeSHAPer's ``forward``, which is already raw).  In
    probability space the predicate above is not merely inconsistent across
    baselines — it is *wrong*: the wrappers' unnormalised residual aggregation
    drives high-confidence flows outside the edge MLP's operating range, the
    float32 softmax saturates, and ``max |Δp|`` collapses to EXACTLY 0.0 on
    flows whose surrogate is demonstrably not flat (measured: 67/150 real flows
    at 0.0 in probability space, 150/150 non-zero in logit space — see
    ``tests/test_baseline_h_full.py``).  That would report a *false* "provably
    flat surrogate" verdict precisely where the model is most confident.

    Args:
        predict:    callable taking a (2, E) edge_index and returning a 1-D
                    tensor of RAW LOGITS (one per class).
        edge_index: (2, E) local-space subgraph edges.

    Returns:
        ``max |predict(full) - predict(empty)|`` in logit space, or
        ``float("nan")`` if the measurement could not be taken.

        NaN, deliberately not 0.0: 0.0 is a semantically loaded *result*
        ("measured, and the surrogate is provably flat"), so a failed
        measurement must never be able to masquerade as one.  Consumers must
        treat NaN as "unmeasured" and skip it, not coerce it to zero.
    """
    try:
        empty = edge_index.new_zeros((2, 0))
        with torch.no_grad():
            out_full = predict(edge_index).reshape(-1)
            out_empty = predict(empty).reshape(-1)
        return float((out_full - out_empty).abs().max().item())
    except Exception as exc:                                  # diagnostics only
        # WARNING, not DEBUG: a NaN in this column with no log line would be a
        # silent failure of the very instrumentation added to expose one.
        logger.warning(f"surrogate_delta unavailable: {type(exc).__name__}: {exc}")
        return float("nan")


def empty_diagnostics(n_local: int = 0, n_subgraph_edges: int = 0) -> dict:
    """Neutral diagnostic block for flows that never reach ``build_h_full``.

    UNMEASURED, not zero-meaning-degenerate: on the wrappers' early-return
    paths (``fallback_reason`` in ``("empty_subgraph", "insufficient_players")``)
    ``build_h_full`` is deliberately NOT run — it is a multi-layer
    message-passing pass whose result the fallback would immediately discard.
    The three h_full-derived counters (``n_h_full_nonzero``,
    ``n_live_edges_into_readout``, ``n_live_edges_into_src``) are therefore
    reported as 0 because nothing measured them, NOT because a measurement
    found dead rows.  In particular ``n_h_full_nonzero == 0`` on such a row
    does not contradict ``surrogate_diagnostics``' post-``build_h_full``
    ``n_h_full_nonzero == n_local`` regression guard.

    The two structural counters are cheap and ARE real when the caller supplies
    them: ``n_local`` is ``len(gnid_to_local)`` and ``n_subgraph_edges`` is
    ``edge_index.size(1)``, neither of which needs ``h_full``.

    Args:
        n_local:          number of nodes in the local subgraph, if known.
        n_subgraph_edges: number of edges in the local subgraph, if known.

    Returns:
        dict with the same five keys ``surrogate_diagnostics`` returns.
    """
    return {
        "n_local":                    int(n_local),
        "n_h_full_nonzero":           0,
        "n_subgraph_edges":           int(n_subgraph_edges),
        "n_live_edges_into_readout":  0,
        "n_live_edges_into_src":      0,
    }


# ── Shared result packing (node-coalition baselines) ───────────────────────────

def pack_node_baseline_result(
    ctx: FlowContext,
    node_scores: np.ndarray,
    fid_plus: float,
    fid_minus: float,
    runtime_s: float,
    fallback_reason: str | None = None,
    diag: dict | None = None,
) -> dict:
    """Pack one flow's result row for a node-coalition baseline.

    Shared by the four wrappers whose attribution space is the target flow's
    k-hop NODE set (GNNShap, GraphSVX, PGExplainer, EdgeSHAPer).  It is
    deliberately not used by ``gnnexplainer_wrapper``, whose rows live in
    feature-group space and have a different schema.

    ``fallback_reason`` and the ``diag`` fields are additive diagnostic metadata
    (extra CSV columns); they never affect the attribution or fidelity numbers.

    Args:
        ctx:             FlowContext for the target flow.
        node_scores:     (N_input,) per-node importance scores.
        fid_plus:        Fidelity+ for this flow.
        fid_minus:       Fidelity- for this flow.
        runtime_s:       wall-clock seconds attributable to the explainer,
                         with diagnostic instrumentation time already excluded.
        fallback_reason: why this row is degenerate, or None if it is not.
        diag:            diagnostic block from ``surrogate_diagnostics`` or
                         ``empty_diagnostics`` (optionally carrying
                         ``surrogate_delta_logit``); missing keys default to the
                         neutral block with ``surrogate_delta_logit = NaN``.

    Returns:
        dict of CSV row fields for this flow.  The ``surrogate_delta_logit``
        column is named for its measurement space: it was ``surrogate_delta``
        while the four wrappers each measured it in whatever space their own
        explainer library happened to consume (logits / log-probabilities /
        probabilities), which made the column neither comparable across
        baselines nor trustworthy as a flatness verdict.  The rename means no
        pre-existing mixed-space CSV on disk can be silently read as the new,
        uniformly logit-space column.
    """
    d = dict(empty_diagnostics())
    d["surrogate_delta_logit"] = float("nan")
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
