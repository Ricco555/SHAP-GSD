"""
C2 — PGExplainer baseline wrapper (journal version).

PGExplainer (Luo et al. 2020) is an inductive parametric explainer that trains
a small MLP once across many graphs and then predicts edge-importance masks at
inference without per-flow gradient descent.

Attribution space: N_input NODES (not 218 feature dims).
The MLP learns which neighbouring-flow edges in the DGL subgraph are important
for the edge-classification decision.  Edge masks are aggregated to per-node
scores and fidelity is measured via fidelity_from_node_mask (top-k nodes).

Note: GNNExplainer (C1) operates in the 218-dim feature space (singleton-graph
trick).  PGExplainer, GNNShap, and GraphSVX all operate in the node coalition
space; this is noted explicitly in the paper's Table 2 header row.
SHAP-GSD is the only method that covers all three granularities.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.baselines.adapter import FlowContext
    from src.explainer.background import BackgroundDistributions

logger = logging.getLogger(__name__)


def _get_dummy_mp_class():
    """Return a DummyMP(MessagePassing) class.  Lazy import avoids a hard
    torch_geometric dependency at module load time."""
    from torch_geometric.nn import MessagePassing

    class DummyMP(MessagePassing):
        """Identity MessagePassing layer exposed so PyG's get_embeddings()
        hook can capture h_full.  PGExplainer uses the captured output as
        node embeddings for its edge-mask MLP."""
        def __init__(self):
            super().__init__(aggr="add")
            self.add_self_loops = False

        def forward(self, x, edge_index):
            return x  # identity: output = captured node embeddings (h_full)

        def message(self, x_j):
            return x_j

    return DummyMP


# ── PGE-compatible model wrapper ───────────────────────────────────────────────

class PGECompatibleWrapper(nn.Module):
    """PyG-native wrapper for PGExplainer with pre-computed DGL embeddings.

    PGExplainer masks edges in the subgraph and measures how the prediction
    changes.  This wrapper holds frozen DGL node embeddings (h_full) and applies
    a one-hop weighted aggregation when PGExplainer supplies an edge_weight mask.

    A DummyMP layer is included so that PyG's get_embeddings() hook can capture
    h_full as the 'node embeddings' used by PGExplainer's edge-mask MLP.

    forward(x, edge_index, edge_weight=None):
        x           : (N, hidden) — node embeddings (h_full passed by the
                      Explainer framework; same as _h_full)
        edge_index  : (2, E) subgraph edges
        edge_weight : (E,) sigmoid mask predicted by PGExplainer
        returns     : (N, num_classes) edge classification broadcast to N nodes
    """

    def __init__(
        self,
        h_full: torch.Tensor,       # (N, hidden)
        edge_mlp: nn.Module,
        x_e_t: torch.Tensor,        # (1, d_e)
        src_local_idx: int,
        dst_local_idx: int,
        num_nodes: int,
    ) -> None:
        super().__init__()
        DummyMP = _get_dummy_mp_class()
        self._dummy_mp = DummyMP()
        self.register_buffer("_h_full", h_full.detach())
        self.edge_mlp = edge_mlp
        self.register_buffer("_x_e_t", x_e_t.detach())
        self._src_local = src_local_idx
        self._dst_local = dst_local_idx
        self._num_nodes = num_nodes

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        h = x if (x is not None and x.shape == self._h_full.shape) else self._h_full

        # Pass h through the dummy MP so PyG's get_embeddings hook captures it.
        h = self._dummy_mp(h, edge_index)   # identity: h unchanged

        if edge_weight is not None and edge_index.size(1) > 0:
            # Weighted one-hop aggregation: h_new[dst] += edge_weight * h[src]
            h_agg = torch.zeros_like(h)
            src_idx, dst_idx = edge_index[0], edge_index[1]
            msgs = edge_weight.view(-1, 1) * h[src_idx]   # (E, hidden)
            h_agg.index_add_(0, dst_idx, msgs)
            h = h + h_agg                                  # residual connection

        h_src = h[self._src_local].unsqueeze(0)            # (1, hidden)
        h_dst = h[self._dst_local].unsqueeze(0)            # (1, hidden)
        combined = torch.cat([h_src, h_dst, self._x_e_t], dim=1)
        logit = self.edge_mlp(combined)                    # (1, num_classes)
        proba = torch.softmax(logit, dim=1)
        return proba.expand(self._num_nodes, -1).contiguous()


# ── helpers ────────────────────────────────────────────────────────────────────


def _edge_mask_to_node_scores(
    edge_mask: np.ndarray,
    edge_index: torch.Tensor,
    N_local: int,
) -> np.ndarray:
    """Aggregate per-edge importance to per-node scores (sum of incident edges)."""
    scores = np.zeros(N_local, dtype=np.float64)
    if edge_index.size(1) == 0:
        return scores
    src_idx = edge_index[0].cpu().numpy()
    dst_idx = edge_index[1].cpu().numpy()
    for e_i, (s, d) in enumerate(zip(src_idx, dst_idx)):
        scores[s] += edge_mask[e_i]
        scores[d] += edge_mask[e_i]
    return scores


def _map_local_to_input(
    local_scores: np.ndarray,
    blocks: list,
    gnid_to_local: dict,
) -> np.ndarray:
    """Map N_local node scores → N_input scores aligned with ctx.base_node_feats."""
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


# ── Training phase ─────────────────────────────────────────────────────────────

def train_pgexplainer(
    model: nn.Module,
    g_train,
    nsm,
    fs,
    sampler,
    device: torch.device,
    n_train: int = 200,
    epochs: int = 30,
    lr: float = 0.003,
    seed: int = 42,
):
    """Train PGExplainer on a random sample of training flows.

    The PGExplainer MLP input dimension is 2 × hidden_size (concatenation of
    node embeddings at each edge endpoint).  This is the same across all flows,
    so the trained MLP is immediately transferable to test flows.

    Args:
        model:    EdgeAwareGraphSAGE (used for model.edge_mlp and model.encode).
        g_train:  DGL training-split graph.
        nsm:      NodeStateManager.
        fs:       FeatureStore for the training split.
        sampler:  TemporalNeighborSampler.
        device:   Compute device for DGL encoding.
        n_train:  Number of training flows sampled for PGExplainer training.
        epochs:   Training epochs (30 is standard for PGExplainer).
        lr:       Optimiser learning rate.
        seed:     RNG seed for reproducible flow sampling.

    Returns:
        Trained torch_geometric.explain.algorithm.PGExplainer instance.
    """
    import dgl as _dgl
    from torch_geometric.explain.algorithm import PGExplainer
    from torch_geometric.explain.config import (
        ExplainerConfig, ExplanationType,
        ModelConfig, ModelMode, ModelTaskLevel, ModelReturnType,
    )
    from src.baselines.adapter import (
        build_flow_context, build_h_full, dgl_subgraph_to_pyg,
    )

    algorithm = PGExplainer(epochs=epochs, lr=lr)

    # connect() is normally called by the Explainer constructor.
    # When training without an Explainer wrapper we must call it manually.
    algorithm.connect(
        ExplainerConfig(
            explanation_type=ExplanationType.phenomenon,
            node_mask_type=None,
            edge_mask_type="object",
        ),
        ModelConfig(
            mode=ModelMode.multiclass_classification,
            task_level=ModelTaskLevel.node,
            return_type=ModelReturnType.probs,
        ),
    )

    all_geids = g_train.edata[_dgl.EID].numpy()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(all_geids, size=min(n_train, len(all_geids)), replace=False)

    edge_mlp_cpu = _move_to_cpu(model.edge_mlp)

    for epoch in range(epochs):
        total_loss = 0.0
        n_used = 0

        for global_eid in chosen:
            try:
                ctx = build_flow_context(
                    int(global_eid), model, g_train, nsm, fs, sampler, device
                )
            except Exception as exc:
                logger.debug(f"train flow EID={global_eid} skipped: {exc}")
                continue

            pyg_data, gnid_to_local = dgl_subgraph_to_pyg(
                ctx.blocks, None, ctx.base_node_feats
            )
            if pyg_data.edge_index.size(1) == 0:
                continue  # no edges — PGExplainer cannot learn from isolated nodes

            h_full = build_h_full(ctx, model, gnid_to_local, pyg_data.edge_index)
            src_local = gnid_to_local.get(ctx.target_src_nid, 0)
            dst_local = gnid_to_local.get(ctx.target_dst_nid, 0)
            N_local = len(gnid_to_local)

            wrapper = PGECompatibleWrapper(
                h_full, edge_mlp_cpu,
                ctx.x_e_t.detach().cpu(),
                src_local, dst_local, N_local,
            ).eval()

            # PGExplainer indexes target as y[index], so target must have
            # shape (N_local,) — not a single-element scalar tensor.
            target_t = torch.full((N_local,), ctx.true_label, dtype=torch.long)
            index_t  = torch.tensor([src_local], dtype=torch.long)

            try:
                loss = algorithm.train(
                    epoch, wrapper,
                    h_full, pyg_data.edge_index,
                    target=target_t, index=index_t,
                )
                loss_val = float(loss)
                if not (loss_val == loss_val):   # NaN check
                    logger.debug(f"PGExplainer NaN loss EID={global_eid}, skipping")
                    continue
                total_loss += loss_val
                n_used += 1
            except Exception as exc:
                logger.debug(f"PGExplainer train step EID={global_eid}: {exc}")

        avg = total_loss / max(n_used, 1)
        if n_used == 0:
            logger.warning(
                f"PGExplainer epoch {epoch+1}/{epochs}: "
                f"0/{len(chosen)} flows produced a valid loss — MLP may be diverging"
            )
        else:
            logger.info(
                f"PGExplainer epoch {epoch+1}/{epochs}: "
                f"avg_loss={avg:.4f}  n={n_used}"
            )

    logger.info("PGExplainer training complete.")
    return algorithm


# ── Inference ──────────────────────────────────────────────────────────────────

def run_pgexplainer_with_model(
    ctx: "FlowContext",
    model: nn.Module,
    algorithm,
    feature_groups: dict,
    background: "BackgroundDistributions",
    g_test,
    top_k: int = 3,
) -> dict:
    """Run trained PGExplainer on one test flow and return fidelity scores.

    Args:
        ctx:          FlowContext from adapter.build_flow_context().
        model:        EdgeAwareGraphSAGE (for edge_mlp access).
        algorithm:    Trained PGExplainer returned by train_pgexplainer().
        feature_groups: (unused — kept for consistent API with other baselines).
        background:   BackgroundDistributions (for background_node_state).
        g_test:       DGL test graph (passed for API consistency; not directly used).
        top_k:        Number of top nodes for fidelity masking.

    Returns:
        dict with keys: edge_id, true_label, predicted_label, p_full,
            node_scores (N_input-dim list), fidelity_plus, fidelity_minus,
            runtime_s.
    """
    from torch_geometric.explain import Explainer
    from src.baselines.adapter import (
        build_h_full,
        dgl_subgraph_to_pyg,
        fidelity_from_node_mask,
        surrogate_diagnostics,
        surrogate_delta,
        SATURATION_EPS,
    )

    t0 = time.time()
    edge_mlp_cpu = _move_to_cpu(model.edge_mlp)

    pyg_data, gnid_to_local = dgl_subgraph_to_pyg(
        ctx.blocks, None, ctx.base_node_feats
    )
    h_full = build_h_full(ctx, model, gnid_to_local, pyg_data.edge_index)
    src_local = gnid_to_local.get(ctx.target_src_nid, 0)
    dst_local = gnid_to_local.get(ctx.target_dst_nid, 0)
    N_local = len(gnid_to_local)

    bg_node = background.background_node_state[ctx.true_label]   # (15,)

    # Diagnostics (instrumentation only).  Timed separately and excluded from
    # runtime_s so the reported runtime keeps its original meaning.
    _t_diag = time.time()
    diag = surrogate_diagnostics(
        h_full, pyg_data.edge_index, src_local, dst_local
    )
    t0 += time.time() - _t_diag

    # Isolated flow: no neighbours → zero attribution, fallback fidelity
    if pyg_data.edge_index.size(1) == 0:
        node_scores_input = np.zeros(len(ctx.input_node_ids))
        fid_plus, fid_minus = fidelity_from_node_mask(
            node_scores_input, ctx, bg_node, model, top_k=top_k
        )
        return _pack_result(ctx, node_scores_input, fid_plus, fid_minus,
                            time.time() - t0,
                            fallback_reason="empty_subgraph", diag=diag)

    wrapper = PGECompatibleWrapper(
        h_full, edge_mlp_cpu,
        ctx.x_e_t.detach().cpu(),
        src_local, dst_local, N_local,
    ).eval()

    _t_diag = time.time()
    diag["surrogate_delta"] = surrogate_delta(
        lambda ei: wrapper(
            h_full, ei, edge_weight=torch.ones(ei.size(1))
        )[0] if ei.size(1) > 0 else wrapper(h_full, ei)[0],
        pyg_data.edge_index,
    )
    t0 += time.time() - _t_diag

    explainer = Explainer(
        model=wrapper,
        algorithm=algorithm,
        explanation_type="phenomenon",
        node_mask_type=None,
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="probs",
        ),
    )

    explanation = explainer(
        x=h_full,
        edge_index=pyg_data.edge_index,
        index=torch.tensor([src_local], dtype=torch.long),
        target=torch.full((N_local,), ctx.true_label, dtype=torch.long),
    )

    edge_mask_np = explanation.edge_mask.detach().cpu().numpy()
    local_scores = _edge_mask_to_node_scores(
        edge_mask_np, pyg_data.edge_index, N_local
    )
    node_scores_input = _map_local_to_input(local_scores, ctx.blocks, gnid_to_local)

    # No fallback branch exists on PGExplainer's inference path; a degenerate
    # row here means the learned edge-mask MLP itself returned an all-zero or
    # numerically saturated (< SATURATION_EPS) mask.
    fallback_reason: str | None = None
    if node_scores_input.size:
        peak = float(np.max(np.abs(node_scores_input)))
        if peak == 0.0:
            fallback_reason = "estimator_all_zero"
        elif peak < SATURATION_EPS:
            fallback_reason = "estimator_saturated"

    # Fidelity uses the actual DGL model and original ctx (GPU tensors preserved)
    fid_plus, fid_minus = fidelity_from_node_mask(
        node_scores_input, ctx, bg_node, model, top_k=top_k
    )

    runtime_s = time.time() - t0
    logger.debug(
        f"PGExplainer EID={ctx.global_eid}: "
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
