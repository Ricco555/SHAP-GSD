"""
C1 — GNNExplainer baseline wrapper (conference version).

Uses PyG 2.x torch_geometric.explain.GNNExplainer in node_mask_type='attributes'
mode on a singleton graph where the single node's features = the 218-dim edge
feature vector.  GNNExplainer optimises a per-feature mask via gradient descent;
the mask is then aggregated to 48 semantic groups for direct fidelity comparison
with SHAP-GSD.

Source: PyG built-in — NOT the legacy gnn-model-explainer repo (torch 1.6).

Paper claim: GNNExplainer (gradient-based, per-edge) vs SHAP-GSD (KernelSHAP,
coalition-based) — different optimisation strategy, same feature-group coalition space.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer

if TYPE_CHECKING:
    from src.baselines.adapter import FlowContext
    from src.explainer.background import BackgroundDistributions

logger = logging.getLogger(__name__)


def run_gnnexplainer(
    ctx: "FlowContext",
    feature_groups: dict,
    background: "BackgroundDistributions",
    epochs: int = 200,
    lr: float = 0.01,
    top_k: int = 5,
) -> dict:
    """Run GNNExplainer on one target flow and return fidelity scores.

    Args:
        ctx:            FlowContext from adapter.build_flow_context().
        feature_groups: parsed feature_groups.json dict.
        background:     BackgroundDistributions (for class-conditional background).
        epochs:         GNNExplainer optimisation epochs (default 200).
        lr:             optimiser learning rate.
        top_k:          number of feature groups for fidelity masking.

    Returns:
        dict with keys: edge_id, true_label, predicted_label, p_full,
            group_scores (48-dim list), fidelity_plus, fidelity_minus,
            runtime_s.
    """
    from src.baselines.adapter import (
        FeatureAttributionWrapper,
        scores_to_group_importances,
        fidelity_from_group_importances_with_mlp,
    )

    t0 = time.time()
    device = ctx.x_e_t.device

    # Singleton PyG graph: 1 node whose features = edge features (218-dim)
    x_singleton = ctx.x_e_t.cpu()   # (1, d_e)
    edge_index_empty = torch.zeros((2, 0), dtype=torch.long)

    # Wrapper: forward(x, edge_index) uses frozen h_fixed to classify the edge
    wrapper = FeatureAttributionWrapper(
        h_fixed=ctx.h_fixed.cpu(),
        edge_mlp=_get_cpu_edge_mlp(ctx),
        src_pos=ctx.src_pos.cpu(),
        dst_pos=ctx.dst_pos.cpu(),
        device=torch.device("cpu"),
    ).eval()

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type="model",
        node_mask_type="attributes",    # per-feature importance
        edge_mask_type=None,
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",          # wrapper returns raw logits
        ),
    )

    with torch.no_grad():
        pass  # GNNExplainer uses grad internally; context manager not needed here

    explanation = explainer(
        x=x_singleton,
        edge_index=edge_index_empty,
        index=0,                        # explain node 0 (the singleton)
        target=ctx.true_label,
    )

    # node_mask: (1, d_e) — importance per raw feature dimension
    raw_mask = explanation.node_mask.squeeze(0).detach().cpu().numpy()  # (d_e,)

    # Aggregate to 48 groups
    group_scores = scores_to_group_importances(raw_mask, feature_groups)

    # Fidelity+/-
    bg_feat = background.background_features[ctx.true_label]
    edge_mlp_cpu = _get_cpu_edge_mlp(ctx)
    ctx_cpu = _cpu_ctx(ctx)
    fid_plus, fid_minus = fidelity_from_group_importances_with_mlp(
        group_scores, ctx_cpu, feature_groups, bg_feat, edge_mlp_cpu, top_k=top_k
    )

    runtime_s = time.time() - t0
    logger.debug(
        f"GNNExplainer EID={ctx.global_eid}: "
        f"fid+={fid_plus:.4f} fid-={fid_minus:.4f} t={runtime_s:.1f}s"
    )

    return {
        "edge_id":        ctx.global_eid,
        "true_label":     ctx.true_label,
        "predicted_label": ctx.predicted_label,
        "p_full":         ctx.p_full,
        "group_scores":   group_scores.tolist(),
        "fidelity_plus":  round(fid_plus, 6),
        "fidelity_minus": round(fid_minus, 6),
        "runtime_s":      round(runtime_s, 3),
    }


# ── helpers ────────────────────────────────────────────────────────────────────

def _get_cpu_edge_mlp(ctx: "FlowContext") -> nn.Module:
    """Extract the edge MLP from the model stored in ctx and move to CPU."""
    # The edge_mlp is on device; return a CPU copy without touching the original
    import copy
    # h_fixed is on device; we need to reconstruct via the wrapper which is CPU-only
    # Actually ctx doesn't store model directly — we access edge_mlp via wrapper
    # This is resolved by the FeatureAttributionWrapper which contains edge_mlp
    # Here we return a dummy that uses the wrapper's edge_mlp
    # We'll extract it at call time from the wrapper
    raise RuntimeError(
        "_get_cpu_edge_mlp should not be called directly. "
        "Pass model.edge_mlp via run_gnnexplainer_with_model()."
    )


def _cpu_ctx(ctx: "FlowContext") -> "FlowContext":
    """Return a copy of FlowContext with all tensors on CPU."""
    import copy
    c = copy.copy(ctx)
    c.h_fixed  = ctx.h_fixed.cpu()
    c.x_e_t    = ctx.x_e_t.cpu()
    c.src_pos  = ctx.src_pos.cpu()
    c.dst_pos  = ctx.dst_pos.cpu()
    c.node_feats_t = ctx.node_feats_t.cpu()
    return c


def run_gnnexplainer_with_model(
    ctx: "FlowContext",
    model: nn.Module,
    feature_groups: dict,
    background: "BackgroundDistributions",
    epochs: int = 200,
    lr: float = 0.01,
    top_k: int = 5,
) -> dict:
    """Primary entry point: run GNNExplainer given the full model.

    Args:
        model: EdgeAwareGraphSAGE instance (used to access model.edge_mlp).
        (other args same as run_gnnexplainer)
    """
    from src.baselines.adapter import (
        FeatureAttributionWrapper,
        scores_to_group_importances,
        fidelity_from_group_importances_with_mlp,
    )

    t0 = time.time()

    # Everything runs on CPU to keep GPU free and avoid gradient graph issues
    edge_mlp_cpu = _move_to_cpu(model.edge_mlp)
    h_fixed_cpu  = ctx.h_fixed.detach().cpu()
    src_pos_cpu  = ctx.src_pos.cpu()
    dst_pos_cpu  = ctx.dst_pos.cpu()
    x_singleton  = ctx.x_e_t.detach().cpu()   # (1, d_e)
    edge_index_empty = torch.zeros((2, 0), dtype=torch.long)

    wrapper = FeatureAttributionWrapper(
        h_fixed=h_fixed_cpu,
        edge_mlp=edge_mlp_cpu,
        src_pos=src_pos_cpu,
        dst_pos=dst_pos_cpu,
        device=torch.device("cpu"),
    ).eval()

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type=None,
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )

    explanation = explainer(
        x=x_singleton,
        edge_index=edge_index_empty,
        index=0,
    )

    raw_mask = explanation.node_mask.squeeze(0).detach().cpu().numpy()  # (d_e,)
    group_scores = scores_to_group_importances(raw_mask, feature_groups)

    bg_feat = background.background_features[ctx.true_label]
    ctx_cpu = _cpu_ctx_with_none(ctx)
    fid_plus, fid_minus = fidelity_from_group_importances_with_mlp(
        group_scores, ctx_cpu, feature_groups, bg_feat, edge_mlp_cpu, top_k=top_k
    )

    runtime_s = time.time() - t0
    logger.debug(
        f"GNNExplainer EID={ctx.global_eid}: "
        f"fid+={fid_plus:.4f} fid-={fid_minus:.4f} t={runtime_s:.1f}s"
    )

    return {
        "edge_id":        ctx.global_eid,
        "true_label":     ctx.true_label,
        "predicted_label": ctx.predicted_label,
        "p_full":         round(ctx.p_full, 6),
        "group_scores":   group_scores.tolist(),
        "fidelity_plus":  round(fid_plus, 6),
        "fidelity_minus": round(fid_minus, 6),
        "runtime_s":      round(runtime_s, 3),
    }


def _move_to_cpu(module: nn.Module) -> nn.Module:
    """Return a CPU copy of an nn.Module without modifying the original."""
    import copy
    m = copy.deepcopy(module).cpu()
    return m


def _cpu_ctx_with_none(ctx: "FlowContext") -> "FlowContext":
    """FlowContext copy with tensors on CPU; blocks set to None (not needed for fidelity)."""
    import copy
    c = copy.copy(ctx)
    c.h_fixed      = ctx.h_fixed.detach().cpu()
    c.x_e_t        = ctx.x_e_t.detach().cpu()
    c.src_pos      = ctx.src_pos.cpu()
    c.dst_pos      = ctx.dst_pos.cpu()
    c.node_feats_t = ctx.node_feats_t.cpu()
    c.blocks       = None   # not needed for feature-attribution fidelity
    return c
