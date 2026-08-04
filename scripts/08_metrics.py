"""
Phase 8 — Quantitative SHAP-GSD metrics for paper Table 2.

Computes Fidelity+, Fidelity−, and Stability on the existing 1,764-flow
explanation set from Phase 6.

Metrics (as defined in research_plan.md §4.6 / §5.4):
  Fidelity+ (sufficiency): P(true_class | all) − P(true_class | top-k masked)
      High → top-k groups are necessary (removing them hurts).
  Fidelity− (necessity):   P(true_class | all) − P(true_class | only top-k kept)
      Low  → top-k groups are sufficient (keeping only them maintains prediction).
  Stability: mean per-group std of SHAP vectors across n_seeds re-runs.
      Low → stable attributions.

Outputs:
  outputs/metrics/fidelity.csv           — per-flow Fidelity+/− (feature groups)
  outputs/metrics/stability.csv          — per-flow stability (50-flow subset)
  outputs/metrics/summary.json           — per-class + overall table values
  outputs/metrics/table2.txt             — ASCII table ready for paper
  outputs/metrics/fidelity_temporal.csv  — per-flow Fidelity+/− (φ_T neighborhood)
  outputs/metrics/summary_temporal.json  — per-class + overall φ_T table values
  outputs/metrics/table2_temporal.txt    — ASCII φ_T table ready for paper
  outputs/metrics/fidelity_novelty.csv   — per-flow Fidelity+/− (node novelty, φ_N)
  outputs/metrics/summary_novelty.json   — per-class + overall φ_N table values
  outputs/metrics/table2_novelty.txt     — ASCII φ_N table ready for paper

Usage:
  python scripts/08_metrics.py --config configs/experiment_unsw.yaml
  python scripts/08_metrics.py --config configs/experiment_unsw.yaml --top-k 10
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import dgl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.feature_store import FeatureStore
from src.model.node_state import NodeStateManager
from src.model.sage_model import EdgeAwareGraphSAGE, build_src_dst_pos
from src.explainer.background import BackgroundDistributions
from src.model.temporal_sampler import TemporalNeighborSampler
from src.explainer.feature_shap import FeatureGroupSHAP
from src.explainer.temporal_shap import TemporalNeighborhoodSHAP
from src.explainer.temporal_fidelity import (
    align_phi_to_records,
    temporal_fidelity_for_flow,
)
from src.explainer.node_shap import (
    NodeNoveltySHAP,
    build_non_target_ids,
    build_non_target_pos,
)
from src.explainer.node_fidelity import align_phi_to_players, novelty_fidelity_for_flow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("shap").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ── model loader (mirrors 06_explain.py) ──────────────────────────────────────

def _load_model(cfg: dict, device: torch.device) -> EdgeAwareGraphSAGE:
    m = cfg["model"]
    artifacts_dir = Path(cfg["output"]["artifacts_dir"])

    best_params_path = artifacts_dir / "best_params.json"
    if best_params_path.exists():
        with open(best_params_path) as f:
            best_params = json.load(f)
        hidden_size = best_params.get("hidden_size", m["hidden_size"])
        num_layers  = best_params.get("num_layers",  m["num_layers"])
        dropout     = best_params.get("dropout",     m["dropout"])
        aggregator  = best_params.get("aggregator",  m["aggregator"])
    else:
        hidden_size = m["hidden_size"]
        num_layers  = m["num_layers"]
        dropout     = m["dropout"]
        aggregator  = m["aggregator"]

    fg_path = Path(cfg["output"]["feature_groups_path"])
    with open(fg_path) as f:
        fg = json.load(f)
    d_e = fg["d_e"]

    label_map_path = artifacts_dir / "label_map.json"
    with open(label_map_path) as f:
        label_map = json.load(f)
    num_classes = len(label_map)

    model = EdgeAwareGraphSAGE(
        node_in_dim=m["node_state_dim"],
        edge_in_dim=d_e,
        hidden_size=hidden_size,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        aggregator=aggregator,
    ).to(device)

    ckpt = artifacts_dir / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    logger.info(f"Model loaded from {ckpt}")
    return model


# ── explanation loader ─────────────────────────────────────────────────────────

def _load_explanations(
    explanations_dir: Path,
    int_to_name: dict[int, str],
) -> list[dict]:
    """Load all per-flow JSON files (non-fixed) from outputs/explanations/."""
    records = []
    for class_dir in sorted(explanations_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for jf in sorted(class_dir.glob("*.json")):
            if jf.stem.endswith("_fixed"):
                continue
            with open(jf) as f:
                d = json.load(f)
            d["_class_name"] = class_name
            records.append(d)
    logger.info(f"Loaded {len(records)} explanation records")
    return records


# ── single-flow forward-pass helper ───────────────────────────────────────────

def _flow_forward(
    global_eid: int,
    true_label: int,
    model: EdgeAwareGraphSAGE,
    g_test: dgl.DGLGraph,
    nsm: NodeStateManager,
    fs: FeatureStore,
    sampler: TemporalNeighborSampler,
    device: torch.device,
) -> tuple[list, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, float]:
    """Sample blocks and precompute reusable tensors for one flow.

    Returns:
        blocks, h_fixed, src_pos, dst_pos, x_e (numpy), p_full (float)
    """
    # Local EID lookup
    global_eids_t = g_test.edata[dgl.EID]
    local_eid = int((global_eids_t == global_eid).nonzero(as_tuple=True)[0][0].item())

    seed_t = torch.tensor([local_eid], dtype=torch.long)
    input_nodes, seed_eids, blocks = sampler.sample_blocks(g_test, seed_t)
    blocks = [b.to(device) for b in blocks]
    input_nodes = input_nodes.to(device)

    target_ts = float(g_test.edata["timestamp"][local_eid].item())
    input_node_ids = input_nodes.cpu().numpy()
    node_feats = np.stack([
        nsm.get_state_at_time(int(nid), target_ts) for nid in input_node_ids
    ])
    node_feats_t = torch.tensor(node_feats, dtype=torch.float32, device=device)

    x_e = fs[global_eid].copy()
    x_e_t = torch.tensor(x_e, dtype=torch.float32, device=device).unsqueeze(0)

    seed_nodes_final = blocks[-1].dstdata[dgl.NID]
    src_pos, dst_pos = build_src_dst_pos(g_test, seed_t, seed_nodes_final)
    src_pos = src_pos.to(device)
    dst_pos = dst_pos.to(device)

    with torch.no_grad():
        h_fixed = model.encode(blocks, node_feats_t)
        logits = model.classify(h_fixed, x_e_t, src_pos, dst_pos)
        proba = torch.softmax(logits, dim=1).cpu().numpy()[0]

    p_full = float(proba[true_label])
    return blocks, h_fixed, src_pos, dst_pos, x_e, p_full


def _masked_proba(
    model: EdgeAwareGraphSAGE,
    h_fixed: torch.Tensor,
    src_pos: torch.Tensor,
    dst_pos: torch.Tensor,
    x_e: np.ndarray,
    true_label: int,
    group_names: list[str],
    groups: dict,
    bg_feat: np.ndarray,
    top_k_idx: np.ndarray,
    mask_top_k: bool,
    device: torch.device,
) -> float:
    """One masked forward pass.

    Args:
        mask_top_k: if True, SET top-k to background (fidelity+ pass).
                    if False, set all EXCEPT top-k to background (fidelity- pass).
    """
    masked = x_e.copy()
    for i, name in enumerate(group_names):
        idxs = groups[name]["indices"]
        is_top_k = i in top_k_idx
        if mask_top_k and is_top_k:
            masked[idxs] = bg_feat[idxs]
        elif not mask_top_k and not is_top_k:
            masked[idxs] = bg_feat[idxs]

    masked_t = torch.tensor(masked, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model.classify(h_fixed, masked_t, src_pos, dst_pos)
        proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return float(proba[true_label])


# ── per-granularity flow context ──────────────────────────────────────────────

class FlowContext(NamedTuple):
    """Everything a per-granularity fidelity pass needs for one flow."""

    local_eid: int
    target_ts_ms: float
    blocks: list
    input_node_ids: np.ndarray      # (N_in,) int, block row order
    base_node_feats: np.ndarray     # (N_in, node_state_dim) float32
    x_e_t: torch.Tensor             # (1, d_e) float32, on device
    src_pos: torch.Tensor
    dst_pos: torch.Tensor
    p_full: float                   # P(true_label | everything present)
    src_nid: int                    # target edge source, global node ID
    dst_nid: int                    # target edge destination, global node ID


def _flow_context(
    global_eid: int,
    true_label: int,
    model: EdgeAwareGraphSAGE,
    g_test: dgl.DGLGraph,
    nsm: NodeStateManager,
    fs: FeatureStore,
    sampler: TemporalNeighborSampler,
    device: torch.device,
) -> FlowContext:
    """Sample blocks and precompute everything a granularity fidelity pass needs.

    Mirrors `_flow_forward`, but additionally returns the local EID, the target
    timestamp, the block input node IDs, the actual node-state matrix, and the
    target edge's endpoints — none of which `_flow_forward` exposes and none of
    which Phase 6 serialises. `_flow_forward`'s signature is deliberately left
    untouched (two existing call sites unpack its six values).

    `input_nodes` is taken directly from the sampler's return value, matching
    how Phase 6 derives it — not reconstructed from `blocks[0].srcdata`.

    `p_full` is computed with the FULL `model.forward()` rather than
    `encode` + `classify`. Numerically identical, but it drops the cached
    embedding, which is unusable for node-state masking (see
    `src/explainer/node_fidelity`).

    Args:
        global_eid: global edge ID of the target flow.
        true_label: class index of the target edge.
        model:      trained EdgeAwareGraphSAGE (eval mode expected).
        g_test:     test-split DGL graph.
        nsm:        NodeStateManager for actual node states.
        fs:         test-split FeatureStore.
        sampler:    deterministic TemporalNeighborSampler.
        device:     torch device.

    Returns:
        A populated FlowContext.
    """
    global_eids_t = g_test.edata[dgl.EID]
    local_eid = int((global_eids_t == global_eid).nonzero(as_tuple=True)[0][0].item())

    seed_t = torch.tensor([local_eid], dtype=torch.long)
    input_nodes, _seed_eids, blocks = sampler.sample_blocks(g_test, seed_t)
    blocks = [b.to(device) for b in blocks]
    input_nodes = input_nodes.to(device)

    target_ts = float(g_test.edata["timestamp"][local_eid].item())
    input_node_ids = input_nodes.cpu().numpy()
    base_node_feats = np.stack([
        nsm.get_state_at_time(int(nid), target_ts) for nid in input_node_ids
    ])
    node_feats_t = torch.tensor(base_node_feats, dtype=torch.float32, device=device)

    x_e = fs[global_eid].copy()
    x_e_t = torch.tensor(x_e, dtype=torch.float32, device=device).unsqueeze(0)

    seed_nodes_final = blocks[-1].dstdata[dgl.NID]
    src_pos, dst_pos = build_src_dst_pos(g_test, seed_t, seed_nodes_final)
    src_pos = src_pos.to(device)
    dst_pos = dst_pos.to(device)

    # Target endpoints — derived exactly as Phase 6 does; not serialised in the
    # explanation JSONs, so they must be re-derived here.
    src_t, dst_t = g_test.find_edges(seed_t)
    src_nid, dst_nid = int(src_t[0]), int(dst_t[0])

    with torch.no_grad():
        logits = model(blocks, node_feats_t, x_e_t, src_pos, dst_pos)
        proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
    p_full = float(proba[true_label])

    assert base_node_feats.shape[0] == len(input_node_ids), (
        f"row misalignment: base_node_feats has {base_node_feats.shape[0]} rows "
        f"but input_node_ids has {len(input_node_ids)}"
    )
    assert src_nid in input_node_ids and dst_nid in input_node_ids, (
        f"target endpoints ({src_nid}, {dst_nid}) not both among the block's "
        f"input nodes — the node-novelty coalition layout would be meaningless"
    )

    return FlowContext(
        local_eid=local_eid,
        target_ts_ms=target_ts,
        blocks=blocks,
        input_node_ids=input_node_ids,
        base_node_feats=base_node_feats,
        x_e_t=x_e_t,
        src_pos=src_pos,
        dst_pos=dst_pos,
        p_full=p_full,
        src_nid=src_nid,
        dst_nid=dst_nid,
    )


# ── fidelity computation ───────────────────────────────────────────────────────

def compute_fidelity(
    records: list[dict],
    model: EdgeAwareGraphSAGE,
    g_test: dgl.DGLGraph,
    nsm: NodeStateManager,
    fs: FeatureStore,
    background: BackgroundDistributions,
    feature_groups: dict,
    sampler: TemporalNeighborSampler,
    device: torch.device,
    top_k: int = 5,
) -> list[dict]:
    """Compute per-flow Fidelity+ and Fidelity− for all explanation records."""
    group_names: list[str] = list(feature_groups["groups"].keys())
    groups: dict = feature_groups["groups"]

    rows = []
    n = len(records)
    t0 = time.time()

    for i, rec in enumerate(records):
        global_eid = rec["edge_id"]
        true_label = rec["true_label"]
        class_name = rec["_class_name"]
        shap_arr = np.array(rec["feature_group_shap"])

        try:
            blocks, h_fixed, src_pos, dst_pos, x_e, p_full = _flow_forward(
                global_eid, true_label, model, g_test, nsm, fs, sampler, device
            )
        except Exception:
            logger.warning(f"Skipping EID {global_eid}: block sampling failed")
            continue

        bg_feat = background.background_features[true_label]  # (d_e,)
        top_k_idx = np.argsort(np.abs(shap_arr))[::-1][:top_k]
        top_k_groups = [group_names[j] for j in top_k_idx]

        p_masked = _masked_proba(
            model, h_fixed, src_pos, dst_pos, x_e, true_label,
            group_names, groups, bg_feat, top_k_idx, mask_top_k=True, device=device,
        )
        p_kept = _masked_proba(
            model, h_fixed, src_pos, dst_pos, x_e, true_label,
            group_names, groups, bg_feat, top_k_idx, mask_top_k=False, device=device,
        )

        rows.append({
            "class_name":      class_name,
            "edge_id":         global_eid,
            "true_label":      true_label,
            "predicted_label": rec["predicted_label"],
            "p_full":          round(p_full, 6),
            "p_masked":        round(p_masked, 6),
            "p_kept":          round(p_kept, 6),
            "fidelity_plus":   round(p_full - p_masked, 6),
            "fidelity_minus":  round(p_full - p_kept, 6),
            "top_k_groups":    "|".join(top_k_groups),
            "runtime_s":       rec.get("runtime_s", None),
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            logger.info(
                f"  Fidelity: {i+1}/{n} flows done  "
                f"({elapsed:.0f}s elapsed, {elapsed/(i+1)*1000:.0f}ms/flow)"
            )

    logger.info(f"Fidelity: {len(rows)}/{n} flows computed")
    return rows


# ── temporal-neighborhood (φ_T) fidelity ───────────────────────────────────────

def _round6(x: float | None) -> float | None:
    """Round to 6 decimals, passing None through (empty CSV cell)."""
    return None if x is None else round(x, 6)


def compute_fidelity_temporal(
    records: list[dict],
    model: EdgeAwareGraphSAGE,
    g_test: dgl.DGLGraph,
    nsm: NodeStateManager,
    fs: FeatureStore,
    background: BackgroundDistributions,
    sampler: TemporalNeighborSampler,
    device: torch.device,
    top_k: int = 3,
) -> tuple[list[dict], dict]:
    """Per-flow Fidelity+/− for the temporal-neighborhood granularity (φ_T).

    `background` is required only to satisfy
    `TemporalNeighborhoodSHAP.__init__`; φ_T absence is node-state rollback, not
    background substitution (see spec §1.2).

    Args:
        records:  Phase 6 explanation records (with neighbor_edge_ids /
                  neighbor_shap fields).
        model:    trained EdgeAwareGraphSAGE (eval mode expected).
        g_test:   test split DGL graph.
        nsm:      NodeStateManager for node-state queries and rollbacks.
        fs:       FeatureStore for target edge features.
        background: BackgroundDistributions — see note above.
        sampler:  deterministic TemporalNeighborSampler.
        device:   torch device.
        top_k:    number of top-|φ_T| neighbors to mask / keep (default 3).

    Returns:
        (rows, diagnostics) where diagnostics counts skipped/degenerate flows.
    """
    temp_shap = TemporalNeighborhoodSHAP(background, nsm, g_test, device)
    model.eval()

    rows: list[dict] = []
    n = len(records)
    n_sample_fail = 0
    n_neighborhood_mismatch = 0
    n_zero_neighborhood = 0
    n_degenerate_minus = 0
    t0 = time.time()

    for i, rec in enumerate(records):
        global_eid = rec["edge_id"]
        true_label = rec["true_label"]
        class_name = rec["_class_name"]
        stored_eids = rec["neighbor_edge_ids"]
        stored_phi = rec["neighbor_shap"]

        # The shared helper also asserts that both target endpoints are among
        # the block's input nodes — an invariant φ_T does not itself rely on.
        # φ_T keeps a single broad `except`, so were it ever to fire here it
        # would be counted under n_sample_fail; the accounting invariant below
        # still holds.
        try:
            ctx = _flow_context(
                global_eid, true_label, model, g_test, nsm, fs, sampler, device
            )
        except Exception:
            logger.warning(f"Skipping EID {global_eid}: block sampling failed")
            n_sample_fail += 1
            continue

        neighbor_records = temp_shap.extract_neighbor_edges(
            ctx.blocks, ctx.local_eid, ctx.target_ts_ms
        )
        phi = align_phi_to_records(neighbor_records, stored_eids, stored_phi)
        if phi is None:
            n_neighborhood_mismatch += 1
            continue

        res = temporal_fidelity_for_flow(
            model=model,
            temp_shap=temp_shap,
            blocks=ctx.blocks,
            neighbor_records=neighbor_records,
            phi_temporal=phi,
            base_node_feats=ctx.base_node_feats,
            input_node_ids=ctx.input_node_ids,
            target_ts_ms=ctx.target_ts_ms,
            x_e_t=ctx.x_e_t,
            src_pos=ctx.src_pos,
            dst_pos=ctx.dst_pos,
            true_label=true_label,
            p_full=ctx.p_full,
            top_k=top_k,
            device=device,
        )

        rows.append({
            "class_name":          class_name,
            "edge_id":             global_eid,
            "true_label":          true_label,
            "predicted_label":     rec["predicted_label"],
            "p_full":              _round6(res["p_full"]),
            "p_masked":            _round6(res["p_masked"]),
            "p_kept":              _round6(res["p_kept"]),
            "fidelity_plus":       _round6(res["fidelity_plus"]),
            "fidelity_minus":      _round6(res["fidelity_minus"]),
            "n_neighbors":         res["n_neighbors"],
            "effective_k":         res["effective_k"],
            "top_k_neighbor_eids": "|".join(str(e) for e in res["top_k_neighbor_eids"]),
            "runtime_temporal_s":  rec.get("runtime_temporal_s", None),
        })

        if res["n_neighbors"] == 0:
            n_zero_neighborhood += 1
        elif res["n_neighbors"] <= top_k:
            n_degenerate_minus += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            logger.info(
                f"  Temporal fidelity: {i+1}/{n} flows done  "
                f"({elapsed:.0f}s elapsed, {elapsed/(i+1)*1000:.0f}ms/flow)"
            )

    assert n == len(rows) + n_sample_fail + n_neighborhood_mismatch, (
        f"flow accounting mismatch: {n} records != {len(rows)} rows + "
        f"{n_sample_fail} sample failures + {n_neighborhood_mismatch} mismatches"
    )

    diagnostics = {
        "n_records":               n,
        "n_rows":                  len(rows),
        "n_sample_fail":           n_sample_fail,
        "n_neighborhood_mismatch": n_neighborhood_mismatch,
        "n_zero_neighborhood":     n_zero_neighborhood,
        "n_degenerate_minus":      n_degenerate_minus,
        "top_k":                   top_k,
    }
    logger.info(
        f"Temporal fidelity: {len(rows)}/{n} flows computed "
        f"({n_zero_neighborhood} empty neighborhoods, "
        f"{n_degenerate_minus} with N<=k)"
    )
    return rows, diagnostics


# ── node-novelty fidelity computation ─────────────────────────────────────────

def compute_fidelity_novelty(
    records: list[dict],
    model: EdgeAwareGraphSAGE,
    g_test: dgl.DGLGraph,
    nsm: NodeStateManager,
    fs: FeatureStore,
    background: BackgroundDistributions,
    sampler: TemporalNeighborSampler,
    device: torch.device,
    top_k: int = 5,
) -> tuple[list[dict], dict]:
    """Per-flow Fidelity+/− for the node-novelty granularity (φ_N).

    `background` is load-bearing: it supplies background_node_state[true_class]
    for the non-target-node replacement mechanism. Pass the real, training-fit
    object — never None, never a zeros array.

    Args:
        records:  Phase-6 explanation records (with `_class_name` injected).
        model:    trained EdgeAwareGraphSAGE.
        g_test:   test-split DGL graph.
        nsm:      NodeStateManager for actual node states.
        fs:       test-split FeatureStore.
        background: training-fit BackgroundDistributions.
        sampler:  deterministic TemporalNeighborSampler.
        device:   torch device.
        top_k:    number of top-|φ_N| players to select per flow.

    Returns:
        (rows, diagnostics) where diagnostics counts skipped/degenerate flows.
    """
    model.eval()
    node_shap = NodeNoveltySHAP(background, device)

    rows: list[dict] = []
    n = len(records)
    n_sample_fail = 0
    n_context_assert_fail = 0
    n_player_set_mismatch = 0
    n_degenerate_minus = 0
    n_flows_with_novelty_in_top_k = 0
    t0 = time.time()

    for i, rec in enumerate(records):
        global_eid = rec["edge_id"]
        true_label = rec["true_label"]
        class_name = rec["_class_name"]
        stored_node_ids = rec["node_ids"]
        stored_node_phi = rec["node_shap"]
        src_novelty_phi = rec["src_novelty_shap"]
        dst_novelty_phi = rec["dst_novelty_shap"]

        # AssertionError is caught first so that a broken player layout is not
        # mislabelled as a block-sampling failure.
        try:
            ctx = _flow_context(
                global_eid, true_label, model, g_test, nsm, fs, sampler, device
            )
        except AssertionError as exc:
            logger.warning(
                f"Skipping EID {global_eid}: flow-context invariant failed: {exc}"
            )
            n_context_assert_fail += 1
            continue
        except Exception:
            logger.warning(
                f"Skipping EID {global_eid}: flow-context construction failed"
            )
            n_sample_fail += 1
            continue

        non_target_ids = build_non_target_ids(
            ctx.input_node_ids, ctx.src_nid, ctx.dst_nid
        )
        non_target_pos = build_non_target_pos(non_target_ids)

        phi = align_phi_to_players(
            non_target_ids, stored_node_ids, stored_node_phi,
            src_novelty_phi, dst_novelty_phi,
        )
        if phi is None:
            logger.warning(f"Skipping EID {global_eid}: player-set mismatch")
            n_player_set_mismatch += 1
            continue

        res = novelty_fidelity_for_flow(
            model=model,
            node_shap=node_shap,
            blocks=ctx.blocks,
            phi_novelty=phi,
            base_node_feats=ctx.base_node_feats,
            input_node_ids=ctx.input_node_ids,
            src_nid=ctx.src_nid,
            dst_nid=ctx.dst_nid,
            non_target_pos=non_target_pos,
            x_e_t=ctx.x_e_t,
            src_pos=ctx.src_pos,
            dst_pos=ctx.dst_pos,
            true_label=true_label,
            p_full=ctx.p_full,
            top_k=top_k,
            device=device,
        )

        rows.append({
            "class_name":         class_name,
            "edge_id":            global_eid,
            "true_label":         true_label,
            "predicted_label":    rec["predicted_label"],
            "p_full":             round(res["p_full"], 6),
            "p_masked":           round(res["p_masked"], 6),
            "p_kept":             round(res["p_kept"], 6),
            "fidelity_plus":      round(res["fidelity_plus"], 6),
            "fidelity_minus":     round(res["fidelity_minus"], 6),
            "n_players":          res["n_players"],
            "n_non_target":       res["n_non_target"],
            "effective_k":        res["effective_k"],
            "n_novelty_in_top_k": res["n_novelty_in_top_k"],
            "top_k_players":      "|".join(res["top_k_players"]),
            "runtime_node_s":     rec.get("runtime_node_s", None),
        })

        if res["n_players"] <= top_k:
            n_degenerate_minus += 1
        if res["n_novelty_in_top_k"] > 0:
            n_flows_with_novelty_in_top_k += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            logger.info(
                f"  Novelty fidelity: {i+1}/{n} flows done  "
                f"({elapsed:.0f}s elapsed, {elapsed/(i+1)*1000:.0f}ms/flow)"
            )

    assert n == len(rows) + n_sample_fail + n_context_assert_fail + n_player_set_mismatch, (
        f"record accounting broken: {n} records != {len(rows)} rows + "
        f"{n_sample_fail} sample fails + {n_context_assert_fail} assert fails + "
        f"{n_player_set_mismatch} player-set mismatches"
    )

    diagnostics = {
        "n_records":                     n,
        "n_rows":                        len(rows),
        "n_sample_fail":                 n_sample_fail,
        "n_context_assert_fail":         n_context_assert_fail,
        "n_player_set_mismatch":         n_player_set_mismatch,
        "n_degenerate_minus":            n_degenerate_minus,
        "n_flows_with_novelty_in_top_k": n_flows_with_novelty_in_top_k,
        "mean_n_players": (
            round(float(np.mean([r["n_players"] for r in rows])), 4) if rows else 0.0
        ),
        "top_k":                         top_k,
    }

    logger.info(
        f"Novelty fidelity: {len(rows)}/{n} flows computed "
        f"({n_degenerate_minus} with P<=k, {n_flows_with_novelty_in_top_k} with a "
        f"novelty player in top-{top_k})"
    )
    return rows, diagnostics


# ── stability computation ──────────────────────────────────────────────────────

def compute_stability(
    records: list[dict],
    model: EdgeAwareGraphSAGE,
    g_test: dgl.DGLGraph,
    nsm: NodeStateManager,
    fs: FeatureStore,
    background: BackgroundDistributions,
    feature_groups: dict,
    sampler: TemporalNeighborSampler,
    device: torch.device,
    n_seeds: int = 3,
    nsamples: int = 512,
    n_per_class: int = 5,
) -> list[dict]:
    """Re-run FeatureGroupSHAP n_seeds times on a subset; measure phi variance."""
    # Select first n_per_class flows per class
    per_class: dict[str, list[dict]] = {}
    for rec in records:
        cls = rec["_class_name"]
        if cls not in per_class:
            per_class[cls] = []
        if len(per_class[cls]) < n_per_class:
            per_class[cls].append(rec)

    subset = [r for recs in per_class.values() for r in recs]
    logger.info(
        f"Stability: {len(subset)} flows ({n_seeds} seeds × nsamples={nsamples})"
    )

    feat_shap = FeatureGroupSHAP(feature_groups, background, device)
    rows = []

    for i, rec in enumerate(subset):
        global_eid = rec["edge_id"]
        true_label = rec["true_label"]
        class_name = rec["_class_name"]

        try:
            blocks, h_fixed, src_pos, dst_pos, x_e, p_full = _flow_forward(
                global_eid, true_label, model, g_test, nsm, fs, sampler, device
            )
            node_feats_needed = True
        except Exception:
            logger.warning(f"Stability: skipping EID {global_eid}")
            continue

        # Re-build node_feats_t (needed by feat_shap.explain)
        global_eids_t = g_test.edata[dgl.EID]
        local_eid = int((global_eids_t == global_eid).nonzero(as_tuple=True)[0][0].item())
        target_ts = float(g_test.edata["timestamp"][local_eid].item())
        seed_t = torch.tensor([local_eid], dtype=torch.long)
        input_nodes_cpu = blocks[0].srcdata[dgl.NID].cpu()
        # input_nodes are block[0].srcdata since that's the deepest input level
        # but _flow_forward already gave us blocks; we need to get input node IDs
        # We can find them from blocks[0].srcdata[dgl.NID]
        input_node_ids = blocks[0].srcdata[dgl.NID].cpu().numpy()
        node_feats = np.stack([
            nsm.get_state_at_time(int(nid), target_ts) for nid in input_node_ids
        ])
        node_feats_t = torch.tensor(node_feats, dtype=torch.float32, device=device)

        phi_runs: list[np.ndarray] = []
        for seed in range(n_seeds):
            np.random.seed(seed * 1000 + i)
            phi_dict, _f_baseline, _f_logit = feat_shap.explain(
                true_class=true_label,
                model=model,
                blocks=blocks,
                node_feats=node_feats_t,
                x_e=x_e,
                src_pos=src_pos,
                dst_pos=dst_pos,
                nsamples=nsamples,
            )
            phi_runs.append(np.array(list(phi_dict.values())))

        phi_matrix = np.stack(phi_runs)  # (n_seeds, K)
        per_group_std = phi_matrix.std(axis=0)   # (K,)
        mean_std = float(per_group_std.mean())
        max_std  = float(per_group_std.max())

        rows.append({
            "class_name": class_name,
            "edge_id":    global_eid,
            "mean_phi_std": round(mean_std, 6),
            "max_phi_std":  round(max_std, 6),
        })
        logger.info(
            f"  Stability [{i+1}/{len(subset)}] {class_name} EID={global_eid}: "
            f"mean_std={mean_std:.4f}"
        )

    return rows


# ── summary + table formatting ─────────────────────────────────────────────────

def _build_summary(
    fidelity_rows: list[dict],
    stability_rows: list[dict],
    top_k: int,
) -> dict:
    """Aggregate per-class and overall means for Table 2."""
    import collections

    fid_by_class: dict[str, list] = collections.defaultdict(list)
    for r in fidelity_rows:
        fid_by_class[r["class_name"]].append(r)

    stab_by_class: dict[str, list] = collections.defaultdict(list)
    for r in stability_rows:
        stab_by_class[r["class_name"]].append(r)

    all_classes = sorted(fid_by_class.keys())
    per_class = {}
    for cls in all_classes:
        frows = fid_by_class[cls]
        srows = stab_by_class.get(cls, [])
        fid_plus  = [r["fidelity_plus"]  for r in frows]
        fid_minus = [r["fidelity_minus"] for r in frows]
        stab      = [r["mean_phi_std"]   for r in srows]
        runtimes  = [r["runtime_s"] for r in frows
                     if r.get("runtime_s") is not None and r["runtime_s"] >= 0]
        per_class[cls] = {
            "n_flows":          len(frows),
            "fidelity_plus":    round(float(np.mean(fid_plus)),  4),
            "fidelity_plus_std": round(float(np.std(fid_plus)),  4),
            "fidelity_minus":   round(float(np.mean(fid_minus)), 4),
            "fidelity_minus_std": round(float(np.std(fid_minus)),4),
            "stability":        round(float(np.mean(stab)), 6) if stab else None,
            "n_stability":      len(srows),
            "runtime_mean_s":   round(float(np.mean(runtimes)), 4) if runtimes else None,
        }

    all_fp  = [r["fidelity_plus"]  for r in fidelity_rows]
    all_fm  = [r["fidelity_minus"] for r in fidelity_rows]
    all_st  = [r["mean_phi_std"]   for r in stability_rows]
    # Exclude negative runtimes (WSL2 clock jitter artefacts)
    all_rt  = [r["runtime_s"] for r in fidelity_rows
               if r.get("runtime_s") is not None and r["runtime_s"] >= 0]
    overall = {
        "n_flows":           len(fidelity_rows),
        "fidelity_plus":     round(float(np.mean(all_fp)),  4),
        "fidelity_plus_std": round(float(np.std(all_fp)),   4),
        "fidelity_minus":    round(float(np.mean(all_fm)),  4),
        "fidelity_minus_std":round(float(np.std(all_fm)),   4),
        "stability":         round(float(np.mean(all_st)),  6) if all_st else None,
        "n_stability":       len(stability_rows),
        "runtime_mean_s":    round(float(np.mean(all_rt)),  4) if all_rt else None,
    }

    return {"top_k": top_k, "per_class": per_class, "overall": overall}


def _format_table(summary: dict, runtime_by_layer: dict | None = None) -> str:
    """ASCII table for paper Table 2 (SHAP-GSD row)."""
    k = summary["top_k"]
    ov = summary["overall"]
    lines = [
        f"SHAP-GSD feature-group metrics  (k={k}, n={ov['n_flows']} flows)",
        "",
        f"{'Class':<14} {'N':>5}  {'Fidelity+':>10}  {'Fidelity−':>10}  {'Stability':>10}",
        "-" * 56,
    ]
    for cls, v in summary["per_class"].items():
        stab_str = f"{v['stability']:.4f}" if v["stability"] is not None else "—"
        lines.append(
            f"{cls:<14} {v['n_flows']:>5}  "
            f"{v['fidelity_plus']:>8.4f}    "
            f"{v['fidelity_minus']:>8.4f}    "
            f"{stab_str:>10}"
        )
    lines.append("-" * 56)
    stab_ov = f"{ov['stability']:.4f}" if ov["stability"] is not None else "—"
    lines.append(
        f"{'Overall':<14} {ov['n_flows']:>5}  "
        f"{ov['fidelity_plus']:>8.4f}±{ov['fidelity_plus_std']:.4f}  "
        f"{ov['fidelity_minus']:>8.4f}±{ov['fidelity_minus_std']:.4f}  "
        f"{stab_ov:>10}"
    )
    lines.append("")
    lines.append(
        "Fidelity+: P_full − P_masked_top_k  (higher = top-k groups more necessary)"
    )
    lines.append(
        "Fidelity−: P_full − P_kept_top_k    (lower  = top-k groups more sufficient)"
    )
    lines.append(
        "Stability: mean per-group φ std across 3 coalition seeds (lower = more stable)"
    )

    # LOCKED 2026-08-04 (coder_instructions_figure_determinism.md S2): per-flow
    # explanation cost is reported end-to-end as the PRIMARY figure, because
    # that is the number that is like-for-like against the baseline explainers'
    # own `runtime_s` column (they report a single per-flow wall-clock total,
    # not a per-layer breakdown). The per-layer breakdown (feature/temporal/
    # node) is kept as a secondary row so the whole-method cost stays
    # decomposable, sourced from runtime_by_layer.json (written by
    # 06_explain.py from each explanation's runtime_feature_s/
    # runtime_temporal_s/runtime_node_s/runtime_s fields).
    if ov.get("runtime_mean_s") is not None:
        lines.append("")
        lines.append(
            f"Per-flow explanation cost (PRIMARY, end-to-end, n={ov['n_flows']} flows, "
            f"like-for-like against baselines): {ov['runtime_mean_s']:.3f} s/flow"
        )
        if runtime_by_layer:
            feat = runtime_by_layer.get("feature_group", {}).get("mean_s")
            temp = runtime_by_layer.get("temporal_neighborhood", {}).get("mean_s")
            node = runtime_by_layer.get("node_novelty", {}).get("mean_s")
            n_rt = runtime_by_layer.get("n_flows")
            if None not in (feat, temp, node):
                lines.append(
                    f"  Secondary — per-layer breakdown (n={n_rt} flows, sums to the "
                    f"end-to-end total): feature={feat:.4f}s + temporal={temp:.4f}s + "
                    f"node={node:.4f}s = {feat + temp + node:.4f}s"
                )
    return "\n".join(lines)


def _aggregate_temporal(rows: list[dict]) -> dict:
    """Mean/std of Fidelity+/− over a subset of φ_T rows.

    Args:
        rows: φ_T fidelity rows, all with non-None fidelity values.

    Returns:
        Dict with n, fidelity_plus, fidelity_plus_std, fidelity_minus,
        fidelity_minus_std — all None when `rows` is empty.
    """
    if not rows:
        return {
            "n": 0,
            "fidelity_plus": None,
            "fidelity_plus_std": None,
            "fidelity_minus": None,
            "fidelity_minus_std": None,
        }
    fp = [r["fidelity_plus"] for r in rows]
    fm = [r["fidelity_minus"] for r in rows]
    return {
        "n":                  len(rows),
        "fidelity_plus":      round(float(np.mean(fp)), 4),
        "fidelity_plus_std":  round(float(np.std(fp)),  4),
        "fidelity_minus":     round(float(np.mean(fm)), 4),
        "fidelity_minus_std": round(float(np.std(fm)),  4),
    }


def _build_summary_temporal(
    rows: list[dict],
    diagnostics: dict,
    top_k: int,
) -> dict:
    """Aggregate per-class and overall φ_T fidelity for the paper's φ_T row.

    Flows with no in-window neighbors (N == 0) are EXCLUDED from every mean —
    they are counted as n_zero_neighborhood instead. Two aggregates are
    reported per class and overall:

      all_nonempty — every flow with N >= 1.
      informative  — flows with N > top_k, i.e. those where Fidelity− is not
                     0 by construction.

    The headline (flat) fidelity_plus comes from `all_nonempty` — masking all N
    neighbors when N <= k is a legitimate full-neighborhood ablation. The
    headline fidelity_minus comes from `informative`, because at N <= k it is 0
    by arithmetic identity and would deflate the mean with no content.

    Denominators differ by key: the fidelity means are over N >= 1 rows (that is
    also what n_flows counts), while `mean_n_neighbors` is over ALL rows
    including N == 0, so it matches the reference neighborhood-size statistic.

    Args:
        rows:        φ_T fidelity rows from compute_fidelity_temporal.
        diagnostics: the diagnostics dict from the same call, copied verbatim.
        top_k:       the --top-k-temporal value actually used.

    Returns:
        Summary dict with top_k, per_class, overall and diagnostics keys.
    """
    import collections

    by_class: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        by_class[r["class_name"]].append(r)

    def _class_block(crows: list[dict]) -> dict:
        nonempty = [r for r in crows if r["n_neighbors"] >= 1]
        informative = [r for r in crows if r["n_neighbors"] > top_k]
        agg_all = _aggregate_temporal(nonempty)
        agg_inf = _aggregate_temporal(informative)
        return {
            "n_flows":             len(nonempty),
            "fidelity_plus":       agg_all["fidelity_plus"],
            "fidelity_plus_std":   agg_all["fidelity_plus_std"],
            "fidelity_minus":      agg_inf["fidelity_minus"],
            "fidelity_minus_std":  agg_inf["fidelity_minus_std"],
            "n_zero_neighborhood": sum(1 for r in crows if r["n_neighbors"] == 0),
            "n_informative":       len(informative),
            "mean_n_neighbors":    (round(float(np.mean([r["n_neighbors"] for r in crows])), 4)
                                    if crows else None),
            "all_nonempty":        agg_all,
            "informative":         agg_inf,
        }

    per_class = {cls: _class_block(by_class[cls]) for cls in sorted(by_class.keys())}

    overall = _class_block(rows)
    overall["n_flows_total"] = diagnostics["n_records"]

    return {
        "top_k":       top_k,
        "per_class":   per_class,
        "overall":     overall,
        "diagnostics": dict(diagnostics),
    }


def _format_table_temporal(summary: dict) -> str:
    """ASCII table for the paper's temporal-neighborhood (φ_T) row.

    The reported Fidelity+ is the all-nonempty mean and the reported Fidelity−
    is the informative (N > k) mean; the secondary columns show the other
    subset so the headline numbers are unambiguous.

    Args:
        summary: output of _build_summary_temporal.

    Returns:
        Formatted multi-line table string.
    """
    def _f(x: float | None) -> str:
        return f"{x:.4f}" if x is not None else "—"

    k = summary["top_k"]
    ov = summary["overall"]
    diag = summary["diagnostics"]
    lines = [
        f"SHAP-GSD temporal-neighborhood metrics  (k={k}, "
        f"n={ov['n_flows']} of {ov['n_flows_total']} flows with N>=1)",
        "",
        f"{'Class':<14} {'N>=1':>6} {'N>k':>6}  {'Fidelity+':>10}  "
        f"{'Fidelity−':>10}  {'meanN':>7}",
        "-" * 62,
    ]
    for cls, v in summary["per_class"].items():
        lines.append(
            f"{cls:<14} {v['n_flows']:>6} {v['n_informative']:>6}  "
            f"{_f(v['fidelity_plus']):>10}  "
            f"{_f(v['fidelity_minus']):>10}  "
            f"{_f(v['mean_n_neighbors']):>7}"
        )
    lines.append("-" * 62)
    lines.append(
        f"{'Overall':<14} {ov['n_flows']:>6} {ov['n_informative']:>6}  "
        f"{_f(ov['fidelity_plus'])}±{_f(ov['fidelity_plus_std'])}  "
        f"{_f(ov['fidelity_minus'])}±{_f(ov['fidelity_minus_std'])}  "
        f"{_f(ov['mean_n_neighbors']):>7}"
    )
    lines.append("")
    lines.append("Secondary (non-headline) subsets:")
    lines.append(
        f"  Fidelity− over ALL N>=1 flows: "
        f"{_f(ov['all_nonempty']['fidelity_minus'])} "
        f"(n={ov['all_nonempty']['n']}) — deflated by the N<=k identity"
    )
    lines.append(
        f"  Fidelity+ over N>k flows only: "
        f"{_f(ov['informative']['fidelity_plus'])} "
        f"(n={ov['informative']['n']})"
    )
    lines.append("")
    lines.append(
        "Fidelity+: P_full − P_masked_top_k   (top-k neighbours rolled back out of node state)"
    )
    lines.append(
        "Fidelity−: P_full − P_kept_top_k     (all but top-k neighbours rolled back)"
    )
    lines.append(
        "Flows with no in-window neighbours (N=0) are EXCLUDED from all means."
    )
    lines.append(
        "Flows with N <= k have Fidelity− = 0 by construction; the \"informative\""
    )
    lines.append(
        "columns restrict to N > k."
    )
    lines.append("")
    lines.append(
        f"Diagnostics: {diag['n_records']} records, {diag['n_rows']} rows, "
        f"{diag['n_zero_neighborhood']} empty neighbourhoods, "
        f"{diag['n_degenerate_minus']} with 0<N<=k, "
        f"{diag['n_sample_fail']} sampling failures, "
        f"{diag['n_neighborhood_mismatch']} neighbourhood mismatches"
    )
    return "\n".join(lines)


def _build_summary_novelty(
    rows: list[dict],
    diagnostics: dict,
    top_k: int,
) -> dict:
    """Aggregate per-class and overall φ_N fidelity for the paper's φ_N row.

    All rows are aggregated — there is no zero-player case for φ_N, so no
    exclusion criterion exists. A secondary `informative` aggregate restricts
    to rows with n_players > top_k, i.e. those where Fidelity− is not 0 by
    construction; the headline numbers are the `all_flows` ones, which keeps
    the φ_N row's n directly comparable with the feature-group row's.

    Args:
        rows:        output rows of compute_fidelity_novelty.
        diagnostics: its diagnostics dict, copied verbatim into the summary.
        top_k:       the --top-k-novelty value actually used.

    Returns:
        JSON-serialisable summary dict.
    """
    import collections

    def _agg(group: list[dict]) -> dict:
        informative = [r for r in group if r["n_players"] > top_k]
        fp = [r["fidelity_plus"] for r in group]
        fm = [r["fidelity_minus"] for r in group]
        i_fp = [r["fidelity_plus"] for r in informative]
        i_fm = [r["fidelity_minus"] for r in informative]
        nov = [r["n_novelty_in_top_k"] for r in group]
        n_nov = sum(1 for v in nov if v > 0)
        return {
            "n_flows":             len(group),
            "fidelity_plus":       round(float(np.mean(fp)), 4) if fp else None,
            "fidelity_plus_std":   round(float(np.std(fp)), 4) if fp else None,
            "fidelity_minus":      round(float(np.mean(fm)), 4) if fm else None,
            "fidelity_minus_std":  round(float(np.std(fm)), 4) if fm else None,
            "n_informative":       len(informative),
            "fidelity_plus_informative": (
                round(float(np.mean(i_fp)), 4) if i_fp else None
            ),
            "fidelity_minus_informative": (
                round(float(np.mean(i_fm)), 4) if i_fm else None
            ),
            "mean_n_players": (
                round(float(np.mean([r["n_players"] for r in group])), 4)
                if group else None
            ),
            "mean_n_non_target": (
                round(float(np.mean([r["n_non_target"] for r in group])), 4)
                if group else None
            ),
            "n_flows_with_novelty_in_top_k": n_nov,
            "frac_novelty_in_top_k": (
                round(n_nov / len(group), 4) if group else None
            ),
            "mean_n_novelty_in_top_k": (
                round(float(np.mean(nov)), 4) if nov else None
            ),
        }

    by_class: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        by_class[r["class_name"]].append(r)

    per_class = {cls: _agg(by_class[cls]) for cls in sorted(by_class.keys())}

    note = (
        "Novelty players rarely reach the top-k: outputs/metrics/novelty_audit.json "
        "reports explanation_json_audit.frac_nonzero_either = 0.3469, i.e. in ~65% of "
        "flows BOTH novelty phi are exactly 0.0 because the novelty state dim is "
        "already 0 for those endpoints, so zeroing it is a no-op. This mechanically "
        "explains the low top-k engagement; it is a diagnostic, not grounds for "
        "re-defining the metric."
    )

    return {
        "top_k": top_k,
        "per_class": per_class,
        "overall": _agg(rows),
        "diagnostics": diagnostics,
        "note": note,
    }


def _format_table_novelty(summary: dict) -> str:
    """ASCII table for the paper's node-novelty (φ_N) fidelity row.

    Args:
        summary: output of _build_summary_novelty.

    Returns:
        The rendered table as a single string.
    """
    k = summary["top_k"]
    ov = summary["overall"]
    diag = summary["diagnostics"]

    def _f(v: "float | None") -> str:
        return f"{v:.4f}" if v is not None else "—"

    lines = [
        f"SHAP-GSD node-novelty metrics  (k={k}, n={ov['n_flows']} flows)",
        "",
        f"{'Class':<14} {'N':>5}  {'Fidelity+':>10}  {'Fidelity−':>10}  "
        f"{'Fid−(inf)':>10}  {'MeanP':>7}  {'Nov@k':>6}",
        "-" * 80,
    ]
    for cls, v in summary["per_class"].items():
        lines.append(
            f"{cls:<14} {v['n_flows']:>5}  "
            f"{_f(v['fidelity_plus']):>10}  "
            f"{_f(v['fidelity_minus']):>10}  "
            f"{_f(v['fidelity_minus_informative']):>10}  "
            f"{_f(v['mean_n_players']):>7}  "
            f"{v['n_flows_with_novelty_in_top_k']:>6}"
        )
    lines.append("-" * 80)
    lines.append(
        f"{'Overall':<14} {ov['n_flows']:>5}  "
        f"{_f(ov['fidelity_plus'])}±{_f(ov['fidelity_plus_std'])}  "
        f"{_f(ov['fidelity_minus'])}±{_f(ov['fidelity_minus_std'])}  "
        f"{_f(ov['fidelity_minus_informative']):>10}  "
        f"{_f(ov['mean_n_players']):>7}  "
        f"{ov['n_flows_with_novelty_in_top_k']:>6}"
    )
    lines.append("")
    lines.append(
        "Headline Fidelity+/Fidelity− are the all-flows means; Fid−(inf) is the "
        "secondary"
    )
    lines.append(
        "  'informative' mean over flows with P > k only."
    )
    lines.append(
        "Fidelity+: P_full − P_masked_top_k   (top-k players masked: novelty flags "
        "zeroed,"
    )
    lines.append(
        "                                      non-target nodes replaced by class "
        "background)"
    )
    lines.append(
        "Fidelity−: P_full − P_kept_top_k     (ALL non-top-k players masked, both "
        "mechanisms)"
    )
    lines.append(
        "Players per flow = 2 novelty flags + M non-target nodes (P >= 2 always;"
    )
    lines.append(
        "  no flow is excluded)."
    )
    lines.append(
        f"Flows with P <= k have Fidelity− = 0 by construction; the 'informative' "
        f"columns"
    )
    lines.append(
        f"  restrict to P > k ({diag['n_degenerate_minus']}/{diag['n_rows']} rows "
        f"differ at k={k})."
    )
    lines.append(
        f"Novelty players reach the top-k in only "
        f"{ov['n_flows_with_novelty_in_top_k']}/{ov['n_flows']} flows; see"
    )
    lines.append(
        "  outputs/metrics/novelty_audit.json (65% of flows have both novelty phi "
        "== 0)."
    )
    if (
        diag["n_sample_fail"]
        or diag["n_context_assert_fail"]
        or diag["n_player_set_mismatch"]
    ):
        lines.append(
            f"WARNING: {diag['n_records']} records but only {diag['n_rows']} rows — "
            f"{diag['n_sample_fail']} sampling failures, "
            f"{diag['n_context_assert_fail']} invariant assert failures, "
            f"{diag['n_player_set_mismatch']} player-set mismatches."
        )
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP-GSD Phase 8 — Quantitative metrics")
    parser.add_argument("--config", required=True,
                        help="Path to experiment YAML config")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top feature groups for fidelity masks (default 5)")
    parser.add_argument("--stability-seeds", type=int, default=3,
                        help="Coalition seeds for stability (default 3)")
    parser.add_argument("--stability-nsamples", type=int, default=512,
                        help="KernelSHAP samples per stability run (default 512)")
    parser.add_argument("--stability-per-class", type=int, default=5,
                        help="Flows per class for stability subset (default 5)")
    parser.add_argument("--skip-stability", action="store_true",
                        help="Skip stability computation (fidelity only)")
    parser.add_argument("--top-k-temporal", type=int, default=3,
                        help="Top-k neighbours for temporal fidelity masks (default 3)")
    parser.add_argument("--skip-temporal-fidelity", action="store_true",
                        help="Skip φ_T fidelity computation")
    parser.add_argument("--top-k-novelty", type=int, default=5,
                        help="Top-k node-novelty players for fidelity masks (default 5)")
    parser.add_argument("--skip-novelty-fidelity", action="store_true",
                        help="Skip φ_N fidelity computation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info("=== Phase 8: SHAP-GSD Quantitative Metrics ===")

    artifacts_dir   = Path(cfg["output"]["artifacts_dir"])
    fs_test_dir     = Path(cfg["output"]["feature_store_dir"]) / "test"
    graphs_dir      = Path(cfg["graph"]["dir"])
    nsm_dir         = Path(cfg["graph"]["node_state_dir"])
    explanations_dir = Path(cfg["output"]["outputs_dir"]) / "explanations"
    out_dir         = Path(cfg["output"]["outputs_dir"]) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        cfg.get("compute", {}).get("device", "cuda")
        if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Device: {device}")

    # --- Load artifacts ---
    logger.info("Loading test graph …")
    g_test, _ = dgl.load_graphs(str(graphs_dir / "test.bin"))
    g_test = g_test[0]

    logger.info("Loading feature store …")
    fs_test = FeatureStore(fs_test_dir)

    logger.info("Loading NodeStateManager …")
    nsm = NodeStateManager.load(nsm_dir)

    logger.info("Loading background distributions …")
    background = BackgroundDistributions.load(artifacts_dir)

    logger.info("Loading model …")
    model = _load_model(cfg, device)

    fg_path = Path(cfg["output"]["feature_groups_path"])
    with open(fg_path) as f:
        feature_groups = json.load(f)

    label_map_path = artifacts_dir / "label_map.json"
    with open(label_map_path) as f:
        label_map = json.load(f)
    int_to_name = {v: k for k, v in label_map.items()}

    sampler = TemporalNeighborSampler(fanouts=cfg["model"]["fanouts"])

    # --- Load explanations ---
    records = _load_explanations(explanations_dir, int_to_name)

    # --- Fidelity ---
    logger.info(f"Computing Fidelity (top-k={args.top_k}) on {len(records)} flows …")
    fidelity_rows = compute_fidelity(
        records, model, g_test, nsm, fs_test, background,
        feature_groups, sampler, device, top_k=args.top_k,
    )

    fid_path = out_dir / "fidelity.csv"
    if fidelity_rows:
        with open(fid_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(fidelity_rows[0].keys()))
            writer.writeheader()
            writer.writerows(fidelity_rows)
    logger.info(f"Fidelity CSV → {fid_path}")

    # --- Temporal-neighborhood (φ_T) fidelity ---
    if not args.skip_temporal_fidelity:
        logger.info(
            f"Computing temporal fidelity (top-k={args.top_k_temporal}) "
            f"on {len(records)} flows …"
        )
        temporal_rows, temporal_diag = compute_fidelity_temporal(
            records, model, g_test, nsm, fs_test, background,
            sampler, device, top_k=args.top_k_temporal,
        )

        fid_t_path = out_dir / "fidelity_temporal.csv"
        if temporal_rows:
            with open(fid_t_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(temporal_rows[0].keys()))
                writer.writeheader()
                writer.writerows(temporal_rows)
        logger.info(f"Temporal fidelity CSV → {fid_t_path}")

        summary_t = _build_summary_temporal(
            temporal_rows, temporal_diag, top_k=args.top_k_temporal
        )
        summary_t_path = out_dir / "summary_temporal.json"
        with open(summary_t_path, "w") as f:
            json.dump(summary_t, f, indent=2)
        logger.info(f"Temporal summary JSON → {summary_t_path}")

        table_t_str = _format_table_temporal(summary_t)
        table_t_path = out_dir / "table2_temporal.txt"
        with open(table_t_path, "w") as f:
            f.write(table_t_str)
        logger.info(f"Temporal table → {table_t_path}")
        logger.info("\n" + table_t_str)

    # --- Node-novelty fidelity (φ_N) ---
    if not args.skip_novelty_fidelity:
        logger.info(
            f"Computing node-novelty Fidelity (top-k={args.top_k_novelty}) on "
            f"{len(records)} flows …"
        )
        novelty_rows, novelty_diag = compute_fidelity_novelty(
            records, model, g_test, nsm, fs_test, background,
            sampler, device, top_k=args.top_k_novelty,
        )

        fid_n_path = out_dir / "fidelity_novelty.csv"
        if novelty_rows:
            with open(fid_n_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(novelty_rows[0].keys()))
                writer.writeheader()
                writer.writerows(novelty_rows)
        logger.info(f"Novelty fidelity CSV → {fid_n_path}")

        novelty_summary = _build_summary_novelty(
            novelty_rows, novelty_diag, top_k=args.top_k_novelty
        )
        novelty_summary_path = out_dir / "summary_novelty.json"
        with open(novelty_summary_path, "w") as f:
            json.dump(novelty_summary, f, indent=2)
        logger.info(f"Novelty summary JSON → {novelty_summary_path}")

        novelty_table = _format_table_novelty(novelty_summary)
        novelty_table_path = out_dir / "table2_novelty.txt"
        with open(novelty_table_path, "w") as f:
            f.write(novelty_table)
        logger.info(f"Novelty table → {novelty_table_path}")
    # --- Stability ---
    stability_rows: list[dict] = []
    if not args.skip_stability:
        logger.info(
            f"Computing Stability ({args.stability_seeds} seeds, "
            f"nsamples={args.stability_nsamples}, "
            f"{args.stability_per_class}/class) …"
        )
        stability_rows = compute_stability(
            records, model, g_test, nsm, fs_test, background,
            feature_groups, sampler, device,
            n_seeds=args.stability_seeds,
            nsamples=args.stability_nsamples,
            n_per_class=args.stability_per_class,
        )
        stab_path = out_dir / "stability.csv"
        if stability_rows:
            with open(stab_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(stability_rows[0].keys()))
                writer.writeheader()
                writer.writerows(stability_rows)
        logger.info(f"Stability CSV → {stab_path}")

    # --- Summary + table ---
    summary = _build_summary(fidelity_rows, stability_rows, top_k=args.top_k)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary JSON → {summary_path}")

    runtime_by_layer_path = out_dir / "runtime_by_layer.json"
    runtime_by_layer = None
    if runtime_by_layer_path.exists():
        with open(runtime_by_layer_path) as f:
            runtime_by_layer = json.load(f)
    else:
        logger.warning(
            f"{runtime_by_layer_path} not found — Table 2 will omit the "
            f"per-layer explanation-cost secondary (06_explain.py writes it)."
        )

    table_str = _format_table(summary, runtime_by_layer)
    table_path = out_dir / "table2.txt"
    with open(table_path, "w") as f:
        f.write(table_str)
    logger.info(f"Table 2 → {table_path}")

    print("\n" + table_str)
    logger.info("Phase 8 complete.")


if __name__ == "__main__":
    main()
