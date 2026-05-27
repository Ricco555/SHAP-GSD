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
  outputs/metrics/fidelity.csv       — per-flow Fidelity+/−
  outputs/metrics/stability.csv      — per-flow stability (50-flow subset)
  outputs/metrics/summary.json       — per-class + overall table values
  outputs/metrics/table2.txt         — ASCII table ready for paper

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
            phi_dict = feat_shap.explain(
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


def _format_table(summary: dict) -> str:
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

    table_str = _format_table(summary)
    table_path = out_dir / "table2.txt"
    with open(table_path, "w") as f:
        f.write(table_str)
    logger.info(f"Table 2 → {table_path}")

    print("\n" + table_str)
    logger.info("Phase 8 complete.")


if __name__ == "__main__":
    main()
