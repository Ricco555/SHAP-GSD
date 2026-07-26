"""
Phase 9 — W (temporal window) sensitivity ablation.

Step 1: Gap distribution analysis across all 1,764 explained flows.
  For each W ∈ {60, 300, 1800, 3600} seconds, computes:
    - fraction of flows with ≥1 in-window neighbour
    - mean/median in-window neighbour count per flow
    - fraction of all sampled neighbour edges that fall inside W

Step 2 (only if a W has >1% in-window-flow rate):
  Re-run FeatureGroupSHAP + TemporalNeighborhoodSHAP on a 50-flow subset
  (5/class) to produce non-zero temporal SHAP values and measure
  Fidelity+ change.

Outputs:
  outputs/w_ablation/gap_stats.json     — per-W gap statistics (Step 1)
  outputs/w_ablation/gap_stats.txt      — readable table for paper
  outputs/w_ablation/W<s>_temporal.csv  — temporal SHAP re-runs (Step 2, if triggered)
  outputs/w_ablation/summary.txt        — final narrative for research_plan.md

Usage:
  python scripts/09_w_ablation.py --config configs/experiment_unsw.yaml
"""

import argparse
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
from src.explainer.temporal_shap import TemporalNeighborhoodSHAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("shap").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

W_VALUES = [60, 300, 1800, 3600]   # seconds
STEP2_THRESHOLD = 0.01              # trigger Step 2 if ≥1% of flows have in-window neighbours


# ── model loader ──────────────────────────────────────────────────────────────

def _load_model(cfg: dict, device: torch.device) -> EdgeAwareGraphSAGE:
    m = cfg["model"]
    artifacts_dir = Path(cfg["output"]["artifacts_dir"])
    best_params_path = artifacts_dir / "best_params.json"
    with open(best_params_path) as f:
        bp = json.load(f)
    with open(artifacts_dir / "label_map.json") as f:
        lm = json.load(f)
    with open(Path(cfg["output"]["feature_groups_path"])) as f:
        fg = json.load(f)
    model = EdgeAwareGraphSAGE(
        node_in_dim=m["node_state_dim"], edge_in_dim=fg["d_e"],
        hidden_size=bp["hidden_size"], num_classes=len(lm),
        num_layers=bp["num_layers"], dropout=bp["dropout"], aggregator=bp["aggregator"],
    ).to(device)
    model.load_state_dict(torch.load(artifacts_dir / "best_model.pt", map_location=device))
    model.eval()
    return model


# ── Step 1: gap distribution ───────────────────────────────────────────────────

def _collect_gaps(
    records: list[dict],
    g_test: dgl.DGLGraph,
    sampler: TemporalNeighborSampler,
) -> list[dict]:
    """For each explained flow, sample blocks and collect all neighbour gaps (s).

    Returns list of dicts with keys: class_name, edge_id, target_ts_ms, gaps_s (list[float]).
    """
    global_eids_t = g_test.edata[dgl.EID]
    flow_gaps: list[dict] = []
    n = len(records)
    t0 = time.time()

    for i, rec in enumerate(records):
        global_eid = rec["edge_id"]
        local_eid_matches = (global_eids_t == global_eid).nonzero(as_tuple=True)[0]
        if len(local_eid_matches) == 0:
            continue
        local_eid = int(local_eid_matches[0].item())
        seed_t = torch.tensor([local_eid], dtype=torch.long)

        try:
            _, _, blocks = sampler.sample_blocks(g_test, seed_t)
        except Exception:
            continue

        target_ts = float(g_test.edata["timestamp"][local_eid].item())
        seen: set[int] = set()
        gaps: list[float] = []
        for block in blocks:
            if dgl.EID not in block.edata:
                continue
            for leid in block.edata[dgl.EID].tolist():
                if leid == local_eid or leid in seen:
                    continue
                seen.add(leid)
                ts = float(g_test.edata["timestamp"][leid].item())
                gaps.append(max((target_ts - ts) / 1000.0, 0.0))

        flow_gaps.append({
            "class_name":   rec["_class_name"],
            "edge_id":      global_eid,
            "target_ts_ms": target_ts,
            "gaps_s":       gaps,
        })

        if (i + 1) % 200 == 0:
            logger.info(f"  Gap collection: {i+1}/{n}  ({time.time()-t0:.0f}s)")

    logger.info(f"Gap collection complete: {len(flow_gaps)}/{n} flows, {time.time()-t0:.0f}s")
    return flow_gaps


def _gap_stats_for_W(flow_gaps: list[dict], W_s: float) -> dict:
    """Aggregate per-W gap statistics across all flows."""
    n_flows = len(flow_gaps)
    flows_with_inwindow = 0
    total_edges = 0
    total_inwindow = 0
    inwindow_counts: list[int] = []
    all_gaps: list[float] = []

    for fg in flow_gaps:
        gaps = np.array(fg["gaps_s"])
        all_gaps.extend(gaps.tolist())
        inw = int((gaps <= W_s).sum())
        inwindow_counts.append(inw)
        total_edges += len(gaps)
        total_inwindow += inw
        if inw > 0:
            flows_with_inwindow += 1

    all_gaps_arr = np.array(all_gaps) if all_gaps else np.array([0.0])
    return {
        "W_s":                       W_s,
        "n_flows":                   n_flows,
        "flows_with_inwindow":       flows_with_inwindow,
        "pct_flows_with_inwindow":   round(100 * flows_with_inwindow / max(n_flows, 1), 2),
        "total_neighbour_edges":     total_edges,
        "total_inwindow_edges":      total_inwindow,
        "pct_edges_inwindow":        round(100 * total_inwindow / max(total_edges, 1), 3),
        "mean_inwindow_per_flow":    round(float(np.mean(inwindow_counts)), 3),
        "median_gap_s":              round(float(np.median(all_gaps_arr)), 1),
        "p5_gap_s":                  round(float(np.percentile(all_gaps_arr, 5)), 1),
        "p95_gap_s":                 round(float(np.percentile(all_gaps_arr, 95)), 1),
    }


def _format_gap_table(all_stats: list[dict]) -> str:
    lines = [
        "W sensitivity — neighbour gap statistics (NF-UNSW-NB15-v3, n=1,764 flows)",
        "",
        f"{'W (s)':>8}  {'Flows w/≥1 nbr':>16}  {'Edges in W':>12}  {'Mean in-W/flow':>16}",
        "-" * 60,
    ]
    for s in all_stats:
        lines.append(
            f"{s['W_s']:>8.0f}  "
            f"{s['flows_with_inwindow']:>6}/{s['n_flows']}  "
            f"({s['pct_flows_with_inwindow']:5.1f}%)   "
            f"{s['pct_edges_inwindow']:>6.2f}%         "
            f"{s['mean_inwindow_per_flow']:>6.3f}"
        )
    lines += [
        "-" * 60,
        f"Median neighbour gap: {all_stats[0]['median_gap_s']:.0f} s  "
        f"(p5={all_stats[0]['p5_gap_s']:.0f}s, p95={all_stats[0]['p95_gap_s']:.0f}s)",
        "",
        "Interpretation: W must exceed the median gap (~{:.0f} s) before".format(
            all_stats[0]["median_gap_s"]
        ),
        "temporal neighbours routinely fall within the window.",
    ]
    return "\n".join(lines)


# ── Step 2: temporal SHAP re-run for informative W values ─────────────────────

def _step2_temporal_shap(
    flow_gaps: list[dict],
    records: list[dict],
    W_s: float,
    model: EdgeAwareGraphSAGE,
    g_test: dgl.DGLGraph,
    nsm: NodeStateManager,
    fs: FeatureStore,
    background: BackgroundDistributions,
    feature_groups: dict,
    sampler: TemporalNeighborSampler,
    device: torch.device,
    n_per_class: int = 5,
    nsamples: int = 512,
) -> list[dict]:
    """Re-run TemporalNeighborhoodSHAP on 5/class subset with overridden W."""
    # Override W in NSM (dynamic — does not rebuild snapshots)
    original_W = nsm._W_ms
    nsm._W_ms = W_s * 1000.0
    logger.info(f"  NSM _W_ms overridden to {W_s * 1000.0:.0f} ms")

    # Build record lookup by edge_id
    rec_by_eid = {r["edge_id"]: r for r in records}

    # Select subset: first n_per_class per class from flows that have ≥1 in-window edge
    per_class: dict[str, list] = {}
    W_ms = W_s * 1000.0
    for fg in flow_gaps:
        if not any(g <= W_s for g in fg["gaps_s"]):
            continue
        cls = fg["class_name"]
        if cls not in per_class:
            per_class[cls] = []
        if len(per_class[cls]) < n_per_class:
            per_class[cls].append(fg["edge_id"])

    subset_eids = [eid for eids in per_class.values() for eid in eids]
    logger.info(f"  Step 2 subset: {len(subset_eids)} flows with in-window neighbours at W={W_s}s")

    if not subset_eids:
        nsm._W_ms = original_W
        return []

    global_eids_t = g_test.edata[dgl.EID]
    temp_shap = TemporalNeighborhoodSHAP(background, nsm, g_test, device)
    rows: list[dict] = []

    for i, global_eid in enumerate(subset_eids):
        rec = rec_by_eid.get(global_eid)
        if rec is None:
            continue
        true_label = rec["true_label"]
        class_name = rec["_class_name"]

        local_eid_t = (global_eids_t == global_eid).nonzero(as_tuple=True)[0]
        if len(local_eid_t) == 0:
            continue
        local_eid = int(local_eid_t[0].item())
        seed_t = torch.tensor([local_eid], dtype=torch.long)

        try:
            inp, _, blocks = sampler.sample_blocks(g_test, seed_t)
            blocks_d = [b.to(device) for b in blocks]
            inp_d = inp.to(device)
        except Exception:
            logger.warning(f"  Skip EID {global_eid}: block sampling failed")
            continue

        target_ts = float(g_test.edata["timestamp"][local_eid].item())
        input_node_ids = inp.numpy()
        base_nf = np.stack([nsm.get_state_at_time(int(n), target_ts) for n in input_node_ids])
        x_e = fs[global_eid].copy()
        x_e_t = torch.tensor(x_e, dtype=torch.float32, device=device).unsqueeze(0)
        nf_t = torch.tensor(base_nf, dtype=torch.float32, device=device)

        seed_nodes = blocks_d[-1].dstdata[dgl.NID]
        sp, dp = build_src_dst_pos(g_test, seed_t, seed_nodes.cpu())
        sp, dp = sp.to(device), dp.to(device)

        # Full prediction
        with torch.no_grad():
            logits = model(blocks_d, nf_t, x_e_t, sp, dp)
            p_full = float(torch.softmax(logits, dim=1).cpu().numpy()[0, true_label])

        # Temporal SHAP with overridden W
        temp_results, _f_baseline_temp, _f_logit_temp = temp_shap.explain(
            target_local_eid=local_eid,
            true_class=true_label,
            model=model,
            blocks=blocks_d,
            input_nodes=inp,
            target_ts_ms=target_ts,
            src_pos=sp, dst_pos=dp,
            x_e=x_e, base_node_feats=base_nf,
            nsamples=nsamples,
        )
        n_inwindow = len(temp_results)
        sum_phi = float(sum(abs(t[2]) for t in temp_results))
        top_phi = float(max((abs(t[2]) for t in temp_results), default=0.0))

        rows.append({
            "W_s":           W_s,
            "class_name":    class_name,
            "edge_id":       global_eid,
            "true_label":    true_label,
            "p_full":        round(p_full, 6),
            "n_inwindow":    n_inwindow,
            "sum_abs_phi_T": round(sum_phi, 6),
            "top_phi_T":     round(top_phi, 6),
        })
        logger.info(
            f"  [{i+1}/{len(subset_eids)}] {class_name} EID={global_eid} "
            f"n_inwindow={n_inwindow} sum|φ_T|={sum_phi:.4f}"
        )

    nsm._W_ms = original_W
    return rows


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP-GSD Phase 9 — W sensitivity ablation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--step2-nsamples", type=int, default=512,
                        help="KernelSHAP samples for Step 2 temporal SHAP re-runs (default 512)")
    parser.add_argument("--step2-per-class", type=int, default=5,
                        help="Flows per class for Step 2 subset (default 5)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info("=== Phase 9: W Sensitivity Ablation ===")

    artifacts_dir = Path(cfg["output"]["artifacts_dir"])
    fs_test_dir   = Path(cfg["output"]["feature_store_dir"]) / "test"
    graphs_dir    = Path(cfg["graph"]["dir"])
    nsm_dir       = Path(cfg["graph"]["node_state_dir"])
    out_dir       = Path(cfg["output"]["outputs_dir"]) / "w_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        cfg.get("compute", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Device: {device}")

    logger.info("Loading graph, FST, NSM, background, model …")
    g_test, _ = dgl.load_graphs(str(graphs_dir / "test.bin"))
    g_test = g_test[0]
    fs_test = FeatureStore(fs_test_dir)
    nsm = NodeStateManager.load(nsm_dir)
    background = BackgroundDistributions.load(artifacts_dir)
    model = _load_model(cfg, device)
    with open(Path(cfg["output"]["feature_groups_path"])) as f:
        feature_groups = json.load(f)
    sampler = TemporalNeighborSampler(fanouts=cfg["model"]["fanouts"])

    # Load all explanation records
    explanations_dir = Path(cfg["output"]["outputs_dir"]) / "explanations"
    records: list[dict] = []
    for cls_dir in sorted(explanations_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for jf in sorted(cls_dir.glob("*.json")):
            if jf.stem.endswith("_fixed"):
                continue
            with open(jf) as f:
                d = json.load(f)
            d["_class_name"] = cls_dir.name
            records.append(d)
    logger.info(f"Loaded {len(records)} explanation records")

    # ── Step 1 ────────────────────────────────────────────────────────────────
    logger.info("Step 1: collecting neighbour gaps for all flows …")
    flow_gaps = _collect_gaps(records, g_test, sampler)

    all_stats: list[dict] = []
    for W_s in W_VALUES:
        s = _gap_stats_for_W(flow_gaps, float(W_s))
        all_stats.append(s)
        logger.info(
            f"  W={W_s:5d}s: {s['pct_flows_with_inwindow']:5.1f}% flows have in-window nbr  "
            f"({s['pct_edges_inwindow']:.2f}% of edges)"
        )

    gap_json_path = out_dir / "gap_stats.json"
    with open(gap_json_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    table_str = _format_gap_table(all_stats)
    gap_txt_path = out_dir / "gap_stats.txt"
    with open(gap_txt_path, "w") as f:
        f.write(table_str)
    print("\n" + table_str + "\n")

    # ── Step 2 ────────────────────────────────────────────────────────────────
    import csv

    step2_results: dict[int, list[dict]] = {}
    triggered_W: list[int] = []

    for s in all_stats:
        W_s = int(s["W_s"])
        rate = s["pct_flows_with_inwindow"] / 100.0
        if rate >= STEP2_THRESHOLD:
            logger.info(
                f"Step 2 triggered for W={W_s}s "
                f"({s['pct_flows_with_inwindow']:.1f}% ≥ threshold {STEP2_THRESHOLD*100:.0f}%)"
            )
            triggered_W.append(W_s)
            rows = _step2_temporal_shap(
                flow_gaps, records, float(W_s), model, g_test, nsm,
                fs_test, background, feature_groups, sampler, device,
                n_per_class=args.step2_per_class,
                nsamples=args.step2_nsamples,
            )
            step2_results[W_s] = rows

            csv_path = out_dir / f"W{W_s}_temporal.csv"
            if rows:
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
            logger.info(f"  Step 2 CSV → {csv_path}")
        else:
            logger.info(
                f"Step 2 NOT triggered for W={W_s}s "
                f"({s['pct_flows_with_inwindow']:.1f}% < threshold)"
            )

    # ── Summary narrative ─────────────────────────────────────────────────────
    summary_lines = [
        "W sensitivity ablation — NF-UNSW-NB15-v3",
        "=" * 60,
        "",
        table_str,
        "",
        "Step 2 (temporal SHAP re-runs):",
    ]
    if triggered_W:
        for W_s in triggered_W:
            rows = step2_results.get(W_s, [])
            if rows:
                phi_vals = [r["sum_abs_phi_T"] for r in rows]
                summary_lines.append(
                    f"  W={W_s}s: {len(rows)} flows re-explained, "
                    f"mean sum|φ_T|={np.mean(phi_vals):.4f}, "
                    f"max={np.max(phi_vals):.4f}"
                )
    else:
        summary_lines.append("  No W value exceeded the 1% in-window-flow threshold.")
        summary_lines.append(
            "  Temporal SHAP remains null across all tested W values on UNSW-NB15."
        )
        summary_lines.append(
            "  This is a dataset characteristic: the 2-day capture window has a"
        )
        summary_lines.append(
            "  median inter-flow gap of ~{:.0f}s, far exceeding W=3600s.".format(
                all_stats[0]["median_gap_s"]
            )
        )

    summary_str = "\n".join(summary_lines)
    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_str)
    logger.info(f"Summary → {summary_path}")
    print(summary_str)
    logger.info("Phase 9 complete.")


if __name__ == "__main__":
    main()
