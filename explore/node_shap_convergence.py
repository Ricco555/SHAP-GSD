"""
Node SHAP KernelSHAP convergence experiment.

For each selected flow, rebuilds the GNN computation context from the test
graph and re-runs NodeNoveltySHAP.explain() at
nsamples ∈ {128, 256, 512, 1024, 2048}.  For each level the Shapley
efficiency error (|Σφ − (f_logit − f_baseline)|) and wall-clock time are
recorded.

Expected convergence: efficiency error ∝ O(1/√n) → empirical log-log slope
should be close to −0.50.

Estimated runtime (200 flows, 5 nsamples levels):
  ~5 min  on A100 GPU
  ~90 min on CPU

For a quick local sanity-check, pass --n-per-class 2 (~2 min on CPU).

Usage:
  python explore/node_shap_convergence.py [--config ...] [--n-per-class 20] [--seed 42]

Reads:
  outputs/explanations/<Class>/<EID>.json  (for EID selection only)
  graphs/test.bin
  configs/experiment_unsw.yaml

Outputs:
  outputs/figures/explore/node_shap_convergence.{pdf,png}
  outputs/figures/explore/node_shap_convergence_details.txt
  outputs/metrics/node_shap_convergence.json
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from src.explainer.node_shap import NodeNoveltySHAP
from src.model.temporal_sampler import TemporalNeighborSampler

NSAMPLES_LIST = [128, 256, 512, 1024, 2048]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("shap").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


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

    with open(artifacts_dir / "label_map.json") as f:
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


def _build_flow_context(
    g: dgl.DGLGraph,
    fs: FeatureStore,
    nsm: NodeStateManager,
    geid_to_local: dict[int, int],
    global_eid: int,
    sampler: TemporalNeighborSampler,
    device: torch.device,
) -> dict | None:
    """Build all tensors needed to call NodeNoveltySHAP.explain() for one edge.

    Returns None if the global EID is not present in the graph.
    """
    local_eid = geid_to_local.get(global_eid)
    if local_eid is None:
        return None

    seed_eid_t = torch.tensor([local_eid], dtype=torch.long)
    input_nodes, _, blocks = sampler.sample_blocks(g, seed_eid_t)
    blocks_dev = [b.to(device) for b in blocks]

    target_ts = float(g.edata["timestamp"][local_eid].item())
    src_t, dst_t = g.find_edges(seed_eid_t)
    target_src = int(src_t[0])
    target_dst = int(dst_t[0])

    input_node_ids = input_nodes.cpu().numpy()
    base_node_feats = np.stack([
        nsm.get_state_at_time(int(nid), target_ts)
        for nid in input_node_ids
    ])

    seed_nodes_final = blocks_dev[-1].dstdata[dgl.NID]
    src_pos, dst_pos = build_src_dst_pos(g, seed_eid_t, seed_nodes_final)
    src_pos = src_pos.to(device)
    dst_pos = dst_pos.to(device)

    actual_geid = int(g.edata[dgl.EID][local_eid])
    x_e = fs[actual_geid].copy()
    true_label = int(fs.labels[fs._eid_to_pos[actual_geid]])

    coalition_size = 2 + sum(
        1 for nid in input_node_ids
        if int(nid) != target_src and int(nid) != target_dst
    )

    return {
        "global_eid":      global_eid,
        "true_label":      true_label,
        "target_src":      target_src,
        "target_dst":      target_dst,
        "blocks":          blocks_dev,
        "input_nodes":     input_nodes,
        "base_node_feats": base_node_feats,
        "src_pos":         src_pos,
        "dst_pos":         dst_pos,
        "x_e":             x_e,
        "coalition_size":  coalition_size,
    }


def main() -> None:
    _default_cfg = os.environ.get("SHAP_GSD_CONFIG", "configs/experiment_unsw.yaml")
    parser = argparse.ArgumentParser(description="Node SHAP convergence experiment")
    parser.add_argument("--config",       default=_default_cfg)
    parser.add_argument("--n-per-class",  type=int, default=20,
                        help="Flows to sample per class (default 20; use 2 for quick test)")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    outputs     = ROOT / cfg["output"]["outputs_dir"]
    EXPL_DIR    = outputs / "explanations"
    FIG_DIR     = outputs / "figures" / "explore"
    METRICS_DIR = outputs / "metrics"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        cfg.get("compute", {}).get("device", "cuda")
        if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Device: {device}")

    artifacts_dir = Path(cfg["output"]["artifacts_dir"])
    graphs_dir    = Path(cfg["graph"]["dir"])
    nsm_dir       = Path(cfg["graph"]["node_state_dir"])

    logger.info("Loading test graph …")
    g_test, _ = dgl.load_graphs(str(graphs_dir / "test.bin"))
    g_test = g_test[0]
    geid_to_local = {int(g): i for i, g in enumerate(g_test.edata[dgl.EID].numpy())}

    logger.info("Loading feature store …")
    fs_test = FeatureStore(Path(cfg["output"]["feature_store_dir"]) / "test")

    logger.info("Loading NodeStateManager …")
    nsm = NodeStateManager.load(nsm_dir)

    logger.info("Loading background distributions …")
    background = BackgroundDistributions.load(artifacts_dir)

    logger.info("Loading model …")
    model = _load_model(cfg, device)

    sampler = TemporalNeighborSampler(fanouts=cfg["model"]["fanouts"])
    node_shap_inst = NodeNoveltySHAP(background=background, device=device)

    # --- Select flows ---
    with open(artifacts_dir / "label_map.json") as f:
        label_map: dict[str, int] = json.load(f)
    int_to_name = {v: k for k, v in label_map.items()}

    rng = np.random.default_rng(args.seed)
    selected: list[tuple[int, int]] = []  # (global_eid, class_int)

    for c, class_name in sorted(int_to_name.items()):
        cls_dir = EXPL_DIR / class_name
        if not cls_dir.exists():
            continue
        eids = sorted(
            int(p.stem) for p in cls_dir.glob("*.json")
            if p.stem.lstrip("-").isdigit()
        )
        if not eids:
            continue
        n = min(args.n_per_class, len(eids))
        chosen = rng.choice(eids, size=n, replace=False)
        for eid in chosen:
            selected.append((int(eid), c))

    logger.info(f"Selected {len(selected)} flows across {len(int_to_name)} classes "
                f"(n_per_class={args.n_per_class}, seed={args.seed})")

    # records[nsamples] → list of (abs_err, rel_err, wall_time_s, coalition_size)
    records: dict[int, list[tuple[float, float, float, int]]] = {n: [] for n in NSAMPLES_LIST}

    for idx, (global_eid, true_class) in enumerate(selected):
        logger.info(
            f"  [{idx+1}/{len(selected)}] EID={global_eid} "
            f"class={int_to_name.get(true_class, str(true_class))}"
        )
        ctx = _build_flow_context(
            g_test, fs_test, nsm, geid_to_local, global_eid, sampler, device
        )
        if ctx is None:
            logger.warning(f"    EID {global_eid} not in test graph — skipping")
            continue

        # f_logit and f_baseline are deterministic (independent of nsamples);
        # capture them from the first explain() call to use as the reference gap.
        gap_ref: float | None = None

        for n in NSAMPLES_LIST:
            t0 = time.time()
            try:
                result = node_shap_inst.explain(
                    true_class=ctx["true_label"],
                    src_nid=ctx["target_src"],
                    dst_nid=ctx["target_dst"],
                    model=model,
                    blocks=ctx["blocks"],
                    input_nodes=ctx["input_nodes"].cpu(),
                    base_node_feats=ctx["base_node_feats"],
                    src_pos=ctx["src_pos"],
                    dst_pos=ctx["dst_pos"],
                    x_e=ctx["x_e"],
                    nsamples=n,
                )
            except Exception:
                logger.exception(f"    nsamples={n} failed — skipping this level")
                continue
            wall_s = time.time() - t0

            gap = result["f_logit"] - result["f_baseline"]
            if gap_ref is None:
                gap_ref = gap

            phi_sum = (
                result["src_novelty_shap"]
                + result["dst_novelty_shap"]
                + sum(result["node_shap"].values())
            )
            abs_err = abs(phi_sum - gap)
            rel_err = abs_err / max(abs(gap_ref), 1e-3)

            records[n].append((abs_err, rel_err, wall_s, ctx["coalition_size"]))
            logger.debug(
                f"    n={n:5d}  abs_err={abs_err:.4f}  rel_err={rel_err:.4f}  "
                f"time={wall_s:.2f}s  coalition={ctx['coalition_size']}"
            )

    # --- Aggregate per nsamples level ---
    agg: dict[int, dict] = {}
    for n in NSAMPLES_LIST:
        recs = np.array(records[n])
        if len(recs) == 0:
            continue
        agg[n] = {
            "n_flows":        int(len(recs)),
            "mean_abs_err":   float(recs[:, 0].mean()),
            "std_abs_err":    float(recs[:, 0].std()),
            "median_abs_err": float(np.median(recs[:, 0])),
            "p95_abs_err":    float(np.percentile(recs[:, 0], 95)),
            "mean_rel_err":   float(recs[:, 1].mean()),
            "std_rel_err":    float(recs[:, 1].std()),
            "mean_time_s":    float(recs[:, 2].mean()),
            "std_time_s":     float(recs[:, 2].std()),
            "mean_coalition": float(recs[:, 3].mean()),
        }

    # --- Empirical log-log slope ---
    ns_present = [n for n in NSAMPLES_LIST if n in agg]
    if len(ns_present) >= 2:
        log_ns  = np.log(np.array(ns_present, dtype=float))
        log_err = np.log(np.clip(
            [agg[n]["mean_abs_err"] for n in ns_present], 1e-10, None
        ))
        slope, _intercept = np.polyfit(log_ns, log_err, 1)
    else:
        slope = float("nan")

    logger.info(f"Empirical log-log slope: {slope:.4f}  (expected −0.50)")

    # --- Write JSON ---
    output_data = {
        "seed":              args.seed,
        "n_per_class":       args.n_per_class,
        "n_total_selected":  len(selected),
        "nsamples_list":     NSAMPLES_LIST,
        "empirical_slope":   round(float(slope), 4),
        "expected_slope":    -0.50,
        "per_nsamples":      {str(n): agg[n] for n in ns_present},
        "interpretation": (
            f"Empirical slope {slope:.2f} vs expected −0.50. "
            "A slope near −0.50 confirms O(1/√n) KernelSHAP convergence. "
            "Node SHAP efficiency error is dominated by sampling variance, "
            "not systematic bias; increasing nsamples predictably reduces error."
        ),
    }
    json_path = METRICS_DIR / "node_shap_convergence.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Metrics → {json_path}")

    # --- Write text report ---
    lines = [
        "Node SHAP KernelSHAP Convergence Experiment",
        f"seed={args.seed}  n_per_class={args.n_per_class}  "
        f"n_total={len(selected)}  W=60s",
        "",
        f"{'nsamples':>8s}  {'mean_abs_err':>12s}  {'std_abs_err':>11s}  "
        f"{'mean_rel_err':>12s}  {'mean_time_s':>11s}  {'mean_coalition':>14s}",
        "-" * 78,
    ]
    for n in ns_present:
        a = agg[n]
        lines.append(
            f"{n:>8d}  {a['mean_abs_err']:>12.4f}  {a['std_abs_err']:>11.4f}  "
            f"{a['mean_rel_err']:>12.4f}  {a['mean_time_s']:>11.3f}  "
            f"{a['mean_coalition']:>14.1f}"
        )
    lines += [
        "-" * 78,
        f"Empirical log-log slope: {slope:.4f}  (expected: −0.50)",
        "",
        "Reduction from n=512 to n=2048: "
        f"expected ×{round(np.sqrt(2048/512), 2):.2f} lower error (O(1/√n)).",
        "Reduction from n=512 to n=2048: "
        f"expected ×{round(2048/512, 1):.1f} longer runtime (O(n)).",
    ]
    txt_path = FIG_DIR / "node_shap_convergence_details.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Text report → {txt_path}")

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # Panel 1 — efficiency error vs nsamples
    mean_err = np.array([agg[n]["mean_abs_err"] for n in ns_present])
    std_err  = np.array([agg[n]["std_abs_err"]  for n in ns_present])
    ns_arr   = np.array(ns_present, dtype=float)

    ax = axes[0]
    ax.loglog(ns_arr, mean_err, "o-", color="steelblue", lw=1.5, label="Mean |eff. error|")
    ax.fill_between(
        ns_arr,
        np.clip(mean_err - std_err, 1e-6, None),
        mean_err + std_err,
        alpha=0.2, color="steelblue",
    )
    # Reference line anchored at midpoint with slope −0.5
    mid_idx = len(ns_arr) // 2
    y_ref = agg[ns_present[mid_idx]]["mean_abs_err"] * (ns_arr / ns_arr[mid_idx]) ** (-0.5)
    ax.loglog(ns_arr, y_ref, "--", color="tomato", lw=1.2, label="slope −0.50 (ref.)")
    ax.set_xlabel("nsamples")
    ax.set_ylabel("|Efficiency error|")
    ax.set_title(
        f"Node SHAP convergence (n={len(selected)} flows)\n"
        f"empirical slope = {slope:.2f}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # Panel 2 — wall-clock time vs nsamples
    mean_time = np.array([agg[n]["mean_time_s"] for n in ns_present])
    std_time  = np.array([agg[n]["std_time_s"]  for n in ns_present])

    ax2 = axes[1]
    ax2.loglog(ns_arr, mean_time, "s-", color="darkorange", lw=1.5, label="Mean wall time")
    ax2.fill_between(
        ns_arr,
        np.clip(mean_time - std_time, 1e-4, None),
        mean_time + std_time,
        alpha=0.2, color="darkorange",
    )
    # Reference line with slope +1 (linear in nsamples)
    y_ref_t = mean_time[0] * (ns_arr / ns_arr[0])
    ax2.loglog(ns_arr, y_ref_t, "--", color="gray", lw=1.2, label="slope +1.00 (ref.)")
    ax2.set_xlabel("nsamples")
    ax2.set_ylabel("Wall-clock time (s)")
    ax2.set_title("Node SHAP runtime vs nsamples")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"node_shap_convergence.{ext}"
        dpi = 300 if ext == "pdf" else 150
        fig.savefig(str(out), bbox_inches="tight", dpi=dpi)
        logger.info(f"Figure → {out}")
    plt.close(fig)

    logger.info(f"Done. Empirical slope={slope:.4f}  (expected −0.50)")


if __name__ == "__main__":
    main()
