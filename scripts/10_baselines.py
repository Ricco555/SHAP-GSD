"""
Phase 10 — Baseline explainer comparison for Table 2 of SHAP-GSD paper.

Runs five baseline explainers on the same flows explained by SHAP-GSD in
Phase 6 (outputs/explanations/) and reports Fidelity+ / Fidelity− per class.

Baselines implemented:
  C1. GNNExplainer  — gradient-based, per-feature (218-dim → 48 groups), feature fidelity
  C2. PGExplainer   — inductive parametric, node-coalition, node fidelity
  C3. GNNShap       — KernelSHAP over edge coalitions, node fidelity
  C4. GraphSVX      — WLS surrogate, node-coalition (structure-only), node fidelity
  C5. EdgeSHAPer    — Monte Carlo marginal contributions, edge → node fidelity

Note on coalition spaces (reported in Table 2 header):
  GNNExplainer operates on 218 raw feature dims (aggregated to 48 groups).
  All others (PGExplainer, GNNShap, GraphSVX, EdgeSHAPer) attribute to nodes
  in the k-hop subgraph.  SHAP-GSD covers all three granularities.

Outputs (outputs/baselines/):
  <baseline>_results.csv     — per-flow: edge_id, true_label, fidelity_plus, fidelity_minus, runtime_s
  comparison_table.txt       — ASCII table (per-class, all baselines)
  summary.json               — machine-readable full results

Usage:
  # Full journal run (all 1764 flows):
  python scripts/10_baselines.py --config configs/experiment_unsw.yaml

  # Conference subset (20 flows × 10 classes = 200):
  python scripts/10_baselines.py --config configs/experiment_unsw.yaml --n-per-class 20

  # Single baseline:
  python scripts/10_baselines.py --config configs/experiment_unsw.yaml --baselines gnnexplainer

  # Skip PGExplainer training (use cached):
  python scripts/10_baselines.py --config configs/experiment_unsw.yaml --skip-pg-train
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
from src.model.sage_model import EdgeAwareGraphSAGE
from src.explainer.background import BackgroundDistributions
from src.model.temporal_sampler import TemporalNeighborSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("shap").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

BASELINE_NAMES = ["gnnexplainer", "pgexplainer", "gnnshap", "graphsvx", "edgeshaper"]


# ── infrastructure loaders ─────────────────────────────────────────────────────

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


def _load_explained_flows(
    explanations_dir: Path,
    n_per_class: int,
    seed: int = 42,
) -> list[dict]:
    """Load per-flow explanation records from outputs/explanations/.

    Each record has at minimum: edge_id, true_label, _class_name.
    Optionally sub-samples to n_per_class per class (0 = all).
    """
    records = []
    for class_dir in sorted(explanations_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        class_records = []
        for jf in sorted(class_dir.glob("*.json")):
            if jf.stem.endswith("_fixed"):
                continue
            with open(jf) as f:
                d = json.load(f)
            d["_class_name"] = class_name
            class_records.append(d)

        if n_per_class > 0 and len(class_records) > n_per_class:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(class_records), size=n_per_class, replace=False)
            class_records = [class_records[i] for i in sorted(idx)]

        records.extend(class_records)
        logger.info(f"  {class_name}: {len(class_records)} flows")

    logger.info(f"Total flows for baseline evaluation: {len(records)}")
    return records


# ── PGExplainer training ───────────────────────────────────────────────────────

def _train_or_load_pgexplainer(
    cfg: dict,
    model: EdgeAwareGraphSAGE,
    nsm: NodeStateManager,
    fs_train: FeatureStore,
    sampler: TemporalNeighborSampler,
    device: torch.device,
    artifacts_dir: Path,
    n_train: int,
    pg_epochs: int,
    pg_lr: float,
    skip_training: bool,
):
    """Train PGExplainer or load a saved checkpoint if available and skip_training set."""
    from src.baselines.pgexplainer_wrapper import train_pgexplainer

    ckpt_path = artifacts_dir / "pgexplainer_algorithm.pt"

    if skip_training and ckpt_path.exists():
        logger.info(f"Loading cached PGExplainer from {ckpt_path}")
        from torch_geometric.explain.algorithm import PGExplainer
        algorithm = PGExplainer(epochs=pg_epochs, lr=pg_lr)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        algorithm.mlp = state["mlp"]
        algorithm.optimizer = None  # not needed for inference
        algorithm._curr_epoch = pg_epochs - 1  # mark training as complete
        return algorithm

    logger.info(
        f"Training PGExplainer on {n_train} training flows "
        f"(epochs={pg_epochs}, lr={pg_lr}) …"
    )
    graphs_dir = Path(cfg["graph"]["dir"])
    g_train, _ = dgl.load_graphs(str(graphs_dir / "train.bin"))
    g_train = g_train[0]

    algorithm = train_pgexplainer(
        model=model,
        g_train=g_train,
        nsm=nsm,
        fs=fs_train,
        sampler=sampler,
        device=device,
        n_train=n_train,
        epochs=pg_epochs,
        lr=pg_lr,
        seed=cfg.get("seed", 42),
    )
    del g_train

    # Cache the MLP weights
    if algorithm.mlp is not None:
        torch.save({"mlp": algorithm.mlp}, ckpt_path)
        logger.info(f"PGExplainer saved → {ckpt_path}")

    return algorithm


# ── per-flow baseline dispatch ────────────────────────────────────────────────

def _run_one_flow(
    baseline_name: str,
    global_eid: int,
    model: EdgeAwareGraphSAGE,
    g_test,
    nsm: NodeStateManager,
    fs_test: FeatureStore,
    sampler: TemporalNeighborSampler,
    feature_groups: dict,
    background: BackgroundDistributions,
    device: torch.device,
    pg_algorithm=None,
    gnnexplainer_epochs: int = 200,
    gnnshap_nsamples: int = 256,
    graphsvx_nsamples: int = 256,
    edgeshaper_M: int = 100,
    top_k_feature: int = 5,
    top_k_node: int = 3,
) -> dict | None:
    """Build FlowContext and run one baseline on one flow."""
    from src.baselines.adapter import build_flow_context

    try:
        ctx = build_flow_context(
            global_eid=global_eid,
            model=model,
            g_test=g_test,
            nsm=nsm,
            fs=fs_test,
            sampler=sampler,
            device=device,
        )
    except Exception as exc:
        logger.warning(f"{baseline_name} EID={global_eid}: build_flow_context failed: {exc}")
        return None

    try:
        if baseline_name == "gnnexplainer":
            from src.baselines.gnnexplainer_wrapper import run_gnnexplainer_with_model
            return run_gnnexplainer_with_model(
                ctx=ctx,
                model=model,
                feature_groups=feature_groups,
                background=background,
                epochs=gnnexplainer_epochs,
                top_k=top_k_feature,
            )

        elif baseline_name == "pgexplainer":
            from src.baselines.pgexplainer_wrapper import run_pgexplainer_with_model
            return run_pgexplainer_with_model(
                ctx=ctx,
                model=model,
                algorithm=pg_algorithm,
                feature_groups=feature_groups,
                background=background,
                g_test=g_test,
                top_k=top_k_node,
            )

        elif baseline_name == "gnnshap":
            from src.baselines.gnnshap_wrapper import run_gnnshap_with_model
            return run_gnnshap_with_model(
                ctx=ctx,
                model=model,
                feature_groups=feature_groups,
                background=background,
                g_test=g_test,
                nsamples=gnnshap_nsamples,
                top_k=top_k_node,
            )

        elif baseline_name == "graphsvx":
            from src.baselines.graphsvx_wrapper import run_graphsvx_with_model
            return run_graphsvx_with_model(
                ctx=ctx,
                model=model,
                feature_groups=feature_groups,
                background=background,
                g_test=g_test,
                num_samples=graphsvx_nsamples,
                top_k=top_k_node,
            )

        elif baseline_name == "edgeshaper":
            from src.baselines.edgeshaper_wrapper import run_edgeshaper_with_model
            return run_edgeshaper_with_model(
                ctx=ctx,
                model=model,
                feature_groups=feature_groups,
                background=background,
                g_test=g_test,
                M=edgeshaper_M,
                top_k=top_k_node,
            )

    except Exception as exc:
        logger.warning(f"{baseline_name} EID={global_eid}: explain failed: {exc}")
        return None

    return None


# ── per-baseline full run ──────────────────────────────────────────────────────

def _run_baseline(
    baseline_name: str,
    records: list[dict],
    model: EdgeAwareGraphSAGE,
    g_test,
    nsm: NodeStateManager,
    fs_test: FeatureStore,
    sampler: TemporalNeighborSampler,
    feature_groups: dict,
    background: BackgroundDistributions,
    device: torch.device,
    output_dir: Path,
    pg_algorithm=None,
    **kwargs,
) -> list[dict]:
    """Run one baseline on all records, save CSV, return results list."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running baseline: {baseline_name.upper()} on {len(records)} flows")
    logger.info(f"{'='*60}")

    results = []
    t_total = time.time()
    n_ok, n_err = 0, 0

    for i, rec in enumerate(records):
        global_eid = rec["edge_id"]
        result = _run_one_flow(
            baseline_name=baseline_name,
            global_eid=global_eid,
            model=model,
            g_test=g_test,
            nsm=nsm,
            fs_test=fs_test,
            sampler=sampler,
            feature_groups=feature_groups,
            background=background,
            device=device,
            pg_algorithm=pg_algorithm,
            **kwargs,
        )
        if result is not None:
            result["_class_name"] = rec.get("_class_name", "unknown")
            results.append(result)
            n_ok += 1
        else:
            n_err += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_total
            rate = (i + 1) / elapsed
            eta = (len(records) - i - 1) / rate if rate > 0 else 0
            logger.info(
                f"  {baseline_name}: {i+1}/{len(records)}  "
                f"ok={n_ok} err={n_err}  "
                f"rate={rate:.1f}f/s  ETA={eta/60:.1f}min"
            )

    # Save CSV
    csv_path = output_dir / f"{baseline_name}_results.csv"
    if results:
        fieldnames = [k for k in results[0].keys() if not k.startswith("_")]
        fieldnames += ["_class_name"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
    logger.info(
        f"{baseline_name}: done. {n_ok}/{len(records)} OK  "
        f"({time.time()-t_total:.0f}s) → {csv_path}"
    )
    return results


# ── comparison table ──────────────────────────────────────────────────────────

def _build_comparison_table(
    all_results: dict[str, list[dict]],
    int_to_name: dict[int, str],
    out_path: Path,
) -> dict:
    """Build per-class and overall comparison table."""
    class_names = sorted(set(int_to_name.values()))
    summary = {}

    for bl_name, records in all_results.items():
        class_fid = {}
        for rec in records:
            cn = rec.get("_class_name", "unknown")
            if cn not in class_fid:
                class_fid[cn] = {"fid_plus": [], "fid_minus": [], "runtime": []}
            class_fid[cn]["fid_plus"].append(rec["fidelity_plus"])
            class_fid[cn]["fid_minus"].append(rec["fidelity_minus"])
            class_fid[cn]["runtime"].append(rec["runtime_s"])

        all_fp = [r["fidelity_plus"]  for r in records]
        all_fm = [r["fidelity_minus"] for r in records]
        all_rt = [r["runtime_s"]      for r in records]

        bl_summary = {
            "per_class": {},
            "overall": {
                "fidelity_plus_mean":  round(float(np.mean(all_fp)) if all_fp else 0, 4),
                "fidelity_plus_std":   round(float(np.std(all_fp))  if all_fp else 0, 4),
                "fidelity_minus_mean": round(float(np.mean(all_fm)) if all_fm else 0, 4),
                "fidelity_minus_std":  round(float(np.std(all_fm))  if all_fm else 0, 4),
                "runtime_mean_s":      round(float(np.mean(all_rt)) if all_rt else 0, 3),
                "n_flows":             len(records),
            },
        }
        for cn, stats in class_fid.items():
            bl_summary["per_class"][cn] = {
                "fidelity_plus_mean":  round(float(np.mean(stats["fid_plus"]))  if stats["fid_plus"]  else 0, 4),
                "fidelity_minus_mean": round(float(np.mean(stats["fid_minus"])) if stats["fid_minus"] else 0, 4),
                "n":                   len(stats["fid_plus"]),
            }
        summary[bl_name] = bl_summary

    # ASCII table
    lines = []
    lines.append("BASELINE EXPLAINER COMPARISON — SHAP-GSD Paper 2 Table 2")
    lines.append("="*80)
    lines.append(
        "Note: GNNExplainer uses feature-group fidelity (top-5 groups, 218→48 dims).")
    lines.append(
        "      All others use node-coalition fidelity (top-3 nodes from k-hop subgraph).")
    lines.append("="*80)
    lines.append("")

    header = f"{'Class':<14}" + "".join(
        f"  {bl:>12}" for bl in BASELINE_NAMES
    )
    lines.append(f"{'':>14}  {'Fidelity+ (↑ better)':^60}")
    lines.append(header)
    lines.append("-" * len(header))

    for cn in sorted(class_names):
        row = f"{cn:<14}"
        for bl in BASELINE_NAMES:
            v = summary.get(bl, {}).get("per_class", {}).get(cn, {}).get("fidelity_plus_mean", float("nan"))
            row += f"  {v:>12.4f}"
        lines.append(row)

    lines.append("-" * len(header))
    overall_row = f"{'Overall':<14}"
    for bl in BASELINE_NAMES:
        v = summary.get(bl, {}).get("overall", {}).get("fidelity_plus_mean", float("nan"))
        overall_row += f"  {v:>12.4f}"
    lines.append(overall_row)

    lines.append("")
    lines.append(f"{'':>14}  {'Fidelity- (↓ better)':^60}")
    lines.append(header)
    lines.append("-" * len(header))

    for cn in sorted(class_names):
        row = f"{cn:<14}"
        for bl in BASELINE_NAMES:
            v = summary.get(bl, {}).get("per_class", {}).get(cn, {}).get("fidelity_minus_mean", float("nan"))
            row += f"  {v:>12.4f}"
        lines.append(row)

    lines.append("-" * len(header))
    overall_row = f"{'Overall':<14}"
    for bl in BASELINE_NAMES:
        v = summary.get(bl, {}).get("overall", {}).get("fidelity_minus_mean", float("nan"))
        overall_row += f"  {v:>12.4f}"
    lines.append(overall_row)

    lines.append("")
    lines.append(f"{'':>14}  {'Runtime mean (s/flow)':^60}")
    lines.append(header)
    lines.append("-" * len(header))
    rt_row = f"{'Mean runtime':<14}"
    for bl in BASELINE_NAMES:
        v = summary.get(bl, {}).get("overall", {}).get("runtime_mean_s", float("nan"))
        rt_row += f"  {v:>12.3f}"
    lines.append(rt_row)

    table_str = "\n".join(lines)
    out_path.write_text(table_str)
    logger.info(f"Comparison table → {out_path}")
    print("\n" + table_str + "\n")

    return summary


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run baseline explainers for Table 2.")
    parser.add_argument("--config", default="configs/experiment_unsw.yaml")
    parser.add_argument(
        "--n-per-class", type=int, default=0,
        help="Flows per class (0 = all explained flows from Phase 6).",
    )
    parser.add_argument(
        "--baselines", default="all",
        help=f"Comma-separated subset of: {','.join(BASELINE_NAMES)}, or 'all'.",
    )
    parser.add_argument(
        "--n-train-pg", type=int, default=200,
        help="Number of training flows for PGExplainer training.",
    )
    parser.add_argument("--pg-epochs",   type=int,   default=30)
    parser.add_argument("--pg-lr",       type=float, default=0.003)
    parser.add_argument(
        "--skip-pg-train", action="store_true",
        help="Use cached PGExplainer checkpoint if available.",
    )
    parser.add_argument("--gnn-epochs",   type=int,   default=200,  help="GNNExplainer epochs.")
    parser.add_argument("--gnn-lr",       type=float, default=0.01, help="GNNExplainer LR.")
    parser.add_argument("--gnnshap-n",    type=int,   default=256,  help="GNNShap nsamples.")
    parser.add_argument("--graphsvx-n",   type=int,   default=256,  help="GraphSVX num_samples.")
    parser.add_argument("--edgeshaper-M", type=int,   default=100,  help="EdgeSHAPer M.")
    parser.add_argument("--top-k-feat",   type=int,   default=5,    help="Top-k for feature fidelity.")
    parser.add_argument("--top-k-node",   type=int,   default=3,    help="Top-k for node fidelity.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── paths ──
    artifacts_dir    = Path(cfg["output"]["artifacts_dir"])
    graphs_dir       = Path(cfg["graph"]["dir"])
    nsm_dir          = Path(cfg["graph"]["node_state_dir"])
    fs_train_dir     = Path(cfg["output"]["feature_store_dir"]) / "train"
    fs_test_dir      = Path(cfg["output"]["feature_store_dir"]) / "test"
    explanations_dir = Path("outputs") / "explanations"
    output_dir       = Path("outputs") / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        cfg.get("compute", {}).get("device", "cuda")
        if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Device: {device}")

    # ── determine which baselines to run ──
    if args.baselines.lower() == "all":
        run_names = BASELINE_NAMES
    else:
        run_names = [b.strip().lower() for b in args.baselines.split(",")]
        invalid = [n for n in run_names if n not in BASELINE_NAMES]
        if invalid:
            raise ValueError(f"Unknown baselines: {invalid}. Valid: {BASELINE_NAMES}")

    # ── load infrastructure ──
    logger.info("Loading test graph …")
    g_test, _ = dgl.load_graphs(str(graphs_dir / "test.bin"))
    g_test = g_test[0]

    logger.info("Loading feature stores …")
    fs_test  = FeatureStore(fs_test_dir)
    fs_train = FeatureStore(fs_train_dir)

    logger.info("Loading NodeStateManager …")
    nsm = NodeStateManager.load(nsm_dir)

    logger.info("Loading background distributions …")
    background = BackgroundDistributions.load(artifacts_dir)

    logger.info("Loading model …")
    model = _load_model(cfg, device)

    fg_path = Path(cfg["output"]["feature_groups_path"])
    with open(fg_path) as f:
        feature_groups = json.load(f)

    with open(artifacts_dir / "label_map.json") as f:
        label_map = json.load(f)
    int_to_name = {v: k for k, v in label_map.items()}

    sampler = TemporalNeighborSampler(fanouts=cfg["model"]["fanouts"])

    # ── load explained flows ──
    logger.info(f"Loading explained flows from {explanations_dir} …")
    records = _load_explained_flows(
        explanations_dir, n_per_class=args.n_per_class, seed=cfg.get("seed", 42)
    )

    # ── PGExplainer training (if needed) ──
    pg_algorithm = None
    if "pgexplainer" in run_names:
        pg_algorithm = _train_or_load_pgexplainer(
            cfg=cfg,
            model=model,
            nsm=nsm,
            fs_train=fs_train,
            sampler=sampler,
            device=device,
            artifacts_dir=artifacts_dir,
            n_train=args.n_train_pg,
            pg_epochs=args.pg_epochs,
            pg_lr=args.pg_lr,
            skip_training=args.skip_pg_train,
        )

    # ── run baselines ──
    all_results: dict[str, list[dict]] = {}
    kwargs = dict(
        gnnexplainer_epochs=args.gnn_epochs,
        gnnshap_nsamples=args.gnnshap_n,
        graphsvx_nsamples=args.graphsvx_n,
        edgeshaper_M=args.edgeshaper_M,
        top_k_feature=args.top_k_feat,
        top_k_node=args.top_k_node,
    )

    for bl_name in run_names:
        results = _run_baseline(
            baseline_name=bl_name,
            records=records,
            model=model,
            g_test=g_test,
            nsm=nsm,
            fs_test=fs_test,
            sampler=sampler,
            feature_groups=feature_groups,
            background=background,
            device=device,
            output_dir=output_dir,
            pg_algorithm=pg_algorithm if bl_name == "pgexplainer" else None,
            **kwargs,
        )
        all_results[bl_name] = results

    # ── comparison table ──
    summary = _build_comparison_table(
        all_results=all_results,
        int_to_name=int_to_name,
        out_path=output_dir / "comparison_table.txt",
    )

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary JSON → {output_dir / 'summary.json'}")

    logger.info("Phase 10 complete.")


if __name__ == "__main__":
    main()
