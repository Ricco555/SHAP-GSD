"""
Phase 7 — Case study visualizations for SHAP-GSD paper.

Produces four-panel figures for four candidate flows:
  Recon (EID 2232037), Generic (EID 2239466),
  Worms (EID 2228887), Backdoor (EID 2229558).

Outputs:
  outputs/figures/case_studies/<class>_<eid>.pdf
  outputs/figures/case_studies/<class>_<eid>.png
  outputs/figures/case_studies/quality_report.txt

Usage:
  python scripts/07_visualize.py --config configs/experiment_unsw.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import dgl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.model.temporal_sampler import TemporalNeighborSampler
from src.visualization.case_study_plots import (
    make_case_study_figure,
    plot_class_feature_summary,
    make_all_classes_figure,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Four paper candidates: (class_name, global_eid)
CANDIDATES = [
    ("Recon",    2232037),
    ("Generic",  2239466),
    ("Worms",    2228887),
    ("Backdoor", 2229558),
]


def _find_local_eid(g: dgl.DGLGraph, global_eid: int) -> int:
    """Return the local EID in g whose global EID matches."""
    global_eids = g.edata[dgl.EID]
    matches = (global_eids == global_eid).nonzero(as_tuple=True)[0]
    if len(matches) == 0:
        raise ValueError(f"Global EID {global_eid} not found in graph")
    return int(matches[0].item())


def _build_topo(
    g: dgl.DGLGraph,
    blocks: list,
    local_eid: int,
    target_ts_ms: float,
) -> dict:
    """Extract hop1/hop2 node sets and all neighbour edge gaps (seconds).

    Args:
        g:            test split DGL graph.
        blocks:       computation blocks from TemporalNeighborSampler.
        local_eid:    local EID of the target edge in g.
        target_ts_ms: target edge timestamp in milliseconds.

    Returns:
        dict with keys 'hop1_nodes', 'hop2_nodes', 'all_neighbor_gaps_s'.
    """
    # Hop membership from block node sets
    seed_set   = set(blocks[1].dstdata[dgl.NID].tolist())
    block1_src = set(blocks[1].srcdata[dgl.NID].tolist())
    block0_src = set(blocks[0].srcdata[dgl.NID].tolist())

    hop1_nodes = list(block1_src - seed_set)
    hop2_nodes = list(block0_src - block1_src)

    # All neighbour edge gaps (target_ts − edge_ts) / 1000.0 s
    seen_leids: set[int] = set()
    gaps: list[float] = []
    for block in blocks:
        if dgl.EID not in block.edata:
            continue
        for leid in block.edata[dgl.EID].tolist():
            if leid == local_eid or leid in seen_leids:
                continue
            seen_leids.add(leid)
            ts = float(g.edata["timestamp"][leid].item())
            gaps.append(max((target_ts_ms - ts) / 1000.0, 0.0))

    return {
        "hop1_nodes":         hop1_nodes,
        "hop2_nodes":         hop2_nodes,
        "all_neighbor_gaps_s": np.array(gaps, dtype=np.float64),
    }


def _quality_score(explanation: dict, topo: dict) -> dict:
    """Compute simple quality metrics for case study selection."""
    shap = np.array(explanation["feature_group_shap"])
    top_phi = float(np.abs(shap).max())
    shap_range = float(np.abs(shap).max() - np.abs(shap).min())
    n_topology = len(topo["hop1_nodes"]) + len(topo["hop2_nodes"])

    node_dict = explanation.get("node_shap_dict", {})
    node_vals = np.array(list(node_dict.values())) if node_dict else np.array([0.0])
    node_range = float(np.abs(node_vals).max())

    gaps = topo["all_neighbor_gaps_s"]
    in_window = int((gaps <= 60.0).sum()) if len(gaps) > 0 else 0
    return {
        "top_feature_phi":    round(top_phi, 4),
        "shap_range":         round(shap_range, 4),
        "n_topology_nodes":   n_topology,
        "node_shap_range":    round(node_range, 4),
        "n_neighbors":        len(gaps),
        "in_window_neighbors": in_window,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP-GSD Phase 7 — Visualizations")
    parser.add_argument("--config", required=True,
                        help="Path to experiment YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info("=== Phase 7: Case Study Visualizations ===")

    graphs_dir  = Path(cfg["graph"]["dir"])
    W_seconds   = float(cfg["model"]["temporal_window_seconds"])
    fanouts     = cfg["model"]["fanouts"]

    # --- Load test graph ---
    logger.info("Loading test graph …")
    g_test, _ = dgl.load_graphs(str(graphs_dir / "test.bin"))
    g_test = g_test[0]

    # --- Build id → IP lookup ---
    nm_path = graphs_dir / "node_id_map.json"
    with open(nm_path) as f:
        ip_to_id: dict[str, int] = json.load(f)
    id2ip: dict[int, str] = {v: k for k, v in ip_to_id.items()}

    # --- Sampler ---
    sampler = TemporalNeighborSampler(fanouts=fanouts)

    # --- Output directory ---
    out_dir = Path("outputs") / "figures" / "case_studies"
    out_dir.mkdir(parents=True, exist_ok=True)

    quality_rows: list[str] = []
    explanations_dir = Path("outputs") / "explanations"

    for class_name, global_eid in CANDIDATES:
        logger.info(f"Processing {class_name} EID={global_eid} …")

        json_path = explanations_dir / class_name / f"{global_eid}_fixed.json"
        if not json_path.exists():
            logger.warning(f"  Missing: {json_path} — skipping")
            continue

        with open(json_path) as f:
            explanation = json.load(f)

        # Find local EID and get target timestamp
        local_eid = _find_local_eid(g_test, global_eid)
        target_ts_ms = float(g_test.edata["timestamp"][local_eid].item())
        logger.info(f"  local_eid={local_eid}, target_ts_ms={target_ts_ms:.0f}")

        # Run temporal sampler
        seed_t = torch.tensor([local_eid], dtype=torch.long)
        input_nodes, seed_eids, blocks = sampler.sample_blocks(g_test, seed_t)

        # Build topology dict
        topo = _build_topo(g_test, blocks, local_eid, target_ts_ms)
        logger.info(
            f"  hop1={len(topo['hop1_nodes'])} hop2={len(topo['hop2_nodes'])} "
            f"n_gaps={len(topo['all_neighbor_gaps_s'])} "
            f"in_window={int((topo['all_neighbor_gaps_s'] <= W_seconds).sum())}"
        )

        # Build figure
        fig = make_case_study_figure(
            explanation=explanation,
            topo=topo,
            id2ip=id2ip,
            class_name=class_name,
            W_seconds=W_seconds,
            top_feat=20,
            global_eid=global_eid,
        )

        # Save PDF and PNG
        stem = f"{class_name}_{global_eid}"
        pdf_path = out_dir / f"{stem}.pdf"
        png_path = out_dir / f"{stem}.png"
        fig.savefig(str(pdf_path), bbox_inches="tight", dpi=300)
        fig.savefig(str(png_path), bbox_inches="tight", dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig)
        logger.info(f"  Saved {pdf_path.name} and {png_path.name}")

        # Quality metrics
        q = _quality_score(explanation, topo)
        row = (
            f"{class_name:12s} EID={global_eid}  "
            f"top_phi={q['top_feature_phi']:.4f}  "
            f"shap_range={q['shap_range']:.4f}  "
            f"n_topo={q['n_topology_nodes']:2d}  "
            f"node_range={q['node_shap_range']:.4f}  "
            f"n_gaps={q['n_neighbors']:3d}  "
            f"in_W={q['in_window_neighbors']}"
        )
        quality_rows.append(row)
        logger.info(f"  Quality: {row}")

    # Write quality report
    report_path = out_dir / "quality_report.txt"
    header = (
        "Case study quality report — SHAP-GSD\n"
        f"W_seconds={W_seconds}\n"
        + "-" * 90 + "\n"
        + "Class        EID         top_phi  shap_range  n_topo  node_range  n_gaps  in_W\n"
        + "-" * 90 + "\n"
    )
    with open(report_path, "w") as f:
        f.write(header)
        for row in quality_rows:
            f.write(row + "\n")

    logger.info(f"Quality report → {report_path}")

    # ── Phase 7B: per-class feature-group summary plots ───────────────────────
    logger.info("=== Phase 7B: Per-class feature-group SHAP summaries ===")
    fg_dir = Path("outputs") / "figures" / "feature_groups"
    fg_dir.mkdir(parents=True, exist_ok=True)

    # Load all explanation JSONs grouped by class
    class_shap: dict[str, np.ndarray] = {}
    group_names_ref: list[str] | None = None

    for cls_dir in sorted(explanations_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        matrices = []
        for jf in sorted(f for f in cls_dir.glob("*.json")
                         if f.stem.isdigit()):
            try:
                rec = json.loads(jf.read_text())
                phi = np.array(rec["feature_group_shap"], dtype=np.float64)
                matrices.append(phi)
                if group_names_ref is None:
                    group_names_ref = rec["feature_group_names"]
            except Exception as e:
                logger.debug(f"Skipping {jf.name}: {e}")
        if matrices:
            class_shap[cls] = np.stack(matrices)
            logger.info(f"  {cls}: {len(matrices)} flows loaded")

    if not class_shap or group_names_ref is None:
        logger.warning("No explanation JSONs found — skipping Phase 7B")
    else:
        import matplotlib.pyplot as _plt

        # Per-class figures
        for cls, mat in class_shap.items():
            fig, ax = _plt.subplots(figsize=(6, 8))
            plot_class_feature_summary(ax, group_names_ref, mat, cls, top_n=20)
            for ext in ("pdf", "png"):
                path = fg_dir / f"{cls}.{ext}"
                fig.savefig(str(path), bbox_inches="tight",
                            dpi=300 if ext == "pdf" else 150)
            _plt.close(fig)
            logger.info(f"  Saved {cls}.pdf / {cls}.png")

        # Combined all-classes figure
        fig_all = make_all_classes_figure(class_shap, group_names_ref, top_n=15)
        for ext in ("pdf", "png"):
            path = fg_dir / f"all_classes.{ext}"
            fig_all.savefig(str(path), bbox_inches="tight",
                            dpi=300 if ext == "pdf" else 150)
        _plt.close(fig_all)
        logger.info(f"  Saved all_classes.pdf / all_classes.png")
        logger.info(f"Feature-group summaries → {fg_dir}")

    logger.info("Phase 7 complete.")


if __name__ == "__main__":
    main()
