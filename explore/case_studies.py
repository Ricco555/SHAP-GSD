"""
Case study figures for all 9 attack classes.

Produces a 4-panel figure per class:
  (a) Feature-group SHAP bar chart (top 20 by |φ|)
  (b) 2-hop neighbourhood topology
  (c) Node SHAP bar chart
  (d) Temporal neighbour gap histogram

Selected flows:
  Classes with temporal neighbours (prioritised):
    Fuzzers   2117155  (28 in-window neighbours)
    Shellcode 2117054  (25)
    Exploits  2117278  (25)
    Analysis  2117614  (25)
    DoS       2117597  (13)
  Classes without temporal neighbours (best by proba × phi_N_frac):
    Backdoor  2174974
    Generic   2247735
    Worms     2310720
  Model/reference case (regenerated for consistency):
    Recon     2232037

Outputs:
  outputs/figures/case_studies/<Class>_<EID>.{pdf,png}
  outputs/figures/case_studies/<Class>_<EID>_summary.json
  outputs/figures/case_studies/quality_report.txt  (all 9 cases)

Reads:
  outputs/explanations/<Class>/<EID>.json
  graphs/test.bin
  graphs/node_id_map.json
  configs/experiment_unsw.yaml
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import dgl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from explore._paths import paths  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.model.temporal_sampler import TemporalNeighborSampler  # noqa: E402
from src.visualization.case_study_plots import make_case_study_figure  # noqa: E402

_P = paths()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Candidate flows ────────────────────────────────────────────────────────────
# (class_name, global_eid, has_temporal_nbrs)
CANDIDATES = [
    ("Fuzzers",   2117155, True),
    ("Shellcode", 2117054, True),
    ("Exploits",  2117278, True),
    ("Analysis",  2117614, True),
    ("DoS",       2117597, True),
    ("Backdoor",  2174974, False),
    ("Generic",   2247735, False),
    ("Worms",     2310720, False),
    ("Recon",     2232037, False),   # model case, regenerated for consistency
]

EXPL_DIR = _P["explanations"]
OUT_DIR  = _P["figures"] / "case_studies"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _find_local_eid(g: dgl.DGLGraph, global_eid: int) -> int:
    matches = (g.edata[dgl.EID] == global_eid).nonzero(as_tuple=True)[0]
    if len(matches) == 0:
        raise ValueError(f"Global EID {global_eid} not found in graph")
    return int(matches[0].item())


def _build_topo(
    g: dgl.DGLGraph,
    blocks: list,
    local_eid: int,
    target_ts_ms: float,
    W_seconds: float,
) -> dict:
    """Extract hop1/hop2 nodes and all neighbour edge gaps."""
    seed_set   = set(blocks[1].dstdata[dgl.NID].tolist())
    block1_src = set(blocks[1].srcdata[dgl.NID].tolist())
    block0_src = set(blocks[0].srcdata[dgl.NID].tolist())

    hop1_nodes = list(block1_src - seed_set)
    hop2_nodes = list(block0_src - block1_src)

    seen: set[int] = set()
    gaps: list[float] = []
    for block in blocks:
        if dgl.EID not in block.edata:
            continue
        for leid in block.edata[dgl.EID].tolist():
            if leid == local_eid or leid in seen:
                continue
            seen.add(leid)
            ts = float(g.edata["timestamp"][leid].item())
            gaps.append(max((target_ts_ms - ts) / 1000.0, 0.0))

    return {
        "hop1_nodes":          hop1_nodes,
        "hop2_nodes":          hop2_nodes,
        "all_neighbor_gaps_s": np.array(gaps, dtype=np.float64),
    }


def _adapt_explanation(plain: dict, src_nid: int, dst_nid: int) -> dict:
    """Convert plain JSON (node_ids + node_shap lists) to figure-compatible dict."""
    node_ids  = plain.get("node_ids", [])
    node_shap = plain.get("node_shap", [])
    node_shap_dict = {
        str(nid): float(phi)
        for nid, phi in zip(node_ids, node_shap)
    }
    adapted = dict(plain)
    adapted["src_nid"]        = src_nid
    adapted["dst_nid"]        = dst_nid
    adapted["node_shap_dict"] = node_shap_dict
    return adapted


def _summary(plain: dict, topo: dict, W_seconds: float) -> dict:
    """Build numeric summary JSON for manuscript/audit use."""
    fg   = np.array(plain["feature_group_shap"])
    ns   = np.array(plain.get("node_shap", []))
    nbrs = plain.get("neighbor_shap", [])
    gaps = topo["all_neighbor_gaps_s"]

    top_fg_i   = int(np.argmax(np.abs(fg)))
    fg_abs     = float(np.abs(fg).sum())
    ns_abs     = float(np.abs(ns).sum()) if len(ns) > 0 else 0.0
    nov_abs    = abs(plain.get("src_novelty_shap", 0.0)) + abs(plain.get("dst_novelty_shap", 0.0))
    total      = fg_abs + ns_abs + nov_abs
    phi_n_frac = (ns_abs + nov_abs) / total if total > 0 else 0.0

    in_window   = int((gaps <= W_seconds).sum()) if len(gaps) > 0 else 0
    top_t_phi   = float(max(abs(x) for x in nbrs)) if nbrs else 0.0

    return {
        "edge_id":            plain["edge_id"],
        "true_label":         plain.get("true_label"),
        "predicted_label":    plain.get("predicted_label"),
        "predicted_proba":    plain.get("predicted_proba"),
        "W_seconds":          W_seconds,
        "src_nid":            plain.get("src_nid"),
        "dst_nid":            plain.get("dst_nid"),
        "n_temporal_nbrs":    len(plain.get("neighbor_edge_ids", [])),
        "n_in_window":        in_window,
        "top_temporal_shap":  top_t_phi,
        "node_shap_list":     plain.get("node_shap", []),
        "src_novelty_shap":   plain.get("src_novelty_shap", 0.0),
        "dst_novelty_shap":   plain.get("dst_novelty_shap", 0.0),
        "top_fg_name":        plain["feature_group_names"][top_fg_i],
        "top_fg_phi":         float(fg[top_fg_i]),
        "sum_phi_N":          ns_abs,
        "phi_N_fraction":     round(phi_n_frac, 4),
        "runtime_s":          plain.get("runtime_s"),
        "json_path":          str(EXPL_DIR / str(plain.get("true_label", "?")) / f"{plain['edge_id']}.json"),
    }


def main() -> None:
    cfg = _P["cfg"]
    W_seconds = float(cfg["model"]["temporal_window_seconds"])
    fanouts   = cfg["model"]["fanouts"]
    graphs_dir = Path(cfg["graph"]["dir"])

    logger.info("Loading test graph …")
    g_test, _ = dgl.load_graphs(str(graphs_dir / "test.bin"))
    g_test = g_test[0]

    with open(graphs_dir / "node_id_map.json") as f:
        ip_to_id: dict[str, int] = json.load(f)
    id2ip: dict[int, str] = {v: k for k, v in ip_to_id.items()}

    # Label map to resolve class name → int for json path
    artifacts_dir = Path(cfg["output"]["artifacts_dir"])
    with open(artifacts_dir / "label_map.json") as f:
        label_map: dict[str, int] = json.load(f)

    sampler   = TemporalNeighborSampler(fanouts=fanouts)
    geid_to_local = {int(g): i for i, g in enumerate(g_test.edata[dgl.EID].numpy())}

    quality_rows: list[str] = []

    for class_name, global_eid, has_nbrs in CANDIDATES:
        logger.info(f"Processing {class_name} EID={global_eid} …")

        json_path = EXPL_DIR / class_name / f"{global_eid}.json"
        if not json_path.exists():
            logger.warning(f"  Missing {json_path} — skipping")
            continue

        with open(json_path) as f:
            plain = json.load(f)

        local_eid     = geid_to_local.get(global_eid)
        if local_eid is None:
            logger.warning(f"  EID {global_eid} not in test graph — skipping")
            continue

        target_ts_ms  = float(g_test.edata["timestamp"][local_eid].item())
        seed_t        = torch.tensor([local_eid], dtype=torch.long)
        input_nodes, _, blocks = sampler.sample_blocks(g_test, seed_t)

        src_t, dst_t  = g_test.find_edges(seed_t)
        src_nid       = int(src_t[0])
        dst_nid       = int(dst_t[0])

        topo = _build_topo(g_test, blocks, local_eid, target_ts_ms, W_seconds)
        logger.info(
            f"  hop1={len(topo['hop1_nodes'])} hop2={len(topo['hop2_nodes'])} "
            f"n_gaps={len(topo['all_neighbor_gaps_s'])} "
            f"in_W={int((topo['all_neighbor_gaps_s'] <= W_seconds).sum())}"
        )

        explanation = _adapt_explanation(plain, src_nid, dst_nid)

        fig = make_case_study_figure(
            explanation=explanation,
            topo=topo,
            id2ip=id2ip,
            class_name=class_name,
            W_seconds=W_seconds,
            top_feat=20,
            global_eid=global_eid,
        )

        stem = f"{class_name}_{global_eid}"
        fig.savefig(str(OUT_DIR / f"{stem}.pdf"), bbox_inches="tight", dpi=300)
        fig.savefig(str(OUT_DIR / f"{stem}.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"  Saved {stem}.pdf / {stem}.png")

        # Summary JSON
        summ = _summary(explanation, topo, W_seconds)
        summ_path = OUT_DIR / f"{stem}_summary.json"
        with open(summ_path, "w") as f:
            json.dump(summ, f, indent=2)
        logger.info(f"  Summary → {summ_path.name}")

        # Quality report row
        fg   = np.array(plain["feature_group_shap"])
        ns   = np.array(plain.get("node_shap", []))
        gaps = topo["all_neighbor_gaps_s"]
        in_W = int((gaps <= W_seconds).sum())
        proba = plain.get("predicted_proba", [0])[plain.get("predicted_label", 0)]
        quality_rows.append((
            class_name, global_eid,
            float(np.abs(fg).max()),
            float(np.abs(fg).max() - np.abs(fg).min()),
            len(topo["hop1_nodes"]) + len(topo["hop2_nodes"]),
            float(np.abs(ns).max()) if len(ns) > 0 else 0.0,
            len(gaps), in_W, proba, has_nbrs,
        ))

    # Write quality report
    header = (
        f"Case study quality report — SHAP-GSD (all 9 attack classes)\n"
        f"W_seconds={W_seconds}\n"
        + "-" * 110 + "\n"
        + f"{'Class':12s} {'EID':>8s}  {'top_phi':>7s}  {'phi_range':>9s}  "
          f"{'n_topo':>6s}  {'node_max':>8s}  {'n_gaps':>6s}  {'in_W':>4s}  "
          f"{'proba':>5s}  {'has_nbrs':>8s}\n"
        + "-" * 110 + "\n"
    )
    rows_txt = []
    for cls, eid, top_phi, phi_r, n_topo, node_max, n_gaps, in_W, proba, has_nbrs in quality_rows:
        rows_txt.append(
            f"{cls:12s} {eid:>8d}  {top_phi:>7.4f}  {phi_r:>9.4f}  "
            f"{n_topo:>6d}  {node_max:>8.4f}  {n_gaps:>6d}  {in_W:>4d}  "
            f"{proba:>5.3f}  {'YES' if has_nbrs else 'no':>8s}"
        )
    report_path = OUT_DIR / "quality_report.txt"
    with open(report_path, "w") as f:
        f.write(header)
        f.write("\n".join(rows_txt) + "\n")
    logger.info(f"Quality report → {report_path}")
    logger.info(f"Done. {len(quality_rows)}/9 case studies produced.")


if __name__ == "__main__":
    main()
