"""
Node novelty audit — scripts/12_novelty_audit.py

Answers three questions about node novelty suppression in SHAP-GSD:

  1. How many unique nodes does the dataset have, and what is their IP type
     breakdown (RFC1918, public, loopback, multicast)?

  2. Across all 1,764 explanation JSONs, how many flows have non-zero
     src_novelty_shap or dst_novelty_shap?  (Fast pass — no artifacts needed.)

  3. What fraction of node observations in the test split have dim-0
     (is_internal) = 1 and dim-1 (novelty = first-seen-in-W) = 1?
     (Full pass — requires NSM + test graph, run on SRCE.)

BACKGROUND
----------
Node SHAP masking toggles _NOVELTY_DIM = 1 (novelty: first seen in window W).
If dim-1 = 0 for a node at the time of the target flow, masking it out has no
effect and src/dst_novelty_shap = 0.  This script measures how often that
suppression actually occurs in practice.

An earlier note in CLAUDE.md claimed "is_internal returns 0 for all nodes"
(because UNSW-NB15 uses public IPs).  This was incorrect: the built graph
contains 8 RFC1918/loopback nodes (10.40.x.x, 192.168.x.x, 127.0.0.1) among
its 44 unique endpoints.  This script verifies the corrected picture.

USAGE
-----
  # Fast pass (JSON scan only, no artifacts):
  python scripts/12_novelty_audit.py --config configs/experiment_unsw.yaml

  # Full pass (adds NSM + test-graph node-state sampling, run on SRCE):
  python scripts/12_novelty_audit.py --config configs/experiment_unsw.yaml --full

  # Adjust sample size for full pass:
  python scripts/12_novelty_audit.py --config configs/experiment_unsw.yaml --full --n-sample 1000

OUTPUTS
-------
  outputs/metrics/novelty_audit.json   machine-readable results
  outputs/metrics/novelty_audit.txt    human-readable report
"""

import argparse
import ipaddress
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── IP classification ─────────────────────────────────────────────────────────

def _ip_type(ip_str: str) -> str:
    """Classify an IP string into one of: rfc1918, loopback, multicast, public."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return "unknown"
    if addr.is_private:
        return "rfc1918"
    if addr.is_loopback:
        return "loopback"
    if addr.is_multicast:
        return "multicast"
    return "public"


# ── Pass 1: node_id_map ───────────────────────────────────────────────────────

def audit_node_map(graphs_dir: Path) -> dict:
    """Count unique nodes and classify by IP type."""
    path = graphs_dir / "node_id_map.json"
    if not path.exists():
        logger.warning(f"node_id_map.json not found at {path}")
        return {}

    ip_to_id: dict[str, int] = json.loads(path.read_text())
    type_counts: dict[str, list[str]] = {}
    for ip in ip_to_id:
        t = _ip_type(ip)
        type_counts.setdefault(t, []).append(ip)

    result = {
        "n_unique_nodes": len(ip_to_id),
        "by_type": {t: {"count": len(ips), "ips": sorted(ips)}
                    for t, ips in sorted(type_counts.items())},
    }
    logger.info(
        f"Unique nodes: {result['n_unique_nodes']}  "
        + "  ".join(f"{t}={v['count']}" for t, v in result["by_type"].items())
    )
    return result


# ── Pass 2: explanation JSON scan ─────────────────────────────────────────────

def audit_explanation_jsons(expl_dir: Path) -> dict:
    """Scan all explanation JSONs for non-zero src/dst novelty shap values."""
    total = 0
    n_nonzero_src = 0
    n_nonzero_dst = 0
    n_nonzero_either = 0
    by_class: dict[str, dict] = {}

    for cls_dir in sorted(expl_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        cls_total = cls_nonzero = 0

        for p in cls_dir.glob("*.json"):
            if not p.stem.lstrip("-").isdigit():
                continue
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue

            total += 1
            cls_total += 1

            src_v = abs(d.get("src_novelty_shap", 0.0))
            dst_v = abs(d.get("dst_novelty_shap", 0.0))
            if src_v > 1e-6:
                n_nonzero_src += 1
            if dst_v > 1e-6:
                n_nonzero_dst += 1
            if src_v > 1e-6 or dst_v > 1e-6:
                n_nonzero_either += 1
                cls_nonzero += 1

        if cls_total > 0:
            by_class[cls_name] = {"n_total": cls_total, "n_nonzero": cls_nonzero}

    logger.info(
        f"JSON scan: {n_nonzero_either}/{total} flows have non-zero novelty shap "
        f"(src: {n_nonzero_src}, dst: {n_nonzero_dst})"
    )
    return {
        "n_total":            total,
        "n_nonzero_src":      n_nonzero_src,
        "n_nonzero_dst":      n_nonzero_dst,
        "n_nonzero_either":   n_nonzero_either,
        "frac_nonzero_either": round(n_nonzero_either / total, 6) if total else 0,
        "by_class":           by_class,
    }


# ── Pass 3: NSM node-state dim sampling ──────────────────────────────────────

def audit_node_states(cfg: dict, n_sample: int, seed: int) -> dict:
    """
    Sample test flows and measure dim-0 (is_internal) and dim-1 (novelty)
    distributions.  Requires NSM + test graph (run on SRCE).
    """
    import torch
    import dgl
    from src.model.node_state import NodeStateManager

    graphs_dir  = Path(cfg["graph"]["dir"])
    nsm_dir     = Path(cfg["graph"]["node_state_dir"])

    logger.info("Loading test graph …")
    g_test, _ = dgl.load_graphs(str(graphs_dir / "test.bin"))
    g_test = g_test[0]

    logger.info("Loading NodeStateManager …")
    nsm = NodeStateManager.load(
        nsm_dir,
        expected_novelty_mode=cfg["model"].get("novelty_mode", "recent_window"),
    )

    n_edges = g_test.num_edges()
    rng = np.random.default_rng(seed)
    sampled_leids = rng.choice(n_edges, size=min(n_sample, n_edges), replace=False)

    dim0_src, dim0_dst = [], []
    dim1_src, dim1_dst = [], []

    for leid in sampled_leids:
        ts = float(g_test.edata["timestamp"][int(leid)].item())
        src_t, dst_t = g_test.find_edges(torch.tensor([int(leid)]))
        src_nid = int(src_t[0])
        dst_nid = int(dst_t[0])

        s_state = nsm.get_state_at_time(src_nid, ts)
        d_state = nsm.get_state_at_time(dst_nid, ts)

        dim0_src.append(float(s_state[0]))
        dim0_dst.append(float(d_state[0]))
        dim1_src.append(float(s_state[1]))
        dim1_dst.append(float(d_state[1]))

    def _stats(arr: list) -> dict:
        a = np.array(arr)
        return {
            "mean":           round(float(a.mean()), 6),
            "max":            round(float(a.max()), 6),
            "n_nonzero":      int((a > 0).sum()),
            "frac_nonzero":   round(float((a > 0).mean()), 6),
        }

    result = {
        "n_flows_sampled":  int(len(sampled_leids)),
        "is_internal_dim0": {"src": _stats(dim0_src), "dst": _stats(dim0_dst)},
        "novelty_dim1":     {"src": _stats(dim1_src), "dst": _stats(dim1_dst)},
    }
    logger.info(
        f"dim-0 is_internal: src frac={result['is_internal_dim0']['src']['frac_nonzero']:.4f}  "
        f"dst frac={result['is_internal_dim0']['dst']['frac_nonzero']:.4f}"
    )
    logger.info(
        f"dim-1 novelty:     src frac={result['novelty_dim1']['src']['frac_nonzero']:.4f}  "
        f"dst frac={result['novelty_dim1']['dst']['frac_nonzero']:.4f}"
    )
    return result


# ── Report writer ─────────────────────────────────────────────────────────────

def _write_report(
    node_map: dict,
    json_audit: dict,
    state_audit: dict | None,
    out_txt: Path,
) -> None:
    lines = [
        "Node Novelty Audit — SHAP-GSD / NF-UNSW-NB15-v3",
        "=" * 60,
        "",
        f"Unique nodes: {node_map.get('n_unique_nodes', 'N/A')}",
    ]

    if "by_type" in node_map:
        for t, v in node_map["by_type"].items():
            lines.append(f"  {t:12s}: {v['count']:3d}  {v['ips']}")
    lines.append("")

    jt = json_audit.get("n_total", 0)
    lines += [
        "── Explanation JSON scan (src/dst_novelty_shap) ──",
        f"  Total explained flows:      {jt}",
        f"  Non-zero src_novelty_shap:  {json_audit.get('n_nonzero_src', 0)} "
        f"({json_audit.get('n_nonzero_src',0)/jt*100:.1f}%)" if jt else "",
        f"  Non-zero dst_novelty_shap:  {json_audit.get('n_nonzero_dst', 0)} "
        f"({json_audit.get('n_nonzero_dst',0)/jt*100:.1f}%)" if jt else "",
        f"  Non-zero either:            {json_audit.get('n_nonzero_either', 0)} "
        f"({json_audit.get('frac_nonzero_either',0)*100:.2f}%)",
        "",
        "  Per-class breakdown:",
    ]
    for cls, v in sorted(json_audit.get("by_class", {}).items()):
        pct = v["n_nonzero"] / v["n_total"] * 100 if v["n_total"] else 0
        lines.append(f"    {cls:12s}: {v['n_nonzero']:3d}/{v['n_total']:3d}  ({pct:.1f}%)")
    lines.append("")

    if state_audit:
        ns = state_audit.get("n_flows_sampled", 0)
        d0 = state_audit.get("is_internal_dim0", {})
        d1 = state_audit.get("novelty_dim1", {})
        lines += [
            "── Node state dim audit (NSM sample) ──",
            f"  Flows sampled: {ns}",
            "",
            "  dim-0 is_internal:",
            f"    src  frac=1: {d0.get('src',{}).get('frac_nonzero',0):.4f}  "
            f"n={d0.get('src',{}).get('n_nonzero',0)}/{ns}",
            f"    dst  frac=1: {d0.get('dst',{}).get('frac_nonzero',0):.4f}  "
            f"n={d0.get('dst',{}).get('n_nonzero',0)}/{ns}",
            "",
            "  dim-1 novelty (first-seen-in-W):",
            f"    src  frac=1: {d1.get('src',{}).get('frac_nonzero',0):.4f}  "
            f"n={d1.get('src',{}).get('n_nonzero',0)}/{ns}",
            f"    dst  frac=1: {d1.get('dst',{}).get('frac_nonzero',0):.4f}  "
            f"n={d1.get('dst',{}).get('n_nonzero',0)}/{ns}",
        ]
    else:
        lines += [
            "── Node state dim audit ──",
            "  (skipped — run with --full on SRCE to get dim-0/dim-1 distributions)",
        ]

    out_txt.write_text("\n".join(lines) + "\n")
    logger.info(f"Report → {out_txt}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config",   default=str(ROOT / "configs" / "experiment_unsw.yaml"))
    parser.add_argument("--full",     action="store_true",
                        help="Also load NSM + test graph to sample actual dim values "
                             "(requires SRCE artifacts)")
    parser.add_argument("--n-sample", type=int, default=500,
                        help="Number of test flows to sample for dim audit (default 500)")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    graphs_dir  = Path(cfg["graph"]["dir"])
    outputs_dir = Path(cfg["output"]["outputs_dir"])
    expl_dir    = outputs_dir / "explanations"
    metrics_dir = outputs_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1
    node_map = audit_node_map(graphs_dir)

    # Pass 2
    logger.info("Scanning explanation JSONs …")
    json_audit = audit_explanation_jsons(expl_dir)

    # Pass 3 (optional)
    state_audit: dict | None = None
    if args.full:
        logger.info(f"Running full node-state dim audit (n_sample={args.n_sample}) …")
        try:
            state_audit = audit_node_states(cfg, args.n_sample, args.seed)
        except Exception:
            logger.exception("Full audit failed — NSM not available? Run on SRCE.")

    # Write outputs
    output = {
        "node_map_audit":        node_map,
        "explanation_json_audit": json_audit,
        "node_state_audit":       state_audit,
    }
    json_path = metrics_dir / "novelty_audit.json"
    json_path.write_text(json.dumps(output, indent=2))
    logger.info(f"JSON → {json_path}")

    _write_report(node_map, json_audit, state_audit, metrics_dir / "novelty_audit.txt")


if __name__ == "__main__":
    main()
