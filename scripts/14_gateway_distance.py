"""
Gateway-distance diagnostic — scripts/14_gateway_distance.py

Stage-0, read-only, no-model, no-GPU topology measurement of the
malicious-only training subgraph. Computes per-node gateway distance
(``d_gw``) and derives a recommended neighbor-sampling depth ``k*``.

This is a diagnostic ONLY: it never feeds its result back into
``model.num_layers``/sampler config automatically (that would be Stage 2,
not authorized by this phase). See ``src/data/gateway_distance.py``'s module
docstring for the full d_gw definitional resolution.

USAGE
-----
  python scripts/14_gateway_distance.py --config configs/experiment_unsw.yaml

Disabled via config (``topology.gateway_distance.enabled: false``) or via
``run_dataset.py --skip-gateway-distance``: the script logs and exits 0
without writing any output files.

OUTPUTS
-------
  <outputs_dir>/topology/gateway_distance.json   machine-readable results
  <outputs_dir>/topology/gateway_distance.txt    human-readable report
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.gateway_distance import run_gateway_distance_diagnostic
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _write_report(result: dict[str, Any], out_txt: Path) -> None:
    """Write a human-readable mirror of the diagnostic result.

    Follows ``scripts/12_novelty_audit.py``'s ``_write_report`` section-header
    style (presentation formatting kept in the script, not in ``src/``).

    Args:
        result: the diagnostic result dict (see ``run_gateway_distance_diagnostic``).
        out_txt: destination path for the report.
    """
    lines = [
        "Gateway-Distance Diagnostic — SHAP-GSD",
        "=" * 60,
        "",
        f"Generated: {result.get('generated_at', 'N/A')}",
        "",
    ]

    ra = result.get("role_assignment", {})
    lines += [
        "── Role assignment ──",
        f"  mode:          {ra.get('mode', 'N/A')}",
        f"  n_nodes_total: {ra.get('n_nodes_total', 'N/A')}",
        f"  n_internal:    {ra.get('n_internal', 'N/A')}",
        f"  n_external:    {ra.get('n_external', 'N/A')}",
        "",
    ]

    gb = result.get("gateway_boundary", {})
    lines += [
        "── Gateway boundary (reporting only) ──",
        f"  n_gateway_nodes:      {gb.get('n_gateway_nodes', 'N/A')}",
        f"  is_single_chokepoint: {gb.get('is_single_chokepoint', 'N/A')}",
        "",
    ]

    dg = result.get("d_gw", {})
    g_stats = dg.get("global", {})
    lines += [
        "── Gateway distance (global) ──",
        f"  mean: {g_stats.get('mean', 'N/A')}  max: {g_stats.get('max', 'N/A')}  "
        f"std: {g_stats.get('std', 'N/A')}",
        f"  n_measured: {g_stats.get('n_measured', 'N/A')}  "
        f"n_unreachable: {g_stats.get('n_unreachable', 'N/A')}",
        "",
        "  Per-class:",
    ]
    for cls, v in sorted(dg.get("per_class", {}).items()):
        lines.append(
            f"    {cls:15s}: mean={v.get('mean')}  max={v.get('max')}  "
            f"n_dst={v.get('n_dst_nodes')}  n_src={v.get('n_src_nodes')}  "
            f"unreachable={v.get('n_unreachable')}"
        )
    lines.append("")

    diam = result.get("diameter", {})
    g_diam = diam.get("global", diam)
    # Annotate estimated (double-sweep lower-bound) diameters per scope so a
    # downstream reader never cites an estimate as exact (spec 10 §2.1/§9 Q3).
    # ``.get(..., True)`` treats a pre-annotation JSON (no flag) as exact.
    g_lb = "  (diameter is a lower bound — one or more components estimated)" \
        if g_diam.get("diameter_exact", True) is False else ""
    lines += [
        "── Diameter (undirected, malicious subgraph) ──",
        f"  global: diameter={g_diam.get('diameter', 'N/A')}  "
        f"n_components={g_diam.get('n_components', 'N/A')}{g_lb}",
    ]
    for cls, v in sorted(diam.get("per_class", {}).items()):
        lb = " (lower bound)" if v.get("diameter_exact", True) is False else ""
        lines.append(
            f"    {cls:15s}: diameter={v.get('diameter')}  "
            f"n_components={v.get('n_components')}{lb}"
        )
    lines.append("")

    pv = result.get("pivot_nodes", {})
    lines += [
        "── Pivot nodes ──",
        f"  global_count: {pv.get('global_count', 'N/A')}",
    ]
    for cls, cnt in sorted(pv.get("per_class_count", {}).items()):
        lines.append(f"    {cls:15s}: {cnt}")
    lines.append("")

    lines += [
        "── k* derivation ──",
        f"  k_star: {result.get('k_star', 'N/A')}  (k_max={result.get('k_max', 'N/A')})",
        "",
    ]

    dc = result.get("drift_check", {})
    lines.append("── Train-half vs train-half drift check ──")
    if dc.get("enabled"):
        lines += [
            f"  k_star_first_half:  {dc.get('k_star_first_half')}",
            f"  k_star_second_half: {dc.get('k_star_second_half')}",
            f"  delta_k_star:       {dc.get('delta_k_star')}",
            f"  mean_d_gw_first_half:  {dc.get('mean_d_gw_first_half')}",
            f"  mean_d_gw_second_half: {dc.get('mean_d_gw_second_half')}",
            f"  delta_mean_d_gw:       {dc.get('delta_mean_d_gw')}",
        ]
    else:
        lines.append("  (disabled via config)")
    lines.append("")

    pc = result.get("posthoc_correlation", {})
    lines += [
        "── Post-hoc correlation: mean d_gw vs per-class test F1 ──",
        f"  available:     {pc.get('available', 'N/A')}",
        f"  spearman_rho:  {pc.get('spearman_rho', 'N/A')}",
        f"  p_value:       {pc.get('p_value', 'N/A')}",
    ]
    if pc.get("note"):
        lines.append(f"  note: {pc['note']}")

    out_txt.write_text("\n".join(lines) + "\n")
    logger.info("Report -> %s", out_txt)


def main() -> None:
    """CLI entry point: run the gateway-distance diagnostic for one experiment config."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiment_unsw.yaml"),
        help="Path to experiment config YAML. Default: configs/experiment_unsw.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    topology_cfg = cfg.get("topology", {}).get("gateway_distance", {})
    if not topology_cfg.get("enabled", True):
        logger.info("gateway-distance disabled via config; skipping")
        sys.exit(0)

    graph_dir = Path(cfg["graph"]["dir"])
    artifacts_dir = Path(cfg["output"]["artifacts_dir"])
    outputs_dir = Path(cfg["output"]["outputs_dir"])
    metrics_json_path = artifacts_dir / "evaluation" / "metrics.json"

    result = run_gateway_distance_diagnostic(
        graph_dir=graph_dir,
        artifacts_dir=artifacts_dir,
        topology_cfg=topology_cfg,
        metrics_json_path=metrics_json_path if metrics_json_path.exists() else None,
    )
    result["generated_at"] = datetime.now(timezone.utc).isoformat()

    topology_out_dir = outputs_dir / "topology"
    topology_out_dir.mkdir(parents=True, exist_ok=True)

    json_path = topology_out_dir / "gateway_distance.json"
    json_path.write_text(json.dumps(result, indent=2))
    logger.info("JSON -> %s", json_path)

    _write_report(result, topology_out_dir / "gateway_distance.txt")

    logger.info("k* = %s (k_max=%s)", result["k_star"], result["k_max"])


if __name__ == "__main__":
    main()
