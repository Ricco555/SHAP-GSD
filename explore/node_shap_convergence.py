"""
Node-layer SHAP exactness audit.

WHAT THIS REPLACED, AND WHY
---------------------------
This script previously reported the Shapley *efficiency residual*
|Sigma phi - (f_logit - f_baseline)| as a function of ``nsamples`` and fitted
a log-log slope to it, claiming an O(1/sqrt(n)) convergence rate.  That metric
measured nothing: ``shap``'s solver eliminates one player and assigns it the
residual by construction (``_kernel.py:737-786``, the ``phi[nonzero_inds[-1]] =
(f(x) - f(null)) - sum(w)`` line), so efficiency holds exactly at every
``nsamples``, independently of sampling error.  The slope, the plateau and the
runtime reference line are all retired -- see review01
``coder_instructions_figure_determinism.md`` sections 6 and 7.

WHAT IS REPORTED INSTEAD
------------------------
The fraction of explained flows whose node-layer attributions are *exact
Shapley values* because ``shap``'s KernelExplainer enumerated the coalition
space exhaustively rather than sampling it, plus the player-count
distribution that determines it.  Every number below is computed from the run
being rendered; ``nsamples`` is read from the run's config.

VERIFIED LIBRARY-INTERNALS CLAIM (blocking prerequisite, section 7)
------------------------------------------------------------------
Installed library: ``shap==0.51.0`` at
``<venv>/lib/python3.12/site-packages/shap/explainers/_kernel.py``.

1. Exhaustive-enumeration condition -- ``_kernel.py:407-411``::

       self.max_samples = 2**30
       if self.M <= 30:
           self.max_samples = 2**self.M - 2
           if self.nsamples > self.max_samples:
               self.nsamples = self.max_samples

   ``nsamples`` is clamped to ``2**M - 2``, the size of the full non-trivial
   coalition space.  The subset-size enumeration loop at ``_kernel.py:434-471``
   then fills every subset size completely (its per-size budget test is at
   ``_kernel.py:451``) and the random-sampling block at ``_kernel.py:478-518``
   is skipped entirely because ``num_full_subsets == num_subset_sizes``.
   Replaying that loop numerically for M = 2..30 confirms the branch is
   exactly equivalent to the closed form ``2**M - 2 <= nsamples`` -- no
   off-by-one.  ``M <= 1`` is handled analytically before any sampling
   (``_kernel.py:385-395``) and is likewise exact.

2. M equals the node-game player count P -- ``_kernel.py:355-361`` sets
   ``self.M`` from ``varying_groups()``, i.e. only columns that differ between
   the instance and the background.  ``src/explainer/node_shap.py:243-244``
   passes ``background_data = np.zeros((1, coalition_size))`` against
   ``foreground_data = np.ones((1, coalition_size))``, so *every* column
   varies and ``M == coalition_size == P`` for every flow.  P is recoverable
   post hoc from the stored explanation as ``len(node_shap) + 2``
   (``node_shap.py:258-260``: column 0 = src novelty, column 1 = dst novelty,
   columns 2.. = non-target nodes).

3. Exhaustive enumeration implies the *exact* Shapley value only if the solver
   does no feature selection.  ``_kernel.py:699-733``: ``nonzero_inds =
   np.arange(self.M)`` and the LARS/Lasso truncation branch is guarded by
   ``if (self.l1_reg not in ["auto", False, 0]) or ...`` (``_kernel.py:703``).
   ``node_shap.py:37`` sets ``_L1_REG = False``, so that branch is not taken
   and the full-rank weighted least squares runs over all P players.

Usage:
  python explore/node_shap_convergence.py [--config ...]

Reads:
  outputs/explanations/<Class>/<EID>.json   (all flows)
  outputs/metrics/summary.json              (cross-referenced stability figure)
  config: explainer.node_nsamples

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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config

# Illustrative sensitivity levels. The CONFIGURED value is the operative one;
# these exist only to show how the exact fraction moves with the budget.
SENSITIVITY_NSAMPLES = [512, 1024, 2048]

# shap clamps nsamples to 2**M - 2 only for M <= 30 (_kernel.py:408).
_SHAP_MAX_ENUMERABLE_M = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def max_exhaustive_players(nsamples: int) -> int:
    """Largest player count P for which shap enumerates all coalitions.

    Mirrors ``shap/explainers/_kernel.py:407-411`` (see module docstring):
    the coalition space is enumerated in full iff ``2**P - 2 <= nsamples``
    (for ``P <= 30``), and ``P <= 1`` is exact analytically.

    Args:
        nsamples: KernelSHAP coalition budget passed to ``shap_values()``.

    Returns:
        The largest P that is computed exhaustively at this budget.
    """
    p = 1
    while p < _SHAP_MAX_ENUMERABLE_M and (2 ** (p + 1) - 2) <= nsamples:
        p += 1
    return p


def is_exhaustive(n_players: int, nsamples: int) -> bool:
    """True iff this flow's node game is solved by exhaustive enumeration.

    Args:
        n_players: node-game player count P (= ``len(node_shap) + 2``).
        nsamples:  KernelSHAP coalition budget.

    Returns:
        Whether shap enumerates rather than samples the coalition space.
    """
    if n_players <= 1:
        return True  # _kernel.py:385-395, analytic branch
    return n_players <= _SHAP_MAX_ENUMERABLE_M and (2 ** n_players - 2) <= nsamples


def collect_player_counts(
    expl_dir: Path,
) -> tuple[np.ndarray, dict[str, int], dict[int, int]]:
    """Read the node-game player count P for every explained flow.

    Args:
        expl_dir: ``outputs/explanations`` directory.

    Returns:
        (array of P per flow, {class name: flow count}, {edge_id: P}).

    Raises:
        SystemExit: if no explanation JSONs are found.
    """
    players: list[int] = []
    per_class: dict[str, int] = {}
    by_eid: dict[int, int] = {}
    for path in sorted(expl_dir.glob("*/*.json")):
        with open(path) as f:
            d = json.load(f)
        node_shap = d["node_shap"]
        node_ids = d["node_ids"]
        if len(node_shap) != len(node_ids):
            raise SystemExit(
                f"{path}: node_shap ({len(node_shap)}) / node_ids "
                f"({len(node_ids)}) length mismatch"
            )
        # +2 for the two target-endpoint novelty flags (node_shap.py:226).
        players.append(len(node_shap) + 2)
        by_eid[int(d["edge_id"])] = len(node_shap) + 2
        per_class[path.parent.name] = per_class.get(path.parent.name, 0) + 1

    if not players:
        raise SystemExit(f"No explanation JSONs under {expl_dir}")
    return np.asarray(players, dtype=int), per_class, by_eid


def crosscheck_player_counts(metrics_dir: Path, players: dict[int, int]) -> str:
    """Verify P = len(node_shap) + 2 against an independent runtime record.

    ``scripts/08_metrics.py``'s novelty-fidelity pass writes an ``n_players``
    column into ``fidelity_novelty.csv``, derived from the live coalition
    layout rather than reconstructed from stored phi. Agreement makes the
    player count observed rather than inferred.

    Args:
        metrics_dir: ``outputs/metrics`` directory.
        players:     {edge_id: P} reconstructed from the explanation JSONs.

    Returns:
        Human-readable status string for the report header.

    Raises:
        SystemExit: on any disagreement — the exactness counts would be wrong.
    """
    csv_path = metrics_dir / "fidelity_novelty.csv"
    if not csv_path.exists():
        return f"{csv_path.name}: NOT PRESENT (cross-check skipped)"

    import csv as _csv

    n_checked = 0
    mismatches: list[str] = []
    with open(csv_path, newline="") as f:
        for row in _csv.DictReader(f):
            if "n_players" not in row or "edge_id" not in row:
                return f"{csv_path.name}: no n_players column (cross-check skipped)"
            eid = int(row["edge_id"])
            if eid not in players:
                continue
            n_checked += 1
            if int(row["n_players"]) != players[eid]:
                mismatches.append(
                    f"EID {eid}: csv {row['n_players']} != json {players[eid]}"
                )
    if mismatches:
        raise SystemExit(
            f"Player-count cross-check FAILED on {len(mismatches)} of "
            f"{n_checked} flows: {mismatches[:5]}"
        )
    return f"{csv_path.name} n_players: {n_checked}/{n_checked} agree"


def read_stability_crossref(metrics_dir: Path) -> dict | None:
    """Existing seed-to-seed stability figure, with its own provenance.

    This is the FEATURE-GROUP layer's figure (``scripts/08_metrics.py``'s
    ``compute_stability`` re-runs ``FeatureGroupSHAP``); it is reported here
    as a labelled cross-reference for the sampled tail, not as a node-layer
    measurement. No node-layer seed-to-seed measurement exists in this run.

    Args:
        metrics_dir: ``outputs/metrics`` directory.

    Returns:
        dict with the value and its denominators, or None if unavailable.
    """
    summary_path = metrics_dir / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        summary = json.load(f)
    overall = summary.get("overall", {})
    if "stability" not in overall:
        return None
    return {
        "value":       float(overall["stability"]),
        "n_flows":     int(overall.get("n_stability", 0)),
        "layer":       "feature_group",
        "source":      str(summary_path.relative_to(metrics_dir.parent.parent)),
        "provenance": (
            "scripts/08_metrics.py compute_stability(); defaults "
            "--stability-seeds 3, --stability-nsamples 512, "
            "--stability-per-class 5"
        ),
    }


def main() -> None:
    _default_cfg = os.environ.get("SHAP_GSD_CONFIG", "configs/experiment_unsw.yaml")
    parser = argparse.ArgumentParser(
        description="Node-layer SHAP exactness audit (exhaustive enumeration share)"
    )
    parser.add_argument("--config", default=_default_cfg)
    args = parser.parse_args()

    cfg = load_config(args.config)
    outputs     = ROOT / cfg["output"]["outputs_dir"]
    EXPL_DIR    = outputs / "explanations"
    FIG_DIR     = outputs / "figures" / "explore"
    METRICS_DIR = outputs / "metrics"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    explainer_cfg = cfg.get("explainer", {})
    if "node_nsamples" not in explainer_cfg:
        raise SystemExit(
            "explainer.node_nsamples missing from the resolved config — "
            "refusing to assume a value."
        )
    node_nsamples = int(explainer_cfg["node_nsamples"])
    logger.info(f"Configured node_nsamples = {node_nsamples} (from {args.config})")

    P, per_class, P_by_eid = collect_player_counts(EXPL_DIR)
    n_flows = int(P.size)
    logger.info(f"Read {n_flows} explanation JSONs across {len(per_class)} classes")

    xcheck = crosscheck_player_counts(METRICS_DIR, P_by_eid)
    logger.info(f"Player-count cross-check: {xcheck}")

    p_max_exact = max_exhaustive_players(node_nsamples)
    exact_mask = np.array([is_exhaustive(int(p), node_nsamples) for p in P])
    n_exact = int(exact_mask.sum())
    n_sampled = n_flows - n_exact
    pct_exact = 100.0 * n_exact / n_flows

    dist = {
        "min":    int(P.min()),
        "median": float(np.median(P)),
        "mean":   round(float(P.mean()), 2),
        "max":    int(P.max()),
    }

    sensitivity = []
    for ns in sorted(set(SENSITIVITY_NSAMPLES) | {node_nsamples}):
        thr = max_exhaustive_players(ns)
        cnt = int(sum(1 for p in P if is_exhaustive(int(p), ns)))
        sensitivity.append({
            "nsamples":         ns,
            "max_exact_P":      thr,
            "n_exact":          cnt,
            "pct_exact":        round(100.0 * cnt / n_flows, 1),
            "is_configured":    ns == node_nsamples,
        })

    histogram = {int(p): int(c) for p, c in zip(*np.unique(P, return_counts=True))}
    stability = read_stability_crossref(METRICS_DIR)

    # ── JSON ──────────────────────────────────────────────────────────────
    output_data = {
        "metric": "node_layer_exhaustive_enumeration_share",
        "shap_version_verified": "0.51.0",
        "exhaustive_condition": "2**P - 2 <= nsamples  (P <= 30)",
        "exhaustive_condition_source":
            "shap/explainers/_kernel.py:407-411 (clamp), :434-471 (enumeration "
            "loop), :478 (sampling skipped), :699/:703 (l1_reg=False -> no "
            "feature selection), :355-361 (M == P given zeros/ones background)",
        "config_path":            str(args.config),
        "node_nsamples":          node_nsamples,
        "n_flows":                n_flows,
        "n_classes":              len(per_class),
        "flows_per_class":        per_class,
        "max_exact_players":      p_max_exact,
        "n_exact":                n_exact,
        "pct_exact":              round(pct_exact, 1),
        "n_sampled":              n_sampled,
        "pct_sampled":            round(100.0 - pct_exact, 1),
        "player_count_crosscheck":  xcheck,
        "player_count_distribution": dist,
        "player_count_histogram": histogram,
        "nsamples_sensitivity":   sensitivity,
        "sampled_tail_error_reference": stability,
    }
    json_path = METRICS_DIR / "node_shap_convergence.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Metrics → {json_path}")

    # ── Text report (explore/AGENT.md §3 block structure) ─────────────────
    lines = [
        "Figure reasoning — node_shap_convergence",
        "========================================",
        "",
        f"config={args.config}",
        f"configured explainer.node_nsamples = {node_nsamples}  "
        f"(read from the resolved config; this run directory carries no "
        f"separate effective-config dump)",
        f"n_flows={n_flows} (all explanation JSONs under "
        f"{EXPL_DIR.relative_to(outputs.parent)}), {len(per_class)} classes",
        f"player-count formula P = len(node_shap) + 2 cross-checked against "
        f"{xcheck}",
        "",
        "WHAT THE FIGURE SHOWS",
        "----------------------",
        "  Left panel: the distribution of node-game player counts P over every",
        "  explained flow, split at the largest P that shap's KernelExplainer",
        "  solves by exhaustive coalition enumeration at the configured budget.",
        "  Right panel: the share of flows receiving exact Shapley values as a",
        "  function of that budget.",
        "",
        "  shap's KernelExplainer clamps nsamples to 2**P - 2 and then",
        "  enumerates every coalition, so a flow whose node game has",
        f"  2**P - 2 <= {node_nsamples} receives the EXACT Shapley value, not a",
        "  sampled estimate. Verified in shap 0.51.0 at",
        "  shap/explainers/_kernel.py:407-411 and :434-471; l1_reg=False",
        "  (node_shap.py:37) keeps the solver at :703 from truncating players.",
        "",
        "KEY FINDINGS",
        "------------",
        f"  * exhaustive when P <= {p_max_exact}  (at nsamples={node_nsamples})",
        f"  * exact  (enumerated) {n_exact:>5d} of {n_flows}  ({pct_exact:.1f}%)",
        f"  * sampled (estimated) {n_sampled:>5d} of {n_flows}  "
        f"({100.0 - pct_exact:.1f}%)",
        f"  * node-game players P = len(node_shap) + 2 novelty flags:",
        f"    min {dist['min']}, median {dist['median']:.0f}, "
        f"mean {dist['mean']:.2f}, max {dist['max']}",
        f"  * sampled tail is exactly the P > {p_max_exact} flows: {n_sampled}",
        "",
        "  P : flows",
    ]
    for p in sorted(histogram):
        tag = "exact " if is_exhaustive(p, node_nsamples) else "sampled"
        lines.append(f"  {p:>2d} : {histogram[p]:>5d}   {tag}")

    lines += [
        "",
        "  Sensitivity to the coalition budget "
        "(illustrative; the configured value is the operative one)",
        f"  {'nsamples':>9s}  {'exhaustive when':>16s}  {'n exact':>8s}  "
        f"{'pct':>6s}",
    ]
    for s in sensitivity:
        mark = "  <-- configured" if s["is_configured"] else ""
        lines.append(
            f"  {s['nsamples']:>9d}  {'P <= ' + str(s['max_exact_P']):>16s}  "
            f"{s['n_exact']:>8d}  {s['pct_exact']:>5.1f}%{mark}"
        )

    lines += [
        "",
        "  Sampling error for the sampled tail",
    ]
    if stability is None:
        lines.append(
            "  UNAVAILABLE — outputs/metrics/summary.json carries no stability "
            "figure for this run."
        )
    else:
        lines += [
            f"  Cross-reference (NOT a node-layer measurement): seed-to-seed "
            f"stability = {stability['value']:.6f}",
            f"    layer: {stability['layer']}   n_flows: {stability['n_flows']}"
            f"   source: {stability['source']}",
            f"    provenance: {stability['provenance']}",
            "  No node-layer seed-to-seed measurement exists in this run. The",
            f"  {n_sampled} sampled flows have no layer-specific sampling-error",
            "  figure; the feature-group value above is the only seed-to-seed",
            "  measurement the pipeline produces and is quoted with its own",
            "  denominators rather than relabelled as a node-layer statistic.",
        ]

    lines += [
        "",
        "PAPER FRAMING",
        "-------------",
        f"  The node-novelty coalition game has a small player set (P = "
        f"{dist['min']}-{dist['max']}, median {dist['median']:.0f} over "
        f"{n_flows} explained",
        f"  flows), and KernelSHAP enumerates the coalition space exhaustively "
        f"whenever",
        f"  2^P - 2 does not exceed the sampling budget. At the configured "
        f"budget of",
        f"  nsamples = {node_nsamples} this holds for P <= {p_max_exact}, so "
        f"{n_exact} of {n_flows} flows ({pct_exact:.1f}%) receive",
        f"  exact Shapley values at the node layer rather than Monte Carlo "
        f"estimates.",
        f"  The remaining {n_sampled} flows ({100.0 - pct_exact:.1f}%) are "
        f"sampled.",
        "",
        "SUGGESTED FIGURE CAPTION",
        "-------------------------",
        f"  Node-layer attribution exactness. Left: distribution of "
        f"node-novelty coalition",
        f"  player counts P = |V_sub \\ endpoints| + 2 novelty flags over the "
        f"{n_flows} explained",
        f"  flows (min {dist['min']}, median {dist['median']:.0f}, "
        f"mean {dist['mean']:.2f}, max {dist['max']}). KernelSHAP enumerates "
        f"the full",
        f"  coalition space when 2^P - 2 <= nsamples, which at the configured "
        f"nsamples = {node_nsamples}",
        f"  means P <= {p_max_exact} (green bars); those "
        f"{n_exact} flows ({pct_exact:.1f}%) receive exact Shapley values, "
        f"while the",
        f"  remaining {n_sampled} ({100.0 - pct_exact:.1f}%) are sampled. "
        f"Right: exact share as a function of the",
        f"  coalition budget, with the configured value circled.",
    ]

    txt_path = FIG_DIR / "node_shap_convergence_details.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Text report → {txt_path}")

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    ps = np.array(sorted(histogram))
    counts = np.array([histogram[int(p)] for p in ps])
    colors = [
        "seagreen" if is_exhaustive(int(p), node_nsamples) else "lightsteelblue"
        for p in ps
    ]
    ax.bar(ps, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(p_max_exact + 0.5, color="tomato", ls="--", lw=1.2)
    ax.text(
        p_max_exact + 0.7, counts.max() * 0.92,
        f"exhaustive: P $\\leq$ {p_max_exact}\n(nsamples = {node_nsamples})",
        fontsize=8, color="tomato", va="top",
    )
    ax.set_xlabel("Node-game players $P$")
    ax.set_ylabel("Flows")
    ax.set_title(
        f"Node-layer player counts (n = {n_flows} flows)\n"
        f"{n_exact} exact ({pct_exact:.1f}%), {n_sampled} sampled"
    )
    ax.grid(True, axis="y", alpha=0.3)

    ax2 = axes[1]
    ns_arr = np.array([s["nsamples"] for s in sensitivity], dtype=float)
    pct_arr = np.array([s["pct_exact"] for s in sensitivity], dtype=float)
    ax2.semilogx(ns_arr, pct_arr, "o-", color="seagreen", lw=1.5, base=2)
    for s in sensitivity:
        if s["is_configured"]:
            ax2.scatter(
                [s["nsamples"]], [s["pct_exact"]], s=120,
                facecolors="none", edgecolors="tomato", lw=1.6, zorder=5,
                label=f"configured ({s['nsamples']})",
            )
    ax2.set_xticks(ns_arr)
    ax2.set_xticklabels([f"{int(n)}" for n in ns_arr])
    ax2.set_xlabel("KernelSHAP coalition budget (nsamples)")
    ax2.set_ylabel("Flows with exact Shapley values (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("Exhaustively enumerated share vs budget")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"node_shap_convergence.{ext}"
        dpi = 300 if ext == "pdf" else 150
        fig.savefig(str(out), bbox_inches="tight", dpi=dpi)
        logger.info(f"Figure → {out}")
    plt.close(fig)

    logger.info(
        f"Done. {n_exact}/{n_flows} ({pct_exact:.1f}%) node games solved by "
        f"exhaustive enumeration at nsamples={node_nsamples} (P <= {p_max_exact})."
    )


if __name__ == "__main__":
    main()
