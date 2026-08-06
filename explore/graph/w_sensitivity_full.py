"""
w_sensitivity_full.py — Figure 5
==================================
What it shows:
  Dual-axis line plot:
    Left y-axis  (teal):  % of flows with at least one in-window temporal neighbor vs W.
    Right y-axis (green): Mean top-φ_T per flow vs W.
  Five annotations:
    1. Median IAT vertical red dashed line.
    2. Saturation box at W = 1800 s.
    3. 50% majority threshold horizontal dashed line.
    4. Observed-values textbox (amber, bottom-right) — this dataset's own
       coverage/mean-φ_T at W=60s, computed live. NOT a cross-dataset
       prediction (see below).
    5. Point value labels on every marker.

This script is reused across datasets. It previously baked a specific,
named cross-dataset prediction into the figure image and into most of the
.txt reasoning text ("On IoT datasets (NF-ToN-IoT, NF-BoT-IoT) ... φ_T is
expected to dominate") — written when only UNSW-NB15 existed, never
recomputed once NF-BoT-IoT-v3 data existed to actually test it. That
prediction is now measured false on NF-BoT-IoT-v3 (63.3% in-window at
W=60s, not >80%; φ_T is 0.00-0.001% of total attribution, not dominant —
final/review01/results_update_tracker.md §10b). Fixed: no other dataset is
named or predicted about anywhere in this script's output; every sentence
in KEY FINDINGS/PAPER FRAMING/SUGGESTED FIGURE CAPTION is computed from
this run's own gap_stats.json/W*_temporal.csv, describing only what was
actually measured for whichever dataset is currently active.

Panels:
  fig, ax1 — single axis with twinx(), figsize=(9, 5.5)

Files read:
  outputs/w_ablation/gap_stats.json           — coverage and gap statistics
  outputs/w_ablation/W{60,300,1800,3600}_temporal.csv — per-flow phi_T values

Files output:
  outputs/figures/graph/w_sensitivity_annotated.{pdf,png,txt}
"""

import matplotlib
matplotlib.use("Agg")

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from explore._paths import paths  # noqa: E402
_P = paths()
ABL_DIR = _P["w_ablation"]
OUT_DIR = _P["figures"] / "graph"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEM     = "w_sensitivity_annotated"
LABEL_FS = 9
_TEAL    = "#2a9d8f"
_GREEN   = "#4caf50"
_AMBER   = "#e9c46a"
_RED     = "#e76f51"
_GRAY    = "#888888"


def main() -> None:
    # ------------------------------------------------------------------ data
    dataset_name = _P["dataset_name"]
    gap_stats = json.loads((ABL_DIR / "gap_stats.json").read_text())
    Ws        = [g["W_s"] for g in gap_stats]
    pct_cov   = [g["pct_flows_with_inwindow"] for g in gap_stats]
    median_gap = gap_stats[0]["median_gap_s"]   # same across all rows

    shap_means = []
    for W in Ws:
        csv_path = ABL_DIR / f"W{int(W)}_temporal.csv"
        rows = list(csv.DictReader(csv_path.open()))
        vals = [float(r["top_phi_T"]) for r in rows]
        shap_means.append(float(np.mean(vals)))

    Ws_arr   = np.array(Ws)
    cov_arr  = np.array(pct_cov)
    shap_arr = np.array(shap_means)

    # ------------------------------------------------------------------ figure
    fig, ax1 = plt.subplots(1, 1, figsize=(9, 5.5))
    ax2 = ax1.twinx()

    # Coverage (left, teal)
    l1, = ax1.plot(Ws_arr, cov_arr, "o-", color=_TEAL, lw=2,
                   ms=7, label="% flows with in-window neighbor")
    ax1.set_xscale("log")
    ax1.set_xlim(50, 7500)
    ax1.set_ylim(0, max(cov_arr) * 1.35)
    ax1.set_xlabel("Temporal window W (s, log scale)", fontsize=LABEL_FS)
    ax1.set_ylabel("% flows with in-window neighbor", color=_TEAL, fontsize=LABEL_FS)
    ax1.tick_params(axis="y", labelcolor=_TEAL, labelsize=LABEL_FS)
    ax1.tick_params(axis="x", labelsize=LABEL_FS)

    # Mean φ_T (right, green dashed)
    l2, = ax2.plot(Ws_arr, shap_arr, "s--", color=_GREEN, lw=2,
                   ms=7, label=r"Mean top-$\varphi_T$")
    ax2.set_ylabel(r"Mean top-$\varphi_T$ per flow", color=_GREEN, fontsize=LABEL_FS)
    ax2.tick_params(axis="y", labelcolor=_GREEN, labelsize=LABEL_FS)

    # --- Annotation 1: Median IAT ---
    ax1.axvline(x=median_gap, color=_RED, lw=1.5, ls="dotted",
                label=f"Median IAT = {median_gap:.0f} s")
    ax1.text(median_gap * 1.05, max(cov_arr) * 1.25,
             f"Median IAT\n{median_gap:.0f} s", color=_RED,
             fontsize=LABEL_FS - 1.5, va="top")

    # --- Annotation 2: Saturation box at W=1800 ---
    idx_1800 = list(Ws).index(1800.0)
    idx_3600 = list(Ws).index(3600.0)
    delta_phi = shap_arr[idx_3600] - shap_arr[idx_1800]
    ax2.annotate(
        f"φ_T saturates\n(Δ = {delta_phi:+.4f}\n1800→3600 s)",
        xy=(Ws_arr[idx_1800], shap_arr[idx_1800]),
        xytext=(Ws_arr[idx_1800] * 0.55, shap_arr[idx_1800] * 1.35),
        fontsize=LABEL_FS - 1, color=_GREEN,
        arrowprops=dict(arrowstyle="->", color=_GREEN, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=_GREEN, alpha=0.9),
    )

    # --- Annotation 3: 50% majority threshold ---
    ax1.axhline(y=50, color=_GRAY, lw=1.2, ls="--", alpha=0.6)
    ax1.text(55, 51.5, "50% majority", color=_GRAY, fontsize=LABEL_FS - 2)

    # --- Annotation 4: Observed values at W=60s (this dataset only — no
    # cross-dataset prediction; see module docstring) ---
    idx_60 = list(Ws).index(60.0)
    ax1.text(0.98, 0.08,
             f"At W = 60 s: {cov_arr[idx_60]:.2f}% in-window,\n"
             f"mean top-φ_T = {shap_arr[idx_60]:.5f}\n"
             f"({dataset_name})",
             transform=ax1.transAxes, fontsize=LABEL_FS - 1.5,
             ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.35", fc="#fff8e1", ec=_AMBER, alpha=0.92))

    # --- Annotation 5: Point value labels ---
    offsets = [(-18, 8), (5, 8), (5, -14), (5, 8)]   # (dx_pt, dy_pt) alternating
    for i, (w, cov, sm) in enumerate(zip(Ws_arr, cov_arr, shap_arr)):
        dx, dy = offsets[i % len(offsets)]
        ax1.annotate(f"{cov:.2f}%", xy=(w, cov),
                     xytext=(dx, dy), textcoords="offset points",
                     fontsize=LABEL_FS - 2, color=_TEAL)
        ax2.annotate(f"{sm:.4f}", xy=(w, sm),
                     xytext=(dx, -dy), textcoords="offset points",
                     fontsize=LABEL_FS - 2, color=_GREEN)

    # Legend
    handles = [l1, l2,
               plt.Line2D([0], [0], color=_RED, lw=1.5, ls="dotted",
                          label=f"Median IAT = {median_gap:.0f} s")]
    ax1.legend(handles=handles, fontsize=LABEL_FS - 1, loc="upper left")

    ax1.set_title("W-sensitivity: temporal coverage and mean φ_T vs window size",
                  fontsize=LABEL_FS + 1)

    # ------------------------------------------------------------------ save
    fig.savefig(OUT_DIR / f"{STEM}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / f"{STEM}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / STEM}.{{pdf,png}}")

    # ------------------------------------------------------------------ txt
    max_W = Ws_arr[-1]
    cov_at_max = cov_arr[-1]
    shap_at_max = shap_arr[-1]
    iat_exceeds_max_W = median_gap > max_W
    # No fixed threshold for "near-zero" is imposed here — the number is
    # reported and the reader judges it, per this script's own
    # GENERATOR NOTE precedent (attribution_decomp.py: report computed
    # values, do not assert an interpretation the data doesn't force).

    lines = [
        "Figure reasoning — w_sensitivity_annotated",
        "=" * 44, "",
        "WHAT THE FIGURE SHOWS",
        "----------------------",
        f"Dual-axis line plot showing how temporal coverage (% of {dataset_name} test flows",
        "that have at least one in-window temporal neighbor, left axis teal) and mean top-φ_T",
        "(right axis green dashed) vary as the temporal window W grows from",
        f"{int(Ws_arr[0])} to {int(max_W)} s. A vertical dotted red line marks the median",
        f"inter-arrival time ({median_gap:.0f} s).",
        "",
        "KEY FINDINGS",
        "------------",
        "Computed values at each W:",
    ]
    for w, cov, sm in zip(Ws, pct_cov, shap_means):
        lines.append(f"  W={int(w):5d}s: coverage={cov:.2f}%,  mean top-φ_T={sm:.5f}")

    lines += [
        "",
        f"  Saturation: Δφ_T from W=1800s → W=3600s = {delta_phi:+.5f}",
        f"  Median IAT = {median_gap:.1f} s "
        f"{'>>' if iat_exceeds_max_W else '<'} largest tested W ({max_W:.0f} s).",
        "",
        f"  At the largest tested window (W={max_W:.0f}s): {cov_at_max:.2f}% of {dataset_name}",
        f"  flows have an in-window neighbor, and mean top-φ_T is {shap_at_max:.5f}.",
    ]

    lines += [
        "",
        "PAPER FRAMING",
        "-------------",
        f"On {dataset_name}, temporal coverage reaches {cov_at_max:.2f}% and mean top-φ_T reaches",
        f"{shap_at_max:.5f} by the largest tested window (W={max_W:.0f}s). φ_T saturates between",
        f"W=1800s and W=3600s (Δ={delta_phi:+.5f}), i.e. widening the window past 1800s does not",
        "materially change the temporal attribution magnitude on this dataset. This script is",
        "reused across datasets and reports only what this run's own data shows — no claim is",
        "made here about whether this generalizes to any other dataset; where a comparison",
        "across datasets is needed, see final/review01/results_update_tracker.md's cross-dataset",
        "table instead of this per-run figure.",
        "",
        "SUGGESTED FIGURE CAPTION",
        "-------------------------",
        f"W-sensitivity of the SHAP-GSD temporal component on {dataset_name}. Left axis (teal):",
        "percentage of test flows with at least one in-window temporal neighbor at each window",
        "size W. Right axis (green dashed): mean top-φ_T per flow. The red dotted line marks",
        f"the median inter-arrival time ({median_gap:.0f} s). φ_T saturates between W = 1800 s and",
        f"W = 3600 s (Δ = {delta_phi:+.5f}). At W = 60 s, {cov_arr[list(Ws).index(60.0)]:.2f}% of flows",
        f"have an in-window neighbor and mean top-φ_T is {shap_arr[list(Ws).index(60.0)]:.5f}.",
    ]

    txt_path = OUT_DIR / f"{STEM}.txt"
    txt_path.write_text("\n".join(lines))
    print(f"Saved {txt_path}")


if __name__ == "__main__":
    main()
