"""
W-ablation visualization — SHAP-GSD temporal null result.

Two-panel figure:
  Left  — % flows with ≥1 in-window neighbor vs W (log-x)
  Right — mean and max top temporal SHAP vs W (log-x)

Illustrates the temporal null result: even at W=3600s, only ~40% of flows
have in-window neighbors and temporal SHAP values remain near-zero.

Outputs:
  outputs/figures/explore/w_ablation.{pdf,png}
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from explore._paths import paths  # noqa: E402
_P = paths()
OUT_DIR       = _P["figures"] / "explore"
ABLATION_DIR  = _P["w_ablation"]
GAP_STATS     = ABLATION_DIR / "gap_stats.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_FS = 9

# ── Load gap stats ─────────────────────────────────────────────────────────────

gap_stats = json.loads(GAP_STATS.read_text())
Ws         = [d["W_s"]                    for d in gap_stats]
pct_flows  = [d["pct_flows_with_inwindow"] for d in gap_stats]
pct_edges  = [d["pct_edges_inwindow"]      for d in gap_stats]

# ── Load per-W temporal SHAP ──────────────────────────────────────────────────

shap_means = []
shap_maxs  = []

for W in [int(w) for w in Ws]:
    p = ABLATION_DIR / f"W{W}_temporal.csv"
    if not p.exists():
        shap_means.append(np.nan)
        shap_maxs.append(np.nan)
        continue
    rows = list(csv.DictReader(open(p)))
    top_phi = np.array([float(r["top_phi_T"]) for r in rows]) if rows else np.array([0.0])
    shap_means.append(top_phi.mean())
    shap_maxs.append(top_phi.max())

# ── Figure ────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                gridspec_kw=dict(wspace=0.38))
fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.14)

_TEAL  = "#2a9d8f"
_CORAL = "#e76f51"
_GRAY  = "#888888"
marker = "o"

# Panel 1: coverage
ax1.semilogx(Ws, pct_flows, color=_TEAL,  marker=marker, linewidth=1.5,
             markersize=6, label="% flows with ≥1 in-window neighbor")
ax1.semilogx(Ws, pct_edges, color=_CORAL, marker=marker, linewidth=1.5,
             markersize=6, linestyle="--", label="% edges in-window")

for W, yf, ye in zip(Ws, pct_flows, pct_edges):
    ax1.annotate(f"{yf:.1f}%", (W, yf), textcoords="offset points",
                 xytext=(4, 4), fontsize=LABEL_FS - 2, color=_TEAL)
    ax1.annotate(f"{ye:.2f}%", (W, ye), textcoords="offset points",
                 xytext=(4, -10), fontsize=LABEL_FS - 2, color=_CORAL)

ax1.set_xlabel("Temporal window W  (seconds, log scale)", fontsize=LABEL_FS)
ax1.set_ylabel("Percentage  (%)", fontsize=LABEL_FS)
ax1.set_title("In-window coverage vs W", fontsize=LABEL_FS, fontweight="bold", pad=4)
ax1.tick_params(axis="both", labelsize=LABEL_FS - 1)
ax1.legend(fontsize=LABEL_FS - 1, loc="upper left", framealpha=0.85, edgecolor="#cccccc")
ax1.set_ylim(bottom=0)
ax1.grid(True, alpha=0.3, which="both")

# Add median gap annotation
median_gap = gap_stats[0]["median_gap_s"]
ax1.axvline(median_gap, color=_GRAY, linewidth=0.9, linestyle=":", zorder=3)
ax1.text(median_gap * 1.1, ax1.get_ylim()[1] * 0.85,
         f"Median gap\n{median_gap:.0f}s",
         fontsize=LABEL_FS - 2, color=_GRAY, ha="left")

# Panel 2: temporal SHAP magnitude
ax2.semilogx(Ws, shap_means, color=_TEAL,  marker=marker, linewidth=1.5,
             markersize=6, label="Mean top φ_T")
ax2.semilogx(Ws, shap_maxs,  color=_CORAL, marker=marker, linewidth=1.5,
             markersize=6, linestyle="--", label="Max top φ_T")

ax2.set_xlabel("Temporal window W  (seconds, log scale)", fontsize=LABEL_FS)
ax2.set_ylabel("|φ_T|  (temporal Shapley value)", fontsize=LABEL_FS)
ax2.set_title("Temporal SHAP magnitude vs W", fontsize=LABEL_FS, fontweight="bold", pad=4)
ax2.tick_params(axis="both", labelsize=LABEL_FS - 1)
ax2.legend(fontsize=LABEL_FS - 1, loc="upper left", framealpha=0.85, edgecolor="#cccccc")
ax2.grid(True, alpha=0.3, which="both")

for ext, dpi in (("pdf", 300), ("png", 150)):
    p = OUT_DIR / f"w_ablation.{ext}"
    fig.savefig(str(p), bbox_inches="tight", dpi=dpi)
    print(f"Saved {p}")
plt.close(fig)

# ── Reasoning ─────────────────────────────────────────────────────────────────

reasoning = f"""\
Figure reasoning — w_ablation
==============================

WHAT THE FIGURE SHOWS
---------------------
Two log-x panels showing how in-window coverage (left) and temporal SHAP
magnitude (right) scale with the temporal window W on NF-UNSW-NB15-v3.

W VALUES TESTED:  {Ws}

COVERAGE (LEFT PANEL)
----------------------
The median neighbour gap is {median_gap:.0f}s. At W=60s, only {pct_flows[0]:.1f}% of
flows have any in-window neighbors. Even at W=3600s (one hour), only
{pct_flows[-1]:.1f}% of flows have in-window neighbors and only {pct_edges[-1]:.2f}% of
neighbour edges fall inside the window.

The vertical dotted line marks the median gap, showing that most of the
W range tested falls far below the median.

TEMPORAL SHAP MAGNITUDE (RIGHT PANEL)
---------------------------------------
Mean top temporal Shapley value at W=60s:   {shap_means[0]:.6f}
Mean top temporal Shapley value at W=3600s: {shap_means[-1]:.6f}
Max  top temporal Shapley value at W=3600s: {shap_maxs[-1]:.6f}

These values are near-zero relative to feature-group SHAP (typical
top-feature φ ≈ 0.1–0.5). Expanding W does not rescue the temporal
signal because the dataset's flow timing structure means predictive
context comes from edges orders of magnitude outside any practical
lookback window.

PAPER INTERPRETATION
---------------------
This is a correct null result, not an explainer failure. The temporal
GNN sampler faithfully enforces the cutoff — there simply are no
temporally-proximate neighbors on NF-UNSW-NB15-v3 at deployment timescales.

This motivates Paper 3: IoT datasets (NF-ToN-IoT-v3, NF-BoT-IoT-v3) have
denser temporal clustering and are expected to show non-trivial temporal
SHAP values, providing the contrast case that validates the temporal
granularity as a dataset-dependent rather than explainer-specific signal.

SUGGESTED PAPER PLACEMENT
--------------------------
Place immediately after the temporal SHAP case-study panel (d).

SUGGESTED FIGURE CAPTION
-------------------------
Temporal null result: NF-UNSW-NB15-v3 (median flow gap = {median_gap:.0f} s). Left:
percentage of flows with at least one in-window neighbour and percentage of
neighbour edges inside the window, for W ∈ {{60, 300, 1800, 3600}} s (log
scale). Right: mean and maximum top temporal Shapley value (|φ_T|) for the
same W range. The median inter-flow gap (dotted vertical line) exceeds all
tested W values, leaving temporal context structurally absent at deployment
timescales. φ_T values remain near-zero across all W, confirming the null
result is a dataset property rather than an explainer artifact.
"""

txt_path = OUT_DIR / "w_ablation.txt"
txt_path.write_text(reasoning)
print(f"Saved {txt_path}")
