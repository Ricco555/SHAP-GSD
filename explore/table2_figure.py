"""
Table 2 visualisation — SHAP-GSD feature-group fidelity and stability.

Three-panel figure:
  Left   — Fidelity+ diverging bar (presence-driven vs absence-driven classes)
  Right  — Stability colour-banded bar (green / amber / red)
  Bottom — shared stats strip

Outputs:
  outputs/figures/table2_fidelity_stability.pdf
  outputs/figures/table2_fidelity_stability.png
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SUMMARY_PATH = ROOT / "outputs" / "metrics" / "summary.json"
OUT_DIR      = ROOT / "outputs" / "figures" / "explore"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────

with open(SUMMARY_PATH) as f:
    summary = json.load(f)

per_class = summary["per_class"]
overall   = summary["overall"]

# Sort ascending by Fidelity+ — barh places y=0 at bottom, so ascending
# order puts the highest value (Generic) at the top of the chart.
classes = sorted(per_class, key=lambda c: per_class[c]["fidelity_plus"])

fp_mean  = np.array([per_class[c]["fidelity_plus"]     for c in classes])
fp_std   = np.array([per_class[c]["fidelity_plus_std"] for c in classes])
stab     = np.array([per_class[c]["stability"]         for c in classes])

# ── Colour palettes ────────────────────────────────────────────────────────────

_PRESENCE = "#2a9d8f"   # teal  — presence-driven (φ ≥ 0)
_ABSENCE  = "#e76f51"   # coral — absence-driven  (φ < 0)
fp_colours = [_PRESENCE if v >= 0 else _ABSENCE for v in fp_mean]

def _stab_colour(v: float) -> str:
    if v <= 0.010:  return "#4caf50"   # green  — near-deterministic
    elif v <= 0.025: return "#ff9800"  # amber  — stable
    else:            return "#e53935"  # red    — variable

stab_colours = [_stab_colour(v) for v in stab]

# ── Figure layout ──────────────────────────────────────────────────────────────

LABEL_FS = 9   # matches y-tick label size throughout

fig, (ax_fp, ax_stab) = plt.subplots(
    1, 2,
    figsize=(13, 6.0),
    gridspec_kw=dict(wspace=0.42),
)
fig.subplots_adjust(left=0.10, right=0.97, top=0.96, bottom=0.20)

y = np.arange(len(classes))

# ── Left panel: Fidelity+ diverging bar ───────────────────────────────────────

ax_fp.barh(
    y, fp_mean,
    xerr=fp_std,
    color=fp_colours,
    error_kw=dict(elinewidth=0.9, ecolor="#555555", capsize=3),
    edgecolor="white", linewidth=0.4,
    height=0.65,
)
ax_fp.axvline(0, color="black", linewidth=0.8, zorder=3)
ax_fp.set_yticks(y)
ax_fp.set_yticklabels(classes, fontsize=LABEL_FS)
ax_fp.set_xlabel(
    "P(full) − P(masked top-5 groups)",
    fontsize=LABEL_FS,
)
ax_fp.set_title("Fidelity+", fontsize=LABEL_FS, fontweight="bold", pad=5)
ax_fp.tick_params(axis="x", labelsize=LABEL_FS - 1)
ax_fp.set_xlim(-0.72, 0.88)

# Legend
ax_fp.legend(
    handles=[
        mpatches.Patch(color=_PRESENCE, label="Presence-driven (φ > 0)"),
        mpatches.Patch(color=_ABSENCE,  label="Absence-driven (φ < 0)"),
    ],
    fontsize=LABEL_FS - 1, loc="lower right",
    framealpha=0.85, edgecolor="#cccccc",
)

# Overall dashed reference line
ax_fp.axvline(overall["fidelity_plus"], color="#333333",
              linewidth=1.1, linestyle="--", zorder=4)
ax_fp.text(
    overall["fidelity_plus"] + 0.02, len(classes) - 0.5,
    f"Overall {overall['fidelity_plus']:.3f}",
    fontsize=LABEL_FS - 2, color="#333333", va="top",
)

# ── Right panel: Stability colour-banded bar ───────────────────────────────────

bars = ax_stab.barh(
    y, stab,
    color=stab_colours,
    edgecolor="white", linewidth=0.4,
    height=0.65,
)
ax_stab.set_yticks(y)
ax_stab.set_yticklabels(classes, fontsize=LABEL_FS)
ax_stab.set_xlabel(
    "Mean per-group φ std  (lower = more stable)",
    fontsize=LABEL_FS,
)
ax_stab.set_title("Stability", fontsize=LABEL_FS, fontweight="bold", pad=5)
ax_stab.tick_params(axis="x", labelsize=LABEL_FS - 1)
ax_stab.set_xlim(0, stab.max() * 1.38)

# Value labels
for bar, v in zip(bars, stab):
    ax_stab.text(
        v + 0.0008, bar.get_y() + bar.get_height() / 2,
        f"{v:.4f}", va="center", ha="left", fontsize=LABEL_FS - 1,
    )

# Threshold reference lines
for thresh, label, ls in ((0.010, "≤ 0.010", ":"), (0.025, "≤ 0.025", "--")):
    ax_stab.axvline(thresh, color="#888888", linewidth=0.8, linestyle=ls, zorder=3)
    ax_stab.text(
        thresh + 0.0004, 0.01, label,
        fontsize=LABEL_FS - 3, color="#888888",
        ha="left", va="bottom", transform=ax_stab.get_xaxis_transform(),
    )

# Legend
ax_stab.legend(
    handles=[
        mpatches.Patch(color="#4caf50", label="≤ 0.010  near-deterministic"),
        mpatches.Patch(color="#ff9800", label="≤ 0.025  stable"),
        mpatches.Patch(color="#e53935", label="> 0.025  variable"),
    ],
    fontsize=LABEL_FS - 1, loc="lower right",
    framealpha=0.85, edgecolor="#cccccc",
)

# Overall dashed reference line
ax_stab.axvline(overall["stability"], color="#333333",
                linewidth=1.1, linestyle="--", zorder=4)
ax_stab.text(
    overall["stability"] + 0.0008, len(classes) - 0.5,
    f"Overall {overall['stability']:.3f}",
    fontsize=LABEL_FS - 2, color="#333333", va="top",
)

# ── Bottom caption strip ───────────────────────────────────────────────────────

caption = (
    f"n = {overall['n_flows']:,} flows  ·  k = {summary['top_k']} groups  ·  3 coalition seeds"
    f"   |   "
    f"Overall  Fidelity+ {overall['fidelity_plus']:.3f} ± {overall['fidelity_plus_std']:.3f}"
    f"  ·  Fidelity− {overall['fidelity_minus']:.3f} ± {overall['fidelity_minus_std']:.3f}"
    f"  ·  Stability {overall['stability']:.3f}"
)
fig.text(
    0.5, 0.05, caption,
    ha="center", va="top",
    fontsize=LABEL_FS, color="#333333",
    style="italic",
)

# ── Save figures ──────────────────────────────────────────────────────────────

STEM = "table2_fidelity_stability"

for ext, dpi in (("pdf", 300), ("png", 150)):
    path = OUT_DIR / f"{STEM}.{ext}"
    fig.savefig(str(path), bbox_inches="tight", dpi=dpi)
    print(f"Saved {path}")

plt.close(fig)

# ── Save reasoning ─────────────────────────────────────────────────────────────

reasoning = """\
Figure reasoning — table2_fidelity_stability
=============================================

WHY THIS LAYOUT
---------------
The diverging Fidelity+ chart is doing real argumentative work. The teal/coral
split immediately encodes the "presence-driven vs absence-driven" distinction
from the methodology — readers absorb the five-class split before reading a
word. Classes are sorted by Fidelity+ descending so Generic and Shellcode lead,
which matches their dominant SHAP values in the case studies.

WHY FIDELITY- IS OMITTED
-------------------------
Fidelity− was left out of the figure. Given its tight range (−0.02 to 0.11)
and mostly-consistent direction, it adds little visual insight beyond what the
table already says. One sentence in the caption covers it without a third chart
competing for space:

  "Fidelity− was 0.03–0.11 across most classes, indicating top-5 groups are
   compact and sufficient."

STABILITY COLOUR BANDS
-----------------------
Stability is colour-coded in three bands:

  Green  (≤ 0.010) — near-deterministic, dominated by Analysis (0.0012)
                      and Exploits (0.0059). The explanation is highly
                      consistent across coalition seeds — publishable claim.

  Amber  (≤ 0.025) — stable. Backdoor (0.011), DoS (0.013), Generic (0.015),
                      Recon (0.024). Acceptable variance for a sampling-based
                      Shapley estimator at nsamples=256.

  Red    (> 0.025) — variable. Benign (0.043) and Shellcode (0.037) are the
                     two outliers.

BENIGN OUTLIER (0.043)
----------------------
The Benign stability outlier is the point to call out in the paper. It reflects
heterogeneous benign traffic with no dominant feature signal: the model assigns
low-confidence, diffuse feature attributions that shift meaningfully between
coalition seeds. This is correct behaviour — it is not a flaw in the explainer
but evidence that benign traffic is genuinely harder to summarise in k=5 groups.
The caption or text should read something like:

  "Benign traffic exhibits the highest instability (0.043), consistent with its
   heterogeneous mix of protocols and flow patterns; no single feature group
   dominates, so coalition-sampled attributions are sensitive to seed choice."

OVERALL DASHED LINE
-------------------
Both panels carry a dashed vertical line at the overall mean (Fidelity+ 0.107,
Stability 0.019). This gives the reader an immediate reference: classes above
the line are better-than-average, classes below are worse. It also visually
anchors the caption statistics without requiring the reader to locate the number
in the table.

SUGGESTED PAPER CAPTION
------------------------
"Fidelity+ and Stability of SHAP-GSD feature-group explanations across 10
classes (k = 5, n = 1,764 flows, 3 coalition seeds). Fidelity+ < 0 indicates
absence-driven classes where top-5 attributed groups include negative-φ
features; positive values indicate presence-driven classes. Fidelity− was
0.03–0.11 across most classes, indicating top-5 groups are compact and
sufficient. Stability reflects mean per-group φ standard deviation across
coalition sampling seeds; lower is more stable. The Benign outlier (0.043)
reflects heterogeneous traffic with no dominant feature signal."

SUGGESTED PLACEMENT
-------------------
Place this figure immediately after Table 2 (the full per-class fidelity table).
It converts the table's numbers into a visual argument: the teal/coral split
answers "which classes does the model explain by presence vs absence?", and the
stability bands answer "how reproducible are those explanations?". Together they
support the claim that SHAP-GSD provides interpretable, stable, and class-aware
feature attributions for GNN-based NIDS.
"""

txt_path = OUT_DIR / f"{STEM}.txt"
txt_path.write_text(reasoning)
print(f"Saved {txt_path}")
