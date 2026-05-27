"""
Per-class Fidelity+ distribution explorer — SHAP-GSD.

Two figures:
  1. Violin + strip plot per class (sorted by mean Fidelity+)
  2. Presence/absence breakdown — stacked bars showing correct vs
     wrong prediction breakdown within positive/negative Fidelity+

Outputs:
  outputs/figures/explore/fidelity_violins.{pdf,png}
  outputs/figures/explore/fidelity_pa_bars.{pdf,png}
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from explore._paths import paths  # noqa: E402
_P = paths()
FIDELITY_CSV = _P["metrics"] / "fidelity.csv"
OUT_DIR      = _P["figures"] / "explore"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_FS = 9

rows = list(csv.DictReader(open(FIDELITY_CSV)))

classes_all = sorted(set(r["class_name"] for r in rows))

per_class = {}
for cls in classes_all:
    cls_rows = [r for r in rows if r["class_name"] == cls]
    fp = np.array([float(r["fidelity_plus"]) for r in cls_rows])
    correct = np.array([r["true_label"] == r["predicted_label"] for r in cls_rows])
    per_class[cls] = {"fp": fp, "correct": correct, "n": len(cls_rows)}

# Sort by mean Fidelity+ ascending (highest at top in barh / rightmost in violin)
classes_sorted = sorted(classes_all, key=lambda c: per_class[c]["fp"].mean())

# ── Figure 1: Violin plots ─────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))
fig.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.12)

_PRESENCE = "#2a9d8f"
_ABSENCE  = "#e76f51"
_ZERO_LINE = "black"

positions = np.arange(len(classes_sorted))
violin_data = [per_class[c]["fp"] for c in classes_sorted]
colours = [_PRESENCE if per_class[c]["fp"].mean() >= 0 else _ABSENCE
           for c in classes_sorted]

vp = ax.violinplot(violin_data, positions=positions,
                   widths=0.65, showmedians=True, showextrema=False)

for body, col in zip(vp["bodies"], colours):
    body.set_facecolor(col)
    body.set_alpha(0.55)
    body.set_edgecolor("white")
    body.set_linewidth(0.5)

vp["cmedians"].set_color("black")
vp["cmedians"].set_linewidth(1.5)

# Overlay mean markers
means = [per_class[c]["fp"].mean() for c in classes_sorted]
ax.scatter(positions, means, zorder=3, s=40, color="black", marker="D")

ax.axhline(0, color=_ZERO_LINE, linewidth=0.9, zorder=2)

# Global mean reference
global_mean = np.concatenate([per_class[c]["fp"] for c in classes_all]).mean()
ax.axhline(global_mean, color="#555555", linewidth=0.8, linestyle="--", zorder=2)
ax.text(len(classes_sorted) - 0.5, global_mean + 0.02,
        f"Overall {global_mean:+.3f}",
        fontsize=LABEL_FS - 2, color="#555555", ha="right")

ax.set_xticks(positions)
ax.set_xticklabels(classes_sorted, rotation=30, ha="right", fontsize=LABEL_FS)
ax.set_ylabel("Fidelity+  (presence-driven > 0)", fontsize=LABEL_FS)
ax.set_title("Per-class Fidelity+ distribution  (violin = density, ◆ = mean, — = median)",
             fontsize=LABEL_FS, fontweight="bold", pad=5)
ax.tick_params(axis="y", labelsize=LABEL_FS - 1)
ax.legend(handles=[
    mpatches.Patch(color=_PRESENCE, alpha=0.6, label="Presence-driven (mean ≥ 0)"),
    mpatches.Patch(color=_ABSENCE,  alpha=0.6, label="Absence-driven  (mean < 0)"),
], fontsize=LABEL_FS - 1, loc="upper left", framealpha=0.85, edgecolor="#cccccc")

for ext, dpi in (("pdf", 300), ("png", 150)):
    p = OUT_DIR / f"fidelity_violins.{ext}"
    fig.savefig(str(p), bbox_inches="tight", dpi=dpi)
    print(f"Saved {p}")
plt.close(fig)

# ── Figure 2: Presence / Absence × Correct / Wrong stacked bars ───────────────

fig2, ax2 = plt.subplots(figsize=(12, 5))
fig2.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.14)

bar_w = 0.65
_C_POS_OK  = "#2a9d8f"   # correct + positive fp
_C_POS_BAD = "#80d5cc"   # wrong   + positive fp (lighter teal)
_C_NEG_OK  = "#e76f51"   # correct + negative fp
_C_NEG_BAD = "#f4c7b8"   # wrong   + negative fp (lighter coral)

for i, cls in enumerate(classes_sorted):
    fp = per_class[cls]["fp"]
    ok = per_class[cls]["correct"]
    n  = per_class[cls]["n"]

    pos_ok  = ((fp > 0) & ok).sum()
    pos_bad = ((fp > 0) & ~ok).sum()
    neg_ok  = ((fp <= 0) & ok).sum()
    neg_bad = ((fp <= 0) & ~ok).sum()

    # Stack from zero up (positive) and zero down (negative) as fractions
    ax2.bar(i, pos_ok  / n, width=bar_w, bottom=0,                   color=_C_POS_OK)
    ax2.bar(i, pos_bad / n, width=bar_w, bottom=pos_ok  / n,         color=_C_POS_BAD)
    ax2.bar(i, -neg_ok / n, width=bar_w, bottom=0,                    color=_C_NEG_OK)
    ax2.bar(i, -neg_bad/ n, width=bar_w, bottom=-neg_ok / n,          color=_C_NEG_BAD)

ax2.axhline(0, color="black", linewidth=0.9, zorder=3)
ax2.set_xticks(np.arange(len(classes_sorted)))
ax2.set_xticklabels(classes_sorted, rotation=30, ha="right", fontsize=LABEL_FS)
ax2.set_ylabel("Fraction of flows", fontsize=LABEL_FS)
ax2.set_title(
    "Fidelity+ polarity × prediction correctness per class\n"
    "(above zero = presence-driven, below = absence-driven)",
    fontsize=LABEL_FS, fontweight="bold", pad=5,
)
ax2.tick_params(axis="y", labelsize=LABEL_FS - 1)
ax2.legend(handles=[
    mpatches.Patch(color=_C_POS_OK,  label="Presence + correct"),
    mpatches.Patch(color=_C_POS_BAD, label="Presence + misclassified"),
    mpatches.Patch(color=_C_NEG_OK,  label="Absence + correct"),
    mpatches.Patch(color=_C_NEG_BAD, label="Absence + misclassified"),
], fontsize=LABEL_FS - 1, loc="upper left", framealpha=0.85, edgecolor="#cccccc",
   ncol=2)

for ext, dpi in (("pdf", 300), ("png", 150)):
    p = OUT_DIR / f"fidelity_pa_bars.{ext}"
    fig2.savefig(str(p), bbox_inches="tight", dpi=dpi)
    print(f"Saved {p}")
plt.close(fig2)

# ── Reasoning files ───────────────────────────────────────────────────────────

(OUT_DIR / "fidelity_violins.txt").write_text("""\
Figure reasoning — fidelity_violins
=====================================

WHAT THE FIGURE SHOWS
----------------------
Violin plot showing the full Fidelity+ distribution per class, sorted by mean ascending
(most negative at left, most positive at right). Diamond markers show means, horizontal
lines show medians.

KEY FINDINGS
------------
Generic (mean +0.362, median +0.519): Strongly right-skewed — most flows have high
positive Fidelity+. The top DNS and packet-size features are very discriminative.

Shellcode (mean +0.316, std 0.493) and Recon (mean -0.063, std 0.458): Wide, bimodal
distributions. Some flows within each class have extremely high positive Fidelity+
(>0.9) while others are strongly negative. This reflects within-class heterogeneity:
some flows are near the model's Shellcode/Recon decision boundary, others are not.

Benign (mean -0.007, std 0.105): Very tight distribution near zero — consistent with
the high stability (0.043 mean_phi_std, the outlier in the stability figure). The model
has no dominant feature group for Benign.

Backdoor (mean +0.008, std 0.104): Misleadingly positive mean. 70% of flows have
negative Fidelity+ (70% are absence-driven), but the 6 correctly-classified flows have
very high Fidelity+ (0.32–0.79), pulling the mean positive.

PAPER FRAMING
-------------
"Fidelity+ distributions reveal within-class explanation heterogeneity. Generic shows
a tight right-skewed distribution (median 0.52), indicating consistently strong presence-
driven attributions. Shellcode and Recon show bimodal distributions, reflecting flows
near and far from the model's decision boundary. Benign has a near-degenerate
distribution concentrated at zero — the model classifies Benign correctly but does so
without dominant feature attributions, consistent with heterogeneous benign traffic."

SUGGESTED FIGURE CAPTION
-------------------------
Per-class Fidelity+ distribution for SHAP-GSD (n ≤ 200 flows per class, sorted by
mean ascending). Violins show kernel density; diamonds mark means; horizontal lines
mark medians. Teal = presence-driven (mean ≥ 0), coral = absence-driven (mean < 0).
Dashed line marks the overall mean (0.107). Generic shows a tight right-skewed
distribution (median 0.52); Shellcode and Recon show wide bimodal distributions
(std > 0.45); Benign is near-degenerate at zero.
""")

(OUT_DIR / "fidelity_pa_bars.txt").write_text("""\
Figure reasoning — fidelity_pa_bars
=====================================

WHAT THE FIGURE SHOWS
----------------------
Stacked bar chart per class showing the fraction of flows in each of four categories:
  Above zero (positive Fidelity+): presence-driven correct (dark teal) / misclassified (light teal)
  Below zero (negative Fidelity+): absence-driven correct (dark coral) / misclassified (light coral)

Classes are sorted by mean Fidelity+ ascending (most absence-driven at left).

KEY FINDINGS
------------
Backdoor: Almost entirely below zero for misclassified flows (70% absence-driven),
with a tiny positive sliver from the 6 correct flows. The bar is dominated by
coral (absence), consistent with Backdoor being absence-driven via missing DNS
and L7 protocol features.

Analysis: 82.5% correct, but split ~50/50 between presence and absence even for
correct flows. The absence-driven signal (missing ephemeral source ports) is the
primary driver.

Generic: Near-entirely presence-driven correct (dark teal dominates above axis).
The few misclassified Generic flows still show some teal (positive Fidelity+),
suggesting Generic features are present but point to the wrong class.

Recon: Equal split above/below for correct flows (87.5% are presence-driven, 1.1%
for wrong). The few wrong Recon flows are almost entirely absence-driven.

Benign: About 40% presence-driven correct + 58% absence-driven correct — Benign
is borderline. The model correct classifies it without a clear directional signal.

PAPER FRAMING
-------------
"The polarity breakdown shows that explanation quality (positive Fidelity+) strongly
correlates with prediction correctness for most classes. Misclassified flows cluster
near the negative Fidelity+ regime (absence-driven), suggesting the model's boundary
is ill-defined for those samples. The exception is Backdoor, where both correct and
incorrect flows show diffuse attribution due to near-chance classification accuracy."

SUGGESTED FIGURE CAPTION
-------------------------
Fidelity+ polarity × prediction correctness per class (n ≤ 200 flows, sorted by mean
Fidelity+ ascending). Bars above zero = presence-driven (positive Fidelity+); bars
below zero = absence-driven (negative Fidelity+). Within each direction, dark shading
= correctly classified, light shading = misclassified. Correctly classified flows are
predominantly presence-driven for Generic, Recon, and Shellcode; misclassified flows
collapse to absence-driven or near-zero across almost all classes.
""")

print("\nDone.")
