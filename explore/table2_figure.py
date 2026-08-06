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

from explore._paths import paths  # noqa: E402
_P = paths()
SUMMARY_PATH = _P["metrics"] / "summary.json"
OUT_DIR      = _P["figures"] / "explore"
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

fidelity_leaders = sorted(classes, key=lambda c: -per_class[c]["fidelity_plus"])[:2]

fm_vals = [per_class[c]["fidelity_minus"] for c in classes]
fm_min, fm_max = min(fm_vals), max(fm_vals)

green_classes = sorted((c for c in classes if per_class[c]["stability"] <= 0.010),
                        key=lambda c: per_class[c]["stability"])
amber_classes = sorted((c for c in classes if 0.010 < per_class[c]["stability"] <= 0.025),
                        key=lambda c: per_class[c]["stability"])
red_classes   = sorted((c for c in classes if per_class[c]["stability"] > 0.025),
                        key=lambda c: -per_class[c]["stability"])

stability_outlier_cls = max(classes, key=lambda c: per_class[c]["stability"])


def _band_desc(band_classes: list) -> str:
    if not band_classes:
        return "no classes fall in this band"
    return ", ".join(f"{c} ({per_class[c]['stability']:.4f})" for c in band_classes)


leaders_desc = " and ".join(fidelity_leaders)

reasoning = f"""\
Figure reasoning — table2_fidelity_stability
=============================================

WHY THIS LAYOUT
---------------
The diverging Fidelity+ chart is doing real argumentative work. The teal/coral
split immediately encodes the "presence-driven vs absence-driven" distinction
from the methodology — readers absorb the class split before reading a
word. Classes are sorted by Fidelity+ descending so {leaders_desc} lead.

WHY FIDELITY- IS OMITTED
-------------------------
Fidelity− was left out of the figure. Given its tight range ({fm_min:.2f} to
{fm_max:.2f}) and mostly-consistent direction, it adds little visual insight
beyond what the table already says. One sentence in the caption covers it
without a third chart competing for space:

  "Fidelity− was {fm_min:.2f}–{fm_max:.2f} across most classes, indicating
   top-5 groups are compact and sufficient."

STABILITY COLOUR BANDS
-----------------------
Stability is colour-coded in three bands:

  Green  (≤ 0.010) — near-deterministic. {_band_desc(green_classes)}. The
                      explanation is highly consistent across coalition seeds.

  Amber  (≤ 0.025) — stable. {_band_desc(amber_classes)}.

  Red    (> 0.025) — variable. {_band_desc(red_classes)}.

STABILITY OUTLIER ({stability_outlier_cls}, {per_class[stability_outlier_cls]['stability']:.3f})
----------------------------------------------------------------------
{stability_outlier_cls} shows the most cross-seed variance of any class in this
run. This warrants qualitative investigation before asserting why — it may
reflect genuinely harder-to-summarise traffic for that class, or it may be an
artefact of a small explanation sample; the numbers alone do not distinguish
the two.

OVERALL DASHED LINE
-------------------
Both panels carry a dashed vertical line at the overall mean (Fidelity+
{overall['fidelity_plus']:.3f}, Stability {overall['stability']:.3f}). This
gives the reader an immediate reference: classes above the line are
better-than-average, classes below are worse. It also visually anchors the
caption statistics without requiring the reader to locate the number in the
table.

SUGGESTED PAPER CAPTION
------------------------
"Fidelity+ and Stability of SHAP-GSD feature-group explanations across
{len(classes)} classes (k = {summary['top_k']}, n = {overall['n_flows']:,}
flows, 3 coalition seeds). Fidelity+ < 0 indicates absence-driven classes
where top-5 attributed groups include negative-φ features; positive values
indicate presence-driven classes. Fidelity− was {fm_min:.2f}–{fm_max:.2f}
across most classes, indicating top-5 groups are compact and sufficient.
Stability reflects mean per-group φ standard deviation across coalition
sampling seeds; lower is more stable. {stability_outlier_cls}
({per_class[stability_outlier_cls]['stability']:.3f}) shows the most
cross-seed variance of any class in this run."

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
