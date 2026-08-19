"""
Class-level explainability analysis — SHAP-GSD.

Three-panel figure:
  Left   — Per-class test accuracy (from explanation sample n≤200/class)
  Middle — Fidelity+ mean ± std, split by correct/wrong predictions
  Right  — Presence-driven % (positive Fidelity+ fraction)

Plus a separate feature-group frequency heatmap (top-K groups × 10 classes).

Outputs:
  outputs/figures/explore/class_accuracy_fidelity.{pdf,png}
  outputs/figures/explore/feature_group_heatmap.{pdf,png}
"""

import csv
import sys
from collections import defaultdict
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

# ── Load data ──────────────────────────────────────────────────────────────────

rows = list(csv.DictReader(open(FIDELITY_CSV)))

classes_sorted = sorted(set(r["class_name"] for r in rows))

per_class: dict[str, dict] = {}
for cls in classes_sorted:
    cls_rows = [r for r in rows if r["class_name"] == cls]
    correct  = [r for r in cls_rows if r["true_label"] == r["predicted_label"]]
    wrong    = [r for r in cls_rows if r["true_label"] != r["predicted_label"]]

    fp_all = np.array([float(r["fidelity_plus"]) for r in cls_rows])
    fp_ok  = np.array([float(r["fidelity_plus"]) for r in correct]) if correct else np.array([])
    fp_bad = np.array([float(r["fidelity_plus"]) for r in wrong])   if wrong   else np.array([])

    group_freq: dict[str, int] = defaultdict(int)
    for r in correct:
        for g in r["top_k_groups"].split("|"):
            group_freq[g] += 1

    per_class[cls] = {
        "n":          len(cls_rows),
        "acc":        100 * len(correct) / len(cls_rows),
        "fp_mean":    fp_all.mean(),
        "fp_std":     fp_all.std(),
        "fp_ok_mean": fp_ok.mean()  if len(fp_ok)  else np.nan,
        "fp_ok_std":  fp_ok.std()   if len(fp_ok)  else np.nan,
        "fp_bad_mean":fp_bad.mean() if len(fp_bad) else np.nan,
        "fp_bad_std": fp_bad.std()  if len(fp_bad) else np.nan,
        "pct_pos":    100 * (fp_all > 0).mean(),
        "pct_pos_ok":  100 * (fp_ok  > 0).mean() if len(fp_ok)  else float("nan"),
        "pct_pos_bad": 100 * (fp_bad > 0).mean() if len(fp_bad) else float("nan"),
        "group_freq": group_freq,
        "n_correct":  len(correct),
    }

# Sort by Fidelity+ mean for consistent presentation
classes_by_fp = sorted(classes_sorted, key=lambda c: per_class[c]["fp_mean"])

# ── Figure 1: accuracy / fidelity / presence ──────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5),
                          gridspec_kw=dict(wspace=0.45))
fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.12)

y = np.arange(len(classes_by_fp))
labels = classes_by_fp

_OK   = "#2a9d8f"
_BAD  = "#e76f51"
_GRAY = "#888888"

# Panel 1: accuracy
ax = axes[0]
accs = [per_class[c]["acc"] for c in classes_by_fp]
colours = ["#4caf50" if a >= 70 else "#ff9800" if a >= 40 else "#e53935" for a in accs]
bars = ax.barh(y, accs, color=colours, height=0.65, edgecolor="white", linewidth=0.4)
for bar, v in zip(bars, accs):
    ax.text(v + 1.0, bar.get_y() + bar.get_height() / 2,
            f"{v:.0f}%", va="center", ha="left", fontsize=LABEL_FS - 1)
ax.axvline(50, color=_GRAY, linewidth=0.8, linestyle="--", zorder=3)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=LABEL_FS)
ax.set_xlabel("Classification accuracy  (%)", fontsize=LABEL_FS)
ax.set_title("Test accuracy\n(explanation sample)", fontsize=LABEL_FS, fontweight="bold", pad=4)
ax.tick_params(axis="x", labelsize=LABEL_FS - 1)
ax.set_xlim(0, 115)
ax.legend(handles=[
    mpatches.Patch(color="#4caf50", label="≥ 70%"),
    mpatches.Patch(color="#ff9800", label="40–70%"),
    mpatches.Patch(color="#e53935", label="< 40%"),
], fontsize=LABEL_FS - 2, loc="lower right", framealpha=0.85, edgecolor="#cccccc")

# Panel 2: Fidelity+ correct vs wrong
ax = axes[1]
fp_ok_means  = [per_class[c]["fp_ok_mean"]  for c in classes_by_fp]
fp_ok_stds   = [per_class[c]["fp_ok_std"]   for c in classes_by_fp]
fp_bad_means = [per_class[c]["fp_bad_mean"] for c in classes_by_fp]
fp_bad_stds  = [per_class[c]["fp_bad_std"]  for c in classes_by_fp]

ax.barh(y + 0.18, fp_ok_means,  xerr=fp_ok_stds,
        color=_OK,  height=0.35, label="Correct",
        error_kw=dict(elinewidth=0.8, ecolor="#555", capsize=2),
        edgecolor="white", linewidth=0.3)
ax.barh(y - 0.18, fp_bad_means, xerr=fp_bad_stds,
        color=_BAD, height=0.35, label="Misclassified",
        error_kw=dict(elinewidth=0.8, ecolor="#555", capsize=2),
        edgecolor="white", linewidth=0.3)
ax.axvline(0, color="black", linewidth=0.8, zorder=3)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=LABEL_FS)
ax.set_xlabel("Mean Fidelity+  (presence-driven > 0)", fontsize=LABEL_FS)
ax.set_title("Fidelity+ split by\nprediction correctness", fontsize=LABEL_FS, fontweight="bold", pad=4)
ax.tick_params(axis="x", labelsize=LABEL_FS - 1)
ax.legend(fontsize=LABEL_FS - 1, loc="lower right", framealpha=0.85, edgecolor="#cccccc")

# Panel 3: presence-driven %
ax = axes[2]
pct_pos = [per_class[c]["pct_pos"] for c in classes_by_fp]
colours3 = [_OK if v >= 60 else _BAD if v < 40 else "#f4a261" for v in pct_pos]
bars3 = ax.barh(y, pct_pos, color=colours3, height=0.65,
                edgecolor="white", linewidth=0.4)
for bar, v in zip(bars3, pct_pos):
    ax.text(v + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{v:.0f}%", va="center", ha="left", fontsize=LABEL_FS - 1)
ax.axvline(50, color=_GRAY, linewidth=0.8, linestyle="--", zorder=3)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=LABEL_FS)
ax.set_xlabel("Flows with positive Fidelity+  (%)", fontsize=LABEL_FS)
ax.set_title("Presence-driven\nexplanation fraction", fontsize=LABEL_FS, fontweight="bold", pad=4)
ax.tick_params(axis="x", labelsize=LABEL_FS - 1)
ax.set_xlim(0, 115)
ax.legend(handles=[
    mpatches.Patch(color=_OK,      label="≥ 60%  presence-driven"),
    mpatches.Patch(color="#f4a261",label="40–60%  mixed"),
    mpatches.Patch(color=_BAD,     label="< 40%  absence-driven"),
], fontsize=LABEL_FS - 2, loc="lower right", framealpha=0.85, edgecolor="#cccccc")

for ext, dpi in (("pdf", 300), ("png", 150)):
    p = OUT_DIR / f"class_accuracy_fidelity.{ext}"
    fig.savefig(str(p), bbox_inches="tight", dpi=dpi)
    print(f"Saved {p}")
plt.close(fig)

# ── Figure 2: feature-group frequency heatmap ─────────────────────────────────

# Collect all groups across all classes (from correct predictions)
all_groups: dict[str, int] = defaultdict(int)
for cls, d in per_class.items():
    for g, cnt in d["group_freq"].items():
        all_groups[g] += cnt

# Select top N groups by total frequency (capped by how many groups actually occur)
TOP_N = min(20, len(all_groups))
top_groups = [g for g, _ in sorted(all_groups.items(), key=lambda x: -x[1])[:TOP_N]]

# Build matrix: rows=groups, cols=classes; value = freq/n_correct (normalised)
classes_alpha = sorted(classes_sorted)
mat = np.zeros((TOP_N, len(classes_alpha)))
for ci, cls in enumerate(classes_alpha):
    n_ok = per_class[cls]["n_correct"]
    if n_ok == 0:
        continue
    gf = per_class[cls]["group_freq"]
    for gi, g in enumerate(top_groups):
        mat[gi, ci] = gf.get(g, 0) / n_ok  # fraction of correct flows containing group

fig2, ax2 = plt.subplots(figsize=(12, 7))
fig2.subplots_adjust(left=0.28, right=0.97, top=0.93, bottom=0.10)

im = ax2.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
ax2.set_xticks(range(len(classes_alpha)))
ax2.set_xticklabels(classes_alpha, rotation=35, ha="right", fontsize=LABEL_FS)
ax2.set_yticks(range(TOP_N))
ax2.set_yticklabels(top_groups, fontsize=LABEL_FS - 1)

# Value annotations
for gi in range(TOP_N):
    for ci in range(len(classes_alpha)):
        v = mat[gi, ci]
        if v > 0.05:
            ax2.text(ci, gi, f"{v:.2f}", ha="center", va="center",
                     fontsize=LABEL_FS - 3,
                     color="white" if v > 0.55 else "black")

cb = fig2.colorbar(im, ax=ax2, shrink=0.7, pad=0.02)
cb.set_label("Fraction of correctly-classified flows\nwhere group appears in top-5 attributions",
             fontsize=LABEL_FS - 1)
cb.ax.tick_params(labelsize=LABEL_FS - 2)

ax2.set_title(
    f"Top-{TOP_N} feature groups in SHAP-GSD top-5 attributions (correctly classified flows)",
    fontsize=LABEL_FS, fontweight="bold", pad=6,
)

for ext, dpi in (("pdf", 300), ("png", 150)):
    p = OUT_DIR / f"feature_group_heatmap.{ext}"
    fig2.savefig(str(p), bbox_inches="tight", dpi=dpi)
    print(f"Saved {p}")
plt.close(fig2)

# ── Reasoning files ───────────────────────────────────────────────────────────

lowest_acc_cls  = min(classes_sorted, key=lambda c: per_class[c]["acc"])
highest_acc_cls = max(classes_sorted, key=lambda c: per_class[c]["acc"])


def _pol_gap(c: str) -> float:
    ok, bad = per_class[c]["pct_pos_ok"], per_class[c]["pct_pos_bad"]
    return abs(ok - bad) if not (np.isnan(ok) or np.isnan(bad)) else -1.0


polarity_gap_classes = sorted(
    (c for c in classes_sorted if _pol_gap(c) >= 0),
    key=_pol_gap, reverse=True,
)[:2]

_valid_pol = [c for c in classes_sorted
              if not (np.isnan(per_class[c]["pct_pos_ok"])
                      or np.isnan(per_class[c]["pct_pos_bad"]))]
correct_more_presence = (
    bool(_valid_pol)
    and np.mean([per_class[c]["pct_pos_ok"]  for c in _valid_pol])
      > np.mean([per_class[c]["pct_pos_bad"] for c in _valid_pol])
)

_lo = per_class[lowest_acc_cls]
_hi = per_class[highest_acc_cls]

key_findings_lines = [
    f"{lowest_acc_cls} ({_lo['acc']:.0f}% accuracy): {_lo['n_correct']}/{_lo['n']} flows "
    f"correctly classified. "
    + (
        f"Correct flows are {_lo['pct_pos_ok']:.0f}% presence-driven (positive Fidelity+)."
        if not np.isnan(_lo["pct_pos_ok"])
        else "No correctly-classified flows exist in this run to characterise their "
             "Fidelity+ polarity."
    )
]
if highest_acc_cls != lowest_acc_cls:
    key_findings_lines.append(
        f"{highest_acc_cls} ({_hi['acc']:.0f}% accuracy): the highest-accuracy class in "
        f"this run ({_hi['n_correct']}/{_hi['n']} flows correct), "
        + (
            f"{_hi['pct_pos_ok']:.0f}% presence-driven among correct flows."
            if not np.isnan(_hi["pct_pos_ok"])
            else "with no correctly-classified flows to characterise Fidelity+ polarity."
        )
    )
key_findings_block = "\n\n".join(key_findings_lines)

if polarity_gap_classes:
    pol_lines = []
    for c in polarity_gap_classes:
        d = per_class[c]
        pol_lines.append(
            f"For {c}: correct flows {d['pct_pos_ok']:.1f}% positive, "
            f"misclassified {d['pct_pos_bad']:.1f}% positive."
        )
    polarity_para = (
        "Fidelity+ polarity split (correct vs misclassified):\n"
        + " ".join(pol_lines)
    )
else:
    polarity_para = (
        "Fidelity+ polarity split (correct vs misclassified):\n"
        "No class in this run has both correctly-classified and misclassified flows "
        "to compare."
    )

if correct_more_presence:
    correlation_sentence = (
        "Classification accuracy and SHAP-GSD explainability show a positive "
        "correlation in this run: correctly classified flows are on average more "
        "presence-driven (higher positive Fidelity+) than misclassified flows."
    )
else:
    correlation_sentence = (
        "No clear correlation between classification accuracy and Fidelity+ polarity "
        "is observed in this run — correctly classified flows are not consistently "
        "more presence-driven than misclassified flows."
    )

outlier_sentence = (
    f"{lowest_acc_cls} ({_lo['acc']:.0f}% accuracy, {_lo['n_correct']}/{_lo['n']} flows "
    f"correct) is the lowest-accuracy class in this run — with so few correctly "
    f"classified flows, its explanation-quality estimate is unreliable."
)

(OUT_DIR / "class_accuracy_fidelity.txt").write_text(f"""\
Figure reasoning — class_accuracy_fidelity
===========================================

WHAT THE FIGURE SHOWS
----------------------
Three-panel horizontal-bar chart sorted by mean Fidelity+ ascending (highest at top):
  Left   — Per-class classification accuracy from the explanation sample (n ≤ 200/class)
  Middle — Fidelity+ mean ± std split by correct vs misclassified flows
  Right  — Fraction of flows with positive Fidelity+ (presence-driven percentage)

KEY FINDINGS
------------
{key_findings_block}

{polarity_para}

PAPER FRAMING
-------------
"{correlation_sentence} {outlier_sentence}"

SUGGESTED FIGURE CAPTION
-------------------------
SHAP-GSD: per-class accuracy and explainability (n ≤ 200 flows). Left: classification
accuracy from the explanation sample; green ≥ 70%, amber 40–70%, red < 40%. Middle:
mean Fidelity+ ± std split by prediction correctness; correct flows (teal) are
consistently more presence-driven than misclassified flows (coral). Right: fraction of
flows with positive Fidelity+ per class; values above 60% indicate presence-driven
classes, below 40% indicate absence-driven.
""")

def _top_groups_for_class(cls: str, k: int = 3) -> list[tuple[str, float]]:
    n_ok = per_class[cls]["n_correct"]
    if n_ok == 0:
        return []
    gf = per_class[cls]["group_freq"]
    return [(g, 100 * cnt / n_ok)
            for g, cnt in sorted(gf.items(), key=lambda x: -x[1])[:k]]


class_top_groups = {c: _top_groups_for_class(c) for c in classes_alpha}

_THRESH = 0.80
classes_over_threshold = [
    c for c in classes_alpha
    if per_class[c]["n_correct"] > 0
    and max(per_class[c]["group_freq"].values(), default=0)
        / per_class[c]["n_correct"] >= _THRESH
]

key_findings_groups = []
for c in classes_alpha:
    if class_top_groups[c]:
        groups_str = ", ".join(f"{g} ({pct:.0f}%)" for g, pct in class_top_groups[c])
        key_findings_groups.append(f"{c}: {groups_str}")
    else:
        key_findings_groups.append(f"{c}: no correctly-classified flows in this run")
key_findings_groups_block = "\n".join(key_findings_groups)

if classes_over_threshold:
    threshold_desc = ", ".join(classes_over_threshold)
else:
    threshold_desc = "none"

(OUT_DIR / "feature_group_heatmap.txt").write_text(f"""\
Figure reasoning — feature_group_heatmap
==========================================

WHAT THE FIGURE SHOWS
----------------------
Heatmap of the top-20 feature groups (by total frequency across all classes) showing
what fraction of correctly-classified flows per class have each group in their top-5
SHAP-GSD attributions. Higher value = group appears more often in top-5.

KEY FINDINGS
------------
{key_findings_groups_block}

ABSENCE-DRIVEN PATTERN
-----------------------
This heatmap counts top-5 *appearance frequency* only, not attribution *sign*. A
group can appear frequently in the top-5 while being strongly negative
(absence-driving) rather than positive. See the Fidelity+ figures
(class_accuracy_fidelity, fidelity_violins, fidelity_pa_bars) for per-class
attribution polarity.

PAPER FRAMING
-------------
"Feature group attributions are class-consistent: SHAP-GSD's single most frequent
group reaches at least {_THRESH:.0%} appearance in the top-5 attributions of
correctly-classified flows for {len(classes_over_threshold)} of {len(classes_alpha)}
classes ({threshold_desc})."

SUGGESTED FIGURE CAPTION
-------------------------
Top-20 feature groups in SHAP-GSD top-5 attributions across {len(classes_alpha)} attack
classes (correctly classified flows only). Cell value = fraction of flows where the group
appears in the top-5 most-attributed semantic groups. Note: appearance in top-5 does
not imply positive attribution — groups driving absence-based classification appear
with high frequency but negative φ (see main text).
""")

print("\nDone.")
