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

pa_stats: dict[str, dict] = {}
for i, cls in enumerate(classes_sorted):
    fp = per_class[cls]["fp"]
    ok = per_class[cls]["correct"]
    n  = per_class[cls]["n"]

    pos_ok  = ((fp > 0) & ok).sum()
    pos_bad = ((fp > 0) & ~ok).sum()
    neg_ok  = ((fp <= 0) & ok).sum()
    neg_bad = ((fp <= 0) & ~ok).sum()

    pa_stats[cls] = dict(pos_ok=pos_ok, pos_bad=pos_bad,
                          neg_ok=neg_ok, neg_bad=neg_bad, n=n)

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

highest_mean_cls = classes_sorted[-1]
lowest_mean_cls  = classes_sorted[0]
highest_std_cls  = max(classes_all, key=lambda c: per_class[c]["fp"].std())


def _fp_stats(cls: str) -> dict:
    fp = per_class[cls]["fp"]
    return dict(mean=fp.mean(), median=float(np.median(fp)), std=fp.std())


_named = set()


def _dedupe_note(cls: str) -> str:
    if cls in _named:
        return " (same class as above)"
    _named.add(cls)
    return ""


_s_hi  = _fp_stats(highest_mean_cls)
hi_note = _dedupe_note(highest_mean_cls)
key_line_hi = (
    f"{highest_mean_cls} (mean {_s_hi['mean']:+.3f}, median {_s_hi['median']:+.3f})"
    f"{hi_note}: the most presence-driven class by mean Fidelity+ in this run."
)

_s_lo = _fp_stats(lowest_mean_cls)
lo_note = _dedupe_note(lowest_mean_cls)
key_line_lo = (
    f"{lowest_mean_cls} (mean {_s_lo['mean']:+.3f}, std {_s_lo['std']:.3f}){lo_note}: "
    f"the most absence-driven class by mean Fidelity+ in this run."
)

_s_std = _fp_stats(highest_std_cls)
std_note = _dedupe_note(highest_std_cls)
key_line_std = (
    f"{highest_std_cls} (std {_s_std['std']:.3f}){std_note}: "
    f"the widest Fidelity+ spread of any class in this run."
)

key_findings_lines = [key_line_hi, key_line_lo, key_line_std]
key_findings_block = "\n\n".join(key_findings_lines)

(OUT_DIR / "fidelity_violins.txt").write_text(f"""\
Figure reasoning — fidelity_violins
=====================================

WHAT THE FIGURE SHOWS
----------------------
Violin plot showing the full Fidelity+ distribution per class, sorted by mean ascending
(most negative at left, most positive at right). Diamond markers show means, horizontal
lines show medians.

KEY FINDINGS
------------
{key_findings_block}

PAPER FRAMING
-------------
"Fidelity+ distributions reveal within-class explanation heterogeneity. {highest_mean_cls}
is the most presence-driven class in this run (mean {_s_hi['mean']:+.3f}, median
{_s_hi['median']:+.3f}). {lowest_mean_cls} is the most absence-driven (mean
{_s_lo['mean']:+.3f}). {highest_std_cls} shows the widest spread (std {_s_std['std']:.3f}),
indicating within-class heterogeneity in how strongly individual flows are explained."

SUGGESTED FIGURE CAPTION
-------------------------
Per-class Fidelity+ distribution for SHAP-GSD (n ≤ 200 flows per class, sorted by
mean ascending). Violins show kernel density; diamonds mark means; horizontal lines
mark medians. Teal = presence-driven (mean ≥ 0), coral = absence-driven (mean < 0).
Dashed line marks the overall mean ({global_mean:+.3f}). {highest_mean_cls} shows the
most presence-driven distribution (mean {_s_hi['mean']:+.3f}); {lowest_mean_cls} is the
most absence-driven (mean {_s_lo['mean']:+.3f}); {highest_std_cls} has the widest spread
(std {_s_std['std']:.3f}).
""")

def _acc(cls: str) -> float:
    return 100 * per_class[cls]["correct"].mean()


def _presence_frac(cls: str) -> float:
    s = pa_stats[cls]
    return (s["pos_ok"] + s["pos_bad"]) / s["n"]


most_presence_cls = max(classes_sorted, key=_presence_frac)
most_absence_cls  = min(classes_sorted, key=_presence_frac)
lowest_acc_cls    = min(classes_sorted, key=_acc)

_pa_named = set()


def _pa_dedupe_note(cls: str) -> str:
    if cls in _pa_named:
        return " (same class as above)"
    _pa_named.add(cls)
    return ""


def _pa_desc(cls: str) -> str:
    s = pa_stats[cls]
    n = s["n"]
    return (
        f"presence {100*(s['pos_ok']+s['pos_bad'])/n:.0f}% "
        f"(correct {100*s['pos_ok']/n:.0f}%, misclassified {100*s['pos_bad']/n:.0f}%), "
        f"absence {100*(s['neg_ok']+s['neg_bad'])/n:.0f}% "
        f"(correct {100*s['neg_ok']/n:.0f}%, misclassified {100*s['neg_bad']/n:.0f}%)"
    )


mp_note = _pa_dedupe_note(most_presence_cls)
ma_note = _pa_dedupe_note(most_absence_cls)
la_note = _pa_dedupe_note(lowest_acc_cls)

key_findings_lines = [
    f"{most_presence_cls}{mp_note}: the most presence-driven class in this run — "
    f"{_pa_desc(most_presence_cls)}.",
]
if ma_note:
    key_findings_lines.append(
        f"{most_absence_cls}{ma_note}: also the most absence-driven class in this run "
        f"(same underlying flow statistics as above)."
    )
else:
    key_findings_lines.append(
        f"{most_absence_cls}: the most absence-driven class in this run — "
        f"{_pa_desc(most_absence_cls)}."
    )
if la_note:
    key_findings_lines.append(
        f"{lowest_acc_cls}{la_note} ({_acc(lowest_acc_cls):.0f}% accuracy): also the "
        f"lowest-accuracy class in this run (same underlying flow statistics as above)."
    )
else:
    key_findings_lines.append(
        f"{lowest_acc_cls} ({_acc(lowest_acc_cls):.0f}% accuracy): the "
        f"lowest-accuracy class in this run — {_pa_desc(lowest_acc_cls)}."
    )
key_findings_block = "\n\n".join(key_findings_lines)

# Aggregate correctness-vs-polarity check across all classes' flows in this run.
_agg_pos_ok  = sum(pa_stats[c]["pos_ok"]  for c in classes_sorted)
_agg_pos_bad = sum(pa_stats[c]["pos_bad"] for c in classes_sorted)
_agg_neg_ok  = sum(pa_stats[c]["neg_ok"]  for c in classes_sorted)
_agg_neg_bad = sum(pa_stats[c]["neg_bad"] for c in classes_sorted)
_agg_n_ok    = _agg_pos_ok + _agg_neg_ok
_agg_n_bad   = _agg_pos_bad + _agg_neg_bad
_presence_frac_ok  = _agg_pos_ok  / _agg_n_ok  if _agg_n_ok  else float("nan")
_presence_frac_bad = _agg_pos_bad / _agg_n_bad if _agg_n_bad else float("nan")
correctness_correlates = (
    not np.isnan(_presence_frac_ok) and not np.isnan(_presence_frac_bad)
    and _presence_frac_ok > _presence_frac_bad
)

if correctness_correlates:
    correlation_sentence = (
        "The polarity breakdown shows that explanation quality (positive Fidelity+) "
        f"is associated with prediction correctness in this run: correctly classified "
        f"flows are presence-driven {100*_presence_frac_ok:.0f}% of the time, versus "
        f"{100*_presence_frac_bad:.0f}% for misclassified flows."
    )
else:
    correlation_sentence = (
        "No clear association between explanation polarity and prediction correctness "
        "is observed in this run."
    )

exception_sentence = (
    f"{lowest_acc_cls} has the lowest accuracy in this run "
    f"({_acc(lowest_acc_cls):.0f}%, {pa_stats[lowest_acc_cls]['n']} flows)."
)

if most_presence_cls == most_absence_cls:
    caption_class_sentence = (
        f"The model produces a single dominant polarity in this run."
    )
else:
    caption_class_sentence = (
        f"Correctly classified flows are predominantly presence-driven for "
        f"{most_presence_cls}, and predominantly absence-driven for {most_absence_cls}."
    )

(OUT_DIR / "fidelity_pa_bars.txt").write_text(f"""\
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
{key_findings_block}

PAPER FRAMING
-------------
"{correlation_sentence} {exception_sentence}"

SUGGESTED FIGURE CAPTION
-------------------------
Fidelity+ polarity × prediction correctness per class (n ≤ 200 flows, sorted by mean
Fidelity+ ascending). Bars above zero = presence-driven (positive Fidelity+); bars
below zero = absence-driven (negative Fidelity+). Within each direction, dark shading
= correctly classified, light shading = misclassified. {caption_class_sentence}
""")

print("\nDone.")
