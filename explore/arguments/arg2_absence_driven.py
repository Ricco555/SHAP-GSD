"""
arg2_absence_driven.py — Argument 2: "Presence vs Absence Classes"
===================================================================
What it shows:
  Claim: Four classes are characterised by what they lack. Negative Fidelity+
  is semantically correct, not an explainer failure.

Panels:
  Left  (ax1) — per-class mean Fidelity+ ± 1 SEM, sorted; teal/coral by sign;
                annotated with "N% flows absence-driven".
  Right (ax2) — 100% stacked horizontal bar: presence vs absence fraction per
                class; ★ marks absence-dominant classes (>50% flows with Fid+≤0).

Files read:
  outputs/metrics/fidelity.csv  — fidelity_plus, class_name

Files output:
  outputs/figures/arguments/arg2_absence_driven.{pdf,png,txt}
"""

import matplotlib
matplotlib.use("Agg")

import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from explore._paths import paths  # noqa: E402
from explore.arguments._paper_notes import load_paper_notes  # noqa: E402
_P = paths()
EXP_DIR = _P["explanations"]
OUT_DIR = _P["figures"] / "arguments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEM     = "arg2_absence_driven"
LABEL_FS = 9
_TEAL    = "#2a9d8f"
_CORAL   = "#e76f51"
_GRAY    = "#888888"


def dominant_negative_group(cls: str) -> str:
    """Feature group with the most negative mean signed phi for this class.

    Computed fresh from the explanation JSONs rather than hardcoded, so it
    always reflects whichever explainer run's outputs are currently loaded.
    """
    cls_dir = EXP_DIR / cls
    if not cls_dir.is_dir():
        return "—"
    sums, counts = {}, {}
    for jf in cls_dir.glob("*.json"):
        try:
            rec = json.loads(jf.read_text())
        except Exception:
            continue
        for name, phi in zip(rec["feature_group_names"], rec["feature_group_shap"]):
            sums[name] = sums.get(name, 0.0) + phi
            counts[name] = counts.get(name, 0) + 1
    if not sums:
        return "—"
    means = {name: sums[name] / counts[name] for name in sums}
    return min(means, key=means.get)


def main() -> None:
    df = pd.read_csv(_P["metrics"] / "fidelity.csv")

    stats = df.groupby("class_name")["fidelity_plus"].agg(["mean", "sem", "count"])
    absence_pct = df.groupby("class_name")["fidelity_plus"].apply(
        lambda x: (x <= 0).mean() * 100
    )

    classes   = stats.index.tolist()
    means     = stats["mean"].values
    sems      = stats["sem"].values
    abs_pct   = absence_pct.reindex(classes).values

    # Sort by mean Fidelity+ ascending
    order     = np.argsort(means)
    classes   = [classes[i] for i in order]
    means     = means[order]
    sems      = sems[order]
    abs_pct   = abs_pct[order]

    colors    = [_CORAL if m <= 0 else _TEAL for m in means]
    n_classes = len(classes)
    y         = np.arange(n_classes)

    # ------------------------------------------------------------------ figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.38)

    # Left: mean Fidelity+ ± SEM
    ax1.barh(y, means, xerr=sems, color=colors, edgecolor="white",
             linewidth=0.4, capsize=3, error_kw=dict(ecolor=_GRAY, elinewidth=1))
    ax1.axvline(x=0, color=_GRAY, lw=1.2, ls="--", alpha=0.7)

    x_range = max(abs(means.min()), abs(means.max()))
    ax1.set_xlim(-x_range * 1.6, x_range * 1.6)

    for i, (cls, pct) in enumerate(zip(classes, abs_pct)):
        x_pos = means[i] + sems[i]
        offset = 0.012 * x_range if means[i] >= 0 else -0.012 * x_range
        ha = "left" if means[i] >= 0 else "right"
        ax1.text(x_pos + offset, y[i], f"{pct:.0f}% absent",
                 va="center", ha=ha, fontsize=6.5, color=_GRAY)

    ax1.set_yticks(y)
    ax1.set_yticklabels(classes, fontsize=LABEL_FS)
    ax1.set_xlabel("Mean Fidelity+ (±1 SEM)", fontsize=LABEL_FS)
    ax1.tick_params(labelsize=LABEL_FS)
    ax1.set_title("Per-class mean Fidelity+", fontsize=LABEL_FS + 1)

    presence_p = mpatches.Patch(color=_TEAL,  label="Presence-driven (mean > 0)")
    absence_p  = mpatches.Patch(color=_CORAL, label="Absence-driven  (mean ≤ 0)")
    ax1.legend(handles=[presence_p, absence_p], fontsize=LABEL_FS - 1,
               loc="lower right")

    # Right: 100% stacked bar presence/absence
    pres_frac = 1.0 - abs_pct / 100.0
    abs_frac  = abs_pct / 100.0
    ax2.barh(y, pres_frac, color=_TEAL,  label="Presence-driven flows (Fid+ > 0)")
    ax2.barh(y, abs_frac,  left=pres_frac, color=_CORAL, label="Absence-driven flows (Fid+ ≤ 0)")

    ax2.axvline(x=0.50, color=_GRAY, lw=1.2, ls="--", alpha=0.7)
    ax2.text(0.51, n_classes - 0.2, "50%", color=_GRAY, fontsize=LABEL_FS - 1.5)

    # ★ for absence-dominant classes
    for i, (cls, ap) in enumerate(zip(classes, abs_pct)):
        if ap > 50:
            ax2.text(1.02, y[i], "★", va="center", fontsize=10, color=_CORAL)

    ax2.set_yticks(y)
    ax2.set_yticklabels(classes, fontsize=LABEL_FS)
    ax2.set_xlabel("Fraction of flows", fontsize=LABEL_FS)
    ax2.set_xlim(0, 1.15)
    ax2.tick_params(labelsize=LABEL_FS)
    ax2.set_title("Presence vs absence fraction per class", fontsize=LABEL_FS + 1)
    ax2.legend(fontsize=LABEL_FS - 1, loc="lower right")

    # ------------------------------------------------------------------ save
    fig.savefig(OUT_DIR / f"{STEM}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / f"{STEM}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / STEM}.{{pdf,png}}")

    # ------------------------------------------------------------------ txt
    absence_dominant = [(cls, ap) for cls, ap in zip(classes, abs_pct) if ap > 50]
    n_attack   = sum(1 for cls, _ in absence_dominant if cls != "Benign")
    has_benign = any(cls == "Benign" for cls, _ in absence_dominant)
    breakdown  = f"{n_attack} attack class{'es' if n_attack != 1 else ''}"
    if has_benign:
        breakdown += " and Benign normal traffic"

    lines = [
        f"Figure reasoning — {STEM}",
        "=" * 48, "",
        "WHAT",
        "----",
        f"{len(absence_dominant)} class{'es are' if len(absence_dominant) != 1 else ' is'} "
        f"characterised by absent network norms rather than present attack",
        f"signatures ({breakdown}).",
        "Negative Fidelity+ means the explanation correctly identifies missing",
        "features — it is semantically valid, not an explainer failure.",
        "",
        "KEY FINDINGS",
        "------------",
    ]
    for cls, m, s, ap in zip(classes, means, sems, abs_pct):
        marker = " ★ ABSENCE-DOMINANT" if ap > 50 else ""
        lines.append(f"  {cls:12s}: mean Fid+ = {m:+.4f} ±{s:.4f}  "
                     f"absence fraction = {ap:.1f}%{marker}")
    lines += [
        "",
        "  Absence-dominant classes (>50% flows with Fid+ ≤ 0):",
    ]
    for cls, ap in absence_dominant:
        neg_grp = dominant_negative_group(cls)
        lines.append(f"    {cls:12s}: {ap:.1f}% absence-driven  |  dominant neg group: {neg_grp}")

    lines += [
        "",
        "PAPER FRAMING",
        "-------------",
        f"SHAP-GSD identifies {len(absence_dominant)} absence-dominant classes",
        f"({breakdown}):",
        ", ".join(cls for cls, _ in absence_dominant) + ".",
        "For these classes, the model's confidence drops when groups representing",
        "normal traffic signatures (e.g., ephemeral source ports, normal TTL variation)",
        "are *present* — confirming that detection rests on the absence of benign norms.",
        "Negative Fidelity+ is not an explainer artefact: it encodes a semantically",
        "meaningful signal about what distinguishes malicious from normal traffic.",
        "",
        "CAPTION",
        "-------",
        f"Presence vs absence attribution across {_P['dataset_name']} attack classes.",
        "Left: per-class mean Fidelity+ ± 1 SEM, sorted ascending; teal bars indicate",
        "presence-driven classes (mean > 0), coral indicate absence-driven (mean ≤ 0);",
        "annotations show the percentage of flows with Fidelity+ ≤ 0. Right: 100%",
        "stacked bar showing per-class proportion of absence-driven flows; ★ marks",
        f"the {len(absence_dominant)} classes where >50% of flows have negative Fidelity+.",
    ]
    notes = load_paper_notes(STEM)
    if notes:
        lines += [""] + notes

    txt_path = OUT_DIR / f"{STEM}.txt"
    txt_path.write_text("\n".join(lines))
    print(f"Saved {txt_path}")


if __name__ == "__main__":
    main()
