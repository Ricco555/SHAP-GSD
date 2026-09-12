"""
attribution_decomp.py — Figure 4(c)
====================================
What it shows:
  Left panel:  Stacked horizontal bars — per-class fraction of total |φ| split into
               φ_F (feature groups), φ_T (temporal neighbors), φ_N (node novelty/structure).
  Right panel: Violin of absolute φ_T per class.

This script is reused across datasets, so the figure annotations and the
.txt reasoning deliberately report computed values only; they do not
assert an interpretation (e.g. "null result", "dataset property"), since
the actual φ_T magnitude is expected to vary per dataset.

Panels:
  fig, (ax1, ax2)  — 1 row × 2 columns, figsize=(13, 5.5)

Files read:
  outputs/explanations/<class>/*.json   — raw SHAP-GSD explanation records

Files output:
  outputs/figures/graph/attribution_decomp.{pdf,png,txt}
"""

import matplotlib
matplotlib.use("Agg")

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from explore._paths import paths  # noqa: E402
_P = paths()
EXP_DIR = _P["explanations"]
OUT_DIR = _P["figures"] / "graph"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEM = "attribution_decomp"
LABEL_FS = 11
_PHI_F = "#2a9d8f"
_PHI_T = "#e9c46a"
_PHI_N = "#e76f51"
_GRAY  = "#888888"


def load_class_data(cls_dir: Path) -> dict:
    """Return per-flow phi_f, phi_t, phi_n totals for one class."""
    phi_f_list, phi_t_list, phi_n_list = [], [], []
    for jf in sorted(cls_dir.glob("[0-9]*.json")):
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        phi_f = sum(abs(v) for v in d.get("feature_group_shap", []))
        phi_t = sum(abs(v) for v in d.get("neighbor_shap", []))
        phi_n = (sum(abs(v) for v in d.get("node_shap", []))
                 + abs(d.get("src_novelty_shap", 0.0))
                 + abs(d.get("dst_novelty_shap", 0.0)))
        phi_f_list.append(phi_f)
        phi_t_list.append(phi_t)
        phi_n_list.append(phi_n)
    return {
        "phi_f": np.array(phi_f_list),
        "phi_t": np.array(phi_t_list),
        "phi_n": np.array(phi_n_list),
    }


def main() -> None:
    class_data = {}
    for cls_dir in sorted(EXP_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        class_data[cls_dir.name] = load_class_data(cls_dir)

    if not class_data:
        print("ERROR: no explanation JSON found under", EXP_DIR)
        sys.exit(1)

    # Compute per-class mean fractions
    rows = []  # (cls, frac_f, frac_t, frac_n, mean_phi_t)
    empty_classes = []
    for cls, d in class_data.items():
        if d["phi_f"].shape[0] == 0:
            empty_classes.append(cls)
            print(f"  {cls}: 0 explained flows — SKIPPED")
            continue
        totals = d["phi_f"] + d["phi_t"] + d["phi_n"]
        safe = np.where(totals > 0, totals, 1.0)
        frac_f = (d["phi_f"] / safe).mean()
        frac_t = (d["phi_t"] / safe).mean()
        frac_n = (d["phi_n"] / safe).mean()
        rows.append((cls, frac_f, frac_t, frac_n, d["phi_t"].mean()))
    if empty_classes:
        print(f"  NOTE: {len(empty_classes)} class(es) skipped for zero "
              f"explained flows: {', '.join(empty_classes)}")

    # Sort by φ_T fraction descending
    rows.sort(key=lambda r: r[2], reverse=True)
    classes   = [r[0] for r in rows]
    frac_f    = np.array([r[1] for r in rows])
    frac_t    = np.array([r[2] for r in rows])
    frac_n    = np.array([r[3] for r in rows])
    mean_phi_t = np.array([r[4] for r in rows])

    # ------------------------------------------------------------------ figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.subplots_adjust(wspace=0.55)

    # --- Left: stacked horizontal bar ---
    y = np.arange(len(classes))
    ax1.barh(y, frac_f, color=_PHI_F, label=r"$\varphi_F$ (feature groups)")
    ax1.barh(y, frac_t, left=frac_f, color=_PHI_T, label=r"$\varphi_T$ (temporal)")
    ax1.barh(y, frac_n, left=frac_f + frac_t, color=_PHI_N, label=r"$\varphi_N$ (node/structure)")

    # Per-class absolute-value annotations
    for i, (cls, ff, ft, fn, _) in enumerate(rows):
        label = rf"$\varphi_F={ff:.2f}$  $\varphi_T={ft:.3f}$  $\varphi_N={fn:.2f}$"
        ax1.text(0.02, y[i], label, va="center", fontsize=9,
                 color="white", transform=ax1.get_yaxis_transform())

    ax1.set_yticks(y)
    ax1.set_yticklabels(classes, fontsize=LABEL_FS)
    ax1.set_xlabel("Fraction of total |φ|", fontsize=LABEL_FS)
    ax1.set_xlim(0, 1.0)
    ax1.tick_params(labelsize=LABEL_FS)
    ax1.legend(fontsize=LABEL_FS - 1, loc="lower right")
    ax1.set_title("Attribution decomposition by granularity", fontsize=LABEL_FS + 1)

    # Textbox: computed mean φ_T fraction (no interpretation asserted)
    overall_ft_pct = frac_t.mean() * 100
    ax1.text(0.02, 0.02,
             rf"mean $\varphi_T$ fraction across classes: {overall_ft_pct:.3f}%",
             transform=ax1.transAxes, fontsize=7.5, va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=_GRAY, alpha=0.85))

    if empty_classes:
        ax1.text(0.02, 0.98,
                 f"skipped (0 flows): {', '.join(sorted(empty_classes))}",
                 transform=ax1.transAxes, fontsize=7, va="top",
                 color="#b23b3b",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white",
                           ec="#b23b3b", alpha=0.85))

    # --- Right: φ_T violin per class ---
    phi_t_per_class = [class_data[cls]["phi_t"] for cls in classes]

    parts = ax2.violinplot(phi_t_per_class, positions=np.arange(len(classes)),
                           vert=False, showmeans=False, showmedians=True,
                           widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor(_PHI_T)
        pc.set_alpha(0.6)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color(_GRAY)

    # Mean diamonds
    for i, phi_t_arr in enumerate(phi_t_per_class):
        ax2.scatter(phi_t_arr.mean(), i, marker="D", s=30, color=_PHI_N,
                    zorder=5, label="mean" if i == 0 else "")

    ax2.set_yticks(np.arange(len(classes)))
    ax2.set_yticklabels(classes, fontsize=LABEL_FS)
    ax2.tick_params(axis="y", pad=8)
    ax2.set_xlabel(r"Absolute $\varphi_T$ per flow", fontsize=LABEL_FS)
    ax2.tick_params(labelsize=LABEL_FS)
    ax2.set_title(r"Per-class $\varphi_T$ distribution", fontsize=LABEL_FS + 1)
    ax2.legend(fontsize=LABEL_FS - 1, loc="lower right")

    all_phi_t = np.concatenate(phi_t_per_class)
    pct_zero = 100.0 * float((all_phi_t == 0).mean())
    ax2.text(0.97, 0.97,
             rf"{pct_zero:.1f}% of flows: $\varphi_T = 0$" + "\n(no in-window neighbors)",
             transform=ax2.transAxes, fontsize=7.5, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=_GRAY, alpha=0.85))

    # ------------------------------------------------------------------ save
    fig.savefig(OUT_DIR / f"{STEM}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / f"{STEM}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / STEM}.{{pdf,png}}")

    # ------------------------------------------------------------------ txt
    lines = ["Figure reasoning — attribution_decomp",
             "=" * 40, ""]

    lines.append("KEY FINDINGS — per-class fractions (computed from data):")
    for cls, ff, ft, fn, _ in rows:
        lines.append(f"  {cls:12s}: phi_F={ff:.3f} ({ff*100:.1f}%)  "
                     f"phi_T={ft:.5f} ({ft*100:.3f}%)  "
                     f"phi_N={fn:.3f} ({fn*100:.1f}%)")
    if empty_classes:
        lines += [
            "",
            f"SKIPPED (0 explained flows): {', '.join(sorted(empty_classes))}",
        ]

    overall_ft = frac_t.mean()
    overall_fn = frac_n.mean()

    # LOCKED 2026-08-04 (coder_instructions_figure_determinism.md S2): phi_T
    # share of total attribution has three defensible denominators. PRIMARY
    # convention (locked): the conditional mean over only the phi_T-bearing
    # flows (nonzero phi_T) — this is the number that describes what phi_T's
    # magnitude looks like *when it fires*. Two secondaries are kept,
    # explicitly labelled: the flow-level mean over ALL explained flows
    # (dilutes the conditional mean by the zero-phi_T flows), and the
    # unweighted per-class mean (gives every class equal weight regardless of
    # its own flow count). Every emission of the two conditional statistics
    # carries its denominator in the same sentence, per the locked convention.
    all_phi_t_flat  = np.concatenate(phi_t_per_class)
    all_frac_t_flat = np.concatenate([
        class_data[cls]["phi_t"] / np.where(
            (class_data[cls]["phi_f"] + class_data[cls]["phi_t"] + class_data[cls]["phi_n"]) > 0,
            (class_data[cls]["phi_f"] + class_data[cls]["phi_t"] + class_data[cls]["phi_n"]),
            1.0,
        )
        for cls in classes
    ])
    n_flows_total = len(all_frac_t_flat)
    bearing_mask = all_phi_t_flat > 0
    n_bearing = int(bearing_mask.sum())
    conditional_ft_pct = (float(all_frac_t_flat[bearing_mask].mean()) * 100
                          if n_bearing > 0 else 0.0)
    flow_level_ft_pct = float(all_frac_t_flat.mean()) * 100

    pct_zero = 100.0 * float((all_phi_t_flat == 0).mean())
    lines += [
        "",
        f"phi_T share of total attribution (PRIMARY, conditional on the {n_bearing} "
        f"phi_T-bearing flows of {n_flows_total} total): {conditional_ft_pct:.2f}%.",
        f"  Secondary — flow-level mean over all {n_flows_total} flows: {flow_level_ft_pct:.3f}%.",
        f"  Secondary — unweighted mean of the {len(classes)} per-class means: {overall_ft*100:.3f}%.",
        f"phi_N: mean fraction across classes = {overall_fn*100:.1f}% (range {frac_n.min()*100:.0f}-{frac_n.max()*100:.0f}%).",
        f"phi_T == 0 for {pct_zero:.1f}% of all explained flows (no in-window temporal neighbors).",
        "phi_N measures GNN computation subgraph node contribution (masking computation",
        "nodes from the coalition alters the predicted logit).",
        "",
        "WHAT THE FIGURE SHOWS",
        "----------------------",
        "Left panel: stacked horizontal bars show the per-class mean fraction of total |φ|",
        "attributed to the three SHAP-GSD granularities (feature groups φ_F, temporal",
        "neighbors φ_T, node state/structure φ_N). Right panel: violin plot of absolute",
        "φ_T per flow per class.",
        "",
        "GENERATOR NOTE",
        "---------------",
        "The numbers above are computed directly from this run's explanation JSONs and are",
        "reported without an asserted interpretation (e.g. whether phi_T's magnitude",
        "constitutes a null result or a meaningful signal, or whether that is a property of",
        "this dataset vs. an artifact) — this script is reused across datasets, and the",
        "correct framing depends on the actual numbers for each specific run.",
        "",
        "SUGGESTED FIGURE CAPTION",
        "-------------------------",
        "Attribution decomposition across SHAP-GSD granularities for all attack classes.",
        "Left: mean fraction of total |φ| assigned to feature-group φ_F (teal), temporal",
        "φ_T (amber), and node-structural φ_N (coral) components per class. Annotations",
        "show absolute mean values. Right: distribution of absolute φ_T per flow.",
    ]

    txt_path = OUT_DIR / f"{STEM}.txt"
    txt_path.write_text("\n".join(lines))
    print(f"Saved {txt_path}")


if __name__ == "__main__":
    main()
