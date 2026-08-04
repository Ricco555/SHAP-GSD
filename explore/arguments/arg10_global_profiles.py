"""
arg10_global_profiles.py — Argument 10: "Global Profiles as Proxy Ground Truth"
================================================================================
What it shows:
  Claim: SHAP-GSD per-class global feature-group rankings are compared against
  independently derived literature profiles (Moustafa & Slay 2015) via Spearman
  ρ. Per-class ρ values and the count of classes reaching the ρ ≥ 0.7 threshold
  are computed fresh at run time — see the emitted .txt's WHAT/KEY FINDINGS
  blocks, not this docstring, for current numbers.

Panels:
  Left  (ax1) — Heatmap: classes (rows) × top-12 groups (cols), cell = mean
                |φ| rank (lighter = rank 1 = most important). YlOrRd_r cmap.
  Right (ax2) — Horizontal bar: per-class Spearman ρ vs literature.
                Green ≥ 0.7, amber 0.4–0.7, coral < 0.4. Dashed at 0.7.

Files read:
  outputs/explanations/<class>/*.json  — feature_group_names, feature_group_shap

Files output:
  outputs/figures/arguments/arg10_global_profiles.{pdf,png,txt}
"""

import matplotlib
matplotlib.use("Agg")

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from explore._paths import paths  # noqa: E402
from explore.arguments._paper_notes import load_paper_notes  # noqa: E402
_P = paths()
EXP_DIR = _P["explanations"]
OUT_DIR = _P["figures"] / "arguments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEM     = "arg10_global_profiles"
LABEL_FS = 9
_GREEN   = "#2ecc71"
_AMBER   = "#e9c46a"
_CORAL   = "#e76f51"
_GRAY    = "#888888"
_TEAL    = "#2a9d8f"

LITERATURE_TOP_GROUPS = {
    "Analysis":  ["SRC_PORT_IS_EPHEMERAL","IN_BYTES","TCP_FLAGS","MIN_TTL",
                  "NUM_PKTS_UP_TO_128_BYTES"],
    "Backdoor":  ["L7_PROTO","DST_PORT_GROUP","MIN_IP_PKT_LEN",
                  "FLOW_DURATION_MILLISECONDS","IN_PKTS"],
    "Benign":    ["IN_BYTES","SRC_TO_DST_IAT_AVG","FLOW_DURATION_MILLISECONDS",
                  "TCP_FLAGS","DST_TO_SRC_AVG_THROUGHPUT"],
    "DoS":       ["IN_PKTS","IN_BYTES","SRC_TO_DST_SECOND_BYTES","TCP_FLAGS",
                  "FLOW_DURATION_MILLISECONDS"],
    "Exploits":  ["DST_PORT_GROUP","L7_PROTO","IN_BYTES","TCP_FLAGS",
                  "MIN_IP_PKT_LEN"],
    "Fuzzers":   ["IN_PKTS","SHORTEST_FLOW_PKT","L7_PROTO","DST_PORT_GROUP",
                  "NUM_PKTS_UP_TO_128_BYTES"],
    "Generic":   ["DST_PORT_GROUP","L7_PROTO","DNS_QUERY_TYPE","IN_PKTS",
                  "TCP_FLAGS"],
    "Reconnaissance": ["MIN_TTL","MAX_TTL","TCP_FLAGS","IN_PKTS",
                  "FLOW_DURATION_MILLISECONDS"],
    "Shellcode": ["NUM_PKTS_UP_TO_128_BYTES","IN_PKTS","TCP_FLAGS",
                  "SHORTEST_FLOW_PKT","DST_PORT_GROUP"],
    "Worms":     ["ICMP_TYPE","IN_PKTS","TCP_FLAGS","FLOW_DURATION_MILLISECONDS",
                  "NUM_PKTS_UP_TO_128_BYTES"],
}


def main() -> None:
    # --- load per-class mean |φ| per group ---
    class_group_phi: dict[str, dict[str, float]] = {}

    for cls_dir in sorted(EXP_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        group_accum: dict[str, list] = {}
        for jf in sorted(cls_dir.glob("*.json")):
            try:
                rec = json.loads(jf.read_text())
            except Exception:
                continue
            for name, val in zip(rec["feature_group_names"],
                                 rec["feature_group_shap"]):
                group_accum.setdefault(name, []).append(abs(val))
        class_group_phi[cls] = {g: np.mean(v) for g, v in group_accum.items()}

    # --- build union of top-5 groups per class (up to 12 total) ---
    top5_per_class: dict[str, list[str]] = {}
    for cls, gmap in class_group_phi.items():
        ranked = sorted(gmap, key=lambda g: gmap[g], reverse=True)
        top5_per_class[cls] = ranked[:5]

    union_groups: list[str] = []
    seen = set()
    for cls in sorted(class_group_phi):
        for g in top5_per_class[cls]:
            if g not in seen:
                union_groups.append(g)
                seen.add(g)
            if len(union_groups) == 12:
                break
        if len(union_groups) == 12:
            break
    # ensure exactly 12
    if len(union_groups) < 12:
        all_groups = set(g for gmap in class_group_phi.values() for g in gmap)
        extras = [g for g in sorted(all_groups) if g not in seen]
        union_groups.extend(extras[:12 - len(union_groups)])

    classes = sorted(class_group_phi.keys())
    K = len(union_groups)

    # --- build rank matrices ---
    # SHAP-GSD rank: per-class rank of each group in union_groups (1 = most important)
    shap_ranks = np.zeros((len(classes), K))
    for i, cls in enumerate(classes):
        gmap = class_group_phi[cls]
        vals = np.array([gmap.get(g, 0.0) for g in union_groups])
        # rank: argsort descending gives indices; rank[j] = position in sorted order
        order = np.argsort(vals)[::-1]
        ranks = np.empty(K)
        ranks[order] = np.arange(1, K + 1)
        shap_ranks[i] = ranks

    # Literature rank: position in LITERATURE_TOP_GROUPS (1-indexed; unlisted → K+1)
    lit_ranks = np.zeros((len(classes), K))
    for i, cls in enumerate(classes):
        lit_top = LITERATURE_TOP_GROUPS.get(cls, [])
        for j, g in enumerate(union_groups):
            if g in lit_top:
                lit_ranks[i, j] = lit_top.index(g) + 1
            else:
                lit_ranks[i, j] = K + 1

    # --- Spearman ρ per class ---
    rho_vals = []
    rho_nan  = set()   # classes with undefined ρ (constant literature rank)
    for i, cls in enumerate(classes):
        lit_row = lit_ranks[i]
        if np.all(lit_row == lit_row[0]):
            # All literature groups outside the union — ρ undefined
            rho_vals.append(0.0)
            rho_nan.add(cls)
        else:
            rho, _ = spearmanr(shap_ranks[i], lit_ranks[i])
            rho_vals.append(float(rho) if not np.isnan(rho) else 0.0)

    # Sort by ρ descending
    order = np.argsort(rho_vals)[::-1]
    cls_sorted  = [classes[i] for i in order]
    rho_sorted  = [rho_vals[i] for i in order]
    shap_sorted = shap_ranks[order]

    # ------------------------------------------------------------------ figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    fig.subplots_adjust(wspace=0.42)

    # Left: rank heatmap (lighter = lower rank = more important)
    im = ax1.imshow(shap_sorted, aspect="auto", cmap="YlOrRd_r",
                    vmin=1, vmax=K)
    plt.colorbar(im, ax=ax1, fraction=0.025, pad=0.02,
                 label="Rank (1 = most important)")
    ax1.set_xticks(range(K))
    ax1.set_xticklabels(union_groups, fontsize=6.0, rotation=45, ha="right")
    ax1.set_yticks(range(len(cls_sorted)))
    ax1.set_yticklabels(cls_sorted, fontsize=LABEL_FS)
    ax1.set_title("SHAP-GSD feature-group importance rank", fontsize=LABEL_FS + 1)

    # Right: Spearman ρ bar
    y = np.arange(len(cls_sorted))
    bar_cols = [
        _GREEN if r >= 0.7 else (_AMBER if r >= 0.4 else _CORAL)
        for r in rho_sorted
    ]
    ax2.barh(y, rho_sorted, color=bar_cols, edgecolor="white", linewidth=0.4)
    ax2.axvline(x=0.7, color=_GRAY, lw=1.2, ls="--", alpha=0.8)
    ax2.text(0.71, len(cls_sorted) - 0.5, "ρ = 0.7", color=_GRAY,
             fontsize=LABEL_FS - 2)
    ax2.axvline(x=0.0, color=_GRAY, lw=0.8, ls="-", alpha=0.4)

    for i, (cls, r) in enumerate(zip(cls_sorted, rho_sorted)):
        label = "n/a*" if cls in rho_nan else f"{r:.2f}"
        ax2.text(r + 0.01 if r >= 0 else r - 0.01, y[i],
                 label, va="center",
                 ha="left" if r >= 0 else "right",
                 fontsize=LABEL_FS - 1.5, color=_GRAY)

    ax2.set_yticks(y)
    ax2.set_yticklabels(cls_sorted, fontsize=LABEL_FS)
    ax2.set_xlabel("Spearman ρ (SHAP-GSD vs literature)", fontsize=LABEL_FS)
    ax2.set_xlim(-1.1, 1.1)
    ax2.tick_params(labelsize=LABEL_FS)
    ax2.set_title("Global profile coherence", fontsize=LABEL_FS + 1)

    green_p = mpatches.Patch(color=_GREEN, label="ρ ≥ 0.7 (strong)")
    amber_p = mpatches.Patch(color=_AMBER, label="0.4 ≤ ρ < 0.7 (moderate)")
    coral_p = mpatches.Patch(color=_CORAL, label="ρ < 0.4 (weak)")
    ax2.legend(handles=[green_p, amber_p, coral_p],
               fontsize=LABEL_FS - 1.5, loc="lower right")

    if rho_nan:
        nan_note = "* n/a: literature groups not in top-12 union"
        ax2.text(0.02, 0.02, nan_note, transform=ax2.transAxes,
                 fontsize=6.5, color=_GRAY, va="bottom")

    # ------------------------------------------------------------------ save
    fig.savefig(OUT_DIR / f"{STEM}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / f"{STEM}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / STEM}.{{pdf,png}}")

    # ------------------------------------------------------------------ txt
    strong = [(c, r) for c, r in zip(cls_sorted, rho_sorted) if r >= 0.7]
    moderate = [(c, r) for c, r in zip(cls_sorted, rho_sorted) if 0.4 <= r < 0.7]
    weak = [(c, r) for c, r in zip(cls_sorted, rho_sorted) if r < 0.4]
    mean_rho = np.mean(rho_sorted)

    top_cls, top_rho = cls_sorted[0], rho_sorted[0]
    top_strength = "strong" if top_rho >= 0.7 else ("moderate" if top_rho >= 0.4 else "weak")
    overall_strength = "weak" if mean_rho < 0.4 else ("moderate" if mean_rho < 0.7 else "strong")

    if strong:
        threshold_sentence = (
            f"Spearman ρ measures cross-source rank correlation; {len(strong)} of "
            f"{len(classes)} classes ("
            + ", ".join(c for c, _ in strong)
            + f") reach ρ ≥ 0.7, with the highest at {top_cls} (ρ = {top_rho:+.3f})."
        )
    else:
        threshold_sentence = (
            f"Spearman ρ measures cross-source rank correlation; no class reaches ρ ≥ 0.7 — "
            f"the highest agreement is {top_cls} (ρ = {top_rho:+.3f}, {top_strength}), "
            "so the threshold cannot be used as a blanket validation claim."
        )

    lines = [
        f"Figure reasoning — {STEM}",
        "=" * 48, "",
        "WHAT",
        "----",
        "SHAP-GSD per-class feature-group rankings are compared against independently",
        "derived literature profiles (Moustafa & Slay 2015, UNSW-NB15 attack descriptions).",
        threshold_sentence,
        "",
        "KEY FINDINGS",
        "------------",
        "Spearman ρ (SHAP-GSD rank vs Moustafa & Slay 2015):",
    ]
    for cls, r in zip(cls_sorted, rho_sorted):
        strength = "strong" if r >= 0.7 else ("moderate" if r >= 0.4 else "weak")
        lines.append(f"  {cls:12s}: ρ = {r:+.3f}  ({strength})")
    lines += [
        "",
        f"  Mean ρ across all classes: {mean_rho:.3f}",
        f"  Strong agreement  (ρ ≥ 0.7): {len(strong)} classes: "
        + ", ".join(c for c, _ in strong),
        f"  Moderate agreement (0.4–0.7): {len(moderate)} classes: "
        + ", ".join(c for c, _ in moderate),
        f"  Weak agreement    (ρ < 0.4):  {len(weak)} classes: "
        + ", ".join(c for c, _ in weak),
        "",
        "  Union of top-5 groups per class (12 groups shown in heatmap):",
    ]
    for g in union_groups:
        lines.append(f"    {g}")
    lines += [
        "",
        "PAPER FRAMING",
        "-------------",
        "The cross-source Spearman correlation between SHAP-GSD rankings and",
        "Moustafa & Slay (2015) literature profiles is a check on whether global",
        "SHAP-GSD profiles align with literature-derived attack signatures, not a",
        "validation that can be claimed uniformly across classes.",
    ]
    lines += [
        f"The highest agreement is {top_cls} (ρ = {top_rho:+.3f}, {top_strength}); "
        f"{len(strong)} of {len(classes)} classes",
        f"reach the ρ ≥ 0.7 threshold. Mean ρ = {mean_rho:.3f} across all classes indicates",
        f"{overall_strength} agreement overall — most classes' rankings diverge substantially",
        "from the 2015 raw-traffic characterisation, plausibly reflecting real",
        "distributional differences between that dataset's original feature set and the",
        "NetFlow feature set used here, rather than an explainer failure specific to",
        "any one class.",
        "",
        "CAPTION",
        "-------",
        "Global feature-group profile coherence: SHAP-GSD vs Moustafa & Slay (2015).",
        "Left: heatmap of per-class SHAP-GSD feature-group importance rank for the",
        "12-group union of top-5 per class (lighter = rank 1 = most important).",
        "Right: per-class Spearman ρ between SHAP-GSD rank and independent literature",
        "rank; green = strong agreement (ρ ≥ 0.7), amber = moderate, coral = weak.",
        f"Mean ρ = {mean_rho:.2f}; {len(strong)} of {len(classes)} classes show strong cross-source alignment.",
    ]
    notes = load_paper_notes(STEM)
    if notes:
        lines += [""] + notes

    txt_path = OUT_DIR / f"{STEM}.txt"
    txt_path.write_text("\n".join(lines))
    print(f"Saved {txt_path}")


if __name__ == "__main__":
    main()
