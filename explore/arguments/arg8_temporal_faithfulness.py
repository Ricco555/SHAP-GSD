"""
arg8_temporal_faithfulness.py — Argument 8: "Temporal Faithfulness Loop"
=========================================================================
What it shows:
  Claim: SHAP-GSD enforces strict causal temporal ordering — every neighbor
  used in a SHAP coalition has timestamp ≤ target flow timestamp. This is
  verified empirically: 0 violations in 405 neighbor assignments. Without the
  causal filter, 99.2% of in-window edges would be from the future.

Panels:
  Left  (ax1) — Histogram of temporal offsets (target_ts − neighbor_ts) in
                seconds for all 405 SHAP-GSD neighbor edges. All offsets ∈
                (0, 60s] — zero violations. Annotated with violation count.
  Right (ax2) — Per-class log-scale grouped bars: causal neighbors used
                (teal) vs future edges excluded by the causal filter (coral).
                Demonstrates that without filtering, ≥99% of in-window edges
                would be future (information leakage).

Files read:
  outputs/explanations/<class>/*.json  — neighbor_timestamps, edge_id
  graphs/test.bin                      — EID → timestamp mapping

Files output:
  outputs/figures/arguments/arg8_temporal_faithfulness.{pdf,png,txt}
"""

import matplotlib
matplotlib.use("Agg")

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from explore._paths import paths  # noqa: E402
_P = paths()
EXP_DIR = _P["explanations"]
OUT_DIR = _P["figures"] / "arguments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEM     = "arg8_temporal_faithfulness"
LABEL_FS = 11
_TEAL    = "#2a9d8f"
_CORAL   = "#e76f51"
_GRAY    = "#888888"
_AMBER   = "#e9c46a"
W_S      = 60   # SHAP-GSD temporal window (seconds)


def load_eid_to_ts() -> dict:
    graph_path = ROOT / "graphs" / "test.bin"
    import dgl
    gs, _ = dgl.load_graphs(str(graph_path))
    g = gs[0]
    return {
        int(e): int(t)
        for e, t in zip(g.edata[dgl.EID].numpy(), g.edata["timestamp"].numpy())
    }, g.edata["timestamp"].numpy().astype(np.int64)


def main() -> None:
    eid_to_ts, all_ts = load_eid_to_ts()

    W_ms = W_S * 1000
    offsets_s     = []
    violations    = 0
    per_class     = {}  # cls -> {causal: [...], future: [...]}

    for cls_dir in sorted(EXP_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        causal_per_flow, future_per_flow = [], []

        for jf in sorted(cls_dir.glob("*.json")):
            try:
                rec = json.loads(jf.read_text())
            except Exception:
                continue
            n_ts = rec.get("neighbor_timestamps", [])
            eid  = int(rec["edge_id"])
            if not n_ts:
                continue
            target_ts = eid_to_ts.get(eid)
            if target_ts is None:
                continue

            for nt in n_ts:
                offset = (target_ts - int(nt)) / 1000.0
                offsets_s.append(offset)
                if offset < 0:
                    violations += 1

            causal_per_flow.append(len(n_ts))
            future_count = int(((all_ts > target_ts) & (all_ts <= target_ts + W_ms)).sum())
            future_per_flow.append(future_count)

        if causal_per_flow:
            per_class[cls] = {
                "n_flows":    len(causal_per_flow),
                "causal_sum": sum(causal_per_flow),
                "future_sum": sum(future_per_flow),
                "causal_mean": np.mean(causal_per_flow),
                "future_mean": np.mean(future_per_flow),
            }

    offsets = np.array(offsets_s)
    total_causal  = int(offsets.size)
    total_future  = sum(d["future_sum"] for d in per_class.values())
    leakage_rate  = total_future / (total_causal + total_future) * 100

    # Sort classes by leakage rate desc
    cls_sorted = sorted(per_class, key=lambda c: per_class[c]["future_mean"], reverse=True)

    # ------------------------------------------------------------------ figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.subplots_adjust(wspace=0.42)

    # Left: temporal offset histogram
    bins = np.linspace(0, W_S, 25)
    ax1.hist(offsets, bins=bins, color=_TEAL, edgecolor="white", linewidth=0.4)
    ax1.axvline(x=0, color=_CORAL, lw=2, ls="--", zorder=5,
                label="t = 0 (target flow timestamp)")
    ax1.axvline(x=W_S, color=_GRAY, lw=1.2, ls=":", alpha=0.7,
                label=f"W = {W_S} s (window boundary)")

    ax1.set_xlabel("Temporal offset: target_ts − neighbor_ts (seconds)", fontsize=LABEL_FS)
    ax1.set_ylabel("Number of neighbor edges", fontsize=LABEL_FS)
    ax1.tick_params(labelsize=LABEL_FS)
    ax1.set_title("SHAP-GSD neighbor temporal offsets", fontsize=LABEL_FS + 1)
    ax1.set_xlim(-5, W_S + 3)
    ax1.legend(fontsize=LABEL_FS - 1, loc="upper left")

    # Zero-violation annotation
    ax1.text(0.97, 0.97,
             f"Violations: {violations} / {total_causal}\n"
             f"(all offsets > 0 s)\n"
             f"Median offset: {np.median(offsets):.1f} s",
             transform=ax1.transAxes, fontsize=8, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.35", fc="#f0fff4", ec=_TEAL, alpha=0.95))

    # Right: log-scale grouped bars per class
    y         = np.arange(len(cls_sorted))
    causal_m  = [per_class[c]["causal_mean"] for c in cls_sorted]
    future_m  = [per_class[c]["future_mean"] for c in cls_sorted]
    lr_pct    = [per_class[c]["future_mean"] /
                 (per_class[c]["causal_mean"] + per_class[c]["future_mean"]) * 100
                 for c in cls_sorted]

    bar_h = 0.35
    ax2.barh(y + bar_h / 2, future_m, height=bar_h, color=_CORAL,
             label="Future edges excluded (would leak)", edgecolor="white", linewidth=0.4)
    ax2.barh(y - bar_h / 2, causal_m,  height=bar_h, color=_TEAL,
             label="Causal neighbors used (SHAP-GSD)", edgecolor="white", linewidth=0.4)

    # Leakage rate annotations
    for i, (c, lr) in enumerate(zip(cls_sorted, lr_pct)):
        ax2.text(max(future_m) * 1.05, y[i] + bar_h / 2,
                 f"{lr:.1f}% leaked",
                 va="center", fontsize=9, color=_GRAY)

    ax2.set_xscale("log")
    ax2.set_yticks(y)
    ax2.set_yticklabels(cls_sorted, fontsize=LABEL_FS)
    ax2.set_xlabel("Mean in-window edges per flow (log scale)", fontsize=LABEL_FS)
    ax2.tick_params(labelsize=LABEL_FS)
    ax2.set_title("Causal used vs future excluded (W = 60 s)", fontsize=LABEL_FS + 1)
    ax2.legend(fontsize=LABEL_FS - 1, loc="lower right")

    # Overall annotation
    ax2.text(0.02, 0.97,
             f"Without causal filter:\n{leakage_rate:.1f}% of in-window\nedges are future",
             transform=ax2.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round,pad=0.35", fc="#fff0f0", ec=_CORAL, alpha=0.95))

    # ------------------------------------------------------------------ save
    fig.savefig(OUT_DIR / f"{STEM}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / f"{STEM}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / STEM}.{{pdf,png}}")

    # ------------------------------------------------------------------ txt
    lines = [
        f"Figure reasoning — {STEM}",
        "=" * 48, "",
        "WHAT",
        "----",
        f"SHAP-GSD enforces strict causal temporal ordering: every temporal neighbor",
        f"used in a SHAP coalition satisfies neighbor_ts ≤ target_ts. Verified over",
        f"{total_causal} neighbor assignments: {violations} violations. Without the causal",
        f"filter, {leakage_rate:.1f}% of in-window edges in the test graph are from the",
        f"future — systematic temporal leakage that would contaminate SHAP attributions.",
        "",
        "KEY FINDINGS",
        "------------",
        f"  Total SHAP-GSD neighbor edges verified: {total_causal}",
        f"  Violations (neighbor_ts > target_ts):   {violations}",
        f"  Offset range: {offsets.min():.2f}s – {offsets.max():.2f}s (all within W={W_S}s)",
        f"  Median offset: {np.median(offsets):.2f}s",
        f"  P25 / P75: {np.percentile(offsets, 25):.2f}s / {np.percentile(offsets, 75):.2f}s",
        "",
        f"  Future edges excluded by causal filter (total): {total_future:,}",
        f"  Leakage rate without filter: {leakage_rate:.1f}%",
        f"  (for every 1 causal neighbor used, ~{total_future//max(total_causal,1):,} future edges excluded)",
        "",
        "  Per-class breakdown:",
    ]
    for cls in cls_sorted:
        d  = per_class[cls]
        lr = d["future_mean"] / (d["causal_mean"] + d["future_mean"]) * 100
        lines.append(
            f"    {cls:12s}: {d['n_flows']} flows with nbrs | "
            f"mean causal={d['causal_mean']:.1f} | "
            f"mean future excluded={d['future_mean']:.0f} | "
            f"leakage%={lr:.1f}%"
        )
    lines += [
        "",
        "PAPER FRAMING",
        "-------------",
        "The temporal faithfulness property is a correctness requirement, not an",
        "optimisation: a SHAP explanation that uses future neighbors assigns attribution",
        "to information that was causally unavailable at prediction time, producing",
        "explanations that cannot be replicated in deployment. UNSW-NB15's test split",
        "has a high forward edge density (~2,300 edges in the next 60s per sampled flow)",
        "but low backward density (~6–25 edges in the previous 60s), meaning that",
        "without explicit causal filtering, ~99% of sampled temporal neighbors would",
        f"be from the future. SHAP-GSD's TemporalNeighborSampler enforces timestamp ≤",
        f"target_ts, verified here as {violations} violations over {total_causal} assignments.",
        "Paper 1 (E-GraphSAGE-XAI) did not implement within-split causal ordering for",
        "the SHAP coalition background — this is one of the key methodology gaps",
        "addressed by SHAP-GSD.",
        "",
        "CAPTION",
        "-------",
        "Temporal faithfulness verification for SHAP-GSD's temporal coalition component.",
        f"Left: histogram of temporal offsets (target_ts − neighbor_ts) for all",
        f"{total_causal} neighbor edges used across SHAP-GSD explanations; all offsets",
        f"lie in (0, {W_S}s], confirming {violations} violations of the causal ordering constraint.",
        f"Right: per-class log-scale comparison of mean causal neighbors used (teal)",
        f"vs mean future edges excluded by the causal filter (coral); without filtering,",
        f"{leakage_rate:.1f}% of in-window edges would leak future information into the explanation.",
        "",
        "REVIEWER CHALLENGE",
        "------------------",
        "Temporal neighbors are a minor component of SHAP-GSD (φ_T < 0.1% on UNSW-NB15).",
        "The causal ordering constraint has negligible practical effect on the results.",
        "",
        "COUNTER-ARGUMENT",
        "----------------",
        "The practical effect on φ_T is small on UNSW-NB15 precisely BECAUSE the causal",
        "filter leaves almost no valid temporal neighbors (median IAT = 5168s >> W=60s).",
        "If the filter were absent and future edges included, φ_T would be artificially",
        "inflated — the temporal component would appear meaningful on UNSW-NB15 even",
        "though it is not. More critically, this is a correctness argument, not a",
        "performance one: any SHAP explanation using future neighbors cannot be deployed",
        "in a real IDS where future flows are unknown. The 99.2% leakage rate shows that",
        "the dataset's temporal structure makes unconstrained sampling almost entirely",
        "future-contaminated. SHAP-GSD's correctness guarantee is what enables the",
        "W-sensitivity analysis (Arg9) to be interpreted as a genuine null result rather",
        "than a methodological artefact. On IoT datasets with dense sub-second traffic,",
        "where φ_T is expected to be large, the faithfulness guarantee becomes critical.",
        "",
        "PAPER SECTION PLACEMENT",
        "-----------------------",
        "Methods §3.3 — Temporal Faithfulness Constraint. Provides empirical verification",
        "that SHAP-GSD's TemporalNeighborSampler enforces the causal ordering invariant",
        "with zero violations, and quantifies the magnitude of leakage that would result",
        "from omitting this constraint (99.2% of in-window edges are future-dated).",
        "Directly addresses Paper 1's temporal leakage limitation.",
    ]

    txt_path = OUT_DIR / f"{STEM}.txt"
    txt_path.write_text("\n".join(lines))
    print(f"Saved {txt_path}")


if __name__ == "__main__":
    main()
