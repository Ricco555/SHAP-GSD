"""
arg5_node_state.py — Argument 5: "Node-State SHAP and Temporal Context"
========================================================================
What it shows:
  Claim: φ_N (node-structural attribution) is non-trivial (13–61% of total |φ|)
  and the GNN computation subgraph encodes state that flat SHAP cannot see.

Panels:
  Left  (ax1) — Scatter: per-flow φ_N vs hour-of-day (from DGL timestamps).
                Top-5 classes by count. Alpha=0.3. Per-hour mean line.
                Controlled-lab caveat annotation if no diurnal pattern.
  Right (ax2) — Violin: per-class φ_N distribution, sort by mean desc.
                ◆ mean markers. Coral fill. "13–61% of total |φ|" textbox.

Files read:
  outputs/explanations/<class>/*.json  — node_shap, src/dst_novelty_shap, edge_id
  graphs/test.bin                      — EID→timestamp mapping

Files output:
  outputs/figures/arguments/arg5_node_state.{pdf,png,txt}
"""

import matplotlib
matplotlib.use("Agg")

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from explore._paths import paths  # noqa: E402
from explore.arguments._paper_notes import load_paper_notes  # noqa: E402
_P = paths()
EXP_DIR = _P["explanations"]
OUT_DIR = _P["figures"] / "arguments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEM     = "arg5_node_state"
LABEL_FS = 9
_PHI_N   = "#e76f51"   # coral — node/structural
_GRAY    = "#888888"

CLASS_COLORS = {
    "Generic":   "#4e79a7",
    "Benign":    "#59a14f",
    "Exploits":  "#f28e2b",
    "Reconnaissance": "#b07aa1",
    "DoS":       "#e15759",
    "Analysis":  "#76b7b2",
    "Backdoor":  "#ff9da7",
    "Fuzzers":   "#9c755f",
    "Shellcode": "#bab0ac",
    "Worms":     "#d37295",
}


def load_eid_to_hour() -> dict:
    """Return {edge_id: hour_utc} from DGL test graph timestamps."""
    graph_path = _P["graphs"] / "test.bin"
    if not graph_path.exists():
        return {}
    try:
        import dgl
        gs, _ = dgl.load_graphs(str(graph_path))
        g = gs[0]
        return {
            int(eid): datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc).hour
            for eid, ts in zip(g.edata[dgl.EID].numpy(),
                               g.edata["timestamp"].numpy())
        }
    except Exception as e:
        print(f"Warning: could not load DGL graph: {e}")
        return {}


def load_novelty_audit() -> dict:
    """Read Phase 12's canonical novelty non-zero audit (>1e-6 threshold)."""
    audit_path = _P["metrics"] / "novelty_audit.json"
    if not audit_path.exists():
        return {}
    return json.loads(audit_path.read_text()).get("explanation_json_audit", {})


def main() -> None:
    eid_to_hour = load_eid_to_hour()
    has_timestamps = bool(eid_to_hour)

    # Load per-flow phi_N per class
    class_phi_n: dict[str, list[float]] = {}
    class_hours: dict[str, list[int]]   = {}
    class_totals: dict[str, list[float]] = {}

    for cls_dir in sorted(EXP_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        phi_n_list, hour_list, total_list = [], [], []
        for jf in sorted(cls_dir.glob("*.json")):
            try:
                rec = json.loads(jf.read_text())
            except Exception:
                continue
            phi_n = (np.sum(np.abs(rec.get("node_shap", [])))
                     + abs(rec.get("src_novelty_shap", 0.0))
                     + abs(rec.get("dst_novelty_shap", 0.0)))
            phi_f = sum(abs(v) for v in rec.get("feature_group_shap", []))
            phi_t = sum(abs(v) for v in rec.get("neighbor_shap", []))
            total = phi_f + phi_t + phi_n
            phi_n_list.append(phi_n)
            total_list.append(total)
            eid = rec.get("edge_id", -1)
            hour_list.append(eid_to_hour.get(int(eid), -1) if has_timestamps else -1)
        class_phi_n[cls]   = phi_n_list
        class_hours[cls]   = hour_list
        class_totals[cls]  = total_list

    # Per-class mean phi_N fraction
    class_frac = {}
    for cls in class_phi_n:
        phi_n_arr = np.array(class_phi_n[cls])
        tot_arr   = np.array(class_totals[cls])
        safe      = np.where(tot_arr > 0, tot_arr, 1.0)
        class_frac[cls] = (phi_n_arr / safe).mean()

    # Sort classes by mean phi_N desc
    sorted_classes = sorted(class_phi_n, key=lambda c: np.mean(class_phi_n[c]), reverse=True)

    # Top-5 classes by n_flows for scatter
    top5 = sorted(class_phi_n, key=lambda c: len(class_phi_n[c]), reverse=True)[:5]

    # ------------------------------------------------------------------ figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.38)

    # Left: scatter phi_N vs hour-of-day (or index if no timestamps)
    if has_timestamps:
        ax1.set_xlabel("Hour of day (UTC)", fontsize=LABEL_FS)
        x_label = "hour-of-day"
    else:
        ax1.set_xlabel("Flow index (no timestamp data)", fontsize=LABEL_FS)
        x_label = "flow index"

    for cls in top5:
        phi_n_arr = np.array(class_phi_n[cls])
        if has_timestamps:
            hours = np.array(class_hours[cls])
            valid = hours >= 0
            x_vals = hours[valid]
            y_vals = phi_n_arr[valid]
        else:
            x_vals = np.arange(len(phi_n_arr))
            y_vals = phi_n_arr
        col = CLASS_COLORS.get(cls, _GRAY)
        ax1.scatter(x_vals, y_vals, alpha=0.25, s=8, color=col, label=cls, rasterized=True)
        # Per-hour mean
        if has_timestamps and len(x_vals) > 0:
            for h in range(24):
                mask = x_vals == h
                if mask.sum() >= 3:
                    ax1.scatter(h, y_vals[mask].mean(), marker="_",
                                s=60, color=col, linewidths=2, zorder=5)

    ax1.set_ylabel(r"$\varphi_N$ per flow", fontsize=LABEL_FS)
    ax1.tick_params(labelsize=LABEL_FS)
    ax1.set_title(r"$\varphi_N$ vs temporal context", fontsize=LABEL_FS + 1)
    ax1.legend(fontsize=LABEL_FS - 1.5, loc="upper right", markerscale=2)

    if has_timestamps:
        ax1.set_xlim(-0.5, 23.5)
        ax1.set_xticks(range(0, 24, 4))
        ax1.text(0.02, 0.97,
                 "Controlled lab env. (UNSW-NB15, 2 days)\n"
                 "Full diurnal pattern not recoverable.",
                 transform=ax1.transAxes, fontsize=7, va="top",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=_GRAY, alpha=0.85))
    else:
        ax1.text(0.5, 0.5, "DGL graph timestamps\nnot available",
                 transform=ax1.transAxes, ha="center", va="center",
                 fontsize=9, color=_GRAY,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=_GRAY))

    # Right: violin per class
    phi_n_arrays = [np.array(class_phi_n[cls]) for cls in sorted_classes]
    parts = ax2.violinplot(phi_n_arrays, positions=np.arange(len(sorted_classes)),
                           vert=False, showmeans=False, showmedians=True, widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor(_PHI_N)
        pc.set_alpha(0.6)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color(_GRAY)

    for i, cls in enumerate(sorted_classes):
        ax2.scatter(np.mean(class_phi_n[cls]), i, marker="D", s=25,
                    color="#264653", zorder=5, label="mean" if i == 0 else "")

    ax2.set_yticks(np.arange(len(sorted_classes)))
    ax2.set_yticklabels(sorted_classes, fontsize=LABEL_FS)
    ax2.set_xlabel(r"$\varphi_N$ per flow", fontsize=LABEL_FS)
    ax2.tick_params(labelsize=LABEL_FS)
    ax2.set_title(r"Per-class $\varphi_N$ distribution", fontsize=LABEL_FS + 1)
    ax2.legend(fontsize=LABEL_FS - 1, loc="lower right")

    frac_min = min(class_frac.values()) * 100
    frac_max = max(class_frac.values()) * 100
    ax2.text(0.97, 0.97,
             fr"$\varphi_N$ = {frac_min:.0f}–{frac_max:.0f}% of total |φ|" + "\n"
             r"(GNN computation subgraph)" + "\n"
             r"Flat SHAP: $\varphi_N \equiv 0$",
             transform=ax2.transAxes, fontsize=7.5, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=_GRAY, alpha=0.85))

    # ------------------------------------------------------------------ save
    fig.savefig(OUT_DIR / f"{STEM}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / f"{STEM}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / STEM}.{{pdf,png}}")

    # --- novelty non-zero rates (from Phase 12's canonical audit, >1e-6) ---
    novelty_audit = load_novelty_audit()
    nov_n_total    = novelty_audit.get("n_total", 0)
    nov_n_nonzero  = novelty_audit.get("n_nonzero_either", 0)
    nov_frac       = novelty_audit.get("frac_nonzero_either", 0.0) * 100
    nov_by_class   = novelty_audit.get("by_class", {})
    nov_class_pct  = {
        c: (v["n_nonzero"] / v["n_total"] * 100 if v.get("n_total") else 0.0)
        for c, v in nov_by_class.items()
    }
    nov_ranked = sorted(nov_class_pct.items(), key=lambda kv: kv[1], reverse=True)
    nov_top2_str = ", ".join(f"{c} {p:.1f}%" for c, p in nov_ranked[:2])
    nov_attack_mean = (
        np.mean([p for c, p in nov_class_pct.items() if c != "Benign"])
        if nov_class_pct else 0.0
    )

    # ------------------------------------------------------------------ txt
    lines = [
        f"Figure reasoning — {STEM}",
        "=" * 48, "",
        "WHAT",
        "----",
        "φ_N (node-structural attribution) captures the contribution of GNN",
        "computation subgraph context: per-node SHAP plus src/dst novelty SHAP.",
        "This is non-trivial on UNSW-NB15, proving that GNN context encodes signal",
        "that flat feature-only SHAP (φ_N ≡ 0) cannot see.",
        "",
        "KEY FINDINGS",
        "------------",
    ]
    for cls in sorted_classes:
        mn  = np.mean(class_phi_n[cls])
        frc = class_frac[cls] * 100
        lines.append(f"  {cls:12s}: mean φ_N = {mn:.5f}  "
                     f"fraction of total |φ| = {frc:.1f}%")
    lines += [
        "",
        f"  φ_N fraction range: {frac_min:.1f}% – {frac_max:.1f}% of total |φ|",
        "  Flat KernelSHAP: φ_N ≡ 0 (computation subgraph not in coalition space)",
        f"  src/dst novelty SHAP non-zero in {nov_frac:.1f}% of flows "
        f"({nov_n_nonzero}/{nov_n_total}, >1e-6 threshold, per novelty_audit.json);",
        f"  {nov_top2_str} highest (mixed-IP topology: 9 RFC1918/loopback)",
        "",
        "PAPER FRAMING",
        "-------------",
        f"φ_N accounts for {frac_min:.0f}–{frac_max:.0f}% of total |φ| across classes,",
        "confirming that the GNN computation subgraph encodes non-trivial context.",
        "This is not a statistical artefact: node_shap is measured by ablating each",
        "computation node from the coalition, so a non-zero value means the node's",
        "state (out-degree, port entropy, rolling byte count, novelty) causally",
        "affects the prediction. Flat KernelSHAP treats all flows as independent",
        "and cannot recover this neighbourhood context. The src/dst novelty components",
        f"are non-zero in {nov_frac:.1f}% of explained flows ({nov_n_nonzero}/{nov_n_total}),",
        f"with attack classes averaging {nov_attack_mean:.1f}% and {nov_top2_str} reaching",
        "the highest rates. The dataset contains a mixed-IP topology (9 RFC1918/loopback",
        "endpoints alongside 34 public IPs), so novelty is a live signal. Full per-class",
        "rates are in §4.5 and novelty_audit.txt.",
        "",
        "CAPTION",
        "-------",
        r"Node-structural attribution (φ_N) across SHAP-GSD classes.",
        "Left: per-flow φ_N vs hour-of-day for the five largest classes; per-hour",
        "mean marked with horizontal ticks; annotation notes the controlled-lab",
        "environment of UNSW-NB15 (2 days, no full diurnal cycle).",
        "Right: violin distribution of φ_N per class (coral fill), sorted by mean",
        fr"φ_N descending; ◆ = mean. φ_N accounts for {frac_min:.0f}–{frac_max:.0f}%",
        "of total |φ|, compared to φ_N ≡ 0 in flat KernelSHAP.",
    ]
    notes = load_paper_notes(STEM, frac_min=frac_min, frac_max=frac_max)
    if notes:
        lines += [""] + notes

    txt_path = OUT_DIR / f"{STEM}.txt"
    txt_path.write_text("\n".join(lines))
    print(f"Saved {txt_path}")


if __name__ == "__main__":
    main()
