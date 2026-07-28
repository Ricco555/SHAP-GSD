"""
TE-G-SAGE minority-class F1 comparison — NF-UNSW-NB15-v3 ONLY.

UNSW-SPECIFIC. The reference F1 values below are the TE-G-SAGE (Paper 1)
published per-class scores on NF-UNSW-NB15-v3. They are meaningless for any
other dataset: the Paper 3 NetFlow datasets (NF-CSE-CIC-IDS2018-v3,
NF-ToN-IoT-v3, NF-BoT-IoT-v3) have disjoint class taxonomies with no
TE-G-SAGE published counterpart, and a name-collision on a literal class
string ("DoS" appears in NF-BoT-IoT-v3 too) would otherwise attach a
UNSW-derived verdict to an unrelated dataset. For that reason this comparison
was removed from the reusable ``src/model/evaluator.py`` pipeline and lives
here as a one-off exploration script for the SHAP-GSD paper only.

What it shows:
  Single panel — grouped horizontal bars, TE-G-SAGE vs SHAP-GSD test F1 for
  the two minority classes TE-G-SAGE reported (Backdoor, DoS), annotated with
  the per-class delta.

Reads:
  <artifacts_dir>/evaluation/metrics.json   — "per_class" F1 written by
                                              scripts/05_evaluate.py

Outputs:
  <outputs_dir>/figures/explore/teg_sage_comparison.pdf
  <outputs_dir>/figures/explore/teg_sage_comparison.png
  <outputs_dir>/figures/explore/teg_sage_comparison.txt

Run:
  python explore/teg_sage_comparison.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from explore._paths import REPO_ROOT, paths  # noqa: E402

_P = paths()
METRICS_JSON = REPO_ROOT / _P["cfg"]["output"]["artifacts_dir"] / "evaluation" / "metrics.json"
OUT_DIR = _P["figures"] / "explore"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEM = "teg_sage_comparison"

# TE-G-SAGE published per-class F1 on NF-UNSW-NB15-v3. UNSW-only by design —
# see the module docstring for why this is not in the reusable pipeline.
TEG_SAGE_F1_BASELINES: dict[str, float] = {
    "Backdoor": 0.071,
    "DoS":      0.26,
}

LABEL_FS = 9
_SHAP_GSD = "#2a9d8f"   # teal  — SHAP-GSD
_TEG_SAGE = "#e9c46a"   # amber — TE-G-SAGE reference
_GRAY = "#888888"


def build_comparison(per_class: dict[str, dict]) -> dict[str, dict]:
    """Build the TE-G-SAGE vs SHAP-GSD per-class F1 delta table.

    Args:
        per_class: the ``per_class`` block of ``evaluation/metrics.json``,
            mapping class name to a dict containing at least an ``f1`` key.

    Returns:
        Mapping of class name to
        ``{"teg_sage_f1", "shap_gsd_f1", "delta", "improved"}``. Classes
        absent from ``per_class`` are skipped.
    """
    comparison: dict[str, dict] = {}
    for cls_name, baseline_f1 in TEG_SAGE_F1_BASELINES.items():
        shap_gsd_f1 = per_class.get(cls_name, {}).get("f1")
        if shap_gsd_f1 is None:
            continue
        comparison[cls_name] = {
            "teg_sage_f1": baseline_f1,
            "shap_gsd_f1": shap_gsd_f1,
            "delta":       round(shap_gsd_f1 - baseline_f1, 4),
            "improved":    shap_gsd_f1 > baseline_f1,
        }
    return comparison


def print_comparison(comparison: dict[str, dict]) -> None:
    """Print the per-class delta table to the console."""
    print("=" * 65)
    print("TE-G-SAGE minority-class targets  (NF-UNSW-NB15-v3)")
    for name, row in comparison.items():
        status = "IMPROVED" if row["improved"] else "NOT MET"
        sign = "+" if row["delta"] >= 0 else ""
        print(
            f"  {name:<10s} TE-G-SAGE={row['teg_sage_f1']:.3f}  "
            f"SHAP-GSD={row['shap_gsd_f1']:.4f}  "
            f"D={sign}{row['delta']:.4f}  [{status}]"
        )
    print("=" * 65)


def plot_comparison(comparison: dict[str, dict], out_dir: Path) -> None:
    """Save the grouped-bar comparison figure as PDF and PNG."""
    names = list(comparison.keys())
    y = np.arange(len(names))
    teg = [comparison[n]["teg_sage_f1"] for n in names]
    gsd = [comparison[n]["shap_gsd_f1"] for n in names]

    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.88, bottom=0.20)

    ax.barh(y + 0.18, teg, height=0.34, color=_TEG_SAGE,
            edgecolor="white", linewidth=0.4, label="TE-G-SAGE (Paper 1)")
    ax.barh(y - 0.18, gsd, height=0.34, color=_SHAP_GSD,
            edgecolor="white", linewidth=0.4, label="SHAP-GSD")

    for yi, name in zip(y, names):
        row = comparison[name]
        sign = "+" if row["delta"] >= 0 else ""
        ax.text(
            max(row["teg_sage_f1"], row["shap_gsd_f1"]) + 0.008, yi,
            f"Δ={sign}{row['delta']:.4f}",
            va="center", ha="left", fontsize=LABEL_FS - 1, color=_GRAY,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=LABEL_FS)
    ax.set_xlabel("Test-split F1", fontsize=LABEL_FS)
    ax.set_xlim(0, max(max(teg), max(gsd)) * 1.45)
    ax.tick_params(axis="x", labelsize=LABEL_FS - 1)
    ax.set_title("Minority-class F1 vs TE-G-SAGE baseline",
                 fontsize=LABEL_FS, fontweight="bold", pad=4)
    ax.legend(fontsize=LABEL_FS - 1, loc="lower right",
              framealpha=0.85, edgecolor="#cccccc")

    for ext, dpi in (("pdf", 300), ("png", 150)):
        p = out_dir / f"{STEM}.{ext}"
        fig.savefig(str(p), bbox_inches="tight", dpi=dpi)
        print(f"Saved {p}")
    plt.close(fig)


def write_reasoning(comparison: dict[str, dict], out_dir: Path) -> Path:
    """Write the mandatory companion .txt reasoning file. Returns its path."""
    rows = "\n".join(
        f"  {n:<10s} TE-G-SAGE={r['teg_sage_f1']:.3f}  "
        f"SHAP-GSD={r['shap_gsd_f1']:.4f}  "
        f"delta={'+' if r['delta'] >= 0 else ''}{r['delta']:.4f}  "
        f"[{'IMPROVED' if r['improved'] else 'NOT MET'}]"
        for n, r in comparison.items()
    )
    reasoning = f"""\
Figure reasoning — {STEM}
=================================

WHAT THE FIGURE SHOWS
----------------------
Grouped horizontal bars comparing the TE-G-SAGE (Paper 1) published per-class
F1 on NF-UNSW-NB15-v3 against this SHAP-GSD run's test-split F1, for the two
minority classes TE-G-SAGE reported: Backdoor (0.071) and DoS (0.26). Each
pair is annotated with the SHAP-GSD minus TE-G-SAGE delta.

MEASURED VALUES
---------------
{rows}

KEY FINDINGS
------------
This comparison is deliberately UNSW-NB15-only. TE-G-SAGE published baselines
exist for no other NetFlow dataset, and the Paper 3 datasets
(NF-CSE-CIC-IDS2018-v3, NF-ToN-IoT-v3, NF-BoT-IoT-v3) have disjoint class
taxonomies, so the comparison was moved out of the reusable evaluator into
this one-off script rather than being gated on class-name string matches.

Note that the two models differ in more than the explainer: SHAP-GSD's model
uses IP-level nodes, temporally-constrained neighbour sampling and a 15-dim
node state, so per-class F1 differences reflect the whole modelling change,
not the explanation method.

PAPER FRAMING
-------------
"On the two minority classes for which TE-G-SAGE reports per-class F1
(Backdoor, DoS), the SHAP-GSD detector's test-split F1 is compared directly
against the published baseline. The comparison is restricted to NF-UNSW-NB15-v3,
the only dataset for which a TE-G-SAGE per-class reference exists."

SUGGESTED FIGURE CAPTION
-------------------------
Minority-class test F1 on NF-UNSW-NB15-v3: TE-G-SAGE published baseline
(amber) against the SHAP-GSD detector (teal), for the two minority classes
TE-G-SAGE reports. Annotations give the per-class difference. The comparison
applies to NF-UNSW-NB15-v3 only; no TE-G-SAGE per-class reference exists for
the other NetFlow datasets evaluated.
"""
    txt_path = out_dir / f"{STEM}.txt"
    txt_path.write_text(reasoning)
    return txt_path


def main() -> None:
    """Load metrics.json, recompute the comparison, print and save it."""
    if not METRICS_JSON.exists():
        raise FileNotFoundError(
            f"Evaluation metrics not found: {METRICS_JSON}\n"
            "Run scripts/05_evaluate.py first."
        )
    with open(METRICS_JSON) as f:
        metrics = json.load(f)

    per_class = metrics.get("per_class", {})
    comparison = build_comparison(per_class)
    if not comparison:
        raise SystemExit(
            f"None of {sorted(TEG_SAGE_F1_BASELINES)} present in "
            f"{METRICS_JSON} per_class — this script is NF-UNSW-NB15-v3 only."
        )

    print_comparison(comparison)
    plot_comparison(comparison, OUT_DIR)
    txt_path = write_reasoning(comparison, OUT_DIR)
    print(f"Saved {txt_path}")


if __name__ == "__main__":
    main()
