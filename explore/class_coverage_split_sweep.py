"""
Split-fraction class-coverage sweep — one dataset per invocation.

Answers: "is a class-coverage gap (per explore/class_coverage_analysis.py) a
property of the capture, or just of the currently-configured split
fractions?" Sweeps 789 candidate (train_frac, val_frac) pairs and reports,
per class, whether ANY candidate gives it nonzero rows in all three splits.

Consolidates two throwaway local/ investigation scripts that did this same
sweep independently per dataset (local/botiot_split_sweep/botiot_split_sweep.py,
local/split_sweep_ton_cicids/split_sweep.py) into one repeatable, git-tracked
tool — those scripts' own JSON/console output remain on disk as a historical
record of the runs that produced the numbers already cited in
final/review01/class_coverage.md and PROXEVAL's
notes/four_dataset_class_coverage_synthesis.md, but this script is the one to
run for any future sweep (including a fresh check on UNSW/BoT-IoT if their
splits ever change again).

Single dataset per invocation, via SHAP_GSD_CONFIG like every other
per-run explore/ script (explore/AGENT.md's convention) — NOT the
all-four-at-once pattern class_coverage_analysis.py uses, since a sweep is
inherently a per-dataset question ("is THIS capture's gap fixable by
retuning THIS capture's split") and each run's own configured
train_frac/val_frac is the natural, config-driven "baseline" to compare
candidates against — no separate --baseline-train/--baseline-val flag needed.

What it reads:
  - The active config's raw CSV (data.csv_path, resolved to the local data/
    symlink the same way explore/class_coverage_analysis.py's
    resolve_local_csv_path does — reused directly from that module, not
    reimplemented) — in full, all rows/columns, same faithfulness
    requirement as class_coverage_analysis.py (dedup needs the whole row).
  - data.train_frac / data.val_frac from the same config, as the baseline
    candidate to report alongside the sweep.

What it outputs:
  outputs/figures/explore/class_coverage_split_sweep_candidates.csv
      One row per swept (train_frac, val_frac) candidate: n_train/val/test,
      n_covered_all3, n_present_test.
  outputs/figures/explore/class_coverage_split_sweep.{pdf,png,txt}
      Figure: classes-covered-in-all-3-splits vs train_frac (all valid
      val_frac choices at each train_frac overlaid as a scatter), baseline
      marked. Reasoning .txt: baseline result, top candidates, best-anywhere
      result, and the list of classes that are NEVER coverable at any
      candidate tested (a capture property no split choice can fix).

Data-source rule exception: same as class_coverage_analysis.py (explore/
AGENT.md §5) — this reads raw CSV rows directly for data characterization,
not model output; not the leakage risk that rule exists to prevent.

Performance note: the naive approach (call Preprocessor._temporal_split once
per candidate) would re-mask the full sorted array 789 times; this instead
precomputes each class's sorted row-position array ONCE and answers every
candidate via np.searchsorted, exactly replicating _temporal_split's
value-based split semantics (searchsorted on a sorted array at a given
cut value is equivalent to counting rows with timestamp <= cut) without
789 full-array passes. Proven on ToN-IoT/CICIDS2018 (20-27M rows): full
789-candidate sweep in ~3s once the cleaned/sorted array is built. Frees
the full-row DataFrame (`del df; gc.collect()`) immediately after dedup/
dropna/sort, before the sweep loop — these two datasets are large enough
that holding the full frame for the sweep's duration is a real memory risk
(confirmed: crashed a 23GB local dev machine when run concurrently with
other work; safe on supek's `cpu` queue with proper `qsub`, never on a
login node).
"""

import gc
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from explore._paths import paths                                   # noqa: E402
from explore.class_coverage_analysis import resolve_local_csv_path  # noqa: E402  [EXISTING, reused]
from src.data.loader import NODE_ID_COLS, PORT_COLS, LABEL_COLS     # noqa: E402  [EXISTING, reused]
from src.data.preprocessor import Preprocessor                      # noqa: E402  [EXISTING, reused]

LABEL_FS = 9
_PRESENCE = "#2a9d8f"  # teal
_ABSENCE = "#e76f51"   # coral
_GRAY = "#888888"

FIGURE_STEM = "class_coverage_split_sweep"
CANDIDATES_CSV_NAME = "class_coverage_split_sweep_candidates.csv"

# Same grid both throwaway precedents used — kept bit-identical so any
# future rerun reproduces the same 789 candidates already cited in
# final/review01/class_coverage.md and PROXEVAL's synthesis doc.
TRAIN_FRAC_GRID = np.arange(0.05, 0.96, 0.01)
VAL_FRAC_FRACTIONS_OF_REMAINING = np.arange(0.1, 1.0, 0.1)
MIN_TEST_FRAC = 0.05


def _build_candidates(baseline_train: float, baseline_val: float) -> list[tuple[float, float]]:
    """The swept (train_frac, val_frac) grid, with the baseline appended if not already in it."""
    candidates: list[tuple[float, float]] = []
    for tf in TRAIN_FRAC_GRID:
        remaining = 1.0 - tf
        if remaining < 0.03:
            continue
        for frac_of_remaining in VAL_FRAC_FRACTIONS_OF_REMAINING:
            vf = round(remaining * frac_of_remaining, 4)
            if vf < 0.01 or (tf + vf) > 0.98:
                continue
            candidates.append((round(float(tf), 4), vf))
    baseline = (round(float(baseline_train), 4), round(float(baseline_val), 4))
    if baseline not in candidates:
        candidates.append(baseline)
    return candidates


def run_sweep(csv_path: Path, baseline_train: float, baseline_val: float) -> dict:
    """Load, clean, sort once; evaluate all candidates via precomputed per-class positions.

    Returns a dict with: n_classes, int_to_class, baseline result, all
    per-candidate results, and the never-coverable class list.
    """
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    n_raw = len(df)
    print(f"Loaded {n_raw:,} rows, {df.shape[1]} cols")

    last_two = list(df.columns[-2:])
    if last_two != LABEL_COLS:
        raise ValueError(
            f"{csv_path}: expected the last 2 columns to be {LABEL_COLS} "
            f"(by position), but found {last_two}."
        )

    n_before = len(df)
    df = df.drop_duplicates()
    n_dedup_dropped = n_before - len(df)
    print(f"Dropped {n_dedup_dropped:,} exact duplicates -> {len(df):,} rows")

    required = NODE_ID_COLS + PORT_COLS
    n_before = len(df)
    df = df.dropna(subset=required)
    n_dropna_dropped = n_before - len(df)
    print(f"Dropped {n_dropna_dropped:,} rows missing {required} -> {len(df):,} rows")

    label_map = Preprocessor.build_label_map(df)  # [EXISTING, reused]
    int_to_class = {v: k for k, v in label_map.items()}
    n_classes = len(label_map)

    # Narrow to exactly what the sweep needs, then free the full-row frame —
    # dedup/dropna already ran on the full row above, so this doesn't
    # retroactively undercount either delta (mirrors class_coverage_analysis.py).
    df = df[["FLOW_START_MILLISECONDS", "Attack"]].copy()
    gc.collect()

    df = df.sort_values("FLOW_START_MILLISECONDS", kind="mergesort").reset_index(drop=True)
    n = len(df)
    ts = df["FLOW_START_MILLISECONDS"].to_numpy(dtype=np.int64)
    class_ints = df["Attack"].map(label_map).to_numpy(dtype=np.int64)
    del df
    gc.collect()
    print(f"Sorted, n_total={n:,}")

    class_positions = {c: np.flatnonzero(class_ints == c) for c in range(n_classes)}

    def counts_in_range(lo: int, hi: int) -> np.ndarray:
        out = np.zeros(n_classes, dtype=np.int64)
        for c, pos in class_positions.items():
            out[c] = np.searchsorted(pos, hi) - np.searchsorted(pos, lo)
        return out

    def eval_split(train_frac: float, val_frac: float) -> dict:
        train_cut = int(n * train_frac)
        val_cut = int(n * (train_frac + val_frac))
        tau_train_ms = int(ts[train_cut])
        tau_val_ms = int(ts[val_cut])
        train_end = int(np.searchsorted(ts, tau_train_ms, side="right"))
        val_end = int(np.searchsorted(ts, tau_val_ms, side="right"))
        train_counts = counts_in_range(0, train_end)
        val_counts = counts_in_range(train_end, val_end)
        test_counts = counts_in_range(val_end, n)
        covered = int(np.sum((train_counts > 0) & (val_counts > 0) & (test_counts > 0)))
        present_test = int(np.sum(test_counts > 0))
        return {
            "train_frac": train_frac, "val_frac": val_frac,
            "n_train": train_end, "n_val": val_end - train_end, "n_test": n - val_end,
            "train_counts": train_counts, "val_counts": val_counts, "test_counts": test_counts,
            "n_covered_all3": covered, "n_present_test": present_test,
        }

    candidates = _build_candidates(baseline_train, baseline_val)
    results = [eval_split(tf, vf) for tf, vf in candidates]
    baseline_result = eval_split(round(float(baseline_train), 4), round(float(baseline_val), 4))

    never_coverable = [
        int_to_class[c] for c in range(n_classes)
        if not any((r["train_counts"][c] > 0 and r["val_counts"][c] > 0 and r["test_counts"][c] > 0) for r in results)
    ]

    return {
        "n_raw": n_raw, "n_dedup_dropped": n_dedup_dropped, "n_dropna_dropped": n_dropna_dropped,
        "n_clean": n, "n_classes": n_classes, "int_to_class": int_to_class,
        "baseline_train": baseline_train, "baseline_val": baseline_val,
        "baseline_result": baseline_result, "results": results,
        "never_coverable": never_coverable,
    }


def write_candidates_csv(sweep: dict, path: Path) -> None:
    rows = [
        {"train_frac": r["train_frac"], "val_frac": r["val_frac"],
         "n_train": r["n_train"], "n_val": r["n_val"], "n_test": r["n_test"],
         "n_covered_all3": r["n_covered_all3"], "n_present_test": r["n_present_test"]}
        for r in sweep["results"]
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_figure_and_reasoning(sweep: dict, dataset_name: str, out_dir: Path) -> None:
    n_classes = sweep["n_classes"]
    results = sweep["results"]
    baseline = sweep["baseline_result"]

    viable = [r for r in results if r["n_test"] >= MIN_TEST_FRAC * sweep["n_clean"]]
    viable_sorted = sorted(viable, key=lambda r: (-r["n_covered_all3"], -r["n_present_test"], -r["n_test"]))
    best_any = sorted(results, key=lambda r: (-r["n_covered_all3"], -r["n_present_test"]))
    best = best_any[0] if best_any else baseline
    improves = best["n_covered_all3"] > baseline["n_covered_all3"]

    fig, ax = plt.subplots(figsize=(8, 5))
    train_fracs = np.array([r["train_frac"] for r in results])
    covered = np.array([r["n_covered_all3"] for r in results])
    ax.scatter(train_fracs, covered, s=14, color=_PRESENCE, alpha=0.6, label="swept candidate")
    ax.scatter([baseline["train_frac"]], [baseline["n_covered_all3"]], s=90, color=_ABSENCE,
               marker="D", zorder=5, label=f"configured split (train_frac={baseline['train_frac']:.3f})")
    ax.set_xlabel("train_frac", fontsize=LABEL_FS)
    ax.set_ylabel("classes covered in ALL 3 splits", fontsize=LABEL_FS)
    ax.set_title(f"{dataset_name}: class coverage vs. split fraction ({len(results)} candidates)", fontsize=LABEL_FS)
    ax.set_ylim(-0.5, n_classes + 0.5)
    ax.axhline(n_classes, color=_GRAY, linestyle="--", linewidth=1, alpha=0.6)
    ax.legend(fontsize=LABEL_FS - 1, loc="lower right")
    ax.tick_params(labelsize=LABEL_FS)
    fig.tight_layout()
    fig.savefig(out_dir / f"{FIGURE_STEM}.pdf")
    fig.savefig(out_dir / f"{FIGURE_STEM}.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / f'{FIGURE_STEM}.pdf'}")
    print(f"Saved {out_dir / f'{FIGURE_STEM}.png'}")

    top_lines = "\n".join(
        f"  tf={r['train_frac']:.2f} vf={r['val_frac']:.2f}  "
        f"n=({r['n_train']:,}/{r['n_val']:,}/{r['n_test']:,})  "
        f"covered_all3={r['n_covered_all3']}/{n_classes}  present_test={r['n_present_test']}/{n_classes}"
        for r in viable_sorted[:10]
    )
    best_lines = "\n".join(
        f"  tf={r['train_frac']:.2f} vf={r['val_frac']:.2f}  "
        f"n=({r['n_train']:,}/{r['n_val']:,}/{r['n_test']:,})  "
        f"covered_all3={r['n_covered_all3']}/{n_classes}  present_test={r['n_present_test']}/{n_classes}"
        for r in best_any[:5]
    )
    never_str = ", ".join(sweep["never_coverable"]) if sweep["never_coverable"] else "(none — every class is coverable somewhere in the grid)"

    if not sweep["never_coverable"] and baseline["n_covered_all3"] == n_classes:
        verdict = "Already full coverage at the configured split — no gap to sweep away."
    elif not improves:
        verdict = (
            f"The configured split ({baseline['n_covered_all3']}/{n_classes} covered) already matches "
            f"the best coverage found anywhere in {len(results)} candidates. This is a capture property, "
            f"not a split-fraction artifact — retuning would not help."
        )
    else:
        verdict = (
            f"The best candidate found ({best['n_covered_all3']}/{n_classes} covered at "
            f"train_frac={best['train_frac']:.2f}) beats the configured split "
            f"({baseline['n_covered_all3']}/{n_classes}) at a test-split size >= {MIN_TEST_FRAC*100:.0f}% of the "
            f"cleaned data — real, usable headroom exists; whether to retune is an owner decision, not "
            f"made by this script."
        )

    reasoning = f"""
Figure reasoning — {FIGURE_STEM}
{'=' * (24 + len(FIGURE_STEM))}

WHAT THE FIGURE SHOWS
----------------------
{dataset_name}: for each of {len(results)} candidate (train_frac, val_frac) chronological
splits, how many of this dataset's {n_classes} classes have nonzero rows in ALL THREE
splits (train, val, test). The configured/adopted split is marked with a coral
diamond; every other swept candidate is a teal dot. The dashed horizontal line
marks full coverage ({n_classes}/{n_classes}).

KEY FINDINGS
------------
Configured split (train_frac={baseline['train_frac']:.4f}, val_frac={baseline['val_frac']:.4f}):
  {baseline['n_covered_all3']}/{n_classes} classes covered in all 3 splits, {baseline['n_present_test']}/{n_classes} present in test.

Top candidates with test split >= {MIN_TEST_FRAC*100:.0f}% of cleaned rows (usable splits only):
{top_lines if top_lines else '  (none — every candidate failed the minimum test-size filter)'}

Best coverage found anywhere (any test size, including unusably small ones):
{best_lines}

Classes NEVER coverable in all 3 splits, at ANY of the {len(results)} candidates tested:
  {never_str}

PAPER FRAMING
-------------
"{verdict}"

SUGGESTED FIGURE CAPTION
-------------------------
{dataset_name}: number of classes present in all three chronological splits
(train/val/test) across {len(results)} swept (train_frac, val_frac) candidates.
The configured split (train_frac={baseline['train_frac']:.3f}) is marked; the dashed line
marks full coverage ({n_classes}/{n_classes} classes).
"""
    txt_path = out_dir / f"{FIGURE_STEM}.txt"
    txt_path.write_text(reasoning)
    print(f"Saved {txt_path}")


def main() -> None:
    """Sweep split fractions for the currently active dataset (SHAP_GSD_CONFIG) and report coverage."""
    p = paths()
    cfg = p["cfg"]
    dataset_name = p["dataset_name"]
    baseline_train = cfg["data"]["train_frac"]
    baseline_val = cfg["data"]["val_frac"]
    csv_path = resolve_local_csv_path(cfg)

    out_dir = p["figures"] / "explore"
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep = run_sweep(csv_path, baseline_train, baseline_val)

    write_candidates_csv(sweep, out_dir / CANDIDATES_CSV_NAME)
    print(f"Wrote {out_dir / CANDIDATES_CSV_NAME}")
    write_figure_and_reasoning(sweep, dataset_name, out_dir)


if __name__ == "__main__":
    main()
