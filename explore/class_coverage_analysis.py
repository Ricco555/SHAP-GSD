"""
Cross-dataset class-coverage analysis (specs/47, specs/48).

One script, one code path, run against all four Paper-3 NetFlow datasets
(NF-UNSW-NB15-v3, NF-BoT-IoT-v3, NF-ToN-IoT-v3, NF-CICIDS2018-v3), replacing
a session's worth of ad-hoc, independently-written, per-dataset class-coverage
investigations (specs/47 §1) with one comparable, reproducible tool. The core
computation (``compute_class_coverage``) is called identically for each
dataset — only the CSV path and the two split fractions vary — so any
cross-dataset comparison in the resulting table is actually comparing like
with like.

What it reads:
  - The four raw NetFlow CSVs, in full (all rows, all columns — NOT a
    ``usecols``-narrowed read; see the "Data-source rule exception" note
    below for why the full-row read is required, not merely convenient).
  - The four experiment configs (``configs/experiment_unsw.yaml``,
    ``configs/experiment_nf_bot_iot_v3.yaml``, ``configs/experiment_nf_ton_iot_v3.yaml``,
    ``configs/experiment_nf_cicids2018_v3.yaml``) for ``data.train_frac``,
    ``data.val_frac``, and to resolve each local CSV path from
    ``data.csv_path`` (see ``resolve_local_csv_path``).

What it outputs (only on a full, non-``--only`` run — see ``--only`` below):
  - ``outputs/figures/explore/class_coverage_by_split.csv`` — one row per
    (dataset, class) pair: raw per-class-per-split row counts.
  - ``outputs/figures/explore/class_coverage_summary.md`` — one row per
    dataset (4 rows total): the dedup rule stated once as a preamble, plus a
    per-dataset class-coverage summary table.
  - ``outputs/figures/explore/class_coverage_timeline.{pdf,png,txt}`` — a
    4-panel (one panel per dataset) per-class flow-density heatmap over
    elapsed capture time, with each dataset's own split-boundary lines
    overlaid.

Data-source rule exception (explore/AGENT.md §5, "never use raw val/test
data directly"): this script reads raw CSV rows directly, which is not the
leakage risk that rule exists to prevent. That rule guards against a figure
using val/test rows a trained model was scored on — a genuine risk, since a
figure built that way could leak information the model never saw during
training into a "results" artifact. This script never touches a model, a
prediction, or a SHAP value at all: it characterizes the RAW INPUT DATASETS'
own temporal and class structure, conceptually BEFORE any train/val/test
split exists as a modeling artifact. This is the same class of exception
already approved for ``explore/class_time_distribution.py`` (data
characterization, not model output) — restated here in this script's own
words per specs/48 §1.1 point 4, rather than silently relying on that
precedent by reference alone.

Output-location note: ``OUT_DIR`` is a hardcoded
``REPO_ROOT/outputs/figures/explore/`` path, NOT resolved via
``explore._paths.paths()``/``SHAP_GSD_CONFIG``. Every current experiment
config (all four) now sets a non-empty ``run.dir``, so there is no single
"active run" this cross-dataset tool could correctly key its output off —
``_paths.paths()``'s config-driven resolution would place this 4-dataset
artifact inside one specific dataset's ``runs/`` subtree (specs/48 §0.2).

``--only`` flag caveat: a partial run (``--only unsw,bot_iot``) prints
per-dataset results to stdout for quick iteration but does NOT write the
three canonical output files — only a full four-dataset run (no ``--only``
flag) produces ``class_coverage_by_split.csv``, ``class_coverage_summary.md``,
and ``class_coverage_timeline.{pdf,png,txt}``. This prevents a dev-iteration
partial run from silently overwriting the canonical four-row/four-panel
artifacts with a one-row/one-panel version.
"""

import argparse
import gc
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from explore._temporal_streaming import (  # noqa: E402  [PROPOSED NEW]
    argsort_and_take,
    pooled_timestamp_stats_from_array,
)
from src.data.loader import NODE_ID_COLS, PORT_COLS, LABEL_COLS  # noqa: E402  [EXISTING, reused]
from src.data.preprocessor import Preprocessor  # noqa: E402      [EXISTING, reused]
from src.utils.config import load_config  # noqa: E402            [EXISTING, reused]

OUT_DIR = REPO_ROOT / "outputs" / "figures" / "explore"   # [PROPOSED NEW] — see module docstring

FIGURE_STEM = "class_coverage_timeline"
BY_SPLIT_CSV_NAME = "class_coverage_by_split.csv"
SUMMARY_MD_NAME = "class_coverage_summary.md"

LABEL_FS = 9            # [EXISTING, reused constant value — explore/AGENT.md §6]
_GRAY = "#888888"       # [EXISTING, reused constant value — explore/AGENT.md §6]
# _PRESENCE/_ABSENCE/_SHAP_GSD/_GNN_EXP intentionally NOT imported/defined:
# none is semantically applicable to a density heatmap (explore/AGENT.md §6's
# constants are for presence/absence-of-attribution-sign figures; this figure
# encodes flow density, an unrelated axis) — per specs/47 §4's explicit
# "do not force an unrelated constant's use just to check a box."

NUM_TIME_BINS = 200     # [PROPOSED NEW] fixed bin count per panel, independent
                         # of each dataset's own row count — keeps the figure's
                         # visual resolution comparable across panels even
                         # though the four datasets differ by 10x+ in row count.

# name -> config path (relative to REPO_ROOT). This is the one hardcoded
# surface specs/47 §6 explicitly sanctions; everything else (train_frac,
# val_frac, local csv path) is read live from each config, never duplicated
# here (specs/47 §6, commit 9346a68 precedent).
DATASETS: dict[str, str] = {
    "unsw":       "configs/experiment_unsw.yaml",
    "bot_iot":    "configs/experiment_nf_bot_iot_v3.yaml",
    "ton_iot":    "configs/experiment_nf_ton_iot_v3.yaml",
    "cicids2018": "configs/experiment_nf_cicids2018_v3.yaml",
}

# DATASETS' iteration order (unsw, bot_iot, ton_iot, cicids2018) is the
# CANONICAL row/panel order for every output artifact (CSV rows, markdown
# table rows, figure panel top-to-bottom) — fixed once here and reused
# everywhere downstream; never re-sorted or re-derived elsewhere.


@dataclass
class ClassCoverageResult:
    """Per-dataset class-coverage computation — one instance per dataset.

    All fields are LIVE-computed by compute_class_coverage; nothing here is
    transcribed from specs/47 or any prior session's numbers.
    """
    csv_path: Path
    train_frac: float
    val_frac: float

    n_raw: int                 # rows read from the CSV, before any cleaning
    n_dedup_dropped: int       # rows removed by drop_duplicates()
    n_dropna_dropped: int      # rows removed by dropna(NODE_ID_COLS+PORT_COLS)
    n_clean: int                # n_raw - n_dedup_dropped - n_dropna_dropped

    tau_train_ms: int
    tau_val_ms: int

    label_map: dict[str, int]           # Benign=0, rest alphabetical (build_label_map)
    ordered_classes: list[str]          # label_map keys, sorted by label_map value

    # per class -> {"train": n, "val": n, "test": n}; every ordered_classes
    # entry present with an explicit int (0 if that split has none) — never
    # an omitted key (specs/47 §8's "reported as 0, not omitted" requirement).
    counts: dict[str, dict[str, int]]

    n_train: int
    n_val: int
    n_test: int
    n_unassigned: int    # n_clean - (n_train + n_val + n_test); see specs/48 §0.6

    # For the figure only (§4) — same cleaned+sorted rows the counts above
    # were computed from, narrowed to exactly the two columns the figure
    # needs, so table and figure are provably the same computation
    # (specs/47 §5.3's "single source of truth" requirement).
    elapsed_hours: np.ndarray    # float64, shape (n_clean,), sorted ascending
    class_of_row: np.ndarray     # str/object, shape (n_clean,), aligned to elapsed_hours
    total_span_hours: float
    tau_train_h: float
    tau_val_h: float


def compute_class_coverage(
    csv_path: Path | str,
    train_frac: float,
    val_frac: float,
) -> ClassCoverageResult:
    """Faithfully replicate load_raw's dedup/dropna + Preprocessor's
    chronological split, on one raw NetFlow CSV, and return per-class
    per-split row counts plus the two cleaning deltas.

    Does NOT call load_raw() itself (specs/48 §0.1 — would force a second
    full CSV read to recover the dedup/dropna deltas it doesn't return, and
    runs numeric-imputation/inf-replacement/dtype-coercion work this
    row-count-only tool never uses) or Preprocessor.fit_transform() (fits a
    StandardScaler/OneHotEncoder this tool never uses). DOES call
    Preprocessor.build_label_map and Preprocessor._temporal_split directly —
    the two pieces of real pipeline logic whose exact behavior this tool's
    whole value proposition depends on reproducing byte-for-byte (specs/47
    §5.3).

    Pure function: no config read, no printing, no file writes — csv_path/
    train_frac/val_frac in, ClassCoverageResult out (specs/48 §2.6).
    """
    csv_path = Path(csv_path)

    # Full-row read, all columns, low_memory=False (specs/48 §0.6 — both are
    # faithfulness requirements, not perf knobs to relax).
    df = pd.read_csv(csv_path, low_memory=False)
    n_raw = len(df)

    # Structural validation, mirroring load_raw's own check (loader.py:154-166)
    # via the SAME imported LABEL_COLS constant — no independently-typed
    # column-name literal here.
    last_two = list(df.columns[-2:])
    if last_two != LABEL_COLS:
        raise ValueError(
            f"{csv_path}: expected the last 2 columns to be {LABEL_COLS} "
            f"(by position), but found {last_two}."
        )

    n_before = len(df)
    df = df.drop_duplicates()
    n_dedup_dropped = n_before - len(df)

    required = NODE_ID_COLS + PORT_COLS
    n_before = len(df)
    df = df.dropna(subset=required)
    n_dropna_dropped = n_before - len(df)
    n_clean = len(df)

    # Narrow to exactly what the rest of this function + the figure need,
    # THEN free the full-row frame — dedup/dropna already ran on the full
    # row above (specs/47 §5.1), so narrowing here does not retroactively
    # undercount either delta. Done before sort/label-map/split purely for
    # memory (ToN-IoT/CICIDS2018 are 20-27M rows x ~55 cols pre-narrow).
    df = df[["FLOW_START_MILLISECONDS", "Attack"]].copy()
    gc.collect()

    # build_label_map is order-independent, so it is computed BEFORE the
    # sort — it never needed sorted input (specs/72 §3.2).
    label_map = Preprocessor.build_label_map(df)                       # [EXISTING, reused]
    ordered_classes = sorted(label_map, key=lambda c: label_map[c])
    int_to_class = {v: k for k, v in label_map.items()}

    # Narrow to compact arrays BEFORE the sort (specs/72 §2.3): ts stays
    # float64 (NOT int64) because FLOW_START_MILLISECONDS may contain NaN
    # (see n_unassigned below, specs/72 §3.2.1); codes is a compact int16
    # class-code array, not an object/string array, so argsort_and_take's
    # reindex step costs ~8.6x less than reindexing raw Attack strings.
    ts = df["FLOW_START_MILLISECONDS"].to_numpy(dtype=np.float64)
    codes = df["Attack"].map(label_map).to_numpy(dtype=np.int16)
    del df
    gc.collect()

    # Replaces sort_values(kind="mergesort") — a cheap paired-array reorder
    # instead of a full-frame permute (specs/72 §1.3).
    ts_sorted, codes_sorted = argsort_and_take(ts, codes)
    del ts, codes
    gc.collect()

    pass1 = pooled_timestamp_stats_from_array(ts_sorted, train_frac, val_frac)

    # Reconstruct the small 2-column, now-genuinely-sorted DataFrame
    # Preprocessor._temporal_split expects (specs/72 §3.2).
    sorted_df = pd.DataFrame({
        "FLOW_START_MILLISECONDS": ts_sorted,
        "Attack": pd.Categorical.from_codes(
            codes_sorted, categories=[int_to_class[i] for i in range(len(label_map))]
        ),
    })

    pre = Preprocessor(train_frac=train_frac, val_frac=val_frac)
    train_df, val_df, test_df = pre._temporal_split(sorted_df)         # [EXISTING, reused]
    tau_train_ms, tau_val_ms = pre.tau_train_ms, pre.tau_val_ms

    counts: dict[str, dict[str, int]] = {}
    for cls in ordered_classes:
        counts[cls] = {
            "train": int((train_df["Attack"] == cls).sum()),
            "val":   int((val_df["Attack"] == cls).sum()),
            "test":  int((test_df["Attack"] == cls).sum()),
        }

    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    n_unassigned = n_clean - (n_train + n_val + n_test)   # specs/48 §0.6

    t0_ms = pass1.t0_ms
    elapsed_hours = (ts_sorted - t0_ms) / (1000.0 * 3600.0)
    total_span_hours = pass1.total_span_hours
    # NOT pass1.tau_train_h/tau_val_h: Pass1Result's tau_train_h/tau_val_h are
    # derived from pooled_timestamp_stats_from_array's OWN row-fraction-cut
    # index convention (int(n*train_frac) - 1, class_time_distribution.py's
    # convention -- see _pass1_from_sorted_ms's docstring), which differs by
    # one rank from Preprocessor._temporal_split's convention
    # (int(n*train_frac), no -1) that ACTUALLY produced tau_train_ms/
    # tau_val_ms above. Using pass1's tau_*_h here would silently derive the
    # figure's split-boundary lines (write_timeline_figure's
    # ax.axvline(r.tau_train_h)) from a different cut than the one that
    # produced the returned tau_train_ms/tau_val_ms and train_df/val_df/
    # test_df -- an internally inconsistent result object whenever the two
    # ranks don't happen to share a tied timestamp value. Recomputed here
    # from the ACTUAL tau_train_ms/tau_val_ms instead, matching the original
    # (pre-refactor) script's formula exactly.
    tau_train_h = (tau_train_ms - t0_ms) / (1000.0 * 3600.0)
    tau_val_h   = (tau_val_ms   - t0_ms) / (1000.0 * 3600.0)

    return ClassCoverageResult(
        csv_path=csv_path, train_frac=train_frac, val_frac=val_frac,
        n_raw=n_raw, n_dedup_dropped=n_dedup_dropped,
        n_dropna_dropped=n_dropna_dropped, n_clean=n_clean,
        tau_train_ms=tau_train_ms, tau_val_ms=tau_val_ms,
        label_map=label_map, ordered_classes=ordered_classes, counts=counts,
        n_train=n_train, n_val=n_val, n_test=n_test, n_unassigned=n_unassigned,
        elapsed_hours=elapsed_hours, class_of_row=sorted_df["Attack"].to_numpy(),
        total_span_hours=total_span_hours,
        tau_train_h=tau_train_h, tau_val_h=tau_val_h,
    )


def resolve_local_csv_path(cfg: dict) -> Path:
    """Re-root cfg['data']['csv_path']'s basename under the local data/ dir.

    Correct uniformly for all four configs: a no-op re-rooting for the three
    whose csv_path is already "data/NF-*.csv", and the mechanism that
    recovers the local path for UNSW's committed cluster Lustre path without
    any dataset-specific special case (specs/47 §6).
    """
    return REPO_ROOT / "data" / Path(cfg["data"]["csv_path"]).name


def write_by_split_csv(results: dict[str, "ClassCoverageResult"], path: Path) -> None:
    """Deliverable 1 — one row per (dataset, class) pair.

    DATASETS order outer, ordered_classes order inner (both canonical orders
    fixed at module scope / on ClassCoverageResult).
    """
    rows = []
    for key in DATASETS:
        r = results[key]
        for cls in r.ordered_classes:
            c = r.counts[cls]
            rows.append({
                "dataset": key,
                "class": cls,
                "class_int": r.label_map[cls],
                "n_train": c["train"],
                "n_val": c["val"],
                "n_test": c["test"],
                "n_total": c["train"] + c["val"] + c["test"],
            })
    pd.DataFrame(rows, columns=[
        "dataset", "class", "class_int", "n_train", "n_val", "n_test", "n_total",
    ]).to_csv(path, index=False)


def write_summary_markdown(results: dict[str, "ClassCoverageResult"], path: Path) -> None:
    """Deliverables 2 + 4 — the one 4-row cross-dataset table, plus the dedup
    rule stated once, explicitly, as a preamble.
    """
    display_names = {key: results[key].csv_path.stem for key in DATASETS}

    dedup_lines = []
    for key in DATASETS:
        r = results[key]
        dedup_lines.append(
            f"- **{display_names[key]}**: `{r.n_raw:,}` raw rows → "
            f"`{r.n_dedup_dropped:,}` exact duplicates dropped → "
            f"`{r.n_dropna_dropped:,}` further rows dropped for a missing "
            f"node/port value → `{r.n_clean:,}` clean rows."
        )
    dedup_block = "\n".join(dedup_lines)

    any_unassigned = any(results[key].n_unassigned != 0 for key in DATASETS)
    unassigned_note = (
        "\nAt least one dataset reports a nonzero \"Unassigned\" count below — "
        "that many clean rows had a `FLOW_START_MILLISECONDS` value that "
        "matched none of the three chronological-split masks (a NaN "
        "timestamp — see specs/48 §0.6/§2.5) and are excluded from all "
        "three split counts."
        if any_unassigned else
        "\nEvery dataset below reports \"Unassigned\" = 0: no row's "
        "`FLOW_START_MILLISECONDS` was NaN on this run."
    )

    table_lines = [
        "| Dataset | Classes | Covered (train+val+test) | Zero-train classes | "
        "Zero-val classes | Zero-test classes | n_train | n_val | n_test | Unassigned |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in DATASETS:
        r = results[key]
        n_classes = len(r.label_map)
        covered = sum(
            1 for cls in r.ordered_classes
            if r.counts[cls]["train"] > 0 and r.counts[cls]["val"] > 0 and r.counts[cls]["test"] > 0
        )

        def _zero_list(split: str) -> str:
            names = [cls for cls in r.ordered_classes if r.counts[cls][split] == 0]
            return ", ".join(names) if names else "none"

        table_lines.append(
            f"| {display_names[key]} | {n_classes} | {covered}/{n_classes} | "
            f"{_zero_list('train')} | {_zero_list('val')} | {_zero_list('test')} | "
            f"{r.n_train:,} | {r.n_val:,} | {r.n_test:,} | {r.n_unassigned:,} |"
        )
    table_block = "\n".join(table_lines)

    md = f"""# Cross-Dataset Class Coverage Summary

Every row count below depends on two cleaning steps applied to each raw CSV,
in this exact order: (1) `drop_duplicates()` — full-row exact match — then
(2) `dropna(subset=["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT",
"L4_DST_PORT"])`. This mirrors `src/data/loader.py:load_raw` exactly (same
order, same columns); different datasets shed very different amounts to it:

{dedup_block}
{unassigned_note}

{table_block}

Per-class raw counts (every dataset x every class, one row each) live in
`{BY_SPLIT_CSV_NAME}` (same directory). The per-class flow-density timeline
figure, showing where each class's traffic sits relative to the split
boundaries, lives in `{FIGURE_STEM}.{{pdf,png}}` (same directory).
"""
    path.write_text(md)


def write_timeline_figure(results: dict[str, "ClassCoverageResult"], out_dir: Path) -> None:
    """Deliverable 3 — one 4-panel (stacked) per-class flow-density heatmap,
    one panel per dataset, split-boundary lines overlaid, plus a companion
    .txt reasoning file.
    """
    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(11, 3.0 * len(DATASETS)))
    if len(DATASETS) == 1:
        axes = [axes]

    for ax, key in zip(axes, DATASETS):
        r = results[key]
        bin_edges = np.linspace(0.0, r.total_span_hours, NUM_TIME_BINS + 1)

        density_rows = []
        for cls in r.ordered_classes:
            counts, _ = np.histogram(
                r.elapsed_hours[r.class_of_row == cls], bins=bin_edges
            )
            counts = counts.astype(float)
            counts[counts == 0] = np.nan
            density_rows.append(counts)
        density = np.vstack(density_rows)

        panel_max = np.nanmax(density)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#eeeeee")

        im = ax.imshow(
            density, aspect="auto", origin="upper", cmap=cmap,
            norm=LogNorm(vmin=1, vmax=panel_max),
            extent=[0, r.total_span_hours, len(r.ordered_classes), 0],
        )
        ax.set_yticks(np.arange(len(r.ordered_classes)) + 0.5)
        ax.set_yticklabels(r.ordered_classes, fontsize=LABEL_FS)
        ax.axvline(r.tau_train_h, color=_GRAY, linestyle="--", linewidth=1.3)
        ax.axvline(r.tau_val_h, color=_GRAY, linestyle="--", linewidth=1.3)
        ax.set_title(r.csv_path.stem, fontsize=LABEL_FS)
        ax.set_xlabel("Elapsed time (hours since first flow)", fontsize=LABEL_FS)
        ax.tick_params(labelsize=LABEL_FS)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
        cbar.set_label("Flow count / bin (log scale)", fontsize=LABEL_FS)
        cbar.ax.tick_params(labelsize=LABEL_FS)

    fig.tight_layout()
    fig.savefig(out_dir / f"{FIGURE_STEM}.pdf")
    fig.savefig(out_dir / f"{FIGURE_STEM}.png", dpi=200)
    plt.close(fig)

    # ── Companion .txt reasoning file ───────────────────────────────────────
    findings_lines = []
    for key in DATASETS:
        r = results[key]
        covered_all3 = [
            cls for cls in r.ordered_classes
            if r.counts[cls]["train"] > 0 and r.counts[cls]["val"] > 0 and r.counts[cls]["test"] > 0
        ]
        zero_lines = []
        for split in ("train", "val", "test"):
            zero_cls = [cls for cls in r.ordered_classes if r.counts[cls][split] == 0]
            if zero_cls:
                zero_lines.append(f"      zero-{split}: {', '.join(zero_cls)}")
        findings_lines.append(
            f"  {r.csv_path.stem}: {len(r.label_map)} classes, "
            f"{len(covered_all3)}/{len(r.label_map)} covered in all three splits."
        )
        findings_lines.extend(zero_lines)
    key_findings_block = "\n".join(findings_lines)

    reasoning = f"""
Figure reasoning — {FIGURE_STEM}
=================================

WHAT THE FIGURE SHOWS
----------------------
{len(DATASETS)} stacked panels, one per dataset (top to bottom:
{", ".join(results[key].csv_path.stem for key in DATASETS)}). Each panel is a
class x time-bin heatmap: rows are classes in label_map integer-code order
(Benign first), columns are {NUM_TIME_BINS} equal-width time bins spanning
that dataset's own elapsed capture time (hours since its first flow), and
color encodes flow count per bin on a log scale (zero-count cells are left
uncolored / masked). Two dashed vertical lines per panel mark that dataset's
own chronological split boundaries, tau_train and tau_val, computed by the
same compute_class_coverage() call that produced the summary table's counts
(single source of truth — specs/47 §5.3). The color scale is NORMALIZED PER
PANEL, not shared across panels: the four datasets' row counts differ by
orders of magnitude, so no reader should compare color intensity ACROSS
panels, only WITHIN one.

KEY FINDINGS
------------
{key_findings_block}

PAPER FRAMING
-------------
This figure is a descriptive characterization of each dataset's own raw
class/time structure relative to its configured chronological split — it
states where each class's flow volume falls in time relative to
tau_train/tau_val, not a claim about how any downstream model result should
be interpreted. A class with visibly no density on one side of a boundary is
the same fact the "Zero-{{train,val,test}} classes" columns in
class_coverage_summary.md report numerically; this figure shows where in
time that gap comes from.

SUGGESTED FIGURE CAPTION
-------------------------
Per-class flow density over elapsed capture time for each of the four
Paper-3 NetFlow datasets (one panel per dataset), color = flow count per
time bin (log scale, normalized independently per panel). Dashed vertical
lines mark each dataset's train/validation and validation/test chronological
split boundaries.
"""
    # Placeholder guard (explore/class_time_distribution.py precedent,
    # specs/48 §4.2): fail loudly rather than write a template that looks
    # complete but has an unfilled slot.
    _placeholder = re.search(r"<[A-Z][^<>]{15,}>", reasoning, re.S)
    if _placeholder is not None:
        sys.stderr.write(
            f"ERROR: {FIGURE_STEM} would emit an unfilled placeholder, refusing to write:\n"
            f"  {_placeholder.group(0)[:120]}...\n"
        )
        sys.exit(1)

    (out_dir / f"{FIGURE_STEM}.txt").write_text(reasoning)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: run compute_class_coverage per dataset (§3.3), then
    (on a full, non-``--only`` run) write the three canonical deliverables.
    """
    parser = argparse.ArgumentParser(
        description="Cross-dataset class-coverage analysis (specs/47, specs/48)."
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated subset of {%s} — dev/debug only. Prints "
             "per-dataset results to stdout but does NOT write the three "
             "canonical output files; omit to run all four and produce "
             "them." % ", ".join(DATASETS),
    )
    args = parser.parse_args(argv)

    keys = list(DATASETS.keys())
    if args.only:
        requested = [k.strip() for k in args.only.split(",") if k.strip()]
        unknown = sorted(set(requested) - set(DATASETS))
        if unknown:
            raise SystemExit(
                f"Unknown dataset key(s) {unknown}. Choices: {sorted(DATASETS)}"
            )
        keys = requested

    is_full_run = keys == list(DATASETS.keys())

    results: dict[str, ClassCoverageResult] = {}
    for key in keys:
        cfg = load_config(REPO_ROOT / DATASETS[key])
        csv_path = resolve_local_csv_path(cfg)
        result = compute_class_coverage(
            csv_path=csv_path,
            train_frac=float(cfg["data"]["train_frac"]),
            val_frac=float(cfg["data"]["val_frac"]),
        )
        results[key] = result
        print(f"[{key}] n_clean={result.n_clean:,}  classes={len(result.label_map)}  "
              f"dedup_dropped={result.n_dedup_dropped:,}  "
              f"dropna_dropped={result.n_dropna_dropped:,}  "
              f"unassigned={result.n_unassigned}")

    if not is_full_run:
        print(
            f"\n--only={args.only} used — canonical cross-dataset deliverables "
            f"({BY_SPLIT_CSV_NAME}, {SUMMARY_MD_NAME}, {FIGURE_STEM}.{{pdf,png,txt}}) "
            "are NOT written on a partial run. Re-run with no --only flag to "
            "produce them."
        )
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_by_split_csv(results, OUT_DIR / BY_SPLIT_CSV_NAME)
    write_summary_markdown(results, OUT_DIR / SUMMARY_MD_NAME)
    write_timeline_figure(results, OUT_DIR)

    print(f"\nWrote {OUT_DIR / BY_SPLIT_CSV_NAME}")
    print(f"Wrote {OUT_DIR / SUMMARY_MD_NAME}")
    print(f"Wrote {OUT_DIR / f'{FIGURE_STEM}.pdf'}")
    print(f"Wrote {OUT_DIR / f'{FIGURE_STEM}.png'}")
    print(f"Wrote {OUT_DIR / f'{FIGURE_STEM}.txt'}")


if __name__ == "__main__":
    main()
