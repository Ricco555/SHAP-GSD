"""
eval02_macro_denominators.py — missing-class-aware macro statistics
===================================================================
Implements specs/64 Part I §4.2 / Part II §14.2.

Why this exists:
  A bare "macro-F1" is not a defensible cross-dataset number when a large
  fraction of a dataset's label space can be absent from its test split — and
  on the runs this repo currently holds that fraction ranges from none at all
  to four fifths. *Which* denominator a table used therefore moves the headline
  number by several fold. This script computes and labels all three
  denominators for every resolved dataset, so no manuscript number has to be
  computed ad hoc anywhere else:

    full_label_space  — every class in label_map.json; absent classes score 0.
    test_supported    — classes with non-zero test support.
    closed_set        — classes present in BOTH the train and the test split.

  A fourth row, ``as_reported_by_pipeline``, carries the pipeline's own
  ``macro_f1``/``weighted_f1`` verbatim together with its self-documenting
  ``macro_f1_convention`` string, so the number already quoted elsewhere sits
  in the same file as the three recomputed ones and can be reconciled against
  them by eye.

Files read (read-only):
  <run>/artifacts/evaluation/metrics.json   — per-class P/R/F1/support (test)
  <run>/artifacts/label_map.json            — the canonical class list
  <run>/feature_store/train/labels.npy      — train-split class support

  Train-split support appears in no metrics file, so the closed-set denominator
  has to come from the feature store. A genuinely absent train feature store is
  a real on-disk state (one resolvable dataset is in it today): that dataset's
  ``closed_set`` row is emitted with NaN macro columns and an explaining
  ``notes`` value — never silently relabelled as ``test_supported``.

Files output (both under ``outputs/figures/evaluation/``):
  macro_stats_by_denominator.csv   — one row per (dataset, denominator)
  macro_stats_by_denominator.md    — the denominator legend + the positive
                                     control's outcome per dataset

Output-location note: ``OUT_DIR`` is the hardcoded
``REPO_ROOT/outputs/figures/evaluation/``, NOT resolved through
``explore._paths.paths()``/``SHAP_GSD_CONFIG`` (specs/64 D3): that variable
names a single active run, while this is a cross-dataset tool whose output
belongs to no one run's subtree. Restated here rather than relied on by
reference (specs/48 §1.1 pt 4).

This module emits no figure, so the figure conventions of
``explore/evaluation/AGENT.md`` do not apply; the ``.md`` companion plays the
role a figure's ``.txt`` reasoning file plays elsewhere.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # explore/AGENT.md mandatory convention

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from explore.evaluation._discover import (  # noqa: E402
    DEFAULT_OUT_DIR,
    DatasetRun,
    DatasetResolutionError,
    add_common_args,
    feature_store_dir,
    resolve_datasets_from_args,
)
from explore.evaluation._load import (  # noqa: E402
    EvalMetrics,
    as_float,
    load_eval_metrics,
    load_train_class_support,
)

log = logging.getLogger("eval02_macro_denominators")

OUT_DIR: Path = DEFAULT_OUT_DIR  # see module docstring for why this is hardcoded

CSV_NAME = "macro_stats_by_denominator.csv"
MD_NAME = "macro_stats_by_denominator.md"

#: The three recomputed denominators plus the pipeline's own reported figure.
DENOMINATOR_FULL = "full_label_space"
DENOMINATOR_TEST = "test_supported"
DENOMINATOR_CLOSED = "closed_set"
DENOMINATOR_REPORTED = "as_reported_by_pipeline"

#: Tolerance for the positive control comparing this script's ``test_supported``
#: macro-F1 against the pipeline's own reported ``macro_f1``. A mismatch is
#: LOGGED and recorded in ``notes``, never asserted: one rounding difference
#: must not abort a multi-dataset run (specs/64 D10).
POSITIVE_CONTROL_TOL = 1e-4

#: Legend lines mapping each ASCII denominator token to the LaTeX symbol the
#: manuscript-planning notation uses. Written verbatim into the ``.md``; the
#: CSV only ever carries the ASCII tokens (specs/64 §14.2).
DENOMINATOR_LEGEND: tuple[tuple[str, str, str], ...] = (
    (DENOMINATOR_FULL, r"$\mathrm{F1}_{\text{macro}}^{\mathcal{L}}$",
     "full label space; classes absent from the test split score 0"),
    (DENOMINATOR_TEST, r"$\mathrm{F1}_{\text{macro}}^{S_{\text{te}}}$",
     "test-supported classes only"),
    (DENOMINATOR_CLOSED,
     r"$\mathrm{F1}_{\text{macro}}^{S_{\text{tr}} \cap S_{\text{te}}}$",
     "closed set: classes present in both the train and the test split"),
    (DENOMINATOR_REPORTED, "—",
     "the pipeline's own macro_f1/weighted_f1, verbatim, under its own "
     "macro_f1_convention"),
)

CSV_COLUMNS: tuple[str, ...] = (
    "dataset",
    "denominator",
    "n_classes_in_denominator",
    "denominator_classes",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "notes",
)


def _macro_over(
    metrics: EvalMetrics, classes: list[str], score_absent_as_zero: bool,
) -> dict[str, float]:
    """Average per-class P/R/F1 over one denominator's class set.

    Args:
        metrics: The dataset's normalised ``metrics.json``.
        classes: The denominator's class set, in canonical order.
        score_absent_as_zero: When ``True``, a class with no ``per_class`` row
            (or zero support) contributes 0.0 rather than being skipped — the
            ``full_label_space`` convention.

    Returns:
        ``{macro_precision, macro_recall, macro_f1, weighted_f1}``; all ``NaN``
        when the class set is empty.
    """
    if not classes:
        return dict.fromkeys(
            ("macro_precision", "macro_recall", "macro_f1", "weighted_f1"),
            float("nan"),
        )
    collected: dict[str, list[float]] = {"precision": [], "recall": [], "f1": []}
    supports: list[float] = []
    for class_name in classes:
        row = metrics.per_class.get(class_name)
        support = metrics.support(class_name)
        for metric in collected:
            if row is None:
                value = 0.0 if score_absent_as_zero else float("nan")
            else:
                value = as_float(row.get(metric))
                if not np.isfinite(value):
                    value = 0.0 if score_absent_as_zero else float("nan")
            collected[metric].append(value)
        supports.append(support if np.isfinite(support) else 0.0)

    weights = np.asarray(supports, dtype=float)
    f1_values = np.asarray(collected["f1"], dtype=float)
    weight_total = float(np.nansum(weights))
    weighted_f1 = (
        float(np.nansum(f1_values * weights) / weight_total)
        if weight_total > 0 else float("nan")
    )
    return {
        "macro_precision": float(np.nanmean(collected["precision"])),
        "macro_recall": float(np.nanmean(collected["recall"])),
        "macro_f1": float(np.nanmean(f1_values)),
        "weighted_f1": weighted_f1,
    }


def compute_denominator_rows(run: DatasetRun) -> list[dict[str, object]]:
    """Compute all four denominator rows for one dataset.

    Args:
        run: The resolved dataset run.

    Returns:
        Four row dicts carrying exactly :data:`CSV_COLUMNS`.
    """
    metrics = load_eval_metrics(run)
    all_classes = list(metrics.class_names)
    test_classes = [c for c in all_classes if metrics.is_present_in_test(c)]

    train_counts = load_train_class_support(run, all_classes)
    if train_counts is None:
        closed_classes: list[str] | None = None
        closed_note = (
            f"train feature store absent at "
            f"{feature_store_dir(run) / 'train' / 'labels.npy'}; closed-set "
            f"denominator not computable"
        )
    else:
        train_classes = {
            name for idx, name in enumerate(all_classes)
            if idx < train_counts.shape[0] and int(train_counts[idx]) > 0
        }
        closed_classes = [c for c in test_classes if c in train_classes]
        closed_note = ""

    # CLAUDE.md CODING STANDARDS 7 — structural invariants, asserted in
    # production code. The positive control below is deliberately NOT an
    # assertion (see POSITIVE_CONTROL_TOL).
    assert set(test_classes).issubset(set(all_classes)), (
        f"{run.name}: test-supported classes are not a subset of label_map.json"
    )
    if closed_classes is not None:
        assert set(closed_classes).issubset(set(test_classes)), (
            f"{run.name}: closed-set classes are not a subset of the "
            f"test-supported classes"
        )

    rows: list[dict[str, object]] = []

    def _row(
        denominator: str, classes: list[str] | None, absent_zero: bool, note: str,
    ) -> dict[str, object]:
        if classes is None:
            stats = dict.fromkeys(
                ("macro_precision", "macro_recall", "macro_f1", "weighted_f1"),
                float("nan"),
            )
            n_classes: object = float("nan")
            joined = ""
        else:
            stats = _macro_over(metrics, classes, absent_zero)
            n_classes = len(classes)
            joined = "|".join(classes)
        return {
            "dataset": run.label,
            "denominator": denominator,
            "n_classes_in_denominator": n_classes,
            "denominator_classes": joined,
            **stats,
            "notes": note,
        }

    rows.append(_row(
        DENOMINATOR_FULL, all_classes, True,
        f"{len(all_classes) - len(test_classes)} of {len(all_classes)} classes "
        f"have zero test support and are scored 0 here",
    ))
    test_row = _row(DENOMINATOR_TEST, test_classes, False, "")
    rows.append(test_row)
    rows.append(_row(DENOMINATOR_CLOSED, closed_classes, False, closed_note))

    reported_note = f"macro_f1_convention: {metrics.macro_f1_convention}"
    delta = abs(float(test_row["macro_f1"]) - metrics.macro_f1)
    if np.isfinite(delta):
        if delta <= POSITIVE_CONTROL_TOL:
            log.info(
                "%s: positive control OK — recomputed test_supported macro-F1 "
                "matches the reported macro_f1 to %.1e (%.6f)",
                run.label, POSITIVE_CONTROL_TOL, metrics.macro_f1,
            )
            reported_note += (
                f"; positive control OK (recomputed test_supported macro-F1 "
                f"agrees within {POSITIVE_CONTROL_TOL:.0e})"
            )
        else:
            log.warning(
                "%s: positive control MISMATCH — recomputed test_supported "
                "macro-F1 %.6f vs reported %.6f (delta %.6f). The reported "
                "figure may use a different convention than its "
                "macro_f1_convention string claims.",
                run.label, float(test_row["macro_f1"]), metrics.macro_f1, delta,
            )
            reported_note += (
                f"; positive control MISMATCH: recomputed test_supported "
                f"macro-F1 differs by {delta:.6f}"
            )
    rows.append({
        "dataset": run.label,
        "denominator": DENOMINATOR_REPORTED,
        "n_classes_in_denominator": metrics.n_classes_present_in_test,
        "denominator_classes": "",
        "macro_precision": float("nan"),
        "macro_recall": float("nan"),
        "macro_f1": metrics.macro_f1,
        "weighted_f1": metrics.weighted_f1,
        "notes": reported_note,
    })
    return rows


def build_macro_stats(runs: list[DatasetRun]) -> pd.DataFrame:
    """Build the full denominator table across every resolved dataset.

    Args:
        runs: The resolved dataset runs, in output order.

    Returns:
        A frame carrying exactly :data:`CSV_COLUMNS`, four rows per dataset.
    """
    rows: list[dict[str, object]] = []
    for run in runs:
        rows.extend(compute_denominator_rows(run))
    return pd.DataFrame(rows, columns=list(CSV_COLUMNS))


def write_legend_markdown(frame: pd.DataFrame, path: Path) -> None:
    """Write the denominator legend and per-dataset spread beside the CSV.

    Args:
        frame: The denominator table.
        path: Destination ``.md`` path.
    """
    lines = [
        "# eval02 — macro statistics under all three denominators",
        "",
        "Generated by `explore/evaluation/eval02_macro_denominators.py` "
        "(specs/64 §14.2).",
        "",
        "`macro_stats_by_denominator.csv` is the single source of truth for "
        "every cross-dataset macro statistic. Any table or figure that quotes a "
        "macro-F1 must state which of these denominators it used — the three "
        "can differ by several fold on a dataset whose test split covers only "
        "part of the label space.",
        "",
        "## Denominator legend",
        "",
        "| token | symbol | meaning |",
        "|---|---|---|",
    ]
    for token, symbol, meaning in DENOMINATOR_LEGEND:
        lines.append(f"| `{token}` | {symbol} | {meaning} |")
    lines += ["", "## Per-dataset spread", ""]
    if frame.empty:
        lines.append("_No dataset resolved._")
    else:
        lines.append("| dataset | denominator | n_classes | macro_f1 | weighted_f1 | notes |")
        lines.append("|---|---|---|---|---|---|")
        for _, row in frame.iterrows():
            lines.append(
                f"| {row['dataset']} | `{row['denominator']}` | "
                f"{row['n_classes_in_denominator']} | "
                f"{row['macro_f1']:.4f} | {row['weighted_f1']:.4f} | "
                f"{row['notes'] or '—'} |"
            )
    lines.append("")
    path.write_text("\n".join(lines))
    log.info("wrote %s", path)


def run(runs: list[DatasetRun], out_dir: Path) -> pd.DataFrame:
    """Build and write both eval02 artifacts.

    Args:
        runs: The resolved dataset runs to ingest.
        out_dir: Directory both artifacts are written into.

    Returns:
        The denominator frame that was written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = build_macro_stats(runs)
    csv_path = out_dir / CSV_NAME
    frame.to_csv(csv_path, index=False)
    log.info("wrote %s (%d rows)", csv_path, len(frame))
    write_legend_markdown(frame, out_dir / MD_NAME)
    return frame


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector, or ``None`` to read ``sys.argv``.

    Returns:
        Process exit code (0 on success, 2 on a dataset-resolution failure).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute macro precision/recall/F1 under all three class-space "
            "denominators, plus the pipeline's own reported figure, for every "
            "resolved dataset (specs/64 Part I §4.2)."
        )
    )
    add_common_args(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        runs = resolve_datasets_from_args(args)
    except DatasetResolutionError as exc:
        log.error("%s", exc)
        return 2

    frame = run(runs, Path(args.out_dir))
    log.info("eval02 complete: %d datasets, %d rows", len(runs), len(frame))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
