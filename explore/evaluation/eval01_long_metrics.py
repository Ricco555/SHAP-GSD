"""
eval01_long_metrics.py — long-format per-dataset metrics aggregator
===================================================================
Implements specs/64 Part I §4.1 / Part II §14.1.

What it does:
  Turns every resolved dataset run's classification metrics and its three
  coalition-space fidelity/stability summaries into ONE tidy table — one row
  per (dataset, class_name, coalition_space, metric) — so that no downstream
  module ever needs dataset-specific knowledge of which classes exist or which
  optional keys a given ``summary*.json`` happens to carry.

  The class-coverage gap (a dataset where a large fraction of the label space
  has zero test support) is encoded *structurally*, in the
  ``class_present_in_test`` column, rather than as a caveat a reader has to
  remember: every downstream aggregation can filter on that column instead of
  re-deriving the rule per module.

Files read (read-only; this package never writes into ``runs/``):
  <run>/artifacts/evaluation/metrics.json      — per-class P/R/F1/support
  <run>/artifacts/label_map.json               — the canonical class list
  <run>/outputs/metrics/summary.json           — phi_F fidelity + intra-run stability
  <run>/outputs/metrics/summary_temporal.json  — phi_T fidelity
  <run>/outputs/metrics/summary_novelty.json   — phi_N fidelity
  <run>/outputs/metrics/stability.csv          — per-flow intra-run phi std (phi_F only)

  ``outputs/metrics/table2*.txt`` is deliberately NOT parsed: those are
  fixed-width human-readable renderings carrying an em-dash sentinel for
  undefined cells, and their machine-readable JSON/CSV siblings carry the same
  numbers plus diagnostics (specs/64 D4).

Files output (both under ``outputs/figures/evaluation/``):
  eval_long_metrics.csv           — the tidy table
  eval_long_metrics_coverage.md   — one row per dataset: coverage preamble

Output-location note: ``OUT_DIR`` is the hardcoded
``REPO_ROOT/outputs/figures/evaluation/``, NOT resolved through
``explore._paths.paths()``/``SHAP_GSD_CONFIG`` (specs/64 D3). ``SHAP_GSD_CONFIG``
names a *single active run*; this is a cross-dataset tool, and every
``configs/experiment_*.yaml`` sets a non-empty ``run.dir``, so a config-driven
resolution would file a cross-dataset artifact inside one dataset's ``runs/``
subtree. Stating this here rather than relying on the precedent in
``explore/class_coverage_analysis.py`` by reference alone (specs/48 §1.1 pt 4).

This module emits no figure, so the figure conventions of
``explore/evaluation/AGENT.md`` do not apply to it; the ``.md`` companion plays
the role the ``.txt`` reasoning file plays for a figure-producing script.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # explore/AGENT.md mandatory convention (no display, no GUI)

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
    resolve_datasets_from_args,
)
from explore.evaluation._load import (  # noqa: E402
    COALITION_SPACES,
    STABILITY_SCOPE_NOTE,
    CoalitionSummary,
    EvalMetrics,
    as_float,
    load_all_coalition_summaries,
    load_eval_metrics,
    load_stability_frame,
)

log = logging.getLogger("eval01_long_metrics")

OUT_DIR: Path = DEFAULT_OUT_DIR  # see module docstring for why this is hardcoded

LONG_CSV_NAME = "eval_long_metrics.csv"
COVERAGE_MD_NAME = "eval_long_metrics_coverage.md"

#: ``coalition_space`` value for the classification metrics (precision, recall,
#: f1, support). Those come from the model's confusion matrix and belong to no
#: coalition space at all; giving them a fourth, explicit value keeps the tidy
#: table's key unique without pretending they are phi_F quantities.
CLASSIFICATION_SPACE = "none"

#: Metrics read out of each ``summary*.json`` per-class dict, in output order.
#: ``fidelity_*_std`` and ``n_flows`` extend Part I §4.1's enum: they sit
#: alongside the mean in the same ``summary*.json`` per-class dict, and their
#: consumer is ``eval08``'s fidelity comparison table, where a cross-dataset
#: fidelity mean is unreadable without its dispersion and the flow count it was
#: computed over. Dropping them here would force a second read of the same file
#: (specs/64 §14.1.2).
FIDELITY_METRIC_KEYS: tuple[str, ...] = (
    "fidelity_plus",
    "fidelity_plus_std",
    "fidelity_minus",
    "fidelity_minus_std",
    "n_flows",
)

#: Intra-run stability metrics. Emitted for every coalition space, but only the
#: feature space can carry a real value (specs/64 §14.1.2).
STABILITY_METRIC_KEYS: tuple[str, ...] = (
    "stability_intrarun_mean",
    "stability_intrarun_max",
)

LONG_COLUMNS: tuple[str, ...] = (
    "dataset",
    "class_name",
    "coalition_space",
    "metric",
    "value",
    "class_present_in_test",
    "notes",
)


#: Tolerance for the positive control comparing ``stability.csv``'s per-class
#: mean against ``summary.json``'s own ``stability`` scalar. The two are
#: independent renderings of the same quantity, and the CSV is the primary
#: source; a disagreement is LOGGED and recorded in ``notes`` rather than
#: asserted, so one rounding difference cannot abort a multi-dataset run.
STABILITY_CONTROL_TOL = 1e-5


def _stability_by_class(run: DatasetRun) -> dict[str, tuple[float, float]]:
    """Reduce a run's per-flow intra-run stability CSV to per-class scalars.

    ``stability.csv``'s columns are ``class_name,edge_id,mean_phi_std,
    max_phi_std`` — feature-group phi only. The two emitted scalars are the
    class's mean of ``mean_phi_std`` and its worst ``max_phi_std``.

    Args:
        run: The resolved dataset run.

    Returns:
        ``{class_name: (intrarun_mean, intrarun_max)}``; empty when the file is
        absent.
    """
    frame = load_stability_frame(run)
    if frame is None or frame.empty:
        return {}
    out: dict[str, tuple[float, float]] = {}
    for class_name, group in frame.groupby("class_name"):
        out[str(class_name)] = (
            float(group["mean_phi_std"].mean()),
            float(group["max_phi_std"].max()),
        )
    return out


def _classification_rows(
    metrics: EvalMetrics, label: str,
) -> list[dict[str, object]]:
    """Emit the precision/recall/f1/support rows for one dataset.

    Every class in ``label_map.json`` gets a row per metric, including classes
    with zero test support: an absent row would silently disappear from a
    downstream join (specs/64 D10).

    Args:
        metrics: The dataset's normalised ``metrics.json``.
        label: Display label written into the ``dataset`` column.

    Returns:
        One dict per emitted row.
    """
    rows: list[dict[str, object]] = []
    schema_note = f"metrics.json schema_source={metrics.schema_source}"
    for class_name in metrics.class_names:
        present = metrics.is_present_in_test(class_name)
        per_class = metrics.per_class.get(class_name)
        note = schema_note
        if per_class is None:
            note = (
                f"{schema_note}; class is in label_map.json but has no per_class "
                f"row in metrics.json"
            )
        for metric in ("precision", "recall", "f1", "support"):
            value = (
                as_float((per_class or {}).get(metric))
                if per_class is not None
                else float("nan")
            )
            rows.append({
                "dataset": label,
                "class_name": class_name,
                "coalition_space": CLASSIFICATION_SPACE,
                "metric": metric,
                "value": value,
                "class_present_in_test": present,
                "notes": note,
            })
    return rows


def _coalition_rows(
    metrics: EvalMetrics,
    label: str,
    summaries: dict[str, CoalitionSummary],
    stability: dict[str, tuple[float, float]],
) -> list[dict[str, object]]:
    """Emit the fidelity and intra-run-stability rows for one dataset.

    Args:
        metrics: The dataset's normalised ``metrics.json`` (for the class list
            and the ``class_present_in_test`` flag).
        label: Display label written into the ``dataset`` column.
        summaries: ``{space: CoalitionSummary}`` for all three spaces.
        stability: Per-class intra-run stability scalars (feature space only).

    Returns:
        One dict per emitted row.
    """
    rows: list[dict[str, object]] = []
    for space in COALITION_SPACES:
        summary = summaries[space]
        for class_name in metrics.class_names:
            present = metrics.is_present_in_test(class_name)
            per_class = summary.per_class.get(class_name)

            if not summary.present:
                base_note = (
                    f"no {space} coalition-space summary on disk at "
                    f"{summary.path}"
                )
            elif per_class is None:
                base_note = (
                    f"class has no entry in {summary.path.name if summary.path else space}"
                    f" (no explained flows for it in this coalition space)"
                )
            else:
                base_note = ""

            for metric in FIDELITY_METRIC_KEYS:
                value = (
                    as_float(per_class.get(metric))
                    if per_class is not None
                    else float("nan")
                )
                rows.append({
                    "dataset": label,
                    "class_name": class_name,
                    "coalition_space": space,
                    "metric": metric,
                    "value": value,
                    "class_present_in_test": present,
                    "notes": base_note,
                })

            # Intra-run stability: real values for the feature space only.
            if space == "feature":
                pair = stability.get(class_name)
                if pair is not None:
                    values = {
                        "stability_intrarun_mean": pair[0],
                        "stability_intrarun_max": pair[1],
                    }
                    stab_note = base_note
                    # Positive control: summary.json carries the same quantity
                    # as a rounded scalar. The CSV is authoritative; a
                    # disagreement would mean the two are not the same
                    # definition, which must be visible rather than silent.
                    reported = (
                        as_float(per_class.get("stability"))
                        if per_class is not None and "stability" in per_class
                        else float("nan")
                    )
                    if np.isfinite(reported):
                        delta = abs(reported - pair[0])
                        if delta > STABILITY_CONTROL_TOL:
                            log.warning(
                                "%s/%s: stability.csv mean %.8g disagrees with "
                                "summary.json's stability %.8g (delta %.2g); "
                                "the two may not be the same statistic",
                                label, class_name, pair[0], reported, delta,
                            )
                            stab_note = (
                                (base_note + "; " if base_note else "")
                                + f"stability.csv mean and summary.json's "
                                  f"'stability' scalar disagree by {delta:.3g}"
                            )
                elif per_class is not None and "stability" in per_class:
                    values = {
                        "stability_intrarun_mean": as_float(per_class["stability"]),
                        "stability_intrarun_max": float("nan"),
                    }
                    stab_note = (
                        "stability.csv unavailable for this class; mean taken "
                        "from summary.json's per-class 'stability' key, max not "
                        "recoverable"
                    )
                else:
                    values = dict.fromkeys(STABILITY_METRIC_KEYS, float("nan"))
                    stab_note = base_note or (
                        "no intra-run stability recorded for this class"
                    )
            else:
                values = dict.fromkeys(STABILITY_METRIC_KEYS, float("nan"))
                stab_note = STABILITY_SCOPE_NOTE

            for metric in STABILITY_METRIC_KEYS:
                rows.append({
                    "dataset": label,
                    "class_name": class_name,
                    "coalition_space": space,
                    "metric": metric,
                    "value": values[metric],
                    "class_present_in_test": present,
                    "notes": stab_note,
                })
    return rows


def build_long_metrics(runs: list[DatasetRun]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the tidy long-format metrics table and its coverage preamble.

    Args:
        runs: The resolved dataset runs to ingest, in output order.

    Returns:
        ``(long_frame, coverage_frame)``. ``long_frame`` carries exactly
        :data:`LONG_COLUMNS`; ``coverage_frame`` has one row per dataset.
    """
    long_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    for run in runs:
        metrics = load_eval_metrics(run)
        summaries = load_all_coalition_summaries(run)
        stability = _stability_by_class(run)
        log.info(
            "%s: %d classes, %d present in test, coalition spaces on disk: %s",
            run.label, metrics.n_classes_total, metrics.n_classes_present_in_test,
            [s for s in COALITION_SPACES if summaries[s].present],
        )

        long_rows.extend(_classification_rows(metrics, run.label))
        long_rows.extend(_coalition_rows(metrics, run.label, summaries, stability))

        coverage_rows.append({
            "dataset": run.label,
            "run_dir": str(run.run_dir),
            "variant_suffix": run.variant_suffix,
            "n_classes_total": metrics.n_classes_total,
            "n_classes_present_in_test": metrics.n_classes_present_in_test,
            "classes_absent_from_test": "|".join(metrics.classes_absent_from_test),
            "schema_source": metrics.schema_source,
            "macro_f1_convention": metrics.macro_f1_convention,
        })

    long_frame = pd.DataFrame(long_rows, columns=list(LONG_COLUMNS))
    coverage_frame = pd.DataFrame(coverage_rows)

    # CLAUDE.md CODING STANDARDS 7 — assertions run in production code.
    assert not long_frame.duplicated(
        subset=["dataset", "class_name", "coalition_space", "metric"]
    ).any(), "eval01 emitted duplicate (dataset, class, coalition_space, metric) keys"
    return long_frame, coverage_frame


def write_coverage_markdown(coverage: pd.DataFrame, path: Path) -> None:
    """Write the per-dataset coverage preamble beside the tidy table.

    The coverage constraint travels with the numbers rather than living as a
    remembered caveat: a reader of ``eval_long_metrics.csv`` who also opens
    this file can see, per dataset, how much of the label space the test split
    actually contains and whether the coverage fields were read or derived.

    Args:
        coverage: One row per dataset, as returned by :func:`build_long_metrics`.
        path: Destination ``.md`` path.
    """
    lines = [
        "# eval01 — class-coverage preamble for `eval_long_metrics.csv`",
        "",
        "Generated by `explore/evaluation/eval01_long_metrics.py` (specs/64 §14.1).",
        "",
        "Every row of `eval_long_metrics.csv` carries a `class_present_in_test`",
        "flag. A cross-dataset macro statistic computed without filtering on it",
        "silently mixes scored classes with classes that had zero test support.",
        "`schema_source = derived` marks a dataset whose `metrics.json` predates",
        "`src/model/evaluator.py`'s self-documenting coverage fields, so its",
        "coverage numbers were reconstructed from `label_map.json` + per-class",
        "support rather than read.",
        "",
    ]
    if coverage.empty:
        lines.append("_No dataset resolved._")
    else:
        cols = [
            "dataset", "n_classes_total", "n_classes_present_in_test",
            "classes_absent_from_test", "schema_source", "run_dir",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for _, row in coverage.iterrows():
            lines.append(
                "| " + " | ".join(str(row[c]) if str(row[c]) else "—" for c in cols) + " |"
            )
        lines.append("")
        lines.append("## `macro_f1_convention`, verbatim per dataset")
        lines.append("")
        for _, row in coverage.iterrows():
            lines.append(f"- **{row['dataset']}**: {row['macro_f1_convention']}")
    lines.append("")
    path.write_text("\n".join(lines))
    log.info("wrote %s", path)


def run(runs: list[DatasetRun], out_dir: Path) -> pd.DataFrame:
    """Build and write both eval01 artifacts.

    Args:
        runs: The resolved dataset runs to ingest.
        out_dir: Directory both artifacts are written into.

    Returns:
        The tidy long-format frame that was written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    long_frame, coverage = build_long_metrics(runs)
    csv_path = out_dir / LONG_CSV_NAME
    long_frame.to_csv(csv_path, index=False)
    log.info("wrote %s (%d rows)", csv_path, len(long_frame))
    write_coverage_markdown(coverage, out_dir / COVERAGE_MD_NAME)
    return long_frame


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector, or ``None`` to read ``sys.argv``.

    Returns:
        Process exit code (0 on success, 2 on a dataset-resolution failure).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate every resolved dataset's classification metrics and "
            "three coalition-space fidelity/stability summaries into one tidy, "
            "long-format table (specs/64 Part I §4.1)."
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
    n_finite = int(np.isfinite(frame["value"].to_numpy(dtype=float)).sum())
    log.info(
        "eval01 complete: %d datasets, %d rows, %d finite values",
        len(runs), len(frame), n_finite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
