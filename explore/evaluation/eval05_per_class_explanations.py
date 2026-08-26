"""
eval05_per_class_explanations.py — per-class explanation aggregation
====================================================================
Implements specs/64 Part I §4.5 / Part II §14.4.

Why this exists:
  The pipeline ships per-*flow* explanation records; nothing aggregates them to
  per-*class* summaries. Both the global-coherence scoring (eval04) and the
  Source-3 consistency check need the same aggregation, so it is built once,
  here, and consumed from this script's CSV rather than recomputed — that is
  what makes "scored against the same vector" a checkable property instead of a
  hope.

What it computes, per (dataset, class):
  - The full feature-group ranking by mean |phi_F| across the class's flows,
    plus the mean magnitudes themselves, so a downstream rank correlation does
    not have to re-walk the JSONs.
  - The fraction of flows with a non-degenerate temporal neighbourhood, and the
    mean |phi_T| magnitude among those.
  - The fraction of flows with a non-zero endpoint-novelty attribution and its
    mean magnitude — expected to be near zero, which is a measured finding and
    not a bug in this aggregation.
  - ``n_degenerate_novelty_players``, aggregated. Without it a near-zero
    novelty aggregate is uninterpretable: it cannot be told apart from a
    measurement artefact. With it, "both endpoints were degenerate coalition
    players in every flow" is directly visible.
  - An explicit ``n_flows_aggregated = 0`` row for every class that has no
    explained flows at all, rather than a missing row that would silently
    disappear from a downstream join.

Files read (read-only):
  <run>/artifacts/label_map.json                        — the canonical class list
  <run>/artifacts/evaluation/metrics.json               — class_present_in_test
  <run>/outputs/explanations/summary.csv                — per-flow top-1 signals
  <run>/outputs/explanations/<Class>/<edge_id>.json     — full per-flow vectors

  The cheap/expensive split is deliberate: the rollup CSV supplies the per-flow
  counts and top-1 signals, and the raw JSONs are opened only for the quantities
  that need a full per-flow vector.

Files output (under ``outputs/figures/evaluation/``):
  per_class_explanation_summary.csv — one row per (dataset, class)

Output-location note: ``OUT_DIR`` is the hardcoded
``REPO_ROOT/outputs/figures/evaluation/``, NOT resolved through
``explore._paths.paths()``/``SHAP_GSD_CONFIG`` (specs/64 D3): that variable
selects one active run, while this script's single output table spans every
resolved dataset and belongs inside no run's subtree. Restated here rather than
relied on by reference (specs/48 §1.1 pt 4).

This module emits no figure, so the figure conventions of
``explore/evaluation/AGENT.md`` do not apply to it.

Feature-group counts are never hardcoded: each record's own
``feature_group_names`` list defines the group set for that dataset.
"""

from __future__ import annotations

import argparse
import json
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
    outputs_dir,
    resolve_datasets_from_args,
)
from explore.evaluation._load import load_eval_metrics  # noqa: E402

log = logging.getLogger("eval05_per_class_explanations")

OUT_DIR: Path = DEFAULT_OUT_DIR  # see module docstring for why this is hardcoded

CSV_NAME = "per_class_explanation_summary.csv"

#: How many top feature groups the ``top5_feature_groups_by_mean_magnitude``
#: column lists. Deliberately NOT wired to the shared ``--top-k`` flag: the
#: column's name states its length, so letting a CLI flag change it would make
#: a three-entry cell sit under a header claiming five. ``full_feature_ranking``
#: carries the complete order for any caller that wants a different k.
DEFAULT_TOP_LIST = 5

#: Separator for the ordered-list columns. Pipe rather than comma so the CSV
#: needs no nested quoting to stay readable.
LIST_SEP = "|"

CSV_COLUMNS: tuple[str, ...] = (
    "dataset",
    "class_name",
    "class_present_in_test",
    "n_flows_aggregated",
    "top5_feature_groups_by_mean_magnitude",
    "full_feature_ranking",
    "full_feature_mean_magnitudes",
    "mean_phi_f_magnitude",
    "frac_flows_with_neighbors",
    "mean_phi_t_magnitude_when_present",
    "frac_flows_novelty_nonzero",
    "mean_novelty_magnitude",
    "mean_n_degenerate_novelty_players",
    "frac_flows_both_endpoints_degenerate",
    "notes",
)


def load_explanation_summary(run: DatasetRun) -> pd.DataFrame | None:
    """Load a run's per-flow explanation rollup.

    Args:
        run: The resolved dataset run.

    Returns:
        The rollup frame, or ``None`` when it is absent (impossible for a run
        that passed the completeness predicate, so ``None`` means a
        ``--run-dir`` pin bypassed it).
    """
    path = outputs_dir(run) / "explanations" / "summary.csv"
    if not path.is_file():
        log.warning("%s: no explanations summary.csv at %s", run.name, path)
        return None
    return pd.read_csv(path)


def iter_class_records(run: DatasetRun, class_name: str) -> list[dict]:
    """Read every per-flow explanation JSON for one class.

    Args:
        run: The resolved dataset run.
        class_name: Class directory name, exactly as it appears in
            ``label_map.json`` (never case-normalised — one resolvable dataset
            mixes capitalised and lowercase class names).

    Returns:
        The decoded records, in sorted filename order. Empty when the class has
        no directory or an empty one.
    """
    class_dir = outputs_dir(run) / "explanations" / class_name
    if not class_dir.is_dir():
        return []
    records: list[dict] = []
    for path in sorted(class_dir.glob("*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("%s: skipping unreadable explanation %s: %s",
                        run.name, path, exc)
    return records


def _empty_row(
    run: DatasetRun, class_name: str, present: bool, note: str,
) -> dict[str, object]:
    """Build the explicit zero-flow row for a class with no explanations.

    Args:
        run: The resolved dataset run.
        class_name: The class with no explained flows.
        present: Whether the class has non-zero test support.
        note: Why the class has no flows.

    Returns:
        A row dict carrying exactly :data:`CSV_COLUMNS`.
    """
    return {
        "dataset": run.label,
        "class_name": class_name,
        "class_present_in_test": present,
        "n_flows_aggregated": 0,
        "top5_feature_groups_by_mean_magnitude": "",
        "full_feature_ranking": "",
        "full_feature_mean_magnitudes": "",
        "mean_phi_f_magnitude": float("nan"),
        "frac_flows_with_neighbors": float("nan"),
        "mean_phi_t_magnitude_when_present": float("nan"),
        "frac_flows_novelty_nonzero": float("nan"),
        "mean_novelty_magnitude": float("nan"),
        "mean_n_degenerate_novelty_players": float("nan"),
        "frac_flows_both_endpoints_degenerate": float("nan"),
        "notes": note,
    }


def aggregate_class(
    run: DatasetRun,
    class_name: str,
    present: bool,
    rollup: pd.DataFrame | None,
    top_list: int = DEFAULT_TOP_LIST,
) -> dict[str, object]:
    """Aggregate one class's per-flow explanations into a single row.

    Args:
        run: The resolved dataset run.
        class_name: Class name as in ``label_map.json``.
        present: Whether the class has non-zero test support.
        rollup: The run's ``summary.csv`` frame, or ``None``.
        top_list: How many groups to list in the top-k column.

    Returns:
        A row dict carrying exactly :data:`CSV_COLUMNS`.
    """
    rows = (
        rollup[rollup["class_name"] == class_name]
        if rollup is not None and "class_name" in rollup.columns
        else None
    )
    n_rollup = 0 if rows is None else int(len(rows))

    records = iter_class_records(run, class_name)
    if not records:
        note = (
            f"no explained flows for this class "
            f"({'directory absent or empty' if n_rollup == 0 else f'{n_rollup} rollup rows but no JSONs'})"
        )
        return _empty_row(run, class_name, present, note)

    # --- feature-group ranking, from each record's own group-name list -------
    accum: dict[str, list[float]] = {}
    phi_f_flow_means: list[float] = []
    for rec in records:
        names = rec.get("feature_group_names") or []
        values = rec.get("feature_group_shap") or []
        if len(names) != len(values):
            log.warning(
                "%s/%s: record %s has %d group names but %d values; skipped",
                run.name, class_name, rec.get("edge_id"), len(names), len(values),
            )
            continue
        magnitudes = [abs(float(v)) for v in values]
        for name, magnitude in zip(names, magnitudes):
            accum.setdefault(name, []).append(magnitude)
        if magnitudes:
            phi_f_flow_means.append(float(np.mean(magnitudes)))

    group_means = {name: float(np.mean(vals)) for name, vals in accum.items()}
    ranking = sorted(group_means, key=lambda g: (-group_means[g], g))

    # --- temporal (phi_T) coalition space -----------------------------------
    phi_t_means: list[float] = []
    n_with_neighbors = 0
    for rec in records:
        neighbor_shap = rec.get("neighbor_shap") or []
        if len(neighbor_shap) >= 1:
            n_with_neighbors += 1
            phi_t_means.append(float(np.mean([abs(float(v)) for v in neighbor_shap])))

    # --- node novelty (phi_N) coalition space -------------------------------
    novelty_magnitudes: list[float] = []
    n_novelty_nonzero = 0
    degenerate_counts: list[float] = []
    n_both_degenerate = 0
    for rec in records:
        src = abs(float(rec.get("src_novelty_shap") or 0.0))
        dst = abs(float(rec.get("dst_novelty_shap") or 0.0))
        novelty_magnitudes.append((src + dst) / 2.0)
        if src > 0.0 or dst > 0.0:
            n_novelty_nonzero += 1
        degenerate = rec.get("n_degenerate_novelty_players")
        if degenerate is not None:
            degenerate_counts.append(float(degenerate))
            if float(degenerate) >= 2.0:
                n_both_degenerate += 1

    n_flows = len(records)
    notes = ""
    if n_rollup and n_rollup != n_flows:
        notes = (
            f"summary.csv lists {n_rollup} flows for this class but "
            f"{n_flows} JSON records were readable"
        )
    if not degenerate_counts:
        notes = (notes + "; " if notes else "") + (
            "records carry no n_degenerate_novelty_players field, so a "
            "near-zero novelty aggregate cannot be attributed to degenerate "
            "coalition players"
        )

    return {
        "dataset": run.label,
        "class_name": class_name,
        "class_present_in_test": present,
        "n_flows_aggregated": n_flows,
        "top5_feature_groups_by_mean_magnitude": LIST_SEP.join(ranking[:top_list]),
        "full_feature_ranking": LIST_SEP.join(ranking),
        "full_feature_mean_magnitudes": LIST_SEP.join(
            f"{group_means[g]:.10g}" for g in ranking
        ),
        "mean_phi_f_magnitude": (
            float(np.mean(phi_f_flow_means)) if phi_f_flow_means else float("nan")
        ),
        "frac_flows_with_neighbors": n_with_neighbors / n_flows,
        "mean_phi_t_magnitude_when_present": (
            float(np.mean(phi_t_means)) if phi_t_means else float("nan")
        ),
        "frac_flows_novelty_nonzero": n_novelty_nonzero / n_flows,
        "mean_novelty_magnitude": (
            float(np.mean(novelty_magnitudes)) if novelty_magnitudes else float("nan")
        ),
        "mean_n_degenerate_novelty_players": (
            float(np.mean(degenerate_counts)) if degenerate_counts else float("nan")
        ),
        "frac_flows_both_endpoints_degenerate": (
            n_both_degenerate / n_flows if degenerate_counts else float("nan")
        ),
        "notes": notes,
    }


def build_per_class_summary(
    runs: list[DatasetRun], top_list: int = DEFAULT_TOP_LIST,
) -> pd.DataFrame:
    """Aggregate every resolved dataset's explanations to one row per class.

    Args:
        runs: The resolved dataset runs, in output order.
        top_list: How many groups to list in the top-k column.

    Returns:
        A frame carrying exactly :data:`CSV_COLUMNS`.
    """
    rows: list[dict[str, object]] = []
    for run in runs:
        metrics = load_eval_metrics(run)
        rollup = load_explanation_summary(run)
        # Count this run's own rows, not the last ``n_classes_total`` of the
        # accumulated list: ``n_classes_total`` is read from metrics.json and
        # need not equal ``len(class_names)`` (it is derived only on the legacy
        # schema), and a value of 0 would make ``rows[-0:]`` the WHOLE list.
        run_rows: list[dict[str, object]] = []
        for class_name in metrics.class_names:
            run_rows.append(aggregate_class(
                run, class_name, metrics.is_present_in_test(class_name),
                rollup, top_list,
            ))
        rows.extend(run_rows)
        n_zero = sum(1 for r in run_rows if r["n_flows_aggregated"] == 0)
        log.info(
            "%s: aggregated %d classes (%d with zero explained flows)",
            run.label, len(run_rows), n_zero,
        )
    return pd.DataFrame(rows, columns=list(CSV_COLUMNS))


def parse_ranking(value: object) -> list[str]:
    """Split a pipe-joined ranking cell back into an ordered list.

    Exposed so eval04 can consume this script's CSV rather than re-deriving
    the same ranking from the JSONs (specs/64 §14.5.1).

    Args:
        value: A ``full_feature_ranking`` / ``top5_*`` cell.

    Returns:
        The ordered group names; empty for an empty or missing cell.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    return [part for part in text.split(LIST_SEP) if part] if text else []


def parse_magnitudes(value: object) -> list[float]:
    """Split a pipe-joined magnitude cell back into floats.

    Args:
        value: A ``full_feature_mean_magnitudes`` cell.

    Returns:
        The magnitudes, aligned to :func:`parse_ranking` of the same row.
    """
    return [float(part) for part in parse_ranking(value)]


def run(
    runs: list[DatasetRun], out_dir: Path, top_list: int = DEFAULT_TOP_LIST,
) -> pd.DataFrame:
    """Build and write the per-class explanation summary.

    Args:
        runs: The resolved dataset runs to aggregate.
        out_dir: Directory the CSV is written into.
        top_list: How many groups to list in the top-k column.

    Returns:
        The frame that was written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = build_per_class_summary(runs, top_list)
    path = out_dir / CSV_NAME
    frame.to_csv(path, index=False)
    log.info("wrote %s (%d rows)", path, len(frame))
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
            "Aggregate per-flow explanation records to one row per "
            "(dataset, class), including explicit zero-flow rows "
            "(specs/64 Part I §4.5)."
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

    # args.top_k is deliberately not passed through — see DEFAULT_TOP_LIST.
    frame = run(runs, Path(args.out_dir))
    log.info(
        "eval05 complete: %d datasets, %d class rows, %d with explained flows",
        len(runs), len(frame), int((frame["n_flows_aggregated"] > 0).sum()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
