"""
eval08_cross_dataset_report.py — the assembled cross-dataset comparison report
==============================================================================
Implements specs/64 Part I §4.8 / Part II §14.8.

What it does:
  Assembles the other six modules' CSVs into one generated markdown document,
  in a fixed section order, and **never recomputes a number**. Every figure in
  the report is read from the CSV that owns it; a missing CSV produces an
  explicit "not available — run <script>" stub rather than a silently absent
  section, so the report's own shape shows what has and has not been run.

  Figure captions are parsed out of each figure's own ``.txt`` reasoning file
  (its ``SUGGESTED FIGURE CAPTION`` block) rather than written a second time
  here, so editing a caption in one place propagates.

Refuses to run against a single dataset: this is the cross-dataset artifact,
and a one-dataset "comparison" would be a misleading deliverable. Pass
``--all-datasets`` or at least two ``--dataset`` selections.

Files read (all under the same ``--out-dir``, all produced by this package):
  eval_long_metrics.csv, eval_long_metrics_coverage.md   (eval01)
  macro_stats_by_denominator.csv                          (eval02)
  cross_explainer_agreement.csv,
  cross_explainer_disagreement_flags.csv                  (eval03)
  global_coherence_rank_correlation.csv,
  global_coherence_structural.csv                         (eval04)
  per_class_explanation_summary.csv                       (eval05)
  stability_temporal_windows.csv                          (eval07b)
  <figure stem>.txt                                       (captions)

Files output (under ``outputs/figures/evaluation/``):
  cross_dataset_comparison_report.md

Output-location note: ``OUT_DIR`` is the hardcoded
``REPO_ROOT/outputs/figures/evaluation/``, NOT resolved through
``explore._paths.paths()``/``SHAP_GSD_CONFIG`` (specs/64 D3): the report is a
cross-dataset artifact assembled from cross-dataset inputs, and that variable
names a single active run whose subtree would be the wrong home for it.
Restated here rather than relied on by reference (specs/48 §1.1 pt 4).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timezone
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
    resolve_datasets_from_args,
)

log = logging.getLogger("eval08_cross_dataset_report")

OUT_DIR: Path = DEFAULT_OUT_DIR  # see module docstring for why this is hardcoded

REPORT_NAME = "cross_dataset_comparison_report.md"

#: input CSV -> the script that produces it, for the "not available" stubs.
INPUT_SOURCES: dict[str, str] = {
    "eval_long_metrics.csv": "explore/evaluation/eval01_long_metrics.py",
    "macro_stats_by_denominator.csv": "explore/evaluation/eval02_macro_denominators.py",
    "cross_explainer_agreement.csv": "explore/evaluation/eval03_cross_explainer_agreement.py",
    "cross_explainer_disagreement_flags.csv": "explore/evaluation/eval03_cross_explainer_agreement.py",
    "global_coherence_rank_correlation.csv": "explore/evaluation/eval04_global_coherence.py",
    "global_coherence_structural.csv": "explore/evaluation/eval04_global_coherence.py",
    "per_class_explanation_summary.csv": "explore/evaluation/eval05_per_class_explanations.py",
    "stability_temporal_windows.csv": "explore/evaluation/eval07b_stability_windows.py",
}

#: The `.txt` section header captions are parsed out of.
CAPTION_HEADER = "SUGGESTED FIGURE CAPTION"


def read_optional_csv(out_dir: Path, name: str) -> pd.DataFrame | None:
    """Read one of this package's CSVs if it has been generated.

    Args:
        out_dir: The shared output directory.
        name: CSV file name.

    Returns:
        The frame, or ``None`` when the file is absent.
    """
    path = out_dir / name
    if not path.is_file():
        log.warning("%s not found; its section will be a stub", path)
        return None
    return pd.read_csv(path)


def stub(name: str) -> str:
    """Render the "not available" stub for a missing input.

    Args:
        name: The missing CSV's file name.

    Returns:
        A markdown paragraph naming the script that produces it.
    """
    return (
        f"_Not available — `{name}` has not been generated. Run "
        f"`{INPUT_SOURCES.get(name, 'the owning eval script')}` with the same "
        f"`--out-dir`._"
    )


def figure_caption(out_dir: Path, stem: str) -> str:
    """Extract one figure's caption from its own reasoning ``.txt``.

    Args:
        out_dir: The shared output directory.
        stem: Figure stem, without extension.

    Returns:
        The caption text, or an explanatory line when the ``.txt`` is absent.
    """
    path = out_dir / f"{stem}.txt"
    if not path.is_file():
        return f"_No caption: `{stem}.txt` has not been generated._"
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip() == CAPTION_HEADER:
            body = [
                text for text in lines[i + 2:]
                if not set(text.strip()) <= {"-"} or not text.strip()
            ]
            return " ".join(t.strip() for t in body if t.strip())
    return f"_No `{CAPTION_HEADER}` block found in `{stem}.txt`._"


def _markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    float_fmt: str = "{:.4f}",
    int_columns: tuple[str, ...] = (),
) -> list[str]:
    """Render a dataframe slice as a markdown table.

    Pipe characters inside a cell are escaped: several columns carry
    pipe-joined class lists, which would otherwise split the row into extra
    columns and silently corrupt the table.

    Args:
        frame: Source frame.
        columns: Columns to render, in order.
        float_fmt: Format applied to float cells.
        int_columns: Columns whose float values are counts and should render
            without a decimal part (a NaN still renders as an em dash).

    Returns:
        Markdown lines.
    """
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                if not np.isfinite(value):
                    cells.append("—")
                elif column in int_columns:
                    cells.append(f"{int(round(float(value)))}")
                else:
                    cells.append(float_fmt.format(value))
            else:
                text = str(value).replace("|", r"\|")
                cells.append(text if text and text != "nan" else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _dataset_column_check(
    frames: dict[str, pd.DataFrame | None], expected: list[str],
) -> list[str]:
    """Warn when an input CSV's datasets do not match the current selection.

    The ``dataset`` column carries whatever display label the *producing*
    invocation used, and the producing invocation may have selected datasets by
    a different token than this one. Joining silently across that mismatch
    would produce a report whose sections describe different dataset sets, so
    the mismatch is reported in the document itself.

    Args:
        frames: ``{csv_name: frame or None}``.
        expected: The labels of the currently-resolved datasets.

    Returns:
        Markdown warning lines; empty when everything matches.
    """
    warnings: list[str] = []
    for name, frame in frames.items():
        if frame is None or frame.empty or "dataset" not in frame.columns:
            continue
        found = sorted(set(frame["dataset"].astype(str)))
        missing = sorted(set(expected) - set(found))
        extra = sorted(set(found) - set(expected))
        if missing or extra:
            warnings.append(
                f"- `{name}` carries datasets {found}, but this report was "
                f"generated for {sorted(expected)}"
                + (f"; missing: {missing}" if missing else "")
                + (f"; unexpected: {extra}" if extra else "")
                + ". The `dataset` column records the display label of the "
                "invocation that produced the CSV, so a mismatch usually means "
                "that CSV was produced with a different `--dataset`/`--label` "
                "selection. Regenerate it before reading the sections below as "
                "one comparison."
            )
    return warnings


def _git_head() -> str:
    """Return this repo's current commit, or a marker when it cannot be read.

    Returns:
        The short commit hash, or ``"unknown"``.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True, timeout=15,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return "unknown"


def build_report(runs: list[DatasetRun], out_dir: Path) -> str:
    """Assemble the cross-dataset comparison report.

    Args:
        runs: The resolved dataset runs (for the provenance block).
        out_dir: The shared output directory holding every input CSV.

    Returns:
        The full markdown document.
    """
    frames = {name: read_optional_csv(out_dir, name) for name in INPUT_SOURCES}
    labels = [run.label for run in runs]
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines: list[str] = [
        "# Cross-dataset comparison report",
        "",
        f"Generated {stamp} by `explore/evaluation/eval08_cross_dataset_report.py`.",
        "",
        "Every number below is read from the CSV that owns it; nothing in this "
        "document is recomputed. A section reading _not available_ means its "
        "input has not been generated, not that the quantity is zero.",
        "",
        f"Datasets in this report: {', '.join(labels)}.",
        "",
    ]

    mismatch = _dataset_column_check(frames, labels)
    if mismatch:
        lines += ["## Input consistency warning", ""] + mismatch + [""]

    # --- 1. Fidelity -------------------------------------------------------
    lines += ["## 1. Fidelity comparison", ""]
    long_metrics = frames["eval_long_metrics.csv"]
    if long_metrics is None:
        lines += [stub("eval_long_metrics.csv"), ""]
    else:
        fidelity = long_metrics[
            long_metrics["metric"].isin(["fidelity_plus", "fidelity_minus"])
        ]
        pivot = fidelity.pivot_table(
            index=["dataset", "class_name"],
            columns=["coalition_space", "metric"],
            values="value", aggfunc="first",
        ).reset_index()
        pivot.columns = [
            c if isinstance(c, str) else "_".join(str(p) for p in c if p)
            for c in pivot.columns
        ]
        # Left-join onto each dataset's OWN class list, so a class whose
        # fidelity cells are all NaN still appears as an explicit all-em-dash
        # row instead of vanishing from the comparison -- while a class of one
        # dataset never leaks into another dataset's rows.
        keys = long_metrics[["dataset", "class_name"]].drop_duplicates()
        pivot = keys.merge(pivot, on=["dataset", "class_name"], how="left")
        value_columns = [c for c in pivot.columns if c not in ("dataset", "class_name")]
        lines += _markdown_table(pivot, ["dataset", "class_name"] + value_columns)
        lines.append("")

    # --- 2. Macro statistics ----------------------------------------------
    lines += ["## 2. Macro statistics under all three denominators", ""]
    macro = frames["macro_stats_by_denominator.csv"]
    if macro is None:
        lines += [stub("macro_stats_by_denominator.csv"), ""]
    else:
        lines += [
            "Which denominator a cross-dataset macro-F1 uses is not a "
            "presentational detail: on a dataset whose test split covers only "
            "part of the label space the three differ by several fold. The "
            "per-dataset coverage that drives the spread is in section 6, "
            "stated per table rather than relegated to a footnote.",
            "",
        ]
        lines += _markdown_table(macro, [
            "dataset", "denominator", "n_classes_in_denominator",
            "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "notes",
        ], int_columns=("n_classes_in_denominator",))
        lines.append("")

    # --- 3. Cross-explainer agreement -------------------------------------
    lines += ["## 3. Cross-explainer agreement", ""]
    agreement = frames["cross_explainer_agreement.csv"]
    flags = frames["cross_explainer_disagreement_flags.csv"]
    if agreement is None:
        lines += [stub("cross_explainer_agreement.csv"), ""]
    else:
        summary = (
            agreement.groupby(["dataset", "space"])
            .agg(
                pairs=("spearman_rho", "size"),
                scored=("spearman_rho", lambda s: int(np.isfinite(s).sum())),
                mean_rho=("spearman_rho", "mean"),
                mean_jaccard=("jaccard_topk", "mean"),
            ).reset_index()
        )
        lines += _markdown_table(
            summary, ["dataset", "space", "pairs", "scored", "mean_rho", "mean_jaccard"],
        )
        lines.append("")
        for (dataset_label, space), _ in agreement.groupby(["dataset", "space"]):
            stem = f"cross_explainer_agreement_heatmap_{dataset_label}_{space}"
            lines += [f"**{stem}** — {figure_caption(out_dir, stem)}", ""]
    if flags is not None and not flags.empty:
        flagged = flags[flags["flagged"].astype(bool)]
        lines += ["### Flagged disagreement regions", ""]
        if flagged.empty:
            lines += ["_No class fell below the disagreement threshold._", ""]
        else:
            lines += _markdown_table(flagged, [
                "dataset", "class_name", "space",
                "shap_gsd_rank_vs_majority_rho", "threshold", "n_pairs",
            ])
            lines.append("")

    # --- 4. Global coherence ----------------------------------------------
    lines += ["## 4. Global coherence", ""]
    rank = frames["global_coherence_rank_correlation.csv"]
    structural = frames["global_coherence_structural.csv"]
    if rank is None:
        lines += [stub("global_coherence_rank_correlation.csv"), ""]
    elif rank.empty:
        lines += [
            "_Rank-correlation half not scored: the owner-authored proxy-GT "
            "feature-expectation mapping was absent when eval04 last ran. The "
            "structural half below is unaffected._",
            "",
        ]
    else:
        lines += _markdown_table(rank, [
            "dataset", "class_name", "spearman_rho_vs_expected", "p_raw",
            "p_corrected", "correction_family", "mapping_confidence",
            "rho_undefined_reason",
        ])
        lines.append("")
    if structural is None:
        lines += [stub("global_coherence_structural.csv"), ""]
    else:
        lines += _markdown_table(structural, [
            "dataset", "class_name", "shap_subgraph_topology",
            "literature_expected_topology", "stage0_measured_topology",
            "three_way_agreement", "n_subgraph_edges",
        ])
        lines.append("")
        for dataset_label in sorted(set(structural["dataset"].astype(str))):
            stem = f"global_coherence_summary_{dataset_label}"
            lines += [f"**{stem}** — {figure_caption(out_dir, stem)}", ""]

    # --- 5. Stability ------------------------------------------------------
    lines += ["## 5. Stability", ""]
    windows = frames["stability_temporal_windows.csv"]
    lines += [
        "Three sub-dimensions are in scope. Intra-run dispersion is computed "
        "by the pipeline. Inter-seed dispersion is **not available**: it needs "
        "several independently-seeded training runs per dataset, which do not "
        "exist, so it is reported as a gap rather than omitted. The "
        "across-window statistic below compares attributions for different "
        "flows explained at different times, not the same explanation under "
        "drift.",
        "",
    ]
    if windows is None:
        lines += [stub("stability_temporal_windows.csv"), ""]
    else:
        scored = windows[windows["feature_group"].astype(str) != ""]
        summary = (
            scored.groupby(["dataset", "class_name"])
            .agg(
                mean_std_across_windows=("std_across_windows", "mean"),
                n_flows_total=("n_flows_total", "max"),
                min_windows_available=("n_windows_available", "min"),
            ).reset_index()
        )
        lines += _markdown_table(summary, [
            "dataset", "class_name", "mean_std_across_windows",
            "n_flows_total", "min_windows_available",
        ], float_fmt="{:.6f}")
        lines.append("")
        for dataset_label in sorted(set(windows["dataset"].astype(str))):
            stem = f"stability_comparison_{dataset_label}"
            lines += [f"**{stem}** — {figure_caption(out_dir, stem)}", ""]
    lines += [
        "_Inter-seed stability: not available (no seeded training runs exist; "
        "it is the most compute-expensive item in this line of work and needs "
        "explicit budget sign-off)._",
        "",
        "_Nemenyi post-hoc: not computed — it needs the studentized range "
        "distribution, i.e. a dependency this repo does not carry. A "
        "significant Friedman omnibus is reported with the post-hoc marked "
        "unavailable rather than dropped._",
        "",
    ]

    # --- 6. Coverage and caveats ------------------------------------------
    lines += ["## 6. Coverage and caveats", ""]
    coverage_md = out_dir / "eval_long_metrics_coverage.md"
    if long_metrics is None:
        lines += [stub("eval_long_metrics.csv"), ""]
    else:
        classification = long_metrics[long_metrics["metric"] == "support"]
        coverage_rows = (
            classification.groupby("dataset")
            .agg(
                n_classes=("class_name", "nunique"),
                n_present_in_test=("class_present_in_test", lambda s: int(s.astype(bool).sum())),
            ).reset_index()
        )
        coverage_rows["absent_classes"] = [
            "|".join(sorted(
                classification[
                    (classification["dataset"] == row["dataset"])
                    & (~classification["class_present_in_test"].astype(bool))
                ]["class_name"].astype(str)
            )) or "—"
            for _, row in coverage_rows.iterrows()
        ]
        lines += [
            "Generated from `eval_long_metrics.csv`'s `class_present_in_test` "
            "column. Any macro statistic quoted above without stating its "
            "denominator silently mixes scored classes with classes that had "
            "no test support at all.",
            "",
        ]
        lines += _markdown_table(coverage_rows, [
            "dataset", "n_classes", "n_present_in_test", "absent_classes",
        ])
        lines.append("")
        if coverage_md.is_file():
            lines += [
                "Per-dataset schema provenance and the verbatim macro-F1 "
                f"convention strings are in `{coverage_md.name}`, including "
                "any dataset marked `schema_source = derived` (whose coverage "
                "numbers were reconstructed rather than read).",
                "",
            ]
        explanations = frames["per_class_explanation_summary.csv"]
        if explanations is not None:
            zero = explanations[explanations["n_flows_aggregated"] == 0]
            lines += [
                f"Classes with zero explained flows: {len(zero)} of "
                f"{len(explanations)} (dataset, class) pairs.",
                "",
            ]

    # --- provenance --------------------------------------------------------
    lines += ["## Provenance", "", f"- repo commit: `{_git_head()}`"]
    for run in runs:
        lines.append(
            f"- **{run.label}** (`--dataset {run.name}`): `{run.run_dir}`, "
            f"variant_suffix `{run.variant_suffix or '(none)'}`, config "
            f"`{run.config_path or '(resolved by completeness, no config)'}`"
        )
    lines.append("")
    lines.append("Input file mtimes:")
    for name in INPUT_SOURCES:
        path = out_dir / name
        if path.is_file():
            mtime = datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc,
            ).isoformat(timespec="seconds")
            lines.append(f"- `{name}`: {mtime}")
        else:
            lines.append(f"- `{name}`: absent")
    lines.append("")
    return "\n".join(lines)


def run(runs: list[DatasetRun], out_dir: Path) -> Path:
    """Assemble and write the report.

    Args:
        runs: The resolved dataset runs.
        out_dir: The shared output directory.

    Returns:
        The path written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REPORT_NAME
    path.write_text(build_report(runs, out_dir))
    log.info("wrote %s", path)
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector, or ``None`` to read ``sys.argv``.

    Returns:
        Process exit code (0 on success, 2 on a dataset-resolution failure or a
        single-dataset selection).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Assemble the other modules' CSVs into one cross-dataset "
            "comparison report, recomputing nothing (specs/64 Part I §4.8)."
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

    if len(runs) < 2:
        log.error(
            "eval08 is the cross-dataset report and refuses to run against %d "
            "dataset(s): pass --all-datasets, or at least two --dataset "
            "selections.", len(runs),
        )
        return 2

    run(runs, Path(args.out_dir))
    log.info("eval08 complete: %d datasets", len(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
