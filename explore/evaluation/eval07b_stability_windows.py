"""
eval07b_stability_windows.py — stability across chronological test windows
===========================================================================
Implements specs/64 Part I §4.7b / Part II §14.7 (re-scoped — read on).

Terminology, which this package treats as binding: "temporal" means two
different things across the two projects, and this module is about the second.
Identifiers here therefore all carry the ``stability_temporal_`` sense, never a
bare ``temporal_``:

  - phi_T *fidelity* is a coalition space — mask a flow's temporal neighbours,
    measure the probability shift. That is eval01's ``coalition_space ==
    "temporal"`` rows, not this file.
  - *stability* across time is this file: does an attribution hold up as the
    traffic distribution drifts across the capture's span?

Re-scoping, stated plainly because the output is weaker than Part I asks for:
  Part I §4.7b specifies re-running explanation generation against two disjoint
  chronological sub-windows of the test split. The pipeline's explanation
  script exposes exactly three flags, none of which selects a time window or an
  edge-id subset, so that invocation is not available today. What IS computable
  from the artifacts on disk is a post-hoc partition of the flows that were
  ALREADY explained, by their own timestamps, and the across-window variance of
  their attributions.

  **This measures whether attributions for DIFFERENT flows explained at
  different times differ — not whether THE SAME explanation is stable under
  drift.** That sentence is repeated in the emitted ``.txt`` and in every CSV
  row's ``notes`` column, because a reader who takes this number for the
  stronger one would overstate the result. The stronger form needs a pipeline
  flag, which is an owner request rather than something this script can supply.

Files read (read-only):
  <run>/outputs/explanations/<Class>/*.json      — per-flow phi_F vectors
  <run>/feature_store/test/edge_indices.npy      — EID -> row alignment
  <run>/feature_store/test/timestamps.npy        — per-row timestamps
  <run>/outputs/metrics/stability.csv            — intra-run stability, for the figure

Files output (under ``outputs/figures/evaluation/``):
  stability_temporal_windows.csv
  stability_comparison_<dataset>.{pdf,png,txt}

Output-location note: ``OUT_DIR`` is the hardcoded
``REPO_ROOT/outputs/figures/evaluation/``, NOT resolved through
``explore._paths.paths()``/``SHAP_GSD_CONFIG`` (specs/64 D3): that variable
selects a single active run, while this script's CSV spans every resolved
dataset and belongs in no one run's subtree. Restated here rather than relied
on by reference (specs/48 §1.1 pt 4).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # explore/AGENT.md mandatory convention

import matplotlib.pyplot as plt
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
from explore.evaluation._load import load_eval_metrics, load_stability_frame  # noqa: E402
from explore.evaluation.eval05_per_class_explanations import (  # noqa: E402
    iter_class_records,
)

log = logging.getLogger("eval07b_stability_windows")

OUT_DIR: Path = DEFAULT_OUT_DIR  # see module docstring for why this is hardcoded

CSV_NAME = "stability_temporal_windows.csv"
FIGURE_STEM_PREFIX = "stability_comparison_"

#: Number of chronological sub-windows the explained flows are partitioned
#: into. Named, never inline, exposed as ``--n-windows``.
DEFAULT_N_WINDOWS = 2

#: Window-boundary modes. ``equal_count`` splits at quantiles of the explained
#: flows' timestamps and is the default: the explained set is a bounded
#: per-class sample rather than the whole test split, so equal-duration bins
#: would produce wildly unequal per-window n.
WINDOW_MODE_EQUAL_COUNT = "equal_count"
WINDOW_MODE_EQUAL_TIME = "equal_time"

#: Repeated into every row's ``notes`` and into the figure's ``.txt``.
WEAKER_STATISTIC_NOTE = (
    "this measures whether attributions for different flows explained at "
    "different times differ, not whether the same explanation is stable under "
    "drift"
)

CSV_COLUMNS: tuple[str, ...] = (
    "dataset",
    "class_name",
    "feature_group",
    "mean_across_windows",
    "std_across_windows",
    "n_windows_available",
    "n_flows_total",
    "window_mode",
    "notes",
)


def build_eid_timestamp_lookup(run: DatasetRun) -> tuple[np.ndarray, np.ndarray]:
    """Load the test split's EID and timestamp arrays.

    Args:
        run: The resolved dataset run.

    Returns:
        ``(edge_indices, timestamps)``, row-aligned.

    Raises:
        FileNotFoundError: When either array is absent.
    """
    fs_test = feature_store_dir(run) / "test"
    eid_path = fs_test / "edge_indices.npy"
    ts_path = fs_test / "timestamps.npy"
    if not eid_path.is_file() or not ts_path.is_file():
        raise FileNotFoundError(
            f"{run.name}: window assignment needs both {eid_path} and {ts_path}"
        )
    edge_indices = np.asarray(np.load(eid_path, mmap_mode="r"))
    timestamps = np.asarray(np.load(ts_path, mmap_mode="r"))
    # CLAUDE.md CRITICAL INVARIANT 1 / CODING STANDARDS 7.
    assert edge_indices.shape == timestamps.shape, (
        f"{run.name}: edge_indices.npy and timestamps.npy differ in length "
        f"({edge_indices.shape} vs {timestamps.shape}); the EID-alignment "
        f"invariant is broken"
    )
    assert bool(np.all(np.diff(edge_indices) >= 0)), (
        f"{run.name}: edge_indices.npy is not ascending, so it cannot be "
        f"searched with np.searchsorted"
    )
    return edge_indices, timestamps


def lookup_timestamp(
    edge_indices: np.ndarray, timestamps: np.ndarray, edge_id: int, run_name: str,
) -> float:
    """Resolve one explained flow's timestamp by its global edge id.

    Args:
        edge_indices: The test split's ascending edge ids.
        timestamps: The row-aligned timestamps.
        edge_id: The explained flow's global edge id.
        run_name: Dataset name, for the error message.

    Returns:
        The flow's timestamp.

    Raises:
        KeyError: When the edge id is not in the test split — ``searchsorted``
            returns an insertion point rather than a miss, so the hit is
            checked explicitly.
    """
    idx = int(np.searchsorted(edge_indices, int(edge_id)))
    if idx >= edge_indices.shape[0] or int(edge_indices[idx]) != int(edge_id):
        raise KeyError(
            f"{run_name}: explained edge id {edge_id} is not present in the "
            f"test split's edge_indices.npy; the EID-alignment invariant "
            f"(CLAUDE.md CRITICAL INVARIANT 1) does not hold for this run"
        )
    return float(timestamps[idx])


def window_boundaries(
    values: np.ndarray, n_windows: int, mode: str,
) -> np.ndarray:
    """Compute chronological window edges over the explained flows' timestamps.

    Args:
        values: Timestamps of every explained flow in the dataset.
        n_windows: How many windows to cut.
        mode: ``equal_count`` or ``equal_time``.

    Returns:
        ``n_windows - 1`` interior boundaries, ascending.

    Raises:
        ValueError: On an unknown mode.
    """
    if n_windows < 2:
        return np.asarray([], dtype=float)
    if mode == WINDOW_MODE_EQUAL_COUNT:
        quantiles = np.linspace(0.0, 1.0, n_windows + 1)[1:-1]
        return np.quantile(values, quantiles)
    if mode == WINDOW_MODE_EQUAL_TIME:
        low, high = float(values.min()), float(values.max())
        return np.linspace(low, high, n_windows + 1)[1:-1]
    raise ValueError(
        f"unknown window mode {mode!r}; expected "
        f"{WINDOW_MODE_EQUAL_COUNT!r} or {WINDOW_MODE_EQUAL_TIME!r}"
    )


def compute_stability_temporal_windows(
    runs: list[DatasetRun], n_windows: int, mode: str,
) -> pd.DataFrame:
    """Compute across-window attribution variance for every dataset.

    Args:
        runs: The resolved dataset runs.
        n_windows: Number of chronological sub-windows.
        mode: Window-boundary mode.

    Returns:
        A frame carrying exactly :data:`CSV_COLUMNS`.
    """
    rows: list[dict[str, object]] = []
    for run in runs:
        metrics = load_eval_metrics(run)
        edge_indices, timestamps = build_eid_timestamp_lookup(run)

        # Pass 1 — collect every explained flow with its timestamp, so window
        # boundaries are dataset-wide and therefore comparable across classes.
        per_class_flows: dict[str, list[tuple[float, list[str], list[float]]]] = {}
        all_times: list[float] = []
        for class_name in metrics.class_names:
            flows: list[tuple[float, list[str], list[float]]] = []
            for rec in iter_class_records(run, class_name):
                names = list(rec.get("feature_group_names") or [])
                values = [float(v) for v in (rec.get("feature_group_shap") or [])]
                if len(names) != len(values) or not names:
                    continue
                stamp = lookup_timestamp(
                    edge_indices, timestamps, int(rec["edge_id"]), run.name,
                )
                flows.append((stamp, names, values))
                all_times.append(stamp)
            per_class_flows[class_name] = flows

        if not all_times:
            log.warning("%s: no explained flows at all; emitting no window rows",
                        run.label)
            continue
        boundaries = window_boundaries(np.asarray(all_times, dtype=float),
                                       n_windows, mode)
        log.info(
            "%s: %d explained flows partitioned into %d %s windows",
            run.label, len(all_times), n_windows, mode,
        )

        # Pass 2 — per class, per feature group, the across-window statistic.
        for class_name in metrics.class_names:
            flows = per_class_flows[class_name]
            if not flows:
                rows.append({
                    "dataset": run.label, "class_name": class_name,
                    "feature_group": "", "mean_across_windows": float("nan"),
                    "std_across_windows": float("nan"), "n_windows_available": 0,
                    "n_flows_total": 0, "window_mode": mode,
                    "notes": f"no explained flows for this class; {WEAKER_STATISTIC_NOTE}",
                })
                continue

            group_names = flows[0][1]
            # {window: {group: [values]}}
            per_window: list[dict[str, list[float]]] = [
                {} for _ in range(n_windows)
            ]
            window_counts = [0] * n_windows
            for stamp, names, values in flows:
                window = int(np.searchsorted(boundaries, stamp, side="right"))
                window = min(window, n_windows - 1)
                window_counts[window] += 1
                for name, value in zip(names, values):
                    per_window[window].setdefault(name, []).append(value)

            # ``n_windows_available`` is counted per feature group below, from
            # the windows that actually carry a value for that group, so the
            # class-level empty-window list is used only for the note.
            empty_windows = [i for i, c in enumerate(window_counts) if c == 0]
            note = WEAKER_STATISTIC_NOTE
            if empty_windows:
                note = (
                    f"window(s) {empty_windows} contain no flow of this class "
                    f"(per-window n = {window_counts}); {note}"
                )
            for group in group_names:
                window_means = [
                    float(np.mean(per_window[i][group]))
                    for i in range(n_windows)
                    if group in per_window[i] and per_window[i][group]
                ]
                rows.append({
                    "dataset": run.label,
                    "class_name": class_name,
                    "feature_group": group,
                    "mean_across_windows": (
                        float(np.mean(window_means)) if window_means else float("nan")
                    ),
                    "std_across_windows": (
                        float(np.std(window_means, ddof=0))
                        if len(window_means) > 1 else float("nan")
                    ),
                    "n_windows_available": len(window_means),
                    "n_flows_total": len(flows),
                    "window_mode": mode,
                    "notes": note,
                })
    return pd.DataFrame(rows, columns=list(CSV_COLUMNS))


# --- figure ---------------------------------------------------------------

LABEL_FS = 9           # explore/AGENT.md §6 style constant
_SHAP_GSD = "#2a9d8f"  # teal  — the intra-run sub-dimension
_GNN_EXP = "#e9c46a"   # amber — the across-window sub-dimension
_GRAY = "#888888"      # the not-available inter-seed gap
# _PRESENCE/_ABSENCE intentionally not used: this figure encodes dispersion
# magnitudes, not the sign of an attribution (specs/47 §4).


def write_stability_figure(
    dataset_label: str,
    windows: pd.DataFrame,
    intra_run: dict[str, float],
    out_dir: Path,
    n_windows: int,
    mode: str,
) -> None:
    """Write one dataset's three-sub-dimension stability comparison trio.

    Args:
        dataset_label: The dataset's display label.
        windows: That dataset's across-window rows.
        intra_run: ``{class_name: intra-run mean phi std}``.
        out_dir: Output directory.
        n_windows: Number of windows used.
        mode: Window-boundary mode used.
    """
    stem = f"{FIGURE_STEM_PREFIX}{dataset_label}"
    per_class = (
        windows[windows["feature_group"] != ""]
        .groupby("class_name")["std_across_windows"].mean()
    )
    classes = list(dict.fromkeys(windows["class_name"]))
    across = [float(per_class.get(c, float("nan"))) for c in classes]
    intra = [float(intra_run.get(c, float("nan"))) for c in classes]

    x = np.arange(len(classes), dtype=float)
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(6.0, 0.8 * len(classes) + 3.0), 4.2))
    ax.bar(x - width, [v if np.isfinite(v) else 0.0 for v in intra], width,
           color=_SHAP_GSD, label="intra-run (coalition sampling)")
    # The inter-seed group is drawn as an explicit empty slot rather than
    # omitted, so the figure's own shape shows what is missing.
    ax.bar(x, np.zeros(len(classes)), width, color="none",
           edgecolor=_GRAY, hatch="//", label="inter-seed (not available)")
    ax.bar(x + width, [v if np.isfinite(v) else 0.0 for v in across], width,
           color=_GNN_EXP, label="across chronological windows")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=LABEL_FS)
    ax.set_ylabel("std of phi_F attribution", fontsize=LABEL_FS)
    ax.legend(fontsize=LABEL_FS - 1, frameon=False)
    ax.set_title(f"{dataset_label} — stability sub-dimensions", fontsize=LABEL_FS + 1)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    scored = [v for v in across if np.isfinite(v)]
    findings = [
        f"  Classes plotted: {len(classes)}.",
        f"  Window mode: {mode}, {n_windows} windows.",
        f"  Classes with an across-window statistic: {len(scored)}.",
    ]
    if scored:
        findings.append(
            f"  Mean across-window std ranges {min(scored):.5f} to "
            f"{max(scored):.5f}."
        )
    intra_scored = [v for v in intra if np.isfinite(v)]
    if intra_scored:
        findings.append(
            f"  Intra-run std ranges {min(intra_scored):.5f} to "
            f"{max(intra_scored):.5f}."
        )
    findings.append(
        "  The inter-seed group is empty by construction: it needs several "
        "independently-seeded training runs per dataset, which do not exist."
    )

    reasoning = f"""Figure reasoning — {stem}
{'=' * (len(stem) + 20)}

WHAT THE FIGURE SHOWS
---------------------
One group of bars per class of {dataset_label}, showing the three stability
sub-dimensions side by side. The first bar is intra-run dispersion — how much
an attribution moves under repeated coalition sampling within a single run.
The second is intentionally empty and hatched: inter-seed dispersion cannot be
computed from the artifacts that exist, and drawing the gap keeps its absence
visible in the deliverable instead of silently dropping a third of the
framing. The third bar is dispersion across chronological sub-windows of the
already-explained flows, averaged over feature groups.

{WEAKER_STATISTIC_NOTE.capitalize()}. The stronger form — re-explaining the
same flows under each window — needs an explanation-generation flag the
pipeline does not currently expose.

KEY FINDINGS
------------
{chr(10).join(findings)}

PAPER FRAMING
-------------
Stability is a three-part claim: an explanation should survive the explainer's
own sampling noise, the model's training randomness, and drift in the traffic
it is applied to. Only the first and (in the weaker form above) the third are
computable from what exists today, and the figure says so rather than
presenting a two-part result as if it were the whole claim.

SUGGESTED FIGURE CAPTION
------------------------
Stability sub-dimensions for {dataset_label}: intra-run dispersion under
coalition sampling, inter-seed dispersion (not available), and dispersion
across {n_windows} chronological sub-windows of the explained flows
({mode} boundaries). The across-window bar compares attributions for different
flows explained at different times, not the same explanation under drift.
"""
    (out_dir / f"{stem}.txt").write_text(reasoning)
    log.info("wrote %s.{pdf,png,txt}", out_dir / stem)


def run(
    runs: list[DatasetRun],
    out_dir: Path,
    n_windows: int = DEFAULT_N_WINDOWS,
    mode: str = WINDOW_MODE_EQUAL_COUNT,
) -> pd.DataFrame:
    """Build and write every eval07b artifact.

    Args:
        runs: The resolved dataset runs.
        out_dir: Output directory.
        n_windows: Number of chronological sub-windows.
        mode: Window-boundary mode.

    Returns:
        The across-window frame that was written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = compute_stability_temporal_windows(runs, n_windows, mode)
    path = out_dir / CSV_NAME
    frame.to_csv(path, index=False)
    log.info("wrote %s (%d rows)", path, len(frame))

    for run_obj in runs:
        subset = frame[frame["dataset"] == run_obj.label]
        if subset.empty:
            continue
        stability = load_stability_frame(run_obj)
        intra: dict[str, float] = {}
        if stability is not None and not stability.empty:
            intra = {
                str(name): float(group["mean_phi_std"].mean())
                for name, group in stability.groupby("class_name")
            }
        write_stability_figure(
            run_obj.label, subset, intra, out_dir, n_windows, mode,
        )
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
            "Partition the already-explained flows into chronological "
            "sub-windows by their own timestamps and compute the per-class "
            "per-feature-group across-window attribution variance "
            "(specs/64 Part I §4.7b, re-scoped by Part II §14.7)."
        )
    )
    add_common_args(parser)
    parser.add_argument(
        "--n-windows", type=int, default=DEFAULT_N_WINDOWS,
        help=f"Number of chronological sub-windows (default: "
             f"{DEFAULT_N_WINDOWS}).",
    )
    parser.add_argument(
        "--window-mode", type=str, default=WINDOW_MODE_EQUAL_COUNT,
        choices=[WINDOW_MODE_EQUAL_COUNT, WINDOW_MODE_EQUAL_TIME],
        help=f"Window-boundary rule (default: {WINDOW_MODE_EQUAL_COUNT}; "
             f"equal-duration bins give wildly unequal per-window n because "
             f"the explained set is a bounded per-class sample).",
    )
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

    frame = run(runs, Path(args.out_dir), int(args.n_windows), str(args.window_mode))
    log.info("eval07b complete: %d rows", len(frame))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
