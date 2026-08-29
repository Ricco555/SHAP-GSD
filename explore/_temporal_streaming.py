"""Shared streaming/temporal-axis utilities for OOM-prone `explore/` scripts.

Factors out the two-pass streaming design (specs/71) and its generalization
to a shared module used by three scripts (specs/72): `class_time_distribution.py`,
`class_coverage_analysis.py`, `class_coverage_split_sweep.py`. All three
scripts currently compute a pooled-axis timestamp summary and/or a genuine
timestamp-sorted reorder of a small number of parallel columns by either
(a) a non-chunked `pd.read_csv` followed by `df.sort_values(kind="mergesort")`
on the whole frame, which allocates several full-length arrays simultaneously
(specs/71 SS2.2's ~22-30 GB estimate at CICIoT2023 scale), or (b) an
equivalent full-frame sort downstream of an already-in-memory DataFrame.

This module provides:
  - Pass 1 (pooled timestamp axis): `pooled_timestamp_stats_from_csv` /
    `pooled_timestamp_stats_from_array`, returning a `Pass1Result`.
  - `argsort_and_take`: a cheap, non-full-frame replacement for
    `df.sort_values(col, kind="mergesort")` over a small number of parallel
    numpy arrays.
  - `per_class_sorted_positions`: an exact per-class row-position index over
    an already time-sorted array of integer class codes.
  - Pass 2 (per-class fine-grained time histograms): `per_class_time_histograms_from_csv`
    / `per_class_time_histograms_from_arrays`, returning a `Pass2Result`, plus
    `cdf_from_hist` / `quantile_from_hist` / `windowed_sum` helpers for
    consuming the histogram without ever holding the exact per-row arrays.

No import of anything from `src/` here — this stays a pure numerical utility
module that callers (which may separately import `Preprocessor`, etc.) sit
around, not inside. Follows the `explore/_paths.py` naming convention:
leading underscore, no `main()`, no CLI, not a runnable script.

specs/71 SS3.2/SS3.3, specs/72 SS1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Pass1Result:
    """Pooled-axis scalars derived from one fully sorted timestamp array.

    Every field is an exact order-statistic or arithmetic derivative of the
    pooled (all-rows, all-classes) timestamp axis -- see
    `class_time_distribution.py`'s original per-row computation (specs/71
    SS3.2) for the quantities this replaces.

    Attributes:
        n: Total row count (including any NaN-timestamp rows, for callers
            that allow them -- see `pooled_timestamp_stats_from_array`).
        t0_ms: Timestamp (milliseconds) of the earliest (finite) row.
        total_span_hours: Elapsed hours from `t0_ms` to the latest (finite)
            row's timestamp.
        largest_gap_size_h: Size, in hours, of the single biggest inter-flow
            gap in the pooled, sorted timestamp axis.
        gap_end_h: Elapsed hours (since `t0_ms`) at the end of that gap.
        zoom_start_h: Dense-region start per the >=5%-of-span rule (falls
            back to the last 10% of the span otherwise).
        tau_train_h: Elapsed hours at the train/val row-fraction cut.
        tau_val_h: Elapsed hours at the val/test row-fraction cut.
        tau_train_ms: Timestamp (milliseconds) at the train/val row-fraction cut.
        tau_val_ms: Timestamp (milliseconds) at the val/test row-fraction cut.
    """

    n: int
    t0_ms: int
    total_span_hours: float
    largest_gap_size_h: float
    gap_end_h: float
    zoom_start_h: float
    tau_train_h: float
    tau_val_h: float
    tau_train_ms: int
    tau_val_ms: int


def _pass1_from_sorted_ms(
    ts_sorted_ms: np.ndarray, train_frac: float, val_frac: float
) -> Pass1Result:
    """Compute every `Pass1Result` field from an already-sorted, finite,
    int64-valued millisecond timestamp array.

    Shared body for both Pass-1 entry points once each has produced a sorted
    array of finite timestamps (in milliseconds). Uses the same row-fraction
    cut convention as `class_time_distribution.py`'s original code
    (`train_cut_idx = int(n * train_frac) - 1`), NOT `Preprocessor.
    _temporal_split`'s convention (`int(n * train_frac)`, no `-1`) -- these
    two conventions already differed in the pre-existing code this module
    replaces, and this function preserves that difference rather than
    unifying it (out of scope for this streaming redesign).

    Args:
        ts_sorted_ms: 1-D int64 array of timestamps in milliseconds, sorted
            ascending, containing no NaN.
        train_frac: Row-fraction of the train/val cut.
        val_frac: Row-fraction of the val/test cut (added to `train_frac`).

    Returns:
        A populated `Pass1Result`.
    """
    n = len(ts_sorted_ms)
    t0_ms = int(ts_sorted_ms[0])
    total_span_hours = (float(ts_sorted_ms[-1]) - t0_ms) / (1000.0 * 3600.0)

    gaps_h = np.diff(ts_sorted_ms) / (1000.0 * 3600.0)
    largest_gap_idx = int(np.argmax(gaps_h))
    largest_gap_size_h = float(gaps_h[largest_gap_idx])
    gap_end_h = (float(ts_sorted_ms[largest_gap_idx + 1]) - t0_ms) / (1000.0 * 3600.0)
    del gaps_h

    zoom_start_h = (
        gap_end_h if largest_gap_size_h >= 0.05 * total_span_hours
        else total_span_hours * 0.90
    )

    train_cut_idx = int(n * train_frac) - 1
    val_cut_idx = int(n * (train_frac + val_frac)) - 1
    train_cut_idx = max(train_cut_idx, 0)
    val_cut_idx = max(val_cut_idx, 0)
    tau_train_ms = int(ts_sorted_ms[train_cut_idx])
    tau_val_ms = int(ts_sorted_ms[val_cut_idx])
    tau_train_h = (float(tau_train_ms) - t0_ms) / (1000.0 * 3600.0)
    tau_val_h = (float(tau_val_ms) - t0_ms) / (1000.0 * 3600.0)

    return Pass1Result(
        n=n, t0_ms=t0_ms, total_span_hours=total_span_hours,
        largest_gap_size_h=largest_gap_size_h, gap_end_h=gap_end_h,
        zoom_start_h=zoom_start_h, tau_train_h=tau_train_h, tau_val_h=tau_val_h,
        tau_train_ms=tau_train_ms, tau_val_ms=tau_val_ms,
    )


def pooled_timestamp_stats_from_csv(
    csv_path: Path | str,
    ts_col: str,
    train_frac: float,
    val_frac: float,
    chunksize: int = 5_000_000,
) -> Pass1Result:
    """Chunked-CSV Pass 1: pooled-axis timestamp statistics.

    Reads `ts_col` only, in chunks, concatenates, sorts in place (plain
    unstable sort -- rank-only use, no tie-break need since only the VALUE at
    a given rank is ever read downstream), and derives every `Pass1Result`
    field. Peak memory ~5 GB at CICIoT2023 scale (specs/71 SS3.2's
    arithmetic): the concatenated array (~4.3 GB at 540.7M rows) plus
    in-place-sort auxiliary stack space (O(log n), not O(n)).

    Used when the caller has NOT already read the file -- currently only
    `class_time_distribution.py`, which reads exactly 2 columns and nothing
    else, ever (specs/72 SS1.2).

    Args:
        csv_path: Path to the raw NetFlow CSV.
        ts_col: Name of the millisecond-timestamp column (e.g.
            "FLOW_START_MILLISECONDS").
        train_frac: Row-fraction of the train/val cut.
        val_frac: Row-fraction of the val/test cut.
        chunksize: Rows per `pd.read_csv` chunk.

    Returns:
        A populated `Pass1Result`.
    """
    ts_chunks = []
    for chunk in pd.read_csv(
        csv_path, usecols=[ts_col], dtype={ts_col: "int64"}, chunksize=chunksize
    ):
        ts_chunks.append(chunk[ts_col].to_numpy())
    ts_all = np.concatenate(ts_chunks) if ts_chunks else np.empty(0, dtype=np.int64)
    del ts_chunks
    ts_all.sort(kind="quicksort")
    result = _pass1_from_sorted_ms(ts_all, train_frac, val_frac)
    del ts_all
    return result


def pooled_timestamp_stats_from_array(
    ts: np.ndarray, train_frac: float, val_frac: float
) -> Pass1Result:
    """Same Pass 1 computation, given an already-in-memory timestamp array.

    Used by `class_coverage_analysis.py` / `class_coverage_split_sweep.py`,
    which have already performed a full-file read + dedup + dropna by the
    time they reach this call (specs/72 SS2) -- there is no CSV left worth
    chunk-reading; the cost this removes is the O(n)-plus-mergesort-scratch
    blowup of computing these scalars via `df.sort_values(...).iloc[k]`,
    replaced by sorting the bare array alone.

    Takes a COPY of `ts` before sorting (never mutates the caller's array in
    place) -- callers still need the ORIGINAL row order of `ts` afterwards to
    build the aligned `(ts, class)` pair for `argsort_and_take`; sorting the
    caller's array in place out from under it would silently desynchronize
    it from a parallel class-code array.

    Accepts `ts` as float64 or int64. NaN entries (float64 only) are handled
    per specs/72 SS3.2.1: NaN sorts last under `np.sort` (numpy's documented
    behavior -- NaN compares greater than any other value), `total_span_hours`
    and the largest-gap search use NaN-aware (`nanmax`/finite-only) reads so
    a trailing NaN block never poisons the span/gap arithmetic, and
    `train_cut`/`val_cut` are computed from `n`, the FULL row count including
    any NaN rows -- exactly matching `Preprocessor._temporal_split`'s and the
    current `class_coverage_analysis.py` code's existing behavior (a NaN
    timestamp row still counts toward `n` when locating the row-fraction cut,
    it just cannot itself be a rank the cut lands on if enough non-NaN rows
    precede it and the cut index is comfortably below the first NaN
    position, which is exactly what happens today).

    Args:
        ts: 1-D int64 or float64 array of timestamps in milliseconds, in
            ANY order (need not already be sorted).
        train_frac: Row-fraction of the train/val cut.
        val_frac: Row-fraction of the val/test cut.

    Returns:
        A populated `Pass1Result`. `n` is the full length of `ts`, including
        any NaN entries.
    """
    ts_sorted = ts.copy()
    ts_sorted.sort(kind="quicksort")  # NaN (float64) sorts last

    n_total = len(ts_sorted)
    if np.issubdtype(ts_sorted.dtype, np.floating):
        finite_mask = np.isfinite(ts_sorted)
        n_finite = int(finite_mask.sum())
        ts_finite = ts_sorted[:n_finite] if n_finite < n_total else ts_sorted
    else:
        n_finite = n_total
        ts_finite = ts_sorted

    ts_finite_i64 = ts_finite.astype(np.int64) if ts_finite.dtype != np.int64 else ts_finite

    t0_ms = int(ts_finite_i64[0])
    total_span_hours = (float(ts_finite_i64[-1]) - t0_ms) / (1000.0 * 3600.0)

    gaps_h = np.diff(ts_finite_i64) / (1000.0 * 3600.0)
    if len(gaps_h) > 0:
        largest_gap_idx = int(np.argmax(gaps_h))
        largest_gap_size_h = float(gaps_h[largest_gap_idx])
        gap_end_h = (float(ts_finite_i64[largest_gap_idx + 1]) - t0_ms) / (1000.0 * 3600.0)
    else:
        largest_gap_size_h = 0.0
        gap_end_h = total_span_hours
    del gaps_h

    zoom_start_h = (
        gap_end_h if largest_gap_size_h >= 0.05 * total_span_hours
        else total_span_hours * 0.90
    )

    # train_cut/val_cut computed from the FULL row count (n_total), including
    # any trailing NaN rows -- matches Preprocessor._temporal_split's /
    # the current code's existing behavior.
    train_cut_idx = int(n_total * train_frac) - 1
    val_cut_idx = int(n_total * (train_frac + val_frac)) - 1
    train_cut_idx = max(min(train_cut_idx, n_finite - 1), 0)
    val_cut_idx = max(min(val_cut_idx, n_finite - 1), 0)
    tau_train_ms = int(ts_finite_i64[train_cut_idx])
    tau_val_ms = int(ts_finite_i64[val_cut_idx])
    tau_train_h = (float(tau_train_ms) - t0_ms) / (1000.0 * 3600.0)
    tau_val_h = (float(tau_val_ms) - t0_ms) / (1000.0 * 3600.0)

    return Pass1Result(
        n=n_total, t0_ms=t0_ms, total_span_hours=total_span_hours,
        largest_gap_size_h=largest_gap_size_h, gap_end_h=gap_end_h,
        zoom_start_h=zoom_start_h, tau_train_h=tau_train_h, tau_val_h=tau_val_h,
        tau_train_ms=tau_train_ms, tau_val_ms=tau_val_ms,
    )


def argsort_and_take(ts: np.ndarray, *aux: np.ndarray) -> tuple[np.ndarray, ...]:
    """Replace `df.sort_values(col, kind="mergesort")` for a small number of
    parallel arrays with a plain (unstable) argsort + explicit numpy take.

    Why this exists (specs/71 SS2.2's diagnosis, generalized, specs/72 SS1.3):
    `DataFrame.sort_values` is expensive for three stacked reasons -- (a) it
    computes an indexer via `nargsort`, which for `kind="mergesort"`
    allocates an O(n) merge-scratch buffer beyond the O(n) indexer itself;
    (b) it then re-indexes EVERY column via `DataFrame.take`, allocating a
    fresh same-length array per column; (c) because the call is typically
    `df = df.sort_values(...)`, the pre-sort frame stays referenced (and
    therefore resident) until the whole right-hand side finishes evaluating,
    so old-frame + new-frame + indexer + mergesort-scratch are alive
    simultaneously. This function fixes (a) by using `kind="quicksort"`
    (numpy's introsort default for `np.argsort` -- O(log n) auxiliary stack,
    not O(n) scratch; unstable, which is fine here because only the VALUE at
    a rank is ever used downstream, never a tie-break-consistent row order),
    and fixes (c) implicitly: the caller is expected to rebind the result
    (e.g. `ts, codes = argsort_and_take(ts, codes)`), after which the old
    `ts`/`codes` names are the only references to the pre-sort arrays, freed
    by ordinary refcounting at reassignment.

    Does NOT fix (b) by itself -- reindexing still allocates one fresh array
    per input array. Callers should pass compact dtypes for `aux` (e.g.
    int16 class codes, not raw object/string arrays) to keep this cost down;
    this function does not silently convert dtypes itself, to avoid a
    surprising behavior change for a general-purpose reorder helper.

    Args:
        ts: 1-D array to sort by (ascending).
        *aux: Zero or more additional 1-D arrays, each the same length as
            `ts`, to be reordered by the same indexer.

    Returns:
        A tuple `(ts_sorted, aux[0]_sorted, aux[1]_sorted, ...)`, one array
        per input, each `take`n by the same indexer, dtype-preserving.
    """
    indexer = np.argsort(ts, kind="quicksort")
    out = [ts.take(indexer)]
    for a in aux:
        out.append(a.take(indexer))
    return tuple(out)


def per_class_sorted_positions(
    class_codes_sorted: np.ndarray, n_classes: int
) -> dict[int, np.ndarray]:
    """Exact per-class row-position index over an already time-sorted array
    of integer class codes.

    `{c: sorted int64 positions where class_codes_sorted == c}` for every
    `c` in `range(n_classes)`. This is `class_coverage_split_sweep.py`'s
    `class_positions` dict comprehension, factored out unchanged (down to
    the `np.flatnonzero` call) -- not an approximation of anything, not a
    histogram. Exists as a named, reusable primitive because
    `class_coverage_split_sweep.py`'s whole design (789-candidate
    `np.searchsorted`-based counting) depends on EXACT positions, not a
    fine-bin histogram (specs/72 SS1.4).

    Args:
        class_codes_sorted: 1-D integer array of class codes, already
            time-sorted (the sort order itself is irrelevant to this
            function -- it just partitions positions by class value -- but
            callers rely on the returned positions being usable with
            `np.searchsorted` against a time-sorted timestamp axis).
        n_classes: Number of distinct class codes, assumed to be
            `0, 1, ..., n_classes - 1`.

    Returns:
        A dict mapping each class code to a sorted int64 array of row
        positions.
    """
    return {
        c: np.flatnonzero(class_codes_sorted == c) for c in range(n_classes)
    }


@dataclass(frozen=True)
class Pass2Result:
    """Per-class fine-grained time-histogram bundle.

    Attributes:
        classes: Sorted list of class labels seen (str or int, caller's
            choice).
        fine_hist: Map from class label to an int64 ndarray of shape
            `(n_fine_bins,)` -- per-class flow counts in each fine time bin.
        bin_edges_h: float64 ndarray of shape `(n_fine_bins + 1,)` -- fine
            bin edges, in elapsed hours since `t0_ms`.
        n_per_class: Map from class label to total row count for that class.
        support_table: Map from class label to `[n_train, n_val, n_test]`
            row counts.
        tie_at_boundary: `{"train": int, "val": int}` -- count of rows whose
            timestamp exactly equals `tau_train_ms` / `tau_val_ms`
            (specs/71 SS3.4's exactness-measurement counter).
    """

    classes: list
    fine_hist: dict
    bin_edges_h: np.ndarray
    n_per_class: dict
    support_table: dict
    tie_at_boundary: dict


def _fine_bin_edges(
    total_span_hours: float,
    fine_bin_seconds: float,
    max_fine_bins: int,
    extra_edges_h: list[float] | None = None,
) -> tuple[np.ndarray, int]:
    """Hour-aligned fine bin edges (specs/72 SS1.5's bin-edge-alignment fix).

    Bin edges are constructed PER-HOUR, not per-span (i.e. NOT a bare
    `np.linspace(0.0, total_span_hours, n_fine_bins + 1)`, which does not, in
    general, nest with the script's own 1-hour hourly-count bins). Every hour
    boundary (0, 1, 2, ..., ceil(total_span_hours)) lands EXACTLY on a fine
    bin edge, so `np.add.reduceat` at hour-boundary bin indices reproduces
    `np.histogram(..., bins=<1-hour edges>)` exactly, not merely to within
    one fine-bin's width.

    `max_fine_bins` still caps the absolute bin count for memory safety; at
    extreme spans `n_fine_bins` is capped and the effective bins-per-hour
    shrinks below `3600.0 / fine_bin_seconds`.

    `extra_edges_h`, if given, are additional exact elapsed-hours values
    (e.g. `Pass1Result.zoom_start_h`) inserted as real bin edges rather than
    left to fall inside whichever fine bin happens to straddle them. Without
    this, a caller that later slices the histogram at such a value (e.g.
    `class_time_distribution.py`'s zoomed-region cut) would silently drop
    or double-count whichever fine bin straddles it, since a fine-bin
    histogram accumulated via `np.histogram` cannot be split after the fact
    — the split must happen at accumulation time, which is exactly what
    inserting the edge here, before any chunk is histogrammed, achieves.
    Each value already coinciding with an existing edge (within floating-
    point tolerance) or falling outside `(0, total_span_hours)` is skipped.

    Args:
        total_span_hours: Total elapsed-hours span to cover.
        fine_bin_seconds: Target fine-bin resolution, in seconds.
        max_fine_bins: Hard cap on the number of fine bins.
        extra_edges_h: Optional list of additional exact elapsed-hours edge
            values to insert into the fine-bin grid.

    Returns:
        `(bin_edges_h, n_fine_bins)`. `n_fine_bins` reflects any bins added
        by `extra_edges_h` (one extra bin per genuinely new edge inserted).
    """
    bins_per_hour = round(3600.0 / fine_bin_seconds)
    n_hours = max(1, int(np.ceil(total_span_hours)))
    if n_hours * bins_per_hour > max_fine_bins:
        # Cap binds: shrink bins-per-hour to the largest integer value that
        # still keeps n_fine_bins an EXACT multiple of n_hours, so every
        # hour boundary still lands exactly on a fine-bin edge under the
        # cap (code review finding: the previous `n_hours / n_fine_bins`
        # spacing formula only nested exactly with hour boundaries in the
        # uncapped case — capped, it silently smeared hour-boundary counts
        # by up to a full bin width, unreachable for any dataset run so far
        # but exactly the failure mode this redesign exists to avoid for an
        # unknown-span dataset like CICIoT2023). At truly extreme spans
        # (`n_hours > max_fine_bins`) even `bins_per_hour = 1` cannot fit
        # under the cap without breaking hour-alignment entirely; in that
        # regime this deliberately keeps hour-alignment and lets
        # `n_fine_bins` exceed `max_fine_bins` rather than silently
        # reintroducing the smearing bug — that span (tens of thousands of
        # hours) is far beyond anything this project has ever seen.
        bins_per_hour = max(1, max_fine_bins // n_hours)
    n_fine_bins = n_hours * bins_per_hour
    bin_edges_h = np.arange(n_fine_bins + 1) / bins_per_hour

    if extra_edges_h:
        # Tolerance for "already coincides with an existing edge": half the
        # smallest bin width this grid can represent, well under any
        # timestamp-derived value's own precision.
        tol = 0.5 * (float(n_hours) / n_fine_bins) * 1e-6
        for e in extra_edges_h:
            e = float(e)
            if not (0.0 < e < bin_edges_h[-1]):
                continue
            pos = int(np.searchsorted(bin_edges_h, e))
            if pos < len(bin_edges_h) and abs(bin_edges_h[pos] - e) <= tol:
                continue  # already an edge, nothing to insert
            if pos > 0 and abs(bin_edges_h[pos - 1] - e) <= tol:
                continue
            bin_edges_h = np.insert(bin_edges_h, pos, e)
        n_fine_bins = len(bin_edges_h) - 1

    return bin_edges_h, n_fine_bins


def per_class_time_histograms_from_csv(
    csv_path: Path | str,
    class_col: str,
    ts_col: str,
    pass1: Pass1Result,
    chunksize: int = 5_000_000,
    fine_bin_seconds: float = 1.0,
    max_fine_bins: int = 5_000_000,
) -> Pass2Result:
    """Chunked-CSV Pass 2: per-class fine-grained time histograms.

    Re-reads `class_col` + `ts_col` in chunks, accumulating one fixed-size,
    fine-grained (default ~1-second resolution, capped) histogram of
    elapsed-time per class, plus per-class/per-split scalar counters. Every
    per-class quantity `class_time_distribution.py` needs (CDF, hourly
    counts, t10/t50/t90, burst/boundary criteria, split support) is
    recomputable from the returned `Pass2Result` via `cdf_from_hist`,
    `quantile_from_hist`, `windowed_sum` -- to a precision far finer than
    anything the script prints (2 decimals of an hour).

    Used only by `class_time_distribution.py` (specs/72 SS1.5).

    `pass1.zoom_start_h` is inserted as an exact fine-bin edge (see
    `_fine_bin_edges`'s `extra_edges_h`) so that a caller slicing the
    returned histogram at the zoom boundary (as `class_time_distribution.py`
    does for its zoomed-region panel/stats) never mis-attributes the fine
    bin that would otherwise straddle it -- the split must happen here, at
    per-chunk `np.histogram` accumulation time, not after the fact.

    Args:
        csv_path: Path to the raw NetFlow CSV.
        class_col: Name of the class/label string column (e.g. "Attack").
        ts_col: Name of the millisecond-timestamp column.
        pass1: The `Pass1Result` already computed for this CSV (supplies
            `t0_ms`, `total_span_hours`, `tau_train_ms`, `tau_val_ms`,
            `zoom_start_h`).
        chunksize: Rows per `pd.read_csv` chunk.
        fine_bin_seconds: Target fine-bin resolution, in seconds.
        max_fine_bins: Hard cap on the number of fine bins (memory safety).

    Returns:
        A populated `Pass2Result`.
    """
    bin_edges_h, n_fine_bins = _fine_bin_edges(
        pass1.total_span_hours, fine_bin_seconds, max_fine_bins,
        extra_edges_h=[pass1.zoom_start_h],
    )

    fine_hist: dict = {}
    n_per_class: dict = {}
    support_table: dict = {}
    tie_at_boundary = {"train": 0, "val": 0}

    for chunk in pd.read_csv(
        csv_path, usecols=[class_col, ts_col],
        dtype={ts_col: "int64"}, chunksize=chunksize,
    ):
        ts = chunk[ts_col].to_numpy()
        elapsed = (ts - pass1.t0_ms) / (1000.0 * 3600.0)
        is_train = ts <= pass1.tau_train_ms
        is_val = (ts > pass1.tau_train_ms) & (ts <= pass1.tau_val_ms)
        is_test = ts > pass1.tau_val_ms
        at_train_boundary = ts == pass1.tau_train_ms
        at_val_boundary = ts == pass1.tau_val_ms

        for cls in chunk[class_col].unique():
            mask = (chunk[class_col] == cls).to_numpy()
            fine_hist.setdefault(cls, np.zeros(n_fine_bins, dtype=np.int64))
            counts, _ = np.histogram(elapsed[mask], bins=bin_edges_h)
            fine_hist[cls] += counts
            n_per_class[cls] = n_per_class.get(cls, 0) + int(mask.sum())
            support_table.setdefault(cls, [0, 0, 0])
            support_table[cls][0] += int((mask & is_train).sum())
            support_table[cls][1] += int((mask & is_val).sum())
            support_table[cls][2] += int((mask & is_test).sum())

        tie_at_boundary["train"] += int(at_train_boundary.sum())
        tie_at_boundary["val"] += int(at_val_boundary.sum())

    classes = sorted(fine_hist.keys())
    return Pass2Result(
        classes=classes, fine_hist=fine_hist, bin_edges_h=bin_edges_h,
        n_per_class=n_per_class, support_table=support_table,
        tie_at_boundary=tie_at_boundary,
    )


def per_class_time_histograms_from_arrays(
    class_codes_sorted: np.ndarray,
    ts_sorted: np.ndarray,
    pass1: Pass1Result,
    fine_bin_seconds: float = 1.0,
    max_fine_bins: int = 5_000_000,
) -> Pass2Result:
    """Same Pass 2 computation, given already-in-memory, already-time-sorted
    parallel arrays instead of a CSV path.

    Not wired into any of the three scripts' call sites in this
    implementation pass -- included for API symmetry with the from-csv
    variant and because computing `support_table`/`tie_at_boundary`/
    `n_per_class` this way is a legitimate, reusable shape for a future
    caller with the same "already-sorted-arrays, want a fine histogram"
    need (specs/72 SS1.5).

    Args:
        class_codes_sorted: 1-D integer array of class codes, time-sorted,
            aligned to `ts_sorted`.
        ts_sorted: 1-D int64 array of timestamps (milliseconds), sorted
            ascending, aligned to `class_codes_sorted`.
        pass1: The `Pass1Result` already computed for these arrays.
        fine_bin_seconds: Target fine-bin resolution, in seconds.
        max_fine_bins: Hard cap on the number of fine bins.

    Returns:
        A populated `Pass2Result`, keyed by the integer class codes present
        in `class_codes_sorted`.
    """
    bin_edges_h, n_fine_bins = _fine_bin_edges(
        pass1.total_span_hours, fine_bin_seconds, max_fine_bins,
        extra_edges_h=[pass1.zoom_start_h],
    )

    elapsed = (ts_sorted.astype(np.float64) - pass1.t0_ms) / (1000.0 * 3600.0)
    is_train = ts_sorted <= pass1.tau_train_ms
    is_val = (ts_sorted > pass1.tau_train_ms) & (ts_sorted <= pass1.tau_val_ms)
    is_test = ts_sorted > pass1.tau_val_ms
    at_train_boundary = ts_sorted == pass1.tau_train_ms
    at_val_boundary = ts_sorted == pass1.tau_val_ms

    fine_hist: dict = {}
    n_per_class: dict = {}
    support_table: dict = {}
    for cls in np.unique(class_codes_sorted):
        cls_key = cls.item() if hasattr(cls, "item") else cls
        mask = class_codes_sorted == cls
        counts, _ = np.histogram(elapsed[mask], bins=bin_edges_h)
        fine_hist[cls_key] = counts.astype(np.int64)
        n_per_class[cls_key] = int(mask.sum())
        support_table[cls_key] = [
            int((mask & is_train).sum()),
            int((mask & is_val).sum()),
            int((mask & is_test).sum()),
        ]

    tie_at_boundary = {
        "train": int(at_train_boundary.sum()),
        "val": int(at_val_boundary.sum()),
    }
    classes = sorted(fine_hist.keys())
    return Pass2Result(
        classes=classes, fine_hist=fine_hist, bin_edges_h=bin_edges_h,
        n_per_class=n_per_class, support_table=support_table,
        tie_at_boundary=tie_at_boundary,
    )


def cdf_from_hist(
    fine_hist_cls: np.ndarray, bin_edges_h: np.ndarray, n_plot_points: int = 2000
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative-fraction curve for one class, downsampled for plotting.

    Returns `(x, y)` arrays: `x` are elapsed-hour bin edges (excluding the
    leading 0), `y` is the cumulative fraction of that class's own flows
    seen by `x`. Downsampled to at most `n_plot_points` points via index
    subsampling of the cumulative array -- plotting one vertex per raw flow
    (as the original per-row `ax.plot(ct, frac, ...)` did) is both a memory
    and rendering-practicality problem at multi-hundred-million-row scale,
    independent of any statistical concern.

    Args:
        fine_hist_cls: int64 array of shape `(n_fine_bins,)`, this class's
            fine-bin counts.
        bin_edges_h: float64 array of shape `(n_fine_bins + 1,)`.
        n_plot_points: Maximum number of points to return.

    Returns:
        `(x, y)`, each a float64 ndarray of length <= `n_plot_points`.
    """
    total = fine_hist_cls.sum()
    if total == 0:
        return np.empty(0), np.empty(0)
    cum = np.cumsum(fine_hist_cls) / total
    x_full = bin_edges_h[1:]
    n = len(x_full)
    if n <= n_plot_points:
        return x_full, cum
    idx = np.linspace(0, n - 1, n_plot_points).astype(np.int64)
    return x_full[idx], cum[idx]


def quantile_from_hist(
    fine_hist_cls: np.ndarray, bin_edges_h: np.ndarray, q: float, offset_h: float = 0.0
) -> float:
    """Elapsed-hours value at which a class's cumulative fraction reaches `q`.

    Replaces the `t_at`/`tz_at` closures in the original per-row code:
    `np.searchsorted` on `cumsum(fine_hist_cls) / fine_hist_cls.sum()`
    against `q`, returning the corresponding `bin_edges_h` value.

    Args:
        fine_hist_cls: int64 array of shape `(n_fine_bins,)`.
        bin_edges_h: float64 array of shape `(n_fine_bins + 1,)`.
        q: Target cumulative fraction, in `[0, 1]`.
        offset_h: Subtracted from the result -- used for the zoomed-region
            case, which reports hours SINCE the zoom start rather than since
            `t0_ms` (matching the original code's `ct_zoom = ct_full[...] -
            zoom_start_h` semantics).

    Returns:
        Elapsed hours (minus `offset_h`) at which the cumulative fraction
        first reaches `q`.
    """
    total = fine_hist_cls.sum()
    if total == 0:
        return 0.0
    cum = np.cumsum(fine_hist_cls) / total
    idx = int(np.searchsorted(cum, q))
    idx = min(idx, len(fine_hist_cls) - 1)
    return float(bin_edges_h[idx + 1]) - offset_h


def windowed_sum(
    fine_hist_cls: np.ndarray, bin_edges_h: np.ndarray, center_h: float, half_width_h: float
) -> int:
    """Sum of a class's fine-bin counts whose bin centers fall within a window.

    Replaces criterion B's exact `np.abs(ct_zoom - tau_h) <= BOUNDARY_WIN_H`
    row-count with a sum over fine bins whose centers fall in
    `[center_h - half_width_h, center_h + half_width_h]`.

    Args:
        fine_hist_cls: int64 array of shape `(n_fine_bins,)`.
        bin_edges_h: float64 array of shape `(n_fine_bins + 1,)`.
        center_h: Window center, in elapsed hours.
        half_width_h: Window half-width, in elapsed hours.

    Returns:
        Sum of `fine_hist_cls` over fine bins whose centers fall in the
        window, as a Python int.
    """
    bin_centers_h = (bin_edges_h[:-1] + bin_edges_h[1:]) / 2.0
    in_window = np.abs(bin_centers_h - center_h) <= half_width_h
    return int(fine_hist_cls[in_window].sum())
