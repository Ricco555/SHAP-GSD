"""Unit tests for explore/_temporal_streaming.py (specs/72 SS5.1).

All fixtures are small, synthetic, seeded CSVs built in `tmp_path` -- no
dependency on any real NetFlow dataset. Covers:

  1 -- `pooled_timestamp_stats_from_csv` vs `pooled_timestamp_stats_from_array`
       agreement (bit-identical `Pass1Result` for the same underlying data).
  2 -- Chunking equivalence (`chunksize=100` vs one large chunk).
  3 -- `argsort_and_take` correctness (sorted, row-aligned, dtype-preserving).
  4 -- `per_class_sorted_positions` correctness (vs `np.flatnonzero` directly).
  5 -- NaN-timestamp handling for the array-based Pass 1 variant.
  6 -- Tie-break invariance: stable vs. unstable sort give the same
       train/val/test assignment when rows share the exact boundary value.
  7 -- The hour-alignment bin-edge fix / `extra_edges_h` zoom-boundary fix:
       a burst of rows in the same ~1-second window as a zoom-region
       boundary must not get smeared across it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from explore._temporal_streaming import (
    Pass1Result,
    argsort_and_take,
    per_class_sorted_positions,
    per_class_time_histograms_from_csv,
    pooled_timestamp_stats_from_array,
    pooled_timestamp_stats_from_csv,
)

TS_COL = "FLOW_START_MILLISECONDS"
CLASS_COL = "Attack"

# A ~10-hour idle gap between two dense blocks -- large enough (relative to
# the ~1 ms inter-row spacing within each block) that the gap-detection rule
# (largest_gap_size_h >= 5% of total_span_hours) unambiguously selects it,
# regardless of the small perturbations individual tests add near the
# boundary.
_GAP_MS = 36_000_000  # 10 hours, in milliseconds
_CLASS_NAMES = ["Benign", "A", "B", "C"]


def _build_two_block_fixture(
    n_per_block: int = 500,
    tie_at_rank: int | None = None,
    tie_width: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a sorted (ts, class) pair: two dense blocks separated by a gap.

    Block 1: `n_per_block` rows, 1000 ms apart, starting at t=0.
    Block 2: `n_per_block` rows, 1000 ms apart, starting `_GAP_MS` after
        block 1's last row.
    Classes cycle through `_CLASS_NAMES`.

    If `tie_at_rank` is given, the value at that rank (and the following
    `tie_width - 1` ranks) is overwritten with the ORIGINAL value at
    `tie_at_rank`, creating a genuine tie block of `tie_width` rows sharing
    one timestamp -- while keeping the array non-decreasing (a prerequisite
    for this being a valid "already sorted" fixture).

    Returns:
        `(ts, classes)`, both length `2 * n_per_block`, sorted ascending by
        `ts` (with a tie block, if requested).
    """
    ts = np.empty(2 * n_per_block, dtype=np.int64)
    for i in range(n_per_block):
        ts[i] = i * 1000
    block2_start = (n_per_block - 1) * 1000 + _GAP_MS
    for i in range(n_per_block):
        ts[n_per_block + i] = block2_start + i * 1000

    classes = np.array(
        [_CLASS_NAMES[i % len(_CLASS_NAMES)] for i in range(2 * n_per_block)],
        dtype=object,
    )

    if tie_at_rank is not None:
        tie_value = ts[tie_at_rank]
        for k in range(tie_width):
            ts[tie_at_rank + k] = tie_value

    return ts, classes


def _write_shuffled_csv(
    tmp_path: Path, ts: np.ndarray, classes: np.ndarray, name: str = "synthetic.csv"
) -> Path:
    """Write `(ts, classes)` to a CSV in a fixed, deterministically shuffled
    row order -- so the streaming utilities are exercised against out-of-order
    input, not input that is accidentally already sorted."""
    df = pd.DataFrame({TS_COL: ts, CLASS_COL: classes})
    rng = np.random.default_rng(1234)
    perm = rng.permutation(len(df))
    df = df.iloc[perm].reset_index(drop=True)
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# 1 -- pooled_timestamp_stats_from_csv vs pooled_timestamp_stats_from_array
# ---------------------------------------------------------------------------

def test_pooled_timestamp_stats_csv_and_array_agree_bit_identically(tmp_path):
    ts, classes = _build_two_block_fixture(n_per_block=500, tie_at_rank=599, tie_width=3)
    csv_path = _write_shuffled_csv(tmp_path, ts, classes)

    from_csv = pooled_timestamp_stats_from_csv(
        csv_path, ts_col=TS_COL, train_frac=0.6, val_frac=0.2, chunksize=5_000_000
    )

    # Array variant fed the SAME (unsorted-row-order) timestamp column, read
    # straight off disk -- not the in-memory `ts` array above, so this is a
    # genuine independent read, not just reusing the same object.
    ts_from_disk = pd.read_csv(csv_path, usecols=[TS_COL])[TS_COL].to_numpy(dtype=np.int64)
    from_array = pooled_timestamp_stats_from_array(ts_from_disk, train_frac=0.6, val_frac=0.2)

    assert from_csv == from_array


# ---------------------------------------------------------------------------
# 2 -- Chunking equivalence
# ---------------------------------------------------------------------------

def test_chunking_equivalence_pass1(tmp_path):
    ts, classes = _build_two_block_fixture(n_per_block=500)
    csv_path = _write_shuffled_csv(tmp_path, ts, classes)

    small_chunks = pooled_timestamp_stats_from_csv(
        csv_path, ts_col=TS_COL, train_frac=0.6, val_frac=0.2, chunksize=100
    )
    one_chunk = pooled_timestamp_stats_from_csv(
        csv_path, ts_col=TS_COL, train_frac=0.6, val_frac=0.2, chunksize=10_000_000
    )

    assert small_chunks == one_chunk


def test_chunking_equivalence_pass2(tmp_path):
    ts, classes = _build_two_block_fixture(n_per_block=500)
    csv_path = _write_shuffled_csv(tmp_path, ts, classes)
    pass1 = pooled_timestamp_stats_from_csv(
        csv_path, ts_col=TS_COL, train_frac=0.6, val_frac=0.2, chunksize=10_000_000
    )

    small_chunks = per_class_time_histograms_from_csv(
        csv_path, class_col=CLASS_COL, ts_col=TS_COL, pass1=pass1, chunksize=100,
        fine_bin_seconds=1.0,
    )
    one_chunk = per_class_time_histograms_from_csv(
        csv_path, class_col=CLASS_COL, ts_col=TS_COL, pass1=pass1, chunksize=10_000_000,
        fine_bin_seconds=1.0,
    )

    assert small_chunks.classes == one_chunk.classes
    assert small_chunks.n_per_class == one_chunk.n_per_class
    assert small_chunks.support_table == one_chunk.support_table
    assert small_chunks.tie_at_boundary == one_chunk.tie_at_boundary
    np.testing.assert_array_equal(small_chunks.bin_edges_h, one_chunk.bin_edges_h)
    for cls in small_chunks.classes:
        np.testing.assert_array_equal(small_chunks.fine_hist[cls], one_chunk.fine_hist[cls])


# ---------------------------------------------------------------------------
# 3 -- argsort_and_take correctness
# ---------------------------------------------------------------------------

def test_argsort_and_take_sorts_and_keeps_aux_aligned_and_dtype_preserving():
    rng = np.random.default_rng(7)
    n = 200
    ts = rng.integers(0, 1000, size=n).astype(np.int64)
    # Force a genuine tie block at a known value.
    ts[10:15] = 500
    aux = rng.integers(0, 4, size=n).astype(np.int16)

    # Build the ground-truth set of (ts, aux) pairs before sorting.
    pairs_before = set(zip(ts.tolist(), aux.tolist()))

    ts_sorted, aux_sorted = argsort_and_take(ts, aux)

    # (a) sorted ascending
    assert np.all(np.diff(ts_sorted) >= 0)
    # (b) aux stays row-aligned -- same multiset of (ts, aux) pairs, just reordered
    pairs_after = set(zip(ts_sorted.tolist(), aux_sorted.tolist()))
    assert pairs_before == pairs_after
    assert sorted(zip(ts.tolist(), aux.tolist())) == sorted(zip(ts_sorted.tolist(), aux_sorted.tolist()))
    # (c) dtype-preserving -- no silent upcast of the compact aux dtype
    assert aux_sorted.dtype == np.int16
    assert ts_sorted.dtype == np.int64


def test_argsort_and_take_no_aux_arrays():
    ts = np.array([3, 1, 2], dtype=np.int64)
    (ts_sorted,) = argsort_and_take(ts)
    np.testing.assert_array_equal(ts_sorted, np.array([1, 2, 3], dtype=np.int64))


# ---------------------------------------------------------------------------
# 4 -- per_class_sorted_positions correctness
# ---------------------------------------------------------------------------

def test_per_class_sorted_positions_matches_flatnonzero_directly():
    rng = np.random.default_rng(3)
    n_classes = 6
    class_codes_sorted = rng.integers(0, n_classes, size=500).astype(np.int16)

    result = per_class_sorted_positions(class_codes_sorted, n_classes)

    assert set(result.keys()) == set(range(n_classes))
    for c in range(n_classes):
        expected = np.flatnonzero(class_codes_sorted == c)
        np.testing.assert_array_equal(result[c], expected)


def test_per_class_sorted_positions_handles_absent_class():
    # Class 2 never appears -- must still be present as an explicit empty array.
    class_codes_sorted = np.array([0, 0, 1, 1, 3, 3], dtype=np.int16)
    result = per_class_sorted_positions(class_codes_sorted, n_classes=4)
    assert 2 in result
    assert len(result[2]) == 0


# ---------------------------------------------------------------------------
# 5 -- NaN-timestamp handling for the array-based Pass 1 variant
# ---------------------------------------------------------------------------

def test_pooled_timestamp_stats_from_array_nan_sorts_last_and_excluded_from_span():
    n_clean = 100
    ts_clean = np.arange(n_clean, dtype=np.float64) * 1000.0
    ts_with_nan = np.concatenate([ts_clean, [np.nan]])
    rng = np.random.default_rng(9)
    ts_with_nan = ts_with_nan[rng.permutation(len(ts_with_nan))]

    result = pooled_timestamp_stats_from_array(ts_with_nan, train_frac=0.6, val_frac=0.2)
    result_clean = pooled_timestamp_stats_from_array(ts_clean, train_frac=0.6, val_frac=0.2)

    # n includes the NaN row (full row count, matching Preprocessor._temporal_split's
    # convention of computing cuts from the full row count).
    assert result.n == n_clean + 1
    # total_span_hours / t0_ms / gap fields are unaffected by the trailing NaN
    # -- computed only from the finite prefix, matching the current
    # np.nanmax-based behavior this replaces.
    assert result.t0_ms == result_clean.t0_ms
    assert result.total_span_hours == pytest.approx(result_clean.total_span_hours)
    assert np.isfinite(result.total_span_hours)
    assert np.isfinite(result.largest_gap_size_h)
    assert np.isfinite(result.gap_end_h)
    assert np.isfinite(result.zoom_start_h)
    assert np.isfinite(result.tau_train_h)
    assert np.isfinite(result.tau_val_h)


def test_pooled_timestamp_stats_from_array_nan_does_not_mutate_caller_array():
    ts = np.array([3.0, 1.0, np.nan, 2.0])
    ts_before = ts.copy()
    pooled_timestamp_stats_from_array(ts, train_frac=0.5, val_frac=0.3)
    np.testing.assert_array_equal(ts, ts_before, err_msg="caller's array must not be mutated in place")


# ---------------------------------------------------------------------------
# 6 -- Tie-break invariance
# ---------------------------------------------------------------------------

def test_tie_break_invariance_split_assignment_identical_stable_vs_unstable():
    """A fixture with several rows sharing the exact tau_train boundary
    millisecond: value-based masking must give the same train/val/test
    assignment regardless of whether the sort that located the boundary
    value was stable (mergesort, the old `sort_values` behavior) or
    unstable (quicksort, `argsort_and_take`'s behavior) -- specs/72 SS4.2
    point 2's invariance argument, checked empirically here rather than
    just asserted in prose.
    """
    n_per_block = 500
    train_frac, val_frac = 0.6, 0.2
    n_total = 2 * n_per_block
    train_cut_idx = int(n_total * train_frac) - 1  # rank at which the tie is placed

    ts, classes = _build_two_block_fixture(
        n_per_block=n_per_block, tie_at_rank=train_cut_idx, tie_width=4
    )
    rng = np.random.default_rng(55)
    perm = rng.permutation(len(ts))
    ts_shuffled = ts[perm]
    codes_shuffled = np.array(
        [_CLASS_NAMES.index(c) for c in classes[perm]], dtype=np.int16
    )

    # Unstable path: argsort_and_take (what the refactored scripts use).
    ts_unstable, codes_unstable = argsort_and_take(ts_shuffled, codes_shuffled)

    # Stable path: what df.sort_values(kind="mergesort") would have done.
    stable_indexer = np.argsort(ts_shuffled, kind="stable")
    ts_stable = ts_shuffled[stable_indexer]
    codes_stable = codes_shuffled[stable_indexer]

    # The sorted VALUE arrays must be identical regardless of tie-break
    # (this is the crux of the invariance argument: only the row order
    # within a tie block can differ, never the values at each rank).
    np.testing.assert_array_equal(ts_unstable, ts_stable)

    def _tau_train_ms(ts_sorted: np.ndarray) -> int:
        return int(ts_sorted[train_cut_idx])

    tau_unstable = _tau_train_ms(ts_unstable)
    tau_stable = _tau_train_ms(ts_stable)
    assert tau_unstable == tau_stable

    # Value-based masking (Preprocessor._temporal_split's actual mechanism:
    # ts <= tau_train_ms) over the ORIGINAL, unsorted array must give the
    # identical train-row count and identical set of row identities
    # (by (ts, class) pair) under both taus.
    train_mask_unstable = ts_shuffled <= tau_unstable
    train_mask_stable = ts_shuffled <= tau_stable
    np.testing.assert_array_equal(train_mask_unstable, train_mask_stable)

    # And the per-split class composition (what compute_class_coverage's
    # `counts` dict reports) is therefore also identical.
    for split_mask_unstable, split_mask_stable in (
        (ts_unstable <= tau_unstable, ts_stable <= tau_stable),
    ):
        n_train_unstable = int(split_mask_unstable.sum())
        n_train_stable = int(split_mask_stable.sum())
        assert n_train_unstable == n_train_stable
        np.testing.assert_array_equal(
            np.bincount(codes_unstable[split_mask_unstable], minlength=len(_CLASS_NAMES)),
            np.bincount(codes_stable[split_mask_stable], minlength=len(_CLASS_NAMES)),
        )


# ---------------------------------------------------------------------------
# 7 -- Hour-alignment bin-edge fix / extra_edges_h zoom-boundary fix
# ---------------------------------------------------------------------------

def test_zoom_boundary_burst_not_smeared_across_boundary(tmp_path):
    """Regression test for the real bug found/fixed during Stage 2: a naive
    `np.linspace(0, total_span_hours, n_fine_bins + 1)` bin grid does not, in
    general, place a fine-bin edge exactly at `zoom_start_h`, so a burst of
    rows landing in the same fine bin as the zoom boundary would be silently
    mis-attributed to the wrong side once the histogram is sliced there.
    `_fine_bin_edges`'s `extra_edges_h` mechanism inserts `zoom_start_h` as a
    genuine bin edge, closing this. This test builds a burst of rows
    clustered within about a second of the boundary and checks the
    fine-histogram-derived zoomed count against a brute-force count computed
    directly from `elapsed_hours >= zoom_start_h`.
    """
    n_per_block = 500
    ts, classes = _build_two_block_fixture(n_per_block=n_per_block)

    # First pass: find out where the gap (and therefore zoom_start_h) will
    # land, using the fixture alone.
    csv_path_base = _write_shuffled_csv(tmp_path, ts, classes, name="base.csv")
    pass1_base = pooled_timestamp_stats_from_csv(
        csv_path_base, ts_col=TS_COL, train_frac=0.6, val_frac=0.2
    )
    boundary_ms = pass1_base.t0_ms + int(round(pass1_base.zoom_start_h * 3600.0 * 1000.0))

    # Now build a burst of rows straddling that exact millisecond, within a
    # ~1-second window on both sides -- the scenario the original bug
    # mishandled.
    rng = np.random.default_rng(21)
    offsets_ms = rng.integers(-900, 900, size=40)
    burst_ts = (boundary_ms + offsets_ms).astype(np.int64)
    burst_classes = np.array(
        [_CLASS_NAMES[i % len(_CLASS_NAMES)] for i in range(len(burst_ts))], dtype=object
    )

    ts_full = np.concatenate([ts, burst_ts])
    classes_full = np.concatenate([classes, burst_classes])
    csv_path = _write_shuffled_csv(tmp_path, ts_full, classes_full, name="burst.csv")

    pass1 = pooled_timestamp_stats_from_csv(
        csv_path, ts_col=TS_COL, train_frac=0.6, val_frac=0.2
    )
    pass2 = per_class_time_histograms_from_csv(
        csv_path, class_col=CLASS_COL, ts_col=TS_COL, pass1=pass1, fine_bin_seconds=1.0,
    )

    # zoom_start_h must land EXACTLY on a fine-bin edge (this is the fix
    # itself, checked directly).
    zoom_idx = int(np.searchsorted(pass2.bin_edges_h, pass1.zoom_start_h))
    assert zoom_idx < len(pass2.bin_edges_h)
    assert pass2.bin_edges_h[zoom_idx] == pytest.approx(pass1.zoom_start_h, abs=1e-9)

    # Brute-force ground truth, straight from the raw arrays.
    df_full = pd.read_csv(csv_path)
    elapsed_full = (df_full[TS_COL].to_numpy(dtype=np.int64) - pass1.t0_ms) / (1000.0 * 3600.0)
    class_full = df_full[CLASS_COL].to_numpy()

    for cls in pass2.classes:
        brute_force_zoomed_n = int(
            np.sum((class_full == cls) & (elapsed_full >= pass1.zoom_start_h))
        )
        fine_hist_zoomed_n = int(pass2.fine_hist[cls][zoom_idx:].sum())
        assert fine_hist_zoomed_n == brute_force_zoomed_n, (
            f"class {cls!r}: fine-histogram zoomed count {fine_hist_zoomed_n} != "
            f"brute-force count {brute_force_zoomed_n}"
        )
