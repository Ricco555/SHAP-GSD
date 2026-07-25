"""
Unit tests for the searchsorted-based EID-to-position lookup in ``FeatureStore``.

These tests do not require a completed pipeline run — they build synthetic
feature stores via ``write_feature_store`` (mirroring the ``test_edges_meta.py``
convention) and exercise ``__getitem__``, ``get_batch``, and ``get_labels_batch``
directly against known, deterministic fixtures.

Covered:
1. Present-EID lookup on both a contiguous range (mirrors the real pipeline)
   and a deliberately non-contiguous sorted array (tests the documented
   contract — sortedness only, not contiguity).
2. Absent-EID handling: below range, above range (the ``pos == n`` boundary
   trap), and strictly between two present entries — all must raise
   ``KeyError``, never silently return a wrong row or raise ``IndexError``.
3. Batch lookups in shuffled order, and batch failure on any absent EID
   (both "absent is first" and "absent is last" sub-cases).
4. ``_eid_start``/``_eid_end`` still match the sorted array's endpoints.
5. A regression oracle reproducing the deleted dict-based lookup exactly.
6. Confirms ``_eid_to_pos`` no longer exists as an attribute.
"""

from pathlib import Path

import numpy as np
import pytest

from src.data.feature_store import FeatureStore, write_feature_store


def _make_store(tmp_path: Path, edge_indices: np.ndarray, d_e: int = 4) -> FeatureStore:
    """Write a synthetic feature store for the given (sorted) edge_indices
    and return it opened as a FeatureStore. Feature row i == geid * 1.0
    broadcast across d_e columns, and label i == geid % 10 — both
    deterministic functions of the global EID, so a test can predict the
    expected row/label for any EID without a separate ground-truth table.
    """
    features = np.tile(edge_indices.astype(np.float32).reshape(-1, 1), (1, d_e))
    timestamps = edge_indices.astype(np.int64) * 1000
    labels = (edge_indices % 10).astype(np.int64)
    write_feature_store(tmp_path, features, edge_indices, timestamps, labels)
    return FeatureStore(tmp_path)


def test_getitem_present_contiguous(tmp_path: Path) -> None:
    """Contiguous edge_indices (mirrors the real pipeline's contiguous-range
    case): first, last, and several random present EIDs return expected rows."""
    edge_indices = np.arange(500, 1500, dtype=np.int64)
    fs = _make_store(tmp_path, edge_indices)

    for eid in [500, 1499]:
        row = fs[eid]
        assert np.allclose(row, float(eid))

    rng = np.random.default_rng(1)
    for eid in rng.choice(edge_indices, size=20, replace=False):
        row = fs[int(eid)]
        assert np.allclose(row, float(eid))


def test_getitem_present_noncontiguous(tmp_path: Path) -> None:
    """Deliberately non-contiguous sorted edge_indices: every present EID
    resolves to its correct row (tests sortedness-only contract)."""
    edge_indices = np.array([3, 7, 8, 19, 42, 1000], dtype=np.int64)
    fs = _make_store(tmp_path, edge_indices)

    for eid in edge_indices:
        row = fs[int(eid)]
        assert np.allclose(row, float(eid))


def test_getitem_absent_below_range(tmp_path: Path) -> None:
    """Query below the first present EID (insertion point 0) raises KeyError,
    not a silent return of position 0's row."""
    edge_indices = np.array([3, 7, 8, 19, 42, 1000], dtype=np.int64)
    fs = _make_store(tmp_path, edge_indices)

    with pytest.raises(KeyError):
        fs[int(edge_indices[0]) - 1]


def test_getitem_absent_above_range(tmp_path: Path) -> None:
    """The pos == n boundary trap: querying above the last present EID must
    raise KeyError, not IndexError. Single most important test in this file."""
    edge_indices = np.array([3, 7, 8, 19, 42, 1000], dtype=np.int64)
    fs = _make_store(tmp_path, edge_indices)

    with pytest.raises(KeyError):
        fs[int(edge_indices[-1]) + 1]


def test_getitem_absent_between_present(tmp_path: Path) -> None:
    """A value strictly between two present entries raises KeyError, proving
    the equality check (not just the bounds check) is exercised."""
    edge_indices = np.array([3, 7, 8, 19, 42, 1000], dtype=np.int64)
    fs = _make_store(tmp_path, edge_indices)

    with pytest.raises(KeyError):
        fs[10]  # strictly between 8 and 19


def test_get_batch_present_shuffled_order(tmp_path: Path) -> None:
    """A batch of present EIDs in shuffled (non-sorted) order returns rows
    matching input order exactly."""
    edge_indices = np.arange(500, 1500, dtype=np.int64)
    fs = _make_store(tmp_path, edge_indices)

    rng = np.random.default_rng(2)
    batch = rng.choice(edge_indices, size=15, replace=False)
    rng.shuffle(batch)

    rows = fs.get_batch(batch)
    for i, eid in enumerate(batch):
        assert np.allclose(rows[i], float(eid))


def test_get_batch_raises_on_any_absent(tmp_path: Path) -> None:
    """A batch mixing present EIDs with one absent EID raises KeyError,
    for both 'absent is last' and 'absent is first' sub-cases."""
    edge_indices = np.arange(500, 1500, dtype=np.int64)
    fs = _make_store(tmp_path, edge_indices)

    present = [510, 520, 530]
    absent = 999999

    with pytest.raises(KeyError):
        fs.get_batch(np.array(present + [absent], dtype=np.int64))

    with pytest.raises(KeyError):
        fs.get_batch(np.array([absent] + present, dtype=np.int64))


def test_get_labels_batch_matches_labels_array(tmp_path: Path) -> None:
    """get_labels_batch on shuffled present EIDs matches the known
    label = geid % 10 fixture function, and raises KeyError on any absent EID
    (both sub-cases)."""
    edge_indices = np.arange(500, 1500, dtype=np.int64)
    fs = _make_store(tmp_path, edge_indices)

    rng = np.random.default_rng(3)
    batch = rng.choice(edge_indices, size=15, replace=False)
    rng.shuffle(batch)

    labels = fs.get_labels_batch(batch)
    expected = (batch % 10).astype(np.int64)
    assert np.array_equal(labels, expected)

    present = [510, 520, 530]
    absent = 999999

    with pytest.raises(KeyError):
        fs.get_labels_batch(np.array(present + [absent], dtype=np.int64))

    with pytest.raises(KeyError):
        fs.get_labels_batch(np.array([absent] + present, dtype=np.int64))


def test_eid_start_end_match_sorted_endpoints(tmp_path: Path) -> None:
    """_eid_start/_eid_end match the sorted edge_indices array's endpoints."""
    edge_indices = np.array([3, 7, 8, 19, 42, 1000], dtype=np.int64)
    fs = _make_store(tmp_path, edge_indices)

    assert fs._eid_start == int(edge_indices[0])
    assert fs._eid_end == int(edge_indices[-1])


def test_regression_matches_old_dict_lookup(tmp_path: Path) -> None:
    """Reproduces the deleted dict-based lookup inline as a reference oracle
    and asserts agreement on 500 sampled EIDs."""
    edge_indices = np.arange(50_000, 100_000, dtype=np.int64)
    fs = _make_store(tmp_path, edge_indices)
    old_eid_to_pos = {int(eid): pos for pos, eid in enumerate(edge_indices)}
    rng = np.random.default_rng(2026)
    sample = rng.choice(edge_indices, size=500, replace=False)
    for eid in sample:
        old_pos = old_eid_to_pos[int(eid)]
        assert fs._pos_of(int(eid)) == old_pos
        assert np.array_equal(fs[int(eid)], fs._mmap[old_pos])
    batch_positions_new = fs._pos_of_batch(sample)
    batch_positions_old = np.array([old_eid_to_pos[int(e)] for e in sample])
    assert np.array_equal(batch_positions_new, batch_positions_old)


def test_no_eid_to_pos_attribute(tmp_path: Path) -> None:
    """The deleted dict-based lookup no longer exists as an attribute."""
    edge_indices = np.arange(500, 1500, dtype=np.int64)
    fs = _make_store(tmp_path, edge_indices)

    assert not hasattr(fs, "_eid_to_pos")


def test_construction_asserts_sortedness(tmp_path: Path) -> None:
    """An unsorted (or duplicate-containing) edge_indices.npy must fail
    loudly at construction time, not silently corrupt later lookups. Writes
    the on-disk files directly (bypassing write_feature_store, which would
    itself hit this same assertion during its own readback check) so the
    violation is only detected by FeatureStore.__init__ itself."""
    unsorted = np.array([3, 19, 8, 7, 42, 1000], dtype=np.int64)
    n, d_e = len(unsorted), 4
    features = np.tile(unsorted.astype(np.float32).reshape(-1, 1), (1, d_e))
    timestamps = unsorted.astype(np.int64) * 1000
    labels = (unsorted % 10).astype(np.int64)

    mmap = np.memmap(tmp_path / "features.dat", dtype=np.float32, mode="w+", shape=(n, d_e))
    mmap[:] = features
    del mmap
    np.save(tmp_path / "edge_indices.npy", unsorted)
    np.save(tmp_path / "timestamps.npy", timestamps)
    np.save(tmp_path / "labels.npy", labels)

    with pytest.raises(AssertionError):
        FeatureStore(tmp_path)


def test_construction_asserts_no_duplicates(tmp_path: Path) -> None:
    """A sorted-but-duplicate-containing edge_indices.npy must also fail
    loudly — strict ascending (not merely non-decreasing) is required."""
    dup = np.array([3, 7, 7, 19, 42, 1000], dtype=np.int64)
    n, d_e = len(dup), 4
    features = np.tile(dup.astype(np.float32).reshape(-1, 1), (1, d_e))
    timestamps = dup.astype(np.int64) * 1000
    labels = (dup % 10).astype(np.int64)

    mmap = np.memmap(tmp_path / "features.dat", dtype=np.float32, mode="w+", shape=(n, d_e))
    mmap[:] = features
    del mmap
    np.save(tmp_path / "edge_indices.npy", dup)
    np.save(tmp_path / "timestamps.npy", timestamps)
    np.save(tmp_path / "labels.npy", labels)

    with pytest.raises(AssertionError):
        FeatureStore(tmp_path)
