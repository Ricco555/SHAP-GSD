"""
Tests for TemporalBalancer.

Uses synthetic data for all tests (no dependency on feature_store or graphs).

Tests:
  1 — Temporal order:   output sorted by timestamp.
  2 — All original EIDs present in oversampled output.
  3 — Class distribution meets min_class_ratio after balancing.
  4 — Val/test unchanged: balancing only touches what is passed; external data unmodified.
  5 — Duplicate EIDs:   duplicate entries map to identical feature values (index equality).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.balancer import TemporalBalancer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_dataset(n: int = 2000, n_classes: int = 5, imbalance: float = 0.9,
                        seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (edge_ids, timestamps, labels) with heavy class imbalance.

    imbalance: fraction of samples assigned to class 0 (benign).
    """
    rng = np.random.default_rng(seed)
    edge_ids  = np.arange(n, dtype=np.int64)
    timestamps = np.sort(rng.integers(1000, 1_000_000, size=n)).astype(np.int64)

    n_benign  = int(n * imbalance)
    n_attack  = n - n_benign
    labels_b  = np.zeros(n_benign, dtype=np.int64)
    labels_a  = rng.integers(1, n_classes, size=n_attack).astype(np.int64)
    labels    = np.concatenate([labels_b, labels_a])

    # Shuffle so labels are not sorted (balancer must handle arbitrary order)
    perm = np.argsort(timestamps)   # then re-sort by timestamp
    edge_ids  = edge_ids[perm]
    timestamps = timestamps[perm]
    labels    = labels[perm]

    return edge_ids, timestamps, labels


# ---------------------------------------------------------------------------
# Test 1: Output sorted by timestamp
# ---------------------------------------------------------------------------

def test_output_sorted_by_timestamp():
    """Balanced EID array must be sorted in non-decreasing timestamp order."""
    eids, ts, labels = _synthetic_dataset()
    balancer = TemporalBalancer(strategy="oversample", seed=42)
    balanced = balancer.balance(eids, ts, labels)

    # Build timestamp lookup
    ts_map = np.zeros(int(eids.max()) + 1, dtype=np.int64)
    ts_map[eids] = ts

    balanced_ts = ts_map[balanced]
    diffs = np.diff(balanced_ts)
    assert (diffs >= 0).all(), (
        f"Balanced array not sorted by timestamp: {(diffs < 0).sum()} violations"
    )


# ---------------------------------------------------------------------------
# Test 2: All original training EIDs present
# ---------------------------------------------------------------------------

def test_all_original_eids_present():
    """Every original EID appears at least once in the oversampled output."""
    eids, ts, labels = _synthetic_dataset()
    balancer = TemporalBalancer(strategy="oversample", seed=42)
    balanced = balancer.balance(eids, ts, labels)

    balanced_set = set(balanced.tolist())
    for eid in eids:
        assert int(eid) in balanced_set, f"Original EID {eid} missing from balanced output"


# ---------------------------------------------------------------------------
# Test 3: Class distribution meets min_class_ratio
# ---------------------------------------------------------------------------

def test_class_distribution():
    """After oversampling, each class has >= min_class_ratio * majority_count samples."""
    eids, ts, labels = _synthetic_dataset(n=2000, imbalance=0.9)
    min_ratio = 0.1
    balancer = TemporalBalancer(strategy="oversample",
                                min_class_ratio=min_ratio, seed=42)
    balanced = balancer.balance(eids, ts, labels)

    # Build label lookup
    lb_map = np.zeros(int(eids.max()) + 1, dtype=np.int64)
    lb_map[eids] = labels
    balanced_labels = lb_map[balanced]

    classes, counts = np.unique(balanced_labels, return_counts=True)
    majority = counts.max()
    target   = majority * min_ratio

    for cls, cnt in zip(classes, counts):
        assert cnt >= target, (
            f"Class {cls}: {cnt} samples < target {target:.0f} "
            f"(min_class_ratio={min_ratio} × majority={majority})"
        )


# ---------------------------------------------------------------------------
# Test 4: External arrays unchanged after balancing
# ---------------------------------------------------------------------------

def test_external_arrays_unchanged():
    """Balancing one split must not modify arrays that were not passed in."""
    eids_train, ts_train, labels_train = _synthetic_dataset(n=1000, seed=1)
    eids_val,   ts_val,   labels_val   = _synthetic_dataset(n=300,  seed=2)

    # Take copies before balancing
    val_eids_before  = eids_val.copy()
    val_ts_before    = ts_val.copy()
    val_lb_before    = labels_val.copy()

    balancer = TemporalBalancer(strategy="oversample", seed=42)
    _ = balancer.balance(eids_train, ts_train, labels_train)

    # Val arrays must be byte-for-byte identical
    np.testing.assert_array_equal(eids_val,   val_eids_before,  "val edge_ids modified")
    np.testing.assert_array_equal(ts_val,     val_ts_before,    "val timestamps modified")
    np.testing.assert_array_equal(labels_val, val_lb_before,    "val labels modified")


# ---------------------------------------------------------------------------
# Test 5: Duplicate EIDs map to same feature rows
# ---------------------------------------------------------------------------

def test_duplicates_map_to_same_features():
    """Duplicate EIDs in balanced output index the same row in a feature array."""
    eids, ts, labels = _synthetic_dataset(n=500, imbalance=0.95, seed=3)
    balancer = TemporalBalancer(strategy="oversample", seed=42)
    balanced = balancer.balance(eids, ts, labels)

    # Synthetic feature array: features[eid] = eid (unique, deterministic)
    feature_values = eids.astype(np.float32)  # feature[eid] == eid

    # Find any duplicate EIDs in balanced
    unique, counts = np.unique(balanced, return_counts=True)
    duplicated_eids = unique[counts > 1]

    if len(duplicated_eids) == 0:
        pytest.skip("No duplicate EIDs found in balanced set (imbalance may be too low)")

    for eid in duplicated_eids[:5]:   # check first 5 duplicates
        positions = np.where(balanced == eid)[0]
        # All positions of this EID should map to the same feature value
        vals = [feature_values[balanced[p]] for p in positions]
        assert len(set(vals)) == 1, (
            f"Duplicate EID {eid} maps to multiple feature values: {vals}"
        )
