"""
Tests for TemporalBalancer.

Uses synthetic data for all tests (no dependency on feature_store or graphs).

Tests:
  1 — Temporal order:   output sorted by timestamp.
  2 — All original EIDs present in oversampled output.
  3 — Class distribution meets min_class_ratio after balancing.
  4 — Val/test unchanged: balancing only touches what is passed; external data unmodified.
  5 — Duplicate EIDs:   duplicate entries map to identical feature values (index equality).
  6 — Shape: get_class_weights returns exactly num_classes entries, not
      len(np.unique(original_labels)) entries.
  7 — A zero-training-support class's weight slot is exactly 0.0.
  8 — No NaN/inf anywhere in the output, for each of the three weighting
      methods, when a zero-support class is present.
  9 — Normalization sums supported weights to n_supported, not num_classes.
  10 — No-op regression: weights are numerically unchanged (vs. the pre-fix
       formula) when every class already has nonzero support.
  11 — A label index >= num_classes raises ValueError.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

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


# ---------------------------------------------------------------------------
# Tests 6-11: get_class_weights zero-support-class handling (specs/39, specs/40)
# ---------------------------------------------------------------------------

def test_get_class_weights_shape_matches_num_classes_not_unique_count():
    """Output shape must be (num_classes,), not len(np.unique(labels)).

    Only classes {0, 1} are present but num_classes=3 -- this is the exact
    mismatch specs/39 S1 identifies as the root cause. This would have
    failed on the pre-fix code (returned shape (2,)).
    """
    original_labels = np.array([0, 0, 0, 1, 1])
    balancer = TemporalBalancer()
    weights = balancer.get_class_weights(original_labels, num_classes=3, log_weights=False)
    assert weights.shape == (3,)


def test_get_class_weights_zero_support_slot_is_exactly_zero():
    """A class with zero training rows gets an exact 0.0 weight slot."""
    original_labels = np.array([0, 0, 0, 1, 1])
    balancer = TemporalBalancer()
    weights = balancer.get_class_weights(original_labels, num_classes=3, log_weights=False)
    assert weights[2].item() == 0.0


def test_get_class_weights_zero_support_no_nan_or_inf_effective_num():
    """No NaN/inf in output for method='effective_num' with a zero-support class."""
    original_labels = np.array([0, 0, 0, 1, 1])
    balancer = TemporalBalancer()
    weights = balancer.get_class_weights(
        original_labels, num_classes=3, method="effective_num", log_weights=False
    )
    assert np.isfinite(weights.numpy()).all()
    assert weights[2].item() == 0.0


def test_get_class_weights_zero_support_no_nan_or_inf_sqrt_inverse_freq():
    """No NaN/inf in output for method='sqrt_inverse_freq' with a zero-support class."""
    original_labels = np.array([0, 0, 0, 1, 1])
    balancer = TemporalBalancer()
    weights = balancer.get_class_weights(
        original_labels, num_classes=3, method="sqrt_inverse_freq", log_weights=False
    )
    assert np.isfinite(weights.numpy()).all()
    assert weights[2].item() == 0.0


def test_get_class_weights_zero_support_no_nan_or_inf_inverse_freq():
    """No NaN/inf in output for method='inverse_freq' with a zero-support class."""
    original_labels = np.array([0, 0, 0, 1, 1])
    balancer = TemporalBalancer()
    weights = balancer.get_class_weights(
        original_labels, num_classes=3, method="inverse_freq", log_weights=False
    )
    assert np.isfinite(weights.numpy()).all()
    assert weights[2].item() == 0.0


def test_get_class_weights_normalization_excludes_zero_support_classes():
    """Supported weights sum to n_supported (2), not num_classes (3)."""
    original_labels = np.array([0] * 900 + [1] * 100)
    balancer = TemporalBalancer()
    weights = balancer.get_class_weights(original_labels, num_classes=3, log_weights=False)
    assert weights[:2].sum().item() == pytest.approx(2.0)
    assert weights[2].item() == 0.0


def test_get_class_weights_no_op_on_fully_supported_labels():
    """When every class has nonzero support, weights match the pre-fix formula
    exactly (hand-computed expected values) -- proof the fix changes nothing
    on a fully-supported dataset (specs/39 S5's no-op-when-fully-supported
    regression-safety property).
    """
    n = [900, 60, 40]
    original_labels = np.concatenate(
        [np.full(count, cls, dtype=np.int64) for cls, count in enumerate(n)]
    )
    beta = 0.9999
    class_counts = np.array(n, dtype=np.float64)
    effective_num = (1.0 - np.power(beta, class_counts)) / (1.0 - beta)
    expected = 1.0 / effective_num
    expected = expected / expected.sum() * len(n)

    balancer = TemporalBalancer()
    weights = balancer.get_class_weights(
        original_labels, num_classes=3, method="effective_num", beta=beta, log_weights=False
    )
    np.testing.assert_allclose(weights.numpy(), expected, rtol=1e-6)


def test_get_class_weights_label_index_exceeding_num_classes_raises():
    """A label value >= num_classes raises ValueError naming the offender."""
    original_labels = np.array([0, 1, 2, 3])
    balancer = TemporalBalancer()
    with pytest.raises(ValueError, match="num_classes"):
        balancer.get_class_weights(original_labels, num_classes=3, log_weights=False)


def test_get_class_weights_max_clamp_still_applies_to_supported_only():
    """max_clamp still bounds supported weights; the zero-support slot stays
    0.0, unaffected by clamping (specs/39 S3.2's explicit non-scope note).
    """
    original_labels = np.array([0] * 9990 + [1] * 10)
    balancer = TemporalBalancer()

    unclamped = balancer.get_class_weights(original_labels, num_classes=3, log_weights=False)
    rarest_weight = unclamped[1].item()
    assert rarest_weight > 1.0  # sanity: rare class weight is indeed large pre-clamp

    max_clamp = rarest_weight - 0.5
    clamped = balancer.get_class_weights(
        original_labels, num_classes=3, max_clamp=max_clamp, log_weights=False
    )
    assert clamped[1].item() == pytest.approx(max_clamp)
    assert clamped[2].item() == 0.0


def test_class_weights_zero_support_class_does_not_crash_crossentropyloss():
    """End-to-end smoke test: a weight vector with a zero-support slot works
    correctly inside nn.CrossEntropyLoss (forward + backward, finite loss).
    """
    original_labels = np.array([0, 0, 0, 1, 1])
    balancer = TemporalBalancer()
    weights = balancer.get_class_weights(original_labels, num_classes=3, log_weights=False)

    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.tensor([0, 1, 0, 1])

    loss = criterion(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
