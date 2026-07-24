"""
Unit tests for the edges_meta.parquet artifact (Phase-2 feature-store reuse).

These tests do not require a completed pipeline run — they exercise
``write_edges_meta`` / ``read_edges_meta`` in isolation on synthetic data, plus
the contiguous-EID concatenation invariant that lets Phase 2 reconstruct the
global temporal order from per-split arrays without re-sorting the CSV.

Covered:
1. Round-trip fidelity — values and dtypes survive write→read, including
   fractional float32 byte values (guards against the int64-truncation regression).
2. Alignment assertion — mismatched-length meta and an ``n_expected`` mismatch
   both raise ``AssertionError`` at write time.
3. Concatenation invariant — concatenating three contiguous-range per-split
   ``edge_indices`` reproduces ``arange(N)`` (mirrors the Phase-2 tripwire).
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow", reason="pyarrow required for edges_meta.parquet")

from src.data.feature_store import read_edges_meta, write_edges_meta


def _make_meta(n: int) -> dict[str, np.ndarray]:
    """Build a synthetic edges_meta dict of length ``n`` with the expected dtypes."""
    rng = np.random.default_rng(7)
    return {
        "src_ip":    np.array([f"10.0.0.{i % 256}" for i in range(n)], dtype=object),
        "dst_ip":    np.array([f"192.168.1.{i % 256}" for i in range(n)], dtype=object),
        # Deliberately fractional to catch an int64-truncation regression.
        "in_bytes":  (rng.random(n).astype(np.float32) * 1000.0 + 0.5).astype(np.float32),
        "out_bytes": (rng.random(n).astype(np.float32) * 500.0 + 0.25).astype(np.float32),
        "dst_port":  rng.integers(0, 65536, size=n).astype(np.int32),
    }


def test_round_trip_values_and_dtypes(tmp_path: Path) -> None:
    """write_edges_meta → read_edges_meta preserves values and dtypes exactly."""
    n = 128
    meta = _make_meta(n)
    write_edges_meta(tmp_path, meta, n_expected=n)

    out = read_edges_meta(tmp_path)

    # Dtypes match Phase-2 call-site expectations.
    assert out["src_ip"].dtype == object
    assert out["dst_ip"].dtype == object
    assert out["in_bytes"].dtype == np.float32
    assert out["out_bytes"].dtype == np.float32
    assert out["dst_port"].dtype == np.int32

    # Values are byte-identical (str objects, exact float32, exact int32).
    assert np.array_equal(out["src_ip"].astype(str), meta["src_ip"].astype(str))
    assert np.array_equal(out["dst_ip"].astype(str), meta["dst_ip"].astype(str))
    assert np.array_equal(out["in_bytes"], meta["in_bytes"])
    assert np.array_equal(out["out_bytes"], meta["out_bytes"])
    assert np.array_equal(out["dst_port"], meta["dst_port"])


def test_round_trip_preserves_fractional_bytes(tmp_path: Path) -> None:
    """Fractional byte values survive — int64 storage would have truncated them."""
    meta = {
        "src_ip":    np.array(["1.1.1.1", "2.2.2.2"], dtype=object),
        "dst_ip":    np.array(["3.3.3.3", "4.4.4.4"], dtype=object),
        "in_bytes":  np.array([12.5, 0.75], dtype=np.float32),
        "out_bytes": np.array([3.25, 100.125], dtype=np.float32),
        "dst_port":  np.array([80, 443], dtype=np.int32),
    }
    write_edges_meta(tmp_path, meta, n_expected=2)
    out = read_edges_meta(tmp_path)

    assert np.array_equal(out["in_bytes"], np.array([12.5, 0.75], dtype=np.float32))
    assert np.array_equal(out["out_bytes"], np.array([3.25, 100.125], dtype=np.float32))


def test_alignment_assert_mismatched_lengths(tmp_path: Path) -> None:
    """Mutually unequal-length meta arrays raise AssertionError at write time."""
    meta = _make_meta(10)
    meta["dst_port"] = meta["dst_port"][:-1]  # length 9 ≠ 10
    with pytest.raises(AssertionError, match="length mismatch"):
        write_edges_meta(tmp_path, meta, n_expected=10)


def test_alignment_assert_n_expected_mismatch(tmp_path: Path) -> None:
    """A length that disagrees with n_expected raises AssertionError."""
    meta = _make_meta(10)
    with pytest.raises(AssertionError, match="!= edge count"):
        write_edges_meta(tmp_path, meta, n_expected=11)


def test_concatenation_invariant() -> None:
    """Concatenating contiguous per-split EID ranges reproduces arange(N)."""
    n_train, n_val, n_test = 100, 40, 25
    n_total = n_train + n_val + n_test
    train_eids = np.arange(0, n_train, dtype=np.int64)
    val_eids = np.arange(n_train, n_train + n_val, dtype=np.int64)
    test_eids = np.arange(n_train + n_val, n_total, dtype=np.int64)

    concat = np.concatenate([train_eids, val_eids, test_eids])
    assert np.array_equal(concat, np.arange(n_total, dtype=np.int64))
