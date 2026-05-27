"""
Tests for FeatureGrouping (semantic SHAP coalition groups).

Loads the actual feature_groups.json written by Phase 1.  The path is
resolved from the config selected by the ``SHAP_GSD_CONFIG`` environment
variable (default: configs/experiment_unsw.yaml) via ``output.feature_groups_path``,
so the tests respect ``run.dir`` automatically for multi-dataset runs.

Tests:
  1 — Full coverage: union of all groups == {0, ..., d_e−1} (no gaps).
  2 — No overlaps:   no column index appears in more than one group.
  3 — DST_PORT_GROUP: contains exactly N_DST_PORT_BINS=16 columns.
  4 — SRC_PORT_IS_EPHEMERAL: contains exactly 1 column.
  5 — get_group_mask: coalition vector → correct d_e-dim boolean mask.
"""

from pathlib import Path

import numpy as np
import pytest

from tests._paths import REPO_ROOT, resolve_cfg
from src.data.feature_groups import FeatureGrouping
from src.data.preprocessor import N_DST_PORT_BINS


def _fg_path() -> Path:
    """Return the feature_groups.json path from the active config."""
    cfg = resolve_cfg()
    return REPO_ROOT / cfg["output"]["feature_groups_path"]


FG_PATH: Path = _fg_path()


@pytest.fixture(scope="module")
def fg() -> FeatureGrouping:
    if not FG_PATH.exists():
        pytest.skip("feature_groups.json not found — run Phase 1 first")
    return FeatureGrouping.from_json(FG_PATH)


# ---------------------------------------------------------------------------
# Test 1: Full coverage
# ---------------------------------------------------------------------------

def test_full_coverage(fg):
    """Union of all group indices == {0, ..., d_e−1}."""
    seen: set[int] = set()
    for name, info in fg.groups.items():
        seen.update(info["indices"])
    expected = set(range(fg.d_e))
    missing = expected - seen
    assert not missing, f"Groups missing columns: {sorted(missing)}"
    extra = seen - expected
    assert not extra, f"Groups reference out-of-range columns: {sorted(extra)}"


# ---------------------------------------------------------------------------
# Test 2: No overlaps
# ---------------------------------------------------------------------------

def test_no_overlaps(fg):
    """No column appears in more than one group."""
    seen: set[int] = set()
    for name, info in fg.groups.items():
        for idx in info["indices"]:
            assert idx not in seen, (
                f"Column {idx} appears in multiple groups (duplicate in '{name}')"
            )
            seen.add(idx)


# ---------------------------------------------------------------------------
# Test 3: DST_PORT_GROUP has N_DST_PORT_BINS columns
# ---------------------------------------------------------------------------

def test_dst_port_group_size(fg):
    """DST_PORT_GROUP must have exactly N_DST_PORT_BINS (16) columns."""
    assert "DST_PORT_GROUP" in fg.groups, "DST_PORT_GROUP group not found"
    n = len(fg.groups["DST_PORT_GROUP"]["indices"])
    assert n == N_DST_PORT_BINS, (
        f"DST_PORT_GROUP has {n} columns, expected {N_DST_PORT_BINS}"
    )


# ---------------------------------------------------------------------------
# Test 4: SRC_PORT_IS_EPHEMERAL has 1 column
# ---------------------------------------------------------------------------

def test_src_port_group_size(fg):
    """SRC_PORT_IS_EPHEMERAL must have exactly 1 column."""
    assert "SRC_PORT_IS_EPHEMERAL" in fg.groups, "SRC_PORT_IS_EPHEMERAL group not found"
    n = len(fg.groups["SRC_PORT_IS_EPHEMERAL"]["indices"])
    assert n == 1, f"SRC_PORT_IS_EPHEMERAL has {n} columns, expected 1"


# ---------------------------------------------------------------------------
# Test 5: get_group_mask produces correct d_e-dim mask
# ---------------------------------------------------------------------------

def test_get_group_mask(fg):
    """Coalition vector → correct boolean mask over d_e columns."""
    K = fg.K
    d_e = fg.d_e

    # All-zero coalition: no columns pass through
    z_none = np.zeros(K, dtype=int)
    mask_none = fg.get_group_mask(z_none)
    assert mask_none.shape == (d_e,)
    assert not mask_none.any(), "All-zero coalition should produce all-False mask"

    # All-one coalition: all columns pass through
    z_all = np.ones(K, dtype=int)
    mask_all = fg.get_group_mask(z_all)
    assert mask_all.all(), "All-one coalition should produce all-True mask"

    # Single-group coalition: only that group's indices are True
    for j, name in enumerate(fg.group_names[:3]):   # test first 3 groups
        z_single = np.zeros(K, dtype=int)
        z_single[j] = 1
        mask = fg.get_group_mask(z_single)
        expected_true = set(fg.groups[name]["indices"])
        actual_true   = set(np.where(mask)[0].tolist())
        assert actual_true == expected_true, (
            f"Group '{name}': mask indices {actual_true} != expected {expected_true}"
        )
