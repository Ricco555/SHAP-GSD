"""
Tests for ``build_singleton_feature_groups`` (paper config (i) coalition space).

Config (i) is raw-feature, ungrouped KernelSHAP: one coalition player per
encoded edge-feature dimension. It reuses ``FeatureGroupSHAP`` unchanged by
feeding it a synthetic ``feature_groups.json`` of d_e singleton groups, so the
correctness of that synthetic file is the whole correctness of the run.

Tests:
  1 — K == d_e == len(groups), and matches the source's d_e.
  2 — every raw column index is covered exactly once, in order.
  3 — the result round-trips through FeatureGrouping (reuses its own
      partition validation and the real json.dumps/loads path).
  4 — singleton types are inherited from the owning semantic group.
  5 — deterministic: two calls produce identical output, key order included.
  6 — the run-scoped UNSW r3_s2 grouping yields exactly K=212 (skipped when
      that artifact is absent, e.g. a fresh clone).
  7-10 — malformed sources are rejected rather than silently mis-grouped.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.feature_groups import (  # noqa: E402
    FeatureGrouping,
    build_singleton_feature_groups,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
R3S2_FG = REPO_ROOT / "runs/nf_unsw_nb15_v3/artifacts/feature_groups.json"


@pytest.fixture()
def source() -> dict:
    """A small semantic grouping: 6 columns in 3 groups of mixed type."""
    return {
        "d_e": 6,
        "K": 3,
        "feature_names": ["A", "B", "P0", "P1", "C_x", "C_y"],
        "groups": {
            "A":          {"indices": [0], "type": "numeric"},
            "B":          {"indices": [1], "type": "numeric"},
            "PORT":       {"indices": [2, 3], "type": "port_bin"},
            "C":          {"indices": [4, 5], "type": "categorical"},
        },
    }


@pytest.fixture()
def source_fixed(source: dict) -> dict:
    """`source` with K corrected to match its four groups."""
    source["K"] = 4
    return source


# ---------------------------------------------------------------------------
# Test 1: cardinality
# ---------------------------------------------------------------------------

def test_k_equals_d_e(source_fixed: dict) -> None:
    """K, d_e and the group count all agree, and d_e is carried through."""
    out = build_singleton_feature_groups(source_fixed)
    assert out["d_e"] == source_fixed["d_e"] == 6
    assert out["K"] == 6
    assert len(out["groups"]) == 6


# ---------------------------------------------------------------------------
# Test 2: exact, ordered coverage
# ---------------------------------------------------------------------------

def test_every_index_covered_exactly_once(source_fixed: dict) -> None:
    """Group i is named feature_names[i] and owns exactly column i."""
    out = build_singleton_feature_groups(source_fixed)
    names = list(out["groups"].keys())
    assert names == source_fixed["feature_names"]

    seen: list[int] = []
    for name, info in out["groups"].items():
        assert info["indices"] == [source_fixed["feature_names"].index(name)]
        seen.extend(info["indices"])
    assert sorted(seen) == list(range(out["d_e"]))
    assert len(seen) == len(set(seen))


# ---------------------------------------------------------------------------
# Test 3: round-trips through the production loader
# ---------------------------------------------------------------------------

def test_round_trips_through_feature_grouping(source_fixed: dict, tmp_path: Path) -> None:
    """The synthetic file loads via FeatureGrouping.from_json and validates."""
    out = build_singleton_feature_groups(source_fixed)
    path = tmp_path / "feature_groups_raw.json"
    path.write_text(json.dumps(out))

    fg = FeatureGrouping.from_json(path)   # runs _validate() internally
    assert fg.K == fg.d_e == 6
    assert fg.group_names == source_fixed["feature_names"]


# ---------------------------------------------------------------------------
# Test 4: type inheritance
# ---------------------------------------------------------------------------

def test_types_inherited_from_owning_group(source_fixed: dict) -> None:
    """Each singleton keeps the type of the semantic group it came from."""
    out = build_singleton_feature_groups(source_fixed)
    assert out["groups"]["A"]["type"] == "numeric"
    assert out["groups"]["P0"]["type"] == "port_bin"
    assert out["groups"]["P1"]["type"] == "port_bin"
    assert out["groups"]["C_y"]["type"] == "categorical"


# ---------------------------------------------------------------------------
# Test 5: determinism
# ---------------------------------------------------------------------------

def test_deterministic(source_fixed: dict) -> None:
    """Repeated calls are byte-identical, key order included."""
    a = json.dumps(build_singleton_feature_groups(source_fixed), sort_keys=False)
    b = json.dumps(build_singleton_feature_groups(source_fixed), sort_keys=False)
    assert a == b


# ---------------------------------------------------------------------------
# Test 6: the real UNSW r3_s2 grouping → exactly 212 players
# ---------------------------------------------------------------------------

def test_r3s2_yields_212_players() -> None:
    """The active paper run's grouping produces the config-(i) K=212 space.

    Guards the spec's explicit warning that the stale top-level
    artifacts/feature_groups.json (d_e=218) must not be used as the source.
    """
    if not R3S2_FG.exists():
        pytest.skip(f"{R3S2_FG} not present — run-scoped artifact, not in git")
    with open(R3S2_FG) as f:
        src = json.load(f)
    assert src["d_e"] == 212, f"expected d_e=212, got {src['d_e']}"
    out = build_singleton_feature_groups(src)
    assert out["K"] == out["d_e"] == 212
    assert len(out["groups"]) == 212
    covered = sorted(i for info in out["groups"].values() for i in info["indices"])
    assert covered == list(range(212))


# ---------------------------------------------------------------------------
# Tests 7-10: malformed sources are rejected
# ---------------------------------------------------------------------------

def test_rejects_k_group_count_mismatch(source: dict) -> None:
    """A source whose K disagrees with its group count is rejected."""
    with pytest.raises(ValueError, match="disagrees"):
        build_singleton_feature_groups(source)   # K=3, 4 groups


def test_rejects_incomplete_coverage(source_fixed: dict) -> None:
    """A source that does not cover every column is rejected."""
    source_fixed["groups"]["C"]["indices"] = [4]
    with pytest.raises(ValueError, match="do not cover"):
        build_singleton_feature_groups(source_fixed)


def test_rejects_duplicate_feature_names(source_fixed: dict) -> None:
    """Non-unique feature names would silently collapse groups — rejected."""
    source_fixed["feature_names"][5] = "A"
    with pytest.raises(ValueError, match="not unique"):
        build_singleton_feature_groups(source_fixed)


def test_rejects_missing_keys() -> None:
    """A source missing a required top-level key is rejected."""
    with pytest.raises(ValueError, match="missing keys"):
        build_singleton_feature_groups({"d_e": 3, "K": 1})
