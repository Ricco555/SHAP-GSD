"""
Tests for explore/class_coverage_analysis.py's compute_class_coverage.

Uses synthetic tmp_path-written CSVs for all tests except the last (no
dependency on the real multi-GB datasets, except the opt-in regression
anchor).

Tests:
  1 — A tie at the cut timestamp lands entirely in the earlier split (never
      split across the train/val or val/test boundary).
  2 — A class with zero rows in one split is reported as an explicit 0, not
      an omitted key.
  3 — Exact duplicate rows are counted by the dedup delta, matching an
      independently-computed drop_duplicates() count.
  4 — A row that is both an exact duplicate AND has a NaN port counts
      against the dedup delta, never the dropna delta (dedup runs first).
  5 — A clean row with a NaN FLOW_START_MILLISECONDS is reconciled via
      n_unassigned, and is not double-counted in n_train/n_val/n_test.
  6 — ordered_classes matches label_map's integer-code order, Benign first.
  7 — Real-data regression anchor against all four actual datasets — gated,
      opt-in via SHAP_GSD_RUN_REAL_DATA_TESTS=1, skipped by default.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from explore.class_coverage_analysis import (
    DATASETS,
    compute_class_coverage,
    resolve_local_csv_path,
)
from src.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent

# All 7 columns compute_class_coverage's underlying read touches:
# IPV4_SRC_ADDR, IPV4_DST_ADDR, L4_SRC_PORT, L4_DST_PORT,
# FLOW_START_MILLISECONDS, Label, Attack (last two positionally required by
# LABEL_COLS).
_COLUMNS = [
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT",
    "FLOW_START_MILLISECONDS", "Label", "Attack",
]


def _write_csv(tmp_path: Path, rows: list[dict], name: str = "synthetic.csv") -> Path:
    df = pd.DataFrame(rows, columns=_COLUMNS)
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def _row(src="10.0.0.1", dst="10.0.0.2", sport=1000, dport=80, ts=1000,
         label=0, attack="Benign") -> dict:
    return {
        "IPV4_SRC_ADDR": src, "IPV4_DST_ADDR": dst,
        "L4_SRC_PORT": sport, "L4_DST_PORT": dport,
        "FLOW_START_MILLISECONDS": ts, "Label": label, "Attack": attack,
    }


# ---------------------------------------------------------------------------
# 1 — tie at cut timestamp lands in earlier split
# ---------------------------------------------------------------------------

def test_tie_at_cut_timestamp_lands_in_earlier_split(tmp_path):
    # 10 rows, timestamps 0..9 ms, all distinct src ports to avoid accidental
    # dedup. train_frac=0.5 -> train_cut = int(10*0.5) = 5 -> tau_train_ms =
    # ts.iloc[5] = 5. Rows 6,7 share the SAME timestamp as tau_train_ms (5)
    # so a tie exists at the cut: with 3 rows sharing ts=5 (indices 5,6,7),
    # ALL must land in train (train_mask uses <=).
    rows = []
    ts_values = [0, 1, 2, 3, 4, 5, 5, 5, 8, 9]
    for i, ts in enumerate(ts_values):
        rows.append(_row(sport=1000 + i, ts=ts, attack="Benign"))
    path = _write_csv(tmp_path, rows)

    result = compute_class_coverage(path, train_frac=0.5, val_frac=0.3)

    assert result.tau_train_ms == 5
    # All three rows with ts == 5 (indices 5, 6, 7) must be in train, since
    # train_mask is ts <= tau_train_ms and none of them can appear in val.
    assert result.n_train == 8  # indices 0..7 (ts 0,1,2,3,4,5,5,5)
    assert result.n_train + result.n_val + result.n_test == result.n_clean


# ---------------------------------------------------------------------------
# 2 — zero-row class reported as zero, not omitted
# ---------------------------------------------------------------------------

def test_zero_row_class_reported_as_zero_not_omitted(tmp_path):
    rows = []
    # Benign spread across the whole span.
    for i in range(20):
        rows.append(_row(sport=2000 + i, ts=i * 100, attack="Benign"))
    # "RareAttack" only appears early (all in train, none in val/test).
    for i in range(3):
        rows.append(_row(sport=3000 + i, ts=i, attack="RareAttack"))
    path = _write_csv(tmp_path, rows)

    result = compute_class_coverage(path, train_frac=0.3, val_frac=0.3)

    assert "RareAttack" in result.counts
    # RareAttack's timestamps (0,1,2) are all before the dense Benign span's
    # split boundary: all 3 rows land in train, and val/test must be explicit
    # 0s (present as keys, not omitted) rather than merely non-negative.
    assert set(result.counts["RareAttack"].keys()) == {"train", "val", "test"}
    assert result.counts["RareAttack"]["train"] == 3
    assert result.counts["RareAttack"]["val"] == 0
    assert result.counts["RareAttack"]["test"] == 0
    assert isinstance(result.counts["RareAttack"]["test"], int)


# ---------------------------------------------------------------------------
# 3 — exact duplicate rows counted by dedup delta
# ---------------------------------------------------------------------------

def test_exact_duplicate_rows_counted_by_dedup_delta(tmp_path):
    rows = []
    base = _row(sport=4000, ts=10, attack="Benign")
    rows.append(base)
    rows.append(dict(base))  # exact duplicate #1
    rows.append(dict(base))  # exact duplicate #2
    other = _row(sport=4001, ts=20, attack="Benign")
    rows.append(other)
    rows.append(dict(other))  # exact duplicate of "other"
    # Two unique rows elsewhere.
    rows.append(_row(sport=4002, ts=30, attack="Benign"))
    rows.append(_row(sport=4003, ts=40, attack="Benign"))
    path = _write_csv(tmp_path, rows)

    # Independently compute the expected dedup delta.
    raw_df = pd.read_csv(path, low_memory=False)
    expected_dedup_dropped = raw_df.duplicated().sum()
    assert expected_dedup_dropped == 3  # 2 dupes of base + 1 dupe of other

    result = compute_class_coverage(path, train_frac=0.6, val_frac=0.2)
    assert result.n_dedup_dropped == expected_dedup_dropped
    assert result.n_dropna_dropped == 0


# ---------------------------------------------------------------------------
# 4 — NaN port counted as dropna, not dedup (ordering pin)
# ---------------------------------------------------------------------------

def test_nan_port_counted_as_dropna_not_dedup(tmp_path):
    # Two byte-identical rows, BOTH carrying a NaN L4_SRC_PORT (this is the
    # only way a duplicate pair can be "byte-identical... also carrying a
    # NaN port" — if only one of the pair had NaN, they would differ in that
    # column and drop_duplicates() would not consider them duplicates at
    # all, since pandas' duplicate comparison requires equal values, and a
    # NaN only compares equal to another NaN, not to a real port number).
    #
    # Under the real dedup-then-dropna order: drop_duplicates() removes the
    # SECOND occurrence first (1 row, counted against n_dedup_dropped), then
    # dropna(subset=ports) removes the single SURVIVING occurrence, which
    # still has a NaN port (1 row, counted against n_dropna_dropped). This
    # is the direct regression pin for specs/47 S5.2's ordering requirement:
    # under the WRONG (reversed) order, dropna would run first and remove
    # BOTH rows in one pass (since both have a NaN port) before dedup ever
    # saw them, giving dedup=0/dropna=2 instead of the correct dedup=1/
    # dropna=1 — a silent reassignment of which delta "owns" the row that
    # was both a duplicate and NaN-port. This test pins the correct split.
    nan_port_row = _row(sport=np.nan, dport=443, ts=50, attack="Benign")
    nan_port_dup = dict(nan_port_row)  # exact duplicate, including the NaN

    rows = [nan_port_row, nan_port_dup]
    # Padding rows so the split logic has enough rows to work with.
    for i in range(5):
        rows.append(_row(sport=6000 + i, ts=100 + i, attack="Benign"))
    path = _write_csv(tmp_path, rows)

    raw_df = pd.read_csv(path, low_memory=False)
    n_before_dedup = len(raw_df)
    after_dedup = raw_df.drop_duplicates()
    dedup_dropped = n_before_dedup - len(after_dedup)
    assert dedup_dropped == 1  # one of the two identical NaN-port rows

    after_dropna = after_dedup.dropna(subset=["IPV4_SRC_ADDR", "IPV4_DST_ADDR",
                                               "L4_SRC_PORT", "L4_DST_PORT"])
    dropna_dropped = len(after_dedup) - len(after_dropna)
    assert dropna_dropped == 1  # the sole survivor still has a NaN port

    # The reversed-order counterfactual (dropna first) would have removed
    # both rows in the dropna pass, since both carry a NaN port — proving
    # the split is order-sensitive, not an equivalent-either-way detail.
    reversed_after_dropna = raw_df.dropna(subset=["IPV4_SRC_ADDR", "IPV4_DST_ADDR",
                                                   "L4_SRC_PORT", "L4_DST_PORT"])
    reversed_dropna_dropped = n_before_dedup - len(reversed_after_dropna)
    assert reversed_dropna_dropped == 2
    assert reversed_dropna_dropped != dropna_dropped

    result = compute_class_coverage(path, train_frac=0.6, val_frac=0.2)
    assert result.n_dedup_dropped == 1
    assert result.n_dropna_dropped == 1


# ---------------------------------------------------------------------------
# 5 — n_unassigned reconciliation on NaN timestamp
# ---------------------------------------------------------------------------

def test_unassigned_reconciliation_nonzero_when_timestamp_is_nan(tmp_path):
    rows = []
    for i in range(10):
        rows.append(_row(sport=7000 + i, ts=i * 10, attack="Benign"))
    # One clean (non-duplicate, non-null-port) row with a NaN timestamp.
    nan_ts_row = _row(sport=7100, ts=np.nan, attack="Benign")
    rows.append(nan_ts_row)
    path = _write_csv(tmp_path, rows)

    result = compute_class_coverage(path, train_frac=0.6, val_frac=0.2)

    assert result.n_unassigned >= 1
    assert result.n_train + result.n_val + result.n_test == result.n_clean - result.n_unassigned
    # sort_values places NaN last, so a NaN FLOW_START_MILLISECONDS row could
    # become elapsed_hours[-1] and poison total_span_hours (-> the figure's
    # np.linspace(0.0, nan, ...) bin edges); this must stay finite regardless.
    assert np.isfinite(result.total_span_hours)


# ---------------------------------------------------------------------------
# 6 — ordered_classes matches label_map int-code order
# ---------------------------------------------------------------------------

def test_ordered_classes_matches_label_map_int_code_order(tmp_path):
    rows = []
    for i in range(5):
        rows.append(_row(sport=8000 + i, ts=i, attack="Benign"))
    rows.append(_row(sport=8100, ts=100, attack="Zebra"))
    rows.append(_row(sport=8101, ts=101, attack="Apple"))
    path = _write_csv(tmp_path, rows)

    result = compute_class_coverage(path, train_frac=0.6, val_frac=0.2)

    assert result.ordered_classes[0] == "Benign"
    assert [result.label_map[c] for c in result.ordered_classes] == list(
        range(len(result.label_map))
    )
    # Alphabetical among non-Benign classes: Apple before Zebra.
    assert result.ordered_classes[1:] == ["Apple", "Zebra"]


# ---------------------------------------------------------------------------
# 7 — real-data regression anchor (gated, opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("SHAP_GSD_RUN_REAL_DATA_TESTS") != "1",
    reason="Reads full multi-GB NetFlow CSVs (up to 5GB); opt-in via "
           "SHAP_GSD_RUN_REAL_DATA_TESTS=1, not part of the default "
           "`pytest tests/ -v` gate.",
)
def test_real_data_regression_anchor():
    results = {}
    for key, cfg_rel in DATASETS.items():
        cfg = load_config(REPO_ROOT / cfg_rel)
        csv_path = resolve_local_csv_path(cfg)
        if not csv_path.exists():
            pytest.skip(f"{csv_path} not found on this checkout")
        results[key] = compute_class_coverage(
            csv_path=csv_path,
            train_frac=float(cfg["data"]["train_frac"]),
            val_frac=float(cfg["data"]["val_frac"]),
        )

    # BoT-IoT: Theft = 0/0/1615 train/val/test.
    bot = results["bot_iot"]
    assert bot.counts["Theft"]["train"] == 0
    assert bot.counts["Theft"]["val"] == 0
    assert bot.counts["Theft"]["test"] == 1615

    # ToN-IoT: five named zero-train classes.
    ton = results["ton_iot"]
    for cls in ("backdoor", "mitm", "password", "ransomware", "xss"):
        assert ton.counts[cls]["train"] == 0, cls
    assert ton.n_dedup_dropped == 1_816_137

    # CICIDS2018: two named zero-train classes.
    cic = results["cicids2018"]
    for cls in ("Bot", "Infilteration"):
        assert cic.counts[cls]["train"] == 0, cls
    assert cic.n_dedup_dropped == 628_474

    # UNSW: zero zero-train classes.
    unsw = results["unsw"]
    zero_train = [cls for cls in unsw.ordered_classes if unsw.counts[cls]["train"] == 0]
    assert zero_train == []
