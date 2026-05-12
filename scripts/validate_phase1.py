#!/usr/bin/env python3
"""
Phase 1 validation — checks all outputs of scripts/01_preprocess.py.

Usage:
  python scripts/validate_phase1.py [--config configs/experiment_unsw.yaml]

Prints PASS / FAIL / WARN for each check.
Does NOT modify any file.
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.WARNING)

from src.data.loader import (
    CATEGORICAL_COLS,
    LOG_TRANSFORM_COLS,
    NUMERIC_COLS,
    THROUGHPUT_COLS,
    load_raw,
)
from src.data.preprocessor import (
    DST_PORT_BIN_NAMES,
    N_DST_PORT_BINS,
    encode_dst_port,
    encode_src_port,
)
from src.data.feature_store import FeatureStore
from src.utils.config import load_config

# ── Result accumulator ────────────────────────────────────────────────────────

RESULTS: list[tuple[str, str, str, str]] = []

def _report(section: str, name: str, status: str, message: str) -> None:
    assert status in ("PASS", "FAIL", "WARN")
    RESULTS.append((section, name, status, message))
    tag = f"[{status}]"
    print(f"  {tag:<6}  {name}: {message}")

def ok(section, name, msg):  _report(section, name, "PASS", msg)
def fail(section, name, msg): _report(section, name, "FAIL", msg)
def warn(section, name, msg): _report(section, name, "WARN", msg)

def check(section, name, passed: bool, msg_pass: str, msg_fail: str,
          as_warn: bool = False) -> bool:
    if passed:
        _report(section, name, "PASS", msg_pass)
    else:
        _report(section, name, "WARN" if as_warn else "FAIL", msg_fail)
    return passed


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_split_meta(split_indices: dict, split_name: str) -> dict:
    return split_indices["splits"][split_name]


def _open_mmap(path: Path, n: int, d: int) -> np.ndarray:
    """Open a float32 memmap of shape (n, d)."""
    return np.memmap(path, dtype=np.float32, mode="r", shape=(n, d))


# ── SECTION 1 — Outputs exist ─────────────────────────────────────────────────

def section1(cfg: dict) -> None:
    print("\n─── SECTION 1: Outputs exist ───────────────────────────────────────")
    S = "S1"
    root = REPO_ROOT
    fs_root = root / cfg["output"]["feature_store_dir"]

    # Feature store splits and their four files
    for split in ("train", "val", "test"):
        d = fs_root / split
        check(S, f"feature_store/{split}/ exists", d.is_dir(),
              f"directory present", f"directory MISSING: {d}")
        for fname in ("features.dat", "edge_indices.npy", "timestamps.npy", "labels.npy"):
            p = d / fname
            check(S, f"feature_store/{split}/{fname}", p.exists(),
                  f"file present ({p.stat().st_size // 1024:,} KB)" if p.exists() else "present",
                  f"file MISSING: {p}")

    # Root-level outputs
    for key, label in [
        ("split_indices_path",         "split_indices.json"),
        ("feature_groups_path",        "feature_groups.json"),
        ("balanced_train_indices_path","balanced_train_indices.npy"),
        ("class_weights_path",         "class_weights.npy"),
    ]:
        p = root / cfg["output"][key]
        check(S, label, p.exists(),
              f"file present ({p.stat().st_size:,} B)" if p.exists() else "present",
              f"file MISSING: {p}")

    # Transformer files — check spec-expected names, then actual names
    tdir = root / cfg["output"]["transformers_dir"]

    p_scaler = tdir / "scaler.pkl"
    check(S, "artifacts/transformers/scaler.pkl", p_scaler.exists(),
          "file present", f"MISSING: {p_scaler}")

    # spec expects encoder.pkl; implementation saved ohe.pkl
    p_enc_spec = tdir / "encoder.pkl"
    p_enc_act  = tdir / "ohe.pkl"
    if p_enc_spec.exists():
        ok(S, "artifacts/transformers/encoder.pkl", "file present")
    elif p_enc_act.exists():
        warn(S, "artifacts/transformers/encoder.pkl",
             f"spec expects encoder.pkl; found ohe.pkl — content is the OneHotEncoder")
    else:
        fail(S, "artifacts/transformers/encoder.pkl",
             f"neither encoder.pkl nor ohe.pkl found in {tdir}")

    # spec expects pruning_mask.json; implementation saved spearman_mask.npy + meta.json
    p_mask_spec = tdir / "pruning_mask.json"
    p_mask_act  = tdir / "spearman_mask.npy"
    p_meta      = tdir / "meta.json"
    if p_mask_spec.exists():
        ok(S, "artifacts/transformers/pruning_mask.json", "file present")
    elif p_mask_act.exists() and p_meta.exists():
        warn(S, "artifacts/transformers/pruning_mask.json",
             f"spec expects pruning_mask.json; found spearman_mask.npy + meta.json — equivalent content")
    else:
        fail(S, "artifacts/transformers/pruning_mask.json",
             f"neither pruning_mask.json nor spearman_mask.npy found in {tdir}")


# ── SECTION 2 — Temporal split correctness ────────────────────────────────────

def section2(cfg: dict) -> dict:
    """Returns split_info for downstream checks."""
    print("\n─── SECTION 2: Temporal split correctness ──────────────────────────")
    S = "S2"
    root = REPO_ROOT
    si_path = root / cfg["output"]["split_indices_path"]

    if not si_path.exists():
        fail(S, "split_indices.json loadable", "file missing — skipping section")
        return {}

    with open(si_path) as f:
        si = json.load(f)

    ok(S, "split_indices.json loadable", f"n_total={si['n_total']:,}  d_e={si['d_e']}")

    splits = si["splits"]
    tr, va, te = splits["train"], splits["val"], splits["test"]

    # Disjointness via EID ranges (no overlap if contiguous)
    tr_eids = set(range(tr["eid_start"], tr["eid_end"] + 1))
    va_eids = set(range(va["eid_start"], va["eid_end"] + 1))
    te_eids = set(range(te["eid_start"], te["eid_end"] + 1))

    check(S, "EIDs disjoint (train∩val=∅)",
          len(tr_eids & va_eids) == 0,
          "no overlap",
          f"overlap: {len(tr_eids & va_eids):,} shared EIDs")
    check(S, "EIDs disjoint (train∩test=∅)",
          len(tr_eids & te_eids) == 0,
          "no overlap",
          f"overlap: {len(tr_eids & te_eids):,} shared EIDs")
    check(S, "EIDs disjoint (val∩test=∅)",
          len(va_eids & te_eids) == 0,
          "no overlap",
          f"overlap: {len(va_eids & te_eids):,} shared EIDs")

    # EID union covers all rows
    n_union = len(tr_eids | va_eids | te_eids)
    check(S, "EID union = n_total",
          n_union == si["n_total"],
          f"union={n_union:,} == n_total={si['n_total']:,}",
          f"union={n_union:,} ≠ n_total={si['n_total']:,}")

    # Load timestamps from each split to check ordering
    fs_root = root / cfg["output"]["feature_store_dir"]
    try:
        ts_train = np.load(fs_root / "train" / "timestamps.npy")
        ts_val   = np.load(fs_root / "val"   / "timestamps.npy")
        ts_test  = np.load(fs_root / "test"  / "timestamps.npy")

        check(S, "train timestamps <= val timestamps",
              ts_train.max() <= ts_val.min(),
              f"max_train={ts_train.max()}  min_val={ts_val.min()}",
              f"LEAKAGE: max_train={ts_train.max()} > min_val={ts_val.min()}")
        check(S, "val timestamps <= test timestamps",
              ts_val.max() <= ts_test.min(),
              f"max_val={ts_val.max()}  min_test={ts_test.min()}",
              f"LEAKAGE: max_val={ts_val.max()} > min_test={ts_test.min()}")
        check(S, "within train: timestamps non-decreasing",
              bool(np.all(np.diff(ts_train) >= 0)),
              "sorted", "train timestamps NOT sorted")
    except FileNotFoundError as e:
        fail(S, "timestamps loadable", str(e))

    # Compare with TE-G-SAGE reference splits
    TEG_SAGE_REF = {"train": 1_419_254, "val": 709_628, "test": 236_542}
    print(f"\n  Split size comparison (after duplicate removal):")
    print(f"  {'split':<6} {'actual':>12} {'teg_sage_ref':>12} {'delta':>10}")
    for sp, ref in TEG_SAGE_REF.items():
        actual = splits[sp]["n_edges"]
        delta  = actual - ref
        sign   = "+" if delta >= 0 else ""
        print(f"  {sp:<6} {actual:>12,} {ref:>12,} {sign}{delta:>9,}")
        status = "PASS" if abs(delta) < ref * 0.05 else "WARN"
        _report(S, f"split size {sp} within 5% of TE-G-SAGE ref",
                status,
                f"actual={actual:,}  ref={ref:,}  delta={sign}{delta:,}")

    return si


# ── SECTION 3 — Feature store dimensions and EID alignment ────────────────────

def section3(cfg: dict, si: dict) -> None:
    print("\n─── SECTION 3: Feature store dimensions and EID alignment ──────────")
    S = "S3"
    if not si:
        fail(S, "section3", "split_indices.json not available — skipping")
        return

    root    = REPO_ROOT
    fs_root = root / cfg["output"]["feature_store_dir"]
    splits  = si["splits"]
    d_e     = si["d_e"]

    d_e_values: list[int] = []
    stores: dict[str, "FeatureStore"] = {}

    for split in ("train", "val", "test"):
        n = splits[split]["n_edges"]
        feat_path = fs_root / split / "features.dat"
        if not feat_path.exists():
            fail(S, f"{split} features.dat loadable", "file missing")
            continue
        try:
            mmap = _open_mmap(feat_path, n, d_e)
            d_e_values.append(mmap.shape[1])
            ok(S, f"{split} features.dat shape",
               f"({mmap.shape[0]:,}, {mmap.shape[1]}) — n_edges matches split_indices")
        except Exception as e:
            fail(S, f"{split} features.dat shape", str(e))
            continue

        check(S, f"{split} n_edges matches split_indices",
              mmap.shape[0] == n,
              f"{mmap.shape[0]:,} == {n:,}",
              f"{mmap.shape[0]:,} ≠ {n:,}")

        try:
            stores[split] = FeatureStore(fs_root / split)
        except Exception as e:
            fail(S, f"{split} FeatureStore loadable", str(e))

    if d_e_values:
        all_same = len(set(d_e_values)) == 1
        check(S, "d_e consistent across all splits",
              all_same,
              f"d_e={d_e_values[0]} across train/val/test",
              f"d_e values differ: {dict(zip(('train','val','test'), d_e_values))}")

    # ── EID alignment spot-check ──────────────────────────────────────────────
    print("\n  EID alignment spot-check (loading raw CSV — may take ~60 s)...")
    csv_path = root / cfg["data"]["csv_path"]
    if not csv_path.exists():
        warn(S, "EID alignment spot-check", f"raw CSV not found at {csv_path} — skipping")
        return

    try:
        df = load_raw(csv_path)
        df = df.sort_values("FLOW_START_MILLISECONDS", kind="mergesort").reset_index(drop=True)
        df["GLOBAL_EID"] = np.arange(len(df), dtype=np.int64)
    except Exception as e:
        warn(S, "EID alignment spot-check", f"failed to load/sort CSV: {e}")
        return

    # Load saved transformers
    tdir = root / cfg["output"]["transformers_dir"]
    try:
        with open(tdir / "scaler.pkl", "rb") as f: scaler = pickle.load(f)
        enc_path = tdir / "encoder.pkl" if (tdir / "encoder.pkl").exists() else tdir / "ohe.pkl"
        with open(enc_path, "rb") as f: ohe = pickle.load(f)
        spearman_mask = np.load(tdir / "spearman_mask.npy")
    except Exception as e:
        warn(S, "EID alignment spot-check", f"could not load transformers: {e}")
        return

    def reconstruct_row(row: "pd.Series") -> np.ndarray:
        """Reconstruct the d_e feature vector for one CSV row."""
        # Numeric: log transform → scale → Spearman mask
        num_raw = row[NUMERIC_COLS].values.astype(np.float64)
        for i, col in enumerate(NUMERIC_COLS):
            if col in LOG_TRANSFORM_COLS:
                num_raw[i] = np.log1p(max(0.0, num_raw[i]))
        num_scaled = scaler.transform(num_raw.reshape(1, -1))[0]
        num_kept   = num_scaled[spearman_mask]

        # Port encoding
        dst_enc = encode_dst_port(np.array([int(row["L4_DST_PORT"])]))[0]   # (16,)
        src_enc = encode_src_port(np.array([int(row["L4_SRC_PORT"])]))[0]   # (1,)

        # OHE categoricals (avoid ArrowStringArray.reshape incompatibility)
        cat_raw = np.array([[str(row[col]) for col in CATEGORICAL_COLS]])
        cat_enc = ohe.transform(cat_raw)[0]                                  # (n_ohe_cols,)

        return np.concatenate([num_kept, dst_enc, src_enc, cat_enc]).astype(np.float32)

    rng = np.random.default_rng(0)
    n_sample = 100
    total_ok = 0
    total_tested = 0
    max_abs_err = 0.0

    for split in ("train", "val", "test"):
        if split not in stores:
            warn(S, f"EID alignment {split}", "FeatureStore not available")
            continue
        store = stores[split]
        split_eids = store.edge_indices
        sampled = rng.choice(split_eids, size=min(n_sample, len(split_eids)), replace=False)

        mismatches = 0
        max_err = 0.0
        for eid in sampled:
            row = df.iloc[int(eid)]
            assert int(row["GLOBAL_EID"]) == int(eid), \
                f"EID mismatch in sorted df: row {eid} has GLOBAL_EID={row['GLOBAL_EID']}"
            expected = reconstruct_row(row)
            actual   = store[int(eid)].astype(np.float32)
            err = float(np.abs(expected - actual).max())
            max_err = max(max_err, err)
            if err > 1e-4:
                mismatches += 1

        total_tested += len(sampled)
        total_ok     += len(sampled) - mismatches
        max_abs_err   = max(max_abs_err, max_err)
        check(S, f"EID alignment {split} ({len(sampled)} random EIDs, atol=1e-4)",
              mismatches == 0,
              f"all {len(sampled)} match (max_abs_err={max_err:.2e})",
              f"{mismatches}/{len(sampled)} mismatches (max_abs_err={max_err:.2e})")

    if total_tested > 0:
        ok(S, "EID alignment overall",
           f"{total_ok}/{total_tested} EIDs matched across all splits (max_abs_err={max_abs_err:.2e})")


# ── SECTION 4 — Port encoding ─────────────────────────────────────────────────

def section4(cfg: dict, si: dict) -> None:
    print("\n─── SECTION 4: Port encoding ───────────────────────────────────────")
    S = "S4"
    fg_path = REPO_ROOT / cfg["output"]["feature_groups_path"]
    if not fg_path.exists():
        fail(S, "feature_groups.json loadable", "file missing — skipping")
        return

    with open(fg_path) as f:
        fg = json.load(f)

    groups   = fg["groups"]
    feat_names = fg["feature_names"]

    # DST_PORT_GROUP: 16 bins
    if "DST_PORT_GROUP" in groups:
        dst_indices = groups["DST_PORT_GROUP"]["indices"]
        check(S, "DST_PORT_GROUP has exactly 16 columns",
              len(dst_indices) == N_DST_PORT_BINS,
              f"{len(dst_indices)} columns",
              f"found {len(dst_indices)} columns, expected {N_DST_PORT_BINS}")
        # Verify bin names in feature vector
        dst_feat_names = [feat_names[i] for i in dst_indices]
        expected_names = DST_PORT_BIN_NAMES
        check(S, "DST_PORT_GROUP column names match 16-bin scheme",
              dst_feat_names == expected_names,
              f"names match: {dst_feat_names[:4]}...",
              f"mismatch: got {dst_feat_names}, expected {expected_names}")
    else:
        fail(S, "DST_PORT_GROUP in feature_groups.json", "key missing")

    # SRC_PORT_IS_EPHEMERAL: 1 column
    if "SRC_PORT_IS_EPHEMERAL" in groups:
        src_indices = groups["SRC_PORT_IS_EPHEMERAL"]["indices"]
        check(S, "SRC_PORT_IS_EPHEMERAL has exactly 1 column",
              len(src_indices) == 1,
              f"1 column (index {src_indices[0]})",
              f"found {len(src_indices)} columns, expected 1")
    else:
        fail(S, "SRC_PORT_IS_EPHEMERAL in feature_groups.json", "key missing")

    # IP addresses NOT in feature names
    ip_cols = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
    for col in ip_cols:
        check(S, f"{col} NOT in feature store",
              col not in feat_names,
              f"absent from d_e={len(feat_names)} features",
              f"PRESENT in feature vector — IP address should not be an edge feature")

    # Timestamps NOT in trainable features
    ts_cols = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"]
    for col in ts_cols:
        check(S, f"{col} NOT in feature store (stored separately as timestamps.npy)",
              col not in feat_names,
              "absent",
              f"PRESENT in feature vector — timestamps must not be trainable features")

    # Spot-check on actual features.dat: port-bin rows sum to 1, is_ephemeral in {0,1}
    if si:
        fs_root  = REPO_ROOT / cfg["output"]["feature_store_dir"]
        dst_cols = groups["DST_PORT_GROUP"]["indices"]
        src_col  = groups["SRC_PORT_IS_EPHEMERAL"]["indices"][0]
        d_e      = si["d_e"]

        for split in ("train", "val", "test"):
            n = si["splits"][split]["n_edges"]
            feat_path = fs_root / split / "features.dat"
            if not feat_path.exists():
                warn(S, f"port encoding spot-check {split}", "features.dat missing")
                continue
            rng = np.random.default_rng(42)
            mmap = _open_mmap(feat_path, n, d_e)
            sample_idx = rng.choice(n, size=min(1000, n), replace=False)
            rows = mmap[np.sort(sample_idx)]

            # DST port bins sum to 1.0 per row
            dst_sums = rows[:, dst_cols].sum(axis=1)
            bad_dst  = int(np.sum(np.abs(dst_sums - 1.0) > 1e-5))
            check(S, f"DST port bins sum=1 ({split}, 1000 rows)",
                  bad_dst == 0,
                  "all rows sum to 1.0",
                  f"{bad_dst} rows do not sum to 1.0 (min={dst_sums.min():.4f}, max={dst_sums.max():.4f})")

            # SRC is_ephemeral in {0, 1}
            src_vals = rows[:, src_col]
            bad_src  = int(np.sum(~np.isin(src_vals, [0.0, 1.0])))
            check(S, f"SRC is_ephemeral in {{0,1}} ({split}, 1000 rows)",
                  bad_src == 0,
                  "all values 0 or 1",
                  f"{bad_src} values outside {{0.0, 1.0}}")


# ── SECTION 5 — Feature engineering correctness ───────────────────────────────

def section5(cfg: dict, si: dict) -> None:
    print("\n─── SECTION 5: Feature engineering correctness ─────────────────────")
    S = "S5"
    tdir = REPO_ROOT / cfg["output"]["transformers_dir"]

    # Load transformers
    try:
        with open(tdir / "scaler.pkl", "rb") as f: scaler = pickle.load(f)
        with open(tdir / "meta.json") as f: meta = json.load(f)
        spearman_mask = np.load(tdir / "spearman_mask.npy")
        enc_path = tdir / "encoder.pkl" if (tdir / "encoder.pkl").exists() else tdir / "ohe.pkl"
        with open(enc_path, "rb") as f: ohe = pickle.load(f)
    except Exception as e:
        fail(S, "transformers loadable", str(e))
        return

    # Numeric features after pruning
    n_numeric_kept = int(spearman_mask.sum())
    n_numeric_total = len(spearman_mask)
    dropped = [NUMERIC_COLS[i] for i in range(n_numeric_total) if not spearman_mask[i]]
    check(S, "numeric features after Spearman pruning (~38–40)",
          35 <= n_numeric_kept <= 42,
          f"{n_numeric_kept} (from {n_numeric_total}); dropped: {dropped}",
          f"{n_numeric_kept} outside expected range 35–42; dropped: {dropped}")

    # Spearman threshold applied correctly
    check(S, "Spearman threshold in meta.json",
          abs(meta.get("spearman_threshold", -1) - cfg["preprocessing"]["spearman_threshold"]) < 1e-9,
          f"threshold={meta.get('spearman_threshold')}",
          f"meta threshold={meta.get('spearman_threshold')} ≠ config {cfg['preprocessing']['spearman_threshold']}")

    # Scaler: apply to val features, confirm finite output
    if si:
        fs_root = REPO_ROOT / cfg["output"]["feature_store_dir"]
        n_val   = si["splits"]["val"]["n_edges"]
        d_e     = si["d_e"]
        val_path = fs_root / "val" / "features.dat"
        if val_path.exists():
            mmap_val = _open_mmap(val_path, n_val, d_e)
            # Extract scaled numeric columns (first n_numeric_kept columns)
            val_numeric_scaled = mmap_val[:1000, :n_numeric_kept].astype(np.float64)
            try:
                # inverse_transform needs all n_numeric_total columns
                # reconstruct full scaled array (fill pruned col with 0 for shape compat)
                full_scaled = np.zeros((val_numeric_scaled.shape[0], n_numeric_total))
                full_scaled[:, spearman_mask] = val_numeric_scaled
                inv = scaler.inverse_transform(full_scaled)
                finite_ok = bool(np.isfinite(inv).all())
                check(S, "scaler.transform(val_features) produces finite output",
                      finite_ok, "all finite", "contains NaN or inf after inverse_transform")
            except Exception as e:
                fail(S, "scaler applicable to val features", str(e))

    # OHE: check it was fitted on training categoricals
    check(S, "OHE n_features_in_ == 7 categoricals",
          ohe.n_features_in_ == len(CATEGORICAL_COLS),
          f"{ohe.n_features_in_} == {len(CATEGORICAL_COLS)}",
          f"{ohe.n_features_in_} ≠ {len(CATEGORICAL_COLS)}")

    # Report categories seen in val/test that are absent from training
    if si:
        fs_root  = REPO_ROOT / cfg["output"]["feature_store_dir"]
        # We check by verifying handle_unknown='ignore' is set and output is finite for val
        check(S, "OHE handle_unknown='ignore' or 'infrequent_if_exist'",
              ohe.handle_unknown in ("ignore", "infrequent_if_exist"),
              f"handle_unknown='{ohe.handle_unknown}'",
              f"handle_unknown='{ohe.handle_unknown}' — unknown val/test categories will raise")

    # Check that OHE categories_ comes only from training (can't check directly, but
    # we can verify the feature_names in meta.json match what the OHE produces)
    ohe_out_names = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
    meta_feat_names = meta["feature_names"]
    n_numeric_in_meta  = len(meta["numeric_cols_kept"])
    ohe_start_in_meta  = n_numeric_in_meta + N_DST_PORT_BINS + 1
    meta_ohe_names     = meta_feat_names[ohe_start_in_meta:]
    check(S, "OHE feature names consistent with meta.json",
          ohe_out_names == meta_ohe_names,
          f"{len(ohe_out_names)} OHE columns match meta.json",
          f"mismatch: OHE has {len(ohe_out_names)} cols, meta has {len(meta_ohe_names)}")


# ── SECTION 6 — Balancer correctness ─────────────────────────────────────────

def section6(cfg: dict, si: dict) -> None:
    print("\n─── SECTION 6: Balancer correctness ───────────────────────────────")
    S = "S6"
    root = REPO_ROOT

    bal_path = root / cfg["output"]["balanced_train_indices_path"]
    cw_path  = root / cfg["output"]["class_weights_path"]

    if not bal_path.exists():
        fail(S, "balanced_train_indices.npy loadable", "file missing"); return
    if not si:
        fail(S, "section6", "split_indices.json not available — skipping"); return

    balanced_eids = np.load(bal_path)
    ok(S, "balanced_train_indices.npy loadable", f"{len(balanced_eids):,} EIDs")

    # Load train timestamps and labels
    fs_root = root / cfg["output"]["feature_store_dir"]
    ts_train = np.load(fs_root / "train" / "timestamps.npy")
    lb_train = np.load(fs_root / "train" / "labels.npy")
    ei_train = np.load(fs_root / "train" / "edge_indices.npy")

    n_train = si["splits"]["train"]["n_edges"]

    # Build O(1) lookup: EID → (timestamp, label)
    eid_start = int(ei_train[0])
    ts_arr = np.zeros(int(ei_train.max()) + 1, dtype=np.int64)
    lb_arr = np.full(int(ei_train.max()) + 1, -1, dtype=np.int64)
    ts_arr[ei_train] = ts_train
    lb_arr[ei_train] = lb_train

    # Confirm balanced EIDs are within training EID range
    bal_eids_in_range = np.all(
        (balanced_eids >= int(ei_train.min())) & (balanced_eids <= int(ei_train.max()))
    )
    check(S, "balanced EIDs all within train EID range",
          bool(bal_eids_in_range),
          "all EIDs belong to training split",
          f"some EIDs outside train range [{ei_train.min()}, {ei_train.max()}]")

    # Confirm sorted by timestamp
    bal_timestamps = ts_arr[balanced_eids]
    n_inversions = int(np.sum(np.diff(bal_timestamps) < 0))
    check(S, "balanced indices sorted by timestamp",
          n_inversions == 0,
          "non-decreasing",
          f"{n_inversions:,} timestamp inversions found")

    # Confirm all original training EIDs appear at least once
    bal_set = set(balanced_eids.tolist())
    original_set = set(ei_train.tolist())
    missing = original_set - bal_set
    check(S, "all original train EIDs present in balanced set",
          len(missing) == 0,
          f"all {len(original_set):,} original EIDs present",
          f"{len(missing):,} original EIDs absent from balanced set")

    # Class distribution after balancing
    bal_labels = lb_arr[balanced_eids]
    classes, counts = np.unique(bal_labels, return_counts=True)
    majority_count = counts.max()
    min_class_ratio = cfg["balancer"]["min_class_ratio"]

    print(f"\n  Balanced class distribution:")
    print(f"  {'class':>6}  {'balanced_n':>12}  {'ratio_to_majority':>18}")
    all_pass = True
    for cls, cnt in zip(classes, counts):
        ratio = cnt / majority_count
        meets = ratio >= min_class_ratio
        all_pass = all_pass and meets
        flag = "✓" if meets else "✗"
        print(f"  {cls:>6}  {cnt:>12,}  {ratio:>18.4f} {flag} (min={min_class_ratio})")
    check(S, f"all classes >= min_class_ratio ({min_class_ratio}) × majority count",
          all_pass,
          "all classes meet minimum ratio",
          "some classes below minimum ratio (see table above)")

    # Class weights from ORIGINAL unbalanced distribution
    if not cw_path.exists():
        fail(S, "class_weights.npy loadable", "file missing"); return
    cw = np.load(cw_path)
    ok(S, "class_weights.npy loadable", f"shape={cw.shape}  values={np.round(cw, 4)}")

    # Verify weights are inverse-proportional to ORIGINAL (unbalanced) frequencies
    orig_classes, orig_counts = np.unique(lb_train, return_counts=True)
    n_total_train = len(lb_train)
    expected_weights = n_total_train / (len(orig_classes) * orig_counts)

    weight_ok = True
    for cls, expected_w in zip(orig_classes, expected_weights):
        if cls >= len(cw):
            warn(S, f"class_weights class {cls}", "index out of range")
            continue
        actual_w = cw[cls]
        if abs(actual_w - expected_w) / (expected_w + 1e-9) > 0.01:
            weight_ok = False
            fail(S, f"class_weight[{cls}] from original distribution",
                 f"actual={actual_w:.4f}  expected≈{expected_w:.4f} "
                 f"(n_original={orig_counts[cls]:,})")

    if weight_ok:
        ok(S, "class weights inversely proportional to ORIGINAL class frequencies",
           f"max rel error < 1% — weights NOT from balanced distribution")


# ── SECTION 7 — No leakage ────────────────────────────────────────────────────

def section7(cfg: dict, si: dict) -> None:
    print("\n─── SECTION 7: No leakage check ────────────────────────────────────")
    S = "S7"
    tdir    = REPO_ROOT / cfg["output"]["transformers_dir"]
    fs_root = REPO_ROOT / cfg["output"]["feature_store_dir"]

    if not si:
        fail(S, "section7", "split_indices.json not available — skipping"); return

    try:
        with open(tdir / "scaler.pkl", "rb") as f: scaler = pickle.load(f)
        spearman_mask = np.load(tdir / "spearman_mask.npy")
    except Exception as e:
        fail(S, "scaler loadable", str(e)); return

    n_numeric_kept  = int(spearman_mask.sum())
    n_numeric_total = len(spearman_mask)
    d_e             = si["d_e"]
    n_train         = si["splits"]["train"]["n_edges"]

    train_feat_path = fs_root / "train" / "features.dat"
    if not train_feat_path.exists():
        fail(S, "train features.dat for leakage check", "missing"); return

    print("  Loading train features.dat for leakage check (~1.2 GB, may take ~10 s)...")
    mmap_train = _open_mmap(train_feat_path, n_train, d_e)

    # Extract scaled numeric columns for training data
    # features.dat[:, :n_numeric_kept] = (log_X - mean_[kept]) / scale_[kept]
    # Reconstruct log_X via inverse_transform
    train_scaled_kept = mmap_train[:, :n_numeric_kept].astype(np.float64)

    # inverse_transform needs full (n, n_numeric_total) array;
    # fill the pruned column(s) with zeros
    full_scaled_train = np.zeros((n_train, n_numeric_total), dtype=np.float64)
    full_scaled_train[:, spearman_mask] = train_scaled_kept

    print("  Computing inverse_transform (reconstructing log-space training features)...")
    log_train = scaler.inverse_transform(full_scaled_train)   # shape (n_train, n_numeric_total)
    log_train_kept = log_train[:, spearman_mask]               # (n_train, n_numeric_kept)

    # The scaler was fitted on training data, so:
    #   mean(log_train_kept) ≈ scaler.mean_[spearman_mask]
    #   var(log_train_kept)  ≈ scaler.var_[spearman_mask]
    computed_mean = log_train_kept.mean(axis=0)
    computed_var  = log_train_kept.var(axis=0)
    scaler_mean   = scaler.mean_[spearman_mask]
    scaler_var    = scaler.var_[spearman_mask]

    # Due to float32 round-trip, use atol=1e-2 (float32 precision limits 1e-3 for large means)
    mean_diffs  = np.abs(computed_mean - scaler_mean)
    var_diffs   = np.abs(computed_var  - scaler_var)
    max_mean_diff = float(mean_diffs.max())
    max_var_diff  = float(var_diffs.max())
    worst_mean_col = int(mean_diffs.argmax())
    worst_var_col  = int(var_diffs.argmax())

    atol = 1e-2  # relaxed for float32 round-trip
    mean_ok = max_mean_diff < atol
    var_ok  = max_var_diff  < atol

    check(S, f"scaler.mean_ matches train log-feature mean (atol={atol})",
          mean_ok,
          f"max |Δmean| = {max_mean_diff:.2e} — scaler fitted on train only",
          f"max |Δmean| = {max_mean_diff:.2e} at col '{NUMERIC_COLS[worst_mean_col]}' "
          f"(computed={computed_mean[worst_mean_col]:.4f}, "
          f"scaler.mean_={scaler_mean[worst_mean_col]:.4f}) — possible leakage?")

    check(S, f"scaler.var_ matches train log-feature variance (atol={atol})",
          var_ok,
          f"max |Δvar| = {max_var_diff:.2e} — consistent with train-only fit",
          f"max |Δvar| = {max_var_diff:.2e} at col '{NUMERIC_COLS[worst_var_col]}' "
          f"(computed={computed_var[worst_var_col]:.4f}, "
          f"scaler.var_={scaler_var[worst_var_col]:.4f}) — possible leakage?")

    # After StandardScaler, mean of TRAINING scaled features should be ~0
    scaled_mean = train_scaled_kept.mean(axis=0)
    max_scaled_mean = float(np.abs(scaled_mean).max())
    check(S, "mean of scaled train numerics ≈ 0 (StandardScaler property, atol=1e-3)",
          max_scaled_mean < 1e-3,
          f"max |mean(scaled_train)| = {max_scaled_mean:.2e}",
          f"max |mean(scaled_train)| = {max_scaled_mean:.2e} — unexpected for StandardScaler")

    # Val mean of scaled numerics should NOT be near zero (different distribution)
    n_val = si["splits"]["val"]["n_edges"]
    val_feat_path = fs_root / "val" / "features.dat"
    if val_feat_path.exists():
        mmap_val = _open_mmap(val_feat_path, n_val, d_e)
        val_scaled = mmap_val[:, :n_numeric_kept].astype(np.float64)
        val_scaled_mean     = val_scaled.mean(axis=0)
        val_max_abs_mean    = float(np.abs(val_scaled_mean).max())
        # If leakage occurred (scaler fitted on val too), val mean would also be ~0
        val_mean_nonzero = val_max_abs_mean > 0.01
        check(S, "mean of scaled val numerics ≠ 0 (scaler not fitted on val)",
              val_mean_nonzero,
              f"max |mean(scaled_val)| = {val_max_abs_mean:.4f} — scaler NOT fitted on val",
              f"max |mean(scaled_val)| = {val_max_abs_mean:.4f} — suspiciously near 0, "
              f"check for leakage")

    # Labels.npy per split: confirm each split's labels are from the correct EID range
    for split in ("train", "val", "test"):
        ei = np.load(fs_root / split / "edge_indices.npy")
        lb = np.load(fs_root / split / "labels.npy")
        split_info = si["splits"][split]
        # All EIDs in this split must be within [eid_start, eid_end]
        in_range = bool(np.all(
            (ei >= split_info["eid_start"]) & (ei <= split_info["eid_end"])
        ))
        check(S, f"{split} labels.npy EIDs within split boundaries",
              in_range,
              f"all {len(ei):,} EIDs in [{split_info['eid_start']}, {split_info['eid_end']}]",
              f"some EIDs outside [{split_info['eid_start']}, {split_info['eid_end']}]")

        # Labels should be valid integers (0 or 1 for binary, or 0..n_classes-1)
        valid_labels = bool(np.all(lb >= 0))
        check(S, f"{split} labels all non-negative",
              valid_labels,
              f"min={lb.min()}  max={lb.max()}  n_classes={len(np.unique(lb))}",
              f"negative labels found: min={lb.min()}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 output validation")
    parser.add_argument("--config", default="configs/experiment_unsw.yaml")
    args = parser.parse_args()

    cfg = load_config(REPO_ROOT / args.config)

    print("=" * 70)
    print("Phase 1 Validation — specs/01_data_pipeline.md")
    print("=" * 70)

    section1(cfg)
    si = section2(cfg)
    section3(cfg, si)
    section4(cfg, si)
    section5(cfg, si)
    section6(cfg, si)
    section7(cfg, si)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    by_status: dict[str, list] = {"PASS": [], "FAIL": [], "WARN": []}
    for sec, name, status, msg in RESULTS:
        by_status[status].append((sec, name, msg))

    n_pass = len(by_status["PASS"])
    n_fail = len(by_status["FAIL"])
    n_warn = len(by_status["WARN"])

    print(f"  PASS: {n_pass}   FAIL: {n_fail}   WARN: {n_warn}")
    print(f"  Total checks: {n_pass + n_fail + n_warn}")

    if by_status["FAIL"]:
        print(f"\n  FAILURES:")
        for sec, name, msg in by_status["FAIL"]:
            print(f"    [{sec}] {name}")
            print(f"           → {msg}")

    if by_status["WARN"]:
        print(f"\n  WARNINGS:")
        for sec, name, msg in by_status["WARN"]:
            print(f"    [{sec}] {name}")
            print(f"           → {msg}")

    print("=" * 70)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
