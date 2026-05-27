"""
Feature engineering and chronological temporal splitting.

TEMPORAL SPLITTING:
Sort all flows by FLOW_START_MILLISECONDS. Assign global EID = position (0-indexed).
Compute cut points via quantile of the sorted index:
  E_train = {e | start(e) <= tau_train}
  E_val   = {e | tau_train < start(e) <= tau_val}
  E_test  = {e | start(e) > tau_val}

All transformers (scaler, Spearman mask, OHE) are fit on training data ONLY.

FEATURE ENGINEERING ORDER:
1. Port encoding  — deterministic, no fitting
2. Log transform  — deterministic, no fitting
3. StandardScaler — fit on train
4. Spearman pruning — mask computed on train
5. OHE categoricals — fit on train

CRITICAL INVARIANT: No val/test data touches any fit step.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.loader import (
    CATEGORICAL_COLS,
    LOG_TRANSFORM_COLS,
    NUMERIC_COLS,
    THROUGHPUT_COLS,
)

logger = logging.getLogger(__name__)

# 16-bin DST port scheme (finalized after dataset pilot — CLAUDE.md phase 4)
# Each entry: (bin_name, predicate_fn)
_DST_PORT_BINS: list[tuple[str, Any]] = [
    ("HTTP",             lambda p: p == 80),
    ("HTTPS",            lambda p: p == 443),
    ("DNS",              lambda p: p == 53),
    ("SSH",              lambda p: p == 22),
    ("FTP",              lambda p: p in (20, 21)),
    ("SMTP",             lambda p: p in (25, 587)),
    ("RDP",              lambda p: p == 3389),    # T1021.001 Lateral Movement
    ("SNMP",             lambda p: p in (161, 162)),  # T1046 Discovery
    ("NTP",              lambda p: p == 123),     # T1124 / T1498.002
    ("IMAP",             lambda p: p in (143, 993)),  # ~2% traffic
    ("SunRPC",           lambda p: p == 111),     # ~3.7% traffic
    ("BitTorrent",       lambda p: 6881 <= p <= 6889),  # ~7% traffic
    ("AIM_ICQ",          lambda p: p == 5190),    # ~2.5% traffic
    ("other_well_known", lambda p: 0 <= p <= 1023),   # remainder
    ("registered",       lambda p: 1024 <= p <= 49151),  # remainder
    ("ephemeral",        lambda p: p >= 49152),
]
DST_PORT_BIN_NAMES: list[str] = [f"DST_PORT_{name}" for name, _ in _DST_PORT_BINS]
N_DST_PORT_BINS: int = len(_DST_PORT_BINS)  # 16


def encode_dst_port(ports: np.ndarray) -> np.ndarray:
    """Map L4_DST_PORT values → 16-bin one-hot array, shape (n, 16).

    Bins are evaluated in order; first match wins.
    Every port maps to exactly one bin (guaranteed by coverage design).
    """
    n = len(ports)
    out = np.zeros((n, N_DST_PORT_BINS), dtype=np.float32)
    # Vectorise each bin with a boolean mask
    unassigned = np.ones(n, dtype=bool)
    for j, (_, pred) in enumerate(_DST_PORT_BINS):
        try:
            # Convert predicate to vectorised boolean
            match = np.array([pred(int(p)) for p in ports], dtype=bool)
        except Exception:
            match = np.zeros(n, dtype=bool)
        hit = match & unassigned
        out[hit, j] = 1.0
        unassigned &= ~hit
    # Any port still unassigned falls to "registered" (bin 14, index 14)
    # In practice this should not occur given the bin design.
    assert unassigned.sum() == 0, (
        f"{unassigned.sum()} ports unassigned after 16-bin encoding"
    )
    return out


def port_to_bin_indices(ports: np.ndarray) -> np.ndarray:
    """Map L4_DST_PORT values → bin indices (0-15), shape (n,), dtype int32.

    Thin wrapper around encode_dst_port: returns argmax of the one-hot output.
    Used by NodeStateManager so that dst_port_entropy and unique_dst_port_count
    operate over the 16 semantic service bins rather than raw port numbers.
    This ensures that 80/8080/8443 (all HTTP/HTTPS) appear as one service,
    not as three distinct high-entropy destinations.
    """
    return encode_dst_port(ports).argmax(axis=1).astype(np.int32)


def encode_src_port(ports: np.ndarray) -> np.ndarray:
    """Map L4_SRC_PORT → binary is_ephemeral (1 if >= 49152), shape (n, 1)."""
    return (np.asarray(ports) >= 49152).astype(np.float32).reshape(-1, 1)


def _spearman_prune_mask(
    X_train: np.ndarray,
    feature_names: list[str],
    threshold: float,
) -> np.ndarray:
    """Return boolean keep-mask for numeric features.

    For each pair (i, j) with i < j where |rho_s| > threshold, column j is
    dropped (greedy upper-triangle rule, same as TE-G-SAGE).
    """
    logger.info(f"Computing Spearman correlation matrix on {X_train.shape} train array")
    df = pd.DataFrame(X_train, columns=feature_names)
    ranks = df.rank(method="average")
    corr = ranks.corr().values
    n = corr.shape[0]

    to_drop: set[int] = set()
    for i in range(n):
        if i in to_drop:
            continue
        for j in range(i + 1, n):
            if j in to_drop:
                continue
            if abs(corr[i, j]) > threshold:
                to_drop.add(j)

    mask = np.array([i not in to_drop for i in range(n)])
    dropped = [feature_names[i] for i in sorted(to_drop)]
    logger.info(
        f"Spearman pruning (threshold={threshold}): "
        f"dropped {len(dropped)} features → {feature_names} surviving {mask.sum()}"
    )
    logger.info(f"  Dropped: {dropped}")
    return mask


class Preprocessor:
    """Full feature engineering pipeline for NF-UNSW-NB15-v3.

    Fit on training split only; transform all splits consistently.
    """

    def __init__(
        self,
        train_frac: float = 0.6,
        val_frac: float = 0.3,
        spearman_threshold: float = 0.995,
        ohe_min_frequency: float = 0.001,
        ohe_handle_unknown: str = "ignore",
    ) -> None:
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.spearman_threshold = spearman_threshold
        self.ohe_min_frequency = ohe_min_frequency
        self.ohe_handle_unknown = ohe_handle_unknown

        # Fitted state (populated by fit_transform)
        self.scaler: StandardScaler | None = None
        self.spearman_mask: np.ndarray | None = None   # bool, shape (n_numeric,)
        self.ohe: OneHotEncoder | None = None
        self.numeric_cols_kept: list[str] = []         # after pruning
        self.feature_names: list[str] = []             # final d_e names
        self.d_e: int = 0

        # Label map (populated by fit_transform via build_label_map)
        self.label_map: dict[str, int] = {}
        self.num_classes: int = 0

        # Split metadata
        self.n_total: int = 0
        self.tau_train_ms: int = 0
        self.tau_val_ms: int = 0
        self.split_sizes: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Sort, split, fit on train, transform all splits.

        Returns three dicts keyed by split name, each containing:
          "features"      : float32 array (n, d_e)
          "edge_indices"  : int64 array   (n,) global EIDs
          "timestamps"    : int64 array   (n,) FLOW_START_MILLISECONDS
          "labels"        : int64 array   (n,) Label (integer class)
        """
        df = df.sort_values("FLOW_START_MILLISECONDS", kind="mergesort").reset_index(
            drop=True
        )
        df["GLOBAL_EID"] = np.arange(len(df), dtype=np.int64)
        self.n_total = len(df)

        # Build label map from full dataset before splitting so that any class
        # absent from train but present in val/test still receives a stable int.
        self.label_map = self.build_label_map(df)
        self.num_classes = len(self.label_map)
        logger.info(
            f"Label map ({self.num_classes} classes): "
            + ", ".join(f"{k}={v}" for k, v in sorted(self.label_map.items(), key=lambda x: x[1]))
        )

        train_df, val_df, test_df = self._temporal_split(df)
        logger.info(
            f"Splits — train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}"
        )

        # Step 1: port encoding (deterministic, no fitting)
        train_port = self._port_features(train_df)
        val_port   = self._port_features(val_df)
        test_port  = self._port_features(test_df)

        # Step 2: log transform on designated numeric cols (deterministic)
        train_num = self._log_transform(train_df)
        val_num   = self._log_transform(val_df)
        test_num  = self._log_transform(test_df)

        # Step 3: StandardScaler — fit on train
        self.scaler = StandardScaler()
        train_scaled = self.scaler.fit_transform(train_num)
        val_scaled   = self.scaler.transform(val_num)
        test_scaled  = self.scaler.transform(test_num)

        # Step 4: Spearman pruning — mask from train
        self.spearman_mask = _spearman_prune_mask(
            train_scaled, NUMERIC_COLS, self.spearman_threshold
        )
        train_scaled = train_scaled[:, self.spearman_mask]
        val_scaled   = val_scaled[:, self.spearman_mask]
        test_scaled  = test_scaled[:, self.spearman_mask]
        self.numeric_cols_kept = [
            c for c, keep in zip(NUMERIC_COLS, self.spearman_mask) if keep
        ]
        logger.info(f"Numeric features after Spearman pruning: {len(self.numeric_cols_kept)}")

        # Step 5: OHE categoricals — fit on train
        train_cat_raw = train_df[CATEGORICAL_COLS].astype(str)
        val_cat_raw   = val_df[CATEGORICAL_COLS].astype(str)
        test_cat_raw  = test_df[CATEGORICAL_COLS].astype(str)

        self.ohe = OneHotEncoder(
            min_frequency=self.ohe_min_frequency,
            handle_unknown=self.ohe_handle_unknown,
            sparse_output=False,
            dtype=np.float32,
        )
        train_ohe = self.ohe.fit_transform(train_cat_raw)
        val_ohe   = self.ohe.transform(val_cat_raw)
        test_ohe  = self.ohe.transform(test_cat_raw)
        logger.info(f"OHE output: {train_ohe.shape[1]} columns for {len(CATEGORICAL_COLS)} categoricals")

        # Assemble final feature matrix: scaled_numeric | port_bins | ohe
        def _concat(scaled: np.ndarray, port: np.ndarray, ohe: np.ndarray) -> np.ndarray:
            return np.concatenate([scaled.astype(np.float32),
                                   port.astype(np.float32),
                                   ohe.astype(np.float32)], axis=1)

        train_X = _concat(train_scaled, train_port, train_ohe)
        val_X   = _concat(val_scaled,   val_port,   val_ohe)
        test_X  = _concat(test_scaled,  test_port,  test_ohe)

        # Encode multi-class Attack labels (must happen before _pack)
        for sub_df in (train_df, val_df, test_df):
            sub_df["_attack_int"] = sub_df["Attack"].map(self.label_map)
            unknown = sub_df["_attack_int"].isna()
            if unknown.any():
                bad = sub_df.loc[unknown, "Attack"].unique().tolist()
                raise ValueError(
                    f"Unknown Attack values: {bad}. "
                    "These classes were absent from the full dataset — "
                    "check the Attack column for unexpected values."
                )
            sub_df["_attack_int"] = sub_df["_attack_int"].astype(np.int64)

        # Build feature name list and record d_e
        ohe_names = list(self.ohe.get_feature_names_out(CATEGORICAL_COLS))
        self.feature_names = (
            self.numeric_cols_kept
            + DST_PORT_BIN_NAMES
            + ["SRC_PORT_IS_EPHEMERAL"]
            + ohe_names
        )
        self.d_e = len(self.feature_names)
        logger.info(f"Final feature dimension d_e = {self.d_e}")

        self.split_sizes = {
            "train": len(train_df),
            "val":   len(val_df),
            "test":  len(test_df),
        }

        def _pack(sub_df: pd.DataFrame, X: np.ndarray) -> dict[str, np.ndarray]:
            return {
                "features":     X,
                "edge_indices": sub_df["GLOBAL_EID"].values.astype(np.int64),
                "timestamps":   sub_df["FLOW_START_MILLISECONDS"].values.astype(np.int64),
                "labels":       sub_df["_attack_int"].values.astype(np.int64),
            }

        return (
            _pack(train_df, train_X),
            _pack(val_df,   val_X),
            _pack(test_df,  test_X),
        )

    @staticmethod
    def build_label_map(df: pd.DataFrame) -> dict[str, int]:
        """Derive a deterministic class → int mapping from df["Attack"].

        ``"Benign"`` always receives int 0.  All remaining unique class names
        are sorted alphabetically and assigned ints 1, 2, ..., N-1.  This
        guarantees identical mappings across independent runs on the same
        dataset.

        Args:
            df: DataFrame containing an ``Attack`` column with class strings.

        Returns:
            Mapping of Attack string → integer label (e.g. ``{"Benign": 0, ...}``).
        """
        classes = df["Attack"].unique().tolist()
        if "Benign" not in classes:
            raise ValueError(
                "Attack column does not contain 'Benign'. "
                "Check that this is a supported NetFlow dataset."
            )
        others = sorted(c for c in classes if c != "Benign")
        label_map: dict[str, int] = {"Benign": 0}
        for i, cls in enumerate(others, start=1):
            label_map[cls] = i
        return label_map

    def save_label_map(self, path: Path | str) -> None:
        """Write self.label_map to a JSON file (Attack string → int).

        The file is compatible with the format read by evaluator.py,
        scripts/06_explain.py, scripts/08_metrics.py, etc.

        Args:
            path: destination file path (will be created/overwritten).
        """
        if not self.label_map:
            raise RuntimeError(
                "label_map is empty — call fit_transform() before save_label_map()."
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.label_map, f, indent=2)
        logger.info(f"label_map.json saved → {path}  ({self.num_classes} classes)")

    def save_transformers(self, transformers_dir: Path | str) -> None:
        """Persist fitted transformers for later use on val/test/inference."""
        transformers_dir = Path(transformers_dir)
        transformers_dir.mkdir(parents=True, exist_ok=True)

        with open(transformers_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(transformers_dir / "ohe.pkl", "wb") as f:
            pickle.dump(self.ohe, f)
        np.save(transformers_dir / "spearman_mask.npy", self.spearman_mask)

        meta = {
            "numeric_cols_kept": self.numeric_cols_kept,
            "categorical_cols": CATEGORICAL_COLS,
            "feature_names": self.feature_names,
            "d_e": self.d_e,
            "spearman_threshold": self.spearman_threshold,
        }
        with open(transformers_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Transformers saved to {transformers_dir}")

    def split_indices_dict(self) -> dict:
        """Return split boundary metadata for split_indices.json."""
        n_train = self.split_sizes["train"]
        n_val   = self.split_sizes["val"]
        n_test  = self.split_sizes["test"]
        return {
            "n_total": self.n_total,
            "d_e": self.d_e,
            "tau_train_ms": int(self.tau_train_ms),
            "tau_val_ms":   int(self.tau_val_ms),
            "splits": {
                "train": {"n_edges": n_train, "eid_start": 0,               "eid_end": n_train - 1},
                "val":   {"n_edges": n_val,   "eid_start": n_train,          "eid_end": n_train + n_val - 1},
                "test":  {"n_edges": n_test,  "eid_start": n_train + n_val,  "eid_end": n_train + n_val + n_test - 1},
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _temporal_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split chronologically at train_frac and train_frac+val_frac quantiles."""
        n = len(df)
        train_cut = int(n * self.train_frac)
        val_cut   = int(n * (self.train_frac + self.val_frac))

        self.tau_train_ms = int(df["FLOW_START_MILLISECONDS"].iloc[train_cut])
        self.tau_val_ms   = int(df["FLOW_START_MILLISECONDS"].iloc[val_cut])

        train_mask = df["FLOW_START_MILLISECONDS"] <= self.tau_train_ms
        val_mask   = (df["FLOW_START_MILLISECONDS"] > self.tau_train_ms) & \
                     (df["FLOW_START_MILLISECONDS"] <= self.tau_val_ms)
        test_mask  = df["FLOW_START_MILLISECONDS"] > self.tau_val_ms

        return df[train_mask].copy(), df[val_mask].copy(), df[test_mask].copy()

    def _log_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply log(1+x) to LOG_TRANSFORM_COLS; return full numeric array."""
        arr = df[NUMERIC_COLS].values.astype(np.float64).copy()
        log_indices = [i for i, c in enumerate(NUMERIC_COLS) if c in LOG_TRANSFORM_COLS]
        arr[:, log_indices] = np.log1p(np.clip(arr[:, log_indices], 0, None))
        return arr

    def _port_features(self, df: pd.DataFrame) -> np.ndarray:
        """Return port encoding array: shape (n, 17) = 16 DST bins + 1 SRC is_ephemeral."""
        dst_enc = encode_dst_port(df["L4_DST_PORT"].values)
        src_enc = encode_src_port(df["L4_SRC_PORT"].values)
        return np.concatenate([dst_enc, src_enc], axis=1)
