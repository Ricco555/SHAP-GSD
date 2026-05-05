"""
Load raw NF-UNSW-NB15-v3 CSV.

Source: https://staff.itee.uq.edu.au/marius/NIDS_datasets/
Columns: 53 NF features + Label + Attack

Cleaning steps:
- Enforce numeric dtypes for quantitative fields
- Drop exact duplicates
- Drop rows missing IPV4_SRC_ADDR, IPV4_DST_ADDR, L4_SRC_PORT, L4_DST_PORT
- Impute: median for numeric NaN, replace inf with NaN first
- Fill throughput columns NaN with 0 (flow duration capped at 120s → zero throughput)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Used for node identity and graph construction — NOT edge features
NODE_ID_COLS: list[str] = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]

# Flow timestamps — used for splits and temporal ordering, NOT edge features
TIMESTAMP_COLS: list[str] = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"]

# Port columns — encoded into edge features (not stored raw)
PORT_COLS: list[str] = ["L4_SRC_PORT", "L4_DST_PORT"]

# Label columns
LABEL_COLS: list[str] = ["Label", "Attack"]

# Throughput columns: NaN means zero-duration flow → zero throughput
THROUGHPUT_COLS: list[str] = [
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
]

# Categorical variables → one-hot encoded in preprocessor
CATEGORICAL_COLS: list[str] = [
    "PROTOCOL",
    "L7_PROTO",
    "ICMP_TYPE",
    "ICMP_IPV4_TYPE",
    "DNS_QUERY_TYPE",
    "DNS_QUERY_ID",
    "FTP_COMMAND_RET_CODE",
]

# All numeric edge-feature columns (before log transform / scaling)
# Excludes: node IDs, timestamps, ports, labels, and categorical columns
NUMERIC_COLS: list[str] = [
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "OUT_PKTS",
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "MIN_TTL",
    "MAX_TTL",
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN",
    "TCP_WIN_MAX_OUT",
    "DNS_TTL_ANSWER",
    "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN",
    "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
]

# Subset of NUMERIC_COLS that benefit from log(1+x) — count/volume/duration fields
LOG_TRANSFORM_COLS: list[str] = [
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN",
    "TCP_WIN_MAX_OUT",
    "DNS_TTL_ANSWER",
    "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN",
    "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
]


def load_raw(csv_path: Path | str) -> pd.DataFrame:
    """Load and clean the raw NF-UNSW-NB15-v3 CSV.

    Returns a cleaned DataFrame retaining all original columns.
    Global EID assignment and splitting are performed downstream in preprocessor.
    """
    csv_path = Path(csv_path)
    logger.info(f"Loading {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    n_before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Dropped {n_before - len(df):,} exact duplicates → {len(df):,} rows")

    # Drop rows missing node-identity or port columns
    required = NODE_ID_COLS + PORT_COLS
    n_before = len(df)
    df = df.dropna(subset=required)
    logger.info(f"Dropped {n_before - len(df):,} rows with missing {required}")

    # Fill throughput NaN with 0 before inf replacement so they stay 0
    for col in THROUGHPUT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Replace inf/-inf with NaN, then impute numerics with median
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            med = df[col].median()
            df[col] = df[col].fillna(med)
            logger.debug(f"Imputed {col}: {n_nan} NaN → median {med:.4g}")

    # Enforce numeric dtype
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["IPV4_SRC_ADDR"] = df["IPV4_SRC_ADDR"].astype(str)
    df["IPV4_DST_ADDR"] = df["IPV4_DST_ADDR"].astype(str)

    logger.info(f"Final shape after cleaning: {df.shape}")
    return df
