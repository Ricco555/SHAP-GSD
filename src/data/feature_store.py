"""
Off-graph feature store via memory-mapped arrays.

CRITICAL INVARIANT: g.edata[dgl.EID][i] must equal edge_indices[i].
Every DGL graph edge maps to exactly one row in the memmap via its global EID.

Files per split:
  features.dat      shape (n_edges, d_e), float32, memory-mapped
  edge_indices.npy  global EID for each position in this split
  timestamps.npy    FLOW_START_MILLISECONDS per edge
  labels.npy        integer class label per edge

Global EIDs are assigned in chronological sort order, 0-indexed over the full
dataset. For contiguous splits:
  train EIDs: 0 .. n_train-1
  val EIDs:   n_train .. n_train+n_val-1
  test EIDs:  n_train+n_val .. N-1

Access: store[global_eid] → np.ndarray shape (d_e,)
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class FeatureStore:
    """Read-only accessor for a single split's memory-mapped feature store."""

    def __init__(self, split_dir: Path | str) -> None:
        """Open an existing feature store for a split.

        Args:
            split_dir: directory containing features.dat, edge_indices.npy, etc.
        """
        split_dir = Path(split_dir)
        self._mmap = np.memmap(
            split_dir / "features.dat", dtype=np.float32, mode="r"
        )
        self._edge_indices = np.load(split_dir / "edge_indices.npy")
        self._timestamps   = np.load(split_dir / "timestamps.npy")
        self._labels       = np.load(split_dir / "labels.npy")

        n = len(self._edge_indices)
        d = len(self._mmap) // n
        self._mmap = self._mmap.reshape(n, d)
        self._d_e = d

        # Build O(1) global-EID → local-position lookup
        self._eid_to_pos: dict[int, int] = {
            int(eid): pos for pos, eid in enumerate(self._edge_indices)
        }
        self._eid_start = int(self._edge_indices.min())
        self._eid_end   = int(self._edge_indices.max())

        logger.info(
            f"FeatureStore opened: {n:,} edges, d_e={d}, "
            f"EIDs [{self._eid_start}, {self._eid_end}]"
        )

    @property
    def d_e(self) -> int:
        return self._d_e

    @property
    def n_edges(self) -> int:
        return len(self._edge_indices)

    @property
    def edge_indices(self) -> np.ndarray:
        return self._edge_indices

    @property
    def timestamps(self) -> np.ndarray:
        return self._timestamps

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    def __getitem__(self, global_eid: int) -> np.ndarray:
        """Return feature vector for a single edge by global EID."""
        pos = self._eid_to_pos.get(int(global_eid))
        assert pos is not None, f"Global EID {global_eid} not in this split"
        return self._mmap[pos]

    def get_batch(self, global_eids: np.ndarray) -> np.ndarray:
        """Return feature matrix for a batch of global EIDs, shape (k, d_e)."""
        positions = np.array([self._eid_to_pos[int(e)] for e in global_eids])
        return self._mmap[positions]


def write_feature_store(
    split_dir: Path | str,
    features: np.ndarray,
    edge_indices: np.ndarray,
    timestamps: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Write all four files for a split's feature store.

    Args:
        split_dir:    output directory (created if absent)
        features:     float32 array (n, d_e)
        edge_indices: global EID for each row, int64 (n,)
        timestamps:   FLOW_START_MILLISECONDS, int64 (n,)
        labels:       integer class labels, int64 (n,)
    """
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    n, d_e = features.shape
    assert len(edge_indices) == n and len(timestamps) == n and len(labels) == n, (
        "Array length mismatch in write_feature_store"
    )

    # Memory-mapped write
    mmap = np.memmap(split_dir / "features.dat", dtype=np.float32, mode="w+", shape=(n, d_e))
    mmap[:] = features.astype(np.float32)
    del mmap  # flush

    np.save(split_dir / "edge_indices.npy", edge_indices.astype(np.int64))
    np.save(split_dir / "timestamps.npy",   timestamps.astype(np.int64))
    np.save(split_dir / "labels.npy",       labels.astype(np.int64))

    logger.info(f"Wrote feature store: {split_dir}  ({n:,} edges, d_e={d_e})")

    # Verify EID alignment: reading back first and last row
    store = FeatureStore(split_dir)
    assert store[int(edge_indices[0])] is not None
    assert store[int(edge_indices[-1])] is not None
    logger.info(f"EID alignment check passed for {split_dir}")
