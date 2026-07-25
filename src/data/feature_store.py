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

        # O(log n) global-EID → local-position lookup via binary search
        # over this sorted array — see _pos_of/_pos_of_batch below. No
        # auxiliary structure is built (this is the point of the change).
        # Sortedness (strictly ascending, no duplicates) is guaranteed by
        # preprocessor.py's sort-then-arange-then-order-preserving-mask
        # pipeline (specs/12 §2), but that guarantee lives in a different
        # module this class does not control — assert it here so a future
        # violation (e.g. a corrupted file or a new split scheme) fails
        # loudly at construction time instead of corrupting lookups
        # silently, per CODING STANDARDS #7 ("assert invariants ... in
        # production code, not only in tests"). One transient boolean array
        # of length n-1, freed immediately after the check.
        assert np.all(self._edge_indices[1:] > self._edge_indices[:-1]), (
            "edge_indices.npy must be strictly ascending with no duplicates "
            "— required for the searchsorted-based EID lookup in _pos_of/"
            "_pos_of_batch to be correct"
        )
        self._eid_start = int(self._edge_indices[0])
        self._eid_end   = int(self._edge_indices[-1])

        logger.info(
            f"FeatureStore opened: {n:,} edges, d_e={d}, "
            f"EIDs [{self._eid_start}, {self._eid_end}]"
        )

    def _pos_of(self, global_eid: int) -> int:
        """Return this split's local row position for a global EID.

        O(log n) binary search over the sorted ``edge_indices`` array — no
        auxiliary structure is built or consulted. Safe because every split's
        ``edge_indices.npy`` is sorted ascending and duplicate-free by
        construction (see specs/12 §2); this method only relies on sortedness,
        not on the stronger contiguous-range property.

        Args:
            global_eid: global EID to look up.

        Returns:
            Local row position into ``self._mmap`` / ``self._labels``.

        Raises:
            KeyError: if ``global_eid`` is not present in this split's
                ``edge_indices``. Every legitimate caller derives its EID from
                this split's own DGL graph (``g_split.edata[dgl.EID]``), which
                by CRITICAL INVARIANT 1 is always present here — a KeyError
                signals programmer error (wrong split's store), not a data
                condition to handle gracefully.
        """
        eid = int(global_eid)
        pos = int(np.searchsorted(self._edge_indices, eid))
        if pos == len(self._edge_indices) or int(self._edge_indices[pos]) != eid:
            raise KeyError(f"Global EID {eid} not in this split")
        return pos

    def _pos_of_batch(self, global_eids: np.ndarray) -> np.ndarray:
        """Vectorized batch equivalent of ``_pos_of`` — O(k log n) total, no
        per-element Python-level loop.

        Args:
            global_eids: numpy array (or list/tuple) of global EIDs to look
                up, any order. Must not be a GPU-resident tensor — call
                .numpy() first, matching every existing caller in
                src/model/trainer.py, src/model/evaluator.py,
                src/model/temporal_sampler.py, and src/explainer/shap_gsd.py.

        Returns:
            int64 array of local row positions, same length and order as
            ``global_eids``.

        Raises:
            KeyError: if any EID in ``global_eids`` is not present in this
                split's ``edge_indices``. Reports the first offending EID (by
                position in the input array) for a debuggable message.
        """
        eids = np.asarray(global_eids, dtype=np.int64)
        n = len(self._edge_indices)
        positions = np.searchsorted(self._edge_indices, eids)
        out_of_bounds = positions == n
        # Clamp before indexing so the bounds check below never IndexErrors —
        # this is the vectorized form of the same pos == n trap as _pos_of.
        safe_positions = np.where(out_of_bounds, 0, positions)
        mismatch = out_of_bounds | (self._edge_indices[safe_positions] != eids)
        if mismatch.any():
            bad_idx = int(np.argmax(mismatch))
            raise KeyError(
                f"Global EID {int(eids[bad_idx])} not in this split "
                f"(batch input position {bad_idx})"
            )
        return positions

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
        """Return feature vector for a single edge by global EID.

        Raises:
            KeyError: if ``global_eid`` is not present in this split.
        """
        return self._mmap[self._pos_of(global_eid)]

    def get_batch(self, global_eids: np.ndarray) -> np.ndarray:
        """Return feature matrix for a batch of global EIDs, shape (k, d_e).

        Args:
            global_eids: numpy array (or list/tuple) of global EIDs. Must not
                be a GPU-resident tensor — call .numpy() first, matching
                every existing caller in src/model/trainer.py,
                src/model/evaluator.py, src/model/temporal_sampler.py, and
                src/explainer/shap_gsd.py.

        Raises:
            KeyError: if any EID in ``global_eids`` is not present in this
                split.
        """
        return self._mmap[self._pos_of_batch(global_eids)]

    def get_labels_batch(self, global_eids: np.ndarray) -> np.ndarray:
        """Return label array for a batch of global EIDs, shape (k,).

        Args:
            global_eids: numpy array (or list/tuple) of global EIDs. Must not
                be a GPU-resident tensor — call .numpy() first, matching
                every existing caller in src/model/trainer.py,
                src/model/evaluator.py, src/model/temporal_sampler.py, and
                src/explainer/shap_gsd.py.

        Raises:
            KeyError: if any EID in ``global_eids`` is not present in this
                split.
        """
        return self._labels[self._pos_of_batch(global_eids)]


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


# Column contract for edges_meta.parquet — the raw columns Phase 2 needs, aligned
# row-for-row with edge_indices.npy so Phase 2 never re-loads/re-sorts the CSV.
_EDGES_META_COLS: tuple[str, ...] = ("src_ip", "dst_ip", "in_bytes", "out_bytes", "dst_port")


def write_edges_meta(
    split_dir: Path | str,
    meta: dict[str, np.ndarray],
    n_expected: int,
) -> None:
    """Write edges_meta.parquet: raw columns Phase 2 needs, aligned to edge_indices.

    Row ``i`` corresponds to ``edge_indices[i]`` by construction — both are
    produced from the same sorted sub-dataframe in Phase 1 (see
    ``preprocessor._pack``). This eliminates Phase 2's independent CSV reload and
    re-sort, making EID alignment hold by construction rather than by coincidence.

    Args:
        split_dir: output directory (created if absent); sits beside features.dat.
        meta: dict with keys ``src_ip``, ``dst_ip`` (str), ``in_bytes``,
            ``out_bytes`` (float32), ``dst_port`` (int32). ``in_bytes``/``out_bytes``
            are float32 — NOT int64 — because loader.py median-imputes to float64
            and today's Phase 2 casts to float32; int64 would truncate.
        n_expected: number of edges in this split (typically
            ``len(data["edge_indices"])``). Asserted against the meta length so
            the alignment contract is self-checking, mirroring
            ``write_feature_store``'s length assertion.

    Raises:
        AssertionError: if the meta arrays are not mutually equal-length, or their
            length does not match ``n_expected``.
    """
    import pandas as pd  # local import; pandas is already a hard dependency

    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    lengths = {k: len(meta[k]) for k in _EDGES_META_COLS}
    assert len(set(lengths.values())) == 1, f"edges_meta length mismatch: {lengths}"
    assert lengths["src_ip"] == n_expected, (
        f"edges_meta length {lengths['src_ip']} != edge count {n_expected}"
    )

    df = pd.DataFrame({
        "src_ip":    np.asarray(meta["src_ip"],    dtype=object),
        "dst_ip":    np.asarray(meta["dst_ip"],    dtype=object),
        "in_bytes":  np.asarray(meta["in_bytes"],  dtype=np.float32),
        "out_bytes": np.asarray(meta["out_bytes"], dtype=np.float32),
        "dst_port":  np.asarray(meta["dst_port"],  dtype=np.int32),
    })
    df.to_parquet(split_dir / "edges_meta.parquet", engine="pyarrow", index=False)
    logger.info(f"Wrote edges_meta.parquet: {split_dir} ({n_expected:,} rows)")


def read_edges_meta(split_dir: Path | str) -> dict[str, np.ndarray]:
    """Read edges_meta.parquet back as a dict of numpy arrays (dtypes preserved).

    Dtypes match what Phase 2 expects at its call sites (so any redundant
    ``.astype`` on the caller side is a no-op): ``src_ip``/``dst_ip`` as object
    arrays of str (matching today's ``df["IPV4_SRC_ADDR"].values``),
    ``in_bytes``/``out_bytes`` as float32, ``dst_port`` as int32.

    Args:
        split_dir: directory containing edges_meta.parquet.

    Returns:
        Dict with keys ``src_ip``, ``dst_ip``, ``in_bytes``, ``out_bytes``,
        ``dst_port``.
    """
    import pandas as pd  # local import; pandas is already a hard dependency

    split_dir = Path(split_dir)
    df = pd.read_parquet(split_dir / "edges_meta.parquet", engine="pyarrow")
    return {
        "src_ip":    df["src_ip"].astype(str).to_numpy(dtype=object),
        "dst_ip":    df["dst_ip"].astype(str).to_numpy(dtype=object),
        "in_bytes":  df["in_bytes"].to_numpy(dtype=np.float32),
        "out_bytes": df["out_bytes"].to_numpy(dtype=np.float32),
        "dst_port":  df["dst_port"].to_numpy(dtype=np.int32),
    }
