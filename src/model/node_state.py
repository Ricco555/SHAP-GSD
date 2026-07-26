"""
15-dimensional per-node state vector. NEW — TE-G-SAGE used constant ones.

NODE STATE SCHEMA:
  Behavioral (11):
  [0]  is_internal       binary  1 if private IP (10.x, 172.16-31.x, 192.168.x) or loopback
                                 DATASET NOTE (CORRECTED): NF-UNSW-NB15-v3 has MIXED IPs.
                                 44 nodes total: 8 RFC1918/loopback (is_internal=1),
                                 36 public/multicast (is_internal=0). Earlier note claiming
                                 all nodes are public was incorrect. Run
                                 scripts/12_novelty_audit.py for full dim-0/dim-1 breakdown.
  [1]  novelty           binary  1 if first seen in window W
  [2]  recency           float   normalized time since last flow [0=just seen, 1=not seen in W]
  [3]  rolling_in_degree  int    incoming edges in W
  [4]  rolling_out_degree int    outgoing edges in W
  [5]  unique_dst_ip_count int   distinct dst IPs sent to in W (fan-out breadth)
  [6]  unique_src_ip_count int   distinct src IPs received from in W
  [7]  unique_dst_port_count int distinct dst ports in outgoing flows in W
  [8]  dst_port_entropy  float   Shannon entropy of dst port dist in W
                                 low=targeted, high=scanning
  [9]  rolling_in_bytes  float   log(1 + sum IN_BYTES) in W
  [10] rolling_out_bytes float   log(1 + sum OUT_BYTES) in W

  Seasonal (4):
  [11] time_sin   float  sin(2π × hour/24) from FLOW_START_MILLISECONDS
  [12] time_cos   float  cos(2π × hour/24) — avoids midnight discontinuity
  [13] volume_deviation float  current volume vs node's historical hourly mean
                               baseline built from training data only
  [14] iat_regularity   float  CV (std/mean) of IATs in W
                               low=machine cadence, high=human irregular

WINDOW: W seconds (default 60). Current time = target edge timestamp.

SEASONALITY:
  hour = (timestamp_ms / 3600000) % 24
  time_sin/cos: same for all nodes in same batch (time context).
  volume_deviation: per-node, per-hour baseline from training only.
    baseline[v][h] = mean(log(1+bytes)) for node v in hour h across training.
    Fallback to global mean for unseen node/hour combinations.
  iat_regularity: from IATs of flows for this node in W.
    Set to 0.0 if < 2 flows in window.

DATASET NOTE: UNSW-NB15 covers ~2 days. ~2 samples per hour-bucket per node.
volume_deviation will have high variance — acknowledged in paper.

SNAPSHOTS (OFF by default — see NodeStateManager docstring): optionally
pre-compute node state every snapshot_interval edges in temporal order.
Mini-batch looks up nearest snapshot <= t_e, applies delta updates. Disabled
by default (snapshot_interval=0) because nothing in the runtime query path
(get_state_at_time / get_batch_states / rollback_edge(s), all below) reads
these snapshots — they all recompute from _histories/_baselines directly.
The only consumer anywhere in the codebase is the optional exploration
figure script explore/graph/topology_panel.py, which already degrades
gracefully (no snapshot overlay) when snapshots.pkl is empty.

EMPIRICAL VALIDATION (2026-07-25, HPC, NF-CSE-CIC-IDS2018-v3, 19.5M edges /
205,801 nodes): the snapshot cache was confirmed to be both the dominant
memory cost AND the dominant runtime cost of Phase 2. Before this fix:
OOM-killed after ~2h wall-clock at ~95.8 GiB peak (96 GB cgroup limit,
exit -9). After: completed in 00:03:55 at ~9.5 GiB peak — a ~10x memory
reduction and a ~30x+ wall-clock reduction. Most of the pre-fix 2h was
spent computing per-node state for every retained snapshot, not building
edge histories.

ROLLBACK (for temporal SHAP coalitions):
When masking neighbor edge e' from node u:
  - incoming: decrement rolling_in_degree, update unique_src_ip_count
  - outgoing: decrement rolling_out_degree, update unique_dst_ip/port counts,
    recalculate dst_port_entropy, update rolling bytes
  - if e' was FIRST edge for u in W: set novelty=0
  - recalculate recency from next-most-recent edge
  - recalculate iat_regularity from remaining IATs
  - time_sin, time_cos, is_internal: NOT rolled back (not edge-dependent)
  - volume_deviation: recompute from updated rolling bytes

Pre-compute per-node sorted edge lists for O(log n) rollback.
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import entropy as scipy_entropy

from src.data.preprocessor import N_DST_PORT_BINS, port_to_bin_indices
from src.utils.memory import _maxrss_mb

logger = logging.getLogger(__name__)


class _NodeHistory:
    """Sorted-by-timestamp record of all edges for one node.

    Built in two phases:
      1. add() calls during graph construction (O(1) amortized)
      2. finalize() converts lists → sorted numpy arrays for O(log n) search
    """

    __slots__ = (
        "first_seen_ms",
        "_ts_list", "_inc_list", "_bytes_list", "_port_list", "_peer_list",
        "ts", "is_incoming", "raw_bytes", "dst_port", "peer_id",
        "_finalized",
    )

    def __init__(self) -> None:
        self.first_seen_ms: int = -1
        # Accumulation buffers (pre-finalize)
        self._ts_list:    list[int]   = []
        self._inc_list:   list[bool]  = []
        self._bytes_list: list[float] = []
        self._port_list:  list[int]   = []
        self._peer_list:  list[int]   = []
        # Sorted numpy arrays (post-finalize)
        self.ts:          Optional[np.ndarray] = None  # int64, ascending
        self.is_incoming: Optional[np.ndarray] = None  # bool
        self.raw_bytes:   Optional[np.ndarray] = None  # float32
        self.dst_port:    Optional[np.ndarray] = None  # int32 — semantic BIN INDEX (0-15), not raw port
        self.peer_id:     Optional[np.ndarray] = None  # int32
        self._finalized = False

    def add(
        self,
        timestamp_ms: int,
        is_incoming: bool,
        bytes_val: float,
        dst_port: int,
        peer_id: int,
    ) -> None:
        """Append one edge record. Call finalize() before any queries."""
        self._ts_list.append(timestamp_ms)
        self._inc_list.append(is_incoming)
        self._bytes_list.append(bytes_val)
        self._port_list.append(dst_port)
        self._peer_list.append(peer_id)
        if self.first_seen_ms < 0 or timestamp_ms < self.first_seen_ms:
            self.first_seen_ms = timestamp_ms

    def finalize(self) -> None:
        """Sort accumulated records by timestamp and convert to numpy arrays."""
        ts_arr = np.array(self._ts_list, dtype=np.int64)
        order = np.argsort(ts_arr, kind="stable")
        self.ts          = ts_arr[order]
        self.is_incoming = np.array(self._inc_list,   dtype=bool)[order]
        self.raw_bytes   = np.array(self._bytes_list,  dtype=np.float32)[order]
        self.dst_port    = np.array(self._port_list,   dtype=np.int32)[order]
        self.peer_id     = np.array(self._peer_list,   dtype=np.int32)[order]
        # Release accumulation buffers
        del self._ts_list, self._inc_list, self._bytes_list, self._port_list, self._peer_list
        self._finalized = True

    def window_bounds(self, t_ms: float, W_ms: float) -> tuple[int, int]:
        """Return (left, right) array indices for records in [t-W, t] (inclusive).

        Uses np.searchsorted for O(log n) binary search on sorted timestamps.
        """
        assert self._finalized, "Call finalize() before querying"
        lo = int(t_ms - W_ms)
        hi = int(t_ms)
        left  = int(np.searchsorted(self.ts, lo, side="left"))
        right = int(np.searchsorted(self.ts, hi, side="right"))
        return left, right


class NodeStateManager:
    """Compute 15-dim per-node temporal state vectors.

    State is computed from a rolling window W around the query time.
    Seasonal features use per-node per-hour baselines from training data only.

    Periodic snapshotting (``build_snapshots``' Pass 2) is OFF by default
    (``snapshot_interval=0``/``None``). It exists as an optional, disabled-
    by-default capability: nothing on the real query path
    (``get_state_at_time``, ``get_batch_states``, ``rollback_edge(s)``) reads
    the snapshot dict — every one of those recomputes state on demand from
    ``_histories``/``_baselines``. Snapshotting was originally meant as a
    fast-lookup cache, but at Paper-3 scale (19-25M edges, up to ~205K nodes)
    it is an O(n_snapshots × n_nodes) memory/time sink (observed ~90-100GB
    for one dataset, projected 600-850GB for another) for a structure nothing
    reads. The ONLY consumer anywhere in the codebase is the optional
    exploration figure script ``explore/graph/topology_panel.py``, which
    already degrades gracefully (no node-state overlay on its plots) when
    ``snapshots.pkl`` holds the empty ``{"times": [], "states": []}`` shape.
    Set ``snapshot_interval`` to a positive int (per-experiment config or
    directly) to re-enable it for a given run — Pass 2 behaves exactly as it
    always has when enabled.

    Typical workflow::

        nsm = NodeStateManager(window_seconds=60, snapshot_interval=0)
        nsm.set_is_internal(is_internal_arr)          # from GraphBuilder
        nsm.build_hourly_baselines(train_src, train_dst, train_ts, in_b, out_b)
        nsm.build_snapshots(all_src, all_dst, all_ts, in_b, out_b, dst_ports)
        state = nsm.get_state_at_time(node_id, timestamp_ms)
        batch = nsm.get_batch_states(node_ids, timestamp_ms)
    """

    NODE_STATE_DIM: int = 15

    def __init__(
        self,
        window_seconds: float = 60.0,
        snapshot_interval: Optional[int] = 0,
    ) -> None:
        """
        Args:
            window_seconds:    rolling window width W in seconds (default 60).
            snapshot_interval: edges between periodic state snapshots. Default
                                0 (disabled — see class docstring for why: no
                                runtime query path reads snapshots). ``None``
                                is also treated as disabled. Set to a positive
                                int to re-enable snapshot generation.
        """
        self.window_seconds    = window_seconds
        self._W_ms: float      = window_seconds * 1000.0
        self.snapshot_interval = snapshot_interval

        # Static is_internal array indexed by node_id (set externally)
        self._is_internal: Optional[np.ndarray] = None

        # Per-node edge history (built by build_snapshots)
        self._histories: dict[int, _NodeHistory] = {}

        # Hourly volume baselines from training: {node_id: {hour: mean_log_bytes}}
        self._baselines: dict[int, dict[int, float]] = {}
        self._global_baseline: float = 0.0

        # Periodic snapshots for fast lookup: parallel lists
        # _snap_times[i]  = timestamp_ms of snapshot i
        # _snap_states[i] = {node_id: np.ndarray(15,)} at that time
        self._snap_times:  list[int]                      = []
        self._snap_states: list[dict[int, np.ndarray]]   = []

        # Lazily-built flat CSR-style index for vectorized get_batch_states
        # (built by _ensure_flat_index on first batch call). None until built.
        # _histories is frozen post-build/load, so no invalidation is ever needed.
        self._flat_index: Optional[dict] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_is_internal(self, is_internal: np.ndarray) -> None:
        """Set the static is_internal feature array indexed by node_id.

        Typically computed via GraphBuilder.compute_is_internal_array().
        """
        self._is_internal = is_internal.astype(np.float32)

    def build_hourly_baselines(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        timestamps_ms: np.ndarray,
        in_bytes: np.ndarray,
        out_bytes: np.ndarray,
    ) -> None:
        """Compute per-node per-hour volume baselines from training data only.

        baseline[v][h] = mean(log(1 + bytes)) for all edges involving node v
        in hour h. Each directed edge contributes:
          - log(1 + out_bytes) to the source node's accumulator
          - log(1 + in_bytes)  to the destination node's accumulator

        Must be called BEFORE build_snapshots. Uses training data only —
        never pass val/test data here.

        Args:
            src_node_ids:  integer node IDs for flow sources (n_train,).
            dst_node_ids:  integer node IDs for flow destinations (n_train,).
            timestamps_ms: FLOW_START_MILLISECONDS (n_train,), used to derive hour.
            in_bytes:      raw IN_BYTES per flow (n_train,).
            out_bytes:     raw OUT_BYTES per flow (n_train,).
        """
        t0 = time.monotonic()
        logger.info(
            f"Building hourly baselines from {len(src_node_ids):,} training edges "
            f"(maxrss={_maxrss_mb():.0f} MiB)"
        )

        # accumulator[node_id][hour] = list of log(1+bytes) values
        accumulator: dict[int, dict[int, list[float]]] = {}

        for i in range(len(timestamps_ms)):
            hour = int((timestamps_ms[i] / 3_600_000) % 24)

            src = int(src_node_ids[i])
            log_out = float(np.log1p(max(0.0, float(out_bytes[i]))))
            if src not in accumulator:
                accumulator[src] = {}
            if hour not in accumulator[src]:
                accumulator[src][hour] = []
            accumulator[src][hour].append(log_out)

            dst = int(dst_node_ids[i])
            log_in = float(np.log1p(max(0.0, float(in_bytes[i]))))
            if dst not in accumulator:
                accumulator[dst] = {}
            if hour not in accumulator[dst]:
                accumulator[dst][hour] = []
            accumulator[dst][hour].append(log_in)

        self._baselines = {
            node: {h: float(np.mean(vals)) for h, vals in hours.items()}
            for node, hours in accumulator.items()
        }

        all_vals: list[float] = [
            v
            for hours in accumulator.values()
            for vals in hours.values()
            for v in vals
        ]
        self._global_baseline = float(np.mean(all_vals)) if all_vals else 0.0

        logger.info(
            f"Hourly baselines built: {len(self._baselines):,} nodes, "
            f"global fallback={self._global_baseline:.4f}, "
            f"elapsed={time.monotonic() - t0:.1f}s, maxrss={_maxrss_mb():.0f} MiB"
        )

    def build_snapshots(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        timestamps_ms: np.ndarray,
        in_bytes: np.ndarray,
        out_bytes: np.ndarray,
        dst_ports: np.ndarray,
        snapshot_interval: Optional[int] = None,
    ) -> None:
        """Build per-node edge histories and (optionally) periodic state snapshots.

        Processes ALL edges (train + val + test) in temporal order so that
        val/test node states correctly reflect prior training history.
        Call build_hourly_baselines() first.

        Pass 1 (edge histories, below) always runs — it populates
        ``_histories``, which every real state query
        (``get_state_at_time``/``get_batch_states``/``rollback_edge(s)``) reads.

        Pass 2 (periodic snapshots) is SKIPPED when ``interval`` resolves to 0
        or None (the default — see class docstring). This is a pure
        performance/memory optimization: nothing on the query path reads
        ``_snap_times``/``_snap_states``, so skipping Pass 2 changes no
        computed value, only whether the (unused-by-runtime) snapshot cache
        gets built. At Paper-3 scale (19-25M edges) Pass 2 was an
        O(n_snapshots × n_nodes) sink of both time and memory.

        Note: extends the spec's (graph, timestamps, snapshot_interval)
        signature to include the raw byte and port data required for
        behavioral feature computation.

        Args:
            src_node_ids:     integer node IDs for flow sources (N,).
            dst_node_ids:     integer node IDs for flow destinations (N,).
            timestamps_ms:    FLOW_START_MILLISECONDS, sorted ascending (N,).
            in_bytes:         raw IN_BYTES per flow (N,).
            out_bytes:        raw OUT_BYTES per flow (N,).
            dst_ports:        L4_DST_PORT per flow (N,) — raw values; converted
                              to semantic bin indices (0-15) internally so that
                              dst_port_entropy reflects service diversity, not
                              raw port diversity.
            snapshot_interval: edges between snapshots; defaults to
                              self.snapshot_interval. 0/None disables Pass 2.
        """
        interval = snapshot_interval if snapshot_interval is not None else self.snapshot_interval
        n = len(src_node_ids)
        logger.info(
            f"Building node histories: {n:,} edges, interval={interval or 'disabled'} "
            f"(maxrss={_maxrss_mb():.0f} MiB)"
        )

        # Pass 1: populate per-node edge lists.
        # Convert raw L4_DST_PORT values to semantic bin indices (0-15) ONCE
        # before the loop so that dst_port_entropy and unique_dst_port_count
        # operate over the 16-service taxonomy, not raw port numbers.
        t0 = time.monotonic()
        bin_indices = port_to_bin_indices(np.asarray(dst_ports))

        self._histories.clear()
        for i in range(n):
            src = int(src_node_ids[i])
            dst = int(dst_node_ids[i])
            ts  = int(timestamps_ms[i])
            ib  = float(in_bytes[i])
            ob  = float(out_bytes[i])
            db  = int(bin_indices[i])   # semantic bin index (0-15)

            if src not in self._histories:
                self._histories[src] = _NodeHistory()
            if dst not in self._histories:
                self._histories[dst] = _NodeHistory()

            # Outgoing record for source node — stores bin index, not raw port
            self._histories[src].add(
                timestamp_ms=ts, is_incoming=False,
                bytes_val=ob, dst_port=db, peer_id=dst,
            )
            # Incoming record for destination node — dst_port unused for in-edges
            self._histories[dst].add(
                timestamp_ms=ts, is_incoming=True,
                bytes_val=ib, dst_port=0, peer_id=src,
            )

        for hist in self._histories.values():
            hist.finalize()

        logger.info(
            f"Edge histories finalized for {len(self._histories):,} nodes "
            f"(pass1 elapsed={time.monotonic() - t0:.1f}s, maxrss={_maxrss_mb():.0f} MiB)"
        )

        # Pass 2: take periodic state snapshots (OFF by default — see docstring)
        self._snap_times.clear()
        self._snap_states.clear()

        if not interval:
            logger.info(
                "Snapshot generation disabled (snapshot_interval=0/None) — "
                "skipping Pass 2; _snap_times/_snap_states left empty."
            )
            return

        t1 = time.monotonic()
        for step in range(interval - 1, n, interval):
            t_ms = int(timestamps_ms[step])
            # Only snapshot nodes that have been seen up to this point
            snap: dict[int, np.ndarray] = {}
            for nid, hist in self._histories.items():
                if 0 <= hist.first_seen_ms <= t_ms:
                    snap[nid] = self._compute_state(nid, float(t_ms))
            self._snap_times.append(t_ms)
            self._snap_states.append(snap)

        logger.info(
            f"Snapshots taken: {len(self._snap_times):,} "
            f"(pass2 elapsed={time.monotonic() - t1:.1f}s, maxrss={_maxrss_mb():.0f} MiB)"
        )

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_state_at_time(self, node_id: int, time_ms: float) -> np.ndarray:
        """Return 15-dim state vector for node_id at time_ms.

        Args:
            node_id: integer node ID from node_id_map.
            time_ms: query timestamp in milliseconds.

        Returns:
            np.ndarray of shape (15,), dtype float32.
        """
        return self._compute_state(int(node_id), float(time_ms))

    def get_batch_states(
        self,
        node_ids: list[int] | np.ndarray,
        time_ms: float,
    ) -> np.ndarray:
        """Return (N, 15) state matrix for a batch of node IDs at a shared time.

        All nodes use the same query timestamp, which is valid because
        mini-batches are sorted by timestamp (temporally close edges).

        Args:
            node_ids: sequence of integer node IDs, length N.
            time_ms:  query timestamp in milliseconds.

        Returns:
            np.ndarray of shape (N, 15), dtype float32.

        Implementation note: the body now builds (once, lazily) an instance-
        cached flat CSR-style index of the already-sorted per-node histories
        and computes all 15 dims via batched NumPy segment reductions
        (``_batch_states_vectorized``). ``_compute_state`` remains the untouched
        parity oracle — bit-exact on the integer/shared-scalar dims and
        allclose (atol 1e-5) on the float64-accumulation dims.
        """
        t = float(time_ms)
        ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
        return self._batch_states_vectorized(ids, t)

    def _ensure_flat_index(self) -> None:
        """Build the flat CSR-style history index and dense per-node arrays once,
        cached on the instance. Idempotent: returns immediately if already built.

        ``_histories`` is frozen after build_snapshots()/load(), so no
        invalidation is needed (unlike the sampler's per-graph ``_csc_cache``).
        A single ``None``-sentinel handle suffices — no keying, no identity
        re-check, no eviction.
        """
        if self._flat_index is not None:
            return

        # --- Node-domain sizing (covers all dense arrays) ---
        n_hist  = (max(self._histories) + 1) if self._histories else 0
        n_base  = (max(self._baselines) + 1) if self._baselines else 0
        n_isint = len(self._is_internal) if self._is_internal is not None else 0
        num_nodes = max(n_hist, n_base, n_isint)

        # --- Flat concatenated history arrays (CSR values) + indptr ---
        seg_len = np.zeros(num_nodes, dtype=np.int64)
        ts_parts:   list[np.ndarray] = []
        inc_parts:  list[np.ndarray] = []
        byt_parts:  list[np.ndarray] = []
        port_parts: list[np.ndarray] = []
        peer_parts: list[np.ndarray] = []
        first_seen_arr = np.full(num_nodes, -1, dtype=np.int64)

        # Iterate nodes in ascending id so segments are contiguous.
        for v in range(num_nodes):
            h = self._histories.get(v)
            if h is None:
                continue
            seg_len[v] = len(h.ts)
            ts_parts.append(h.ts)
            inc_parts.append(h.is_incoming)
            byt_parts.append(h.raw_bytes)
            port_parts.append(h.dst_port)
            peer_parts.append(h.peer_id)
            first_seen_arr[v] = h.first_seen_ms

        if ts_parts:
            flat_ts   = np.concatenate(ts_parts).astype(np.int64, copy=False)
            flat_inc  = np.concatenate(inc_parts).astype(bool, copy=False)
            flat_bytes = np.concatenate(byt_parts).astype(np.float32, copy=False)
            flat_port = np.concatenate(port_parts).astype(np.int32, copy=False)
            flat_peer = np.concatenate(peer_parts).astype(np.int64)
        else:
            flat_ts   = np.empty(0, dtype=np.int64)
            flat_inc  = np.empty(0, dtype=bool)
            flat_bytes = np.empty(0, dtype=np.float32)
            flat_port = np.empty(0, dtype=np.int32)
            flat_peer = np.empty(0, dtype=np.int64)

        node_indptr = np.zeros(num_nodes + 1, dtype=np.int64)
        np.cumsum(seg_len, out=node_indptr[1:])

        # Production assert (CODING STANDARDS 7) — flat-length parity.
        assert (
            node_indptr[-1] == len(flat_ts) == len(flat_inc)
            == len(flat_bytes) == len(flat_port) == len(flat_peer)
        ), "flat CSR index length mismatch"

        # --- Composite key for the window-bounds global searchsorted (Finding 3,
        #     spec 23 §3; mirrors temporal_sampler Optimization B / spec 22 §2.2). ---
        if flat_ts.size:
            # Precondition asserts (CODING STANDARDS 7) — load-bearing, fail loud.
            assert int(flat_ts.min()) >= 0, (
                "flat_ts has negative timestamps; the composite window-bounds key "
                "would let a low-node segment's slot collide into another node's "
                "key range (FLOW_START_MILLISECONDS is epoch-ms, always >= 0)."
            )
            max_ts = int(flat_ts.max())
            big = max_ts + 1                       # tight, not arbitrary
            # Overflow bound: the LEFT key reaches lo_c + (num_nodes-1)*big with the
            # upper lo-clamp lo_c == big (§3.2), i.e. num_nodes*big. Guard THAT max
            # key (NOT Opt-B's (n-1)*big+max_ts form, which is one below it and would
            # permit num_nodes*big == 2**63 → silent int64 wrap).
            assert num_nodes * big < 2**63, (
                "composite window-bounds key overflow: num_nodes*big must stay in "
                "int64 — fail loudly rather than silently wrap on a larger graph."
            )
            node_of_slot = np.repeat(np.arange(num_nodes, dtype=np.int64), seg_len)
            composite = flat_ts.astype(np.int64) + node_of_slot * big
            # Belt-and-braces: the searchsorted precondition itself.
            if composite.size > 1:
                assert bool(np.all(np.diff(composite) >= 0)), (
                    "composite window-bounds key not globally non-decreasing; the "
                    "segmented searchsorted identity would be invalid."
                )
        else:
            big = 1
            composite = np.empty(0, dtype=np.int64)

        # --- Dense per-node arrays ---
        is_internal_arr = np.zeros(num_nodes, dtype=np.float32)
        if self._is_internal is not None:
            is_internal_arr[: len(self._is_internal)] = self._is_internal

        baseline_arr = np.full(
            (num_nodes, 24), np.float32(self._global_baseline), dtype=np.float32
        )
        for node, hours in self._baselines.items():
            if 0 <= node < num_nodes:
                for hour, val in hours.items():
                    baseline_arr[node, hour] = val

        self._flat_index = {
            "num_nodes":      num_nodes,
            "node_indptr":    node_indptr,
            "flat_ts":        flat_ts,
            "flat_inc":       flat_inc,
            "flat_bytes":     flat_bytes,
            "flat_port":      flat_port,
            "flat_peer":      flat_peer,
            "first_seen_arr": first_seen_arr,
            "is_internal_arr": is_internal_arr,
            "baseline_arr":   baseline_arr,
            "composite":      composite,
            "big":            big,
        }

    def _batch_states_vectorized(
        self,
        node_ids: np.ndarray,   # int64, shape (N,), order-significant, dups allowed
        t: float,
    ) -> np.ndarray:            # (N, 15) float32
        """Vectorized equivalent of stacking _compute_state over node_ids at t.

        Builds (once, cached) the flat CSR-style index, then computes all 15
        dims via batched NumPy segment reductions. Bit-parity with the scalar
        oracle on the integer/exact dims; allclose (atol 1e-5) on the
        float64-accumulation dims.
        """
        self._ensure_flat_index()
        fi = self._flat_index

        num_nodes       = fi["num_nodes"]
        node_indptr     = fi["node_indptr"]
        flat_ts         = fi["flat_ts"]
        flat_inc        = fi["flat_inc"]
        flat_bytes      = fi["flat_bytes"]
        flat_port       = fi["flat_port"]
        flat_peer       = fi["flat_peer"]
        first_seen_arr  = fi["first_seen_arr"]
        is_internal_arr = fi["is_internal_arr"]
        baseline_arr    = fi["baseline_arr"]
        composite       = fi["composite"]
        big             = fi["big"]

        N = int(node_ids.shape[0])
        state = np.zeros((N, 15), dtype=np.float32)

        # Defaults for reduction-fed dims (so empty batch / empty windows
        # return the correct no-history vector without special-casing).
        state[:, 2] = 1.0   # recency defaults to 1.0 (not seen in W)

        # --- Setup: validity mask ---
        valid = (node_ids >= 0) & (node_ids < num_nodes)
        safe_ids = np.where(valid, node_ids, 0)   # clamp for safe gathers

        # Dim 0 — is_internal (bit-exact); zero-padded array reproduces the
        # scalar node_id < len(is_internal) guard.
        if num_nodes > 0:
            state[:, 0] = np.where(
                valid, is_internal_arr[safe_ids], np.float32(0.0)
            )

        # Dim 1 — novelty (HIGHEST PARITY RISK). Float window start, NOT int(lo),
        # AND the first_seen >= 0 sentinel exclusion AND valid.
        w_start = t - self._W_ms
        fs = (
            first_seen_arr[safe_ids] if num_nodes > 0
            else np.full(N, -1, dtype=np.int64)
        )
        state[:, 1] = (
            (fs >= 0) & (w_start <= fs) & (fs <= t) & valid
        ).astype(np.float32)

        # Dims 11/12 — seasonal, shared scalar t (bit-exact).
        hour_frac = (t / 3_600_000.0) % 24.0
        state[:, 11] = np.float32(np.sin(2.0 * np.pi * hour_frac / 24.0))
        state[:, 12] = np.float32(np.cos(2.0 * np.pi * hour_frac / 24.0))
        hour_int = int(hour_frac)

        # --- Window bounds (shared int-truncated scalar, per window_bounds) ---
        lo = int(t - self._W_ms)        # int-truncated, matches _NodeHistory.window_bounds
        hi = int(t)                     # int-truncated, matches _NodeHistory.window_bounds
        if flat_ts.size:
            # Clamp query bounds into the composite-key's valid domain. flat_ts >= 0
            # (asserted at build), so these clamps preserve the per-segment counts
            # EXACTLY (see §3.3 correctness):
            #   lo -> [0, big]      (UPPER clamp is `big`, NOT big-1 — see the trap
            #                        note in §3.3: big-1 undercounts when lo > max_ts
            #                        and a segment holds ts == max_ts).
            #   hi -> [-1, big-1]
            lo_c = min(max(lo, 0), big)
            hi_c = min(max(hi, -1), big - 1)
            win_lo = np.searchsorted(composite, lo_c + safe_ids * big, side="left")
            win_hi = np.searchsorted(composite, hi_c + safe_ids * big, side="right")
            win_len = win_hi - win_lo
        else:
            win_lo  = np.zeros(N, dtype=np.int64)
            win_len = np.zeros(N, dtype=np.int64)
        win_len[~valid] = 0             # invalid ids -> empty window (unchanged intent)

        # Dim 2 — recency (bit-exact): last element of each non-empty window.
        nz = win_len > 0
        if nz.any():
            last_ts = flat_ts[win_lo[nz] + win_len[nz] - 1].astype(np.float64)
            state[nz, 2] = np.minimum(
                1.0, (t - last_ts) / self._W_ms
            ).astype(np.float32)

        # --- Loop-free ragged gather over in-window records ---
        total = int(win_len.sum())
        if total > 0:
            seg_id = np.repeat(np.arange(N), win_len)          # (total,) batch row
            ends = np.cumsum(win_len)
            starts = ends - win_len
            ramp = np.arange(total) - np.repeat(starts, win_len)
            gather_idx = np.repeat(win_lo, win_len) + ramp     # abs flat offset

            g_ts   = flat_ts[gather_idx]
            g_inc  = flat_inc[gather_idx]
            g_byt  = flat_bytes[gather_idx]
            g_port = flat_port[gather_idx]
            g_peer = flat_peer[gather_idx]

            out_m = ~g_inc
            in_m  = g_inc

            # Dims 3, 4 — in/out degree (bit-exact).
            state[:, 3] = np.bincount(seg_id[in_m],  minlength=N).astype(np.float32)
            state[:, 4] = np.bincount(seg_id[out_m], minlength=N).astype(np.float32)

            # Dims 5, 6 — unique dst/src IP count (composite-key unique).
            max_peer = int(flat_peer.max()) if flat_peer.size else 0
            K = max_peer + 1
            assert (N - 1) * K + max_peer < 2**63, (
                "composite src/dst-unique key overflow"
            )
            if out_m.any():
                comp = seg_id[out_m].astype(np.int64) * K + g_peer[out_m]
                uc = np.unique(comp)
                state[:, 5] = np.bincount(
                    (uc // K).astype(np.int64), minlength=N
                ).astype(np.float32)
            if in_m.any():
                comp = seg_id[in_m].astype(np.int64) * K + g_peer[in_m]
                uc = np.unique(comp)
                state[:, 6] = np.bincount(
                    (uc // K).astype(np.int64), minlength=N
                ).astype(np.float32)

            # Dims 7, 8 — unique dst-port count & dst-port entropy (out-edges).
            port_hist = np.zeros((N, N_DST_PORT_BINS), dtype=np.int64)
            if out_m.any():
                key = (
                    seg_id[out_m].astype(np.int64) * N_DST_PORT_BINS
                    + g_port[out_m].astype(np.int64)
                )
                port_hist = np.bincount(
                    key, minlength=N * N_DST_PORT_BINS
                ).reshape(N, N_DST_PORT_BINS)
            state[:, 7] = (port_hist > 0).sum(axis=1).astype(np.float32)
            nz8 = port_hist.sum(axis=1) > 0
            if nz8.any():
                state[nz8, 8] = scipy_entropy(
                    port_hist[nz8], axis=1
                ).astype(np.float32)

            # Dims 9, 10 — rolling in/out bytes (float64 accum → allclose).
            in_sum  = np.bincount(seg_id[in_m],  weights=g_byt[in_m],  minlength=N)
            out_sum = np.bincount(seg_id[out_m], weights=g_byt[out_m], minlength=N)
            state[:, 9]  = np.log1p(in_sum).astype(np.float32)
            state[:, 10] = np.log1p(out_sum).astype(np.float32)

            # Dim 14 — IAT regularity (CV), direction-agnostic full window slice.
            if total >= 2:
                d = np.diff(g_ts.astype(np.float64))
                same = seg_id[1:] == seg_id[:-1]
                d_seg = seg_id[1:][same]
                dv = d[same]
                cnt  = np.bincount(d_seg, minlength=N).astype(np.float64)
                ssum = np.bincount(d_seg, weights=dv,      minlength=N)
                sq   = np.bincount(d_seg, weights=dv * dv, minlength=N)
                ok = cnt > 0
                mean = np.zeros(N); mean[ok] = ssum[ok] / cnt[ok]
                var  = np.zeros(N); var[ok]  = sq[ok] / cnt[ok] - mean[ok] ** 2
                var  = np.maximum(var, 0.0)
                std  = np.sqrt(var)
                good = ok & (mean != 0.0)
                cv = np.zeros(N); cv[good] = std[good] / mean[good]
                state[:, 14] = cv.astype(np.float32)

        # Dim 13 — volume deviation: (dim9 + dim10) - baseline[node, hour].
        rolling_bytes = state[:, 9] + state[:, 10]
        if num_nodes > 0:
            base = baseline_arr[safe_ids, hour_int]
            base = np.where(valid, base, np.float32(self._global_baseline))
        else:
            base = np.full(N, np.float32(self._global_baseline))
        state[:, 13] = (rolling_bytes - base).astype(np.float32)

        return state

    def rollback_edge(
        self,
        node_id: int,
        edge_timestamp_ms: float,
        edge_direction: str,
        edge_features: dict,
        query_time_ms: float,
    ) -> np.ndarray:
        """Compute node state as if one specific edge were absent.

        Used by SHAP temporal coalitions (Phase 6). Does NOT modify
        internal state; safe to call concurrently on different edges.

        Args:
            node_id:           the node whose state to compute.
            edge_timestamp_ms: timestamp of the edge to exclude (ms).
            edge_direction:    "incoming" or "outgoing".
            edge_features:     dict with at least key 'peer_id' (int).
            query_time_ms:     the state query time (ms).

        Returns:
            np.ndarray of shape (15,), dtype float32.
        """
        return self._compute_state(
            int(node_id),
            float(query_time_ms),
            excluded_edges=[(float(edge_timestamp_ms), edge_direction,
                             int(edge_features.get("peer_id", -1)))],
        )

    def rollback_edges(
        self,
        node_id: int,
        query_time_ms: float,
        excluded: list[tuple[float, str, int]],
    ) -> np.ndarray:
        """Compute node state excluding multiple specific edges simultaneously.

        Required for temporal SHAP coalitions where several neighbor edges
        are absent at once. Does NOT modify internal state.

        Args:
            node_id:        the node whose state to compute.
            query_time_ms:  state query time in milliseconds.
            excluded:       list of (edge_timestamp_ms, direction, peer_id)
                            tuples — each edge treated as if absent.

        Returns:
            np.ndarray of shape (15,), dtype float32.
        """
        return self._compute_state(
            int(node_id),
            float(query_time_ms),
            excluded_edges=excluded,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: Path | str) -> None:
        """Persist histories, baselines, and snapshots to output_dir/.

        Saves:
          baselines.pkl    per-node hourly baselines and global fallback
          histories.pkl    per-node _NodeHistory objects (numpy arrays)
          snapshots.pkl    periodic snapshot times and state dicts
          meta.pkl         window_seconds and snapshot_interval
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()

        with open(output_dir / "baselines.pkl", "wb") as f:
            pickle.dump(
                {"baselines": self._baselines, "global": self._global_baseline}, f,
                protocol=4,
            )
        with open(output_dir / "histories.pkl", "wb") as f:
            pickle.dump(self._histories, f, protocol=4)
        with open(output_dir / "snapshots.pkl", "wb") as f:
            pickle.dump(
                {"times": self._snap_times, "states": self._snap_states}, f,
                protocol=4,
            )
        if self._is_internal is not None:
            np.save(output_dir / "is_internal.npy", self._is_internal)
        with open(output_dir / "meta.pkl", "wb") as f:
            pickle.dump(
                {"window_seconds": self.window_seconds,
                 "snapshot_interval": self.snapshot_interval}, f,
            )
        logger.info(
            f"NodeStateManager saved → {output_dir} "
            f"(save elapsed={time.monotonic() - t0:.1f}s, maxrss={_maxrss_mb():.0f} MiB)"
        )

    @classmethod
    def load(cls, output_dir: Path | str) -> "NodeStateManager":
        """Restore a NodeStateManager from a previously saved directory."""
        output_dir = Path(output_dir)

        with open(output_dir / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
        nsm = cls(
            window_seconds=meta["window_seconds"],
            snapshot_interval=meta["snapshot_interval"],
        )
        with open(output_dir / "baselines.pkl", "rb") as f:
            bl = pickle.load(f)
        nsm._baselines       = bl["baselines"]
        nsm._global_baseline = bl["global"]

        with open(output_dir / "histories.pkl", "rb") as f:
            nsm._histories = pickle.load(f)

        with open(output_dir / "snapshots.pkl", "rb") as f:
            snaps = pickle.load(f)
        nsm._snap_times  = snaps["times"]
        nsm._snap_states = snaps["states"]

        is_int_path = output_dir / "is_internal.npy"
        if is_int_path.exists():
            nsm._is_internal = np.load(is_int_path)

        logger.info(
            f"NodeStateManager loaded from {output_dir}: "
            f"{len(nsm._histories):,} nodes, "
            f"{len(nsm._snap_times):,} snapshots"
        )
        return nsm

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute_state(
        self,
        node_id: int,
        time_ms: float,
        excluded_edges: Optional[list[tuple[float, str, int]]] = None,
    ) -> np.ndarray:
        """Compute the 15-dim state vector for node_id at time_ms.

        excluded_edges: list of (timestamp_ms, direction, peer_id) tuples.
        Each matching edge is treated as absent (rollback for SHAP coalitions).

        Returns float32 array of shape (15,).
        """
        state = np.zeros(self.NODE_STATE_DIM, dtype=np.float32)

        # [0] is_internal — static, does not change with time
        if self._is_internal is not None and 0 <= node_id < len(self._is_internal):
            state[0] = self._is_internal[node_id]

        hist = self._histories.get(node_id)

        if hist is None or hist.first_seen_ms < 0:
            # Node has no recorded edges: all behavioral features are 0,
            # recency = 1.0 (not seen in W), novelty = 0
            state[2] = 1.0
            self._fill_seasonal(state, time_ms, node_id, rolling_bytes=0.0)
            state[14] = 0.0
            return state

        # Retrieve window records via binary search
        left, right = hist.window_bounds(time_ms, self._W_ms)

        ts    = hist.ts[left:right]
        inc   = hist.is_incoming[left:right]
        byt   = hist.raw_bytes[left:right]
        ports = hist.dst_port[left:right]
        peers = hist.peer_id[left:right]

        # Rollback: exclude listed edges (each matched once, in order)
        _first_seen_rolled_back = False
        if excluded_edges and len(ts) > 0:
            keep = np.ones(len(ts), dtype=bool)
            for (ex_ts, ex_dir, ex_peer) in excluded_edges:
                is_target_dir = inc if ex_dir == "incoming" else ~inc
                match = (
                    (ts == int(ex_ts))
                    & is_target_dir
                    & (peers == int(ex_peer))
                )
                active = np.where(match & keep)[0]
                if len(active) > 0:
                    keep[active[0]] = False
                    if int(ex_ts) == hist.first_seen_ms:
                        _first_seen_rolled_back = True
            ts, inc, byt, ports, peers = (
                ts[keep], inc[keep], byt[keep],
                ports[keep], peers[keep],
            )

        w_start = time_ms - self._W_ms

        # [1] novelty: node's effective first appearance is within window W.
        # When the first-seen edge is rolled back, the node is no longer novel
        # (the evidence of its initial appearance has been masked).
        if _first_seen_rolled_back:
            state[1] = 0.0
        else:
            state[1] = float(w_start <= hist.first_seen_ms <= time_ms)

        # [2] recency: normalized time since most-recent edge in W (0=just seen)
        if len(ts) > 0:
            state[2] = min(1.0, (time_ms - float(ts.max())) / self._W_ms)
        else:
            state[2] = 1.0

        in_mask  = inc
        out_mask = ~inc

        # [3] rolling_in_degree
        state[3] = float(in_mask.sum())

        # [4] rolling_out_degree
        state[4] = float(out_mask.sum())

        # [5] unique_dst_ip_count
        state[5] = float(len(np.unique(peers[out_mask]))) if out_mask.any() else 0.0

        # [6] unique_src_ip_count
        state[6] = float(len(np.unique(peers[in_mask]))) if in_mask.any() else 0.0

        # [7] unique_dst_port_count
        out_ports = ports[out_mask]
        state[7]  = float(len(np.unique(out_ports))) if out_mask.any() else 0.0

        # [8] dst_port_entropy (Shannon)
        if out_mask.any():
            _, counts = np.unique(out_ports, return_counts=True)
            state[8] = float(scipy_entropy(counts))

        # [9] rolling_in_bytes
        state[9]  = float(np.log1p(byt[in_mask].sum()))  if in_mask.any()  else 0.0

        # [10] rolling_out_bytes
        state[10] = float(np.log1p(byt[out_mask].sum())) if out_mask.any() else 0.0

        # [11–13] seasonal
        rolling_bytes = state[9] + state[10]
        self._fill_seasonal(state, time_ms, node_id, rolling_bytes)

        # [14] iat_regularity: CV of inter-arrival times across all window edges
        state[14] = self._iat_regularity(ts)

        return state

    def _fill_seasonal(
        self,
        state: np.ndarray,
        time_ms: float,
        node_id: int,
        rolling_bytes: float,
    ) -> None:
        """Fill state[11:14] with time encoding and volume deviation."""
        hour_frac = (time_ms / 3_600_000.0) % 24.0

        # [11] time_sin, [12] time_cos — circular encoding avoids midnight discontinuity
        state[11] = float(np.sin(2.0 * np.pi * hour_frac / 24.0))
        state[12] = float(np.cos(2.0 * np.pi * hour_frac / 24.0))

        # [13] volume_deviation: current log-bytes minus per-node per-hour baseline
        hour_int = int(hour_frac)
        baseline = self._baselines.get(node_id, {}).get(hour_int, self._global_baseline)
        state[13] = rolling_bytes - float(baseline)

    @staticmethod
    def _iat_regularity(ts: np.ndarray) -> float:
        """CV (std/mean) of inter-arrival times. Returns 0.0 if < 2 edges."""
        if len(ts) < 2:
            return 0.0
        iats = np.diff(ts.astype(np.float64))
        mean_iat = iats.mean()
        if mean_iat == 0.0:
            return 0.0
        return float(iats.std() / mean_iat)
