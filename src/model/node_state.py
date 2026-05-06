"""
15-dimensional per-node state vector. NEW — Paper 1 used constant ones.

NODE STATE SCHEMA:
  Behavioral (11):
  [0]  is_internal       binary  1 if private IP (10.x, 172.16-31.x, 192.168.x)
                                 DATASET NOTE: UNSW-NB15 uses public IPs, not RFC1918.
                                 This feature returns 0 for all nodes in this dataset.
                                 Limitation accepted and noted in paper. Correct behaviour
                                 for real deployments and Papers 3–4 (CIC-IDS, ToN-IoT, BoT-IoT).
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

SNAPSHOTS: Pre-compute node state every snapshot_interval edges in temporal
order. Mini-batch looks up nearest snapshot <= t_e, applies delta updates.

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
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import entropy as scipy_entropy

from src.data.preprocessor import port_to_bin_indices

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

    Typical workflow::

        nsm = NodeStateManager(window_seconds=60, snapshot_interval=1000)
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
        snapshot_interval: int = 1000,
    ) -> None:
        """
        Args:
            window_seconds:    rolling window width W in seconds (default 60).
            snapshot_interval: edges between periodic state snapshots (default 1000).
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
        logger.info(
            f"Building hourly baselines from {len(src_node_ids):,} training edges"
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
            f"global fallback={self._global_baseline:.4f}"
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
        """Build per-node edge histories and periodic state snapshots.

        Processes ALL edges (train + val + test) in temporal order so that
        val/test node states correctly reflect prior training history.
        Call build_hourly_baselines() first.

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
            snapshot_interval: edges between snapshots; defaults to self.snapshot_interval.
        """
        interval = snapshot_interval if snapshot_interval is not None else self.snapshot_interval
        n = len(src_node_ids)
        logger.info(
            f"Building node histories: {n:,} edges, interval={interval}"
        )

        # Pass 1: populate per-node edge lists.
        # Convert raw L4_DST_PORT values to semantic bin indices (0-15) ONCE
        # before the loop so that dst_port_entropy and unique_dst_port_count
        # operate over the 16-service taxonomy, not raw port numbers.
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

        logger.info(f"Edge histories finalized for {len(self._histories):,} nodes")

        # Pass 2: take periodic state snapshots
        self._snap_times.clear()
        self._snap_states.clear()

        for step in range(interval - 1, n, interval):
            t_ms = int(timestamps_ms[step])
            # Only snapshot nodes that have been seen up to this point
            snap: dict[int, np.ndarray] = {}
            for nid, hist in self._histories.items():
                if 0 <= hist.first_seen_ms <= t_ms:
                    snap[nid] = self._compute_state(nid, float(t_ms))
            self._snap_times.append(t_ms)
            self._snap_states.append(snap)

        logger.info(f"Snapshots taken: {len(self._snap_times):,}")

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
        """
        t = float(time_ms)
        return np.stack([self._compute_state(int(nid), t) for nid in node_ids])

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
            exclude_ts=float(edge_timestamp_ms),
            exclude_direction=edge_direction,
            exclude_peer_id=int(edge_features.get("peer_id", -1)),
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
        logger.info(f"NodeStateManager saved → {output_dir}")

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
        exclude_ts: Optional[float] = None,
        exclude_direction: Optional[str] = None,
        exclude_peer_id: Optional[int] = None,
    ) -> np.ndarray:
        """Compute the 15-dim state vector for node_id at time_ms.

        The optional exclude_* arguments implement rollback for SHAP: the
        specified edge is treated as absent without modifying stored data.

        Returns float32 array of shape (15,).
        """
        state = np.zeros(self.NODE_STATE_DIM, dtype=np.float32)

        # [0] is_internal — static, does not change with time
        if self._is_internal is not None and node_id < len(self._is_internal):
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

        # Rollback: exclude the first matching edge record
        _first_seen_rolled_back = False
        if exclude_ts is not None and len(ts) > 0:
            is_target_dir = inc if exclude_direction == "incoming" else ~inc
            match = (
                (ts == int(exclude_ts))
                & is_target_dir
                & (peers == exclude_peer_id)
            )
            first_hit = np.where(match)[0]
            if len(first_hit) > 0:
                keep = np.ones(len(ts), dtype=bool)
                keep[first_hit[0]] = False
                ts, inc, byt, ports, peers = (
                    ts[keep], inc[keep], byt[keep],
                    ports[keep], peers[keep],
                )
                # Track if the global first-seen edge was the one removed
                _first_seen_rolled_back = (int(exclude_ts) == hist.first_seen_ms)

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
