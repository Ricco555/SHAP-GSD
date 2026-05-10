"""
Construct DGL graphs from preprocessed splits.

FOUR CHANGES FROM PAPER 1:

CHANGE 1 — IP-level nodes (not IP:PORT):
  TE-G-SAGE: node = "10.0.0.5:49152"
  SHAP-GSD:  node = "10.0.0.5"
  Ports move to edge features (encoded in preprocessor).
  Avoids host history fragmentation across ephemeral ports.

CHANGE 2 — Port encoding (already done in preprocessor):
  DST_PORT → 16-bin semantic one-hot (HTTP, HTTPS, DNS, SSH, FTP, SMTP,
    RDP[T1021.001], SNMP[T1046], NTP[T1124/T1498.002],
    IMAP[T1071.003], SunRPC[T1046], BitTorrent[T1571], AIM-ICQ[T1071.005],
    other_well_known, registered, ephemeral)
  SRC_PORT → 1 binary is_ephemeral column

CHANGE 3 — Timestamps on edges:
  g.edata['timestamp'] = FLOW_START_MILLISECONDS as int64
  Used by TemporalNeighborSampler to filter neighbors.

CHANGE 4 — Node state from NodeStateManager (15-dim):
  Not constant ones. See node_state.py.

Construction steps:
1. Map IPV4_SRC_ADDR / IPV4_DST_ADDR → node IDs. Save node_id_map.json.
2. Create directed edges (src_node_id → dst_node_id) per flow.
3. Set g.edata: EID (global), timestamp, label.
4. Initialize node state snapshots via NodeStateManager.
5. Persist graph using DGL native functions:
     dgl.save_graphs('graphs/train.bin', [g_train])
     g_list, _ = dgl.load_graphs('graphs/train.bin')
   Also save node_id_map.json (IP string → node int ID).

VALIDATION:
  g.edata[dgl.EID][i] → correct feature_store row
  g.edata['timestamp'][i] == original FLOW_START_MILLISECONDS
  node IDs are IP-level (no port suffix)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import dgl
import numpy as np
import torch

logger = logging.getLogger(__name__)

# RFC1918 private ranges — used for the is_internal node feature.
# DATASET NOTE: UNSW-NB15 uses public IPs; this always returns False for that
# dataset. Feature is correct for real deployments and Papers 3–4.
_RFC1918_PREFIXES: tuple[tuple, ...] = (
    (10,),              # 10.0.0.0/8
    (172, 16, 31),      # 172.16.0.0/12  — stored as (a, lo, hi)
    (192, 168),         # 192.168.0.0/16
)


def _is_internal_ip(ip_str: str) -> bool:
    """Return True if ip_str is an RFC1918 private address."""
    try:
        parts = ip_str.split(".")
        if len(parts) != 4:
            return False
        a, b = int(parts[0]), int(parts[1])
        return (
            a == 10
            or (a == 172 and 16 <= b <= 31)
            or (a == 192 and b == 168)
        )
    except (ValueError, IndexError):
        return False


class GraphBuilder:
    """Build per-split DGL graphs with consistent IP-level node IDs.

    All three splits share the same node_id_map so the same physical host
    always has the same node index, enabling cross-split neighbor sampling
    via the TemporalNeighborSampler.
    """

    def __init__(
        self,
        graph_dir: Path | str,
        node_id_map: Optional[dict[str, int]] = None,
    ) -> None:
        """
        Args:
            graph_dir:    directory for output .bin files and node_id_map.json.
            node_id_map:  pre-built IP→node_id mapping (pass when loading an
                          existing map; otherwise call build_global_node_map).
        """
        self.graph_dir: Path = Path(graph_dir)
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.node_id_map: dict[str, int] = node_id_map or {}
        self._num_nodes: int = len(self.node_id_map)

    # ------------------------------------------------------------------
    # Node mapping
    # ------------------------------------------------------------------

    def build_global_node_map(
        self,
        all_src_ips: np.ndarray,
        all_dst_ips: np.ndarray,
    ) -> dict[str, int]:
        """Build a deterministic IP→node_id mapping from all splits combined.

        Sorted lexicographically so node IDs are stable across identical runs.

        Args:
            all_src_ips: IPV4_SRC_ADDR strings from all splits concatenated.
            all_dst_ips: IPV4_DST_ADDR strings from all splits concatenated.

        Returns:
            Mapping from IP string to integer node ID (0-indexed).
        """
        unique_ips = sorted(
            set(str(ip) for ip in all_src_ips)
            | set(str(ip) for ip in all_dst_ips)
        )
        self.node_id_map = {ip: idx for idx, ip in enumerate(unique_ips)}
        self._num_nodes = len(self.node_id_map)
        logger.info(f"Global node map: {self._num_nodes:,} unique IPs")
        return self.node_id_map

    def save_node_id_map(self) -> None:
        """Persist node_id_map.json to graph_dir."""
        assert self.node_id_map, "node_id_map is empty; call build_global_node_map first"
        path = self.graph_dir / "node_id_map.json"
        with open(path, "w") as f:
            json.dump(self.node_id_map, f)
        logger.info(f"Saved node_id_map → {path}  ({self._num_nodes:,} nodes)")

    @classmethod
    def load_node_id_map(cls, graph_dir: Path | str) -> dict[str, int]:
        """Load a previously saved node_id_map.json from graph_dir."""
        path = Path(graph_dir) / "node_id_map.json"
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def compute_is_internal_array(node_id_map: dict[str, int]) -> np.ndarray:
        """Return a bool array of shape (num_nodes,) where arr[node_id] = is_internal.

        Used to initialise NodeStateManager.set_is_internal().
        """
        n = len(node_id_map)
        arr = np.zeros(n, dtype=np.float32)
        for ip, idx in node_id_map.items():
            arr[idx] = float(_is_internal_ip(ip))
        n_internal = int(arr.sum())
        logger.info(
            f"is_internal: {n_internal}/{n} nodes are RFC1918 "
            f"(expected 0 for UNSW-NB15 dataset)"
        )
        return arr

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_split_graph(
        self,
        split_name: str,
        src_ips: np.ndarray,
        dst_ips: np.ndarray,
        global_eids: np.ndarray,
        timestamps: np.ndarray,
        labels: np.ndarray,
    ) -> dgl.DGLGraph:
        """Build a DGL graph for one split.

        Sets edata:
          dgl.EID       int64  global edge ID (position in sorted dataset)
          'timestamp'   int64  FLOW_START_MILLISECONDS
          'label'       int64  integer class label

        num_nodes equals the global count so IDs are consistent across splits.
        Edge order matches the input arrays exactly, preserving EID alignment.

        Args:
            split_name:  "train", "val", or "test" (used only for logging).
            src_ips:     IPV4_SRC_ADDR strings (n,).
            dst_ips:     IPV4_DST_ADDR strings (n,).
            global_eids: chronologically assigned EID per edge (n,), int64.
            timestamps:  FLOW_START_MILLISECONDS per edge (n,), int64.
            labels:      integer class labels (n,), int64.

        Returns:
            dgl.DGLGraph with edata set.

        Raises:
            AssertionError: if EID alignment is violated.
        """
        assert self.node_id_map, "Call build_global_node_map first"
        n = len(src_ips)
        assert len(dst_ips) == n, "src/dst length mismatch"
        assert len(global_eids) == n and len(timestamps) == n and len(labels) == n

        src_ids = np.fromiter(
            (self.node_id_map[str(ip)] for ip in src_ips),
            dtype=np.int64,
            count=n,
        )
        dst_ids = np.fromiter(
            (self.node_id_map[str(ip)] for ip in dst_ips),
            dtype=np.int64,
            count=n,
        )

        g = dgl.graph((src_ids, dst_ids), num_nodes=self._num_nodes)
        g.edata[dgl.EID]     = torch.tensor(global_eids, dtype=torch.int64)
        g.edata["timestamp"] = torch.tensor(timestamps,  dtype=torch.int64)
        g.edata["label"]     = torch.tensor(labels,      dtype=torch.int64)

        # CRITICAL INVARIANT: EID alignment must hold exactly
        assert np.array_equal(g.edata[dgl.EID].numpy(), global_eids), (
            f"EID alignment violated in {split_name} graph"
        )

        logger.info(
            f"Built {split_name} graph: {n:,} edges, {self._num_nodes:,} nodes, "
            f"EID [{int(global_eids.min())}, {int(global_eids.max())}], "
            f"ts [{int(timestamps.min())}, {int(timestamps.max())}]"
        )
        return g

    def save_split_graph(self, split_name: str, g: dgl.DGLGraph) -> None:
        """Persist graph to graph_dir/<split_name>.bin using DGL native format."""
        path = self.graph_dir / f"{split_name}.bin"
        dgl.save_graphs(str(path), [g])
        logger.info(f"Saved {split_name} graph → {path}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_eid_alignment(
        self,
        g: dgl.DGLGraph,
        feature_store_dir: Path | str,
    ) -> None:
        """Assert g.edata[dgl.EID][i] == feature_store/edge_indices.npy[i] for all i.

        Raises AssertionError on mismatch with a descriptive message.
        """
        feature_store_dir = Path(feature_store_dir)
        stored = np.load(feature_store_dir / "edge_indices.npy")
        graph  = g.edata[dgl.EID].numpy()

        assert len(graph) == len(stored), (
            f"Edge count mismatch: graph={len(graph):,}, store={len(stored):,}"
        )
        n_mismatch = int((graph != stored).sum())
        assert n_mismatch == 0, (
            f"EID alignment violated: {n_mismatch:,} mismatched EIDs "
            f"between graph and {feature_store_dir / 'edge_indices.npy'}"
        )
        logger.info(f"EID alignment validated for {feature_store_dir} ({len(graph):,} edges)")
