"""
Temporal neighborhood coalition SHAP. Second SHAP-GSD granularity.

Answers: "Which recent flows in the 2-hop neighborhood contributed?"

Coalition space: N = |E_sub| neighbor edges from the computation blocks
(all edges in blocks, excluding the target edge itself).

Masking edge e_i (z_T[i]=0) — Option C:
  1. For each endpoint of e_i: call NodeStateManager.rollback_edges() with
     all absent edges incident to that node in this coalition.
  Multiple absent edges sharing an endpoint are handled in a single
  rollback_edges call (processed in the correct window-filter pass).

Node embeddings are recomputed per coalition from the rolled-back states.
The target edge features x_e are fixed (they are not varied here).

Output: list of (global_eid, timestamp_ms, signed_φ) sorted by |φ| descending.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import dgl
    from src.model.sage_model import EdgeAwareGraphSAGE
    from src.model.node_state import NodeStateManager
    from src.explainer.background import BackgroundDistributions

logger = logging.getLogger(__name__)


@dataclass
class _NeighborEdge:
    """Metadata for one neighbor edge in the 2-hop subgraph."""
    local_eid: int    # local EID in g_split
    global_eid: int   # global EID → feature store key
    timestamp_ms: float
    src_nid: int      # global node IDs in g_split
    dst_nid: int


class TemporalNeighborhoodSHAP:
    """KernelSHAP over 2-hop temporal neighborhood edges for one target edge."""

    def __init__(
        self,
        background: "BackgroundDistributions",
        nsm: "NodeStateManager",
        g_split: "dgl.DGLGraph",
        device: torch.device,
    ) -> None:
        """
        Args:
            background: BackgroundDistributions (training split).
            nsm:        NodeStateManager (loaded from node_state_snapshots/).
            g_split:    DGL split graph (test or val) with edata['timestamp']
                        and edata[dgl.EID] mapping local → global EIDs.
            device:     torch device.
        """
        import dgl as _dgl
        self.background = background
        self.nsm = nsm
        self.g_split = g_split
        self.device = device
        self._dgl = _dgl

    def _extract_neighbor_edges(
        self,
        blocks: list,
        target_local_eid: int,
        target_ts_ms: float | None = None,
    ) -> list[_NeighborEdge]:
        """Collect neighbor edges in blocks that fall within the node-state window.

        Only edges whose timestamp satisfies target_ts_ms - W_ms <= ts <= target_ts_ms
        are included. Edges outside the window cannot affect node states via rollback
        and therefore have zero marginal contribution — excluding them makes the
        coalition space semantically meaningful and computationally efficient.

        If target_ts_ms is None, all edges are returned (backward-compat).
        """
        dgl = self._dgl
        W_ms = self.nsm._W_ms
        seen: set[int] = set()
        records: list[_NeighborEdge] = []

        for block in blocks:
            local_eids_t: torch.Tensor = block.edata[dgl.EID]
            if local_eids_t.numel() == 0:
                continue

            local_eids = local_eids_t.tolist()
            unique_local: list[int] = []
            for leid in local_eids:
                if leid != target_local_eid and leid not in seen:
                    seen.add(leid)
                    unique_local.append(leid)

            if not unique_local:
                continue

            ul_t = torch.tensor(unique_local, dtype=torch.long)
            srcs, dsts = self.g_split.find_edges(ul_t)
            geids = self.g_split.edata[dgl.EID][ul_t]
            timestamps = self.g_split.edata["timestamp"][ul_t]

            for i, leid in enumerate(unique_local):
                ts = float(timestamps[i])
                # Filter to in-window edges only when target_ts_ms is provided
                if target_ts_ms is not None and (target_ts_ms - ts) > W_ms:
                    continue
                records.append(_NeighborEdge(
                    local_eid=leid,
                    global_eid=int(geids[i]),
                    timestamp_ms=ts,
                    src_nid=int(srcs[i]),
                    dst_nid=int(dsts[i]),
                ))

        return records

    def extract_neighbor_edges(
        self,
        blocks: list,
        target_local_eid: int,
        target_ts_ms: float,
    ) -> list[_NeighborEdge]:
        """Public wrapper over _extract_neighbor_edges for metric code.

        Returns the same in-window neighbor records, in the same order, that
        explain() uses as KernelSHAP coalition columns.

        Args:
            blocks:           DGL computation blocks from the sampler.
            target_local_eid: local EID of the target edge in g_split.
            target_ts_ms:     target edge timestamp in milliseconds.

        Returns:
            List of _NeighborEdge records (may be empty).
        """
        return self._extract_neighbor_edges(
            blocks, target_local_eid, target_ts_ms=target_ts_ms
        )

    def build_masked_node_feats(
        self,
        neighbor_records: list[_NeighborEdge],
        absent_idx: "np.ndarray | list[int]",
        base_node_feats: np.ndarray,
        input_node_ids: np.ndarray,
        target_ts_ms: float,
    ) -> np.ndarray:
        """Node-state matrix for one temporal coalition (Option C masking).

        Absent neighbor edges are removed by recomputing the affected endpoints'
        15-dim states via NodeStateManager.rollback_edges — never by DGL graph
        surgery. The caller's `blocks` object is reused unchanged.

        Args:
            neighbor_records: in-window neighbor edges, index-aligned with the
                              KernelSHAP coalition columns.
            absent_idx:       indices into neighbor_records that are ABSENT
                              (coalition bit 0). Empty → base_node_feats is
                              returned unmodified (a copy is NOT made).
            base_node_feats:  float32 (N_in, node_state_dim) with all neighbors
                              present; row j corresponds to input_node_ids[j].
            input_node_ids:   int array (N_in,) of global node IDs for blocks[0]
                              input, in block row order.
            target_ts_ms:     state query time (target edge timestamp, ms).

        Returns:
            float32 array (N_in, node_state_dim). Either `base_node_feats` itself
            (no absences) or a modified copy. Callers must treat it as read-only.
        """
        assert base_node_feats.shape[0] == len(input_node_ids), (
            f"row misalignment: base_node_feats has {base_node_feats.shape[0]} rows "
            f"but input_node_ids has {len(input_node_ids)}"
        )

        if len(absent_idx) == 0:
            # All neighbors present — pre-computed states are already correct
            return base_node_feats

        # Build per-node exclusion lists for all absent edges
        node_exclusions: dict[int, list[tuple[float, str, int]]] = {}
        for i in absent_idx:
            rec = neighbor_records[i]
            node_exclusions.setdefault(rec.src_nid, []).append(
                (rec.timestamp_ms, "outgoing", rec.dst_nid)
            )
            node_exclusions.setdefault(rec.dst_nid, []).append(
                (rec.timestamp_ms, "incoming", rec.src_nid)
            )

        # Recompute states with rollbacks for affected nodes
        modified_nf = base_node_feats.copy()
        for j, nid in enumerate(input_node_ids):
            excl = node_exclusions.get(int(nid))
            if excl:
                modified_nf[j] = self.nsm.rollback_edges(
                    int(nid), target_ts_ms, excl
                )
        return modified_nf

    def explain(
        self,
        target_local_eid: int,
        true_class: int,
        model: "EdgeAwareGraphSAGE",
        blocks: list,
        input_nodes: torch.Tensor,
        target_ts_ms: float,
        src_pos: torch.Tensor,
        dst_pos: torch.Tensor,
        x_e: np.ndarray,
        base_node_feats: np.ndarray,
        nsamples: int = 1024,
    ) -> list[tuple[int, float, float]]:
        """Run KernelSHAP over the 2-hop temporal neighborhood.

        Args:
            target_local_eid: local EID of the target edge in g_split.
            true_class:       integer class label of the target edge.
            model:            trained EdgeAwareGraphSAGE (eval mode expected).
            blocks:           DGL computation blocks from the sampler.
            input_nodes:      global node IDs for blocks[0] input, shape (N_in,).
            target_ts_ms:     target edge timestamp in milliseconds.
            src_pos:          int64 tensor — seed edge source positions into h.
            dst_pos:          int64 tensor — seed edge destination positions into h.
            x_e:              float32 numpy array (d_e,) — target edge features.
            base_node_feats:  float32 numpy array (N_in, 15) — node states with
                              all neighbors present (pre-computed by caller).
            nsamples:         KernelSHAP coalition samples (default 1024).

        Returns:
            List of (global_eid, timestamp_ms, signed_φ) sorted by |φ| descending.
            Empty list if the 2-hop neighborhood is empty.
        """
        import shap

        neighbor_records = self._extract_neighbor_edges(
            blocks, target_local_eid, target_ts_ms=target_ts_ms
        )
        N = len(neighbor_records)
        if N == 0:
            logger.debug("Temporal SHAP: no in-window neighbors, returning []")
            return [], 0.0, 0.0

        model.eval()
        input_node_ids = input_nodes.cpu().numpy()
        x_e_t = torch.tensor(x_e, dtype=torch.float32, device=self.device).unsqueeze(0)

        def _predict_fn(coalition_matrix: np.ndarray) -> np.ndarray:
            results: list[float] = []
            for row in coalition_matrix:
                absent_idx = np.where(row == 0)[0]
                nf = self.build_masked_node_feats(
                    neighbor_records, absent_idx, base_node_feats,
                    input_node_ids, target_ts_ms,
                )
                nf_t = torch.tensor(nf, dtype=torch.float32, device=self.device)

                with torch.no_grad():
                    logit = model(blocks, nf_t, x_e_t, src_pos, dst_pos)
                results.append(logit[0, true_class].item())
            return np.array(results, dtype=np.float64)

        background_data = np.zeros((1, N), dtype=np.float32)
        foreground_data = np.ones((1, N), dtype=np.float32)
        explainer = shap.KernelExplainer(_predict_fn, background_data)
        f_baseline = float(np.squeeze(explainer.expected_value))
        f_logit = float(_predict_fn(foreground_data)[0])
        phi_raw = explainer.shap_values(
            foreground_data,
            nsamples=nsamples,
            silent=True,
        )
        phi = np.array(phi_raw).squeeze()
        if N == 1:
            phi = np.array([float(phi)])

        logger.debug(
            f"Temporal SHAP: {N} neighbors, class={true_class}, sum(φ)={phi.sum():.4f}, "
            f"efficiency_err={abs(phi.sum() - (f_logit - f_baseline)):.4f}"
        )

        results_list = [
            (rec.global_eid, rec.timestamp_ms, float(phi[i]))
            for i, rec in enumerate(neighbor_records)
        ]
        results_list.sort(key=lambda t: abs(t[2]), reverse=True)
        return results_list, f_baseline, f_logit
