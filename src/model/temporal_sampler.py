"""
Timestamp-filtered neighbor sampler with exact per-edge temporal cutoffs.

PAPER 1 BUG: Default DGL NeighborSampler draws from ALL edges in the split.
A flow at t=1 can have 2-hop neighbors at t=2,000,000 (within-split leakage).

FIX: When sampling k-hop neighbors for target edge e at t_e, only include
edges e' where start_time(e') <= t_e.

IMPLEMENTATION — Option B (exact per-edge cutoffs):
Override sample_blocks to receive seed EIDs directly from the DataLoader.
For each mini-batch:
  1. Extract per-edge timestamps; assert non-decreasing order (shuffle=False guard).
  2. Build per-node cutoff tensor via scatter_min over incident seed-edge timestamps.
     Seed-edge endpoints: cutoff = min(their seed-edge timestamps).
     All other nodes: cutoff = batch_max_time (fallback, same as Option A for
     nodes not directly adjacent to any seed edge).
  3. Build a temporally filtered subgraph: edge i valid iff ts[i] <= cutoff[dst[i]].
  4. Sample fan-out on the filtered subgraph.
  5. After each hop, propagate cutoffs to next-level seed nodes using the
     timestamps of the just-sampled edges (scatter_min over hop-src IDs).
     This gives EXACT guarantees for multi-hop neighborhoods:
       hop-1 nodes: cutoff <= t_e exactly
       hop-2 nodes: cutoff <= timestamp of the hop-1 edge that reached them <= t_e

Paper claim (exact, no approximation table needed):
  For any target edge e at time t_e, every edge in its sampled k-hop
  computation graph has timestamp <= t_e.

The assertion in sample_blocks is self-enforcing: if the caller shuffles the
DataLoader, the violation fires immediately at runtime rather than silently
producing incorrect blocks.

CRITICAL TEST — zero violations required:
  For 1000 random target edges (tested as individual batches or in sorted
  mini-batches of realistic size): every sampled neighbor edge timestamp
  <= target edge timestamp.
"""

import logging
from typing import Optional

import dgl
import torch

logger = logging.getLogger(__name__)


class TemporalNeighborSampler(dgl.dataloading.BlockSampler):
    """Exact temporal neighbor sampler for edge-classification GNN training.

    Subclasses dgl.dataloading.BlockSampler and overrides sample_blocks so
    the DataLoader passes seed edge IDs directly, enabling per-edge cutoffs.

    Usage with DGL DataLoader::

        sampler = TemporalNeighborSampler(fanouts=[25, 15])
        loader = dgl.dataloading.DataLoader(
            g, balanced_train_eids, sampler,
            batch_size=512, shuffle=False,   # shuffle=False is enforced by assertion
            drop_last=False, num_workers=0,
        )
        for input_nodes, seed_eids, blocks in loader:
            node_feats = nsm.get_batch_states(input_nodes, batch_timestamp)
            edge_feats = feature_store.get_batch(seed_eids.numpy())
            logits = model(blocks, node_feats, edge_feats)
    """

    def __init__(
        self,
        fanouts: list[int],
        timestamp_key: str = "timestamp",
    ) -> None:
        """
        Args:
            fanouts:       per-hop max neighbors, e.g. [25, 15] for 2-layer SAGE.
                           fanouts[0] = first hop (closest to input features),
                           fanouts[-1] = last hop (closest to output).
            timestamp_key: edata key holding int64 FLOW_START_MILLISECONDS.
        """
        super().__init__()
        self.fanouts = fanouts
        self.timestamp_key = timestamp_key

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def sample_blocks(
        self,
        g: dgl.DGLGraph,
        seed_eids: torch.Tensor,
        exclude_eids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[dgl.DGLGraph]]:
        """Build multi-hop computation blocks with exact temporal cutoffs.

        Called by DGL DataLoader once per mini-batch. seed_eids are the global
        edge IDs for this batch (from balanced_train_indices, sorted by
        timestamp when shuffle=False).

        Args:
            g:          the split graph (train/val/test) with edata['timestamp'].
            seed_eids:  int64 tensor of seed edge IDs for this batch, shape (B,).
            exclude_eids: ignored (no link-prediction negatives).

        Returns:
            input_nodes: global node IDs needed for the first GNN layer, (N_in,).
            seed_eids:   unchanged, passed through for feature/label lookup.
            blocks:      list of DGL blocks, one per GNN layer (len == len(fanouts)).

        Raises:
            AssertionError: if seed_eids timestamps are not non-decreasing.
                            Fix: set shuffle=False in the DataLoader.
        """
        seed_eids = seed_eids.long()
        ts = g.edata[self.timestamp_key][seed_eids]

        # Self-enforcing sort-order invariant — fires at the DataLoader, not at
        # the paper review stage.
        if ts.shape[0] > 1:
            violations = int((ts[1:] < ts[:-1]).sum())
            assert violations == 0, (
                f"Seed EIDs are not timestamp-sorted ({violations} violations). "
                "Set shuffle=False in the DataLoader — this is required for "
                "the temporal neighbour sampler to provide exact cutoffs."
            )

        # Step 1: per-node cutoff tensor.
        # Seed-edge endpoints get the min of their incident seed-edge timestamps.
        # All other nodes default to batch_max_time (identical to Option A fallback).
        cutoffs = self._build_cutoffs(g, seed_eids, ts)

        # Step 2: seed nodes = endpoints of seed edges (deduplicated).
        src_s, dst_s = g.find_edges(seed_eids)
        seed_nodes = torch.unique(torch.cat([src_s, dst_s]))

        # Step 3: multi-hop sampling with cutoff propagation between hops.
        blocks: list[dgl.DGLGraph] = []
        curr_seeds = seed_nodes
        curr_cutoffs = cutoffs

        # DGL convention: enumerate hops from OUTPUT to INPUT, then reverse.
        for hop_idx, fanout in enumerate(reversed(self.fanouts)):
            sg = self._filtered_subgraph(g, curr_cutoffs)

            frontier = dgl.sampling.sample_neighbors(
                sg, curr_seeds, fanout, edge_dir="in"
            )
            block = dgl.to_block(frontier, curr_seeds)
            blocks.insert(0, block)

            # Input nodes of this block become seed nodes for the next hop.
            next_seeds = block.srcdata[dgl.NID]

            # Propagate cutoffs: next-hop nodes get min(their seed cutoff,
            # timestamp of the hop-edge that connected them here).
            # This makes the 2nd-hop guarantee EXACT rather than approximate.
            if hop_idx < len(self.fanouts) - 1:
                curr_cutoffs = self._propagate_cutoffs(
                    frontier, curr_cutoffs
                )

            curr_seeds = next_seeds

        # input_nodes = srcdata[NID] of the very first block (deepest hop)
        input_nodes = blocks[0].srcdata[dgl.NID]
        return input_nodes, seed_eids, blocks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_cutoffs(
        self,
        g: dgl.DGLGraph,
        seed_eids: torch.Tensor,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """Build per-node temporal cutoff tensor from seed edge timestamps.

        Seed-edge endpoints: cutoff[v] = min(t for all seed edges incident to v).
        Other nodes:         cutoff[v] = batch_max_time.
        """
        batch_max = int(ts.max())
        cutoffs = torch.full((g.num_nodes(),), batch_max, dtype=torch.int64)
        src, dst = g.find_edges(seed_eids)
        cutoffs.scatter_reduce_(0, src, ts, reduce="amin", include_self=True)
        cutoffs.scatter_reduce_(0, dst, ts, reduce="amin", include_self=True)
        return cutoffs

    def _filtered_subgraph(
        self,
        g: dgl.DGLGraph,
        cutoffs: torch.Tensor,
    ) -> dgl.DGLGraph:
        """Return a view of g containing only edges with ts[e] <= cutoffs[dst[e]].

        Uses relabel_nodes=False so global node IDs are preserved throughout
        the sampling pipeline, enabling correct scatter_reduce_ in propagation.
        """
        ts = g.edata[self.timestamp_key]
        _, dst = g.edges()
        valid_mask = ts <= cutoffs[dst]
        valid_eids = valid_mask.nonzero(as_tuple=False).view(-1)
        return g.edge_subgraph(valid_eids, relabel_nodes=False)

    def _propagate_cutoffs(
        self,
        frontier: dgl.DGLGraph,
        cutoffs: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate temporal cutoffs to the next hop's seed nodes.

        For each sampled hop-1 edge (u → v) with timestamp t_uv, node u
        (which becomes a hop-2 seed) gets cutoff min(cutoffs[u], t_uv).
        This ensures: hop-2 neighbors of u have timestamp <= t_uv <= t_e.

        Args:
            frontier: graph returned by sample_neighbors for this hop.
            cutoffs:  current per-node cutoff tensor (will not be mutated).

        Returns:
            New cutoff tensor with hop-source nodes updated.
        """
        new_cutoffs = cutoffs.clone()
        if frontier.num_edges() == 0:
            return new_cutoffs
        hop_src, _ = frontier.edges()
        hop_ts = frontier.edata[self.timestamp_key]
        new_cutoffs.scatter_reduce_(
            0, hop_src, hop_ts, reduce="amin", include_self=True
        )
        return new_cutoffs
