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
  3. Build, per seed node, the set of eligible in-edges (ts[e] <= cutoff[dst]) via
     a precomputed per-node timestamp-sorted CSC + np.searchsorted (O(log deg)
     per seed instead of an O(E) full-graph boolean mask every hop).
  4. Sample fan-out from the eligible set per seed node (take-all if the
     eligible count is <= fanout, else an O(fanout) Floyd draw).
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

PERFORMANCE (spec 18/19): the per-hop candidate generation is
O(seeds * (log deg + fanout)) via a precomputed per-node timestamp-sorted
in-adjacency (CSC) + np.searchsorted eligible-prefix + an O(fanout) Floyd
draw, replacing an earlier O(E)-per-hop full-split boolean mask
(`_filtered_subgraph` + dgl.sampling.sample_neighbors). This is E-independent
per call and required for Paper-3 scale (see specs/18, specs/19).

BLOCK-EID CONTRACT: block.edata[dgl.EID] holds the g-LOCAL edge position
(index into the split graph passed to sample_blocks) of every sampled edge —
not a frontier-local or subgraph-local position. Downstream consumers
(temporal_shap.py, 07_visualize.py, 09_w_ablation.py, explore/case_studies.py)
index the split graph directly with this field and require the true g-local
position. Mechanism: DGL 2.5's dgl.to_block unconditionally overwrites any
pre-set frontier.edata[dgl.EID] with frontier-local induced edge IDs
(utils/internal.py extract_edge_subframes, store_ids=True), so the g-local
EID must be restored AFTER to_block, composed through the induced IDs
to_block writes: block.edata[dgl.EID] = frontier.edata['_leid'][induced].
"""

import hashlib
import logging
from typing import Optional

import dgl
import numpy as np
import torch

logger = logging.getLogger(__name__)


def floyd_sample(rng: "np.random.Generator", p: int, k: int) -> np.ndarray:
    """Draw k distinct ints from [0, p) in O(k) time/space (Floyd's algorithm)."""
    sel: dict[int, None] = {}
    for j in range(p - k, p):
        t = int(rng.integers(0, j + 1))
        if t in sel:
            sel[j] = None
        else:
            sel[t] = None
    out = np.fromiter(sel.keys(), dtype=np.int64, count=k)
    return out


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
        deterministic: bool = True,
    ) -> None:
        """
        Args:
            fanouts:       per-hop max neighbors, e.g. [25, 15] for 2-layer SAGE.
                           fanouts[0] = first hop (closest to input features),
                           fanouts[-1] = last hop (closest to output).
            timestamp_key: edata key holding int64 FLOW_START_MILLISECONDS.
            deterministic: when True (default), the per-hop subsampling draw is
                           seeded from a hash of seed_eids, so identical
                           (g, seed_eids) always yields identical blocks. When
                           False, draws from a fresh np.random.default_rng()
                           each call — a debug/reproducibility escape hatch,
                           not a config-driven hyperparameter.
        """
        super().__init__()
        self.fanouts = fanouts
        self.timestamp_key = timestamp_key
        self.deterministic = deterministic
        self._csc_cache: dict[int, tuple] = {}  # id(g) -> (g, csc arrays)

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

        # Deterministic-from-seed RNG (or a fresh non-deterministic one) drives
        # the Floyd draw used whenever a node's eligible in-edge count exceeds
        # the fanout for that hop.
        if self.deterministic:
            h = hashlib.sha256(seed_eids.numpy().tobytes()).digest()
            rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        else:
            rng = np.random.default_rng()

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
            frontier = self._sample_frontier(g, curr_seeds, curr_cutoffs, fanout, rng)

            block = dgl.to_block(frontier, curr_seeds)
            # dgl.to_block unconditionally overwrites block.edata[dgl.EID]
            # with frontier-local induced edge IDs (DGL 2.5 internals), so
            # the g-local EID carried on the frontier as '_leid' is lost
            # unless restored here. Downstream consumers (temporal_shap.py,
            # visualize/w_ablation scripts) index g directly with
            # block.edata[dgl.EID] and require the true g-local position, so
            # restore it by composing through the induced IDs to_block just
            # wrote.
            block.edata[dgl.EID] = frontier.edata["_leid"][block.edata[dgl.EID]]
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
            frontier: graph returned by _sample_frontier for this hop.
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

    # ------------------------------------------------------------------
    # Searchsorted CSC + Floyd-draw candidate generation (spec 18/19)
    # ------------------------------------------------------------------

    def _get_csc(
        self, g: dgl.DGLGraph
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
    ]:
        """Return (indptr, in_src, in_ts, in_leid, geid, composite, big) for g,
        cached per graph.

        Precomputes a per-node timestamp-sorted in-adjacency (CSC-style):
        for destination node v, the in-edges with local EIDs
        in_leid[indptr[v]:indptr[v+1]] are sorted by timestamp ascending,
        with sources in_src[...] and timestamps in_ts[...] aligned.

        The last two returned members support the vectorized segmented
        searchsorted in `_sample_frontier` (spec 22 / spec 20 §2.3):
          composite[i] = in_ts[i] + dst_of_slot[i] * big
        with ``big = int(in_ts.max()) + 1``. Because CSC slots are grouped
        ascending by dst and ts-ascending within each segment, and every
        in_ts < big, ``composite`` is GLOBALLY non-decreasing — so a single
        np.searchsorted over all seeds reproduces the per-seed eligible count
        exactly. For an empty graph, ``big = 1`` and ``composite`` is empty.

        Cache is keyed by id(g) but stores (g, csc) and verifies `stored_g is
        g` on hit — a bare id(g)-keyed cache can otherwise return stale
        arrays if a graph object is garbage-collected and a new object
        reuses the same id() (CPython does reuse ids of freed objects).
        composite/big are graph-static, so they live in the same cached tuple
        and are built once per graph object.
        """
        key = id(g)
        cached = self._csc_cache.get(key)
        if cached is not None:
            stored_g, csc = cached
            if stored_g is g:
                return csc
            # id() reused by a different graph object — stale entry, rebuild.

        src, dst = g.edges()
        src = src.numpy().astype(np.int64)
        dst = dst.numpy().astype(np.int64)
        ts = g.edata[self.timestamp_key].numpy().astype(np.int64)
        n = g.num_nodes()

        # Precondition — assert, don't assume (CODING STANDARDS 7).
        assert np.all(np.diff(ts) >= 0), (
            "Edge timestamps are not globally non-decreasing in edge index; "
            "the stable group-by-dst CSC would not be per-segment sorted."
        )

        order = np.argsort(dst, kind="stable")  # stable ⇒ ts-sorted per segment
        in_src = src[order]
        in_ts = ts[order]
        in_leid = order  # local eid of each CSC slot
        counts = np.bincount(dst, minlength=n)
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])

        # Assert per-segment sortedness explicitly as well (belt and braces).
        seg_ok = np.ones(len(in_ts), dtype=bool)
        if len(in_ts) > 1:
            same_seg = np.repeat(np.arange(n), counts)
            seg_ok[1:] = (np.diff(in_ts) >= 0) | (same_seg[1:] != same_seg[:-1])
        assert bool(seg_ok.all()), "per-destination CSC segment not ts-sorted"

        geid = (
            g.edata[dgl.EID].numpy().astype(np.int64)[in_leid]
            if dgl.EID in g.edata
            else in_leid.copy()
        )

        # Composite key for segmented searchsorted (spec 22 / spec 20 §2.3).
        # CSC slots are grouped ascending-dst, ts-ascending within each segment, so
        #   composite[i] = in_ts[i] + dst_of_slot[i] * big
        # is GLOBALLY non-decreasing (across segments dst*big dominates because
        # every in_ts < big; within a segment in_ts is already non-decreasing). This
        # lets one np.searchsorted over ALL seeds reproduce the per-seed eligible
        # count exactly (spec 22 §2.3).
        if in_ts.size:
            # Precondition asserts (CODING STANDARDS 7) — load-bearing, not
            # decorative; without them the identity can silently break (spec 20 §2.8).
            assert int(in_ts.min()) >= 0, (
                "in_ts has negative timestamps; composite key would let a low-dst "
                "segment's slot collide into the previous segment's key range."
            )
            big = int(in_ts.max()) + 1  # tight, not arbitrary (spec 20 §2.3)
            assert (n - 1) * big + int(in_ts.max()) < 2**63, (
                "composite key overflow: (num_nodes-1)*big + max_ts must stay in "
                "int64 — fail loudly rather than silently wrap on a larger graph."
            )
            dst_of_slot = dst[order].astype(np.int64)          # dst aligned to CSC slots
            composite = in_ts + dst_of_slot * big
            # Belt-and-braces: the searchsorted precondition itself.
            if composite.size > 1:
                assert bool(np.all(np.diff(composite) >= 0)), (
                    "composite key not globally non-decreasing; segmented "
                    "searchsorted identity would be invalid."
                )
        else:
            big = 1
            composite = np.empty(0, dtype=np.int64)

        csc = (indptr, in_src, in_ts, in_leid, geid, composite, big)
        self._csc_cache[key] = (g, csc)
        return csc

    def eligible_local_eids(
        self, g: dgl.DGLGraph, node: int, cutoff: int
    ) -> np.ndarray:
        """Local EIDs of node's in-edges with ts <= cutoff, in timestamp order."""
        csc = self._get_csc(g)
        indptr, in_ts, in_leid = csc[0], csc[2], csc[3]
        lo, hi = int(indptr[node]), int(indptr[node + 1])
        p = int(np.searchsorted(in_ts[lo:hi], cutoff, side="right"))
        return in_leid[lo:lo + p]

    def _sample_frontier(
        self,
        g: dgl.DGLGraph,
        curr_seeds: torch.Tensor,
        cutoffs: torch.Tensor,
        fanout: int,
        rng: "np.random.Generator",
    ) -> dgl.DGLGraph:
        """Build one hop's frontier via searchsorted CSC + Floyd-draw sampling.

        Replaces the O(E)-per-hop `_filtered_subgraph` + dgl.sampling.
        sample_neighbors pair: for each seed node, the eligible in-edges
        (ts <= cutoff[node]) are located in O(log deg) via np.searchsorted on
        the precomputed per-node timestamp-sorted CSC, then either taken
        whole (eligible count <= fanout) or subsampled via an O(fanout)
        Floyd draw.

        Returns a frontier graph carrying edata[timestamp_key], edata['_leid']
        (g-local edge position — required by sample_blocks' EID-restoration
        line) and edata['_geid'] (global EID — carried for round-trip
        convenience, recoverable as g.edata[dgl.EID][block.edata[dgl.EID]]).

        The per-seed Python loop is now vectorized (spec 22): a single
        segmented composite-key np.searchsorted computes every seed's eligible
        count p; take-all seeds (p <= fanout) are built with one repeat/cumsum
        multi-range idiom; only the p>fanout hub seeds keep a Python
        floyd_sample loop, iterated in seeds-array order so output is
        bit-identical (same edges, same draws) to the pre-vectorization
        per-seed loop.

        Scope of that bit-identity: it holds for a SINGLE _sample_frontier
        call given identical (curr_seeds, cutoffs, rng). It does NOT extend to
        full multi-hop sample_blocks output — the frontier edge order changes
        (take-all block before subsample block), so a hop-1 reorder shifts the
        induced block.srcdata[NID] order, hence the next hop's curr_seeds order
        and the shared-rng floyd-draw sequence. Multi-hop sample_blocks output
        is therefore a valid, deterministic, leak-free sample but is NOT
        bit-identical to the pre-vectorization build in the mixed
        take-all/subsample regime (safe: message passing is permutation-
        invariant and every temporal-cutoff invariant still holds).
        """
        indptr, in_src, in_ts, in_leid, geid, composite, big = self._get_csc(g)
        seeds = curr_seeds.numpy().astype(np.int64)
        cut = cutoffs.numpy().astype(np.int64)

        # Cutoff-domain precondition (spec 22 §2.4) — the composite identity
        # requires every cutoff c < big (equivalently c <= in_ts.max()); a
        # cutoff above in_ts.max() could over-count into higher-dst segments,
        # a temporal-leakage-critical failure. Guarded for the empty-seed /
        # empty-graph case. Boundary c == in_ts.max() passes (big = max+1).
        if seeds.size and in_ts.size:
            assert int(cut[seeds].max()) < big, (
                "a cutoff exceeds max timestamp; the composite key would over-count "
                "into higher-dst segments (temporal-leakage-critical — fail loud)."
            )

        # p[i] = eligible in-edge count of seed[i] via one segmented searchsorted
        # over the cached composite key (spec 22 §2.3). p = pos_global - lo.
        if seeds.size and in_ts.size:
            lo = indptr[seeds]                               # (S,)
            keys = cut[seeds] + seeds * big                  # int64 (§2.4 bounds it)
            pos_global = np.searchsorted(composite, keys, side="right")
            p = pos_global - lo                              # (S,) >= 0
        else:
            lo = indptr[seeds] if seeds.size else np.empty(0, dtype=np.int64)
            p = np.zeros(seeds.shape[0], dtype=np.int64)

        nz = p > 0
        sub_mask = nz & (p > fanout)

        # Take-all branch (0 < p <= fanout): vectorized repeat/cumsum multi-range
        # idiom (spec 22 §3). Each seed's block is [lo, lo+1, ..., lo+p-1],
        # identical to the per-seed np.arange(lo, lo+p). No RNG consumed.
        take_mask = nz & (p <= fanout)
        lo_take = lo[take_mask]
        p_take = p[take_mask]
        seeds_take = seeds[take_mask]
        total_take = int(p_take.sum())
        if total_take:
            ends = np.cumsum(p_take)
            starts = ends - p_take
            ramp = np.arange(total_take) - np.repeat(starts, p_take)
            pos_take = np.repeat(lo_take, p_take) + ramp
            dst_take = np.repeat(seeds_take, p_take)
        else:
            pos_take = np.empty(0, dtype=np.int64)
            dst_take = np.empty(0, dtype=np.int64)

        # Subsample branch (p > fanout): a Python loop over ONLY the hub seeds,
        # in ascending seeds-array order, issuing floyd_sample(rng, p, fanout)
        # in the exact same relative order as the pre-vectorization per-seed
        # loop — bit-identical RNG provenance (spec 22 §4).
        sub_idx = np.nonzero(sub_mask)[0]        # ascending == seeds-array order
        pos_sub_parts, dst_sub_parts = [], []
        for i in sub_idx:
            v = int(seeds[i]); lv = int(lo[i]); pv = int(p[i])
            chosen = lv + floyd_sample(rng, pv, fanout)      # SAME call, SAME order
            pos_sub_parts.append(chosen)
            dst_sub_parts.append(np.full(fanout, v, dtype=np.int64))
        if pos_sub_parts:
            pos_sub = np.concatenate(pos_sub_parts)
            dst_sub = np.concatenate(dst_sub_parts)
        else:
            pos_sub = np.empty(0, dtype=np.int64)
            dst_sub = np.empty(0, dtype=np.int64)

        # Frontier assembly from one combined pos array (spec 22 §5) — all of
        # timestamp/_leid/_geid/fsrc indexed by the same pos, preserving the
        # block-EID contract by construction.
        pos = np.concatenate([pos_take, pos_sub])
        fdst = np.concatenate([dst_take, dst_sub])
        fsrc = in_src[pos]

        frontier = dgl.graph(
            (torch.from_numpy(fsrc), torch.from_numpy(fdst)),
            num_nodes=g.num_nodes(),
        )
        frontier.edata[self.timestamp_key] = torch.from_numpy(in_ts[pos])
        frontier.edata["_leid"] = torch.from_numpy(in_leid[pos])
        frontier.edata["_geid"] = torch.from_numpy(geid[pos])
        return frontier
