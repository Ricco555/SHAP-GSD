"""
Extract top-K connected explanatory subgraph from temporal SHAP values.

Algorithm:
  1. Rank all neighbor edges by |φ_T| descending.
  2. Init subgraph with target edge e.
  3. Greedily add the highest-|φ_T| edge that shares ≥ 1 endpoint
     with the current subgraph.
  4. Stop at K edges or when no remaining edges are connected.

Default K=10 (configurable via cfg['explainer']['subgraph_k']).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def extract_top_k_subgraph(
    neighbor_shap: list[tuple[int, float, float]],
    target_edge_id: int,
    target_src: int,
    target_dst: int,
    edge_endpoints: dict[int, tuple[int, int]],
    k: int = 10,
) -> list[tuple[int, float]]:
    """Greedily extract the top-K connected subgraph from temporal φ values.

    Args:
        neighbor_shap:   list of (global_eid, timestamp_ms, phi) from TemporalSHAP,
                         already sorted by |phi| descending.
        target_edge_id:  global EID of the target edge (anchor for connectivity).
        target_src:      global node ID of target edge source.
        target_dst:      global node ID of target edge destination.
        edge_endpoints:  dict mapping global_eid → (src_nid, dst_nid) for all
                         neighbor edges (needed for connectivity check).
        k:               maximum subgraph size (number of edges, including target).

    Returns:
        List of (global_eid, shap_weight) tuples for the top-K connected subgraph,
        in order of inclusion. The target edge is not included (it is the anchor).
    """
    if not neighbor_shap:
        return []

    # Track node set in current subgraph
    subgraph_nodes: set[int] = {target_src, target_dst}
    subgraph_edges: list[tuple[int, float]] = []

    # Sort by |phi| descending (caller already sorts, but be safe)
    ranked = sorted(neighbor_shap, key=lambda t: abs(t[2]), reverse=True)

    remaining = list(ranked)

    while remaining and len(subgraph_edges) < k:
        added_any = False
        for i, (geid, _ts, phi) in enumerate(remaining):
            endpoints = edge_endpoints.get(geid)
            if endpoints is None:
                continue
            src, dst = endpoints
            if src in subgraph_nodes or dst in subgraph_nodes:
                subgraph_edges.append((geid, phi))
                subgraph_nodes.add(src)
                subgraph_nodes.add(dst)
                remaining.pop(i)
                added_any = True
                break  # restart from highest-ranked remaining

        if not added_any:
            # No remaining edge is connected to the current subgraph
            break

    logger.debug(
        f"Subgraph extractor: selected {len(subgraph_edges)}/{len(neighbor_shap)} "
        f"neighbor edges (k={k})"
    )
    return subgraph_edges
