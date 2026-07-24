"""
Gateway-distance diagnostic — core computational logic for pipeline phase 14.

Refines ``specs/05_graph_node_distance.md`` (approved, local-only). Committed
scope is **Stage 0 only**: a read-only, no-model, no-GPU topology measurement
of the malicious-only training subgraph. This module never feeds its results
back into ``model.num_layers``/sampler code (that would be Stage 2, out of
scope for this implementation).

DEFINITIONAL RESOLUTION (read this before touching anything below)
--------------------------------------------------------------------
Spec 05 defines ``d_gw`` twice, inconsistently: the literal §3.2/§3.3 text
computes distance to the *boundary node set G* (internal nodes with >=1
boundary-crossing edge), which is identically 0 for every victim on a direct
-injection dataset (the attacker->victim edge itself is boundary-crossing).
§1/§2-F2/§4/§9.5 instead assert ``d_gw = 1`` for UNSW victims and require a
4-node chain fixture to yield ``k* == 4``. Both readings cannot hold at once.

This module commits to the following resolution (see the implementation
spec for the full derivation):

    d_gw(v) := shortest REVERSE-hop distance (following edges backward,
    the same "in" direction TemporalNeighborSampler uses — see
    src/model/temporal_sampler.py) from v to the nearest node NOT in
    V_int (an external/unprotected node), computed over the
    malicious-only edge set. d_gw(v) = 0 iff v is not in V_int; missing
    from the result dict (treated as +inf by callers) if no external
    node is reverse-reachable from v.

Equivalently, and far cheaper to compute: a single multi-source FORWARD BFS
seeded at distance 0 on every external-role node that appears as a malicious
source, propagated forward along malicious edges, gives the identical
distances (same edges, opposite traversal order) — that is what
``compute_d_gw`` actually implements.

The boundary set ``G`` is retained as a derived REPORTING diagnostic only
(``compute_gateway_set``): ``G := {v in V_int : d_gw(v) == 1}`` — internal
nodes exactly one hop from the external side — never the BFS target.

Convention note: this module never accepts an "edge_dir" config knob. The
reverse-hop / forward-BFS equivalence above is a fixed algorithmic identity,
not a tunable choice, so it is documented here rather than exposed as a dead
config key.
"""

from __future__ import annotations

import ipaddress
import json
import logging
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import dgl  # noqa: F401  (only used for type hints / isinstance checks below)
except ImportError:  # pragma: no cover - dgl is a hard runtime dependency elsewhere
    dgl = None  # type: ignore[assignment]

from src.data.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 3.1 — Role assignment
# ---------------------------------------------------------------------------

def _ip_in_any_prefix(ip_str: str, prefixes: list[str]) -> bool:
    """Return True if ``ip_str`` falls within any CIDR in ``prefixes``.

    Mirrors ``scripts/12_novelty_audit.py``'s existing use of the stdlib
    ``ipaddress`` module — no new dependency introduced.

    Args:
        ip_str: dotted-quad IPv4 (or IPv6) address string.
        prefixes: list of CIDR strings, e.g. ``["10.0.0.0/8"]``.

    Returns:
        True if the address is a member of any network in ``prefixes``.
        False if the address string is unparseable or matches nothing.
    """
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    for p in prefixes:
        try:
            if addr in ipaddress.ip_network(p):
                return True
        except ValueError:
            continue
    return False


def assign_roles(
    node_id_map: dict[str, int],
    mal_src_ids: np.ndarray,
    mal_dst_ids: np.ndarray,
    internal_prefixes: str | list[str] = "auto",
) -> tuple[np.ndarray, str]:
    """Classify every node as internal (protected) or external (unprotected).

    Three modes:
      - ``"auto"``: reuse ``GraphBuilder.compute_is_internal_array`` (RFC1918)
        OR-ed with a local loopback check, per node. If the resulting set has
        zero internal nodes among malicious *destinations* specifically, fall
        back to treating every malicious-destination node as internal (the
        UNSW-NB15 direct-public-IP case).
      - explicit list of CIDR strings: build the internal mask from those
        prefixes only.

    Args:
        node_id_map: IP string -> node id, as saved by GraphBuilder.
        mal_src_ids: malicious-edge source node ids (n_mal,), int64.
        mal_dst_ids: malicious-edge destination node ids (n_mal,), int64.
        internal_prefixes: ``"auto"`` or a list of CIDR strings.

    Returns:
        Tuple ``(is_internal, mode)`` where ``is_internal`` is a bool array
        of shape ``(n_nodes,)`` indexed by node id, and ``mode`` is one of
        ``"rfc1918_auto"``, ``"victim_fallback"``, ``"explicit"``.
    """
    n_nodes = len(node_id_map)

    if internal_prefixes == "auto":
        is_internal = GraphBuilder.compute_is_internal_array(node_id_map).astype(bool)
        # Local loopback check — GraphBuilder's RFC1918 helper does not
        # cover 127.0.0.0/8; OR it in here rather than modifying the
        # shared, load-bearing graph_builder.py function (out of scope).
        ip_by_id = {idx: ip for ip, idx in node_id_map.items()}
        for idx in range(n_nodes):
            if is_internal[idx]:
                continue
            ip = ip_by_id.get(idx)
            if ip is None:
                continue
            try:
                if ipaddress.ip_address(ip).is_loopback:
                    is_internal[idx] = True
            except ValueError:
                continue

        unique_mal_dst = np.unique(mal_dst_ids) if mal_dst_ids.size else np.array([], dtype=np.int64)
        n_internal_among_dst = int(is_internal[unique_mal_dst].sum()) if unique_mal_dst.size else 0
        if n_internal_among_dst == 0:
            logger.warning(
                "assign_roles: zero internal nodes among malicious destinations "
                "under RFC1918/loopback auto-detection — falling back to "
                "victim-inference mode (every malicious-flow destination node "
                "treated as internal). Inferred internal set size: %d",
                int(unique_mal_dst.size),
            )
            is_internal = np.zeros(n_nodes, dtype=bool)
            is_internal[unique_mal_dst] = True
            mode = "victim_fallback"
        else:
            mode = "rfc1918_auto"
    else:
        prefixes = list(internal_prefixes)
        is_internal = np.zeros(n_nodes, dtype=bool)
        for ip, idx in node_id_map.items():
            if _ip_in_any_prefix(ip, prefixes):
                is_internal[idx] = True
        mode = "explicit"

    assert is_internal.shape[0] == n_nodes, "role assignment must cover every node"
    return is_internal, mode


# ---------------------------------------------------------------------------
# 3.2 — Malicious-only subgraph extraction
# ---------------------------------------------------------------------------

def build_malicious_subgraph(
    g_train: "dgl.DGLGraph",
    label_map: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the malicious-only (non-Benign) edge subset of a train graph.

    Args:
        g_train: training-split DGL graph with ``edata["label"]`` set
            (integer class ids, per ``GraphBuilder.build_split_graph``).
        label_map: class name -> integer id mapping (``artifacts/label_map.json``).
            ``label_map["Benign"]`` is looked up rather than assuming the
            literal int 0, per CLAUDE.md "do NOT hardcode class names".

    Returns:
        Tuple ``(mal_src_ids, mal_dst_ids, mal_labels)``, int64 arrays of
        equal length, index-aligned to ``g_train``'s (chronological) edge
        order.
    """
    benign_id = label_map["Benign"]
    labels = g_train.edata["label"].numpy()
    src, dst = g_train.edges(form="uv")
    src = src.numpy().astype(np.int64)
    dst = dst.numpy().astype(np.int64)

    mask = labels != benign_id
    mal_src_ids = src[mask]
    mal_dst_ids = dst[mask]
    mal_labels = labels[mask].astype(np.int64)
    return mal_src_ids, mal_dst_ids, mal_labels


# ---------------------------------------------------------------------------
# 3.3 — Gateway distance
# ---------------------------------------------------------------------------

def compute_d_gw(
    mal_src_ids: np.ndarray,
    mal_dst_ids: np.ndarray,
    is_internal: np.ndarray,
) -> dict[int, float]:
    """Compute gateway distance for every node reachable from an external frontier.

    Implements the reverse-hop-distance-to-nearest-external-node definition
    via the equivalent (and much cheaper) multi-source forward BFS: seed at
    distance 0 every external node (``is_internal[r] is False``) that
    appears in ``mal_src_ids``, then propagate forward along malicious edges.

    Args:
        mal_src_ids: malicious-edge source node ids (n_mal,), int64.
        mal_dst_ids: malicious-edge destination node ids (n_mal,), int64.
        is_internal: bool array (n_nodes,) from ``assign_roles``.

    Returns:
        Mapping ``{node_id: distance}`` for every node reached by the BFS.
        Node ids never reached are absent (callers treat missing as +inf).
    """
    adjacency: dict[int, set[int]] = {}
    for s, d in zip(mal_src_ids.tolist(), mal_dst_ids.tolist()):
        adjacency.setdefault(s, set()).add(d)

    dist: dict[int, float] = {}
    queue: deque[int] = deque()

    roots = sorted({s for s in mal_src_ids.tolist() if not is_internal[s]})
    for r in roots:
        if r not in dist:
            dist[r] = 0.0
            queue.append(r)

    while queue:
        u = queue.popleft()
        for v in sorted(adjacency.get(u, ())):
            if v not in dist:
                dist[v] = dist[u] + 1.0
                queue.append(v)

    return dist


def derive_k_star(
    d_gw: dict[int, float],
    mal_dst_ids: np.ndarray,
    is_internal: np.ndarray,
    k_max: int,
) -> tuple[int, dict[str, Any]]:
    """Derive the recommended neighbor-sampling depth ``k*`` from ``d_gw``.

    Args:
        d_gw: output of ``compute_d_gw``.
        mal_dst_ids: malicious-edge destination node ids (n_mal,), int64.
        is_internal: bool array (n_nodes,) from ``assign_roles``.
        k_max: upper clip on the reported ``k*`` (config: ``topology.gateway_distance.k_max``).

    Returns:
        Tuple ``(k_star, stats)`` where ``stats`` has keys ``max_d_gw``,
        ``n_measured``, ``n_unreachable``.
    """
    unique_dst = np.unique(mal_dst_ids) if mal_dst_ids.size else np.array([], dtype=np.int64)
    internal_dst = [int(v) for v in unique_dst if is_internal[v]]

    finite_vals = [d_gw[v] for v in internal_dst if v in d_gw]
    n_unreachable = len(internal_dst) - len(finite_vals)

    if not finite_vals:
        logger.warning(
            "derive_k_star: no internal malicious destination has a finite "
            "gateway distance — degenerate input, clamping k_star to k_max=%d",
            k_max,
        )
        k_star = k_max
        max_d_gw: Optional[float] = None
    else:
        max_d_gw = max(finite_vals)
        k_star = int(min(max_d_gw + 1, k_max))

    stats = {
        "max_d_gw": max_d_gw,
        "n_measured": len(finite_vals),
        "n_unreachable": n_unreachable,
    }
    return k_star, stats


def compute_gateway_set(
    is_internal: np.ndarray,
    d_gw: dict[int, float],
) -> tuple[set[int], bool]:
    """Derive the boundary node set ``G`` as a reporting diagnostic.

    ``G := {v : is_internal[v] and d_gw[v] == 1}`` — internal nodes exactly
    one hop from the external side. This is NOT the BFS target (see module
    docstring) — purely a reporting quantity per spec 05 §3.2/§6 item 1.

    Args:
        is_internal: bool array (n_nodes,) from ``assign_roles``.
        d_gw: output of ``compute_d_gw``.

    Returns:
        Tuple ``(G, is_chokepoint)`` where ``is_chokepoint`` is True iff
        ``len(G) == 1``.
    """
    gateway = {v for v, d in d_gw.items() if is_internal[v] and d == 1}
    return gateway, len(gateway) == 1


# ---------------------------------------------------------------------------
# 3.4 — Per-class structural profile
# ---------------------------------------------------------------------------

def per_class_stats(
    mal_src_ids: np.ndarray,
    mal_dst_ids: np.ndarray,
    mal_labels: np.ndarray,
    d_gw: dict[int, float],
    int_to_name: dict[int, str],
) -> dict[str, dict[str, Any]]:
    """Compute per-attack-class gateway-distance and endpoint-count stats.

    Args:
        mal_src_ids: malicious-edge source node ids (n_mal,), int64.
        mal_dst_ids: malicious-edge destination node ids (n_mal,), int64.
        mal_labels: malicious-edge integer class labels (n_mal,), int64.
        d_gw: output of ``compute_d_gw``.
        int_to_name: integer class id -> class name (inverse of label_map).

    Returns:
        Mapping ``{class_name: {"mean", "max", "std", "n_dst_nodes",
        "n_src_nodes", "n_unreachable"}}``.
    """
    result: dict[str, dict[str, Any]] = {}
    for c in sorted(set(mal_labels.tolist())):
        rows = mal_labels == c
        dst_c = np.unique(mal_dst_ids[rows])
        src_c = np.unique(mal_src_ids[rows])

        finite_vals = [d_gw[v] for v in dst_c.tolist() if v in d_gw]
        n_unreachable = len(dst_c) - len(finite_vals)

        name = int_to_name[int(c)]
        result[name] = {
            "mean": float(np.mean(finite_vals)) if finite_vals else None,
            "max": float(np.max(finite_vals)) if finite_vals else None,
            "std": float(np.std(finite_vals)) if finite_vals else None,
            "n_dst_nodes": int(len(dst_c)),
            "n_src_nodes": int(len(src_c)),
            "n_unreachable": int(n_unreachable),
        }
    return result


def compute_pivot_nodes(
    mal_src_ids: np.ndarray,
    mal_dst_ids: np.ndarray,
    mal_labels: Optional[np.ndarray] = None,
    int_to_name: Optional[dict[int, str]] = None,
) -> tuple[set[int], dict[str, int]]:
    """Identify pivot nodes — malicious destinations that are also malicious sources.

    A pivot node is a kill-chain signature: a node that was attacked (a
    malicious destination) and later originates malicious traffic itself
    (a malicious source), anywhere in the malicious subgraph, any class.

    Args:
        mal_src_ids: malicious-edge source node ids (n_mal,), int64.
        mal_dst_ids: malicious-edge destination node ids (n_mal,), int64.
        mal_labels: optional per-edge class labels, required (with
            ``int_to_name``) to compute the per-class pivot breakdown.
        int_to_name: optional integer class id -> class name mapping.

    Returns:
        Tuple ``(P, per_class_count)`` where ``P`` is the global pivot node
        id set, and ``per_class_count`` maps class name -> number of that
        class's victims that are also pivots (empty dict if
        ``mal_labels``/``int_to_name`` not given).
    """
    dst_set = set(np.unique(mal_dst_ids).tolist())
    src_set = set(np.unique(mal_src_ids).tolist())
    pivots = dst_set & src_set

    per_class_count: dict[str, int] = {}
    if mal_labels is not None and int_to_name is not None:
        for c in sorted(set(mal_labels.tolist())):
            rows = mal_labels == c
            dst_c = set(np.unique(mal_dst_ids[rows]).tolist())
            name = int_to_name[int(c)]
            per_class_count[name] = len(dst_c & pivots)

    return pivots, per_class_count


def compute_diameter(
    mal_src_ids: np.ndarray,
    mal_dst_ids: np.ndarray,
    mal_labels: Optional[np.ndarray] = None,
    int_to_name: Optional[dict[int, str]] = None,
) -> dict[str, Any]:
    """Compute the undirected diameter of the malicious subgraph, per component.

    Args:
        mal_src_ids: malicious-edge source node ids (n_mal,), int64.
        mal_dst_ids: malicious-edge destination node ids (n_mal,), int64.
        mal_labels: optional per-edge class labels, to also compute the
            per-class diameter breakdown (restricting edges to that class
            first).
        int_to_name: optional integer class id -> class name mapping.

    Returns:
        ``{"global": {"diameter", "n_components", "component_diameters"},
        "per_class": {name: {...}}}`` if ``mal_labels``/``int_to_name`` are
        given, otherwise just the flat global stats dict
        ``{"diameter", "n_components", "component_diameters"}``.
    """
    import networkx as nx

    def _diameter_stats(src: np.ndarray, dst: np.ndarray) -> dict[str, Any]:
        g = nx.Graph()
        g.add_edges_from(zip(src.tolist(), dst.tolist()))
        if g.number_of_nodes() == 0:
            return {"diameter": None, "n_components": 0, "component_diameters": []}
        component_diameters = [
            nx.diameter(g.subgraph(comp)) for comp in nx.connected_components(g)
        ]
        return {
            "diameter": max(component_diameters) if component_diameters else None,
            "n_components": len(component_diameters),
            "component_diameters": component_diameters,
        }

    global_stats = _diameter_stats(mal_src_ids, mal_dst_ids)

    if mal_labels is None or int_to_name is None:
        return global_stats

    per_class: dict[str, dict[str, Any]] = {}
    for c in sorted(set(mal_labels.tolist())):
        rows = mal_labels == c
        name = int_to_name[int(c)]
        per_class[name] = _diameter_stats(mal_src_ids[rows], mal_dst_ids[rows])

    return {"global": global_stats, "per_class": per_class}


# ---------------------------------------------------------------------------
# 3.5 — Train-half vs train-half drift check
# ---------------------------------------------------------------------------

def drift_check(
    g_train: "dgl.DGLGraph",
    label_map: dict[str, int],
    is_internal: np.ndarray,
    k_max: int,
) -> dict[str, Any]:
    """Compare gateway-distance statistics between the two halves of the train split.

    Edge insertion order in ``g_train`` is chronological by construction
    (``GraphBuilder.build_split_graph``), so no separate timestamp re-sort is
    needed — the edges are simply split at the midpoint index. This never
    touches val/test data (CLAUDE.md Invariants 2/5): the function signature
    accepts a single training-split graph only.

    Args:
        g_train: training-split DGL graph (chronologically ordered edges).
        label_map: class name -> integer id mapping.
        is_internal: bool array (n_nodes,) from ``assign_roles``, computed
            once on the full train split and reused for both halves (role
            assignment itself is not re-derived per half).
        k_max: same clip used by ``derive_k_star``.

    Returns:
        Dict with keys ``k_star_first_half``, ``k_star_second_half``,
        ``delta_k_star``, ``mean_d_gw_first_half``, ``mean_d_gw_second_half``,
        ``delta_mean_d_gw``.
    """
    n_edges = g_train.num_edges()
    mid = n_edges // 2

    benign_id = label_map["Benign"]
    labels = g_train.edata["label"].numpy()
    src, dst = g_train.edges(form="uv")
    src = src.numpy().astype(np.int64)
    dst = dst.numpy().astype(np.int64)

    def _half_stats(lo: int, hi: int) -> tuple[int, Optional[float]]:
        half_labels = labels[lo:hi]
        half_src = src[lo:hi]
        half_dst = dst[lo:hi]
        mask = half_labels != benign_id
        h_src, h_dst = half_src[mask], half_dst[mask]
        d_gw_half = compute_d_gw(h_src, h_dst, is_internal)
        k_star_half, stats = derive_k_star(d_gw_half, h_dst, is_internal, k_max)
        finite_vals = [
            d_gw_half[v]
            for v in np.unique(h_dst).tolist()
            if is_internal[v] and v in d_gw_half
        ]
        mean_d_gw = float(np.mean(finite_vals)) if finite_vals else None
        return k_star_half, mean_d_gw

    k_star_first, mean_first = _half_stats(0, mid)
    k_star_second, mean_second = _half_stats(mid, n_edges)

    delta_mean: Optional[float]
    if mean_first is not None and mean_second is not None:
        delta_mean = mean_second - mean_first
    else:
        delta_mean = None

    return {
        "k_star_first_half": k_star_first,
        "k_star_second_half": k_star_second,
        "delta_k_star": k_star_second - k_star_first,
        "mean_d_gw_first_half": mean_first,
        "mean_d_gw_second_half": mean_second,
        "delta_mean_d_gw": delta_mean,
    }


# ---------------------------------------------------------------------------
# 3.6 — Post-hoc correlation with per-class F1
# ---------------------------------------------------------------------------

def spearman_dgw_vs_f1(
    per_class: dict[str, dict[str, Any]],
    metrics_json_path: Optional[Path],
) -> dict[str, Any]:
    """Correlate per-class mean gateway distance against per-class test F1.

    Gracefully degrades (returns ``available: False`` rather than raising)
    when ``metrics_json_path`` is missing — e.g. a ``--to-phase``/``--from-phase``
    sub-range that excluded phase 05. This must never short-circuit the rest
    of the diagnostic: callers compute this step last, after items 1-3 are
    already assembled.

    Args:
        per_class: output of ``per_class_stats``.
        metrics_json_path: path to ``artifacts/evaluation/metrics.json``, or
            None if it is known not to exist.

    Returns:
        ``{"available": bool, "spearman_rho": float | None,
        "p_value": float | None, "note": str | None}``.
    """
    if metrics_json_path is None or not metrics_json_path.exists():
        logger.warning(
            "spearman_dgw_vs_f1: artifacts/evaluation/metrics.json not found — "
            "skipping post-hoc d_gw/F1 correlation (items 1-3 are unaffected "
            "and still emitted)."
        )
        return {
            "available": False,
            "spearman_rho": None,
            "p_value": None,
            "note": (
                "artifacts/evaluation/metrics.json not found — phase 05 has "
                "not run for this range, or a --to-phase/--from-phase "
                "sub-range excluded it."
            ),
        }

    from scipy.stats import spearmanr

    metrics = json.loads(metrics_json_path.read_text())
    per_class_metrics = metrics.get("per_class", {})

    names = sorted(set(per_class.keys()) & set(per_class_metrics.keys()))
    d_gw_means = [per_class[n]["mean"] for n in names]
    f1_scores = [per_class_metrics[n]["f1"] for n in names]

    # Drop classes where mean d_gw is None (no finite destinations).
    paired = [(d, f) for d, f in zip(d_gw_means, f1_scores) if d is not None]
    if len(paired) < 2:
        return {
            "available": True,
            "spearman_rho": None,
            "p_value": None,
            "note": "fewer than 2 classes with finite mean d_gw — correlation undefined",
        }

    d_vals, f_vals = zip(*paired)
    rho, p_value = spearmanr(d_vals, f_vals)

    if rho is None or (isinstance(rho, float) and np.isnan(rho)):
        return {
            "available": True,
            "spearman_rho": None,
            "p_value": None,
            "note": "zero variance in per-class mean d_gw — correlation undefined",
        }

    return {
        "available": True,
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "note": None,
    }


# ---------------------------------------------------------------------------
# 3.7 — Top-level orchestration (I/O boundary)
# ---------------------------------------------------------------------------

def run_gateway_distance_diagnostic(
    graph_dir: Path,
    artifacts_dir: Path,
    topology_cfg: dict[str, Any],
    metrics_json_path: Optional[Path],
) -> dict[str, Any]:
    """Run the full Stage-0 gateway-distance diagnostic and assemble the result dict.

    The only function in this module that touches disk. Not exercised by the
    pytest gate (its inputs are real artifacts that do not exist at gate
    time, immediately after phase 02) — covered instead by
    ``scripts/14_gateway_distance.py``'s own run.

    Args:
        graph_dir: ``cfg["graph"]["dir"]`` (run.dir-prefixed), containing
            ``train.bin`` and ``node_id_map.json``.
        artifacts_dir: ``cfg["output"]["artifacts_dir"]`` (run.dir-prefixed),
            containing ``label_map.json``.
        topology_cfg: ``cfg["topology"]["gateway_distance"]`` dict — keys
            ``internal_prefixes``, ``drift_check``, ``k_max`` (``enabled`` is
            handled by the calling script, not here).
        metrics_json_path: path to ``artifacts/evaluation/metrics.json``, or
            None if the caller has already determined it does not exist.

    Returns:
        Dict matching the phase-14 output-contract schema (see
        ``specs/07_phase14_gateway_distance_impl.md`` §7.1).
    """
    if dgl is None:  # pragma: no cover - dgl is a hard dependency in practice
        raise ImportError("dgl is required to run the gateway-distance diagnostic")

    graphs, _ = dgl.load_graphs(str(Path(graph_dir) / "train.bin"))
    g_train = graphs[0]
    node_id_map = GraphBuilder.load_node_id_map(graph_dir)

    label_map_path = Path(artifacts_dir) / "label_map.json"
    with open(label_map_path) as f:
        label_map = json.load(f)
    int_to_name = {v: k for k, v in label_map.items()}

    internal_prefixes = topology_cfg.get("internal_prefixes", "auto")
    k_max = int(topology_cfg.get("k_max", 4))
    do_drift_check = bool(topology_cfg.get("drift_check", True))

    mal_src_ids, mal_dst_ids, mal_labels = build_malicious_subgraph(g_train, label_map)
    is_internal, mode = assign_roles(node_id_map, mal_src_ids, mal_dst_ids, internal_prefixes)

    d_gw = compute_d_gw(mal_src_ids, mal_dst_ids, is_internal)
    # derive_k_star's own stats duplicate global_d_gw_stats (computed below
    # independently); only k_star itself is needed here.
    k_star, _ = derive_k_star(d_gw, mal_dst_ids, is_internal, k_max)
    gateway_set, is_chokepoint = compute_gateway_set(is_internal, d_gw)

    global_dst = np.unique(mal_dst_ids) if mal_dst_ids.size else np.array([], dtype=np.int64)
    internal_dst = [int(v) for v in global_dst if is_internal[v]]
    global_finite = [d_gw[v] for v in internal_dst if v in d_gw]
    global_d_gw_stats = {
        "mean": float(np.mean(global_finite)) if global_finite else None,
        "max": float(np.max(global_finite)) if global_finite else None,
        "std": float(np.std(global_finite)) if global_finite else None,
        "n_measured": len(global_finite),
        "n_unreachable": len(internal_dst) - len(global_finite),
    }

    per_class = per_class_stats(mal_src_ids, mal_dst_ids, mal_labels, d_gw, int_to_name)
    diameter = compute_diameter(mal_src_ids, mal_dst_ids, mal_labels, int_to_name)
    pivots, per_class_pivots = compute_pivot_nodes(mal_src_ids, mal_dst_ids, mal_labels, int_to_name)

    if do_drift_check:
        drift = drift_check(g_train, label_map, is_internal, k_max)
        drift["enabled"] = True
    else:
        drift = {"enabled": False}

    correlation = spearman_dgw_vs_f1(per_class, metrics_json_path)

    n_nodes = len(node_id_map)
    n_internal = int(is_internal.sum())

    result: dict[str, Any] = {
        "role_assignment": {
            "mode": mode,
            "n_nodes_total": n_nodes,
            "n_internal": n_internal,
            "n_external": n_nodes - n_internal,
        },
        "gateway_boundary": {
            "n_gateway_nodes": len(gateway_set),
            "is_single_chokepoint": is_chokepoint,
        },
        "d_gw": {
            "global": global_d_gw_stats,
            "per_class": per_class,
        },
        "diameter": diameter,
        "pivot_nodes": {
            "global_count": len(pivots),
            "per_class_count": per_class_pivots,
        },
        "k_star": k_star,
        "k_max": k_max,
        "drift_check": drift,
        "posthoc_correlation": correlation,
    }
    return result
