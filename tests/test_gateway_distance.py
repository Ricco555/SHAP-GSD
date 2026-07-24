"""
Tests for gateway-distance diagnostic (src/data/gateway_distance.py) — phase 14.

Fully self-contained on synthetic data (bipartite + chain fixtures), following
the same convention as tests/test_temporal_sampler.py / tests/test_node_state.py.
No dependency on graphs/*.bin, artifacts/label_map.json, or tests/_paths.py's
SHAP_GSD_CONFIG fixture — this file must remain gate-safe: it runs immediately
after phase 02 (scripts/run_dataset.py's pytest gate), strictly before phase 14
or any dataset run.dir artifacts exist.

DEFINITIONAL NOTE: d_gw is the reverse-hop distance to the nearest external
node (equivalently, the result of a forward multi-source BFS seeded at
external "frontier" nodes, propagated along malicious edges) — see
src/data/gateway_distance.py's module docstring for the full resolution of
spec 05's internal inconsistency. Expected values below (e.g.
d_gw(victim) == 1 for the bipartite fixture, not 0) follow that resolution,
not a literal "distance to boundary set G" reading.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dgl

from src.data.gateway_distance import (
    assign_roles,
    build_malicious_subgraph,
    compute_d_gw,
    compute_diameter,
    compute_gateway_set,
    compute_pivot_nodes,
    derive_k_star,
    drift_check,
    per_class_stats,
    spearman_dgw_vs_f1,
)

LABEL_MAP = {"Benign": 0, "AttackClass": 1}
INT_TO_NAME = {v: k for k, v in LABEL_MAP.items()}


# ---------------------------------------------------------------------------
# Fixtures (in-test, hand-built, no disk I/O)
# ---------------------------------------------------------------------------

def _bipartite_fixture(n_attackers: int = 4, n_victims: int = 10):
    """4 external attacker nodes -> n_victims internal victim nodes (complete bipartite).

    Attacker IPs are public; victim IPs are RFC1918, so "auto" role assignment
    correctly infers roles via GraphBuilder.compute_is_internal_array without
    needing the victim-fallback path (that path is exercised separately by
    ``test_assign_roles_victim_fallback``).
    """
    node_id_map: dict[str, int] = {}
    idx = 0
    attacker_ids = []
    for i in range(n_attackers):
        node_id_map[f"8.8.8.{i + 1}"] = idx
        attacker_ids.append(idx)
        idx += 1
    victim_ids = []
    for i in range(n_victims):
        node_id_map[f"10.0.0.{i + 1}"] = idx
        victim_ids.append(idx)
        idx += 1

    src, dst = [], []
    for a in attacker_ids:
        for v in victim_ids:
            src.append(a)
            dst.append(v)

    mal_src = np.array(src, dtype=np.int64)
    mal_dst = np.array(dst, dtype=np.int64)
    return node_id_map, mal_src, mal_dst, attacker_ids, victim_ids


def _chain_fixture():
    """E (external) -> A -> B -> C, three internal nodes A/B/C, single chain."""
    node_id_map = {
        "8.8.8.1": 0,   # E
        "10.0.0.1": 1,  # A
        "10.0.0.2": 2,  # B
        "10.0.0.3": 3,  # C
    }
    mal_src = np.array([0, 1, 2], dtype=np.int64)
    mal_dst = np.array([1, 2, 3], dtype=np.int64)
    return node_id_map, mal_src, mal_dst


def _pivot_fixture():
    """A single genuine mid-path node M: dst of one malicious edge, src of another.

    E -> M (M is attacked), M -> V (M then attacks V) — M is the only pivot.
    """
    node_id_map = {
        "8.8.8.1": 0,   # E (external)
        "10.0.0.1": 1,  # M (internal, pivot)
        "10.0.0.2": 2,  # V (internal, victim only)
    }
    mal_src = np.array([0, 1], dtype=np.int64)
    mal_dst = np.array([1, 2], dtype=np.int64)
    return node_id_map, mal_src, mal_dst


def _pivot_with_shortcut_fixture():
    """E(external) -> M(internal), E -> W(internal), M -> W: M is a pivot
    (dst of E->M, src of M->W) but the direct E->W edge means W is still
    only 1 hop from an external root, so max(d_gw)==1 despite pivot_count!=0.
    Counterexample to the converse of "pivot_count==0 => max(d_gw)<=1".
    """
    node_id_map = {
        "8.8.8.1": 0,   # E (external)
        "10.0.0.1": 1,  # M (internal, pivot)
        "10.0.0.2": 2,  # W (internal, reached directly by E too)
    }
    mal_src = np.array([0, 0, 1], dtype=np.int64)
    mal_dst = np.array([1, 2, 2], dtype=np.int64)
    return node_id_map, mal_src, mal_dst


def _incomplete_bipartite_chain_fixture():
    """E1->V1, E2->{V1,V2}, E3->{V2,V3}, E4->{V3,V4} — miniature of the real
    UNSW Worms per-class subgraph (28/40 attacker-victim edges): every edge
    is external->internal (pivot_count==0), every victim is 1 hop from some
    external attacker (max d_gw==1), but the undirected shape is a long path
    (diameter==7). Demonstrates diameter is independent of pivot/d_gw.
    """
    node_id_map = {
        "8.8.8.1": 0, "8.8.8.2": 1, "8.8.8.3": 2, "8.8.8.4": 3,  # E1..E4 (external)
        "10.0.0.1": 4, "10.0.0.2": 5, "10.0.0.3": 6, "10.0.0.4": 7,  # V1..V4 (internal)
    }
    mal_src = np.array([0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    mal_dst = np.array([4, 4, 5, 5, 6, 6, 7], dtype=np.int64)
    return node_id_map, mal_src, mal_dst


def _internal_to_external_fixture():
    """attacker(external) -> victim(internal) -> c2(external): exercises
    a malicious internal->external edge, invisible under UNSW's
    victim_fallback mode but present on multi-subnet Paper-3 datasets.
    """
    node_id_map = {
        "8.8.8.1": 0,  # external attacker
        "10.0.0.1": 1,  # internal victim (also a malicious source -> external C2)
        "9.9.9.9": 2,  # external C2/exfil destination
    }
    mal_src = np.array([0, 1], dtype=np.int64)
    mal_dst = np.array([1, 2], dtype=np.int64)
    return node_id_map, mal_src, mal_dst


def _unreachable_external_dst_fixture():
    """8.8.8.1(ext) -> 10.0.0.1(int) [normal, reachable], plus an isolated
    10.0.0.2(int) -> 9.9.9.9(ext) edge whose internal source is never reached
    from any external BFS root. Node 3 (external) is absent from d_gw
    entirely (unreachable), unlike _internal_to_external_fixture() where the
    external destination IS reached (d_gw=2). This isolates n_unreachable as
    the discriminator: pre-fix (unfiltered dst_c) counts the unreachable
    external node 3 against n_unreachable; post-fix (is_internal-filtered)
    does not, since node 3 is never an internal destination to begin with.
    """
    node_id_map = {
        "8.8.8.1": 0,   # external attacker (reachable root)
        "10.0.0.1": 1,  # internal victim (reachable, d_gw=1)
        "10.0.0.2": 2,  # internal node (never reached by any BFS root)
        "9.9.9.9": 3,   # external node (isolated exfil dst, unreachable)
    }
    mal_src = np.array([0, 2], dtype=np.int64)
    mal_dst = np.array([1, 3], dtype=np.int64)
    return node_id_map, mal_src, mal_dst


def _make_synthetic_graph(mal_src: np.ndarray, mal_dst: np.ndarray, n_nodes: int) -> "dgl.DGLGraph":
    """Build a synthetic DGL graph with monotonic timestamps, mirroring
    tests/test_temporal_sampler.py's ``_make_synthetic_graph`` convention.
    Every edge carries the single non-Benign label (this is already the
    malicious-only edge set by fixture construction).
    """
    n = len(mal_src)
    ts = np.arange(1000, 1000 + n * 10, 10, dtype=np.int64)
    labels = np.full(n, LABEL_MAP["AttackClass"], dtype=np.int64)

    g = dgl.graph((torch.tensor(mal_src), torch.tensor(mal_dst)), num_nodes=n_nodes)
    g.edata["timestamp"] = torch.tensor(ts, dtype=torch.int64)
    g.edata["label"] = torch.tensor(labels, dtype=torch.int64)
    return g


# ---------------------------------------------------------------------------
# 1 — Role partition is total
# ---------------------------------------------------------------------------

def test_assign_roles_partition_total():
    node_id_map, mal_src, mal_dst, attacker_ids, victim_ids = _bipartite_fixture()
    is_internal, mode = assign_roles(node_id_map, mal_src, mal_dst, internal_prefixes="auto")

    n_nodes = len(node_id_map)
    assert is_internal.shape[0] == n_nodes
    assert int(is_internal.sum()) + int((~is_internal).sum()) == n_nodes
    assert mode == "rfc1918_auto"
    for v in victim_ids:
        assert is_internal[v]
    for a in attacker_ids:
        assert not is_internal[a]


def test_assign_roles_victim_fallback():
    """All-public-IP dataset (UNSW-NB15-like): zero internal nodes among
    malicious destinations under RFC1918/loopback auto-detection triggers
    the victim-inference fallback.
    """
    node_id_map = {"1.1.1.1": 0, "2.2.2.2": 1, "3.3.3.3": 2}
    mal_src = np.array([0, 0], dtype=np.int64)
    mal_dst = np.array([1, 2], dtype=np.int64)

    is_internal, mode = assign_roles(node_id_map, mal_src, mal_dst, internal_prefixes="auto")
    assert mode == "victim_fallback"
    assert is_internal[1] and is_internal[2]
    assert not is_internal[0]


def test_assign_roles_explicit_prefixes():
    node_id_map, mal_src, mal_dst, attacker_ids, victim_ids = _bipartite_fixture()
    is_internal, mode = assign_roles(
        node_id_map, mal_src, mal_dst, internal_prefixes=["10.0.0.0/8"]
    )
    assert mode == "explicit"
    for v in victim_ids:
        assert is_internal[v]
    for a in attacker_ids:
        assert not is_internal[a]


# ---------------------------------------------------------------------------
# 2 — compute_d_gw correctness
# ---------------------------------------------------------------------------

def test_compute_d_gw_bipartite():
    node_id_map, mal_src, mal_dst, _, victim_ids = _bipartite_fixture()
    is_internal, _ = assign_roles(node_id_map, mal_src, mal_dst, "auto")
    d_gw = compute_d_gw(mal_src, mal_dst, is_internal)
    for v in victim_ids:
        assert d_gw[v] == 1.0


def test_compute_d_gw_chain():
    node_id_map, mal_src, mal_dst = _chain_fixture()
    is_internal, _ = assign_roles(node_id_map, mal_src, mal_dst, "auto")
    d_gw = compute_d_gw(mal_src, mal_dst, is_internal)
    assert d_gw[1] == 1.0  # A
    assert d_gw[2] == 2.0  # B
    assert d_gw[3] == 3.0  # C


def test_compute_d_gw_unreachable_absent():
    node_id_map, mal_src, mal_dst, _, _ = _bipartite_fixture()
    isolated_idx = len(node_id_map)
    node_id_map["10.0.0.99"] = isolated_idx  # never appears in any malicious edge

    is_internal, _ = assign_roles(node_id_map, mal_src, mal_dst, "auto")
    d_gw = compute_d_gw(mal_src, mal_dst, is_internal)
    assert isolated_idx not in d_gw


# ---------------------------------------------------------------------------
# 3 — derive_k_star correctness
# ---------------------------------------------------------------------------

def test_derive_k_star_bipartite():
    node_id_map, mal_src, mal_dst, _, _ = _bipartite_fixture()
    is_internal, _ = assign_roles(node_id_map, mal_src, mal_dst, "auto")
    d_gw = compute_d_gw(mal_src, mal_dst, is_internal)

    k_star, stats = derive_k_star(d_gw, mal_dst, is_internal, k_max=4)
    assert k_star == 2
    assert stats["max_d_gw"] == 1.0
    assert stats["n_unreachable"] == 0


def test_derive_k_star_chain():
    node_id_map, mal_src, mal_dst = _chain_fixture()
    is_internal, _ = assign_roles(node_id_map, mal_src, mal_dst, "auto")
    d_gw = compute_d_gw(mal_src, mal_dst, is_internal)

    k_star, stats = derive_k_star(d_gw, mal_dst, is_internal, k_max=4)
    assert k_star == 4
    assert stats["max_d_gw"] == 3.0


def test_derive_k_star_clip():
    node_id_map, mal_src, mal_dst = _chain_fixture()
    is_internal, _ = assign_roles(node_id_map, mal_src, mal_dst, "auto")
    d_gw = compute_d_gw(mal_src, mal_dst, is_internal)

    k_star, _ = derive_k_star(d_gw, mal_dst, is_internal, k_max=2)
    assert k_star == 2  # raw value would be 4; clip must actually fire


def test_derive_k_star_degenerate_no_finite_values():
    """No internal malicious destination is reverse-reachable from an
    external frontier: derive_k_star must clamp to k_max and warn, not raise.
    """
    is_internal = np.array([False, True])  # node 1 is internal but unreached
    d_gw: dict[int, float] = {}
    mal_dst = np.array([1], dtype=np.int64)

    k_star, stats = derive_k_star(d_gw, mal_dst, is_internal, k_max=4)
    assert k_star == 4
    assert stats["n_measured"] == 0
    assert stats["n_unreachable"] == 1


# ---------------------------------------------------------------------------
# 4 — compute_gateway_set
# ---------------------------------------------------------------------------

def test_compute_gateway_set_bipartite():
    node_id_map, mal_src, mal_dst, _, victim_ids = _bipartite_fixture()
    is_internal, _ = assign_roles(node_id_map, mal_src, mal_dst, "auto")
    d_gw = compute_d_gw(mal_src, mal_dst, is_internal)

    gateway, is_chokepoint = compute_gateway_set(is_internal, d_gw)
    assert gateway == set(victim_ids)
    assert not is_chokepoint  # |G| == 10 != 1


def test_compute_gateway_set_single_victim_is_chokepoint():
    node_id_map, mal_src, mal_dst, _, _ = _bipartite_fixture(n_attackers=4, n_victims=1)
    is_internal, _ = assign_roles(node_id_map, mal_src, mal_dst, "auto")
    d_gw = compute_d_gw(mal_src, mal_dst, is_internal)

    gateway, is_chokepoint = compute_gateway_set(is_internal, d_gw)
    assert len(gateway) == 1
    assert is_chokepoint


# ---------------------------------------------------------------------------
# 5 — compute_diameter
# ---------------------------------------------------------------------------

def test_compute_diameter_bipartite():
    _, mal_src, mal_dst, _, _ = _bipartite_fixture()
    result = compute_diameter(mal_src, mal_dst)
    assert result["diameter"] == 2
    assert result["n_components"] == 1


def test_compute_diameter_chain():
    _, mal_src, mal_dst = _chain_fixture()
    result = compute_diameter(mal_src, mal_dst)
    assert result["diameter"] == 3


# ---------------------------------------------------------------------------
# 6 — compute_pivot_nodes
#
# DEVIATION FROM SPEC LITERAL WORDING: the implementation spec's §8.2 item 6
# asserts "bipartite/chain fixtures ... -> pivot_count == 0". This is correct
# for the bipartite fixture (attacker and victim sets are disjoint by
# construction) but internally inconsistent for the chain fixture as defined
# in the spec's own §8.1 item 2 (E -> A -> B -> C): by construction A and B
# are each the destination of one malicious edge AND the source of the next,
# so compute_pivot_nodes's own spec'd definition (§3.4: dst_set ∩ src_set)
# correctly reports {A, B} as pivots for the chain fixture. This rests solely
# on that plain dst_set ∩ src_set definition — see specs/07 §3.4's corrected
# cross-invariant note (post-review: pivot_count == 0 only ever *implies*
# max(d_gw) <= 1, never the converse, and diameter is independent of both)
# for why no biconditional claim is invoked here. The "genuine mid-path node
# -> pivot_count == 1" fixture is implemented as _pivot_fixture(), separate
# from the chain fixture, per the spec's own description of that third case.
# ---------------------------------------------------------------------------

def test_compute_pivot_nodes_bipartite_zero():
    _, mal_src, mal_dst, _, _ = _bipartite_fixture()
    pivots, _ = compute_pivot_nodes(mal_src, mal_dst)
    assert len(pivots) == 0


def test_compute_pivot_nodes_chain_is_nonzero():
    _, mal_src, mal_dst = _chain_fixture()
    pivots, _ = compute_pivot_nodes(mal_src, mal_dst)
    assert pivots == {1, 2}  # A and B relay malicious traffic — see deviation note above


def test_compute_pivot_nodes_single_mid_path_node():
    _, mal_src, mal_dst = _pivot_fixture()
    pivots, _ = compute_pivot_nodes(mal_src, mal_dst)
    assert pivots == {1}  # M only


def test_compute_pivot_nodes_per_class_breakdown():
    _, mal_src, mal_dst = _pivot_fixture()
    mal_labels = np.full(len(mal_src), LABEL_MAP["AttackClass"], dtype=np.int64)
    pivots, per_class_count = compute_pivot_nodes(mal_src, mal_dst, mal_labels, INT_TO_NAME)
    assert pivots == {1}
    assert per_class_count["AttackClass"] == 1


def test_pivot_zero_and_low_d_gw_does_not_imply_low_diameter():
    """Regression test for the corrected (non-)relationship between
    pivot_count, max(d_gw), and diameter — see specs/07_phase14_gateway_
    distance_impl.md §3.4's corrected note: pivot_count == 0 only ever
    *implies* max(d_gw) <= 1 (never the converse), and undirected diameter
    is independent of both. This fixture has pivot_count == 0 and
    max(d_gw) == 1 simultaneously with diameter == 7.
    """
    node_id_map, mal_src, mal_dst = _incomplete_bipartite_chain_fixture()
    is_internal, _ = assign_roles(node_id_map, mal_src, mal_dst, "auto")
    assert is_internal[4] and not is_internal[0]  # expected internal/external split

    pivots, _ = compute_pivot_nodes(mal_src, mal_dst)
    assert len(pivots) == 0

    d_gw = compute_d_gw(mal_src, mal_dst, is_internal)
    internal_ids = [4, 5, 6, 7]
    assert max(d_gw[v] for v in internal_ids) == 1.0

    result = compute_diameter(mal_src, mal_dst)
    assert result["diameter"] == 7


def test_low_d_gw_does_not_imply_pivot_zero():
    """Regression test for the converse-false half of the corrected §3.4
    note: max(d_gw) <= 1 does NOT imply pivot_count == 0. E->M, E->W, M->W
    gives pivot_count == 1 (M is both dst of E->M and src of M->W) while
    max(d_gw) == 1 (W is reached directly from external root E, one hop,
    regardless of the longer M->W path). The forward implication
    (pivot_count==0 => max(d_gw)<=1) is unaffected by this — this fixture
    has a nonzero pivot count, so it is not a counterexample to that
    direction, only to its converse.
    """
    node_id_map, mal_src, mal_dst = _pivot_with_shortcut_fixture()
    is_internal, _ = assign_roles(node_id_map, mal_src, mal_dst, "auto")
    assert is_internal[1] and is_internal[2] and not is_internal[0]

    pivots, _ = compute_pivot_nodes(mal_src, mal_dst)
    assert pivots == {1}  # M is a pivot: dst of E->M, src of M->W

    d_gw = compute_d_gw(mal_src, mal_dst, is_internal)
    internal_ids = [1, 2]
    assert max(d_gw[v] for v in internal_ids) == 1.0


# ---------------------------------------------------------------------------
# 7 — Determinism
# ---------------------------------------------------------------------------

def test_determinism():
    node_id_map, mal_src, mal_dst, _, _ = _bipartite_fixture()
    is_internal, _ = assign_roles(node_id_map, mal_src, mal_dst, "auto")

    d1 = compute_d_gw(mal_src, mal_dst, is_internal)
    d2 = compute_d_gw(mal_src, mal_dst, is_internal)
    assert d1 == d2

    k1, s1 = derive_k_star(d1, mal_dst, is_internal, k_max=4)
    k2, s2 = derive_k_star(d2, mal_dst, is_internal, k_max=4)
    assert k1 == k2
    assert s1 == s2


# ---------------------------------------------------------------------------
# 8 — Train-only / no-leakage framing (drift_check + build_malicious_subgraph)
# ---------------------------------------------------------------------------

def test_build_malicious_subgraph_matches_fixture():
    node_id_map, mal_src, mal_dst, _, _ = _bipartite_fixture()
    n_nodes = len(node_id_map)
    g_train = _make_synthetic_graph(mal_src, mal_dst, n_nodes)

    src_out, dst_out, labels_out = build_malicious_subgraph(g_train, LABEL_MAP)
    assert np.array_equal(src_out, mal_src)
    assert np.array_equal(dst_out, mal_dst)
    assert np.all(labels_out == LABEL_MAP["AttackClass"])


def test_drift_check_uses_only_the_passed_train_graph():
    """drift_check's signature accepts a single graph — structurally, there
    is no val/test input path for it to leak from (CLAUDE.md Invariants 2/5
    analogue at the module level).
    """
    import inspect

    sig = inspect.signature(drift_check)
    graph_params = [
        name for name, p in sig.parameters.items()
        if "graph" in name or name == "g_train"
    ]
    assert graph_params == ["g_train"]


def test_drift_check_splits_edges_exactly_in_half_no_overlap():
    node_id_map, mal_src, mal_dst, _, _ = _bipartite_fixture()
    n_nodes = len(node_id_map)
    g_train = _make_synthetic_graph(mal_src, mal_dst, n_nodes)

    n_edges = g_train.num_edges()
    mid = n_edges // 2
    first_idx = set(range(0, mid))
    second_idx = set(range(mid, n_edges))

    assert first_idx.isdisjoint(second_idx)
    assert len(first_idx) + len(second_idx) == n_edges


def test_drift_check_bipartite_no_drift():
    node_id_map, mal_src, mal_dst, _, _ = _bipartite_fixture()
    n_nodes = len(node_id_map)
    g_train = _make_synthetic_graph(mal_src, mal_dst, n_nodes)
    is_internal, _ = assign_roles(node_id_map, mal_src, mal_dst, "auto")

    result = drift_check(g_train, LABEL_MAP, is_internal, k_max=4)
    assert result["k_star_first_half"] == result["k_star_second_half"] == 2
    assert result["delta_k_star"] == 0


# ---------------------------------------------------------------------------
# 9 — Graceful degradation (spearman_dgw_vs_f1 with missing metrics.json)
# ---------------------------------------------------------------------------

def test_per_class_stats_independent_of_correlation_step():
    """Items 1-3 (per_class_stats included) must succeed even though
    metrics.json is never referenced here — there is no code path where a
    missing metrics.json could short-circuit them.
    """
    node_id_map, mal_src, mal_dst, _, victim_ids = _bipartite_fixture()
    is_internal, _ = assign_roles(node_id_map, mal_src, mal_dst, "auto")
    d_gw = compute_d_gw(mal_src, mal_dst, is_internal)
    mal_labels = np.full(len(mal_src), LABEL_MAP["AttackClass"], dtype=np.int64)

    stats = per_class_stats(mal_src, mal_dst, mal_labels, d_gw, INT_TO_NAME, is_internal)
    assert stats["AttackClass"]["mean"] == 1.0
    assert stats["AttackClass"]["n_dst_nodes"] == len(victim_ids)


def test_per_class_stats_excludes_external_destinations_from_dgw_aggregation():
    """A malicious internal->external edge (e.g. a victim exfiltrating to a
    C2 host) must not pull an external destination into the d_gw mean/max/
    std/n_unreachable aggregation — those restrict to internal destinations
    only, mirroring the global-stats block in run_gateway_distance_diagnostic.
    n_dst_nodes/n_src_nodes remain full, unfiltered counts.
    """
    node_id_map, mal_src, mal_dst = _internal_to_external_fixture()
    is_internal, mode = assign_roles(node_id_map, mal_src, mal_dst, "auto")
    assert mode == "rfc1918_auto"  # node 1 is internal among malicious dsts -> no fallback
    d_gw = compute_d_gw(mal_src, mal_dst, is_internal)
    mal_labels = np.full(len(mal_src), LABEL_MAP["AttackClass"], dtype=np.int64)

    stats = per_class_stats(mal_src, mal_dst, mal_labels, d_gw, INT_TO_NAME, is_internal)
    cls = stats["AttackClass"]
    assert cls["mean"] == 1.0        # only node 1 (internal) counted, not node 2 (external, d_gw=2)
    assert cls["max"] == 1.0
    assert cls["std"] == 0.0
    assert cls["n_unreachable"] == 0
    assert cls["n_dst_nodes"] == 2   # full footprint still includes the external C2 node


def test_per_class_stats_excludes_unreachable_external_destination_from_n_unreachable():
    """Complements test_per_class_stats_excludes_external_destinations_from_
    dgw_aggregation: that test's external destination happens to be BFS-
    reachable (d_gw=2), so n_unreachable==0 regardless of whether the
    is_internal filter is applied (pre-fix would also report 0). This
    fixture's external destination is never reached by any BFS root, so it
    isolates n_unreachable as the sole discriminator between pre-fix
    (unfiltered dst_c, which would count the unreachable external node,
    n_unreachable=1) and post-fix (is_internal-filtered, n_unreachable=0)
    behavior.
    """
    node_id_map, mal_src, mal_dst = _unreachable_external_dst_fixture()
    is_internal, mode = assign_roles(node_id_map, mal_src, mal_dst, "auto")
    assert mode == "rfc1918_auto"
    d_gw = compute_d_gw(mal_src, mal_dst, is_internal)
    assert d_gw == {0: 0.0, 1: 1.0}  # nodes 2, 3 never reached
    mal_labels = np.full(len(mal_src), LABEL_MAP["AttackClass"], dtype=np.int64)

    stats = per_class_stats(mal_src, mal_dst, mal_labels, d_gw, INT_TO_NAME, is_internal)
    cls = stats["AttackClass"]
    assert cls["mean"] == 1.0
    assert cls["n_unreachable"] == 0  # pre-fix would have reported 1 (node 3 counted)
    assert cls["n_dst_nodes"] == 2    # full footprint: nodes 1 and 3


def test_spearman_dgw_vs_f1_graceful_degrade_on_missing_metrics():
    per_class = {
        "AttackClass": {
            "mean": 1.0, "max": 1.0, "std": 0.0,
            "n_dst_nodes": 10, "n_src_nodes": 4, "n_unreachable": 0,
        }
    }
    result = spearman_dgw_vs_f1(per_class, metrics_json_path=None)
    assert result["available"] is False
    assert result["spearman_rho"] is None
    assert result["p_value"] is None
    assert result["note"] is not None


def test_spearman_dgw_vs_f1_graceful_degrade_on_nonexistent_path():
    per_class = {
        "AttackClass": {
            "mean": 1.0, "max": 1.0, "std": 0.0,
            "n_dst_nodes": 10, "n_src_nodes": 4, "n_unreachable": 0,
        }
    }
    result = spearman_dgw_vs_f1(per_class, metrics_json_path=Path("/nonexistent/metrics.json"))
    assert result["available"] is False
    assert result["note"] is not None


def test_spearman_dgw_vs_f1_zero_variance_returns_none_not_nan(tmp_path):
    """UNSW's documented headline case (spec §7.1): every class has the same
    mean d_gw. spearmanr returns NaN for zero-variance input — this must be
    caught and reported as an explicit None + note, not a bare NaN leaking
    into the JSON output. Uses a real metrics.json on disk (tmp_path) so the
    JSON-load + name-intersection branch actually executes, not just the
    missing-file guard.
    """
    import json

    per_class = {
        "Generic": {"mean": 1.0, "max": 1.0, "std": 0.0, "n_dst_nodes": 10, "n_src_nodes": 4, "n_unreachable": 0},
        "Backdoor": {"mean": 1.0, "max": 1.0, "std": 0.0, "n_dst_nodes": 8, "n_src_nodes": 3, "n_unreachable": 0},
        "Fuzzers": {"mean": 1.0, "max": 1.0, "std": 0.0, "n_dst_nodes": 6, "n_src_nodes": 2, "n_unreachable": 0},
    }
    metrics = {
        "per_class": {
            "Generic": {"precision": 0.9, "recall": 0.9, "f1": 0.91, "support": 100},
            "Backdoor": {"precision": 0.1, "recall": 0.1, "f1": 0.03, "support": 50},
            "Fuzzers": {"precision": 0.5, "recall": 0.5, "f1": 0.55, "support": 30},
        }
    }
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics))

    result = spearman_dgw_vs_f1(per_class, metrics_json_path=metrics_path)
    assert result["available"] is True
    assert result["spearman_rho"] is None
    assert result["p_value"] is None
    assert "zero variance" in result["note"]


def test_spearman_dgw_vs_f1_computes_real_correlation(tmp_path):
    """Varying per-class mean d_gw paired with varying per-class F1 exercises
    the actual scipy.stats.spearmanr call and the class-name intersection
    join against metrics.json's per_class keys.
    """
    import json

    per_class = {
        "Generic":     {"mean": 1.0, "max": 1.0, "std": 0.0, "n_dst_nodes": 10, "n_src_nodes": 4, "n_unreachable": 0},
        "Backdoor":    {"mean": 2.0, "max": 2.0, "std": 0.0, "n_dst_nodes": 8, "n_src_nodes": 3, "n_unreachable": 0},
        "Fuzzers":     {"mean": 3.0, "max": 3.0, "std": 0.0, "n_dst_nodes": 6, "n_src_nodes": 2, "n_unreachable": 0},
        "Exploits":    {"mean": 4.0, "max": 4.0, "std": 0.0, "n_dst_nodes": 4, "n_src_nodes": 1, "n_unreachable": 0},
    }
    # Perfectly monotonic decreasing F1 as mean d_gw increases -> rho == -1.0.
    metrics = {
        "per_class": {
            "Generic":  {"precision": 0.9, "recall": 0.9, "f1": 0.90, "support": 100},
            "Backdoor": {"precision": 0.7, "recall": 0.7, "f1": 0.70, "support": 50},
            "Fuzzers":  {"precision": 0.5, "recall": 0.5, "f1": 0.50, "support": 30},
            "Exploits": {"precision": 0.3, "recall": 0.3, "f1": 0.30, "support": 20},
            # Extra class present in metrics.json but absent from per_class —
            # must be excluded by the intersection, not raise a KeyError.
            "Worms":    {"precision": 0.1, "recall": 0.1, "f1": 0.10, "support": 5},
        }
    }
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics))

    result = spearman_dgw_vs_f1(per_class, metrics_json_path=metrics_path)
    assert result["available"] is True
    assert result["spearman_rho"] == -1.0
    assert result["p_value"] is not None
    assert result["note"] is None
