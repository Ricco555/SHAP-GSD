"""Tests for the shared baseline surrogate embedding matrix (adapter.build_h_full).

Background — the defect being guarded against
---------------------------------------------
All four coalition baselines (PGExplainer, GNNShap, GraphSVX, EdgeSHAPer) build
an ``(N_local, hidden)`` node-embedding matrix ``h_full`` and then run their own
one-hop coalition-weighted aggregation on top of it.  Each wrapper used to carry
its own identical copy of a ``_build_h_full`` that wrote ``ctx.h_fixed`` into the
two seed rows and left EVERY other row at zero.  Because the surrogate's message
for an edge is ``h_full[src]``, a zero source row makes an edge informationally
inert: including or excluding it cannot change the surrogate's output at all, so
the estimators were structurally unable to attribute importance to neighbourhood
nodes.

These tests run against the real artifacts on disk (graphs/, feature_store/,
artifacts/, node_state_snapshots/) and are skipped when those are absent, so a
fresh clone still passes the gate.  They deliberately do NOT require the vendored
third-party explainer libraries — ``build_h_full`` and its inputs/outputs are
entirely on the SHAP-GSD side.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests._paths import REPO_ROOT, resolve_cfg

pytest.importorskip("dgl")
pytest.importorskip("torch_geometric")

import dgl  # noqa: E402

N_TEST_FLOWS = 12

#: Fraction of eligible flows that must become sensitive to a neighbourhood
#: node's in-edge.  Measured in LOGIT space, not probability space: the
#: wrappers' UNNORMALISED residual aggregation (h = h_full + sum of neighbour
#: rows, which this fix deliberately leaves alone) drives the readout far
#: outside the edge MLP's operating range on high-confidence flows, so in
#: probability space the float32 softmax is saturated and the measurement rests
#: on the representation floor rather than on the surrogate.
#:
#: Evidence, 150-flow probe (spec 27 §1.4): probability space 83/150 = 55.3%
#: sensitive, with 67 flows at EXACTLY 0.0 and mean p_full = 0.9999964 over
#: that group; the same experiment in logit space gives 150/150 sensitive,
#: median logit delta 2.61.  The legacy zero-fill control is exactly 0.0 in
#: BOTH spaces on 150/150.
#:
#: Re-measured on the N_TEST_FLOWS = 12 fixture (stage 4): logit space
#: **12/12 = 100%** sensitive, per-flow |Δlogit| min 1.042, median 2.71, max
#: 3.59; the SAME 12 flows measure only 7/12 = 58.3% in probability space
#: (every fixture flow has p_full >= 0.993, six of them >= 0.99999).  The
#: legacy zero-fill control is exactly 0.0 on 12/12.  Hence 1.0.
SENSITIVE_FRACTION = 1.0

#: Minimum |Δ| in LOGIT space for a coalition change to count as "the edge
#: moved the surrogate" rather than float32 noise.  Its job is to separate
#: signal from noise, not to encode an effect size: it sits ~1000x below the
#: smallest per-flow delta actually observed (1.042 on the fixture, 0.157 over
#: the 150-flow probe) while still being far above float32 logit resolution and
#: above the legacy control's exact 0.0.  Setting it at the observed p05 (~1.0)
#: would make a CORRECT implementation fail whenever the fixture draw happened
#: to include a low-delta flow.
LOGIT_SENSITIVITY_EPS = 1e-3

#: Minimum |Δ| between the per-block and the superseded flattened (hop-mixed)
#: h_full on non-seed rows.  Per-flow max|Δ| measured 1.00-3.63 over 40 real
#: flows (spec 27 §6.3c) and 1.284-2.215 over this 12-flow fixture (stage 4,
#: all 12 flows have non-seed rows), so
#: this bar is ~1000x below the observed minimum — robust to the fixture draw,
#: but fails immediately if the flattened single-graph pass ever returns.
PER_BLOCK_VS_FLAT_EPS = 1e-3


# ── artifact discovery ────────────────────────────────────────────────────────

def _candidate_roots() -> list[dict[str, Path]]:
    """Layouts to try, most specific first: config-driven, then repo-root legacy."""
    layouts: list[dict[str, Path]] = []
    try:
        cfg = resolve_cfg()
        layouts.append({
            "artifacts": REPO_ROOT / cfg["output"]["artifacts_dir"],
            "graphs": REPO_ROOT / cfg["graph"]["dir"],
            "nsm": REPO_ROOT / cfg["graph"]["node_state_dir"],
            "fs_test": REPO_ROOT / cfg["output"]["feature_store_dir"] / "test",
            "cfg": cfg,
        })
    except Exception:                                    # pragma: no cover
        pass
    cfg_default = None
    try:
        from src.utils.config import load_config
        cfg_default = load_config(REPO_ROOT / "configs/default.yaml")
    except Exception:                                    # pragma: no cover
        pass
    layouts.append({
        "artifacts": REPO_ROOT / "artifacts",
        "graphs": REPO_ROOT / "graphs",
        "nsm": REPO_ROOT / "node_state_snapshots",
        "fs_test": REPO_ROOT / "feature_store" / "test",
        "cfg": cfg_default,
    })
    return layouts


def _resolve_layout() -> dict | None:
    """First layout on disk with everything build_flow_context needs."""
    for lay in _candidate_roots():
        if lay["cfg"] is None:
            continue
        needed = [
            lay["artifacts"] / "best_model.pt",
            lay["artifacts"] / "best_params.json",
            lay["artifacts"] / "feature_groups.json",
            lay["artifacts"] / "label_map.json",
            lay["graphs"] / "test.bin",
            lay["fs_test"] / "features.dat",
            lay["nsm"] / "histories.pkl",
        ]
        if all(p.exists() for p in needed):
            return lay
    return None


LAYOUT = _resolve_layout()
requires_artifacts = pytest.mark.skipif(
    LAYOUT is None,
    reason="no complete local run artifacts (graphs/, feature_store/, artifacts/)",
)


@pytest.fixture(scope="module")
def infra() -> dict:
    """Model, test graph, feature store, node states and sampler — loaded once."""
    from src.data.feature_store import FeatureStore
    from src.model.node_state import NodeStateManager
    from src.model.sage_model import EdgeAwareGraphSAGE
    from src.model.temporal_sampler import TemporalNeighborSampler

    art = LAYOUT["artifacts"]
    bp = json.loads((art / "best_params.json").read_text())
    fg = json.loads((art / "feature_groups.json").read_text())
    lm = json.loads((art / "label_map.json").read_text())
    m = LAYOUT["cfg"]["model"]

    model = EdgeAwareGraphSAGE(
        node_in_dim=m["node_state_dim"],
        edge_in_dim=fg["d_e"],
        hidden_size=bp["hidden_size"],
        num_classes=len(lm),
        num_layers=bp["num_layers"],
        dropout=bp["dropout"],
        aggregator=bp["aggregator"],
    )
    model.load_state_dict(torch.load(art / "best_model.pt", map_location="cpu"))
    model.eval()

    return {
        "model": model,
        "g_test": dgl.load_graphs(str(LAYOUT["graphs"] / "test.bin"))[0][0],
        "fs": FeatureStore(LAYOUT["fs_test"]),
        "nsm": NodeStateManager.load(LAYOUT["nsm"]),
        "sampler": TemporalNeighborSampler(fanouts=bp["fanouts"]),
        "device": torch.device("cpu"),
    }


@pytest.fixture(scope="module")
def flows(infra) -> list[dict]:
    """Per-flow context + local subgraph + both old and new h_full matrices."""
    from src.baselines.adapter import (
        build_flow_context, build_h_full, dgl_subgraph_to_pyg,
    )

    g_test = infra["g_test"]
    geids = g_test.edata[dgl.EID].numpy()
    rng = np.random.default_rng(42)
    # Sample from the tail of the split so the subgraphs are not degenerate
    # (early flows have almost no temporally-eligible history).
    tail = geids[len(geids) // 2:]
    chosen = sorted(rng.choice(tail, size=N_TEST_FLOWS, replace=False).tolist())

    out = []
    for geid in chosen:
        ctx = build_flow_context(
            int(geid), infra["model"], g_test, infra["nsm"], infra["fs"],
            infra["sampler"], infra["device"],
        )
        pyg, g2l = dgl_subgraph_to_pyg(ctx.blocks, None, ctx.base_node_feats)
        if pyg.edge_index.size(1) == 0:
            continue
        out.append({
            "ctx": ctx,
            "edge_index": pyg.edge_index,
            "gnid_to_local": g2l,
            "h_new": build_h_full(ctx, infra["model"], g2l, pyg.edge_index),
            "h_old": _legacy_zero_fill_h_full(ctx.h_fixed, ctx.blocks, g2l),
            "seed_local": {
                g2l[g] for g in ctx.blocks[-1].dstdata[dgl.NID].cpu().tolist()
                if g in g2l
            },
        })
    assert out, "no non-isolated test flow could be built"
    return out


def _legacy_zero_fill_h_full(h_fixed, blocks, gnid_to_local):
    """The superseded implementation, kept verbatim as the before/after control."""
    h_full = torch.zeros(len(gnid_to_local), h_fixed.size(1), dtype=torch.float32)
    for i, gnid in enumerate(blocks[-1].dstdata[dgl.NID].cpu().tolist()):
        local_idx = gnid_to_local.get(gnid)
        if local_idx is not None:
            h_full[local_idx] = h_fixed[i].detach().cpu()
    return h_full


def _surrogate_p(h_full, edge_index, ctx, src_local, dst_local, edge_mlp) -> float:
    """The one-hop residual surrogate shared by all four wrappers → p(true)."""
    h = h_full.clone()
    if edge_index.size(1) > 0:
        agg = torch.zeros_like(h)
        agg.index_add_(0, edge_index[1], h[edge_index[0]])
        h = h + agg
    combined = torch.cat(
        [h[src_local].unsqueeze(0), h[dst_local].unsqueeze(0), ctx.x_e_t.detach().cpu()],
        dim=1,
    )
    with torch.no_grad():
        return float(torch.softmax(edge_mlp(combined), dim=1)[0, ctx.true_label])


def _surrogate_logit(
    h_full: torch.Tensor,
    edge_index: torch.Tensor,
    ctx: object,
    src_local: int,
    dst_local: int,
    edge_mlp: torch.nn.Module,
) -> float:
    """The same one-hop residual surrogate as ``_surrogate_p``, read PRE-softmax.

    Identical body to ``_surrogate_p`` except for the final line: it returns the
    raw edge-MLP logit for ``ctx.true_label`` instead of the softmax
    probability.  Kept as a separate function rather than a flag on
    ``_surrogate_p`` because two must-pass-unmodified tests
    (``test_empty_coalition_reproduces_model_prediction`` and the ``d_old``
    control below) depend on ``_surrogate_p`` staying exactly as it is.

    Args:
        h_full:     (N_local, hidden) surrogate node-embedding matrix.
        edge_index: (2, E) coalition edges in local index space.
        ctx:        FlowContext for the target flow.
        src_local:  local index of the target edge's source endpoint.
        dst_local:  local index of the target edge's destination endpoint.
        edge_mlp:   the model's CPU edge MLP.

    Returns:
        The unnormalised logit for the flow's true class.
    """
    h = h_full.clone()
    if edge_index.size(1) > 0:
        agg = torch.zeros_like(h)
        agg.index_add_(0, edge_index[1], h[edge_index[0]])
        h = h + agg
    combined = torch.cat(
        [h[src_local].unsqueeze(0), h[dst_local].unsqueeze(0), ctx.x_e_t.detach().cpu()],
        dim=1,
    )
    with torch.no_grad():
        return float(edge_mlp(combined)[0, ctx.true_label])


def _flattened_h_full(
    ctx: object,
    model: torch.nn.Module,
    gnid_to_local: dict[int, int],
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """The superseded hop-mixed implementation, kept verbatim as the Bug-2 control.

    Reproduces ``adapter.build_h_full`` as it stood BEFORE the layer-depth fix
    (``src/baselines/adapter.py:369-402`` at that revision): one homogeneous
    ``dgl.graph`` built from the concatenated ``edge_index`` — every hop's edges
    flattened together — and then every ``conv``/``bn`` pair run over that same
    graph.  A 1-hop-shell node therefore received a second round of neighbour
    aggregation the real model never computes for it.

    Touches only ``dgl.graph`` and ``model.convs`` / ``model.bns`` — no
    ``fc_self`` / ``fc_neigh`` internals — so it does not hard-code DGL's
    ``SAGEConv`` implementation details.  Returns the PRE-overwrite matrix (the
    seed overwrite is identical under both implementations and so cannot
    discriminate between them).

    Args:
        ctx:           FlowContext for the target flow.
        model:         the trained EdgeAwareGraphSAGE, in eval mode.
        gnid_to_local: global node ID → local index, from dgl_subgraph_to_pyg.
        edge_index:    (2, E) local-space subgraph edges, flattened over hops.

    Returns:
        (N_local, hidden) float32 CPU tensor, seed rows NOT overwritten.
    """
    n_local = len(gnid_to_local)
    input_gnids = ctx.blocks[0].srcdata[dgl.NID].cpu().tolist()
    input_pos = {g: i for i, g in enumerate(input_gnids)}
    rows = [
        input_pos[gnid]
        for gnid, _ in sorted(gnid_to_local.items(), key=lambda kv: kv[1])
    ]
    x_local = torch.tensor(ctx.base_node_feats[rows], dtype=torch.float32)

    if edge_index.numel() == 0:
        src_t = torch.empty(0, dtype=torch.int64)
        dst_t = torch.empty(0, dtype=torch.int64)
    else:
        src_t = edge_index[0].cpu().long()
        dst_t = edge_index[1].cpu().long()
    g_local = dgl.graph((src_t, dst_t), num_nodes=n_local)

    with torch.no_grad():
        h = x_local
        for conv, bn in zip(model.convs, model.bns):
            h = conv(g_local, h)
            h = bn(h)
            h = torch.relu(h)
        return h.detach().cpu().float()


# ── (a) every node gets a real, node-specific embedding ───────────────────────

@requires_artifacts
def test_non_seed_rows_are_non_zero_and_node_specific(flows):
    """Non-seed rows must be non-zero AND distinct — not zero-fill, not a constant."""
    n_checked = 0
    for f in flows:
        h = f["h_new"]
        non_seed = [i for i in range(h.size(0)) if i not in f["seed_local"]]
        if not non_seed:
            continue
        n_checked += 1
        rows = h[non_seed]
        assert torch.isfinite(rows).all()
        assert (rows.abs().sum(dim=1) > 0).all(), (
            "a non-seed row is all-zero — the zero-fill defect has regressed"
        )
        if len(non_seed) > 1:
            # Not a single constant broadcast to every node.
            assert float(rows.std(dim=0).max()) > 0.0, (
                "every non-seed node received the SAME embedding — not node-specific"
            )
    assert n_checked > 0, "no flow with a non-seed node was available"


@requires_artifacts
def test_shape_dtype_and_device(flows):
    """Contract: (N_local, hidden) float32 on CPU — the wrappers all run on CPU."""
    for f in flows:
        h = f["h_new"]
        assert h.shape == (len(f["gnid_to_local"]), f["ctx"].h_fixed.size(1))
        assert h.dtype == torch.float32
        assert h.device.type == "cpu"


# ── (b) the seed rows still reproduce the real model prediction ───────────────

@requires_artifacts
def test_seed_rows_equal_h_fixed(flows):
    """The two seed rows must be bit-identical to ctx.h_fixed.

    The surrogate reads out at the target edge's endpoints, so any drift here
    would silently change what the model is claimed to have predicted for the
    flow being explained.
    """
    for f in flows:
        ctx = f["ctx"]
        g2l = f["gnid_to_local"]
        h_fixed = ctx.h_fixed.detach().cpu()
        for i, gnid in enumerate(ctx.blocks[-1].dstdata[dgl.NID].cpu().tolist()):
            if gnid in g2l:
                assert torch.equal(f["h_new"][g2l[gnid]], h_fixed[i])


@requires_artifacts
def test_empty_coalition_reproduces_model_prediction(infra, flows):
    """With no coalition edges the surrogate must return exactly ctx.p_full."""
    edge_mlp = infra["model"].edge_mlp
    for f in flows:
        ctx = f["ctx"]
        g2l = f["gnid_to_local"]
        p = _surrogate_p(
            f["h_new"], f["edge_index"].new_zeros((2, 0)), ctx,
            g2l[ctx.target_src_nid], g2l[ctx.target_dst_nid], edge_mlp,
        )
        assert p == pytest.approx(ctx.p_full, abs=1e-5)


# ── (c) the sensitivity proof ─────────────────────────────────────────────────

@requires_artifacts
def test_non_seed_edge_now_changes_the_surrogate(infra, flows):
    """A neighbourhood node's in-edge must now move the surrogate; before, it could not.

    The edge chosen has a NON-SEED source and a readout destination, so under the
    old zero-fill matrix its message was the zero vector and its marginal
    contribution to any coalition was identically zero.  The comparison is made
    in the small-coalition regime — where a Shapley estimator draws most of its
    samples — because at the full coalition the surrogate's unnormalised
    residual sum saturates the softmax for both matrices.

    The two sides are measured in DIFFERENT spaces, deliberately (spec 27 §6.2):

    * ``d_new`` is measured in LOGIT space.  Every fixture flow is a
      high-confidence one (p_full >= 0.993), and the wrappers' unnormalised
      residual aggregation pushes the readout well outside the edge MLP's
      operating range, so the float32 softmax saturates and probability-space
      deltas collapse onto the representation floor — 67/150 flows at
      *exactly* 0.0 in the spec-27 probe.  That is a property of the masking
      mechanism, not of ``h_full``.  Logit space is the strictly STRONGER
      measurement (150/150 vs 83/150 on that probe), not a relaxation.
    * ``d_old`` stays in PROBABILITY space, unchanged.  It is exactly zero *by
      construction* (a zero source row contributes a zero message), it is the
      historical-defect anchor, and it remains exactly 0.0 in logit space too —
      so leaving it alone costs nothing.
    """
    edge_mlp = infra["model"].edge_mlp
    n_checked = 0
    n_sensitive = 0
    for f in flows:
        ctx, ei, g2l = f["ctx"], f["edge_index"], f["gnid_to_local"]
        sl, dl = g2l[ctx.target_src_nid], g2l[ctx.target_dst_nid]
        cand = [
            j for j in range(ei.size(1))
            if int(ei[1, j]) in (sl, dl) and int(ei[0, j]) not in f["seed_local"]
        ]
        if not cand:
            continue
        n_checked += 1
        j = cand[0]
        others = np.array([k for k in range(ei.size(1)) if k != j])
        rng = np.random.default_rng(11)
        C = torch.tensor(rng.permutation(others)[:min(3, len(others))].copy(),
                         dtype=torch.long)
        C_with = torch.cat([C, torch.tensor([j], dtype=torch.long)])

        d_old = abs(_surrogate_p(f["h_old"], ei[:, C], ctx, sl, dl, edge_mlp)
                    - _surrogate_p(f["h_old"], ei[:, C_with], ctx, sl, dl, edge_mlp))
        d_new = abs(_surrogate_logit(f["h_new"], ei[:, C], ctx, sl, dl, edge_mlp)
                    - _surrogate_logit(f["h_new"], ei[:, C_with], ctx, sl, dl, edge_mlp))

        assert d_old == 0.0, (
            "control failed: the legacy zero-fill matrix was expected to be "
            "exactly insensitive to a non-seed-sourced edge"
        )
        if d_new > LOGIT_SENSITIVITY_EPS:
            n_sensitive += 1

    assert n_checked > 0, "no flow offered a non-seed-sourced edge into the readout"
    assert n_sensitive >= max(1, int(SENSITIVE_FRACTION * n_checked)), (
        f"only {n_sensitive}/{n_checked} flows became sensitive to a "
        f"neighbourhood node's in-edge (expected >= {SENSITIVE_FRACTION:.0%})"
    )


@requires_artifacts
def test_diagnostics_report_all_rows_live(flows):
    """Regression guard: surrogate_diagnostics must see no dead rows any more."""
    from src.baselines.adapter import surrogate_diagnostics

    for f in flows:
        ctx, g2l = f["ctx"], f["gnid_to_local"]
        d = surrogate_diagnostics(
            f["h_new"], f["edge_index"],
            g2l[ctx.target_src_nid], g2l[ctx.target_dst_nid],
        )
        assert d["n_h_full_nonzero"] == d["n_local"]


# ── (d) layer-depth correctness: per-block, not flattened ─────────────────────

@requires_artifacts
def test_per_block_pass_differs_from_flattened(infra, flows):
    """The per-block pass must measurably differ from the old hop-mixed one.

    This is the regression anchor for Bug 2 (spec 26 §1.4).  The control is the
    FLATTENED implementation (``_flattened_h_full``), NOT
    ``_legacy_zero_fill_h_full`` — the zero-fill matrix is the control for the
    *previous* fix, and both the flattened and the per-block implementations
    differ from it, so comparing against it would prove nothing about layer
    depth.

    Only the non-seed rows can discriminate: the seed rows are overwritten with
    ``ctx.h_fixed`` verbatim under both implementations.  Those non-seed rows
    are exactly the neighbour embeddings that feed every baseline's coalition
    readout, so a difference here is a difference in the reported attributions.

    What this does NOT assert: a numeric equality for a 1-hop-shell row against
    a hand-built ``relu(bn(fc_self(...)))`` reference.  That would hard-code
    DGL's SAGEConv internals into the test; the self-lift behaviour is pinned
    separately by ``test_sageconv_self_lift_on_zero_in_degree``.
    """
    model = infra["model"]
    n_checked = 0
    for f in flows:
        h_new = f["h_new"]
        non_seed = [i for i in range(h_new.size(0)) if i not in f["seed_local"]]
        if not non_seed:
            continue
        n_checked += 1
        h_flat = _flattened_h_full(
            f["ctx"], model, f["gnid_to_local"], f["edge_index"],
        )
        assert h_flat.shape == h_new.shape
        delta = float((h_new[non_seed] - h_flat[non_seed]).abs().max())
        assert delta > PER_BLOCK_VS_FLAT_EPS, (
            f"per-block h_full is indistinguishable from the flattened, "
            f"hop-mixed control on the non-seed rows (max|Δ|={delta:.3e}) — "
            "the layer-depth fix is not in effect"
        )
    assert n_checked > 0, "no flow with a non-seed node was available"


# ── (e) device safety ─────────────────────────────────────────────────────────

def test_resolve_compute_device_reads_model_parameters():
    """_resolve_compute_device must report the model's parameter device, not CPU.

    No CUDA hardware is touched: ``torch.device("cuda:0")`` is a pure value
    object.  Also pins the zero-parameter case, where the device would be
    unresolvable and every downstream device assert vacuous.
    """
    from src.baselines.adapter import _resolve_compute_device

    class _FakeParam:
        """Stand-in for a model parameter carrying only a device."""

        device = torch.device("cuda:0")

    class _FakeModel:
        """Model stub whose parameters live on a non-CPU device."""

        def parameters(self):
            """Yield one parameter-like object on cuda:0."""
            yield _FakeParam()

    assert _resolve_compute_device(_FakeModel()) == torch.device("cuda:0")

    class _EmptyModel:
        """Model stub with no parameters at all."""

        def parameters(self):
            """Yield nothing."""
            return iter([])

    with pytest.raises(AssertionError, match="no parameters"):
        _resolve_compute_device(_EmptyModel())


@requires_artifacts
def test_build_h_full_follows_the_model_device(monkeypatch, infra, flows):
    """build_h_full must take its device from the model and refuse a mismatch.

    The fixture's ctx tensors are CPU-resident.  With
    ``_resolve_compute_device`` monkeypatched to claim ``cuda:0``, the
    device-agreement asserts must fire and name the device.

    HONESTY NOTE (spec 27 §5.2, risk R2) — what this does and does not prove.
    It proves the fixed behaviour: the device is resolved through the named
    helper and every ctx tensor is checked against it, so a caller that mixes
    devices fails loudly here instead of producing silently wrong numbers.  It
    is NOT a counterfactual catch of Bug 1: pre-fix there was no device
    resolution to monkeypatch, and pre-fix code would simply have SUCCEEDED
    here, building everything on the CPU.  The only test that would have caught
    Bug 1 as it was written is an end-to-end GPU run, which cannot execute in
    this environment (no CUDA device).
    """
    import src.baselines.adapter as adapter

    monkeypatch.setattr(
        adapter, "_resolve_compute_device",
        lambda model: torch.device("cuda:0"),
    )
    f = flows[0]
    with pytest.raises((AssertionError, RuntimeError)) as exc:
        adapter.build_h_full(
            f["ctx"], infra["model"], f["gnid_to_local"], f["edge_index"],
        )
    msg = str(exc.value)
    assert "cuda" in msg.lower()
    # Must be the DEVICE guard specifically, not the eval-mode or layer-count
    # assert that runs before it.
    assert "device mismatch" in msg


# ── (f) third-party behaviour the per-block design rests on ───────────────────

def test_sageconv_self_lift_on_zero_in_degree():
    """A zero-in-degree node's SAGEConv output must be the self term alone.

    This is the mathematical fact the per-block design's self-lift depends on
    (spec 26 §3.2): for a node that appears in ``blocks[i]`` only as a source,
    the convolution has an empty in-neighbourhood, so it must not raise and its
    output must be exactly ``fc_self(x)`` — no invented neighbour message.  If
    a future DGL release changes this (e.g. starts raising on zero in-degree,
    as GraphConv/GATConv do), it fails here with a clear cause rather than
    silently changing every baseline's neighbour attributions.
    """
    from dgl.nn import SAGEConv

    g = dgl.graph((torch.tensor([0]), torch.tensor([1])), num_nodes=3)
    x = torch.randn(3, 4)

    for aggregator in ("mean", "pool", "lstm"):
        torch.manual_seed(0)
        conv = SAGEConv(4, 8, aggregator).eval()
        with torch.no_grad():
            h = conv(g, x)                      # must not raise on node 2
        assert torch.isfinite(h).all(), f"{aggregator}: non-finite output"
        with torch.no_grad():
            self_only = conv.fc_self(x[2])
        assert torch.allclose(h[2], self_only, atol=1e-6), (
            f"{aggregator}: a zero-in-degree node's output is no longer "
            "fc_self(x) alone — the self-lift assumption is broken"
        )

    # The 'gcn' aggregator is explicitly NOT covered: it has no fc_self at all,
    # so the exact self-lift identity above does not apply to it.  No config in
    # this repo selects it (configs/default.yaml, configs/tuning_grid.yaml and
    # artifacts/best_params.json are all "mean"), which is why build_h_full's
    # docstring scopes its self-lift claim to mean/pool/lstm.
    assert not hasattr(SAGEConv(4, 8, "gcn"), "fc_self")


# ── guard rails ───────────────────────────────────────────────────────────────

@requires_artifacts
def test_rejects_train_mode_model(infra, flows):
    """Train-mode BatchNorm over a handful of nodes must fail loud, not silently."""
    from src.baselines.adapter import build_h_full

    model = infra["model"]
    f = flows[0]
    model.train()
    try:
        with pytest.raises(AssertionError):
            build_h_full(f["ctx"], model, f["gnid_to_local"], f["edge_index"])
    finally:
        model.eval()


@requires_artifacts
def test_wrappers_share_one_implementation():
    """All four wrappers must call the shared helper, not a local copy."""
    import inspect
    import src.baselines.adapter as adapter

    wrappers = [
        "graphsvx_wrapper", "gnnshap_wrapper",
        "pgexplainer_wrapper", "edgeshaper_wrapper",
    ]
    assert callable(adapter.build_h_full)
    for name in wrappers:
        src_text = (REPO_ROOT / "src" / "baselines" / f"{name}.py").read_text()
        assert "def _build_h_full" not in src_text, (
            f"{name} still defines its own _build_h_full copy"
        )
        assert "build_h_full(" in src_text, f"{name} does not call build_h_full"
    # PGExplainer has TWO call sites: inference AND train_pgexplainer.  Training
    # the mask-MLP on degenerate embeddings and applying it to correct ones would
    # be worse than either.
    pg = (REPO_ROOT / "src" / "baselines" / "pgexplainer_wrapper.py").read_text()
    assert pg.count("h_full = build_h_full(") == 2, (
        "pgexplainer_wrapper must build h_full via the shared helper in BOTH "
        "train_pgexplainer() and run_pgexplainer_with_model()"
    )
    assert inspect.signature(adapter.build_h_full).parameters.keys() >= {
        "ctx", "model", "gnid_to_local", "edge_index"
    }
