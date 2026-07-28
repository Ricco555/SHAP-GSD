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
    Bug 1 as it was written is an end-to-end GPU run — not part of this
    automated suite (it needs real CUDA hardware, which this machine happens
    to have but a fresh clone's CI/dev environment may not).  A manual
    end-to-end GPU run of build_h_full was performed once, out-of-band, on
    this repo's development machine and succeeded on 12/12 real sampled flows
    (see specs/27 §8 R1 addendum); that check is not reproduced here because
    it depends on real trained artifacts and a CUDA device, neither of which
    this suite's other tests require.
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


def test_wrappers_share_one_implementation():
    """All four wrappers must call the shared helper, not a local copy.

    Not gated behind ``@requires_artifacts``: it only reads wrapper source
    files and ``inspect.signature``, so it must run on a fresh clone too.
    """
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


# ── shared node-baseline helpers (Unit A) ─────────────────────────────────────

#: The four wrappers whose result rows live in node-coalition space and which
#: must therefore share one packer, one degeneracy classifier and one timing
#: convention.  ``gnnexplainer_wrapper`` is deliberately absent: its rows are in
#: feature-group space with a different schema.
NODE_COALITION_WRAPPERS = [
    "graphsvx_wrapper", "gnnshap_wrapper",
    "pgexplainer_wrapper", "edgeshaper_wrapper",
]

#: Exact CSV field set every node-coalition baseline row must carry.  Frozen
#: here so extracting the packer cannot silently add, drop or rename a column.
EXPECTED_ROW_KEYS = {
    "edge_id", "true_label", "predicted_label", "p_full", "node_scores",
    "fidelity_plus", "fidelity_minus", "runtime_s", "fallback_reason",
    "n_local", "n_h_full_nonzero", "n_subgraph_edges",
    "n_live_edges_into_readout", "n_live_edges_into_src", "surrogate_delta_logit",
}


class _StubCtx:
    """Minimal stand-in for FlowContext — the packer reads only these four."""

    def __init__(self) -> None:
        self.global_eid = 7
        self.true_label = 1
        self.predicted_label = 1
        self.p_full = 0.9876543


def test_pack_node_baseline_result_row_contract():
    """The shared packer must emit exactly the historical row schema.

    No artifacts needed: the packer reads four scalars off ctx and merges a
    diagnostic block, so a stub ctx exercises the whole contract.
    """
    from src.baselines.adapter import (
        empty_diagnostics, pack_node_baseline_result,
    )

    row = pack_node_baseline_result(
        _StubCtx(), np.array([0.5, 0.25]), 0.1, 0.2, 1.23456,
    )
    assert set(row) == EXPECTED_ROW_KEYS
    assert row["edge_id"] == 7
    assert row["node_scores"] == [0.5, 0.25]
    assert row["p_full"] == 0.987654          # rounded to 6dp
    assert row["runtime_s"] == 1.235          # rounded to 3dp
    assert row["fallback_reason"] is None
    # No diag supplied → neutral block, surrogate_delta_logit NaN (not 0.0, which
    # would read as "measured and constant").
    assert np.isnan(row["surrogate_delta_logit"])
    for k, v in empty_diagnostics().items():
        assert row[k] == v

    # A supplied diag block overrides the neutral defaults, including a real
    # surrogate_delta_logit measurement.
    diag = dict(empty_diagnostics(n_local=4, n_subgraph_edges=9))
    diag["surrogate_delta_logit"] = 0.25
    row2 = pack_node_baseline_result(
        _StubCtx(), np.zeros(3), 0.0, 0.0, 0.0,
        fallback_reason="empty_subgraph", diag=diag,
    )
    assert set(row2) == EXPECTED_ROW_KEYS
    assert row2["fallback_reason"] == "empty_subgraph"
    assert row2["n_local"] == 4
    assert row2["n_subgraph_edges"] == 9
    assert row2["surrogate_delta_logit"] == 0.25


def test_classify_degenerate_attribution():
    """All-zero, saturated, informative and empty must be told apart."""
    from src.baselines.adapter import (
        SATURATION_EPS, classify_degenerate_attribution,
    )

    assert classify_degenerate_attribution(np.zeros(5)) == "estimator_all_zero"
    assert classify_degenerate_attribution(
        np.full(5, SATURATION_EPS / 10.0)
    ) == "estimator_saturated"
    # Sign must not matter — the classifier reads magnitudes.
    assert classify_degenerate_attribution(
        np.array([0.0, -SATURATION_EPS / 2.0])
    ) == "estimator_saturated"
    # Exactly at the threshold is NOT saturated (strict <).
    assert classify_degenerate_attribution(
        np.array([SATURATION_EPS])
    ) is None
    assert classify_degenerate_attribution(np.array([0.0, 0.3, -1.0])) is None
    assert classify_degenerate_attribution(np.array([])) is None


def test_empty_diagnostics_reports_known_structure_only():
    """Cheap structural counters are real; h_full-derived ones are unmeasured."""
    from src.baselines.adapter import empty_diagnostics, surrogate_diagnostics

    d = empty_diagnostics(n_local=6, n_subgraph_edges=11)
    assert d["n_local"] == 6
    assert d["n_subgraph_edges"] == 11
    for k in ("n_h_full_nonzero", "n_live_edges_into_readout",
              "n_live_edges_into_src"):
        assert d[k] == 0
    # Same key set as a real measurement, so pack_node_baseline_result can
    # merge either one interchangeably.
    real = surrogate_diagnostics(
        torch.ones(3, 4), torch.tensor([[0, 1], [1, 2]]), 1, 2
    )
    assert set(d) == set(real)
    # The docstring must state the UNMEASURED semantics — the only thing
    # stopping a reader from mistaking a fallback row for a dead-row bug.
    assert "UNMEASURED" in empty_diagnostics.__doc__


def test_wrappers_share_the_node_baseline_packer():
    """No wrapper may keep a private _pack_result copy.

    Source-text only (no artifacts, no vendored explainer libraries), matching
    ``test_wrappers_share_one_implementation``'s idiom.
    """
    import src.baselines.adapter as adapter

    assert callable(adapter.pack_node_baseline_result)
    for name in NODE_COALITION_WRAPPERS:
        src_text = (REPO_ROOT / "src" / "baselines" / f"{name}.py").read_text()
        assert "def _pack_result" not in src_text, (
            f"{name} still defines its own _pack_result copy"
        )
        assert "pack_node_baseline_result(" in src_text, (
            f"{name} does not call the shared packer"
        )
    # gnnexplainer_wrapper is in feature-group space — it must NOT be folded in.
    gnnexp = REPO_ROOT / "src" / "baselines" / "gnnexplainer_wrapper.py"
    if gnnexp.exists():
        assert "pack_node_baseline_result" not in gnnexp.read_text(), (
            "gnnexplainer_wrapper has a different row schema and must not use "
            "the node-coalition packer"
        )


def test_wrappers_share_the_degeneracy_classifier():
    """All four wrappers — GraphSVX included — must classify degenerate rows.

    GraphSVX previously had NO all-zero/saturation check at all, so it
    under-reported its own degenerate rows relative to the other three.  An
    end-to-end saturated-phi test would require the vendored GraphSVX library,
    which this module deliberately does not depend on; the classifier's own
    unit test plus this source assertion is the coverage that is actually
    available on a fresh clone.
    """
    for name in NODE_COALITION_WRAPPERS:
        src_text = (REPO_ROOT / "src" / "baselines" / f"{name}.py").read_text()
        assert "classify_degenerate_attribution(" in src_text, (
            f"{name} does not use the shared degeneracy classifier"
        )
        assert 'fallback_reason = "estimator_all_zero"' not in src_text, (
            f"{name} still assigns the degeneracy string inline"
        )
        assert 'fallback_reason = "estimator_saturated"' not in src_text, (
            f"{name} still assigns the degeneracy string inline"
        )
    # GraphSVX must not let the new check clobber its pre-existing reason.
    gsvx = (REPO_ROOT / "src" / "baselines" / "graphsvx_wrapper.py").read_text()
    assert "fallback_reason or classify_degenerate_attribution(" in gsvx, (
        "GraphSVX must preserve an already-set fallback_reason "
        "(e.g. empty_phi_or_neighbours)"
    )


def test_wrappers_exclude_diagnostic_time_without_mutating_t0():
    """Diagnostic time must be accumulated, not folded back into t0.

    ``t0 += ...`` silently redefined what t0 meant partway through each
    function; the ``diag_s`` accumulator keeps t0 meaning "start" at every exit.
    """
    for name in NODE_COALITION_WRAPPERS:
        src_text = (REPO_ROOT / "src" / "baselines" / f"{name}.py").read_text()
        assert "t0 +=" not in src_text, (
            f"{name} still mutates t0 to exclude diagnostic time"
        )
        assert "diag_s = 0.0" in src_text, f"{name} has no diag_s accumulator"
        # Every runtime computation must subtract the accumulator.
        assert src_text.count("time.time() - t0") == \
            src_text.count("time.time() - t0 - diag_s"), (
            f"{name} computes a runtime that does not exclude diag_s"
        )


def test_build_h_full_runs_after_the_early_return_checks():
    """build_h_full must not run for flows that immediately fall back.

    Pinning this needs no artifacts: the textual order of the isolated-flow
    guard and the build_h_full call in each wrapper's inference function is the
    property.
    """
    for name in NODE_COALITION_WRAPPERS:
        src_text = (REPO_ROOT / "src" / "baselines" / f"{name}.py").read_text()
        # PGExplainer also builds h_full inside train_pgexplainer(), which sits
        # ABOVE the inference function; scope the check to the inference half.
        if name == "pgexplainer_wrapper":
            src_text = src_text[src_text.index("def run_pgexplainer_with_model"):]
        # Qualified form: the bare "edge_index.size(1) == 0" also occurs in
        # module-level helpers ABOVE the inference function (e.g. EdgeSHAPer's
        # _edge_shap_to_node_scores), which would make this assertion vacuous.
        i_guard = src_text.index("pyg_data.edge_index.size(1) == 0")
        i_build = src_text.index("build_h_full(ctx")
        assert i_guard < i_build, (
            f"{name} builds h_full before the isolated-flow early return, "
            "wasting a full message-passing pass on a discarded result"
        )

    # GNNShap's SECOND early return (too few coalition players) must also
    # precede the build.
    gs = (REPO_ROOT / "src" / "baselines" / "gnnshap_wrapper.py").read_text()
    assert gs.index("N_local < 3") < gs.index("build_h_full(ctx"), (
        "gnnshap_wrapper builds h_full before the insufficient_players return"
    )


# ── runtime exercise of the early-return paths (no artifacts, no vendored libs) ─

class _StubEdgeMLPModel(torch.nn.Module):
    """Only ``edge_mlp`` is touched before any wrapper's early return."""

    def __init__(self) -> None:
        super().__init__()
        self.edge_mlp = torch.nn.Linear(4, 2)


class _StubBackground:
    background_node_state = {1: np.zeros(15, dtype=np.float32)}


class _StubFlowCtx(_StubCtx):
    """Stub ctx carrying the few fields the early-return paths read."""

    def __init__(self) -> None:
        super().__init__()
        self.input_node_ids = np.arange(3)
        self.base_node_feats = np.zeros((3, 15), dtype=np.float32)
        self.x_e_t = torch.zeros(1, 4)
        self.blocks = []
        self.target_src_nid = 1
        self.target_dst_nid = 2


def _stub_adapter(monkeypatch, n_edges: int) -> None:
    """Point the wrappers at a 2-node local subgraph with ``n_edges`` edges.

    ``build_h_full`` is replaced by a landmine: reaching it on an early-return
    path is exactly the waste this reordering removes.
    """
    from torch_geometric.data import Data
    import src.baselines.adapter as adapter

    edge_index = torch.zeros((2, n_edges), dtype=torch.long)
    data = Data(x=torch.zeros(2, 15), edge_index=edge_index)
    monkeypatch.setattr(
        adapter, "dgl_subgraph_to_pyg", lambda *a, **k: (data, {1: 0, 2: 1})
    )
    monkeypatch.setattr(
        adapter, "fidelity_from_node_mask", lambda *a, **k: (0.1, 0.2)
    )

    def _landmine(*a, **k):
        raise AssertionError(
            "build_h_full ran on an early-return path — a full multi-layer "
            "message-passing pass whose result is immediately discarded"
        )

    monkeypatch.setattr(adapter, "build_h_full", _landmine)


@pytest.mark.parametrize("wrapper_name", NODE_COALITION_WRAPPERS)
def test_empty_subgraph_path_reports_real_n_local(monkeypatch, wrapper_name):
    """Every wrapper's isolated-flow row: full schema, real n_local, no h_full.

    Guards the diagnostic-accuracy half of the reordering: skipping
    ``build_h_full`` must not silently turn ``n_local`` into 0.
    """
    _stub_adapter(monkeypatch, n_edges=0)
    import importlib

    mod = importlib.import_module(f"src.baselines.{wrapper_name}")
    fn_name = f"run_{wrapper_name.replace('_wrapper', '')}_with_model"
    run = getattr(mod, fn_name)

    args = [_StubFlowCtx(), _StubEdgeMLPModel().eval()]
    if wrapper_name == "pgexplainer_wrapper":
        args.append(None)                     # trained algorithm — unused here
    args += [{}, _StubBackground(), None]

    row = run(*args)
    assert set(row) == EXPECTED_ROW_KEYS
    assert row["fallback_reason"] == "empty_subgraph"
    assert row["n_local"] == 2, "real len(gnid_to_local) must survive"
    assert row["n_subgraph_edges"] == 0
    assert row["n_h_full_nonzero"] == 0        # UNMEASURED, per empty_diagnostics
    assert row["runtime_s"] >= 0.0


def test_insufficient_players_path_keeps_the_real_edge_count(monkeypatch):
    """GNNShap's second early return also precedes build_h_full.

    Its subgraph is non-empty (it passed the first guard), so
    ``n_subgraph_edges`` is a real, free measurement and must not be zeroed.
    """
    _stub_adapter(monkeypatch, n_edges=2)
    from src.baselines.gnnshap_wrapper import run_gnnshap_with_model

    row = run_gnnshap_with_model(
        _StubFlowCtx(), _StubEdgeMLPModel().eval(), {}, _StubBackground(), None
    )
    assert row["fallback_reason"] == "insufficient_players"
    assert row["n_local"] == 2
    assert row["n_subgraph_edges"] == 2
    assert row["n_h_full_nonzero"] == 0


# ── Unit B: surrogate_delta is a LOGIT-space measurement ──────────────────────

#: Small synthetic surrogate dimensions for the wrapper output-space tests:
#: hidden size, edge-feature width, class count.  The wrappers' edge MLP input
#: is ``2 * _H + _DE`` (h_src ‖ h_dst ‖ x_e).
_H, _DE, _C = 3, 4, 2


def _toy_edge_mlp() -> torch.nn.Module:
    """Deterministic edge MLP with a large output scale.

    The scale matters: it pushes the readout into the regime where float32
    softmax saturates, which is exactly the condition under which a
    probability-space delta collapses to 0.0 while the logit-space delta does
    not.  Seeded so the assertions below are reproducible.
    """
    torch.manual_seed(1234)
    mlp = torch.nn.Sequential(torch.nn.Linear(2 * _H + _DE, _C))
    with torch.no_grad():
        mlp[0].weight.mul_(25.0)
        mlp[0].bias.zero_()
    return mlp.eval()


def _toy_surrogate_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(h, edge_index, x_e_t) for a 3-node, 2-edge synthetic local subgraph."""
    torch.manual_seed(99)
    h = torch.randn(3, _H)
    edge_index = torch.tensor([[1, 2], [0, 0]], dtype=torch.long)
    x_e_t = torch.randn(1, _DE)
    return h, edge_index, x_e_t


def test_surrogate_delta_is_computed_in_logit_space():
    """The function must report max|Δ| over RAW class scores, unrounded.

    Direct unit test with a synthetic ``predict``: no artifacts and no vendored
    explainer library, so it runs on a fresh clone.  The full-coalition and
    empty-coalition returns are chosen so the answer is exact and so that the
    per-class maximum — not the first class, and not the true class — is what
    comes back.
    """
    from src.baselines.adapter import surrogate_delta

    def predict(ei: torch.Tensor) -> torch.Tensor:
        # Two classes; the SECOND moves further, so a max (not [0]) is required.
        return (torch.tensor([-4.0, 11.5]) if ei.size(1) > 0
                else torch.tensor([-1.5, 2.25]))

    edge_index = torch.tensor([[1, 2], [0, 0]], dtype=torch.long)
    delta = surrogate_delta(predict, edge_index)
    assert delta == pytest.approx(9.25, abs=1e-6)   # |11.5 - 2.25|, not |−4 −(−1.5)|

    # A provably flat surrogate is exactly 0.0 — the one meaning the docstring
    # reserves for that value.
    assert surrogate_delta(lambda ei: torch.tensor([0.5, -0.5]), edge_index) == 0.0


def test_surrogate_delta_returns_nan_not_zero_on_failure():
    """A failed measurement must never be mistakable for a flat surrogate.

    Pins the docstring correction: the fallback is ``float("nan")``, not the
    ``0.0`` the docstring previously claimed.  0.0 already means "measured, and
    provably flat", so conflating the two would turn every measurement error
    into a false flatness verdict.
    """
    from src.baselines.adapter import surrogate_delta

    def exploding(ei: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("synthetic surrogate failure")

    out = surrogate_delta(exploding, torch.tensor([[1], [0]], dtype=torch.long))
    assert np.isnan(out)
    assert out != 0.0
    # The docstring must say so, since NaN-vs-0.0 is the whole contract.
    assert 'float("nan")' in surrogate_delta.__doc__
    assert "or 0.0 if it cannot" not in surrogate_delta.__doc__, (
        "the docstring still advertises the 0.0 fallback the code never had"
    )


def test_surrogate_delta_predict_parameter_is_typed():
    """``predict`` carries a Callable annotation, not a bare untyped name."""
    import inspect
    from src.baselines.adapter import surrogate_delta

    ann = inspect.signature(surrogate_delta).parameters["predict"].annotation
    assert ann is not inspect.Parameter.empty, "predict is still untyped"
    # adapter.py uses `from __future__ import annotations`, so this is a string.
    assert "Callable" in str(ann)


def test_graphsvx_forward_is_log_softmax_of_forward_logits():
    """GraphSVX's library-facing forward must stay in LOG-probability space.

    Relational, deliberately: asserting ``forward == log_softmax(forward_logits)``
    proves both halves at once — that ``forward_logits`` really is the
    pre-normalisation reading, and that ``forward`` (which the vendored
    GraphSVX consumes as ``self.model(x, edge_index).exp()[node_index]``, see
    the wrapper's own class docstring) was not accidentally moved to raw logits.
    """
    from src.baselines.graphsvx_wrapper import GraphSVXNodeWrapper

    h, edge_index, x_e_t = _toy_surrogate_inputs()
    n = h.size(0)
    w = GraphSVXNodeWrapper(_toy_edge_mlp(), x_e_t, 0, 1, n).eval()

    with torch.no_grad():
        logits = w.forward_logits(h, edge_index)
        out = w(h, edge_index)
    assert logits.shape == (1, _C)
    assert out.shape == (n, _C)
    assert torch.allclose(
        torch.log_softmax(logits, dim=1).expand(n, -1), out, atol=1e-6
    )
    # What GraphSVX actually reads: .exp() of a row must be a distribution.
    assert torch.allclose(out.exp().sum(dim=1), torch.ones(n), atol=1e-5)
    # And the logits must NOT already be a distribution — i.e. this is a real
    # change of space, not a no-op rename.
    assert not torch.allclose(logits.exp().sum(), torch.tensor(1.0), atol=1e-3)


@pytest.mark.parametrize("weighted", [False, True])
def test_pgexplainer_forward_is_softmax_of_forward_logits(weighted):
    """PGExplainer's library-facing forward must stay in PROBABILITY space.

    The wrapper is registered with PyG under ``ModelReturnType.probs`` /
    ``return_type="probs"`` (both call sites in pgexplainer_wrapper.py), so
    PGExplainer's own mask loss reads these rows as a probability vector.
    Parametrised over ``edge_weight`` so the weighted-aggregation branch — the
    one PGExplainer actually exercises — is covered too.
    """
    from src.baselines.pgexplainer_wrapper import PGECompatibleWrapper

    h, edge_index, x_e_t = _toy_surrogate_inputs()
    n = h.size(0)
    w = PGECompatibleWrapper(h, _toy_edge_mlp(), x_e_t, 0, 1, n).eval()
    ew = torch.ones(edge_index.size(1)) * 0.7 if weighted else None

    with torch.no_grad():
        logits = w.forward_logits(h, edge_index, ew)
        out = w(h, edge_index, ew)
    assert logits.shape == (1, _C)
    assert torch.allclose(
        torch.softmax(logits, dim=1).expand(n, -1), out, atol=1e-6
    )
    assert torch.allclose(out.sum(dim=1), torch.ones(n), atol=1e-5)
    assert bool(((out >= 0.0) & (out <= 1.0)).all())
    assert not torch.allclose(logits.sum(), torch.tensor(1.0), atol=1e-3)


@pytest.mark.parametrize("weighted", [False, True])
def test_gnnshap_forward_fn_default_stays_probabilities(weighted):
    """GNNShap's library-facing forward_fn must stay in PROBABILITY space.

    ``return_logits`` lives on the FACTORY, never on the returned closure:
    GNNShap invokes ``forward_fn(model, node_features, edge_index, node_idx)``
    positionally, so a flag on the closure would be a positional-collision
    hazard.  Asserting the two closures are softmax-related proves the default
    one (handed to the WLS solver) is untouched.
    """
    from src.baselines.gnnshap_wrapper import _make_forward_fn

    h, edge_index, x_e_t = _toy_surrogate_inputs()
    mlp = _toy_edge_mlp()
    probs_fn = _make_forward_fn(mlp, x_e_t, 0, 1)
    logit_fn = _make_forward_fn(mlp, x_e_t, 0, 1, return_logits=True)
    ew = torch.ones(edge_index.size(1)) * 0.7 if weighted else None

    with torch.no_grad():
        p = probs_fn(None, h, edge_index, 0, ew)
        z = logit_fn(None, h, edge_index, 0, ew)
    assert p.shape == (_C,) and z.shape == (_C,)
    assert torch.allclose(torch.softmax(z.unsqueeze(0), dim=1).squeeze(0), p,
                          atol=1e-6)
    assert torch.allclose(p.sum(), torch.tensor(1.0), atol=1e-5)
    assert not torch.allclose(z.sum(), torch.tensor(1.0), atol=1e-3)


def test_all_wrappers_emit_the_logit_named_column_only():
    """Every wrapper writes ``surrogate_delta_logit`` and nobody writes the old key.

    Source-text (this module's idiom for cross-wrapper contracts — no artifacts,
    no vendored libraries).  The rename is what stops a historical mixed-space
    CSV from being silently reinterpreted as the new uniform column, so a single
    wrapper left behind would defeat it.
    """
    for name in NODE_COALITION_WRAPPERS:
        src_text = (REPO_ROOT / "src" / "baselines" / f"{name}.py").read_text()
        assert 'diag["surrogate_delta_logit"]' in src_text, (
            f"{name} does not record the logit-space column"
        )
        assert 'diag["surrogate_delta"]' not in src_text, (
            f"{name} still records the old ambiguous-space column"
        )

    # Three of the four must take the measurement off a logits-only entry point.
    for name, needle in (
        ("graphsvx_wrapper",    "wrapper.forward_logits("),
        ("pgexplainer_wrapper", "wrapper.forward_logits("),
        ("gnnshap_wrapper",     "return_logits=True"),
    ):
        src_text = (REPO_ROOT / "src" / "baselines" / f"{name}.py").read_text()
        assert needle in src_text, (
            f"{name} does not measure surrogate_delta on raw logits"
        )
    # EdgeSHAPer is the deliberate exception: its own API mandates raw logits
    # from the model, so forward IS the logit entry point.
    es = (REPO_ROOT / "src" / "baselines" / "edgeshaper_wrapper.py").read_text()
    assert "def forward_logits" not in es, (
        "EdgeSHAPModelWrapper.forward already returns raw logits; a "
        "forward_logits added 'for symmetry' would be dead code"
    )


def test_no_repo_code_expects_a_zero_surrogate_delta_fallback():
    """Nothing anywhere coerces the column's NaN to 0.0 or tests it for equality.

    The old docstring advertised a ``0.0`` failure fallback the code never
    implemented.  Correcting the prose is only safe if no consumer was written
    against the wrong contract — this walks the whole source tree rather than
    trusting that.
    """
    hits: list[str] = []
    for sub in ("src", "scripts", "explore", "tests"):
        root = REPO_ROOT / sub
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path.name == Path(__file__).name:
                continue
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if "surrogate_delta" not in line:
                    continue
                if any(tok in line for tok in
                       ("fillna", "== 0.0", "!= 0.0", "or 0.0", "nan_to_num")):
                    hits.append(f"{path.relative_to(REPO_ROOT)}:{i}: {line.strip()}")
    assert not hits, (
        "code treats a surrogate_delta failure as 0.0, conflating it with a "
        "provably flat surrogate:\n" + "\n".join(hits)
    )
