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

#: Fraction of eligible flows that must become sensitive.  Not 1.0: on a
#: minority of flows the wrappers' UNNORMALISED residual aggregation
#: (h = h_full + sum of neighbour rows, unchanged by this fix) drives the
#: readout far enough outside the edge MLP's operating range that the softmax
#: saturates and no coalition change is visible in probability space.  That is
#: a separate property of the masking mechanism, which this fix deliberately
#: leaves alone; measured at 143/149 (96%) over a 150-flow sample.
SENSITIVE_FRACTION = 0.75


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
        d_new = abs(_surrogate_p(f["h_new"], ei[:, C], ctx, sl, dl, edge_mlp)
                    - _surrogate_p(f["h_new"], ei[:, C_with], ctx, sl, dl, edge_mlp))

        assert d_old == 0.0, (
            "control failed: the legacy zero-fill matrix was expected to be "
            "exactly insensitive to a non-seed-sourced edge"
        )
        if d_new > 1e-9:
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
