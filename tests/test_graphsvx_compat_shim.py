"""
Tests for graphsvx_wrapper.py's torch_geometric-version compatibility shim
(_ensure_pyg_gnnexplainer_compat, specs/43, specs/44).

Root cause (specs/43 sec 1): the vendored GraphSVX checkout's
src/explainers.py does, at module level, `from torch_geometric.nn import
GNNExplainer as GNNE`. torch_geometric >= 2.x relocated GNNExplainer to
torch_geometric.explain.algorithm.GNNExplainer, so that import fails on
every module load -- confirmed by direct HPC reproduction (torch_geometric
2.6.1) and reproduced locally against this checkout's own torch_geometric
2.7.0 (both lack torch_geometric.nn.GNNExplainer, confirmed this session).

These tests do NOT require the vendored GraphSVX checkout to exist locally
(it doesn't -- external/graphsvx is absent, confirmed this session). They
verify the shim's monkey-patching behavior against the REAL, installed
torch_geometric in this environment, mocking only what specs/43 sec 7
requires mocking (the "already present" and "absent from both locations"
cases -- the "absent from torch_geometric.nn, present under
explain.algorithm" case is this environment's actual real state and needs
no mocking at all).

Every test restores torch_geometric.nn's real attribute state afterward,
matching test_baseline_wrapper_paths.py's own documented convention --
torch_geometric is a real, shared, already-imported module in the pytest
process, and a test leaving it mutated would corrupt any other test file
that imports torch_geometric.nn later in the same session.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch_geometric.nn as pyg_nn  # noqa: E402
from src.baselines.graphsvx_wrapper import (  # noqa: E402
    _ensure_pyg_gnnexplainer_compat,
)


@pytest.fixture
def _clean_gnnexplainer_attr():
    """Snapshot and restore torch_geometric.nn.GNNExplainer's presence/
    identity around each test, regardless of what the test does to it."""
    had_attr = hasattr(pyg_nn, "GNNExplainer")
    original = getattr(pyg_nn, "GNNExplainer", None)
    yield
    if had_attr:
        pyg_nn.GNNExplainer = original
    elif hasattr(pyg_nn, "GNNExplainer"):
        del pyg_nn.GNNExplainer


def test_noop_when_already_present(_clean_gnnexplainer_attr) -> None:
    """If torch_geometric.nn.GNNExplainer already exists (older PyG, or a
    future PyG that reinstates it), the shim must not touch it."""
    sentinel = object()
    pyg_nn.GNNExplainer = sentinel

    _ensure_pyg_gnnexplainer_compat()

    assert pyg_nn.GNNExplainer is sentinel


def test_patches_relocated_class_when_absent(_clean_gnnexplainer_attr) -> None:
    """Real environment state (confirmed this session): torch_geometric.nn
    lacks GNNExplainer, torch_geometric.explain.algorithm has it. No
    mocking needed for this half -- this IS the checkout's actual state."""
    if hasattr(pyg_nn, "GNNExplainer"):
        del pyg_nn.GNNExplainer
    from torch_geometric.explain.algorithm import GNNExplainer as expected

    # Reproduce the vendored file's exact failing statement shape before
    # the shim runs -- confirms this is the real statement that breaks,
    # not merely a correlated hasattr() check.
    with pytest.raises(ImportError):
        exec("from torch_geometric.nn import GNNExplainer as GNNE", {})

    _ensure_pyg_gnnexplainer_compat()

    assert pyg_nn.GNNExplainer is expected
    # The vendored file's exact import statement now succeeds.
    ns: dict = {}
    exec("from torch_geometric.nn import GNNExplainer as GNNE", ns)
    assert ns["GNNE"] is expected


def test_raises_when_both_locations_absent(
    monkeypatch, _clean_gnnexplainer_attr
) -> None:
    """If a future/older torch_geometric lacks GNNExplainer in BOTH known
    locations, the shim must fail loudly (RuntimeError), not silently
    degrade -- specs/43 sec 4.2's regression pin."""
    if hasattr(pyg_nn, "GNNExplainer"):
        del pyg_nn.GNNExplainer

    import torch_geometric.explain.algorithm as pyg_explain_algo

    monkeypatch.delattr(pyg_explain_algo, "GNNExplainer", raising=False)
    # Force the shim's own `from torch_geometric.explain.algorithm import
    # GNNExplainer` to fail regardless of caching by making the attribute
    # genuinely absent on the real module object for the duration of the
    # test (monkeypatch restores it automatically afterward).

    with pytest.raises(RuntimeError, match="GNNExplainer"):
        _ensure_pyg_gnnexplainer_compat()

    # Must not have been silently patched to anything.
    assert not hasattr(pyg_nn, "GNNExplainer")
