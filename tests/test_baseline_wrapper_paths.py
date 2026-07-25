"""
Tests for the portable path-resolution wiring in the three baseline-explainer
wrappers (gnnshap_wrapper.py, graphsvx_wrapper.py, edgeshaper_wrapper.py).

Complements ``tests/test_baseline_paths.py`` (which exercises
``resolve_baseline_dir`` directly, in isolation from any wrapper). This file
covers two things that module leaves untested:

  1. Current-working-directory independence of ``resolve_baseline_dir`` --
     the task's explicit requirement that the resolved path is correct
     "regardless of current working directory," not merely "when cwd happens
     to be the repo root" (which is all a naive relative-path implementation
     would need to pass).
  2. That each of the three wrapper modules actually *wires up*
     resolve_baseline_dir correctly at import time -- i.e. each wrapper's
     module-level constant (GNNSHAP_DIR / GRAPHSVX_DIR / EDGESHAPER_SRC)
     reflects the env-var-override-vs-default precedence, not just that the
     shared helper function does in isolation.

Requires none of GNNShap/GraphSVX/EdgeSHAPer to be installed: the wrapper
modules only import torch/torch.nn/numpy at module level (verified by direct
import in this environment); their actual third-party imports
(`from gnnshap.explainer import ...`, etc.) are lazy, deferred to inside
function bodies that this test never calls.

Mechanics note: each wrapper's path constant is computed ONCE at import time
(``GNNSHAP_DIR = resolve_baseline_dir(...)`` at module scope), so testing the
env-var-override path requires ``monkeypatch.setenv`` followed by
``importlib.reload(module)`` -- reading ``module.GNNSHAP_DIR`` fresh off the
reloaded module, never a `from ... import GNNSHAP_DIR` binding captured
before the reload (which would be stale). Each reload test restores the
env var and reloads again at the end so the module is left in its default
state for any other test collected in the same session.
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines._paths import REPO_ROOT, resolve_baseline_dir  # noqa: E402
import src.baselines.gnnshap_wrapper as gnnshap_wrapper  # noqa: E402
import src.baselines.graphsvx_wrapper as graphsvx_wrapper  # noqa: E402
import src.baselines.edgeshaper_wrapper as edgeshaper_wrapper  # noqa: E402


# ── 1. cwd independence of the shared resolver ──────────────────────────────

def test_resolve_baseline_dir_default_independent_of_cwd(monkeypatch, tmp_path) -> None:
    """Default resolution must not depend on the process's current working
    directory -- REPO_ROOT is derived from _paths.py's on-disk location
    (__file__), not from os.getcwd()."""
    monkeypatch.delenv("SHAP_GSD_GNNSHAP_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    result = resolve_baseline_dir("SHAP_GSD_GNNSHAP_DIR", "gnnshap")
    assert result == str(REPO_ROOT / "external" / "gnnshap")


def test_resolve_baseline_dir_override_independent_of_cwd(monkeypatch, tmp_path) -> None:
    """Env-var override resolution must also be cwd-independent -- an
    absolute override path must resolve identically no matter where the
    process happens to be running from."""
    another_cwd = tmp_path / "somewhere_else"
    another_cwd.mkdir()
    override_dir = tmp_path / "clone_root"
    monkeypatch.setenv("SHAP_GSD_GRAPHSVX_DIR", str(override_dir))
    monkeypatch.chdir(another_cwd)
    result = resolve_baseline_dir("SHAP_GSD_GRAPHSVX_DIR", "graphsvx")
    assert result == str(override_dir)


# ── 2. each wrapper's module-level constant wires resolve_baseline_dir up ──

def test_gnnshap_wrapper_dir_default(monkeypatch) -> None:
    """GNNSHAP_DIR falls back to <repo_root>/external/gnnshap on reload with
    no env var set."""
    monkeypatch.delenv("SHAP_GSD_GNNSHAP_DIR", raising=False)
    try:
        importlib.reload(gnnshap_wrapper)
        assert gnnshap_wrapper.GNNSHAP_DIR == str(REPO_ROOT / "external" / "gnnshap")
    finally:
        importlib.reload(gnnshap_wrapper)


def test_gnnshap_wrapper_dir_override(monkeypatch, tmp_path) -> None:
    """GNNSHAP_DIR reflects SHAP_GSD_GNNSHAP_DIR after reload."""
    monkeypatch.setenv("SHAP_GSD_GNNSHAP_DIR", str(tmp_path))
    try:
        importlib.reload(gnnshap_wrapper)
        assert gnnshap_wrapper.GNNSHAP_DIR == str(tmp_path)
    finally:
        monkeypatch.delenv("SHAP_GSD_GNNSHAP_DIR", raising=False)
        importlib.reload(gnnshap_wrapper)


def test_graphsvx_wrapper_dir_default(monkeypatch) -> None:
    """GRAPHSVX_DIR falls back to <repo_root>/external/graphsvx on reload
    with no env var set."""
    monkeypatch.delenv("SHAP_GSD_GRAPHSVX_DIR", raising=False)
    try:
        importlib.reload(graphsvx_wrapper)
        assert graphsvx_wrapper.GRAPHSVX_DIR == str(REPO_ROOT / "external" / "graphsvx")
    finally:
        importlib.reload(graphsvx_wrapper)


def test_graphsvx_wrapper_dir_override(monkeypatch, tmp_path) -> None:
    """GRAPHSVX_DIR reflects SHAP_GSD_GRAPHSVX_DIR after reload."""
    monkeypatch.setenv("SHAP_GSD_GRAPHSVX_DIR", str(tmp_path))
    try:
        importlib.reload(graphsvx_wrapper)
        assert graphsvx_wrapper.GRAPHSVX_DIR == str(tmp_path)
    finally:
        monkeypatch.delenv("SHAP_GSD_GRAPHSVX_DIR", raising=False)
        importlib.reload(graphsvx_wrapper)


def test_edgeshaper_wrapper_dir_default(monkeypatch) -> None:
    """EDGESHAPER_SRC falls back to <repo_root>/external/edgeshaper/src on
    reload with no env var set (note the /src subpath, unique to
    EdgeSHAPer's importable-module layout)."""
    monkeypatch.delenv("SHAP_GSD_EDGESHAPER_DIR", raising=False)
    try:
        importlib.reload(edgeshaper_wrapper)
        assert edgeshaper_wrapper.EDGESHAPER_SRC == str(
            REPO_ROOT / "external" / "edgeshaper" / "src"
        )
    finally:
        importlib.reload(edgeshaper_wrapper)


def test_edgeshaper_wrapper_dir_override_appends_src(monkeypatch, tmp_path) -> None:
    """EDGESHAPER_SRC appends /src to the SHAP_GSD_EDGESHAPER_DIR override
    (the clone root), not the override verbatim -- regression guard for the
    same /src-must-append-on-both-branches bug covered for the bare resolver
    in test_baseline_paths.py, now checked through the actual wrapper."""
    monkeypatch.setenv("SHAP_GSD_EDGESHAPER_DIR", str(tmp_path))
    try:
        importlib.reload(edgeshaper_wrapper)
        assert edgeshaper_wrapper.EDGESHAPER_SRC == str(tmp_path / "src")
    finally:
        monkeypatch.delenv("SHAP_GSD_EDGESHAPER_DIR", raising=False)
        importlib.reload(edgeshaper_wrapper)
