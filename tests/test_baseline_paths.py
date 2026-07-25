"""
Tests for ``src/baselines/_paths.py``'s ``resolve_baseline_dir`` helper.

Exercises the env-var-override-vs-default precedence logic directly against
the shared resolver used by gnnshap_wrapper.py, graphsvx_wrapper.py, and
edgeshaper_wrapper.py. Requires none of GNNShap/GraphSVX/EdgeSHAPer/torch/dgl
to be installed or present -- ``src/baselines/_paths.py`` is stdlib-only by
design, so this test module only imports ``os``, ``pathlib``, and
``src.baselines._paths``.

Tests:
  1 — default used when env var unset (no extra segments).
  2 — default used when env var unset, with an extra segment (EdgeSHAPer shape).
  3 — env var override used verbatim when no extra segments given.
  4 — env var override + extra segment: the /src regression guard (spec §I.2).
  5 — empty-string env var falls back to default (not treated as "set").
  6 — return type is always str, never pathlib.Path.
  7 — REPO_ROOT resolves to the actual repo root from this file's location.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines._paths import REPO_ROOT, resolve_baseline_dir  # noqa: E402


def test_default_when_env_var_unset(monkeypatch) -> None:
    """No env var set -> default repo-relative external/<clone_name> path."""
    monkeypatch.delenv("SHAP_GSD_GNNSHAP_DIR", raising=False)
    result = resolve_baseline_dir("SHAP_GSD_GNNSHAP_DIR", "gnnshap")
    assert result == str(REPO_ROOT / "external" / "gnnshap")


def test_default_with_extra_segment_when_env_var_unset(monkeypatch) -> None:
    """Default branch appends *extra segments (EdgeSHAPer's /src shape)."""
    monkeypatch.delenv("SHAP_GSD_EDGESHAPER_DIR", raising=False)
    result = resolve_baseline_dir("SHAP_GSD_EDGESHAPER_DIR", "edgeshaper", "src")
    assert result == str(REPO_ROOT / "external" / "edgeshaper" / "src")


def test_env_var_override_no_extra(monkeypatch, tmp_path) -> None:
    """Env var set, no extra segments -> the override value is used verbatim."""
    monkeypatch.setenv("SHAP_GSD_GRAPHSVX_DIR", str(tmp_path))
    result = resolve_baseline_dir("SHAP_GSD_GRAPHSVX_DIR", "graphsvx")
    assert result == str(tmp_path)


def test_env_var_override_with_extra_appends_subpath(monkeypatch, tmp_path) -> None:
    """Regression test for the spec §I.2 bug: *extra must append on BOTH
    branches, not just the default branch.

    A naive `os.environ.get(env_var, default_with_extra)` implementation
    would return the override verbatim with no /src appended, breaking
    EdgeSHAPer's `from edgeshaper import Edgeshaper` import. This test fails
    if that bug is reintroduced.
    """
    monkeypatch.setenv("SHAP_GSD_EDGESHAPER_DIR", str(tmp_path))
    result = resolve_baseline_dir("SHAP_GSD_EDGESHAPER_DIR", "edgeshaper", "src")
    assert result == str(tmp_path / "src")


def test_empty_string_env_var_falls_back_to_default(monkeypatch) -> None:
    """An env var set to the empty string is treated as unset, not as override."""
    monkeypatch.setenv("SHAP_GSD_GNNSHAP_DIR", "")
    result = resolve_baseline_dir("SHAP_GSD_GNNSHAP_DIR", "gnnshap")
    assert result == str(REPO_ROOT / "external" / "gnnshap")


def test_return_type_is_str_not_path(monkeypatch, tmp_path) -> None:
    """Guards against a future edit reintroducing a Path return (see spec §I.3:
    sys.path membership checks and os.chdir call sites require str)."""
    monkeypatch.delenv("SHAP_GSD_GNNSHAP_DIR", raising=False)
    default_result = resolve_baseline_dir("SHAP_GSD_GNNSHAP_DIR", "gnnshap")
    assert isinstance(default_result, str)
    assert not isinstance(default_result, Path)

    monkeypatch.setenv("SHAP_GSD_GNNSHAP_DIR", str(tmp_path))
    override_result = resolve_baseline_dir("SHAP_GSD_GNNSHAP_DIR", "gnnshap")
    assert isinstance(override_result, str)
    assert not isinstance(override_result, Path)


def test_repo_root_is_actual_repo_root() -> None:
    """REPO_ROOT (parents[2] from _paths.py's on-disk location) resolves to
    the actual SHAP-GSD repo root, not merely an assumed directory depth."""
    assert (REPO_ROOT / "README.md").exists()
    assert (REPO_ROOT / "src" / "baselines" / "_paths.py").exists()
