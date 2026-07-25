"""Shared path-resolution helper for third-party baseline-explainer checkouts.

Each of the three baseline wrappers that vendor a non-pip-installable
third-party research repo (gnnshap_wrapper.py, graphsvx_wrapper.py,
edgeshaper_wrapper.py) resolves its clone directory through this module
instead of hardcoding an absolute, developer-machine-specific path.

Deliberately import-light: stdlib only (os, pathlib). No torch/dgl/numpy
import here, so importing this module never requires GNNShap, GraphSVX,
EdgeSHAPer, or any heavy third-party dependency to be installed or present
on disk -- this keeps resolve_baseline_dir unit-testable in isolation
(see tests/test_baseline_paths.py).
"""

from __future__ import annotations

import os
from pathlib import Path

# src/baselines/_paths.py -> parents[0]=src/baselines, [1]=src, [2]=repo root.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def resolve_baseline_dir(env_var: str, clone_name: str, *extra: str) -> str:
    """Resolve a third-party baseline explainer's local checkout directory.

    Precedence:
      1. ``env_var``, if set in the environment to a non-empty value --
         interpreted as the clone *root* (where the developer ran
         ``git clone``), for a developer who keeps the checkout outside this
         repo. ``extra`` path segments are appended to it identically to the
         default branch below, so the override always means "where did you
         clone the repo," never "where is the importable subpath."
      2. Otherwise, the repo-relative default:
         ``<repo_root>/external/<clone_name>``.

    Args:
        env_var: Name of the environment variable that may override the
            default, e.g. ``"SHAP_GSD_GNNSHAP_DIR"``.
        clone_name: Subdirectory name under ``external/``, e.g. ``"gnnshap"``.
        *extra: Additional path segments appended to the clone root in BOTH
            branches -- e.g. ``"src"`` for EdgeSHAPer, whose importable
            module lives one level inside its checkout
            (``EdgeSHAPer/src/edgeshaper.py``, not ``EdgeSHAPer/edgeshaper.py``).

    Returns:
        Absolute path as a plain ``str`` -- never ``pathlib.Path``. Callers
        pass this directly to ``os.chdir()``, ``sys.path.insert()``,
        ``x in sys.path`` membership checks, and ``os.path.join()``, all of
        which expect ``str`` (a ``Path`` would never equality-match the
        strings already in ``sys.path``, silently defeating the
        already-inserted membership check and re-inserting a duplicate).
    """
    override = os.environ.get(env_var)
    base = Path(override) if override else REPO_ROOT / "external" / clone_name
    if extra:
        base = base.joinpath(*extra)
    return str(base)
