"""
Tests for the ``run.dir`` output-path prefixing in ``src/utils/config.py``.

Exercises ``_apply_run_dir_prefix`` directly on in-memory dicts (no YAML I/O)
so the allowlist and the flat-output-path schema contract (spec 06 §2.1.3-B)
are pinned down independently of on-disk config files.

Tests:
  1 — topology prefixed:        new allowlist entry rewrites a relative key.
  2 — data.csv_path untouched:  read-only input path is never prefixed.
  3 — output/graph unchanged:   existing prefixing behavior preserved.
  4 — absolute paths untouched: absolute values are returned as-is.
  5 — no-op when run.dir empty/absent: cfg returned unchanged.
  6 — flat-schema safety:       nested dict under an allowlisted section is skipped.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import _apply_run_dir_prefix  # noqa: E402


def test_topology_relative_key_is_prefixed() -> None:
    """A relative output key under ``topology`` gets the run.dir prefix.

    This is the new behavior added by the allowlist extension and the core
    regression guard for spec 05's future ``topology:`` block.
    """
    cfg: dict = {
        "run": {"dir": "runs/foo"},
        "topology": {"output_path": "outputs/topology"},
    }
    result = _apply_run_dir_prefix(cfg)
    assert result["topology"]["output_path"] == str(
        Path("runs/foo") / "outputs/topology"
    )


def test_data_csv_path_is_not_prefixed() -> None:
    """``data.csv_path`` is a read-only input path and must never be prefixed."""
    cfg: dict = {
        "run": {"dir": "runs/foo"},
        "data": {"csv_path": "data/NF-UNSW-NB15-v3.csv"},
    }
    result = _apply_run_dir_prefix(cfg)
    assert result["data"]["csv_path"] == "data/NF-UNSW-NB15-v3.csv"


def test_output_and_graph_prefixing_unchanged() -> None:
    """Relative keys under ``output`` and ``graph`` are still prefixed as before."""
    cfg: dict = {
        "run": {"dir": "runs/foo"},
        "output": {"outputs_dir": "outputs"},
        "graph": {"graph_dir": "graphs"},
    }
    result = _apply_run_dir_prefix(cfg)
    assert result["output"]["outputs_dir"] == str(Path("runs/foo") / "outputs")
    assert result["graph"]["graph_dir"] == str(Path("runs/foo") / "graphs")


def test_absolute_paths_are_not_modified() -> None:
    """An absolute path under an allowlisted section is left untouched."""
    abs_path = "/mnt/hpc/scratch/outputs"
    cfg: dict = {
        "run": {"dir": "runs/foo"},
        "output": {"outputs_dir": abs_path},
        "topology": {"output_path": abs_path},
    }
    result = _apply_run_dir_prefix(cfg)
    assert result["output"]["outputs_dir"] == abs_path
    assert result["topology"]["output_path"] == abs_path


def test_empty_run_dir_is_a_noop() -> None:
    """An empty ``run.dir`` returns the config unchanged."""
    cfg: dict = {
        "run": {"dir": ""},
        "output": {"outputs_dir": "outputs"},
    }
    result = _apply_run_dir_prefix(cfg)
    assert result["output"]["outputs_dir"] == "outputs"


def test_absent_run_section_is_a_noop() -> None:
    """A missing ``run`` section returns the config unchanged."""
    cfg: dict = {"output": {"outputs_dir": "outputs"}}
    result = _apply_run_dir_prefix(cfg)
    assert result["output"]["outputs_dir"] == "outputs"


def test_flat_schema_nested_value_is_not_prefixed() -> None:
    """A nested dict under an allowlisted section is skipped (not a ``str``).

    Enforces the flat-output-path schema (spec 06 §2.1.3-B): the loop walks
    only one level deep, so a nested ``topology.gateway_distance.output_path``
    is NOT silently prefixed. This documents why the schema must stay flat
    rather than the loop being made recursive — any output-path key must sit
    exactly one level below its section root.
    """
    cfg: dict = {
        "run": {"dir": "runs/foo"},
        "topology": {
            "gateway_distance": {"output_path": "outputs/nested"},
            "k_max": 3,
        },
    }
    result = _apply_run_dir_prefix(cfg)
    # Nested dict is left entirely unchanged: the shallow loop never reaches it.
    assert result["topology"]["gateway_distance"]["output_path"] == "outputs/nested"
    # Non-string scalar under the section is likewise untouched.
    assert result["topology"]["k_max"] == 3


def test_topology_list_value_is_not_prefixed() -> None:
    """A flat list value under ``topology`` is not mangled (spec 06 §2.1.5-d).

    Spec 05 §9.2 gives ``internal_prefixes`` a *list* shape; a list is not a
    ``str`` so the ``isinstance(v, str)`` guard skips it, and its relative-looking
    entries (e.g. CIDR-like tokens) are never rewritten into run.dir paths.
    """
    cfg: dict = {
        "run": {"dir": "runs/foo"},
        "topology": {"internal_prefixes": ["10.0.0.0/8", "192.168.0.0/16"]},
    }
    result = _apply_run_dir_prefix(cfg)
    assert result["topology"]["internal_prefixes"] == [
        "10.0.0.0/8",
        "192.168.0.0/16",
    ]
