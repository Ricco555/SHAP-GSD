"""
Tests for phase-5 label-map resolution (``scripts/05_evaluate.py``).

Regression guard for the bug where ``scripts/run_dataset.py`` invoked
``05_evaluate.py`` with only ``--config``, so no ``--label-map`` was supplied
and per-class F1 scores were labelled from ``Evaluator.DEFAULT_CLASS_NAMES``
(TE-G-SAGE's UNSW-NB15 ordering) instead of the dataset's own dynamically
derived class map. Per-class scores were therefore silently permuted for any
dataset whose alphabetical class ordering differs.

Tests:
  1 — explicit --label-map wins and is returned as an absolute path.
  2 — no --label-map: resolves to <artifacts_dir>/label_map.json.
  3 — an orchestrator-materialised config (run.dir set) resolves inside that
      run's own artifacts dir, not the shared top-level artifacts/.
  4 — resolve_class_names uses names from a real label_map.json.
  5 — a label_map.json outranks an explicit class_names list.
  6 — resolve_class_names falls back to DEFAULT_CLASS_NAMES when absent.
"""

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_phase5():
    """Import ``scripts/05_evaluate.py`` (module name is not a valid identifier)."""
    spec = importlib.util.spec_from_file_location(
        "phase05_evaluate", REPO_ROOT / "scripts" / "05_evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# resolve_label_map_path
# ---------------------------------------------------------------------------

def test_explicit_label_map_wins(tmp_path) -> None:
    """An explicit --label-map path overrides the artifacts-dir default."""
    phase5 = _load_phase5()
    explicit = tmp_path / "custom_label_map.json"
    cfg = {"output": {"artifacts_dir": "artifacts"}}
    assert phase5.resolve_label_map_path(cfg, explicit) == explicit
    # A relative --label-map is resolved against the repo root, not the cwd.
    assert phase5.resolve_label_map_path(cfg, Path("runs/x/lm.json")) == (
        REPO_ROOT / "runs" / "x" / "lm.json"
    )


def test_default_resolves_to_artifacts_dir() -> None:
    """Without --label-map the run's own artifacts/label_map.json is used."""
    phase5 = _load_phase5()
    cfg = {"output": {"artifacts_dir": "artifacts"}}
    assert phase5.resolve_label_map_path(cfg, None) == (
        REPO_ROOT / "artifacts" / "label_map.json"
    )


def test_orchestrated_run_resolves_inside_run_dir(tmp_path) -> None:
    """A run_dataset.py-materialised config resolves inside runs/<id>/artifacts.

    ``load_config`` prefixes output paths with ``run.dir``; the resolved label
    map must follow that prefix so each dataset gets its own class names.
    """
    from src.utils.config import load_config

    phase5 = _load_phase5()
    overlay = tmp_path / "experiment_ds.yaml"
    overlay.write_text(
        'data:\n  csv_path: "data/ds.csv"\nrun:\n  dir: "runs/ds"\n'
    )
    cfg = load_config(overlay, default_path=REPO_ROOT / "configs" / "default.yaml")

    resolved = phase5.resolve_label_map_path(cfg, None)
    assert resolved == REPO_ROOT / "runs" / "ds" / "artifacts" / "label_map.json"


# ---------------------------------------------------------------------------
# Evaluator class-name resolution (the consumer of the resolved path)
# ---------------------------------------------------------------------------

def test_class_names_come_from_real_label_map(tmp_path) -> None:
    """A real label_map.json drives class names, not the hardcoded default."""
    from src.model.evaluator import DEFAULT_CLASS_NAMES, resolve_class_names

    lmap = tmp_path / "label_map.json"
    lmap.write_text(json.dumps({"Benign": 0, "Bruteforce": 1, "Reconnaissance": 2}))

    names = resolve_class_names(None, lmap)
    assert names == ["Benign", "Bruteforce", "Reconnaissance"]
    assert names != DEFAULT_CLASS_NAMES[: len(names)]


def test_label_map_beats_explicit_class_names(tmp_path) -> None:
    """label_map.json takes precedence over a caller-supplied name list."""
    from src.model.evaluator import resolve_class_names

    lmap = tmp_path / "label_map.json"
    lmap.write_text(json.dumps({"Benign": 0, "Attack": 1}))

    assert resolve_class_names(["wrong", "names"], lmap) == ["Benign", "Attack"]


def test_missing_label_map_falls_back_to_defaults(tmp_path) -> None:
    """A non-existent label map path falls back to DEFAULT_CLASS_NAMES."""
    from src.model.evaluator import DEFAULT_CLASS_NAMES, resolve_class_names

    assert resolve_class_names(None, tmp_path / "nope.json") == DEFAULT_CLASS_NAMES
