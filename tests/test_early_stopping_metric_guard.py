"""
Tests for the Phase 3 selection-metric guard (``scripts/03_tune.py``).

Regression guard for specs/34 §0.1: Phase 3 shard jobs call
``scripts/03_tune.py`` directly and therefore bypass ``run_dataset.py``'s
bare-stub config guard entirely. Without a check inside ``03_tune.py``
itself, an experiment config that forgets to declare its own
``model.early_stopping_metric`` would silently inherit
``configs/default.yaml``'s ``"macro_f1"`` default and tune all 108 trials
against the wrong objective — invisibly, since the merged config always
has SOME value for the key (default.yaml supplies it), so checking the
merged dict can never catch this. The guard therefore reads the experiment
YAML directly, before the merge.

Tests:
  1 — an experiment config with its own model.early_stopping_metric passes.
  2 — a bare {data, run, seed} stub (no model: block at all) raises, message
      names the file.
  3 — a model: block present but missing early_stopping_metric raises.
  4 — the real shipped experiment_unsw.yaml passes (on-disk pin).
  5 — the real shipped experiment_nf_{bot_iot,cicids2018,ton_iot}_v3.yaml all
      pass (on-disk pin — specs/34 grounding fact 9).
  6 — a YAML with a null model: key (``model:`` with nothing under it) raises
      KeyError naming the file/key, not TypeError — regression guard for a
      ``raw.get("model", {})`` bug caught in independent review (YAML parses
      a bare ``model:`` to ``None``, not ``{}``).
  7 — main() actually calls the guard before any expensive I/O (graph/
      feature-store loading) — wiring pin, not just the helper in isolation.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_phase3():
    """Import ``scripts/03_tune.py`` (module name is not a valid identifier)."""
    spec = importlib.util.spec_from_file_location(
        "phase03_tune", REPO_ROOT / "scripts" / "03_tune.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_config(tmp_path: Path, doc: dict) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def test_config_with_own_metric_passes(tmp_path: Path) -> None:
    """T1 — a config declaring its own early_stopping_metric raises nothing."""
    phase3 = _load_phase3()
    config_path = _write_config(
        tmp_path,
        {"data": {}, "run": {}, "seed": 42, "model": {"early_stopping_metric": "composite"}},
    )
    phase3._assert_early_stopping_metric_declared(config_path)


def test_bare_stub_raises(tmp_path: Path) -> None:
    """T2 — a {data, run, seed} stub with no model: block raises, names the file."""
    phase3 = _load_phase3()
    config_path = _write_config(tmp_path, {"data": {}, "run": {}, "seed": 42})
    with pytest.raises(KeyError) as excinfo:
        phase3._assert_early_stopping_metric_declared(config_path)
    message = excinfo.value.args[0]
    assert str(config_path) in message
    assert "early_stopping_metric" in message


def test_model_block_without_metric_raises(tmp_path: Path) -> None:
    """T3 — a model: block missing early_stopping_metric still raises."""
    phase3 = _load_phase3()
    config_path = _write_config(
        tmp_path,
        {"data": {}, "run": {}, "seed": 42, "model": {"num_layers": 2}},
    )
    with pytest.raises(KeyError) as excinfo:
        phase3._assert_early_stopping_metric_declared(config_path)
    message = excinfo.value.args[0]
    assert str(config_path) in message
    assert "early_stopping_metric" in message


def test_real_unsw_config_passes() -> None:
    """T4 — on-disk pin: the shipped, tuned UNSW config declares its own metric."""
    phase3 = _load_phase3()
    phase3._assert_early_stopping_metric_declared(
        REPO_ROOT / "configs" / "experiment_unsw.yaml"
    )


@pytest.mark.parametrize(
    "filename",
    [
        "experiment_nf_bot_iot_v3.yaml",
        "experiment_nf_cicids2018_v3.yaml",
        "experiment_nf_ton_iot_v3.yaml",
    ],
)
def test_real_paper3_configs_pass(filename: str) -> None:
    """T5 — on-disk pin: the regenerated Paper-3 starter configs each declare
    their own early_stopping_metric (specs/34 grounding fact 9)."""
    phase3 = _load_phase3()
    phase3._assert_early_stopping_metric_declared(REPO_ROOT / "configs" / filename)


def test_null_model_block_raises_keyerror_not_typeerror(tmp_path: Path) -> None:
    """T6 — ``model:`` with nothing under it (YAML null, not {}) still raises
    KeyError naming the file/key, not TypeError."""
    phase3 = _load_phase3()
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("data: {}\nrun: {}\nseed: 42\nmodel:\n", encoding="utf-8")
    with pytest.raises(KeyError) as excinfo:
        phase3._assert_early_stopping_metric_declared(config_path)
    message = excinfo.value.args[0]
    assert str(config_path) in message
    assert "early_stopping_metric" in message


def test_main_calls_guard_before_expensive_io(tmp_path: Path, monkeypatch) -> None:
    """T7 — wiring pin: main() calls the guard as its first action, before
    graph/feature-store loading, not just as a helper nobody invokes."""
    phase3 = _load_phase3()
    config_path = _write_config(tmp_path, {"data": {}, "run": {}, "seed": 42})
    calls: list[str] = []
    monkeypatch.setattr(
        phase3,
        "_assert_early_stopping_metric_declared",
        lambda p: calls.append(str(p)),
    )
    # cfg deliberately has no "compute" key: if main() reached past the guard
    # it would KeyError here, before ever touching dgl/graphs — proving the
    # guard runs first without needing real feature_store/graphs fixtures.
    with pytest.raises(KeyError):
        phase3.main({}, config_path)
    assert calls == [str(config_path)]
