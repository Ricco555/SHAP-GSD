"""
Tests for Phase-3 tuning-config resolution (``src/model/tuner.py``,
``scripts/03_tune.py``, ``configs/tuning_grid.yaml``).

Regression guard for the defect where ``configs/tuning_grid.yaml``'s ``trial:``
block was never read by any code: ``scripts/03_tune.py``'s ``__main__`` block
silently backfilled a hardcoded ``cfg.setdefault("tuning", {...})`` (20 epochs /
patience 5) that coincidentally equalled the YAML's values, so editing the YAML
had zero effect on any run. A second, dead ``selection_metric`` concept was
accepted by ``HyperparameterTuner.__init__`` and read by nothing.

All tests here are artifact-free (``tmp_path`` + ``yaml`` + ``load_config`` +
direct imports only) and run in the default ``pytest tests/ -v`` gate — no
``feature_store/``, no ``graphs/``, no checkpoint, no skip guard.

Fixture teeth: the tmp_path fixture uses ``trial: {max_epochs: 7, patience: 3}``,
values producible by NO pre-fix or wrong-source code path. Excluded on purpose:
20/5 (the deleted hardcoded ``__main__`` defaults), 50/10
(``configs/default.yaml``'s ``model.max_epochs``/``model.patience``, which
``load_config`` merges underneath the grid file), and 40/20 (the real shipped
``trial:`` values, which an implementation that ignored its argument and
re-read the on-disk YAML would produce).

Tests:
   1 — resolves 7/3 from a tmp grid file (core value assertion).
   2 — missing ``trial:`` block raises, message names the file and the block.
   3 — missing ``trial.max_epochs`` raises, message names file + key.
   4 — missing ``trial.patience`` raises, message names file + key.
   5 — no silent default: with ``model.max_epochs``/``patience`` present via the
       merged ``default.yaml`` (50/10), a missing ``trial:`` still raises.
   6 — unknown keys under ``trial:`` (stale ``selection_metric:``) are tolerated.
   7 — constructor propagation: resolver output kwargs reach the tuner.
   8 — ``search_space`` fail-fast guard present in ``03_tune.py``; old
       ``.get()``-with-eager-fallback gone.
   9 — on-disk pin: real ``tuning_grid.yaml`` has ``trial: 40 / 20``.
  10 — the real shipped file resolves to 40/20 (no missing required key).
  11 — the real ``search_space`` still enumerates exactly 108 configurations.
  12 — the ``__main__`` landmine is gone from ``scripts/03_tune.py``.
  13 — ``selection_metric=`` kwarg is rejected with ``TypeError``.
  14 — a validly-constructed tuner has no ``selection_metric`` attribute.
  15 — the shipped YAML's ``trial:`` block has no ``selection_metric`` key.
  16 — ``scripts/03_tune.py`` no longer mentions ``selection_metric`` at all.
  17 — ``SELECTION_METRIC_CURVE_KEY`` maps the three policies + fallback, and
       ``run()`` actually uses the constant.
  18 — ``run()`` records ``selection_metric_used`` in the trial result and in
       the ``best_params.json`` payload.
  19 — the resume path still subscripts only ``best_val_macro_f1``.
"""

import inspect
import itertools
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.tuner import (  # noqa: E402
    SELECTION_METRIC_CURVE_KEY,
    HyperparameterTuner,
    resolve_trial_settings,
)
from src.utils.config import load_config  # noqa: E402

GRID_PATH = REPO_ROOT / "configs" / "tuning_grid.yaml"
DEFAULT_PATH = REPO_ROOT / "configs" / "default.yaml"
TUNE_SCRIPT = REPO_ROOT / "scripts" / "03_tune.py"

# Minimal grid; shape only, never asserted on.
_MINIMAL_SEARCH_SPACE: dict = {"hidden_size": [64], "dropout": [0.1]}


def _write_grid(tmp_path: Path, trial: dict | None,
                search_space: dict | None = None) -> Path:
    """Write a throwaway ``tuning_grid.yaml`` and return its path.

    Args:
        tmp_path: pytest ``tmp_path`` fixture directory.
        trial: value for the ``trial:`` block; ``None`` omits the block entirely.
        search_space: value for ``search_space:``; defaults to a minimal grid.

    Returns:
        Path to the written YAML file.
    """
    doc: dict = {"search_space": search_space or _MINIMAL_SEARCH_SPACE}
    if trial is not None:
        doc["trial"] = trial
    path = tmp_path / "tuning_grid.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def _load_grid(tmp_path: Path, trial: dict | None,
               search_space: dict | None = None) -> dict:
    """Write and load a throwaway grid config, merged against the real defaults.

    ``default_path`` is passed explicitly: without it ``load_config`` would look
    for a ``default.yaml`` sibling of the tmp file. Merging the REAL
    ``configs/default.yaml`` is deliberate — it puts ``model.max_epochs: 50`` and
    ``model.patience: 10`` into the returned dict, so a wrong implementation that
    read the merged ``model:`` block would produce 50/10 and fail loudly.

    Args:
        tmp_path: pytest ``tmp_path`` fixture directory.
        trial: value for the ``trial:`` block; ``None`` omits the block.
        search_space: value for ``search_space:``; defaults to a minimal grid.

    Returns:
        The merged config dict.
    """
    return load_config(_write_grid(tmp_path, trial, search_space),
                       default_path=DEFAULT_PATH)


def _tune_script_source() -> str:
    """Return ``scripts/03_tune.py`` as text.

    Read as text rather than imported: the module name starts with a digit and
    its module-level ``dgl``/``FeatureStore`` imports would execute for no
    benefit (specs/33 §III.1.1).
    """
    return TUNE_SCRIPT.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1-7 — budget resolution
# ---------------------------------------------------------------------------

def test_resolve_trial_settings_returns_yaml_values(tmp_path: Path) -> None:
    """T1 — the resolver returns the fixture's OWN 7/3, not any other source.

    This is the core regression assertion: it is about a *value*, not about the
    function existing. 7/3 is unreachable from the deleted hardcoded 20/5, from
    ``default.yaml``'s merged ``model:`` 50/10, and from the shipped 40/20.
    """
    grid_cfg = _load_grid(tmp_path, {"max_epochs": 7, "patience": 3})
    assert resolve_trial_settings(grid_cfg) == {
        "max_epochs_per_trial": 7,
        "patience": 3,
    }


def test_missing_trial_block_raises(tmp_path: Path) -> None:
    """T2 — no ``trial:`` block is a hard error naming the config file."""
    grid_cfg = _load_grid(tmp_path, None)
    with pytest.raises(KeyError) as excinfo:
        resolve_trial_settings(grid_cfg)
    message = excinfo.value.args[0]
    assert "configs/tuning_grid.yaml" in message
    assert "trial:" in message
    assert "Refusing to guess" in message


def test_missing_max_epochs_raises(tmp_path: Path) -> None:
    """T3 — ``trial.max_epochs`` absent is a hard error naming the key."""
    grid_cfg = _load_grid(tmp_path, {"patience": 3})
    with pytest.raises(KeyError) as excinfo:
        resolve_trial_settings(grid_cfg)
    message = excinfo.value.args[0]
    assert "configs/tuning_grid.yaml" in message
    assert "trial.max_epochs" in message
    assert "Refusing to substitute a default" in message


def test_missing_patience_raises(tmp_path: Path) -> None:
    """T4 — ``trial.patience`` absent is a hard error naming the key."""
    grid_cfg = _load_grid(tmp_path, {"max_epochs": 7})
    with pytest.raises(KeyError) as excinfo:
        resolve_trial_settings(grid_cfg)
    message = excinfo.value.args[0]
    assert "configs/tuning_grid.yaml" in message
    assert "trial.patience" in message


def test_no_silent_fallback_to_merged_model_block(tmp_path: Path) -> None:
    """T5 — a plausible-but-wrong fallback source is present, and still unused.

    The loaded config genuinely contains ``model.max_epochs == 50`` and
    ``model.patience == 10`` (merged from the real ``configs/default.yaml``), so
    an implementation with ``.get(..., cfg["model"]["max_epochs"])`` — or with
    any hardcoded 20/5 default — would return a value instead of raising. This
    pins that it raises.
    """
    grid_cfg = _load_grid(tmp_path, None)
    assert grid_cfg["model"]["max_epochs"] == 50
    assert grid_cfg["model"]["patience"] == 10
    with pytest.raises(KeyError):
        resolve_trial_settings(grid_cfg)

    # And a partially-specified block must not backfill the other key either.
    for partial in ({"max_epochs": 7}, {"patience": 3}):
        with pytest.raises(KeyError):
            resolve_trial_settings(_load_grid(tmp_path, partial))


def test_unknown_trial_keys_are_tolerated(tmp_path: Path) -> None:
    """T6 — extra keys under ``trial:`` are ignored, never rejected.

    The un-synced ``../SHAP-GSD-hpc`` mirror still carries the deleted
    ``selection_metric:`` key (specs/32 §6); a strict schema check would turn a
    config-sync convenience into a hard Phase-3 failure on the cluster.
    """
    grid_cfg = _load_grid(tmp_path, {
        "max_epochs": 7,
        "patience": 3,
        "selection_metric": "val_macro_f1",
        "future_key": 1,
    })
    assert resolve_trial_settings(grid_cfg) == {
        "max_epochs_per_trial": 7,
        "patience": 3,
    }


def test_resolved_settings_reach_the_tuner(tmp_path: Path) -> None:
    """T7 — the resolver's return keys ARE the constructor's parameter names.

    ``scripts/03_tune.py`` splats the resolver's output (``**trial_settings``),
    so the two form one contract; this pins it.
    """
    grid_cfg = _load_grid(tmp_path, {"max_epochs": 7, "patience": 3})
    tuner = HyperparameterTuner(
        search_space=_MINIMAL_SEARCH_SPACE,
        fixed_params={},
        **resolve_trial_settings(grid_cfg),
    )
    assert tuner.max_epochs_per_trial == 7
    assert tuner.patience == 3


def test_search_space_is_fail_fast_in_tune_script() -> None:
    """T8 — ``search_space`` resolution fails fast; the eager fallback is gone.

    ``main()`` cannot be called in an artifact-free gate (it needs graphs, a
    FeatureStore and a NodeStateManager), so the guard is pinned by source text:
    the old ``grid_cfg.get("search_space", cfg["tuning"][...])`` had its second
    argument evaluated eagerly and would ``KeyError`` on every call once the
    ``__main__`` landmine was removed.
    """
    source = _tune_script_source()
    assert 'grid_cfg.get("search_space"' not in source
    assert 'if "search_space" not in grid_cfg:' in source
    assert (
        "configs/tuning_grid.yaml is missing the required 'search_space:' "
    ) in source
    assert 'tuning_ss  = grid_cfg["search_space"]' in source


# ---------------------------------------------------------------------------
# 9-12 — the shipped config and the landmine
# ---------------------------------------------------------------------------

def test_shipped_trial_budget_is_40_20() -> None:
    """T9 — the real ``configs/tuning_grid.yaml`` pins ``max_epochs``/``patience``.

    Guards against an accidental revert to the old 20/5 budget. If this budget
    is DELIBERATELY changed, update this test AND ``specs/33`` §II.2's walltime
    arithmetic AND ``README.md``'s Phase 3 section in the same change.
    """
    grid = yaml.safe_load(GRID_PATH.read_text(encoding="utf-8"))
    assert grid["trial"]["max_epochs"] == 40
    assert grid["trial"]["patience"] == 20


def test_shipped_grid_file_resolves() -> None:
    """T10 — the shipped file is not missing a required key and yields 40/20."""
    grid_cfg = load_config(GRID_PATH, default_path=DEFAULT_PATH)
    assert resolve_trial_settings(grid_cfg) == {
        "max_epochs_per_trial": 40,
        "patience": 20,
    }


def test_shipped_search_space_still_yields_108_configs() -> None:
    """T11 — the grid stays frozen at 3x3x4x3 = 108 configurations.

    Exercises the real enumeration path (``HyperparameterTuner._configs``),
    which is artifact-free, rather than recomputing the cross-product test-side.
    """
    grid_cfg = load_config(GRID_PATH, default_path=DEFAULT_PATH)
    tuner = HyperparameterTuner(
        search_space=grid_cfg["search_space"],
        fixed_params=grid_cfg.get("fixed", {}),
        **resolve_trial_settings(grid_cfg),
    )
    assert len(tuner._configs()) == 108
    # Every combination is distinct — no silent collapse of a grid axis.
    axis_lengths = [len(v) for v in grid_cfg["search_space"].values()]
    assert sorted(axis_lengths) == [3, 3, 3, 4]
    assert len(list(itertools.product(*grid_cfg["search_space"].values()))) == 108


def test_tuning_landmine_is_gone_from_tune_script() -> None:
    """T12 — no ``cfg["tuning"]`` backfill survives in ``scripts/03_tune.py``.

    A text scan is a weak pin (stage 5 must confirm by reading the diff), but it
    is the only cheap guard against reintroducing a code path with no runtime
    hook: every other test in this module would pass in the half-fixed state
    where ``main()`` is fixed but the ``__main__`` ``setdefault`` survives.
    """
    source = _tune_script_source()
    assert 'setdefault("tuning"' not in source
    assert 'cfg["tuning"]' not in source
    assert '"max_epochs_per_trial": 20' not in source
    assert '"patience_per_trial"' not in source
    assert "resolve_trial_settings(grid_cfg)" in source


# ---------------------------------------------------------------------------
# 13-19 — selection_metric deletion, and what replaces it
# ---------------------------------------------------------------------------

def test_selection_metric_kwarg_is_rejected() -> None:
    """T13 — ``selection_metric=`` raises ``TypeError`` naming that parameter.

    All other required arguments are supplied, so the ``TypeError`` can only be
    about the unexpected keyword — a bare ``HyperparameterTuner(search_space={},
    fixed_params={}, selection_metric="x")`` would also raise for the two newly
    required budget parameters and would therefore have no teeth.
    """
    with pytest.raises(TypeError) as excinfo:
        HyperparameterTuner(
            search_space=_MINIMAL_SEARCH_SPACE,
            fixed_params={},
            max_epochs_per_trial=7,
            patience=3,
            selection_metric="val_macro_f1",
        )
    assert "selection_metric" in str(excinfo.value)


def test_tuner_has_no_selection_metric_attribute() -> None:
    """T14 — the dead attribute is gone from a validly-constructed tuner."""
    tuner = HyperparameterTuner(
        search_space=_MINIMAL_SEARCH_SPACE,
        fixed_params={},
        max_epochs_per_trial=7,
        patience=3,
    )
    assert not hasattr(tuner, "selection_metric")


def test_shipped_yaml_has_no_selection_metric_key() -> None:
    """T15 — ``trial.selection_metric`` is deleted from the shipped config."""
    grid = yaml.safe_load(GRID_PATH.read_text(encoding="utf-8"))
    assert "selection_metric" not in grid["trial"]


def test_tune_script_never_mentions_selection_metric() -> None:
    """T16 — ``scripts/03_tune.py`` no longer passes or names the dead knob."""
    assert "selection_metric" not in _tune_script_source()


def test_selection_metric_curve_key_mapping() -> None:
    """T17 — concept (b) survives the deletion, correctly sourced.

    Imports the module-level constant rather than replicating the dict literal:
    asserting a test-side copy against a source-side copy would have no teeth.
    The final assertion pins that ``run()`` actually consumes the constant, so
    it cannot drift away from the live code path.
    """
    assert SELECTION_METRIC_CURVE_KEY == {
        "composite": "val_composite_f1",
        "minority_macro_f1": "val_minority_macro_f1",
        "macro_f1": "val_macro_f1",
    }
    # The call-site fallback for an unrecognised early_stopping_metric policy.
    assert SELECTION_METRIC_CURVE_KEY.get("accuracy", "val_macro_f1") == "val_macro_f1"

    run_src = inspect.getsource(HyperparameterTuner.run)
    assert 'SELECTION_METRIC_CURVE_KEY.get(stopping_metric, "val_macro_f1")' in run_src


def test_run_records_selection_metric_used() -> None:
    """T18 — trial results and ``best_params.json`` carry the metric actually used.

    Asserted on ``run()``'s source rather than by executing it: ``run()`` needs a
    DGL graph, a FeatureStore, a NodeStateManager, a Trainer and a model, none of
    which exist in the artifact-free gate (specs/33 §IV.3 sanctions this
    fallback). The VALUE side of the field — which curve key each policy maps to
    — is covered for real by ``test_selection_metric_curve_key_mapping``.
    """
    run_src = inspect.getsource(HyperparameterTuner.run)
    # Added to each trial result dict...
    assert '"selection_metric_used": metric_key,' in run_src
    # ...and to the best_params.json payload. (Pin updated for the specs/35
    # sharding build: best_idx ceased to exist — selection now goes through
    # src.model.selection.select_best, which returns the winning RECORD, and
    # the payload reads the field off that record. Same property, new source.)
    assert '"selection_metric_used": best.get("selection_metric_used")' in run_src
    # The resume-compatible field name is retained alongside it.
    assert '"best_val_macro_f1": best_val_f1,' in run_src


def test_resume_path_only_reads_best_val_macro_f1() -> None:
    """T19 — a legacy ``tuning_results.json`` still resumes.

    Entries written before this build have ``best_val_macro_f1`` but no
    ``selection_metric_used``; the resume branch must not subscript the new
    field. Safe as a source assertion because the new field is only ever
    written as a dict-literal key, never read back via ``result[...]``.
    """
    run_src = inspect.getsource(HyperparameterTuner.run)
    assert 'result["best_val_macro_f1"]' in run_src
    assert 'result["selection_metric_used"]' not in run_src

    # A legacy entry is fully consumable by the resume expression.
    legacy_entry: dict = {"trial": 0, "params": {}, "best_val_macro_f1": 0.9,
                          "best_epoch": 3, "elapsed_s": 1.0}
    assert legacy_entry["best_val_macro_f1"] == 0.9
    assert "selection_metric_used" not in legacy_entry
