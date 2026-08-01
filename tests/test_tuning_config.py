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
  20-29 — ``resolve_selection_settings`` (specs/37, specs/38 §2.3): YAML
       value resolution, missing-block/missing-key/invalid-value fail-fast,
       and the shipped ``configs/tuning_grid.yaml``'s ``selection:`` block.
  30-32 — ``HyperparameterTuner._check_shard_resume_header`` covers the
       three new selection-policy header fields identically to the
       pre-existing identity fields (specs/38 §2.7) — net-new coverage,
       this static method had no test anywhere before this fix.
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
    resolve_selection_settings,
    resolve_trial_settings,
)
from src.utils.config import load_config  # noqa: E402

GRID_PATH = REPO_ROOT / "configs" / "tuning_grid.yaml"
DEFAULT_PATH = REPO_ROOT / "configs" / "default.yaml"
TUNE_SCRIPT = REPO_ROOT / "scripts" / "03_tune.py"

# Minimal grid; shape only, never asserted on.
_MINIMAL_SEARCH_SPACE: dict = {"hidden_size": [64], "dropout": [0.1]}


def _write_grid(tmp_path: Path, trial: dict | None,
                search_space: dict | None = None,
                selection: dict | None = None) -> Path:
    """Write a throwaway ``tuning_grid.yaml`` and return its path.

    Args:
        tmp_path: pytest ``tmp_path`` fixture directory.
        trial: value for the ``trial:`` block; ``None`` omits the block entirely.
        search_space: value for ``search_space:``; defaults to a minimal grid.
        selection: value for ``selection:``; ``None`` omits the block entirely
            (specs/38 §8.3's ``resolve_selection_settings`` tests).

    Returns:
        Path to the written YAML file.
    """
    doc: dict = {"search_space": search_space or _MINIMAL_SEARCH_SPACE}
    if trial is not None:
        doc["trial"] = trial
    if selection is not None:
        doc["selection"] = selection
    path = tmp_path / "tuning_grid.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def _load_grid(tmp_path: Path, trial: dict | None,
               search_space: dict | None = None,
               selection: dict | None = None) -> dict:
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
        selection: value for ``selection:``; ``None`` omits the block entirely.

    Returns:
        The merged config dict.
    """
    return load_config(_write_grid(tmp_path, trial, search_space, selection),
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
        tie_band_pp=0.0, tie_break_axes=[],
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
        tie_band_pp=0.0, tie_break_axes=[],
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
            tie_band_pp=0.0, tie_break_axes=[],
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
        tie_band_pp=0.0, tie_break_axes=[],
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
    # ...and to the best_params.json payload. (Pin updated for the specs/38
    # selection-noise-fix build: selection now goes through
    # src.model.selection.select_best_tie_aware, which returns a dict with
    # a "winner" record, and the payload reads the field off that record.
    # Same property, new source.)
    assert '"selection_metric_used": result["winner"].get("selection_metric_used")' in run_src
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


# ---------------------------------------------------------------------------
# 20-29 — resolve_selection_settings (specs/37, specs/38 §2.3)
# ---------------------------------------------------------------------------

def test_resolve_selection_settings_returns_yaml_value(tmp_path: Path) -> None:
    """T20 — the resolver returns the fixture's OWN values, not a default."""
    grid_cfg = _load_grid(
        tmp_path, {"max_epochs": 7, "patience": 3},
        selection={
            "tie_band_pp": 3.5,
            "tie_break_axes": [{"axis": "hidden_size", "cheapest": "first"}],
        },
    )
    assert resolve_selection_settings(grid_cfg) == {
        "tie_band_pp": 3.5,
        "tie_break_axes": [{"axis": "hidden_size", "cheapest": "first"}],
    }


def test_missing_selection_block_raises(tmp_path: Path) -> None:
    """T21 — no ``selection:`` block at all is a hard error naming the file."""
    grid_cfg = _load_grid(tmp_path, {"max_epochs": 7, "patience": 3})
    with pytest.raises(KeyError) as excinfo:
        resolve_selection_settings(grid_cfg)
    message = excinfo.value.args[0]
    assert "configs/tuning_grid.yaml" in message
    assert "'selection:'" in message


def test_missing_tie_band_pp_key_raises(tmp_path: Path) -> None:
    """T22 — ``selection.tie_band_pp`` absent is a hard error naming the key."""
    grid_cfg = _load_grid(
        tmp_path, {"max_epochs": 7, "patience": 3},
        selection={"tie_break_axes": []},
    )
    with pytest.raises(KeyError) as excinfo:
        resolve_selection_settings(grid_cfg)
    message = excinfo.value.args[0]
    assert "selection.tie_band_pp" in message


def test_missing_tie_break_axes_key_raises(tmp_path: Path) -> None:
    """T23 — ``selection.tie_break_axes`` absent is a hard error naming the key."""
    grid_cfg = _load_grid(
        tmp_path, {"max_epochs": 7, "patience": 3},
        selection={"tie_band_pp": 2.0},
    )
    with pytest.raises(KeyError) as excinfo:
        resolve_selection_settings(grid_cfg)
    message = excinfo.value.args[0]
    assert "selection.tie_break_axes" in message


def test_resolve_selection_settings_rejects_unknown_axis_against_search_space(
    tmp_path: Path,
) -> None:
    """T24 — an axis name absent from the live ``search_space`` is rejected,
    naming the axis."""
    grid_cfg = _load_grid(
        tmp_path, {"max_epochs": 7, "patience": 3},
        selection={
            "tie_band_pp": 2.0,
            "tie_break_axes": [{"axis": "nonexistent_axis", "cheapest": "first"}],
        },
    )
    with pytest.raises(ValueError) as excinfo:
        resolve_selection_settings(grid_cfg)
    assert "nonexistent_axis" in str(excinfo.value)


def test_resolve_selection_settings_rejects_invalid_cheapest_value(
    tmp_path: Path,
) -> None:
    """T25 — a ``cheapest`` value other than ``first``/``last`` is rejected."""
    grid_cfg = _load_grid(
        tmp_path, {"max_epochs": 7, "patience": 3},
        selection={
            "tie_band_pp": 2.0,
            "tie_break_axes": [{"axis": "hidden_size", "cheapest": "middle"}],
        },
    )
    with pytest.raises(ValueError) as excinfo:
        resolve_selection_settings(grid_cfg)
    assert "middle" in str(excinfo.value)


def test_resolve_selection_settings_rejects_negative_tie_band_pp(
    tmp_path: Path,
) -> None:
    """T26 — a negative ``tie_band_pp`` is rejected."""
    grid_cfg = _load_grid(
        tmp_path, {"max_epochs": 7, "patience": 3},
        selection={"tie_band_pp": -1.0, "tie_break_axes": []},
    )
    with pytest.raises(ValueError) as excinfo:
        resolve_selection_settings(grid_cfg)
    assert "must be >= 0" in str(excinfo.value)


def test_resolve_selection_settings_rejects_non_numeric_tie_band_pp(
    tmp_path: Path,
) -> None:
    """T27 — an empty ``tie_band_pp:`` value (parses to YAML ``null``/``None``)
    must raise ``ValueError``, never a bare ``TypeError`` from an unguarded
    ``float(None)``."""
    grid_cfg = _load_grid(
        tmp_path, {"max_epochs": 7, "patience": 3},
        selection={"tie_band_pp": None, "tie_break_axes": []},
    )
    with pytest.raises(ValueError) as excinfo:
        resolve_selection_settings(grid_cfg)
    assert "must be a number" in str(excinfo.value)


def test_resolve_selection_settings_rejects_non_list_tie_break_axes(
    tmp_path: Path,
) -> None:
    """T28 — an empty ``tie_break_axes:`` value (parses to ``None``) must raise
    ``ValueError`` (via ``validate_tie_break_axes``'s ``isinstance`` guard),
    never a bare ``TypeError`` from ``for entry in None``."""
    grid_cfg = _load_grid(
        tmp_path, {"max_epochs": 7, "patience": 3},
        selection={"tie_band_pp": 2.0, "tie_break_axes": None},
    )
    with pytest.raises(ValueError) as excinfo:
        resolve_selection_settings(grid_cfg)
    assert "must be a list" in str(excinfo.value)


def test_run_writes_the_same_11_key_payload_as_promote() -> None:
    """specs/38 §6.1's claim that best_params.json is "identical shape from
    both call sites" is only half-proved by P11 (tests/test_promote_best.py,
    which proves promote()'s behavior) and T18 (which pins exactly one
    key's substring) unless HyperparameterTuner.run's OWN write path is
    also pinned. Source-level assertion (specs/33 §IV.3's sanctioned
    fallback: run() needs a real graph/FeatureStore/Trainer, unavailable in
    the artifact-free gate) that run() uses select_best_tie_aware (never
    the legacy select_best) and writes all 11 best_params.json keys.

    `"select_best("` is deliberately checked as NOT a substring of the
    source: it is not a substring of `"select_best_tie_aware("` either (the
    next characters are `_tie_aware(`), so this negative assertion has
    real teeth against a partial revert that reintroduces the old
    argmax-only call alongside the new one.
    """
    run_src = inspect.getsource(HyperparameterTuner.run)
    assert "select_best_tie_aware(" in run_src
    assert "select_best(" not in run_src
    for key in (
        "best_params", "best_val_macro_f1", "selection_metric_used",
        "best_trial", "argmax_trial", "argmax_val_macro_f1",
        "tie_band_pp", "tie_break_axes", "tie_set_trials",
        "tie_set_size", "selection_method",
    ):
        assert f'"{key}":' in run_src
    assert '"selection_method":      "tie_band_axis_priority"' in run_src


def test_shipped_tuning_grid_selection_resolves() -> None:
    """T29 — the real, shipped ``configs/tuning_grid.yaml`` parses via
    ``resolve_selection_settings``, and its default ``tie_break_axes``
    matches specs/38 §5's documented ordering verbatim."""
    grid_cfg = load_config(GRID_PATH, default_path=DEFAULT_PATH)
    result = resolve_selection_settings(grid_cfg)
    assert result["tie_band_pp"] == 2.0
    assert result["tie_break_axes"] == [
        {"axis": "batch_size", "cheapest": "last"},
        {"axis": "fanouts", "cheapest": "first"},
        {"axis": "hidden_size", "cheapest": "first"},
    ]


# ---------------------------------------------------------------------------
# 30-32 — HyperparameterTuner._check_shard_resume_header covers the three
# new selection-policy fields (specs/38 §2.7). This static method had NO
# existing test anywhere in the suite before this fix (verified: `grep -rln
# "_check_shard_resume_header" tests/*.py` matched nothing pre-fix) — net-new
# coverage, not an extension of something that already existed.
# ---------------------------------------------------------------------------

def _base_resume_header(**overrides) -> dict:
    """A minimal, internally-consistent header-shaped dict covering every
    field ``_check_shard_resume_header`` iterates, so a single-field
    override in ``stored`` is the only difference from ``current``."""
    base = {
        "shard_index": 0,
        "num_shards": 3,
        "n_total": 12,
        "partition_scheme": "beta",
        "shard_axes": ["beta"],
        "owned_values": {"beta": "x"},
        "grid_fingerprint": "deadbeef" * 8,
        "tie_band_pp": 2.0,
        "tie_break_axes": [{"axis": "batch_size", "cheapest": "last"}],
        "search_space": {"batch_size": [512, 1024, 2048]},
        "pbs_jobid": "1.testhost",
    }
    base.update(overrides)
    return base


def test_resume_guard_agrees_when_identical() -> None:
    """Sanity check for the fixture itself: identical stored/current headers
    (pbs_jobid may legitimately differ — that is a warning, not a failure)
    must NOT raise. Without this, a bug in ``_base_resume_header`` could
    make every T30-T32 test below pass for the wrong reason."""
    current = _base_resume_header()
    stored = _base_resume_header(pbs_jobid="999.other")
    HyperparameterTuner._check_shard_resume_header(
        stored, current, Path("/tmp/dummy_shard.json")
    )


def test_resume_guard_covers_tie_band_pp() -> None:
    """T30 — a stored shard header with a different ``tie_band_pp`` than
    the current invocation fails closed on resume, exactly like the
    pre-existing identity fields."""
    current = _base_resume_header()
    stored = _base_resume_header(tie_band_pp=5.0)
    with pytest.raises(RuntimeError) as excinfo:
        HyperparameterTuner._check_shard_resume_header(
            stored, current, Path("/tmp/dummy_shard.json")
        )
    assert "tie_band_pp" in str(excinfo.value)


def test_resume_guard_covers_tie_break_axes() -> None:
    """T31 — same, for ``tie_break_axes``."""
    current = _base_resume_header()
    stored = _base_resume_header(
        tie_break_axes=[{"axis": "fanouts", "cheapest": "first"}]
    )
    with pytest.raises(RuntimeError) as excinfo:
        HyperparameterTuner._check_shard_resume_header(
            stored, current, Path("/tmp/dummy_shard.json")
        )
    assert "tie_break_axes" in str(excinfo.value)


def test_resume_guard_covers_search_space() -> None:
    """T32 — same, for ``search_space``."""
    current = _base_resume_header()
    stored = _base_resume_header(search_space={"batch_size": [512, 1024]})
    with pytest.raises(RuntimeError) as excinfo:
        HyperparameterTuner._check_shard_resume_header(
            stored, current, Path("/tmp/dummy_shard.json")
        )
    assert "search_space" in str(excinfo.value)
