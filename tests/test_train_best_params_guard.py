"""
Tests for scripts/04_train.py's shard-residue guard on best_params.json.

Regression guard for a code-review finding: sharded Phase 3 (specs/34,
specs/35) never writes best_params.json itself -- scripts/promote_best.py
does, only after ALL shards finish. Before this fix, a missing
best_params.json was ALWAYS a soft WARNING and 04_train.py silently trained
on default hyperparameters -- exactly the state shard mode leaves on disk
until promote_best.py is run, discarding the entire point of the 108-trial
grid search with nothing but a WARNING in the log. A directory holding
tuning_results_shard_*.json files but no best_params.json is a forgotten
promote step, not "no tuning happened yet", and must hard-fail. A directory
with no shard residue at all (fresh run / ablations / tests) is the genuine
no-tuning-yet case and must keep the pre-existing soft-warning behavior.

Both tests exercise the guard directly via main() and rely on it firing
BEFORE any graph/feature-store I/O -- no artifacts fixtures are needed.
"""

import importlib.util
import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_phase4():
    """Import ``scripts/04_train.py`` (module name is not a valid identifier)."""
    spec = importlib.util.spec_from_file_location(
        "phase04_train", REPO_ROOT / "scripts" / "04_train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shard_residue_without_promote_raises(tmp_path: Path) -> None:
    """A tuning dir holding sharded Phase-3 result files but no
    best_params.json must hard-fail, naming the missing file, the shard
    file(s) found, and scripts/promote_best.py -- raised before any
    graph/feature-store I/O."""
    phase4 = _load_phase4()
    tuning_dir = tmp_path / "tuning"
    tuning_dir.mkdir()
    (tuning_dir / "tuning_results_shard_fan15-10_hid64.json").write_text("{}")
    (tuning_dir / "tuning_results_shard_fan15-10_hid128.json").write_text("{}")

    with pytest.raises(RuntimeError) as excinfo:
        phase4.main(
            {"compute": {"device": "cpu"}}, tuning_dir / "best_params.json"
        )
    message = str(excinfo.value)
    assert "best_params.json" in message
    assert "promote_best.py" in message
    assert "tuning_results_shard_fan15-10_hid64.json" in message


def test_no_shard_residue_only_warns(tmp_path: Path, caplog) -> None:
    """No shard files at all (fresh run / ablations / tests) is the
    genuine no-tuning-yet case: the guard must NOT raise, only warn, and
    execution must fall through to the next step unchanged (proven here by
    a deterministic KeyError from the next step reading cfg["graph"],
    which a minimal cfg fixture deliberately omits)."""
    phase4 = _load_phase4()
    tuning_dir = tmp_path / "tuning"
    tuning_dir.mkdir()

    with caplog.at_level(logging.WARNING):
        with pytest.raises(KeyError):
            phase4.main(
                {"compute": {"device": "cpu"}}, tuning_dir / "best_params.json"
            )
    assert any(
        "best_params.json not found" in r.getMessage() for r in caplog.records
    )
