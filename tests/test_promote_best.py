"""
Tests for the sharded-Phase-3 promote step (``scripts/promote_best.py``) —
specs/34 §4.1/§7.11, specs/35 Part VII.2 (P1-P9) plus promote-write
atomicity.

All tests are artifact-free and run in the default ``pytest tests/ -v``
gate — no skip markers. ``promote_best.py``'s import chain is stdlib-only
(via ``src.model.selection``), so nothing here needs torch/dgl.

Fixture: a builder that writes a complete synthetic 3-shard set (the SYNTH
2x3x2 grid partitioned on ``beta``, N=3, 4 trials per shard) into
``tmp_path``, headers fully populated — including a REAL
``compute_grid_fingerprint`` over a realistic ``model:`` block, so the
fingerprint-mismatch test (P5) exercises the exact R6 hole end to end: a
changed ``composite_minority_weight`` alone must make promote refuse the
merge.

Covered:
  P1 — happy path + idempotence (byte-identical re-run); exact
       best_params.json key set; merged bare-list tuning_results.json
       sorted by global trial index; no in-flight .tmp files left behind.
  P2 — missing shard file -> PromoteError naming the unclaimed shard_index.
  P3 — duplicated trial index across shards -> PromoteError naming the
       index and BOTH filenames.
  P4 — torn (truncated, unparseable) shard file -> PromoteError naming the
       file, never a bare json.JSONDecodeError.
  P5 — grid_fingerprint mismatch caused by a changed
       composite_minority_weight -> refusal to merge (the R6 guard).
  P6 — header disagreement on n_total / selection_metric_used -> named
       field + files.
  P7 — per-file incompleteness (a record missing vs the file's own
       owned_indices) -> named file + missing trial indices.
  P8 — --num-shards operator cross-check mismatch.
  P9 — tie-break through promote: equal best_val_macro_f1 across two
       shards -> best_trial is the LOWER global index.
  Atomicity — gate failures happen before any write (pre-existing outputs
       untouched); an injected os.replace failure on best_params.json
       leaves the pre-existing target intact and no .tmp behind, while the
       already-completed tuning_results.json write is a valid, complete
       artifact (the .tmp + os.replace contract of promote's own writes).

NOT covered here (other streams / discharged elsewhere per specs/35
§VII.2's note): shard-selection helpers themselves
(tests/test_shard_selection.py) and ``HyperparameterTuner.run()``'s shard
mode end to end (needs graphs/FeatureStore/Trainer — covered by the S11
source pins, the shared-function architecture, and the specs/35 §VII.3
operational dry run).
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.selection import (  # noqa: E402
    build_shard_header,
    compute_grid_fingerprint,
    resolve_shard,
)


def _load_promote():
    """Import ``scripts/promote_best.py`` (test_evaluate_label_map.py pattern)."""
    spec = importlib.util.spec_from_file_location(
        "promote_best", REPO_ROOT / "scripts" / "promote_best.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


promote_best = _load_promote()
PromoteError = promote_best.PromoteError


# ---------------------------------------------------------------------------
# Synthetic fixture — SYNTH grid partitioned on "beta" (N=3, 4 trials/shard)
# ---------------------------------------------------------------------------

# Same synthetic grid as tests/test_shard_selection.py: shape AND axis names
# disjoint from the real 3x3x4x3 grid.
SYNTH = {"alpha": [1, 2], "beta": ["x", "y", "z"], "gamma": [0.5, 0.25]}
N_SHARDS = 3
N_TOTAL = 12

# Realistic-shaped effective model: block — carries the three composite-metric
# keys (the R6 hole) plus the five fixed-overwrite keys.
MODEL_BLOCK = {
    "aggregator": "mean",
    "batch_size": 1024,
    "composite_minority_weight": 0.5,
    "dropout": 0.2,
    "early_stopping_metric": "composite",
    "fanouts": [25, 15],
    "hidden_size": 128,
    "learning_rate": 0.001,
    "max_epochs": 100,
    "minority_class_threshold": 5000,
    "node_state_dim": 15,
    "num_classes": 10,
    "num_layers": 2,
    "patience": 20,
    "snapshot_interval": 3600,
    "temporal_window_seconds": 86400,
    "weight_decay": 0.0005,
}
TRIAL_BLOCK = {"max_epochs": 40, "patience": 20}


def _default_f1(trial: int) -> float:
    """Deterministic, distinct per-trial scores: trial 11 wins (0.65)."""
    return round(0.10 + 0.05 * trial, 4)


def _record(trial: int, f1: float) -> dict:
    """One trial record with the exact per-record schema tuner.run writes."""
    return {
        "trial": trial,
        "params": {"alpha": None, "trial_marker": trial},
        "best_val_macro_f1": f1,
        "selection_metric_used": "val_composite_f1",
        "best_epoch": 7,
        "elapsed_s": 100.0 + trial,
    }


def _write_shard_set(
    tuning_dir: Path,
    f1_for=None,
    model_block_for_shard: dict | None = None,
) -> list[Path]:
    """Write a complete, consistent 3-shard file set into tuning_dir.

    Args:
        tuning_dir: target directory (created if needed).
        f1_for: optional trial -> f1 override (defaults to _default_f1).
        model_block_for_shard: optional {shard_index: model_block} override,
            used to force a genuine fingerprint divergence (P5).

    Returns:
        The shard file paths, in shard-index order.
    """
    tuning_dir.mkdir(parents=True, exist_ok=True)
    f1_for = f1_for or _default_f1
    paths = []
    for k in range(N_SHARDS):
        spec = resolve_shard(SYNTH, "beta", f"{k}/{N_SHARDS}")
        block = (model_block_for_shard or {}).get(k, MODEL_BLOCK)
        # spec + a "grid_fingerprint" key is exactly the shape
        # HyperparameterTuner.run() passes to build_shard_header (specs/35
        # §III.5) -- calling the SAME helper here (rather than hand-listing
        # the 12 header fields in this fixture) means the fixture cannot
        # silently drift from the real write path (finding #6).
        spec["grid_fingerprint"] = compute_grid_fingerprint(block, SYNTH, TRIAL_BLOCK)
        spec["tie_band_pp"] = 0.0
        spec["tie_break_axes"] = []
        spec["search_space"] = SYNTH
        header = build_shard_header(
            spec,
            "val_composite_f1",
            pbs_jobid=f"99999{k}.testhost",
            started_at="2026-08-01T09:00:00+02:00",
        )
        results = [_record(i, f1_for(i)) for i in spec["owned_indices"]]
        path = tuning_dir / spec["filename"]
        with open(path, "w") as f:
            json.dump({"header": header, "results": results}, f, indent=2)
        paths.append(path)
    return paths


def _rewrite(path: Path, mutate) -> None:
    """Load a shard file, apply mutate(data) in place, write it back."""
    with open(path) as f:
        data = json.load(f)
    mutate(data)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# P1 — happy path + idempotence + no .tmp residue
# ---------------------------------------------------------------------------

def test_p1_happy_path_outputs_and_schema(tmp_path):
    _write_shard_set(tmp_path)
    payload = promote_best.promote(tmp_path)

    # best_params.json: EXACTLY the 11-key schema (specs/38 §6.1).
    best_path = tmp_path / "best_params.json"
    with open(best_path) as f:
        on_disk = json.load(f)
    assert set(on_disk) == {
        "best_params",
        "best_val_macro_f1",
        "selection_metric_used",
        "best_trial",
        "argmax_trial",
        "argmax_val_macro_f1",
        "tie_band_pp",
        "tie_break_axes",
        "tie_set_trials",
        "tie_set_size",
        "selection_method",
    }
    assert on_disk == payload
    # Winner: trial 11 has the max f1 under _default_f1.
    assert on_disk["best_trial"] == 11
    assert on_disk["best_val_macro_f1"] == _default_f1(11)
    assert on_disk["selection_metric_used"] == "val_composite_f1"
    assert on_disk["best_params"] == _record(11, 0.0)["params"]
    assert on_disk["argmax_trial"] == 11
    assert on_disk["argmax_val_macro_f1"] == _default_f1(11)
    assert on_disk["tie_band_pp"] == 0.0
    assert on_disk["tie_break_axes"] == []
    assert on_disk["tie_set_trials"] == [11]
    assert on_disk["tie_set_size"] == 1
    assert on_disk["selection_method"] == "tie_band_axis_priority"

    # tuning_results.json: a bare list of ALL records sorted by global trial
    # index — the same shape a single-job run produces.
    with open(tmp_path / "tuning_results.json") as f:
        merged = json.load(f)
    assert isinstance(merged, list)
    assert [r["trial"] for r in merged] == list(range(N_TOTAL))

    # No in-flight temp file left behind by promote's atomic writes.
    assert list(tmp_path.glob("*.tmp")) == []


def test_p1_idempotent_byte_identical_rerun(tmp_path):
    _write_shard_set(tmp_path)
    promote_best.promote(tmp_path)
    first_results = (tmp_path / "tuning_results.json").read_bytes()
    first_best = (tmp_path / "best_params.json").read_bytes()

    promote_best.promote(tmp_path)  # re-run over the same shard files
    assert (tmp_path / "tuning_results.json").read_bytes() == first_results
    assert (tmp_path / "best_params.json").read_bytes() == first_best
    assert list(tmp_path.glob("*.tmp")) == []


def test_p1_overwrites_preexisting_single_job_results(tmp_path):
    """A stale tuning_results.json from an abandoned single-job attempt is
    overwritten by the merged shard results (specs/35 §V.5 step 2)."""
    _write_shard_set(tmp_path)
    stale = [{"trial": 0, "params": {}, "best_val_macro_f1": 0.99}]
    with open(tmp_path / "tuning_results.json", "w") as f:
        json.dump(stale, f)
    promote_best.promote(tmp_path)
    with open(tmp_path / "tuning_results.json") as f:
        merged = json.load(f)
    assert len(merged) == N_TOTAL
    assert merged != stale


# ---------------------------------------------------------------------------
# P2 — missing shard file
# ---------------------------------------------------------------------------

def test_p2_missing_shard_file(tmp_path):
    paths = _write_shard_set(tmp_path)
    # Shard 1 owns beta="y"; its file is paths[1].
    paths[1].unlink()
    with pytest.raises(PromoteError) as exc:
        promote_best.promote(tmp_path)
    assert f"no shard file claims shard_index 1 of {N_SHARDS}" in str(exc.value)
    # Gate failure — nothing was promoted.
    assert not (tmp_path / "best_params.json").exists()
    assert not (tmp_path / "tuning_results.json").exists()


def test_p2_no_shard_files_at_all(tmp_path):
    with pytest.raises(PromoteError) as exc:
        promote_best.promote(tmp_path)
    assert "tuning_results_shard" in str(exc.value)


def test_p2_missing_directory(tmp_path):
    with pytest.raises(PromoteError):
        promote_best.promote(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# P3 — duplicated trial index across shard files
# ---------------------------------------------------------------------------

def test_p3_duplicated_trial_index_names_index_and_both_files(tmp_path):
    paths = _write_shard_set(tmp_path)
    # Trial 0 is owned by shard 0 (beta="x"). Inject a duplicate record for
    # trial 0 into shard 2's file (beta="z").
    _rewrite(
        paths[2],
        lambda d: d["results"].append(_record(0, 0.5)),
    )
    with pytest.raises(PromoteError) as exc:
        promote_best.promote(tmp_path)
    msg = str(exc.value)
    assert "trial index 0 appears in more than one shard file" in msg
    assert paths[0].name in msg
    assert paths[2].name in msg
    # The per-file foreign-trial check fires too (trial 0 is not in shard
    # 2's own owned_indices).
    assert "does not own" in msg


def test_p3_duplicate_within_one_file(tmp_path):
    paths = _write_shard_set(tmp_path)
    # Duplicate shard 1's own first owned trial inside its own file.
    spec1 = resolve_shard(SYNTH, "beta", f"1/{N_SHARDS}")
    dup_trial = spec1["owned_indices"][0]
    _rewrite(
        paths[1],
        lambda d: d["results"].append(_record(dup_trial, 0.2)),
    )
    with pytest.raises(PromoteError) as exc:
        promote_best.promote(tmp_path)
    msg = str(exc.value)
    assert "duplicate trial records" in msg
    assert paths[1].name in msg


# ---------------------------------------------------------------------------
# P4 — torn (truncated, unparseable) shard file
# ---------------------------------------------------------------------------

def test_p4_torn_file_is_promote_error_not_jsondecodeerror(tmp_path):
    paths = _write_shard_set(tmp_path)
    text = paths[0].read_text()
    paths[0].write_text(text[: len(text) // 2])  # truncate mid-JSON
    with pytest.raises(PromoteError) as exc:
        promote_best.promote(tmp_path)
    assert not isinstance(exc.value, json.JSONDecodeError)
    msg = str(exc.value)
    assert "torn or unparseable" in msg
    assert paths[0].name in msg


def test_p4_non_object_top_level_and_missing_keys(tmp_path):
    paths = _write_shard_set(tmp_path)
    paths[0].write_text(json.dumps([1, 2, 3]))  # a list, not an object
    _rewrite(paths[1], lambda d: d.pop("header"))
    with pytest.raises(PromoteError) as exc:
        promote_best.promote(tmp_path)
    msg = str(exc.value)
    # Both offenders reported at once, by name.
    assert paths[0].name in msg
    assert paths[1].name in msg
    assert "expected object" in msg
    assert "header" in msg


# ---------------------------------------------------------------------------
# P5 — fingerprint mismatch on a changed composite_minority_weight (R6)
# ---------------------------------------------------------------------------

def test_p5_fingerprint_mismatch_from_changed_composite_minority_weight(
    tmp_path,
):
    """The exact R6 hole end to end: one shard's fingerprint was computed
    under composite_minority_weight 0.6 instead of 0.5 — nothing else
    differs — and promote must refuse to merge."""
    drifted = dict(MODEL_BLOCK, composite_minority_weight=0.6)
    # Sanity: the single changed key really flips the fingerprint.
    assert compute_grid_fingerprint(
        drifted, SYNTH, TRIAL_BLOCK
    ) != compute_grid_fingerprint(MODEL_BLOCK, SYNTH, TRIAL_BLOCK)

    paths = _write_shard_set(tmp_path, model_block_for_shard={2: drifted})
    with pytest.raises(PromoteError) as exc:
        promote_best.promote(tmp_path)
    msg = str(exc.value)
    assert "grid_fingerprint" in msg
    assert "refusing to merge" in msg
    # The disagreeing file is named.
    assert paths[2].name in msg
    # Nothing was written.
    assert not (tmp_path / "best_params.json").exists()


# ---------------------------------------------------------------------------
# P6 — header disagreement on other agreement fields
# ---------------------------------------------------------------------------

def test_p6_header_disagreement_n_total(tmp_path):
    paths = _write_shard_set(tmp_path)
    _rewrite(paths[1], lambda d: d["header"].__setitem__("n_total", 999))
    with pytest.raises(PromoteError) as exc:
        promote_best.promote(tmp_path)
    msg = str(exc.value)
    assert "'n_total'" in msg
    assert paths[1].name in msg


def test_tie_band_pp_is_a_required_header_agreement_field(tmp_path):
    """specs/38 §8.2: two shard files disagreeing on tie_band_pp must
    refuse to merge, naming the field — the same _AGREEMENT_FIELDS
    machinery already exercised for n_total/selection_metric_used above,
    now covering the fix's own three new header fields."""
    paths = _write_shard_set(tmp_path)
    _rewrite(paths[1], lambda d: d["header"].__setitem__("tie_band_pp", 5.0))
    with pytest.raises(PromoteError) as exc:
        promote_best.promote(tmp_path)
    msg = str(exc.value)
    assert "'tie_band_pp'" in msg
    assert paths[1].name in msg


def test_tie_break_axes_is_a_required_header_agreement_field(tmp_path):
    paths = _write_shard_set(tmp_path)
    _rewrite(
        paths[1],
        lambda d: d["header"].__setitem__(
            "tie_break_axes", [{"axis": "gamma", "cheapest": "first"}]
        ),
    )
    with pytest.raises(PromoteError) as exc:
        promote_best.promote(tmp_path)
    msg = str(exc.value)
    assert "'tie_break_axes'" in msg
    assert paths[1].name in msg


def test_search_space_is_a_required_header_agreement_field(tmp_path):
    paths = _write_shard_set(tmp_path)
    modified_ss = dict(SYNTH)
    modified_ss["alpha"] = [1, 2, 3]
    _rewrite(
        paths[1],
        lambda d: d["header"].__setitem__("search_space", modified_ss),
    )
    with pytest.raises(PromoteError) as exc:
        promote_best.promote(tmp_path)
    msg = str(exc.value)
    assert "'search_space'" in msg
    assert paths[1].name in msg


def test_p6_header_disagreement_selection_metric(tmp_path):
    paths = _write_shard_set(tmp_path)
    _rewrite(
        paths[0],
        lambda d: d["header"].__setitem__(
            "selection_metric_used", "val_macro_f1"
        ),
    )
    with pytest.raises(PromoteError) as exc:
        promote_best.promote(tmp_path)
    msg = str(exc.value)
    assert "'selection_metric_used'" in msg
    assert paths[0].name in msg


# ---------------------------------------------------------------------------
# P7 — per-file incompleteness vs the file's OWN owned_indices
# ---------------------------------------------------------------------------

def test_p7_incomplete_shard_names_file_and_missing_trials(tmp_path):
    paths = _write_shard_set(tmp_path)
    spec1 = resolve_shard(SYNTH, "beta", f"1/{N_SHARDS}")
    dropped = spec1["owned_indices"][-1]
    _rewrite(paths[1], lambda d: d["results"].pop())
    with pytest.raises(PromoteError) as exc:
        promote_best.promote(tmp_path)
    msg = str(exc.value)
    assert paths[1].name in msg
    assert f"missing trials [{dropped}]" in msg
    assert "re-run that shard" in msg


# ---------------------------------------------------------------------------
# P8 — --num-shards operator cross-check
# ---------------------------------------------------------------------------

def test_p8_num_shards_cross_check(tmp_path):
    _write_shard_set(tmp_path)
    with pytest.raises(PromoteError) as exc:
        promote_best.promote(tmp_path, num_shards=5)
    msg = str(exc.value)
    assert "5" in msg and str(N_SHARDS) in msg
    assert "num_shards" in msg
    # The matching value passes.
    promote_best.promote(tmp_path, num_shards=N_SHARDS)


def test_p8_cli_exit_codes(tmp_path):
    _write_shard_set(tmp_path)
    assert promote_best.main(
        ["--tuning-dir", str(tmp_path), "--num-shards", "5"]
    ) == 1
    assert promote_best.main(
        ["--tuning-dir", str(tmp_path), "--num-shards", str(N_SHARDS)]
    ) == 0


# ---------------------------------------------------------------------------
# P10 — fail-closed when every header consistently carries a missing/non-int
# num_shards or n_total (regression: these must not silently disable the
# shard-index-cover and global-coverage checks)
# ---------------------------------------------------------------------------

def test_p10_all_headers_missing_num_shards_fails_closed(tmp_path):
    """Every shard file agreeing on a MISSING num_shards must still be a
    gate failure, not a silent pass-through that then skips the
    shard-index-cover check (validate_shards §V.4 check 1)."""
    paths = _write_shard_set(tmp_path)
    for p in paths:
        _rewrite(p, lambda d: d["header"].pop("num_shards"))
    with pytest.raises(PromoteError) as exc:
        promote_best.promote(tmp_path)
    msg = str(exc.value)
    assert "num_shards" in msg
    assert "must be present and an int" in msg
    assert not (tmp_path / "best_params.json").exists()


def test_p10_all_headers_non_int_n_total_fails_closed(tmp_path):
    """Every shard file agreeing on a non-int (string) n_total must still
    be a gate failure, not silently skip the global-coverage check."""
    paths = _write_shard_set(tmp_path)
    for p in paths:
        _rewrite(p, lambda d: d["header"].__setitem__("n_total", "12"))
    with pytest.raises(PromoteError) as exc:
        promote_best.promote(tmp_path)
    msg = str(exc.value)
    assert "n_total" in msg
    assert "must be present and an int" in msg
    assert not (tmp_path / "best_params.json").exists()


def test_p10_all_headers_bool_num_shards_fails_closed(tmp_path):
    """A bool ``num_shards`` (e.g. JSON ``true``) must be rejected too:
    ``bool`` is a subclass of ``int`` in Python, so a bare
    ``isinstance(value, int)`` check treats ``True`` as the valid int 1.
    With 3 real shard files present but ``num_shards`` corrupted to
    ``True`` (== 1), the OLD code's check 3 would run ``range(1)`` --
    verifying only that shard_index 0 exists -- and silently promote from
    2 missing shards. This must fail closed exactly like a missing or
    string value does."""
    paths = _write_shard_set(tmp_path)
    assert len(paths) == N_SHARDS == 3
    for p in paths:
        _rewrite(p, lambda d: d["header"].__setitem__("num_shards", True))
    with pytest.raises(PromoteError) as exc:
        promote_best.promote(tmp_path)
    msg = str(exc.value)
    assert "num_shards" in msg
    assert "must be present and an int" in msg
    assert not (tmp_path / "best_params.json").exists()


def test_p10_all_headers_bool_n_total_fails_closed(tmp_path):
    """A bool ``n_total`` (e.g. JSON ``true``) must be rejected too, for
    the same subclass-of-int reason as num_shards above: the OLD code's
    check 5 would run ``range(1)`` and accept a merge covering only trial
    0 out of the real 12-trial grid as "complete"."""
    paths = _write_shard_set(tmp_path)
    for p in paths:
        _rewrite(p, lambda d: d["header"].__setitem__("n_total", True))
    with pytest.raises(PromoteError) as exc:
        promote_best.promote(tmp_path)
    msg = str(exc.value)
    assert "n_total" in msg
    assert "must be present and an int" in msg
    assert not (tmp_path / "best_params.json").exists()


# ---------------------------------------------------------------------------
# P9 — tie-break through promote
# ---------------------------------------------------------------------------

def test_p9_tie_breaks_to_lower_global_index_across_shards(tmp_path):
    """Trials 3 (shard 1, beta='y') and 10 (shard 2, beta='z') carry equal,
    maximal scores — the LOWER global index must win, regardless of the
    glob order the files are read in."""

    def f1_for(trial: int) -> float:
        return 0.9 if trial in (3, 10) else _default_f1(trial) / 10.0

    _write_shard_set(tmp_path, f1_for=f1_for)
    payload = promote_best.promote(tmp_path)
    assert payload["best_trial"] == 3
    assert payload["best_val_macro_f1"] == 0.9


# ---------------------------------------------------------------------------
# P11 — tie-aware selection through promote is NOT shard-local
# (specs/37 §3.2's revision, specs/38 §8.2)
# ---------------------------------------------------------------------------

def test_p11_promote_uses_tie_aware_selection_across_shards(tmp_path):
    """The argmax trial and the tie-break-preferred, in-band trial live in
    DIFFERENT shard files — proves select_best_tie_aware's winner is not
    accidentally shard-local, and is not decided by elapsed_s (the flaw
    specs/37 §3.2 revised away from: this fixture gives every trial an
    elapsed_s that would pick the WRONG winner if elapsed_s were consulted).

    Built from enumerate_grid(SYNTH) directly (not _record()'s placeholder
    {"alpha": None, "trial_marker": trial} params) so a real per-axis
    tie-break lookup actually runs, per specs/38 §8.2's explicit guidance.
    """
    from src.model.selection import enumerate_grid

    combos = enumerate_grid(SYNTH)
    # argmax: first trial with gamma == 0.5 ("expensive" by the configured
    # tie-break below).
    argmax_trial = next(i for i, c in enumerate(combos) if c["gamma"] == 0.5)
    # tie-break-preferred: first trial with gamma == 0.25 ("cheapest": last
    # -> preferred) AND a DIFFERENT beta value, so it lives in a different
    # shard file than argmax_trial (shards partition on "beta").
    tie_trial = next(
        i for i, c in enumerate(combos)
        if c["gamma"] == 0.25 and c["beta"] != combos[argmax_trial]["beta"]
    )
    assert combos[argmax_trial]["beta"] != combos[tie_trial]["beta"]

    tie_band_pp = 2.0
    tie_break_axes = [{"axis": "gamma", "cheapest": "last"}]
    argmax_f1 = 0.90
    tie_f1 = argmax_f1 - (tie_band_pp / 100.0) / 2  # inside the band
    other_f1 = 0.10

    def f1_for(trial: int) -> float:
        if trial == argmax_trial:
            return argmax_f1
        if trial == tie_trial:
            return tie_f1
        return other_f1

    tmp_path.mkdir(parents=True, exist_ok=True)
    for k in range(N_SHARDS):
        spec = resolve_shard(SYNTH, "beta", f"{k}/{N_SHARDS}")
        spec["grid_fingerprint"] = compute_grid_fingerprint(
            MODEL_BLOCK, SYNTH, TRIAL_BLOCK
        )
        # Deliberately NOT the fixture's default 0.0/[] -- a real band and a
        # real tie-break ordering, on trials whose params carry real axis
        # values (combos[i], not _record()'s placeholder).
        spec["tie_band_pp"] = tie_band_pp
        spec["tie_break_axes"] = tie_break_axes
        spec["search_space"] = SYNTH
        header = build_shard_header(
            spec,
            "val_composite_f1",
            pbs_jobid=f"77777{k}.testhost",
            started_at="2026-08-01T09:00:00+02:00",
        )
        results = [
            {
                "trial": i,
                "params": combos[i],
                "best_val_macro_f1": f1_for(i),
                "selection_metric_used": "val_composite_f1",
                "best_epoch": 7,
                # Every candidate's elapsed_s is set so that "lowest
                # elapsed_s in the merged tie set" would pick the WRONG
                # trial (argmax_trial, not tie_trial) -- a regression here
                # fails loudly if elapsed_s is ever reconsulted.
                "elapsed_s": 1.0 if i == argmax_trial else 99999.0,
            }
            for i in spec["owned_indices"]
        ]
        path = tmp_path / spec["filename"]
        with open(path, "w") as f:
            json.dump({"header": header, "results": results}, f, indent=2)

    payload = promote_best.promote(tmp_path)
    assert payload["argmax_trial"] == argmax_trial
    assert payload["argmax_val_macro_f1"] == argmax_f1
    assert payload["best_trial"] == tie_trial
    assert payload["best_val_macro_f1"] == tie_f1
    assert payload["tie_set_size"] == 2
    assert sorted(payload["tie_set_trials"]) == sorted([argmax_trial, tie_trial])
    assert payload["selection_method"] == "tie_band_axis_priority"


# ---------------------------------------------------------------------------
# P12 — a v1-schema shard file set (missing the three selection fields)
# fails with a clear, informative PromoteError, not a bare KeyError
# ---------------------------------------------------------------------------

def test_p12_promote_raises_clear_error_on_missing_selection_header_fields(
    tmp_path,
):
    """Simulates shard files produced under SHARD_SCHEMA_VERSION < 2 (before
    this fix): every header uniformly missing tie_band_pp/tie_break_axes/
    search_space. validate_shards()'s agreement loop cannot catch a
    UNIFORMLY missing field (specs/38 §1.4's documented residual gap) —
    this is the local presence check in promote() (specs/38 §4.2) that
    closes it with a named, operator-actionable error instead."""
    paths = _write_shard_set(tmp_path)

    def _drop_selection_fields(d):
        for f in ("tie_band_pp", "tie_break_axes", "search_space"):
            d["header"].pop(f, None)

    for p in paths:
        _rewrite(p, _drop_selection_fields)

    with pytest.raises(PromoteError) as exc:
        promote_best.promote(tmp_path)
    msg = str(exc.value)
    assert "tie_band_pp" in msg
    assert "tie_break_axes" in msg
    assert "search_space" in msg
    assert "SHARD_SCHEMA_VERSION" in msg
    assert not (tmp_path / "best_params.json").exists()


# ---------------------------------------------------------------------------
# Atomicity of promote's own writes (.tmp + os.replace)
# ---------------------------------------------------------------------------

def test_gate_failure_never_touches_preexisting_outputs(tmp_path):
    """All gates run before any write: with a failing shard set, promote
    must leave pre-existing tuning_results.json / best_params.json
    byte-identical and no .tmp behind."""
    paths = _write_shard_set(tmp_path)
    sentinel_results = b'[{"sentinel": true}]'
    sentinel_best = b'{"sentinel": "best"}'
    (tmp_path / "tuning_results.json").write_bytes(sentinel_results)
    (tmp_path / "best_params.json").write_bytes(sentinel_best)
    paths[1].unlink()  # trips the shard-index-cover gate

    with pytest.raises(PromoteError):
        promote_best.promote(tmp_path)

    assert (tmp_path / "tuning_results.json").read_bytes() == sentinel_results
    assert (tmp_path / "best_params.json").read_bytes() == sentinel_best
    assert list(tmp_path.glob("*.tmp")) == []


def test_promote_write_failure_leaves_target_intact_no_tmp(
    tmp_path, monkeypatch
):
    """Inject an os.replace failure on the best_params.json write: the
    pre-existing best_params.json must be untouched (the .tmp + os.replace
    contract — a crash mid-write can never leave a truncated file at the
    final name), the .tmp is cleaned up, and the tuning_results.json write
    that already completed is a valid, complete artifact."""
    _write_shard_set(tmp_path)
    stale_best = b'{"stale": "previous run"}'
    (tmp_path / "best_params.json").write_bytes(stale_best)

    real_replace = os.replace

    def failing_replace(src, dst, *args, **kwargs):
        if str(dst).endswith("best_params.json"):
            raise OSError("injected replace failure")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        promote_best.promote(tmp_path)

    # Target never overwritten with partial content; .tmp cleaned up.
    assert (tmp_path / "best_params.json").read_bytes() == stale_best
    assert list(tmp_path.glob("*.tmp")) == []

    # The first write (tuning_results.json) completed atomically before the
    # injected failure and is a valid, complete artifact.
    with open(tmp_path / "tuning_results.json") as f:
        merged = json.load(f)
    assert [r["trial"] for r in merged] == list(range(N_TOTAL))

    # After the failure is gone, a re-run promotes cleanly.
    monkeypatch.setattr(os, "replace", real_replace)
    payload = promote_best.promote(tmp_path)
    with open(tmp_path / "best_params.json") as f:
        assert json.load(f) == payload
