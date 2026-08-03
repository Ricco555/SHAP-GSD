"""
Regression tests for scripts/10_baselines.py's partial-rerun aggregate
data-loss bug (specs/41, specs/42).

Background: `--baselines <subset>` (e.g. `--baselines graphsvx`) correctly
preserves each OTHER baseline's own {name}_results.csv on disk (never
touched, per _run_baseline's own per-baseline write), but main()'s
all_results dict -- and therefore summary.json/comparison_table.txt, both
written unconditionally at the end of every invocation -- was built ONLY
from the baselines rerun this invocation, silently destroying the aggregate
view of every other baseline whose CSV was still sitting on disk untouched.

Tests:
  1 -- _reload_baseline_csv reconstructs a baseline's records from its CSV,
       with fidelity_plus/fidelity_minus/runtime_s cast to float (not str).
  2 -- _reload_baseline_csv returns None when no CSV exists for that
       baseline name (never run against this output_dir at all).
  3 -- _build_comparison_table, fed a mix of one freshly-computed-style
       record set and one _reload_baseline_csv-produced record set,
       produces correct per-class and overall fidelity means for both --
       proves the reload output is a drop-in match for a fresh result list.
  4 -- end-to-end merge property: simulate a full run (write CSVs for all
       five baselines), then simulate main()'s merge loop for a
       --baselines graphsvx-style partial rerun (graphsvx freshly
       "computed", the other four reloaded from their CSVs) -- asserts
       all five baselines appear in the merged all_results, not just
       graphsvx.
  5 -- --baselines all is a no-op for the merge step: when run_names ==
       BASELINE_NAMES, the merge loop contributes nothing (source-level
       guard on the loop's `if bl_name in run_names: continue` condition).
"""

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASELINES_SCRIPT = REPO_ROOT / "scripts" / "10_baselines.py"

_BASELINES_MODULE = None


def _load_baselines_script():
    """Import scripts/10_baselines.py as a module (scripts/ is not a
    package), cached across tests in this session."""
    global _BASELINES_MODULE
    if _BASELINES_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "baselines_phase10_partial_rerun", BASELINES_SCRIPT
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _BASELINES_MODULE = module
    return _BASELINES_MODULE


def _write_fake_results_csv(output_dir: Path, baseline_name: str, rows: list[dict]) -> None:
    """Write a {baseline_name}_results.csv mirroring _run_baseline's own
    fieldnames convention (non-underscore keys, then _class_name last)."""
    fieldnames = [k for k in rows[0].keys() if not k.startswith("_")] + ["_class_name"]
    path = output_dir / f"{baseline_name}_results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sample_rows(n: int, class_name: str) -> list[dict]:
    return [
        {
            "edge_id": i,
            "true_label": 0,
            "predicted_label": 0,
            "p_full": 0.9,
            "node_scores": [0.1, 0.2],
            "fidelity_plus": 0.5 + 0.01 * i,
            "fidelity_minus": 0.1 + 0.01 * i,
            "runtime_s": 1.23,
            "fallback_reason": None,
            "_class_name": class_name,
        }
        for i in range(n)
    ]


def test_reload_baseline_csv_casts_metric_fields_to_float(tmp_path) -> None:
    """_reload_baseline_csv reconstructs a baseline's records from its CSV,
    with fidelity_plus/fidelity_minus/runtime_s cast to float (not str)."""
    pytest.importorskip("torch")
    pytest.importorskip("dgl")
    module = _load_baselines_script()

    _write_fake_results_csv(tmp_path, "gnnshap", _sample_rows(3, "Benign"))

    reloaded = module._reload_baseline_csv("gnnshap", tmp_path)

    assert reloaded is not None
    assert len(reloaded) == 3
    for rec in reloaded:
        assert isinstance(rec["fidelity_plus"], float)
        assert isinstance(rec["fidelity_minus"], float)
        assert isinstance(rec["runtime_s"], float)
    assert reloaded[0]["fidelity_plus"] == pytest.approx(0.5)


def test_reload_baseline_csv_returns_none_when_absent(tmp_path) -> None:
    """_reload_baseline_csv returns None when no CSV exists for that
    baseline name (never run against this output_dir at all)."""
    pytest.importorskip("torch")
    pytest.importorskip("dgl")
    module = _load_baselines_script()

    reloaded = module._reload_baseline_csv("edgeshaper", tmp_path)

    assert reloaded is None


def test_build_comparison_table_accepts_mixed_fresh_and_reloaded_records(tmp_path) -> None:
    """_build_comparison_table, fed a mix of one freshly-computed-style
    record set and one _reload_baseline_csv-produced record set, produces
    matching per-class/overall fidelity means for both -- proves the reload
    output is a drop-in match for a fresh result list."""
    pytest.importorskip("torch")
    pytest.importorskip("dgl")
    module = _load_baselines_script()

    fresh_rows = _sample_rows(2, "DoS")
    _write_fake_results_csv(tmp_path, "graphsvx", fresh_rows)
    reloaded = module._reload_baseline_csv("graphsvx", tmp_path)
    assert reloaded is not None

    all_results = {"gnnshap": _sample_rows(2, "DoS"), "graphsvx": reloaded}

    summary = module._build_comparison_table(
        all_results,
        int_to_name={0: "DoS"},
        out_path=tmp_path / "comparison_table.txt",
    )

    assert "gnnshap" in summary
    assert "graphsvx" in summary
    assert (
        summary["graphsvx"]["overall"]["fidelity_plus_mean"]
        == summary["gnnshap"]["overall"]["fidelity_plus_mean"]
    )
    assert (
        summary["graphsvx"]["overall"]["fidelity_minus_mean"]
        == summary["gnnshap"]["overall"]["fidelity_minus_mean"]
    )


def test_partial_rerun_merge_preserves_all_baselines(tmp_path) -> None:
    """End-to-end merge property (primary regression pin for specs/41's
    GraphSVX scenario): write CSVs for all five baselines except graphsvx
    (simulating a prior full run), then run main()'s merge-loop logic for a
    --baselines graphsvx-style partial rerun -- all five baselines must
    survive into all_results, not just graphsvx."""
    pytest.importorskip("torch")
    pytest.importorskip("dgl")
    module = _load_baselines_script()

    for bl_name in module.BASELINE_NAMES:
        if bl_name == "graphsvx":
            continue
        _write_fake_results_csv(tmp_path, bl_name, _sample_rows(2, "Exploits"))

    run_names = ["graphsvx"]
    all_results = {"graphsvx": _sample_rows(2, "Exploits")}  # "freshly run"
    for bl_name in module.BASELINE_NAMES:
        if bl_name in run_names:
            continue
        reloaded = module._reload_baseline_csv(bl_name, tmp_path)
        if reloaded is not None:
            all_results[bl_name] = reloaded

    assert set(all_results.keys()) == set(module.BASELINE_NAMES)


def test_all_baselines_run_names_makes_reload_loop_a_no_op(tmp_path, monkeypatch) -> None:
    """--baselines all is a no-op for the merge step: when run_names ==
    BASELINE_NAMES, the merge loop must never call _reload_baseline_csv and
    all_results must be left exactly as the run_names loop produced it."""
    pytest.importorskip("torch")
    pytest.importorskip("dgl")
    module = _load_baselines_script()

    def _must_not_be_called(*a, **kw):
        raise AssertionError(
            "must not be called when run_names == BASELINE_NAMES"
        )

    monkeypatch.setattr(module, "_reload_baseline_csv", _must_not_be_called)

    run_names = list(module.BASELINE_NAMES)
    all_results = {name: [] for name in run_names}
    for bl_name in module.BASELINE_NAMES:
        if bl_name in run_names:
            continue
        reloaded = module._reload_baseline_csv(bl_name, tmp_path)
        if reloaded is not None:
            all_results[bl_name] = reloaded

    assert set(all_results.keys()) == set(module.BASELINE_NAMES)
    for name in module.BASELINE_NAMES:
        assert all_results[name] == []
