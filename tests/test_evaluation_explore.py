"""
Tests for the ``explore/evaluation/`` package (specs/64 Part II §20).

Every fixture is synthetic and written under ``tmp_path``; the only test that
touches the real ``runs/`` tree is the opt-in anchor gated on
``SHAP_GSD_RUN_REAL_DATA_TESTS=1`` (and even that one writes exclusively into
``tmp_path``). No test writes into the repo's real ``configs/``, ``runs/`` or
``outputs/`` directories: every script invocation passes BOTH ``--out-dir`` and
``--runs-root``, because both default to real repo paths.

Test map (spec §20 names are the contract and are not renamed):
  20.1  test_both_metrics_json_schemas_load_identically
  20.2  test_zero_support_class_emits_explicit_row
  20.3  test_three_denominators_differ_on_absent_classes
  20.4  test_benjamini_hochberg_matches_reference_vector
  20.5  test_resolution_rejects_stub_and_variant_runs
  20.5b test_ambiguous_resolution_raises_and_names_candidates
  20.6  test_no_hardcoded_dataset_names
  20.7  test_no_bare_temporal_identifier
  20.8  test_coalition_space_stability_rows_are_nan_not_missing
  20.9  test_missing_mapping_csv_exits_zero
  20.10 test_mapping_csv_unknown_feature_group_raises
  20.11 test_real_data_anchor                      (gated, opt-in)

Additional coverage, for behaviour the two coding passes added beyond §20:
  test_classification_rows_use_coalition_space_none
  test_missing_structural_expectations_csv_reports_not_supplied
  test_node_coalition_agreement_is_nan_with_alignment_note
  test_absent_train_feature_store_yields_nan_closed_set_row
  test_out_of_split_subgraph_edge_id_is_a_hard_error
  test_empty_temporal_window_is_reported_not_dropped
  test_eval08_dataset_label_mismatch_emits_consistency_warning
  test_eval08_refuses_fewer_than_two_datasets
  test_paired_wilcoxon_refuses_too_few_pairs
  test_rank_agreement_constant_vector_is_nan_not_zero
  test_rank_correlation_scores_a_present_mapping
  test_topology_classifiers_cover_every_category

The last two exist because no dataset on disk carries an authored proxy-GT
mapping table or literature structural expectation, so eval04's two scoring
paths -- the spec's stated central contribution -- would otherwise be exercised
by neither a test nor a real run, and only their degrade-cleanly branches would
be covered.
"""

import ast
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from explore.evaluation import _discover, _stats  # noqa: E402
from explore.evaluation import eval01_long_metrics as eval01  # noqa: E402
from explore.evaluation import eval02_macro_denominators as eval02  # noqa: E402
from explore.evaluation import eval03_cross_explainer_agreement as eval03  # noqa: E402
from explore.evaluation import eval04_global_coherence as eval04  # noqa: E402
from explore.evaluation import eval05_per_class_explanations as eval05  # noqa: E402
from explore.evaluation import eval07b_stability_windows as eval07b  # noqa: E402
from explore.evaluation import eval08_cross_dataset_report as eval08  # noqa: E402
from explore.evaluation._discover import (  # noqa: E402
    DatasetResolutionError,
    available_datasets,
    outputs_dir,
    resolve_dataset,
)
from explore.evaluation._load import (  # noqa: E402
    DERIVED_CONVENTION_NOTE,
    STABILITY_SCOPE_NOTE,
    load_eval_metrics,
)

# The four artifacts _discover.REQUIRED_ARTIFACTS insists on. Referenced through
# the module constant rather than re-listed, so a change there fails loudly here
# instead of silently diverging.
_REQUIRED = _discover.REQUIRED_ARTIFACTS

_GROUPS = ["G1", "G2", "G3", "G4"]


# ---------------------------------------------------------------------------
# Synthetic fixture builders
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _touch(path: Path, text: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _make_shell_run(runs_root: Path, dir_name: str, present: tuple[str, ...]) -> Path:
    """Create a run directory carrying only the named REQUIRED_ARTIFACTS."""
    run_dir = runs_root / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for rel in present:
        _touch(run_dir / rel)
    return run_dir


def _complete_shell(runs_root: Path, dir_name: str) -> Path:
    return _make_shell_run(runs_root, dir_name, _REQUIRED)


def _metrics_payload(per_class: dict, *, coverage: bool, class_names: list[str]) -> dict:
    absent = [c for c in class_names if float(per_class.get(c, {}).get("support", 0)) == 0]
    scored = [c for c in class_names if c not in absent]
    macro_f1 = (
        float(np.mean([per_class[c]["f1"] for c in scored])) if scored else float("nan")
    )
    payload = {
        "accuracy": 0.95,
        "macro_f1": macro_f1,
        "weighted_f1": 0.96,
        "n_test_edges": int(sum(float(v.get("support", 0)) for v in per_class.values())),
        "per_class": per_class,
    }
    if coverage:
        payload.update({
            "macro_f1_convention": "averaged over classes present in the test split",
            "n_classes_total": len(class_names),
            "n_classes_present_in_test": len(scored),
            "classes_absent_from_test": absent,
        })
    return payload


def _explanation_record(
    edge_id: int,
    *,
    group_shap: list[float],
    node_ids: list[int],
    subgraph_edge_ids: list[int],
    neighbor_shap: list[float] | None = None,
    src_novelty: float = 0.0,
    dst_novelty: float = 0.0,
    degenerate: int | None = 2,
) -> dict:
    record = {
        "edge_id": edge_id,
        "feature_group_names": list(_GROUPS),
        "feature_group_shap": group_shap,
        "neighbor_edge_ids": list(subgraph_edge_ids),
        "neighbor_shap": neighbor_shap if neighbor_shap is not None else [0.01],
        "node_ids": node_ids,
        "node_shap": [0.1] * len(node_ids),
        "src_novelty_shap": src_novelty,
        "dst_novelty_shap": dst_novelty,
        "subgraph_edge_ids": list(subgraph_edge_ids),
        "subgraph_shap_weights": [0.01] * len(subgraph_edge_ids),
    }
    if degenerate is not None:
        record["n_degenerate_novelty_players"] = degenerate
    return record


def make_run(
    runs_root: Path,
    dir_name: str,
    *,
    class_names: list[str],
    per_class: dict,
    coverage: bool = True,
    summary_per_class: dict | None = None,
    extra_spaces: dict[str, dict] | None = None,
    explanations: dict[str, list[dict]] | None = None,
    stability_rows: list[dict] | None = None,
    train_labels: list[int] | None = None,
    test_edge_ids: list[int] | None = None,
    test_timestamps: list[float] | None = None,
    test_endpoints: list[tuple[str, str]] | None = None,
    baselines: dict[str, pd.DataFrame] | None = None,
    feature_groups: list[str] | None = None,
) -> Path:
    """Write one synthetic, resolution-complete run directory.

    Only the pieces a given test needs are written; everything optional that is
    left out reproduces a real on-disk absence (e.g. no train feature store).
    """
    run_dir = runs_root / dir_name
    label_map = {name: idx for idx, name in enumerate(class_names)}
    _write_json(run_dir / "artifacts" / "label_map.json", label_map)
    _write_json(
        run_dir / "artifacts" / "evaluation" / "metrics.json",
        _metrics_payload(per_class, coverage=coverage, class_names=class_names),
    )

    summary = {"top_k": 5, "per_class": summary_per_class or {}, "overall": {}}
    _write_json(run_dir / "outputs" / "metrics" / "summary.json", summary)
    for space, per_class_block in (extra_spaces or {}).items():
        _write_json(
            run_dir / "outputs" / "metrics" / f"summary_{space}.json",
            {"top_k": 5, "per_class": per_class_block, "overall": {}},
        )

    if stability_rows is not None:
        pd.DataFrame(stability_rows).to_csv(
            run_dir / "outputs" / "metrics" / "stability.csv", index=False,
        )

    explanations = explanations or {}
    rollup_rows = []
    exp_dir = run_dir / "outputs" / "explanations"
    exp_dir.mkdir(parents=True, exist_ok=True)
    for class_name, records in explanations.items():
        for record in records:
            _write_json(exp_dir / class_name / f"{record['edge_id']}.json", record)
            rollup_rows.append({
                "edge_id": record["edge_id"],
                "class_name": class_name,
                "true_label": label_map[class_name],
                "predicted_label": label_map[class_name],
                "correct": True,
            })
    pd.DataFrame(
        rollup_rows,
        columns=["edge_id", "class_name", "true_label", "predicted_label", "correct"],
    ).to_csv(exp_dir / "summary.csv", index=False)

    if train_labels is not None:
        fs_train = run_dir / "feature_store" / "train"
        fs_train.mkdir(parents=True, exist_ok=True)
        np.save(fs_train / "labels.npy", np.asarray(train_labels, dtype=np.int64))

    if test_edge_ids is not None:
        fs_test = run_dir / "feature_store" / "test"
        fs_test.mkdir(parents=True, exist_ok=True)
        eids = np.asarray(sorted(test_edge_ids), dtype=np.int64)
        np.save(fs_test / "edge_indices.npy", eids)
        if test_timestamps is not None:
            np.save(fs_test / "timestamps.npy",
                    np.asarray(test_timestamps, dtype=np.float64))
        endpoints = test_endpoints or [
            (f"10.0.0.{i % 250}", f"10.1.0.{i % 250}") for i in range(len(eids))
        ]
        pd.DataFrame({
            "src_ip": [s for s, _ in endpoints],
            "dst_ip": [d for _, d in endpoints],
        }).to_parquet(fs_test / "edges_meta.parquet")

    if feature_groups is not None:
        _write_json(
            run_dir / "artifacts" / "feature_groups.json",
            {"groups": {name: [name.lower()] for name in feature_groups}},
        )

    for name, frame in (baselines or {}).items():
        baselines_dir = run_dir / "outputs" / "baselines"
        baselines_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(baselines_dir / f"{name}{eval03.BASELINE_SUFFIX}", index=False)

    assert not _discover.missing_required_artifacts(run_dir), (
        f"synthetic run {dir_name} is not resolution-complete"
    )
    return run_dir


@pytest.fixture(autouse=True)
def _reset_discover_log_dedup():
    """Clear ``_discover``'s per-root INFO de-duplication between tests."""
    _discover._logged_roots.clear()
    yield
    _discover._logged_roots.clear()


def _full_run(tmp_path: Path) -> tuple[Path, Path, "_discover.DatasetRun"]:
    """A resolution-complete run rich enough for eval03/eval04/eval07b.

    Classes: ``Benign`` and ``Probe`` with explained flows, ``Ghost`` with zero
    test support and no explanations directory at all.
    """
    runs_root = tmp_path / "runs"
    out_dir = tmp_path / "out"
    class_names = ["Benign", "Ghost", "Probe"]
    per_class = {
        "Benign": {"precision": 0.99, "recall": 0.98, "f1": 0.985, "support": 300},
        "Ghost": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
        "Probe": {"precision": 0.6, "recall": 0.5, "f1": 0.545, "support": 120},
    }
    benign_eids = [10, 11, 12]
    probe_eids = [20, 21, 22]
    explanations = {
        "Benign": [
            _explanation_record(
                eid, group_shap=[0.5, 0.2, 0.1, 0.05],
                node_ids=[1, 2, 3, 4], subgraph_edge_ids=[10, 11],
            )
            for eid in benign_eids
        ],
        "Probe": [
            _explanation_record(
                eid, group_shap=[0.05, 0.4, 0.3, 0.1],
                node_ids=[5, 6, 7, 8], subgraph_edge_ids=[20, 21],
            )
            for eid in probe_eids
        ],
    }
    all_eids = sorted(benign_eids + probe_eids)
    class_column = ["Benign"] * 3 + ["Probe"] * 3
    baseline_frames = {}
    for name in ("pgexplainer", "gnnshap"):
        baseline_frames[name] = pd.DataFrame({
            "edge_id": all_eids,
            # Six entries against SHAP-GSD's four node_ids: the real, measured
            # length mismatch that makes node-coalition agreement uncomputable.
            "node_scores": [json.dumps([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
                            for _ in all_eids],
            "_class_name": class_column,
        })
    # One feature-group baseline, so the alignable half of eval03 is exercised
    # alongside the unalignable node half.
    baseline_frames["gnnexplainer"] = pd.DataFrame({
        "edge_id": all_eids,
        "group_scores": [json.dumps([0.4, 0.3, 0.15, 0.05]) for _ in all_eids],
        "_class_name": class_column,
    })
    make_run(
        runs_root, "nf_alpha_v3",
        class_names=class_names, per_class=per_class,
        summary_per_class={
            "Benign": {"n_flows": 3, "fidelity_plus": 0.1, "fidelity_plus_std": 0.01,
                       "fidelity_minus": 0.02, "fidelity_minus_std": 0.01,
                       "stability": 0.001},
            "Probe": {"n_flows": 3, "fidelity_plus": 0.2, "fidelity_plus_std": 0.02,
                      "fidelity_minus": 0.03, "fidelity_minus_std": 0.01,
                      "stability": 0.002},
        },
        extra_spaces={
            "temporal": {"Benign": {"n_flows": 3, "fidelity_plus": 0.0,
                                    "fidelity_minus": None}},
            "novelty": {"Benign": {"n_flows": 3, "fidelity_plus": 0.0,
                                   "fidelity_minus": 0.0}},
        },
        explanations=explanations,
        train_labels=[0, 0, 0, 1, 2, 2],
        test_edge_ids=all_eids,
        test_timestamps=[1.0, 2.0, 3.0, 100.0, 101.0, 102.0],
        baselines=baseline_frames,
        feature_groups=list(_GROUPS),
    )
    run = resolve_dataset("alpha", runs_root=runs_root)
    return runs_root, out_dir, run


# ---------------------------------------------------------------------------
# 20.1 — both metrics.json schemas load identically
# ---------------------------------------------------------------------------

def test_both_metrics_json_schemas_load_identically(tmp_path):
    # Deliberately synthetic: since the 2026-08-26 Phase-5 backfill every real
    # dataset carries the 9-key schema, so this test is the ONLY thing that
    # exercises _load.load_eval_metrics' derive-when-absent branch.
    class_names = ["Benign", "Ghost", "Probe"]
    per_class = {
        "Benign": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 100},
        "Ghost": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
        "Probe": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 50},
    }
    native_root = tmp_path / "native_runs"
    derived_root = tmp_path / "derived_runs"
    make_run(native_root, "nf_alpha_v3", class_names=class_names,
             per_class=per_class, coverage=True)
    make_run(derived_root, "nf_alpha_v3", class_names=class_names,
             per_class=per_class, coverage=False)

    native = load_eval_metrics(resolve_dataset("alpha", runs_root=native_root))
    derived = load_eval_metrics(resolve_dataset("alpha", runs_root=derived_root))

    assert native.schema_source == "native"
    assert derived.schema_source == "derived"
    assert native.n_classes_total == derived.n_classes_total == 3
    assert native.n_classes_present_in_test == derived.n_classes_present_in_test == 2
    assert native.classes_absent_from_test == derived.classes_absent_from_test == ["Ghost"]
    # The derived branch must MARK itself, not silently invent a convention.
    assert derived.macro_f1_convention == DERIVED_CONVENTION_NOTE
    assert native.macro_f1_convention != DERIVED_CONVENTION_NOTE


# ---------------------------------------------------------------------------
# 20.2 — a zero-support class emits an explicit row, never a missing one
# ---------------------------------------------------------------------------

def test_zero_support_class_emits_explicit_row(tmp_path):
    runs_root, out_dir, run = _full_run(tmp_path)

    long_frame = eval01.run([run], out_dir)
    per_class_frame = eval05.run([run], out_dir)

    # --- eval05: n_flows_aggregated == 0 literally, per spec §20.2 ----------
    ghost_rows = per_class_frame[per_class_frame["class_name"] == "Ghost"]
    assert len(ghost_rows) == 1, "the zero-flow class must not disappear"
    assert int(ghost_rows.iloc[0]["n_flows_aggregated"]) == 0
    assert bool(ghost_rows.iloc[0]["class_present_in_test"]) is False

    # --- eval01: the long table has no n_flows_aggregated column ------------
    # SPEC DIVERGENCE (reported): §20.2 names `n_flows_aggregated` for BOTH
    # outputs, but eval01's tidy schema is (metric, value); its per-class flow
    # count is the `n_flows` METRIC, sourced from summary*.json. For a class
    # with no entry there the honest value is NaN ("not measured"), not 0
    # ("measured as none"). What §20.2 actually guards — that the row EXISTS —
    # is asserted here in eval01's own schema.
    ghost_long = long_frame[long_frame["class_name"] == "Ghost"]
    assert not ghost_long.empty
    assert not ghost_long["class_present_in_test"].any()
    support = ghost_long[
        (ghost_long["coalition_space"] == eval01.CLASSIFICATION_SPACE)
        & (ghost_long["metric"] == "support")
    ]
    assert len(support) == 1 and float(support.iloc[0]["value"]) == 0.0
    for space in eval01.COALITION_SPACES:
        n_flows = ghost_long[
            (ghost_long["coalition_space"] == space)
            & (ghost_long["metric"] == "n_flows")
        ]
        assert len(n_flows) == 1, f"{space}: n_flows row for Ghost is missing"
        assert np.isnan(float(n_flows.iloc[0]["value"]))

    # Both artifacts actually landed in the tmp out-dir, not the repo's.
    assert (out_dir / eval01.LONG_CSV_NAME).is_file()
    assert (out_dir / eval05.CSV_NAME).is_file()
    assert "Ghost" in (out_dir / eval01.LONG_CSV_NAME).read_text()


def test_classification_rows_use_coalition_space_none(tmp_path):
    """The P/R/F1/support rows belong to no coalition space (code addition)."""
    runs_root, out_dir, run = _full_run(tmp_path)
    long_frame = eval01.run([run], out_dir)

    assert eval01.CLASSIFICATION_SPACE == "none"
    assert eval01.CLASSIFICATION_SPACE not in eval01.COALITION_SPACES
    classification = long_frame[
        long_frame["coalition_space"] == eval01.CLASSIFICATION_SPACE
    ]
    assert set(classification["metric"]) == {"precision", "recall", "f1", "support"}
    assert set(long_frame["coalition_space"]) == {
        eval01.CLASSIFICATION_SPACE, *eval01.COALITION_SPACES
    }
    # The tidy key stays unique with the fourth value present.
    assert not long_frame.duplicated(
        subset=["dataset", "class_name", "coalition_space", "metric"]
    ).any()


def test_absent_degenerate_novelty_field_is_nan_with_note(tmp_path):
    """Records without ``n_degenerate_novelty_players`` (BoT-IoT's real state).

    Without that field a near-zero novelty aggregate is uninterpretable — it
    cannot be told apart from a measurement artefact — so its absence must be
    stated in ``notes``, not silently averaged away.
    """
    runs_root = tmp_path / "runs"
    out_dir = tmp_path / "out"
    make_run(
        runs_root, "nf_alpha_v3",
        class_names=["Benign", "Probe"],
        per_class={
            "Benign": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 100},
            "Probe": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 50},
        },
        explanations={
            # Carries the field: the aggregate is attributable.
            "Benign": [
                _explanation_record(
                    eid, group_shap=[0.5, 0.2, 0.1, 0.05], node_ids=[1, 2],
                    subgraph_edge_ids=[eid], degenerate=2,
                )
                for eid in (10, 11)
            ],
            # Does NOT carry it: the aggregate is not attributable.
            "Probe": [
                _explanation_record(
                    eid, group_shap=[0.1, 0.4, 0.3, 0.05], node_ids=[3, 4],
                    subgraph_edge_ids=[eid], degenerate=None,
                )
                for eid in (20, 21)
            ],
        },
    )
    run = resolve_dataset("alpha", runs_root=runs_root)
    frame = eval05.run([run], out_dir)
    by_class = {row["class_name"]: row for _, row in frame.iterrows()}

    with_field = by_class["Benign"]
    assert float(with_field["mean_n_degenerate_novelty_players"]) == pytest.approx(2.0)
    assert float(with_field["frac_flows_both_endpoints_degenerate"]) == pytest.approx(1.0)
    assert "n_degenerate_novelty_players" not in str(with_field["notes"])

    without_field = by_class["Probe"]
    assert int(without_field["n_flows_aggregated"]) == 2, "the flows were still read"
    assert np.isnan(float(without_field["mean_n_degenerate_novelty_players"]))
    assert np.isnan(float(without_field["frac_flows_both_endpoints_degenerate"]))
    assert "records carry no n_degenerate_novelty_players field" in str(
        without_field["notes"]
    )
    assert "cannot be attributed to degenerate" in str(without_field["notes"])


def test_three_way_agreement_categories_and_precedence():
    """The 3-string -> 1-category function, including the two added values."""
    fan_out, fan_in = eval04.TOPOLOGY_FAN_OUT, eval04.TOPOLOGY_FAN_IN
    other = eval04.TOPOLOGY_OTHER
    assert eval04.three_way_agreement(fan_out, fan_out, fan_out) == eval04.AGREEMENT_FULL
    assert eval04.three_way_agreement(fan_out, fan_in, fan_out) == (
        eval04.AGREEMENT_SHAP_STAGE0
    )
    assert eval04.three_way_agreement(fan_out, fan_out, fan_in) == (
        eval04.AGREEMENT_SHAP_LIT
    )
    assert eval04.three_way_agreement(other, fan_in, fan_in) == (
        eval04.AGREEMENT_STAGE0_LIT
    )
    assert eval04.three_way_agreement(fan_out, fan_in, other) == eval04.AGREEMENT_NONE
    # specs/64 §14.5.2's sixth value: a non-discriminative Stage-0 leg.
    for stage0 in (eval04.TOPOLOGY_INDETERMINATE,
                   eval04.STAGE0_INDETERMINATE_IDENTICAL):
        assert eval04.three_way_agreement(fan_out, fan_out, stage0) == (
            eval04.AGREEMENT_STAGE0_INDETERMINATE
        )
    # The seventh value (code addition) takes PRECEDENCE over the sixth: with no
    # literature leg authored, the comparison has nothing to be indeterminate
    # about, and mis-ordering these would hide the unauthored-input state.
    assert eval04.three_way_agreement(
        fan_out, eval04.LITERATURE_NOT_SUPPLIED, eval04.TOPOLOGY_INDETERMINATE,
    ) == eval04.AGREEMENT_LITERATURE_MISSING
    assert eval04.three_way_agreement(
        fan_out, eval04.LITERATURE_NOT_SUPPLIED, fan_out,
    ) == eval04.AGREEMENT_LITERATURE_MISSING
    # ... and it is never conflated with a genuine three-way disagreement.
    assert len({
        eval04.AGREEMENT_FULL, eval04.AGREEMENT_SHAP_STAGE0,
        eval04.AGREEMENT_SHAP_LIT, eval04.AGREEMENT_STAGE0_LIT,
        eval04.AGREEMENT_NONE, eval04.AGREEMENT_STAGE0_INDETERMINATE,
        eval04.AGREEMENT_LITERATURE_MISSING,
    }) == 7


# ---------------------------------------------------------------------------
# 20.3 — the three denominators genuinely differ
# ---------------------------------------------------------------------------

def test_three_denominators_differ_on_absent_classes(tmp_path):
    runs_root = tmp_path / "runs"
    out_dir = tmp_path / "out"
    class_names = ["Benign", "Ghost", "NewAttack"]
    per_class = {
        "Benign": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 100},
        # Zero test support: only full_label_space scores it (as 0).
        "Ghost": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
        # Test-only: present in test, absent from train -> outside closed_set.
        "NewAttack": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 50},
    }
    make_run(runs_root, "nf_alpha_v3", class_names=class_names,
             per_class=per_class, train_labels=[0, 0, 0, 1, 1])
    run = resolve_dataset("alpha", runs_root=runs_root)

    frame = eval02.run([run], out_dir)
    by_denominator = {row["denominator"]: row for _, row in frame.iterrows()}

    n_classes = {
        key: by_denominator[key]["n_classes_in_denominator"]
        for key in (eval02.DENOMINATOR_FULL, eval02.DENOMINATOR_TEST,
                    eval02.DENOMINATOR_CLOSED)
    }
    macro = {
        key: float(by_denominator[key]["macro_f1"]) for key in n_classes
    }
    assert n_classes[eval02.DENOMINATOR_FULL] == 3
    assert n_classes[eval02.DENOMINATOR_TEST] == 2
    assert n_classes[eval02.DENOMINATOR_CLOSED] == 1
    assert len(set(n_classes.values())) == 3
    assert macro[eval02.DENOMINATOR_FULL] == pytest.approx((0.9 + 0.0 + 0.5) / 3)
    assert macro[eval02.DENOMINATOR_TEST] == pytest.approx((0.9 + 0.5) / 2)
    assert macro[eval02.DENOMINATOR_CLOSED] == pytest.approx(0.9)
    assert len({round(v, 10) for v in macro.values()}) == 3

    # The pipeline's own figure sits beside them, labelled, never merged in.
    reported = by_denominator[eval02.DENOMINATOR_REPORTED]
    assert "macro_f1_convention" in str(reported["notes"])
    assert (out_dir / eval02.CSV_NAME).is_file()


def test_absent_train_feature_store_yields_nan_closed_set_row(tmp_path):
    """No ``feature_store/train/`` (BoT-IoT's real state) -> explicit NaN row."""
    runs_root = tmp_path / "runs"
    out_dir = tmp_path / "out"
    make_run(
        runs_root, "nf_alpha_v3",
        class_names=["Benign", "Probe"],
        per_class={
            "Benign": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 100},
            "Probe": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 50},
        },
        train_labels=None,  # the point of the test
    )
    run = resolve_dataset("alpha", runs_root=runs_root)
    frame = eval02.run([run], out_dir)

    closed = frame[frame["denominator"] == eval02.DENOMINATOR_CLOSED]
    assert len(closed) == 1, "the closed_set row is emitted, never dropped"
    row = closed.iloc[0]
    assert np.isnan(float(row["macro_f1"]))
    assert np.isnan(float(row["n_classes_in_denominator"]))
    assert "train feature store absent" in str(row["notes"])
    # The other three denominators are unaffected.
    assert np.isfinite(float(
        frame[frame["denominator"] == eval02.DENOMINATOR_TEST].iloc[0]["macro_f1"]
    ))


# ---------------------------------------------------------------------------
# 20.4 — Benjamini-Hochberg against a hand-computed reference
# ---------------------------------------------------------------------------

def test_benjamini_hochberg_matches_reference_vector():
    # p = [0.01, 0.04, 0.03], n = 3. Sorted: 0.01, 0.03, 0.04.
    # Naive p * n / rank: 0.03, 0.045, 0.04 -> NON-monotone (0.045 > 0.04).
    # Reverse cumulative minimum: 0.03, 0.04, 0.04.
    # Restored to input order: 0.03, 0.04, 0.04.
    got = _stats.benjamini_hochberg([0.01, 0.04, 0.03])
    assert got == pytest.approx([0.03, 0.04, 0.04])
    # Without monotonicity enforcement the middle entry would be 0.045.
    assert got[1] < 0.045

    # NaN passthrough: NaNs stay NaN and are excluded from n (here n = 2, so
    # 0.01 -> 0.02 and 0.03 -> 0.03; an n = 3 would have given 0.03 and 0.045).
    with_nan = _stats.benjamini_hochberg([0.01, float("nan"), 0.03])
    assert np.isnan(with_nan[1])
    assert with_nan[0] == pytest.approx(0.02)
    assert with_nan[2] == pytest.approx(0.03)

    # All-NaN family: all NaN out, never 0.0.
    assert np.all(np.isnan(_stats.benjamini_hochberg([np.nan, np.nan])))

    # Clipping at 1.0.
    assert np.all(_stats.benjamini_hochberg([0.9, 0.95, 0.99]) <= 1.0)

    # apply_family is the single writer of p_corrected + correction_family.
    frame = pd.DataFrame({"p_raw": [0.01, 0.04, 0.03]})
    out = _stats.apply_family(frame, "dataset=alpha;space=feature_group")
    assert list(out["p_raw"]) == [0.01, 0.04, 0.03], "p_raw must never be overwritten"
    assert out["p_corrected"].to_numpy() == pytest.approx([0.03, 0.04, 0.04])
    assert set(out["correction_family"]) == {"dataset=alpha;space=feature_group"}
    with pytest.raises(KeyError):
        _stats.apply_family(pd.DataFrame({"other": [1.0]}), "family")


def test_paired_wilcoxon_refuses_too_few_pairs():
    small = _stats.paired_wilcoxon([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])
    assert np.isnan(small.p_raw)
    assert small.n_pairs == 3
    assert "usable pairs" in small.note

    identical = _stats.paired_wilcoxon([1.0] * 8, [1.0] * 8)
    assert np.isnan(identical.p_raw)
    assert "differences are zero" in identical.note

    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    b = [1.4, 2.6, 3.3, 4.9, 5.2, 6.8, 7.1, 8.7]
    ok = _stats.paired_wilcoxon(a, b)
    assert ok.n_pairs == 8 and np.isfinite(ok.p_raw) and ok.note == ""

    with pytest.raises(ValueError):
        _stats.paired_wilcoxon([1.0, 2.0], [1.0])


def test_rank_agreement_constant_vector_is_nan_not_zero():
    constant = _stats.rank_agreement([1.0, 1.0, 1.0, 1.0], [4.0, 3.0, 2.0, 1.0], 2)
    assert np.isnan(constant.spearman_rho), "a constant vector makes rho UNDEFINED"
    assert constant.spearman_rho is not None and constant.spearman_rho != 0.0
    assert "constant" in constant.note and "NOT the same as" in constant.note
    # The Jaccard overlap is still computable and still reported.
    assert np.isfinite(constant.jaccard_topk)

    agree = _stats.rank_agreement([4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0], 2)
    assert agree.spearman_rho == pytest.approx(1.0)
    assert agree.jaccard_topk == pytest.approx(1.0)
    assert np.isfinite(agree.kendall_tau)
    assert _stats.jaccard([], []) != _stats.jaccard([1], [1])
    assert np.isnan(_stats.jaccard([], []))


# ---------------------------------------------------------------------------
# 20.5 / 20.5b — dataset resolution
# ---------------------------------------------------------------------------

def _resolution_fixture(tmp_path: Path) -> Path:
    """Both historical runs/ layouts, side by side, fully synthetic."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    # Layout A (pre-2026-08-26): the STUB sits at the bare base id and the
    # canonical run is suffixed. A "prefer bare" rule resolves this one wrong.
    stub = runs_root / "nf_foo_v3"
    (stub / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
    (stub / "outputs" / "metrics").mkdir(parents=True, exist_ok=True)
    _complete_shell(runs_root, "nf_foo_v3_r3_s2")

    # Layout B (current): canonical at the bare base id, incomplete variants
    # beside it. A "prefer suffixed" rule resolves this one wrong.
    bar = _complete_shell(runs_root, "nf_bar_v3")
    (bar / "outputs_variant").mkdir(parents=True, exist_ok=True)
    (runs_root / "nf_bar_v3_stub").mkdir(parents=True, exist_ok=True)
    _make_shell_run(runs_root, "nf_bar_v3_binary", ("artifacts/label_map.json",))
    _make_shell_run(runs_root, "nf_bar_v3_r1_r01_A",
                    ("artifacts/evaluation/metrics.json",))

    # Never a candidate: SKIP_DIR_NAMES.
    _complete_shell(runs_root / "archive", "nf_baz_v3")
    return runs_root


def test_resolution_rejects_stub_and_variant_runs(tmp_path):
    runs_root = _resolution_fixture(tmp_path)

    foo = resolve_dataset("foo", runs_root=runs_root)
    assert foo.run_dir.name == "nf_foo_v3_r3_s2"
    assert foo.run_dir.name != "nf_foo_v3", "resolved to the empty stub"
    assert foo.variant_suffix == "_r3_s2"

    bar = resolve_dataset("bar", runs_root=runs_root)
    assert bar.run_dir.name == "nf_bar_v3"
    assert bar.variant_suffix == ""
    assert not bar.run_dir.name.endswith(("_stub", "_binary", "_r1_r01_A"))

    # Derived, never asserted against a literal count of runs on disk.
    assert available_datasets(runs_root) == ["bar", "foo"]
    assert "baz" not in available_datasets(runs_root)

    # A sibling directory that merely looks like outputs/ is never traversed.
    for run in (foo, bar):
        assert outputs_dir(run).name == "outputs"
        assert "outputs_variant" not in str(outputs_dir(run))


def test_ambiguous_resolution_raises_and_names_candidates(tmp_path):
    runs_root = _resolution_fixture(tmp_path)
    second = _complete_shell(runs_root, "nf_bar_v3_r2_final")

    with pytest.raises(DatasetResolutionError) as excinfo:
        resolve_dataset("bar", runs_root=runs_root)
    message = str(excinfo.value)
    assert "nf_bar_v3" in message and "nf_bar_v3_r2_final" in message
    assert "--run-dir" in message
    assert "prefer" in message, "the message must state that NO tiebreak applies"
    # The ambiguous base is excluded from --all-datasets rather than guessed at.
    assert "bar" not in available_datasets(runs_root)

    # The documented escape hatch resolves it, and still completeness-checks.
    pinned = resolve_dataset("bar", runs_root=runs_root,
                             overrides={"bar": second})
    assert pinned.run_dir == second.resolve()

    # Zero complete candidates: the "pipeline has not finished" state, which no
    # real dataset is in today.
    incomplete_root = tmp_path / "incomplete"
    _make_shell_run(incomplete_root, "nf_qux_v3", ("artifacts/label_map.json",))
    with pytest.raises(DatasetResolutionError) as excinfo:
        resolve_dataset("qux", runs_root=incomplete_root)
    message = str(excinfo.value)
    assert "nf_qux_v3" in message
    assert _REQUIRED[0] in message, "the first missing artifact must be named"

    # An ambiguous PREFIX picks nothing either.
    prefix_root = tmp_path / "prefix"
    _complete_shell(prefix_root, "nf_zebra_v3")
    _complete_shell(prefix_root, "nf_zealot_v3")
    with pytest.raises(DatasetResolutionError) as excinfo:
        resolve_dataset("ze", runs_root=prefix_root)
    assert "ambiguous prefix" in str(excinfo.value)
    # ... while the full key still resolves.
    assert resolve_dataset("zebra", runs_root=prefix_root).run_dir.name == "nf_zebra_v3"


# ---------------------------------------------------------------------------
# 20.6 / 20.7 — source-text pins
# ---------------------------------------------------------------------------

_PACKAGE_FILES = sorted((REPO_ROOT / "explore" / "evaluation").glob("*.py"))

_DATASET_NAME_PATTERNS = (
    "nf_unsw", "nf_bot_iot", "nf_ton_iot", "nf_cicids",
    "UNSW", "BoT-IoT", "ToN-IoT", "CICIDS", "CICIoT",
)

_BARE_TEMPORAL_RE = re.compile(r"^_?temporal_|[^a-z_]temporal_[a-z]")


def _module_string_literals(tree: ast.AST) -> list[str]:
    """Every string constant in a module except its docstrings."""
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            text = ast.get_docstring(node, clean=False)
            if text is not None:
                docstrings.add(text)
    return [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and node.value not in docstrings
    ]


def test_no_hardcoded_dataset_names():
    assert _PACKAGE_FILES, "explore/evaluation/ has no modules to scan"
    offenders: list[str] = []
    for path in _PACKAGE_FILES:
        tree = ast.parse(path.read_text())
        for literal in _module_string_literals(tree):
            lowered = literal.lower()
            for pattern in _DATASET_NAME_PATTERNS:
                if pattern.lower() in lowered:
                    offenders.append(f"{path.name}: {literal!r} contains {pattern!r}")
    assert not offenders, (
        "explore/evaluation/ must derive every dataset name from disk "
        "(specs/64 §13.3, owner constraint 3):\n" + "\n".join(offenders)
    )


def test_no_bare_temporal_identifier():
    """D6 / Part I §2: `temporal` alone is ambiguous and is never a bare name."""
    assert _PACKAGE_FILES
    offenders: list[str] = []
    for path in _PACKAGE_FILES:
        tree = ast.parse(path.read_text())
        defined: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                defined.append(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined.append(node.id)
            elif isinstance(node, ast.arg):
                defined.append(node.arg)
        for name in defined:
            if _BARE_TEMPORAL_RE.search(name):
                offenders.append(f"{path.name}: definition {name!r}")
        # Output-column names are literals, so they are pinned too.
        for literal in _module_string_literals(tree):
            if _BARE_TEMPORAL_RE.search(literal) and "." not in literal:
                offenders.append(f"{path.name}: literal {literal!r}")
    assert not offenders, (
        "every temporal identifier must be fidelity_temporal_* or "
        "stability_temporal_* (specs/64 D6):\n" + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# 20.8 — non-feature stability rows are NaN with a note, not absent
# ---------------------------------------------------------------------------

def test_coalition_space_stability_rows_are_nan_not_missing(tmp_path):
    runs_root = tmp_path / "runs"
    out_dir = tmp_path / "out"
    class_names = ["Benign", "Probe"]
    per_class = {
        "Benign": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 100},
        "Probe": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 50},
    }
    # summary_temporal.json / summary_novelty.json exist and carry per-class
    # fidelity, but NO `stability` key -- exactly the real on-disk shape.
    fidelity_only = {
        name: {"n_flows": 5, "fidelity_plus": 0.01, "fidelity_minus": None}
        for name in class_names
    }
    make_run(
        runs_root, "nf_alpha_v3", class_names=class_names, per_class=per_class,
        summary_per_class={
            name: {"n_flows": 5, "fidelity_plus": 0.1, "fidelity_minus": 0.02,
                   "stability": 0.001}
            for name in class_names
        },
        extra_spaces={"temporal": fidelity_only, "novelty": dict(fidelity_only)},
        stability_rows=[
            {"class_name": "Benign", "edge_id": 1, "mean_phi_std": 0.001,
             "max_phi_std": 0.004},
            {"class_name": "Probe", "edge_id": 2, "mean_phi_std": 0.002,
             "max_phi_std": 0.005},
        ],
    )
    run = resolve_dataset("alpha", runs_root=runs_root)
    frame = eval01.run([run], out_dir)

    for space in ("temporal", "novelty"):
        for metric in eval01.STABILITY_METRIC_KEYS:
            rows = frame[
                (frame["coalition_space"] == space) & (frame["metric"] == metric)
            ]
            assert len(rows) == len(class_names), (
                f"{space}/{metric}: rows are missing, not NaN"
            )
            assert rows["value"].isna().all()
            assert (rows["notes"] == STABILITY_SCOPE_NOTE).all()

    # The feature space, by contrast, carries real values from stability.csv.
    feature = frame[
        (frame["coalition_space"] == "feature")
        & (frame["metric"] == "stability_intrarun_mean")
    ]
    assert np.isfinite(feature["value"].to_numpy(dtype=float)).all()
    # NaN is emitted as an empty CSV cell, never as a 0.
    csv_text = (out_dir / eval01.LONG_CSV_NAME).read_text()
    assert STABILITY_SCOPE_NOTE.split("(")[0].strip() in csv_text


# ---------------------------------------------------------------------------
# 20.9 / 20.10 — the owner-authored mapping tables
# ---------------------------------------------------------------------------

def test_missing_mapping_csv_exits_zero(tmp_path, caplog):
    runs_root, out_dir, run = _full_run(tmp_path)
    # eval04 reads eval05's CSV rather than recomputing the ranking.
    eval05.run([run], out_dir)
    missing = tmp_path / "nowhere" / "proxy_gt_feature_expectation_mapping.csv"
    assert not missing.exists()

    with caplog.at_level(logging.WARNING):
        code = eval04.main([
            "--dataset", "alpha",
            "--runs-root", str(runs_root),
            "--out-dir", str(out_dir),
            "--mapping-csv", str(missing),
        ])

    assert code == 0, "an unauthored mapping table is a normal state, not a failure"
    assert any(
        record.levelno == logging.WARNING and str(missing) in record.getMessage()
        for record in caplog.records
    ), "the absent mapping file must be named in a WARNING"
    # The §4.4a half is skipped -- as an EMPTY, header-carrying CSV.
    rank_csv = out_dir / eval04.RANK_CSV_NAME
    assert rank_csv.is_file()
    assert list(pd.read_csv(rank_csv).columns) == list(eval04.RANK_CSV_COLUMNS)
    assert pd.read_csv(rank_csv).empty
    # The §4.4b structural half is computed and written anyway.
    structural = pd.read_csv(out_dir / eval04.STRUCTURAL_CSV_NAME)
    assert set(structural["class_name"]) == {"Benign", "Ghost", "Probe"}
    assert list(structural.columns) == list(eval04.STRUCTURAL_CSV_COLUMNS)


def test_missing_structural_expectations_csv_reports_not_supplied(tmp_path, caplog):
    """The --structural-expectations-csv absence path (code addition to §18)."""
    runs_root, out_dir, run = _full_run(tmp_path)
    eval05.run([run], out_dir)
    absent = tmp_path / "nowhere" / eval04.STRUCTURAL_EXPECTATION_CSV_NAME

    with caplog.at_level(logging.WARNING):
        code = eval04.main([
            "--dataset", "alpha",
            "--runs-root", str(runs_root),
            "--out-dir", str(out_dir),
            "--mapping-csv", str(tmp_path / "nowhere" / eval04.MAPPING_CSV_NAME),
            "--structural-expectations-csv", str(absent),
        ])

    assert code == 0
    assert str(absent) in caplog.text
    structural = pd.read_csv(out_dir / eval04.STRUCTURAL_CSV_NAME)
    assert set(structural["literature_expected_topology"]) == {
        eval04.LITERATURE_NOT_SUPPLIED
    }
    # An unauthored literature leg is its OWN category, never scored as a
    # genuine three-way disagreement.
    assert set(structural["three_way_agreement"]) == {
        eval04.AGREEMENT_LITERATURE_MISSING
    }
    assert eval04.AGREEMENT_LITERATURE_MISSING != eval04.AGREEMENT_NONE
    # Supplying the table populates the leg.
    supplied = out_dir / eval04.STRUCTURAL_EXPECTATION_CSV_NAME
    pd.DataFrame([
        {"dataset": run.label, "class_name": "Probe",
         "literature_expected_topology": eval04.TOPOLOGY_FAN_OUT},
    ]).to_csv(supplied, index=False)
    loaded = eval04.load_structural_expectations(supplied)
    assert loaded[(run.label, "Probe")] == eval04.TOPOLOGY_FAN_OUT
    with pytest.raises(ValueError, match="missing column"):
        bad = out_dir / "bad_structural.csv"
        pd.DataFrame([{"dataset": "x"}]).to_csv(bad, index=False)
        eval04.load_structural_expectations(bad)


def test_mapping_csv_unknown_feature_group_raises(tmp_path):
    mapping = tmp_path / eval04.MAPPING_CSV_NAME
    pd.DataFrame([
        {"dataset": "alpha", "class_name": "Probe",
         "shap_gsd_feature_group": "G1",
         "expected_rank_or_direction": 1, "source_citation": "cite",
         "mapper_confidence": "high"},
        {"dataset": "alpha", "class_name": "Probe",
         "shap_gsd_feature_group": "TYPO_GROUP",
         "expected_rank_or_direction": 2, "source_citation": "cite",
         "mapper_confidence": "high"},
    ]).to_csv(mapping, index=False)
    observed = {"alpha": set(_GROUPS)}

    with pytest.raises(ValueError) as excinfo:
        eval04.load_mapping_csv(mapping, observed)
    assert "TYPO_GROUP" in str(excinfo.value)
    assert "alpha" in str(excinfo.value)

    # A valid table loads, and is what makes the rank half computable.
    good = tmp_path / "good_mapping.csv"
    pd.DataFrame([
        {"dataset": "alpha", "class_name": "Probe",
         "shap_gsd_feature_group": group,
         "expected_rank_or_direction": rank, "source_citation": "cite",
         "mapper_confidence": "high"}
        for rank, group in enumerate(_GROUPS, start=1)
    ]).to_csv(good, index=False)
    assert eval04.load_mapping_csv(good, observed) is not None

    # An out-of-vocabulary confidence is a hard error too.
    bad_conf = tmp_path / "bad_conf.csv"
    frame = pd.read_csv(good)
    frame.loc[0, "mapper_confidence"] = "very_high"
    frame.to_csv(bad_conf, index=False)
    with pytest.raises(ValueError, match="mapper_confidence"):
        eval04.load_mapping_csv(bad_conf, observed)


def _probe_mapping_frame(label: str, ordered_groups: list[str]) -> pd.DataFrame:
    """A valid mapping table for class ``Probe``, ranking ``ordered_groups`` 1..n."""
    return pd.DataFrame([
        {"dataset": label, "class_name": "Probe",
         "shap_gsd_feature_group": group, "expected_rank_or_direction": rank,
         "source_citation": "cite", "mapper_confidence": "high"}
        for rank, group in enumerate(ordered_groups, start=1)
    ])


def test_rank_correlation_scores_a_present_mapping(tmp_path):
    """§4.4a's scoring half, with the mapping table actually supplied.

    No dataset on disk has an authored mapping table, so without this test the
    whole rank-correlation path -- including the sign convention that decides
    whether "agrees with the literature" comes out positive or negative -- is
    exercised by nothing at all. ``Probe``'s synthetic |phi_F| ranking is
    G2 > G3 > G4 > G1, so a mapping listing exactly that order must score
    rho = +1 and the reversed order rho = -1. A sign flip in
    ``_expected_rank_vector``/``compute_rank_correlation`` would invert the
    manuscript's headline claim while every other test still passed.
    """
    runs_root, out_dir, run = _full_run(tmp_path)
    per_class = eval05.run([run], out_dir)
    observed = {run.label: set(_GROUPS)}

    agreeing = tmp_path / "agreeing_mapping.csv"
    _probe_mapping_frame(run.label, ["G2", "G3", "G4", "G1"]).to_csv(
        agreeing, index=False,
    )
    mapping = eval04.load_mapping_csv(agreeing, observed)
    assert mapping is not None
    frame = eval04.compute_rank_correlation(per_class, mapping)

    assert list(frame.columns) == list(eval04.RANK_CSV_COLUMNS)
    assert len(frame) == 1, "one row per (dataset, class) named by the mapping"
    row = frame.iloc[0]
    assert row["class_name"] == "Probe"
    assert float(row["spearman_rho_vs_expected"]) == pytest.approx(1.0)
    assert str(row["rho_undefined_reason"]) == ""
    assert int(row["n_shap_features_ranked"]) == len(_GROUPS)
    assert int(row["n_expected_features_mapped"]) == len(_GROUPS)
    assert int(row["n_flows_aggregated"]) == 3
    assert str(row["mapping_confidence"]) == "high"
    # Part I §5: both p-values present, neither overwriting the other, and the
    # family named in the row itself.
    assert np.isfinite(float(row["p_raw"]))
    assert np.isfinite(float(row["p_corrected"]))
    assert run.label in str(row["correction_family"])

    reversed_csv = tmp_path / "reversed_mapping.csv"
    _probe_mapping_frame(run.label, ["G1", "G4", "G3", "G2"]).to_csv(
        reversed_csv, index=False,
    )
    reversed_frame = eval04.compute_rank_correlation(
        per_class, eval04.load_mapping_csv(reversed_csv, observed),
    )
    assert float(reversed_frame.iloc[0]["spearman_rho_vs_expected"]) == pytest.approx(
        -1.0
    ), "a mapping that inverts the observed ranking must score negative, not positive"

    # A class the mapping names but the explanations do not cover still emits a
    # row, with the reason recorded rather than the class disappearing.
    ghost = tmp_path / "ghost_mapping.csv"
    pd.DataFrame([
        {"dataset": run.label, "class_name": "Ghost",
         "shap_gsd_feature_group": "G1", "expected_rank_or_direction": 1,
         "source_citation": "cite", "mapper_confidence": "low"},
    ]).to_csv(ghost, index=False)
    ghost_frame = eval04.compute_rank_correlation(
        per_class, eval04.load_mapping_csv(ghost, observed),
    )
    assert len(ghost_frame) == 1
    assert np.isnan(float(ghost_frame.iloc[0]["spearman_rho_vs_expected"]))
    assert str(ghost_frame.iloc[0]["rho_undefined_reason"]) != ""
    assert str(ghost_frame.iloc[0]["mapping_confidence"]) == "low"


def test_topology_classifiers_cover_every_category():
    """Both structural legs' classifiers, including the discriminating branches.

    §14.5.2 keys the Stage-0 leg on ``(n_src_nodes, n_dst_nodes)`` rather than
    on ``d_gw`` precisely because the gateway distance is degenerate on real
    data; this pins that the node-count rule actually discriminates, and that
    the identical-across-classes fallback is a distinct value rather than a
    silent ``other``.
    """
    fan_out_pairs = [("a", f"d{i}") for i in range(6)]
    fan_in_pairs = [(f"s{i}", "z") for i in range(6)]
    persistent_pairs = [("a", "b")] * 8 + [("a", "c")]
    assert eval04.classify_subgraph_topology(fan_out_pairs)[0] == (
        eval04.TOPOLOGY_FAN_OUT
    )
    assert eval04.classify_subgraph_topology(fan_in_pairs)[0] == (
        eval04.TOPOLOGY_FAN_IN
    )
    assert eval04.classify_subgraph_topology(persistent_pairs)[0] == (
        eval04.TOPOLOGY_PERSISTENT
    )
    assert eval04.classify_subgraph_topology([])[0] == eval04.TOPOLOGY_INDETERMINATE
    # Two src, two dst, no modal pair reaching the persistence threshold.
    assert eval04.classify_subgraph_topology(
        [("a", "b"), ("a", "c"), ("d", "b"), ("d", "c")]
    )[0] == eval04.TOPOLOGY_OTHER

    entries = {
        "Fan_in": {"n_src_nodes": 60.0, "n_dst_nodes": 1.0},
        "Fan_out": {"n_src_nodes": 1.0, "n_dst_nodes": 40.0},
        "Persistent": {"n_src_nodes": 1.0, "n_dst_nodes": 1.0},
        "Other": {"n_src_nodes": 4.0, "n_dst_nodes": 3.0},
    }
    expected = {
        "Fan_in": eval04.TOPOLOGY_FAN_IN,
        "Fan_out": eval04.TOPOLOGY_FAN_OUT,
        "Persistent": eval04.TOPOLOGY_PERSISTENT,
        "Other": eval04.TOPOLOGY_OTHER,
    }
    for class_name, want in expected.items():
        got, evidence = eval04.classify_stage0_topology(entries[class_name], entries)
        assert got == want, f"{class_name}: {got!r} != {want!r} ({evidence})"

    # A dataset whose classes are structurally identical: the measurement cannot
    # discriminate, and that is its OWN value, never scored as `other`.
    identical = {
        name: {"n_src_nodes": 2.0, "n_dst_nodes": 2.0}
        for name in ("A", "B", "C")
    }
    got, evidence = eval04.classify_stage0_topology(identical["A"], identical)
    assert got == eval04.STAGE0_INDETERMINATE_IDENTICAL
    assert "cannot discriminate" in evidence
    assert eval04.classify_stage0_topology(None, entries)[0] == (
        eval04.TOPOLOGY_INDETERMINATE
    )
    assert eval04.classify_stage0_topology(
        {"n_src_nodes": float("nan"), "n_dst_nodes": 1.0}, entries,
    )[0] == eval04.TOPOLOGY_INDETERMINATE


def test_out_of_split_subgraph_edge_id_is_a_hard_error():
    """The EID-alignment guard no real dataset currently triggers."""
    edge_indices = np.asarray([10, 11, 12, 20], dtype=np.int64)
    assert eval04.resolve_edge_rows(edge_indices, [10, 20], "alpha") == [0, 3]
    with pytest.raises(KeyError) as excinfo:
        eval04.resolve_edge_rows(edge_indices, [10, 999], "alpha")
    assert "999" in str(excinfo.value)
    assert "EID-alignment" in str(excinfo.value)
    # An id past the end of the array is a miss, not an IndexError.
    with pytest.raises(KeyError):
        eval04.resolve_edge_rows(edge_indices, [21], "alpha")
    with pytest.raises(KeyError):
        eval07b.lookup_timestamp(
            edge_indices, np.asarray([1.0, 2.0, 3.0, 4.0]), 999, "alpha",
        )


# ---------------------------------------------------------------------------
# eval03 — the node-coalition gap is reported, not papered over
# ---------------------------------------------------------------------------

def test_node_coalition_agreement_is_nan_with_alignment_note(tmp_path):
    runs_root, out_dir, run = _full_run(tmp_path)

    agreement = eval03.compute_agreement([run], top_k=3)
    node_rows = agreement[agreement["space"] == eval03.SPACE_NODE_COALITION]
    assert not node_rows.empty, "no node-coalition rows were emitted at all"

    # Every node-coalition cell is NaN by structural necessity...
    assert node_rows["spearman_rho"].isna().all()
    assert node_rows["kendall_tau"].isna().all()
    assert node_rows["jaccard_topk"].isna().all()
    # ... and every one of them says WHY, specifically.
    assert node_rows["notes"].str.contains("no accompanying node-id column").all()
    # Rows involving SHAP-GSD carry the live-measured length evidence, so the
    # gap is demonstrated rather than merely asserted.
    involving = node_rows[
        (node_rows["explainer_a"] == eval03.SHAP_GSD)
        | (node_rows["explainer_b"] == eval03.SHAP_GSD)
    ]
    assert not involving.empty
    assert involving["notes"].str.contains("shared flows").all()
    assert involving["notes"].str.contains("matches SHAP-GSD's node_ids length "
                                           "in 0 of them").all()

    # The alignable half still works, so the NaNs above are specific to the
    # node space rather than eval03 failing wholesale.
    feature_rows = agreement[agreement["space"] == eval03.SPACE_FEATURE_GROUP]
    assert not feature_rows.empty
    scored = feature_rows[feature_rows["n_flows"] > 0]
    assert np.isfinite(scored["spearman_rho"].to_numpy(dtype=float)).all()
    assert np.isfinite(scored["p_corrected"].to_numpy(dtype=float)).all()
    assert scored["correction_family"].str.startswith("dataset=").all()
    # BH families are per (dataset, space), never mixed across spaces.
    assert set(feature_rows["correction_family"]) != set(
        node_rows["correction_family"]
    )

    flags = eval03.compute_disagreement_flags(agreement, eval03.DEFAULT_RHO_THRESHOLD)
    node_flags = flags[flags["space"] == eval03.SPACE_NODE_COALITION]
    assert not node_flags.empty
    # NaN never reads as "below threshold": an uncomputable rho is not a
    # disagreement finding.
    assert not node_flags["flagged"].any()
    assert (node_flags["n_pairs"] == 0).all()
    assert node_flags["notes"].str.contains("no computable rho").all()


# ---------------------------------------------------------------------------
# eval07b — an empty temporal window is reported, not silently dropped
# ---------------------------------------------------------------------------

def test_empty_temporal_window_is_reported_not_dropped(tmp_path):
    runs_root, out_dir, run = _full_run(tmp_path)
    # Benign's flows (ts 1-3) and Probe's (ts 100-102) sit in disjoint halves,
    # so with two equal-count windows each class occupies exactly one.
    frame = eval07b.compute_stability_temporal_windows(
        [run], n_windows=2, mode=eval07b.WINDOW_MODE_EQUAL_COUNT,
    )
    assert list(frame.columns) == list(eval07b.CSV_COLUMNS)

    for class_name in ("Benign", "Probe"):
        rows = frame[
            (frame["class_name"] == class_name) & (frame["feature_group"] != "")
        ]
        assert len(rows) == len(_GROUPS)
        assert (rows["n_windows_available"] == 1).all()
        # One window means no across-window spread is defined -> NaN, not 0.0.
        assert rows["std_across_windows"].isna().all()
        assert rows["mean_across_windows"].notna().all()
        assert rows["notes"].str.contains("contain no flow of this class").all()
        assert rows["notes"].str.contains(eval07b.WEAKER_STATISTIC_NOTE).all()

    # A class with no explained flows at all still gets one explicit row.
    ghost = frame[frame["class_name"] == "Ghost"]
    assert len(ghost) == 1
    assert int(ghost.iloc[0]["n_flows_total"]) == 0
    assert int(ghost.iloc[0]["n_windows_available"]) == 0
    assert "no explained flows" in str(ghost.iloc[0]["notes"])

    with pytest.raises(ValueError, match="unknown window mode"):
        eval07b.window_boundaries(np.asarray([1.0, 2.0, 3.0]), 2, "sliding")


# ---------------------------------------------------------------------------
# eval08 — cross-dataset report guards
# ---------------------------------------------------------------------------

def test_eval08_dataset_label_mismatch_emits_consistency_warning(tmp_path):
    runs_root, out_dir, run = _full_run(tmp_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The same underlying run, written under a DIFFERENT display label by an
    # earlier invocation (the real `unsw` vs `unsw_nb15` situation).
    pd.DataFrame([{
        "dataset": "alpha_other_label", "class_name": "Benign",
        "coalition_space": "feature", "metric": "fidelity_plus", "value": 0.1,
        "class_present_in_test": True, "notes": "",
    }]).to_csv(out_dir / "eval_long_metrics.csv", index=False)

    report = eval08.build_report([run], out_dir)
    assert "## Input consistency warning" in report
    assert "alpha_other_label" in report
    assert run.label in report
    assert "eval_long_metrics.csv" in report

    # Matching labels produce no warning section.
    pd.DataFrame([{
        "dataset": run.label, "class_name": "Benign",
        "coalition_space": "feature", "metric": "fidelity_plus", "value": 0.1,
        "class_present_in_test": True, "notes": "",
    }]).to_csv(out_dir / "eval_long_metrics.csv", index=False)
    clean = eval08.build_report([run], out_dir)
    assert "## Input consistency warning" not in clean
    # A missing input is an explicit stub, never a silently empty section.
    assert "has not been generated" in clean


def test_eval08_refuses_fewer_than_two_datasets(tmp_path, caplog):
    runs_root, out_dir, run = _full_run(tmp_path)

    with caplog.at_level(logging.ERROR):
        code = eval08.main([
            "--dataset", "alpha",
            "--runs-root", str(runs_root),
            "--out-dir", str(out_dir),
        ])

    assert code == 2
    # eval08 returns 2 for a resolution failure too; pin the right branch.
    assert "refuses to run" in caplog.text
    assert not (out_dir / eval08.REPORT_NAME).exists()

    # With two datasets it runs.
    make_run(
        runs_root, "nf_beta_v3",
        class_names=["Benign"],
        per_class={"Benign": {"precision": 0.9, "recall": 0.9, "f1": 0.9,
                              "support": 10}},
    )
    code = eval08.main([
        "--dataset", "alpha", "--dataset", "beta",
        "--runs-root", str(runs_root),
        "--out-dir", str(out_dir),
    ])
    assert code == 0
    assert (out_dir / eval08.REPORT_NAME).is_file()


# ---------------------------------------------------------------------------
# 20.11 — real-data anchor (gated, opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("SHAP_GSD_RUN_REAL_DATA_TESTS") != "1",
    reason="Reads the real runs/ tree; opt-in via SHAP_GSD_RUN_REAL_DATA_TESTS=1, "
           "not part of the default `pytest tests/ -v` gate.",
)
def test_real_data_anchor(tmp_path):
    runs_root = REPO_ROOT / "runs"
    if not runs_root.is_dir():
        pytest.skip("no runs/ directory on this checkout")
    names = available_datasets(runs_root)
    # No absolute count is asserted: the assertion is >= 1 and "matches what
    # resolution found", never "== 3" or "== 4".
    assert len(names) >= 1, "resolution found no complete run under runs/"

    runs = [resolve_dataset(name, runs_root=runs_root) for name in names]
    for run in runs:
        basename = run.run_dir.name
        assert not basename.endswith("_stub"), basename
        assert not basename.endswith("_binary"), basename
        assert not re.search(r"_r1_", basename), basename

    out_dir = tmp_path / "out"  # never the repo's real outputs/ tree
    long_frame = eval01.run(runs, out_dir)
    macro_frame = eval02.run(runs, out_dir)

    for run in runs:
        rows = long_frame[long_frame["dataset"] == run.label]
        assert not rows.empty, f"{run.label} missing from eval_long_metrics.csv"
        label_map = json.loads(
            (_discover.artifacts_dir(run) / "label_map.json").read_text()
        )
        assert rows["class_name"].nunique() == len(label_map), run.label

        metrics_raw = json.loads(
            (_discover.artifacts_dir(run) / "evaluation" / "metrics.json").read_text()
        )
        if "macro_f1_convention" not in metrics_raw:
            continue
        test_row = macro_frame[
            (macro_frame["dataset"] == run.label)
            & (macro_frame["denominator"] == eval02.DENOMINATOR_TEST)
        ]
        assert len(test_row) == 1, run.label
        assert float(test_row.iloc[0]["macro_f1"]) == pytest.approx(
            float(metrics_raw["macro_f1"]), abs=1e-4,
        ), f"{run.label}: §14.2 positive control failed"
