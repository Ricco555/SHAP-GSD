"""
Tests for the Phase-3 shard-selection helpers (``src/model/selection.py``)
and the tuner-side full-list-iteration invariant (``src/model/tuner.py``) —
specs/34 §3.3/§3.4/§7.11, specs/35 Part VII.1 (Stream A items).

All tests are artifact-free (no graphs, no feature store, no checkpoints)
and run in the default ``pytest tests/ -v`` gate — no skip markers.
``src.model.selection`` is stdlib-only, so nothing here imports torch/dgl.

The synthetic grid ``SYNTH`` (2x3x2 = 12 trials, axis names disjoint from
the real grid's) is the proof of the specs/34 §3.3 invariant's GENERALITY:
any hardcoded stride (``// 36``, ``% 3``) or axis-name assumption in the
ownership path fails on it. This synthetic-grid test IS the invariant
proof, not a supplement to a real-grid test (specs/35 §VII.1).

Covered here (Stream A: scripts/03_tune.py + src/model/tuner.py):
  S1 — full-list-iteration / dict-lookup ownership invariant on SYNTH:
       every non-empty axis subset, every valid k — brute-force parity,
       pairwise disjointness, full cover.
  S2 — real-grid (configs/tuning_grid.yaml) cover/disjointness for every
       recommended axis combination; 9-way partition detail pins.
  S3 — batch_size-exclusion assertion (dominant-cost axis guard).
  S4 — unknown-axis-name assertion (typo fails loudly).
  S5 — K/N validation: wrong N vs the live search_space's computed
       cardinality, out-of-range k, malformed K/N strings, duplicate axes.
  S6 — select_best dense/sparse parity, lowest-global-index tie-break,
       order independence, empty list, 0.0-scoring sole record.
  S7 — compute_grid_fingerprint: sha256 format, insertion-order
       invariance (sort_keys pin), and the 8-key subsumption flips
       (5 fixed-overwrite keys + the 3 composite/R6 keys).
  S8 — write_json_atomic: .tmp + os.replace happy path; failure injection
       leaves the original target intact and no .tmp behind.
  S9 — verbatim shard-filename pin: specs/34 §3.5's exact example string
       tuning_results_shard_fan35-25_hid256.json for shard 8/9.
  S10 — stdlib-only import contract of src/model/selection.py.
  S11 (partial, tuner-side) — source pin that run() iterates the FULL
       configs list, best_idx is gone, and _configs delegates to
       enumerate_grid.

NOT covered here (other streams / discharged elsewhere per specs/35
§VII.2's note): promote_best.py end-to-end (tests/test_promote_best.py),
run()'s shard mode end to end (needs graphs/FeatureStore/Trainer — covered
by the source pins, the shared-function architecture, and the specs/35
§VII.3 operational dry run).
"""

import ast
import itertools
import json
import random
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.selection import (  # noqa: E402
    FORBIDDEN_SHARD_AXES,
    SHARD_FILE_PREFIX,
    compute_grid_fingerprint,
    enumerate_grid,
    resolve_shard,
    select_best,
    write_json_atomic,
)

# Synthetic grid: shape (2x3x2 = 12) AND axis names disjoint from the real
# 3x3x4x3 grid, so index arithmetic tuned to today's grid cannot pass.
SYNTH = {"alpha": [1, 2], "beta": ["x", "y", "z"], "gamma": [0.5, 0.25]}
SYNTH_N_TOTAL = 12


def _real_search_space() -> dict:
    with open(REPO_ROOT / "configs" / "tuning_grid.yaml") as f:
        return yaml.safe_load(f)["search_space"]


def _brute_force_owned(search_space: dict, owned_values: dict) -> list[int]:
    """The specs/34 §3.1 predicate, written independently of resolve_shard:
    a dict lookup against each trial's own recorded axis values."""
    return [
        i for i, c in enumerate(enumerate_grid(search_space))
        if all(c[a] == v for a, v in owned_values.items())
    ]


def _axis_subsets(search_space: dict):
    axes = list(search_space.keys())
    for r in range(1, len(axes) + 1):
        yield from itertools.combinations(axes, r)


# ──────────────────────────────────────────────────────────────────────────
# S1 — full-list-iteration / dict-lookup ownership invariant, SYNTHETIC grid
# ──────────────────────────────────────────────────────────────────────────

def test_s1_synthetic_grid_ownership_is_value_lookup_cover_and_disjoint():
    """For EVERY non-empty axis subset of SYNTH and every valid k:
    resolve_shard's owned_indices equal the brute-force dict-lookup
    predicate's; across all k the sets are pairwise disjoint and union to
    range(12). Proves ownership is a value lookup on the trial's own dict
    on a grid where hardcoded index arithmetic cannot accidentally pass."""
    assert len(enumerate_grid(SYNTH)) == SYNTH_N_TOTAL

    for axes in _axis_subsets(SYNTH):
        n = 1
        for a in axes:
            n *= len(SYNTH[a])

        seen: dict[int, int] = {}  # global trial index -> owning k
        for k in range(n):
            spec = resolve_shard(SYNTH, ",".join(axes), f"{k}/{n}")

            assert spec["partition_scheme"] == "axis"
            assert spec["shard_index"] == k
            assert spec["num_shards"] == n
            assert spec["n_total"] == SYNTH_N_TOTAL
            assert spec["shard_axes"] == [a for a in SYNTH if a in axes]

            # Brute-force parity: the dict-lookup predicate, independently.
            assert spec["owned_indices"] == _brute_force_owned(
                SYNTH, spec["owned_values"]
            ), f"axes={axes}, k={k}"

            # Every owned index satisfies the predicate on ITS OWN dict.
            grid = enumerate_grid(SYNTH)
            for i in spec["owned_indices"]:
                assert all(
                    grid[i][a] == v for a, v in spec["owned_values"].items()
                )

            # Disjointness: no index owned by two shards.
            for i in spec["owned_indices"]:
                assert i not in seen, (
                    f"axes={axes}: index {i} owned by shards "
                    f"{seen[i]} and {k}"
                )
                seen[i] = k

        # Full cover: union over all k is exactly range(12).
        assert sorted(seen) == list(range(SYNTH_N_TOTAL)), f"axes={axes}"


def test_s1_synthetic_cli_axis_order_is_normalized():
    """CLI axis order must not change the partition: reversed --shard-axes
    yields the same normalized axes, owned indices and filename."""
    a = resolve_shard(SYNTH, "alpha,gamma", "1/4")
    b = resolve_shard(SYNTH, "gamma,alpha", "1/4")
    assert a["shard_axes"] == b["shard_axes"] == ["alpha", "gamma"]
    assert a["owned_indices"] == b["owned_indices"]
    assert a["owned_values"] == b["owned_values"]
    assert a["filename"] == b["filename"]
    assert a["filename"].startswith(SHARD_FILE_PREFIX)
    assert a["filename"].endswith(".json")


# ──────────────────────────────────────────────────────────────────────────
# S2 — real-grid cover/disjointness for every recommended axis combination
# ──────────────────────────────────────────────────────────────────────────

REAL_AXIS_COMBOS = [
    "fanouts",
    "hidden_size",
    "dropout",
    "fanouts,hidden_size",
    "fanouts,dropout",
    "hidden_size,dropout",
    "fanouts,hidden_size,dropout",
]


@pytest.mark.parametrize("axes", REAL_AXIS_COMBOS)
def test_s2_real_grid_cover_and_disjointness(axes):
    ss = _real_search_space()
    n_total = len(enumerate_grid(ss))
    assert n_total == 108  # the real 3x3x4x3 grid

    axis_list = axes.split(",")
    n = 1
    for a in axis_list:
        n *= len(ss[a])

    all_owned: list[int] = []
    for k in range(n):
        spec = resolve_shard(ss, axes, f"{k}/{n}")
        assert spec["n_total"] == 108
        # Disjointness against everything seen so far.
        assert not set(spec["owned_indices"]) & set(all_owned)
        all_owned.extend(spec["owned_indices"])
    # Full cover of range(108).
    assert sorted(all_owned) == list(range(108))


def test_s2_recommended_9way_partition_detail():
    """The owner-decided fanouts x hidden_size partition: 9 shards of 12
    trials; shard 8 owns exactly the global indices 96..107 (cross-checks
    specs/34 §2's index decomposition as evidence, not implementation)."""
    ss = _real_search_space()
    for k in range(9):
        spec = resolve_shard(ss, "fanouts,hidden_size", f"{k}/9")
        assert spec["num_shards"] == 9
        assert len(spec["owned_indices"]) == 12
    spec8 = resolve_shard(ss, "fanouts,hidden_size", "8/9")
    assert spec8["owned_indices"] == list(range(96, 108))
    assert spec8["owned_values"] == {"fanouts": [35, 25], "hidden_size": 256}


# ──────────────────────────────────────────────────────────────────────────
# S3 — batch_size-exclusion assertion
# ──────────────────────────────────────────────────────────────────────────

def test_s3_batch_size_axis_is_rejected():
    ss = _real_search_space()
    assert "batch_size" in FORBIDDEN_SHARD_AXES
    with pytest.raises(ValueError) as exc:
        resolve_shard(ss, "fanouts,batch_size", "0/9")
    assert "batch_size" in str(exc.value)
    assert "specs/34" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        resolve_shard(ss, "batch_size", "0/3")
    assert "batch_size" in str(exc.value)
    assert "specs/34" in str(exc.value)


def test_s3_exclusion_is_membership_not_positional():
    """The guard must key on the axis NAME, not the axis's position in
    today's grid: a synthetic grid with batch_size in a different position
    still trips it, and a same-position alien axis does not."""
    ss = {"alpha": [1, 2], "batch_size": [8, 16], "beta": ["x", "y"]}
    with pytest.raises(ValueError, match="batch_size"):
        resolve_shard(ss, "batch_size", "0/2")
    # An allowed axis in the grid position batch_size occupies in the real
    # grid (last) must NOT be rejected.
    spec = resolve_shard(ss, "beta", "0/2")
    assert spec["owned_values"] == {"beta": "x"}


# ──────────────────────────────────────────────────────────────────────────
# S4 — unknown-axis-name assertion
# ──────────────────────────────────────────────────────────────────────────

def test_s4_unknown_axis_fails_loudly():
    ss = _real_search_space()
    with pytest.raises(ValueError) as exc:
        resolve_shard(ss, "fanouts,hiden_size", "0/9")
    msg = str(exc.value)
    assert "hiden_size" in msg  # the offending name
    for valid in sorted(ss):
        assert valid in msg  # the valid axis names are listed


def test_s4_unknown_axis_on_synthetic_grid():
    with pytest.raises(ValueError) as exc:
        resolve_shard(SYNTH, "delta", "0/2")
    msg = str(exc.value)
    assert "delta" in msg
    assert "alpha" in msg and "beta" in msg and "gamma" in msg


# ──────────────────────────────────────────────────────────────────────────
# S5 — K/N validation
# ──────────────────────────────────────────────────────────────────────────

def test_s5_wrong_n_names_the_computed_cardinality():
    """N must equal the product of the chosen axes' cardinalities computed
    from the LIVE search_space — never a literal. fanouts x hidden_size on
    the real grid is 3x3 = 9, so "0/8" must raise naming both numbers."""
    ss = _real_search_space()
    with pytest.raises(ValueError) as exc:
        resolve_shard(ss, "fanouts,hidden_size", "0/8")
    msg = str(exc.value)
    assert "8" in msg   # the supplied N
    assert "9" in msg   # the computed product


def test_s5_out_of_range_k():
    ss = _real_search_space()
    # k == N (one past the last valid shard index).
    with pytest.raises(ValueError):
        resolve_shard(ss, "fanouts,hidden_size", "9/9")
    # Negative k ("-1/9" also fails the \d+/\d+ format — either way raises).
    with pytest.raises(ValueError):
        resolve_shard(ss, "fanouts,hidden_size", "-1/9")


@pytest.mark.parametrize("bad_shard", ["abc", "1/2/3", "4", "4/", "/9", ""])
def test_s5_malformed_kn_string(bad_shard):
    """--shard must be exactly two ints separated by / — anything else
    raises ValueError before any ownership computation."""
    ss = _real_search_space()
    with pytest.raises(ValueError):
        resolve_shard(ss, "fanouts,hidden_size", bad_shard)


def test_s5_duplicate_axes_rejected():
    ss = _real_search_space()
    with pytest.raises(ValueError) as exc:
        resolve_shard(ss, "fanouts,fanouts", "0/9")
    assert "duplicate" in str(exc.value).lower()


# ──────────────────────────────────────────────────────────────────────────
# S6 — select_best dense/sparse parity + lowest-index tie-break
# ──────────────────────────────────────────────────────────────────────────

def _record(trial: int, f1: float) -> dict:
    return {
        "trial": trial,
        "params": {"hidden_size": 64 + trial},
        "best_val_macro_f1": f1,
        "selection_metric_used": "val_composite_f1",
    }


def test_s6_dense_sparse_parity():
    dense = [_record(i, f1) for i, f1 in enumerate(
        [0.3, 0.5, 0.9, 0.4, 0.7, 0.1]
    )]
    winner_dense = select_best(dense)
    assert winner_dense is not None
    assert winner_dense["trial"] == 2

    # Sparse sublist (shard-style: gaps in the global index) that still
    # contains the winner -> the SAME record wins.
    sparse = [dense[0], dense[2], dense[4]]
    winner_sparse = select_best(sparse)
    assert winner_sparse == winner_dense
    assert winner_sparse["trial"] == 2


def test_s6_tie_breaks_to_lowest_global_trial_index():
    records = [
        _record(0, 0.5),
        _record(2, 0.9),
        _record(4, 0.9),  # equal to trial 2 -> trial 2 must win (strict >)
        _record(5, 0.7),
    ]
    best = select_best(records)
    assert best is not None
    assert best["trial"] == 2


def test_s6_input_order_independence():
    """The sorted() inside select_best is load-bearing: promote concatenates
    shard files in glob order, not trial order."""
    records = [
        _record(7, 0.9),
        _record(1, 0.9),  # tie -> global trial 1 must win regardless of order
        _record(3, 0.4),
        _record(0, 0.2),
    ]
    for _ in range(20):
        shuffled = records[:]
        random.shuffle(shuffled)
        best = select_best(shuffled)
        assert best is not None
        assert best["trial"] == 1


def test_s6_empty_and_zero_score():
    assert select_best([]) is None
    # The -1.0 sentinel pin: a 0.0-scoring sole record IS returned.
    sole = _record(0, 0.0)
    assert select_best([sole]) == sole


# ──────────────────────────────────────────────────────────────────────────
# S7 — compute_grid_fingerprint pins (format, canonicalization, subsumption)
# ──────────────────────────────────────────────────────────────────────────

def _real_shaped_model_block() -> dict:
    """The 17 effective post-merge model: keys as of today's tree
    (specs/35 §II.2.3), with representative JSON-native values."""
    return {
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
        "patience": 10,
        "snapshot_interval": 3600,
        "temporal_window_seconds": 86400,
        "weight_decay": 0.0005,
    }


_S7_TRIAL = {"max_epochs": 40, "patience": 20}


def test_s7_fingerprint_format_is_sha256_hex():
    ss = _real_search_space()
    fp = compute_grid_fingerprint(_real_shaped_model_block(), ss, _S7_TRIAL)
    assert fp.startswith("sha256:")
    digest = fp[len("sha256:"):]
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_s7_insertion_order_invariance():
    """Two dicts with identical content but different insertion order must
    hash identically — the sort_keys canonical-JSON pin. Verified for the
    model block AND the search_space."""
    ss = _real_search_space()
    model = _real_shaped_model_block()
    model_reversed = dict(reversed(list(model.items())))
    ss_reversed = dict(reversed(list(ss.items())))
    assert model == model_reversed and list(model) != list(model_reversed)

    base = compute_grid_fingerprint(model, ss, _S7_TRIAL)
    assert compute_grid_fingerprint(model_reversed, ss, _S7_TRIAL) == base
    assert compute_grid_fingerprint(model, ss_reversed, _S7_TRIAL) == base
    assert compute_grid_fingerprint(model_reversed, ss_reversed, _S7_TRIAL) == base


# The 8 keys specs/35 §II.2.3 requires the hash to subsume: the five
# fixed-overwrite keys plus the three composite-metric keys read by
# trainer.py:238-240 (the R6 hole — composite_minority_weight and friends).
S7_REQUIRED_KEY_PERTURBATIONS = {
    # 5 fixed-overwrite keys (03_tune.py fixed: merge)
    "num_layers": 3,
    "aggregator": "pool",
    "learning_rate": 0.002,
    "temporal_window_seconds": 43200,
    "node_state_dim": 16,
    # 3 composite/R6 keys
    "early_stopping_metric": "macro_f1",
    "minority_class_threshold": 6000,
    "composite_minority_weight": 0.6,
}


@pytest.mark.parametrize("key", sorted(S7_REQUIRED_KEY_PERTURBATIONS))
def test_s7_each_required_key_independently_flips_the_hash(key):
    """Changing any ONE of the 8 required keys — holding everything else
    constant — must change the fingerprint (the subsumption assertion:
    hashing the entire effective model: dict covers all of them)."""
    ss = _real_search_space()
    model = _real_shaped_model_block()
    base = compute_grid_fingerprint(model, ss, _S7_TRIAL)

    perturbed = dict(model)
    perturbed[key] = S7_REQUIRED_KEY_PERTURBATIONS[key]
    assert perturbed[key] != model[key]  # the perturbation is real
    assert compute_grid_fingerprint(perturbed, ss, _S7_TRIAL) != base


# ──────────────────────────────────────────────────────────────────────────
# S8 — atomicity of write_json_atomic (.tmp + os.replace)
# ──────────────────────────────────────────────────────────────────────────

def test_s8_happy_path_no_tmp_left(tmp_path):
    target = tmp_path / "out.json"
    obj = {"header": {"schema_version": 1}, "results": [1, 2, 3]}
    write_json_atomic(target, obj)
    assert target.exists()
    with open(target) as f:
        assert json.load(f) == obj
    assert list(tmp_path.glob("*.tmp")) == []


def test_s8_failure_leaves_target_intact_and_no_tmp(tmp_path):
    target = tmp_path / "out.json"
    original = {"kept": True, "results": [42]}
    write_json_atomic(target, original)

    with pytest.raises(TypeError):
        write_json_atomic(target, {"bad": object()})  # not JSON-serializable

    # Original content untouched — the truncate-then-crash failure mode of
    # the old open(path, "w") write is impossible by construction.
    with open(target) as f:
        assert json.load(f) == original
    assert list(tmp_path.glob("*.tmp")) == []


def test_s8_tmp_name_outside_promote_glob(tmp_path):
    """The in-flight temp name must not match promote_best.py's
    tuning_results_shard*.json glob (it must not end in .json)."""
    target = tmp_path / (SHARD_FILE_PREFIX + "_fan35-25_hid256.json")
    write_json_atomic(target, [])
    tmp_name = target.name + ".tmp"
    assert not tmp_name.endswith(".json")


# ──────────────────────────────────────────────────────────────────────────
# S9 — verbatim shard-filename pin (specs/34 §3.5's exact example)
# ──────────────────────────────────────────────────────────────────────────

def test_s9_filename_pin_verbatim():
    """Shard 8/9 of the recommended fanouts x hidden_size partition —
    (fanouts=[35,25], hidden_size=256) — must produce EXACTLY the filename
    specs/34 §3.5 / specs/35 §II.2.4 specify. Full-string equality, not
    prefix/suffix matching (test_s1_synthetic_cli_axis_order_is_normalized
    only checks prefix/suffix)."""
    ss = _real_search_space()
    spec = resolve_shard(ss, "fanouts,hidden_size", "8/9")
    assert spec["filename"] == "tuning_results_shard_fan35-25_hid256.json"
    assert spec["owned_values"] == {"fanouts": [35, 25], "hidden_size": 256}


def test_s9_cli_axis_order_normalizes_to_same_verbatim_filename():
    """--shard-axes hidden_size,fanouts is the SAME partition: same verbatim
    filename and same owned_indices as fanouts,hidden_size."""
    ss = _real_search_space()
    a = resolve_shard(ss, "fanouts,hidden_size", "8/9")
    b = resolve_shard(ss, "hidden_size,fanouts", "8/9")
    assert b["filename"] == "tuning_results_shard_fan35-25_hid256.json"
    assert a["owned_indices"] == b["owned_indices"]


# ──────────────────────────────────────────────────────────────────────────
# S10 — stdlib-only import contract of src/model/selection.py
# ──────────────────────────────────────────────────────────────────────────

def test_s10_selection_module_imports_are_stdlib_only():
    """src/model/selection.py must import ONLY stdlib modules — no torch/
    dgl/numpy/pandas/yaml and no src.* imports — so promote_best.py stays
    runnable on a login node with no scientific stack installed
    (specs/34 §4.3, specs/35 §II.1). AST scan, itself stdlib-only."""
    source = (REPO_ROOT / "src" / "model" / "selection.py").read_text()
    tree = ast.parse(source)

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # No relative imports (level > 0 would reach back into src.*).
            assert node.level == 0, "relative import in selection.py"
            assert node.module is not None
            imported.add(node.module.split(".")[0])

    forbidden = {
        "torch", "dgl", "numpy", "pandas", "yaml", "scipy", "sklearn",
        "shap", "networkx", "matplotlib", "seaborn", "tqdm", "src",
    }
    assert not imported & forbidden, (
        f"selection.py imports non-stdlib module(s): "
        f"{sorted(imported & forbidden)}"
    )
    # Positive check: every imported top-level module is stdlib.
    non_stdlib = imported - set(sys.stdlib_module_names)
    assert not non_stdlib, (
        f"selection.py imports outside the stdlib: {sorted(non_stdlib)}"
    )


# ──────────────────────────────────────────────────────────────────────────
# S11 (tuner-side) — full-list-iteration source pin on src/model/tuner.py
# ──────────────────────────────────────────────────────────────────────────

def test_s11_tuner_iterates_full_configs_list():
    """Source pin for the specs/34 §3.3 invariant: run() enumerates the
    FULL configs list (never truncated/sliced/pre-filtered), best_idx is
    gone, and _configs delegates enumeration to enumerate_grid. A text
    scan is a weak pin — stage 5 must also verify by reading the diff —
    but it catches the obvious regressions (a filtered-list rewrite)."""
    src = (REPO_ROOT / "src" / "model" / "tuner.py").read_text()
    assert "for i, trial_params in enumerate(configs)" in src
    assert "best_idx" not in src
    assert "[c for c in configs" not in src  # no pre-loop filtering
    assert "configs[:" not in src            # no slicing
    assert "enumerate_grid(self.search_space)" in src


def test_s11_tune_script_shard_flags_and_pairing_guard():
    """scripts/03_tune.py exposes --shard-axes/--shard and enforces the
    both-or-neither pairing guard before any graph/feature-store I/O."""
    src = (REPO_ROOT / "scripts" / "03_tune.py").read_text()
    assert '"--shard-axes"' in src
    assert '"--shard"' in src
    assert "resolve_shard" in src
    assert "compute_grid_fingerprint" in src
    assert "(shard_axes is None) != (shard is None)" in src
