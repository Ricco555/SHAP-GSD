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
  S12 — build_shard_header: single-source-of-truth 15-key header schema.
  S13 — select_best_tie_aware / validate_tie_break_axes: tie-band
       selection + config-driven axis-priority tie-break (specs/37 §3,
       specs/38 §1.2).

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
    SHARD_SCHEMA_VERSION,
    build_shard_header,
    compute_grid_fingerprint,
    enumerate_grid,
    resolve_shard,
    select_best,
    select_best_tie_aware,
    validate_tie_break_axes,
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
        names: set[str] = set()    # filenames seen so far in this partition
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

            # Filename uniqueness: no two shards of the SAME partition may
            # resolve to the same on-disk filename (finding #1's failure
            # mode -- a colliding filename means one shard's completed
            # trials get silently overwritten by write_json_atomic's
            # unconditional os.replace).
            assert spec["filename"] not in names, (
                f"axes={axes}, k={k}: filename {spec['filename']!r} "
                "collides with an earlier shard of the same partition"
            )
            names.add(spec["filename"])

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
    names: set[str] = set()
    for k in range(n):
        spec = resolve_shard(ss, axes, f"{k}/{n}")
        assert spec["n_total"] == 108
        # Disjointness against everything seen so far.
        assert not set(spec["owned_indices"]) & set(all_owned)
        all_owned.extend(spec["owned_indices"])
        # Filename uniqueness within this partition (finding #1).
        assert spec["filename"] not in names, (
            f"axes={axes}, k={k}: filename {spec['filename']!r} collides "
            "with an earlier shard of the same partition"
        )
        names.add(spec["filename"])
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


def test_s9_filename_is_injective_on_signed_values():
    """Distinct signed axis values (1 vs -1, 0.1 vs -0.1) must not collapse
    to the same filename slug: the old _slug() stripped a leading '-' as
    generic punctuation, so two shards could resolve to the IDENTICAL
    filename and silently overwrite each other's results via
    write_json_atomic's unconditional os.replace. Today's real grid never
    exercises this (all-positive axis values), but resolve_shard must stay
    correct for any future signed axis."""
    grid = {"delta": [1, -1], "eps": [0.1, -0.1]}
    n = len(grid["delta"]) * len(grid["eps"])
    filenames = [
        resolve_shard(grid, "delta,eps", f"{k}/{n}")["filename"]
        for k in range(n)
    ]
    assert len(set(filenames)) == n, (
        f"filenames collided for distinct signed axis-value combinations: "
        f"{filenames}"
    )


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
    # Finding #6 (header-schema extraction): tuner.py must call the shared
    # build_shard_header() helper rather than hand-maintain its own copy of
    # the 12-key header dict literal. A regression here would silently
    # re-introduce the drift risk build_shard_header exists to close (a
    # future header field added to one copy and not the other).
    assert "build_shard_header(shard, selection_metric_used)" in src
    assert '"schema_version": SHARD_SCHEMA_VERSION' not in src


def test_s11_tune_script_shard_flags_and_pairing_guard():
    """scripts/03_tune.py exposes --shard-axes/--shard and enforces the
    both-or-neither pairing guard before any graph/feature-store I/O."""
    src = (REPO_ROOT / "scripts" / "03_tune.py").read_text()
    assert '"--shard-axes"' in src
    assert '"--shard"' in src
    assert "resolve_shard" in src
    assert "compute_grid_fingerprint" in src
    assert "(shard_axes is None) != (shard is None)" in src


def test_s11_tune_script_attaches_selection_fields_to_shard_spec():
    """specs/38 §3.3: scripts/03_tune.py must attach tie_band_pp/
    tie_break_axes/search_space onto shard_spec before HyperparameterTuner
    ever sees it -- build_shard_header (src/model/selection.py) hard-
    subscripts all three off the shard dict it is given, so a dropped
    attachment here would only surface at runtime in shard mode (GPU/HPC
    path, not exercised by any test) as a bare KeyError. Pins the supply
    side of the same seam test_s13_result_never_depends_on_elapsed_s and
    the P11/P12 promote_best.py tests pin the consumption side of."""
    src = (REPO_ROOT / "scripts" / "03_tune.py").read_text()
    for field in ("tie_band_pp", "tie_break_axes", "search_space"):
        assert f'shard_spec["{field}"]' in src


# ──────────────────────────────────────────────────────────────────────────
# S12 — build_shard_header: single-source-of-truth header schema
# (extracted from HyperparameterTuner.run()'s inline dict literal into
# src/model/selection.py, specs/35 §III.5; findings #4-6). Previously this
# helper had no DIRECT test — only indirect coverage via
# tests/test_promote_best.py's fixture, which always passes explicit
# pbs_jobid/started_at and therefore never exercises the os.environ /
# datetime.now() default branches actually used by the real tuner.py write
# path.
# ──────────────────────────────────────────────────────────────────────────

def _dummy_shard() -> dict:
    """A minimal, valid resolve_shard()-shaped dict plus grid_fingerprint
    and the three selection-policy fields, exactly the shape
    build_shard_header's docstring requires."""
    ss = _real_search_space()
    spec = resolve_shard(ss, "fanouts,hidden_size", "8/9")
    spec["grid_fingerprint"] = "deadbeef" * 8
    spec["tie_band_pp"] = 2.0
    spec["tie_break_axes"] = [{"axis": "batch_size", "cheapest": "last"}]
    spec["search_space"] = ss
    return spec


def test_s12_header_has_exact_15_key_schema():
    """The header dict written to disk must have EXACTLY the 15 keys the
    shard-file format (specs/35 §III.5, specs/38 §1.3) and
    validate_shards() depend on — no more, no fewer. A drifted key set is
    exactly the failure mode this extraction exists to prevent (a
    hand-maintained second copy silently gaining/losing a field)."""
    shard = _dummy_shard()
    header = build_shard_header(
        shard, "val_composite_f1",
        pbs_jobid="123.testhost", started_at="2026-08-01T09:00:00+02:00",
    )
    assert set(header) == {
        "schema_version", "shard_index", "num_shards", "n_total",
        "partition_scheme", "shard_axes", "owned_values", "owned_indices",
        "selection_metric_used", "grid_fingerprint", "tie_band_pp",
        "tie_break_axes", "search_space", "pbs_jobid", "started_at",
    }
    assert header["schema_version"] == SHARD_SCHEMA_VERSION
    assert header["shard_index"] == shard["shard_index"]
    assert header["num_shards"] == shard["num_shards"]
    assert header["n_total"] == shard["n_total"]
    assert header["partition_scheme"] == shard["partition_scheme"]
    assert header["shard_axes"] == shard["shard_axes"]
    assert header["owned_values"] == shard["owned_values"]
    assert header["owned_indices"] == shard["owned_indices"]
    assert header["selection_metric_used"] == "val_composite_f1"
    assert header["grid_fingerprint"] == shard["grid_fingerprint"]
    assert header["pbs_jobid"] == "123.testhost"
    assert header["started_at"] == "2026-08-01T09:00:00+02:00"
    assert header["tie_band_pp"] == shard["tie_band_pp"]
    assert header["tie_break_axes"] == shard["tie_break_axes"]
    assert header["search_space"] == shard["search_space"]


def test_s12_explicit_overrides_pass_through_unchanged():
    """pbs_jobid/started_at, when given, are used verbatim — no env/clock
    lookup happens even if PBS_JOBID is set in the test environment."""
    shard = _dummy_shard()
    header = build_shard_header(
        shard, "macro_f1", pbs_jobid="999.supek", started_at="frozen-value",
    )
    assert header["pbs_jobid"] == "999.supek"
    assert header["started_at"] == "frozen-value"


def test_s12_default_pbs_jobid_reads_environment(monkeypatch):
    """Absent an explicit pbs_jobid, the header falls back to
    os.environ['PBS_JOBID'] -- this is the real tuner.py write path's
    behavior, previously only reachable by actually running under PBS."""
    shard = _dummy_shard()

    monkeypatch.setenv("PBS_JOBID", "42.supek.example")
    header = build_shard_header(shard, "macro_f1")
    assert header["pbs_jobid"] == "42.supek.example"

    monkeypatch.delenv("PBS_JOBID", raising=False)
    header = build_shard_header(shard, "macro_f1")
    assert header["pbs_jobid"] == "none"


def test_s12_default_started_at_is_isoformat_now():
    """Absent an explicit started_at, the header stamps a real
    datetime.now().astimezone().isoformat() string (parseable, timezone-
    aware, close to 'now') rather than a placeholder."""
    from datetime import datetime

    shard = _dummy_shard()
    before = datetime.now().astimezone()
    header = build_shard_header(shard, "macro_f1")
    after = datetime.now().astimezone()

    parsed = datetime.fromisoformat(header["started_at"])
    assert parsed.tzinfo is not None
    assert before <= parsed <= after


def test_shard_schema_version_bumped_to_2():
    """specs/38 §8.2: SHARD_SCHEMA_VERSION must be 2 (was 1 pre-fix), and
    every header build_shard_header produces stamps that version. This
    pins the VERSION NUMBER specifically; test_s12_header_has_exact_15_key_schema
    (above) pins the header's key SET -- the two are deliberately separate
    assertions so a regression in either is caught independently."""
    assert SHARD_SCHEMA_VERSION == 2
    header = build_shard_header(_dummy_shard(), "val_composite_f1")
    assert header["schema_version"] == 2
    assert len(header) == 15


# ──────────────────────────────────────────────────────────────────────────
# S13 — select_best_tie_aware / validate_tie_break_axes: tie-band selection
# + config-driven axis-priority tie-break (specs/36 D2, specs/37 §3,
# specs/38 §1.1/§1.2). validate_tie_break_axes is exercised indirectly
# through select_best_tie_aware (which calls it first, per specs/38 §0.2's
# shared-validator decision) rather than directly -- its only independent
# behavior (structural axis/cheapest validation) is fully covered by the
# rejection tests below, and it has no other caller-visible contract.
# ──────────────────────────────────────────────────────────────────────────

def _tb_record(trial: int, f1: float, **params) -> dict:
    """A trial record whose "params" dict carries REAL per-axis values (not
    _record()'s placeholder {"hidden_size": 64 + trial}), so a tie-break
    axis lookup can actually run against it."""
    return {
        "trial": trial,
        "params": params,
        "best_val_macro_f1": f1,
        "selection_metric_used": "val_composite_f1",
        "elapsed_s": 100.0 + trial,
    }


def test_s13_zero_band_matches_legacy_select_best():
    """tie_band_pp=0.0, tie_break_axes=[] degenerates to plain select_best,
    byte-for-byte, on any fixture with a UNIQUE top score (specs/38 §1.2's
    docstring guarantee) -- the primary regression property for the new
    mechanism (specs/37 §3.1)."""
    fixtures = [
        [_record(i, f1) for i, f1 in enumerate([0.3, 0.5, 0.9, 0.4, 0.7, 0.1])],
        [_record(7, 0.9), _record(1, 0.3), _record(3, 0.4), _record(0, 0.2)],
    ]
    for fixture in fixtures:
        expected = select_best(fixture)
        result = select_best_tie_aware(fixture, 0.0, [], {})
        assert result["winner"] == expected
        assert result["argmax"] == expected
        assert result["tie_set_size"] == 1
        assert result["tie_set_trials"] == [expected["trial"]]


def test_s13_zero_band_with_exact_score_tie():
    """At tie_band_pp=0.0, `winner == argmax == select_best(results)` still
    holds when two-plus records share the exact max score, but
    tie_set_size is NOT 1 in that case -- it is the number of tied
    records. Reuses S6's own tied-score fixture so this is pinned against
    the exact scenario select_best's own tie-break test exercises."""
    records = [
        _record(0, 0.5),
        _record(2, 0.9),
        _record(4, 0.9),  # exact tie with trial 2 at the top
        _record(5, 0.7),
    ]
    expected = select_best(records)
    result = select_best_tie_aware(records, 0.0, [], {})
    assert result["winner"] == expected
    assert result["winner"]["trial"] == 2  # lowest of the tied indices
    assert result["tie_set_trials"] == [2, 4]
    assert result["tie_set_size"] == 2  # NOT 1 -- do not conflate with the above


def test_s13_tie_band_boundary_is_inclusive():
    """Pins the `<=` (not `<`) band-membership semantics (specs/37 §3.3):
    a candidate exactly `tie_band_pp` below the max is included.

    tie_band_pp=6.25 (band=0.0625) with max_score=1.0 is deliberately
    chosen so the boundary score (0.9375) and the band width are both
    EXACTLY representable in binary floating point -- with an ordinary
    decimal band like 2.0 (0.02), `1.0 - 0.98` evaluates to
    0.020000000000000018 in IEEE 754 double precision, which is NOT
    `<= 0.02`, making an exact-boundary assertion flaky by construction
    rather than a real test of the `<=` semantics.
    """
    records = [
        _record(0, 1.00),    # max
        _record(1, 0.9375),  # exactly at the 6.25pp boundary -> included
        _record(2, 0.94),    # just inside -> included
        _record(3, 0.93),    # just outside -> excluded
    ]
    result = select_best_tie_aware(records, 6.25, [], {})
    assert set(result["tie_set_trials"]) == {0, 1, 2}
    assert 3 not in result["tie_set_trials"]


def test_s13_tie_break_prefers_configured_cheapest_axis():
    """Two in-band trials differing only on batch_size; batch_size=2048
    ranks cheaper by the configured ordering and wins even though it is
    not the argmax."""
    ss = {"batch_size": [512, 1024, 2048]}
    records = [
        _tb_record(0, 0.90, batch_size=512),   # argmax
        _tb_record(1, 0.89, batch_size=2048),  # in-band, configured-cheaper
    ]
    result = select_best_tie_aware(
        records, 2.0, [{"axis": "batch_size", "cheapest": "last"}], ss
    )
    assert result["winner"]["trial"] == 1
    assert result["argmax"]["trial"] == 0


def test_s13_tie_break_cascades_through_axes_in_order():
    """Three in-band trials tying on the FIRST configured axis but
    differing on the second must be decided by the second axis -- proves
    the cascade, not just first-axis-only resolution."""
    ss = {"batch_size": [512, 1024, 2048], "hidden_size": [64, 128, 256]}
    axes = [
        {"axis": "batch_size", "cheapest": "last"},
        {"axis": "hidden_size", "cheapest": "first"},
    ]
    records = [
        _tb_record(0, 0.90, batch_size=2048, hidden_size=256),   # argmax
        _tb_record(1, 0.89, batch_size=2048, hidden_size=64),    # ties on
        #                                                          batch_size,
        #                                                          wins on
        #                                                          hidden_size
        _tb_record(2, 0.885, batch_size=1024, hidden_size=64),   # loses on
        #                                                          batch_size
        #                                                          alone
    ]
    result = select_best_tie_aware(records, 2.0, axes, ss)
    assert result["winner"]["trial"] == 1


def test_s13_single_clear_winner_outside_any_band():
    """The production-shaped case: at a real, non-degenerate band
    (tie_band_pp=2.0, matching the shipped configs/tuning_grid.yaml
    default), exactly one trial is inside it. This exercises the
    tie_break_axes loop over a SINGLETON candidate list (the immediate
    `len(candidates) == 1` break) and pins that the cheaper-but-losing
    trials (outside the band) are correctly never even considered --
    unlike every other S13 tie-break test, which always has >= 2 in-band
    candidates."""
    ss = {"batch_size": [512, 1024, 2048], "hidden_size": [64, 128]}
    axes = [
        {"axis": "batch_size", "cheapest": "last"},
        {"axis": "hidden_size", "cheapest": "first"},
    ]
    records = [
        _tb_record(0, 0.90, batch_size=512, hidden_size=128),   # sole winner
        _tb_record(1, 0.50, batch_size=2048, hidden_size=64),   # cheaper by
        #                                                         config, but
        #                                                         WAY outside
        #                                                         the band
        _tb_record(2, 0.40, batch_size=2048, hidden_size=64),
    ]
    result = select_best_tie_aware(records, 2.0, axes, ss)
    assert result["winner"]["trial"] == 0
    assert result["argmax"]["trial"] == 0
    assert result["tie_set_trials"] == [0]
    assert result["tie_set_size"] == 1


def test_s13_residual_tie_falls_back_to_lowest_trial_index():
    """In-band trials identical on every configured tie_break_axes entry
    fall back to the lowest global trial index -- select_best's own
    tie-break direction."""
    ss = {"batch_size": [512, 1024]}
    axes = [{"axis": "batch_size", "cheapest": "last"}]
    records = [
        _tb_record(5, 0.90, batch_size=1024),
        _tb_record(2, 0.89, batch_size=1024),  # identical on the only axis
    ]
    result = select_best_tie_aware(records, 2.0, axes, ss)
    assert result["winner"]["trial"] == 2


def test_s13_unknown_tie_break_axis_rejected():
    with pytest.raises(ValueError) as excinfo:
        select_best_tie_aware(
            [_tb_record(0, 0.9, batch_size=512)], 2.0,
            [{"axis": "nope", "cheapest": "first"}], {"batch_size": [512]},
        )
    assert "nope" in str(excinfo.value)


def test_s13_invalid_cheapest_value_rejected():
    with pytest.raises(ValueError) as excinfo:
        select_best_tie_aware(
            [_tb_record(0, 0.9, batch_size=512)], 2.0,
            [{"axis": "batch_size", "cheapest": "middle"}], {"batch_size": [512]},
        )
    assert "middle" in str(excinfo.value)


def test_s13_tie_set_member_missing_axis_key_raises():
    """A tie-set record whose params dict is missing a tie_break_axes-named
    key must raise, naming the trial and the axis -- never silently
    skipped or treated as maximally cheap/expensive."""
    records = [
        _tb_record(0, 0.90, batch_size=512),
        _tb_record(1, 0.89, hidden_size=64),  # no "batch_size" key at all
    ]
    with pytest.raises(ValueError) as excinfo:
        select_best_tie_aware(
            records, 2.0, [{"axis": "batch_size", "cheapest": "last"}],
            {"batch_size": [512, 1024]},
        )
    msg = str(excinfo.value)
    assert "trial 1" in msg
    assert "batch_size" in msg


def test_s13_tie_set_member_out_of_search_space_value_raises():
    """A tie-set record's params value that is not a member of that axis's
    search_space list must raise the same way -- naming the trial, the
    axis, and the offending value."""
    records = [
        _tb_record(0, 0.90, batch_size=512),
        _tb_record(1, 0.89, batch_size=9999),  # not in search_space
    ]
    with pytest.raises(ValueError) as excinfo:
        select_best_tie_aware(
            records, 2.0, [{"axis": "batch_size", "cheapest": "last"}],
            {"batch_size": [512, 1024]},
        )
    msg = str(excinfo.value)
    assert "trial 1" in msg
    assert "9999" in msg


def test_s13_negative_tie_band_pp_rejected():
    with pytest.raises(ValueError):
        select_best_tie_aware([_record(0, 0.9)], -1.0, [], {})


def test_s13_empty_results_raises():
    """specs/38 §0.1: select_best_tie_aware itself raises ValueError on an
    empty results list -- the production call sites (tuner.py,
    promote_best.py) guard this before calling it, but the function's own
    contract must hold for any other caller."""
    with pytest.raises(ValueError):
        select_best_tie_aware([], 2.0, [], {})


def test_s13_dense_sparse_parity():
    """Mirrors S6's dense/sparse parity: a dense (single-job) result list
    and an equivalent sparse/shuffled (merged-shards) sublist produce the
    identical winner/argmax/tie_set_trials -- the direct cross-shard-safety
    property specs/37 §3.2's revision exists for."""
    dense = [_record(i, f1) for i, f1 in enumerate(
        [0.3, 0.5, 0.9, 0.4, 0.89, 0.1]
    )]
    result_dense = select_best_tie_aware(dense, 2.0, [], {})

    sparse = [dense[0], dense[2], dense[4], dense[5]]
    result_sparse = select_best_tie_aware(sparse, 2.0, [], {})

    assert result_sparse["winner"] == result_dense["winner"]
    assert result_sparse["argmax"] == result_dense["argmax"]
    assert result_sparse["tie_set_trials"] == result_dense["tie_set_trials"]


def test_s13_tie_set_trials_sorted_and_complete():
    records = [
        _record(5, 0.90),
        _record(1, 0.89),
        _record(9, 0.50),
        _record(3, 0.881),
    ]
    result = select_best_tie_aware(records, 2.0, [], {})
    assert result["tie_set_trials"] == [1, 3, 5]


def test_s13_result_never_depends_on_elapsed_s():
    """The direct regression test for the flaw specs/37 §3.2 revised away
    from: two in-band candidates with wildly different elapsed_s but
    identical tie_break_axes values must be decided by the trial-index
    fallback, and a fixture that omits elapsed_s entirely must still
    produce a valid winner -- proving the field is never read at all."""
    ss = {"batch_size": [512, 1024]}
    axes = [{"axis": "batch_size", "cheapest": "last"}]  # ties on this axis
    records = [
        {"trial": 5, "params": {"batch_size": 1024}, "best_val_macro_f1": 0.90,
         "selection_metric_used": "val_composite_f1", "elapsed_s": 1.0},
        {"trial": 2, "params": {"batch_size": 1024}, "best_val_macro_f1": 0.89,
         "selection_metric_used": "val_composite_f1", "elapsed_s": 99999.0},
    ]
    result = select_best_tie_aware(records, 2.0, axes, ss)
    assert result["winner"]["trial"] == 2  # trial-index fallback, not elapsed_s

    records_no_elapsed = [
        {"trial": 5, "params": {"batch_size": 1024}, "best_val_macro_f1": 0.90,
         "selection_metric_used": "val_composite_f1"},
        {"trial": 2, "params": {"batch_size": 1024}, "best_val_macro_f1": 0.89,
         "selection_metric_used": "val_composite_f1"},
    ]
    result2 = select_best_tie_aware(records_no_elapsed, 2.0, axes, ss)
    assert result2["winner"]["trial"] == 2
