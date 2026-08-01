"""
Tests for the Phase-3 S1 seeding fix (specs/36 §2, specs/37 §2(a),
specs/38 §2.5): src/model/tuner.py must seed torch.manual_seed(seed + i)
immediately before EdgeAwareGraphSAGE construction in the per-trial loop,
not rely on Trainer.train's later seed call, which seeds AFTER the model
(and its randomly-initialized weights) already exist.

Before this fix, `scripts/03_tune.py` never called `torch.manual_seed`/
`np.random.seed`, so a trial's initial weights were drawn from whatever
ambient RNG state existed at that point in the process -- a function of
process entropy and of every RNG draw every preceding trial in the same
process happened to consume (specs/36 §2.2, four fresh-process
measurements, SHA-256 of state_dict). This made a sharded run's trial i
start from different weights than the same trial i in a single-job run,
purely because of unrelated prior-trial RNG consumption.

Constraint (specs/36 §6.4, standing rule, restated here because it governs
every test in this file): assert on state_dict SHA-256 checksums, never on
end-to-end metric agreement; never use a small-scale (<=118k-edge) run as a
reproducibility control. specs/36 §4.2 measured that a small-scale check
reports "reproducible" and is wrong by ~17x at production scale -- this
project's own standing rule this file must not violate.
"""

import hashlib
import inspect
import subprocess
import sys
from pathlib import Path

import torch

from src.model.sage_model import EdgeAwareGraphSAGE
from src.model.tuner import HyperparameterTuner

REPO_ROOT = Path(__file__).resolve().parent.parent

# Minimal, cheap-to-construct dimensions -- pure nn.Module.__init__, no
# GPU/graph/feature-store dependency (specs/33 §IV.3's artifact-free-gate
# sanction, already used elsewhere in this test suite).
_CTOR_KWARGS = dict(
    node_in_dim=15, edge_in_dim=32, hidden_size=64,
    num_classes=4, num_layers=2, dropout=0.2, aggregator="mean",
)


def _state_dict_sha256(model: torch.nn.Module) -> str:
    """SHA-256 over every state_dict tensor, in sorted key order (so the
    checksum does not depend on nn.Module's internal parameter-registration
    order, only on the actual weight values)."""
    h = hashlib.sha256()
    for key in sorted(model.state_dict()):
        h.update(key.encode("utf-8"))
        h.update(model.state_dict()[key].cpu().numpy().tobytes())
    return h.hexdigest()


def test_seeded_construction_is_bit_identical_across_processes_or_calls():
    """Formalizes specs/36 §2.2's manual probe as a permanent regression
    test: two constructions, each preceded by the SAME torch.manual_seed
    call, produce bit-identical weights. This is the property the S1 fix
    relies on -- seeding immediately before construction makes trial i's
    initial weights a pure function of the seed, not of ambient RNG state."""
    torch.manual_seed(123)
    m1 = EdgeAwareGraphSAGE(**_CTOR_KWARGS)
    torch.manual_seed(123)
    m2 = EdgeAwareGraphSAGE(**_CTOR_KWARGS)
    assert _state_dict_sha256(m1) == _state_dict_sha256(m2)


def test_seeded_construction_is_bit_identical_across_fresh_processes():
    """specs/36 §2.2's OWN verification method, formalized as a permanent
    regression test: TWO SEPARATE, FRESH PYTHON PROCESSES (not two calls
    within one process -- see the in-process test above, and contrast with
    specs/36 §2.2's proc-3/proc-4 row, where `torch.initial_seed()` at
    process start differed between the two processes yet
    `manual_seed(90)` still produced the identical checksum
    `e4ab902faae5d571` in both). An in-process reseed cannot catch a
    regression where initialization also depends on ambient
    process-entropy state alongside the seed -- only a real process
    boundary exercises that. specs/36 §6.4 explicitly sanctions a
    weight-checksum assertion as "scale-independent and cheap"; this is
    Appendix A's `init_check` probe, made permanent."""
    snippet = (
        "import hashlib, sys, torch\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from src.model.sage_model import EdgeAwareGraphSAGE\n"
        "torch.manual_seed(123)\n"
        "m = EdgeAwareGraphSAGE(node_in_dim=15, edge_in_dim=32, hidden_size=64, "
        "num_classes=4, num_layers=2, dropout=0.2, aggregator='mean')\n"
        "h = hashlib.sha256()\n"
        "for key in sorted(m.state_dict()):\n"
        "    h.update(key.encode('utf-8'))\n"
        "    h.update(m.state_dict()[key].cpu().numpy().tobytes())\n"
        "print(h.hexdigest())\n"
    )
    outputs = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        outputs.append(proc.stdout.strip())
    assert outputs[0] != ""
    assert outputs[0] == outputs[1]


def test_unseeded_construction_is_not_reliably_reproducible():
    """Negative control: WITHOUT a reseed between constructions, two
    back-to-back EdgeAwareGraphSAGE constructions consume the ongoing RNG
    stream and do NOT produce identical weights -- this is exactly the
    pre-fix defect (specs/36 §2.2's "within-process" row), pinned here so
    the positive test above cannot be trivially satisfied by
    EdgeAwareGraphSAGE always initializing to the same constant regardless
    of RNG state."""
    torch.manual_seed(999)
    m1 = EdgeAwareGraphSAGE(**_CTOR_KWARGS)
    m2 = EdgeAwareGraphSAGE(**_CTOR_KWARGS)  # no reseed in between
    assert _state_dict_sha256(m1) != _state_dict_sha256(m2)


def test_trial_seed_is_independent_of_prior_rng_consumption():
    """The specific property specs/36 §2.3/§5 identifies as what makes a
    sharded run's trial i comparable to the same trial i in a single-job
    run: reseeding with seed_i immediately before construction reproduces
    the SAME weights regardless of how many RNG draws happened earlier in
    the process (simulating a differently-laid-out prior trial history --
    different dropout, different batch counts, unrelated tensor ops)."""
    seed_i = 777
    torch.manual_seed(seed_i)
    baseline = EdgeAwareGraphSAGE(**_CTOR_KWARGS)
    baseline_hash = _state_dict_sha256(baseline)

    # Simulate a differently-laid-out prior trial history: consume an
    # arbitrary, unrelated number of RNG draws before reseeding.
    torch.manual_seed(1)
    _ = torch.randn(4096, 4096)
    _ = EdgeAwareGraphSAGE(**{**_CTOR_KWARGS, "dropout": 0.4})
    _ = torch.randn(17)

    torch.manual_seed(seed_i)
    replay = EdgeAwareGraphSAGE(**_CTOR_KWARGS)
    assert _state_dict_sha256(replay) == baseline_hash


def test_tuner_seeds_before_model_construction():
    """Wiring/regression pin: the seed call precedes construction in
    tuner.py's per-trial loop -- guards against a future refactor silently
    reordering the two statements back to the pre-fix order. Secondary to
    the two outcome-based tests above (CLAUDE.md's executed-verification
    preference for behavior over structural pins), not a substitute for
    them."""
    src = inspect.getsource(HyperparameterTuner.run)
    seed_idx = src.index("torch.manual_seed(seed + i)")
    construct_idx = src.index("model = EdgeAwareGraphSAGE(")
    assert seed_idx < construct_idx


def test_tuner_seed_uses_the_same_per_trial_seed_passed_to_trainer():
    """The seed value used at construction (`seed + i`) must be the exact
    same expression already passed to Trainer.train at the existing call
    site -- specs/38 §2.5 requires reusing `seed + i`, not introducing a
    second, independent seed source that could silently drift from it."""
    src = inspect.getsource(HyperparameterTuner.run)
    assert src.count("seed + i") >= 2  # the new construction-site seed...
    assert "trainer.train(" in src
    # ...and the pre-existing Trainer.train(..., seed=seed + i, ...) call.
    assert "seed=seed + i" in src
