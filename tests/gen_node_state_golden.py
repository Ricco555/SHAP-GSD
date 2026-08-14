"""Generate the pre-change golden fixture for node-state `recent_window` mode.

specs/60 §6.1. This module is NOT a pytest test module — it deliberately
exposes no ``test_*`` names, so pytest collects nothing from it. Run it as::

    python -m tests.gen_node_state_golden

It writes ``tests/fixtures/node_state_golden_recent_window.npz``, which pins
`novelty_mode="recent_window"` (the published, peer-reviewed behaviour) to the
outputs of the code as it stood **before** ``model.novelty_mode`` existed.

PROVENANCE IS THE WHOLE POINT. A golden regenerated *after*
``src/model/node_state.py`` is edited turns the T1 identity test into a
self-comparison and silently proves nothing. Two guards exist:

  1. The fixture records ``source_sha256`` (sha256 of ``src/model/node_state.py``
     as read at generation time) and ``git_head`` (``git rev-parse HEAD``).
     A reviewer verifies provenance with::

         git show <git_head>:src/model/node_state.py | sha256sum

     and compares against ``source_sha256``. This works on a dirty working
     tree, unlike a commit-order check.
  2. The manager is constructed INLINE here rather than via
     ``tests/test_node_state.py::_build_nsm`` — that helper gains a
     ``novelty_mode`` parameter and a ``set_train_nodes`` call, so a golden
     derived from it would drift with the very code it must be independent of.

The inline construction reproduces the current ``_build_nsm`` order exactly:
``NodeStateManager(window_seconds=1.0, snapshot_interval=5)`` →
``build_hourly_baselines(src[:2], dst[:2], ...)`` → ``build_snapshots(...)`` →
``set_is_internal(np.zeros(4))``. No ``set_train_nodes``, no ``novelty_mode``.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.node_state import NodeStateManager  # noqa: E402

# Same five edges as tests/test_node_state.py::_PARITY_EDGES. Duplicated
# verbatim (not imported) so this generator does not depend on a test module
# stage 3/4 is editing.
_PARITY_EDGES: list[dict] = [
    {"ts": 100, "src": 0, "dst": 1, "in_bytes": 0,   "out_bytes": 100, "dst_port": 80},
    {"ts": 200, "src": 0, "dst": 2, "in_bytes": 0,   "out_bytes": 200, "dst_port": 443},
    {"ts": 300, "src": 0, "dst": 1, "in_bytes": 0,   "out_bytes": 150, "dst_port": 80},
    {"ts": 400, "src": 3, "dst": 0, "in_bytes": 500, "out_bytes": 0,   "dst_port": 9999},
    {"ts": 500, "src": 0, "dst": 2, "in_bytes": 0,   "out_bytes": 300, "dst_port": 22},
]

# Query times: the five from test_batch_states_oracle_parity plus three more
# (a pre-first-edge time, a mid-capture time, a far-future time).
QUERY_TIMES: tuple[float, ...] = (
    -500.0, 50.0, 350.0, 600.0, 5000.0, 100.0, 1100.0, 100000.0,
)
NODE_IDS: tuple[int, ...] = (0, 1, 2, 3)

FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "node_state_golden_recent_window.npz"
SOURCE_PATH = REPO_ROOT / "src" / "model" / "node_state.py"


def build_reference_manager(
    novelty_mode: str | None = None,
) -> NodeStateManager:
    """Construct the reference manager inline (see module docstring).

    Args:
        novelty_mode: ``None`` (the generation-time default) omits the
            ``novelty_mode`` constructor argument entirely, reproducing the
            pre-change construction exactly. A string is forwarded to the
            constructor — used by ``test_default_mode_matches_golden`` to prove
            that an explicit ``"recent_window"`` is bit-identical to omitting
            it. No ``set_train_nodes`` call is made in either case.

    Returns:
        A NodeStateManager built with NO training-node set.
    """
    edges = _PARITY_EDGES
    n = len(edges)
    src = np.array([e["src"] for e in edges], dtype=np.int64)
    dst = np.array([e["dst"] for e in edges], dtype=np.int64)
    ts = np.array([e["ts"] for e in edges], dtype=np.int64)
    ib = np.array([e["in_bytes"] for e in edges], dtype=np.float32)
    ob = np.array([e["out_bytes"] for e in edges], dtype=np.float32)
    dp = np.array([e["dst_port"] for e in edges], dtype=np.int32)

    kwargs = {} if novelty_mode is None else {"novelty_mode": novelty_mode}
    nsm = NodeStateManager(window_seconds=1.0, snapshot_interval=5, **kwargs)
    half = max(1, n // 2)
    nsm.build_hourly_baselines(
        src[:half], dst[:half], ts[:half], ib[:half], ob[:half]
    )
    nsm.build_snapshots(src, dst, ts, ib, ob, dp, snapshot_interval=5)
    max_node = max(int(src.max()), int(dst.max())) + 1
    nsm.set_is_internal(np.zeros(max_node, dtype=np.float32))
    return nsm


def compute_golden(nsm: NodeStateManager) -> dict[str, np.ndarray]:
    """Compute every golden array from a reference manager.

    Args:
        nsm: manager from :func:`build_reference_manager`.

    Returns:
        Mapping of fixture key → array, ready for ``np.savez``.
    """
    scalar = np.stack([
        np.stack([nsm._compute_state(int(v), float(t)) for v in NODE_IDS])
        for t in QUERY_TIMES
    ])
    batch_all = np.stack([
        nsm.get_batch_states(np.array(NODE_IDS, dtype=np.int64), float(t))
        for t in QUERY_TIMES
    ])
    batch_odd = nsm.get_batch_states(
        np.array([-1, 9999, 3, 0, 0], dtype=np.int64), 600.0
    )
    rollback_first = nsm.rollback_edge(
        node_id=0,
        edge_timestamp_ms=100.0,
        edge_direction="outgoing",
        edge_features={"peer_id": 1},
        query_time_ms=600.0,
    )
    return {
        "scalar": scalar,
        "batch_all": batch_all,
        "batch_odd": batch_odd,
        "rollback_first": rollback_first,
        "query_times": np.array(QUERY_TIMES, dtype=np.float64),
        "node_ids": np.array(NODE_IDS, dtype=np.int64),
    }


def _git_head() -> str:
    """Return the current git HEAD sha, or 'unknown' outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def main() -> None:
    """Generate and write the golden fixture."""
    nsm = build_reference_manager()
    arrays = compute_golden(nsm)
    arrays["source_sha256"] = np.array(
        hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()
    )
    arrays["git_head"] = np.array(_git_head())
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE_PATH, **arrays)
    print(f"wrote {FIXTURE_PATH}")
    print(f"  source_sha256 = {arrays['source_sha256']}")
    print(f"  git_head      = {arrays['git_head']}")


if __name__ == "__main__":
    main()
