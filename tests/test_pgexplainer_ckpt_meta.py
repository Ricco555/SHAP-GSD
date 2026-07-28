"""Tests for ``src/baselines/_ckpt_meta.py`` — the PGExplainer checkpoint
staleness guard (specs/28, specs/29).

The guard exists because a PGExplainer mask-MLP cached at
``<artifacts_dir>/pgexplainer_algorithm.pt`` is fit on embeddings produced by
``src.baselines.adapter.build_h_full`` with a specific ``best_model.pt``.  The
MLP's input dimension does not change when ``build_h_full``'s *semantics*
change, so a stale checkpoint loads without error and emits well-formed, wrong
edge masks straight into Table 2.  A JSON sidecar records what the checkpoint
was trained against and is validated BEFORE ``torch.load`` ever touches the
pickled payload.

Two sections:

  Section 1 — unit tests for ``_ckpt_meta`` (spec 29 §I.4.3 cases 1-12).
      ``_ckpt_meta`` is stdlib-only.  Every one of these tests passes an
      explicit ``schema_version=`` (Rule 1 of spec 29 §I.4.1) so no code path
      resolves the live ``adapter.H_FULL_SCHEMA_VERSION`` — which would import
      torch.  The two exceptions are case 5b (which exists precisely to prove
      the live constant is read at CALL time) and case 12 (a subprocess), both
      guarded with ``pytest.importorskip``.  The expected version number is a
      file-local ``_TEST_SCHEMA_VERSION`` and is NEVER read from ``adapter``
      (Rule 2), so a future ``H_FULL_SCHEMA_VERSION`` bump leaves this section
      green.

  Section 2 — integration tests for ``scripts/10_baselines.py``'s
      ``_train_or_load_pgexplainer``.  These prove the predicate is actually
      WIRED IN: that a stale checkpoint is moved aside and retrained rather
      than silently ``torch.load``-ed, and that a fresh save writes the
      checkpoint AND its sidecar together.  They necessarily import torch,
      dgl and torch_geometric (the script does, at module level), so each is
      individually ``importorskip``-guarded and Section 1 still stands alone
      torch-free.  This is a deliberate deviation from spec 29 §I.4's "no
      torch except 5b" wording, forced by the integration coverage being
      required in this same file.
"""

import hashlib
import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines._ckpt_meta import (  # noqa: E402
    SIDECAR_SCHEMA,
    SIDECAR_SUFFIX,
    STALE_SUFFIX,
    STATUS_CURRENT,
    STATUS_UNCACHED,
    build_sidecar_payload,
    check_checkpoint,
    file_sha256,
    move_aside,
    sidecar_path_for,
    write_sidecar,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Fixture-local, NEVER read from adapter (spec 29 §I.4.1 Rule 2): a bump of the
# live H_FULL_SCHEMA_VERSION must not break a single test in this file.
_TEST_SCHEMA_VERSION: int = 3


# ── shared helpers ────────────────────────────────────────────────────────────

def _make_pair(
    tmp_path: Path,
    *,
    schema_version: int = _TEST_SCHEMA_VERSION,
    model_bytes: bytes = b"model-v1",
) -> tuple[Path, Path, Path]:
    """Create a fake best_model.pt + checkpoint + a valid matching sidecar.

    The ``.pt`` files hold arbitrary bytes: nothing under test ever
    ``torch.load``s them, which is precisely the property being guarded
    (specs/28 §2.2).  The sidecar is written with an explicit
    ``schema_version`` so no code path here consults the live adapter constant.

    Args:
        tmp_path: pytest ``tmp_path`` directory to populate.
        schema_version: value recorded in the sidecar.
        model_bytes: raw bytes of the fake ``best_model.pt``.

    Returns:
        ``(ckpt_path, best_model_path, sidecar_path)``.
    """
    best_model = tmp_path / "best_model.pt"
    best_model.write_bytes(model_bytes)

    ckpt = tmp_path / "pgexplainer_algorithm.pt"
    ckpt.write_bytes(b"not-a-real-checkpoint")

    sidecar = write_sidecar(
        ckpt,
        build_sidecar_payload(
            best_model,
            pg_epochs=30,
            pg_lr=0.003,
            n_train=200,
            seed=42,
            schema_version=schema_version,
        ),
    )
    return ckpt, best_model, sidecar


def _write_raw_sidecar(ckpt_path: Path, text: str) -> Path:
    """Overwrite a checkpoint's sidecar with raw text (valid JSON or not).

    Args:
        ckpt_path: Checkpoint whose sidecar path is derived.
        text: Exact file content to write.

    Returns:
        The sidecar path written.
    """
    sidecar = sidecar_path_for(ckpt_path)
    sidecar.write_text(text)
    return sidecar


# ══ Section 1 — unit tests for _ckpt_meta (spec 29 §I.4.3) ════════════════════

def test_sha256_matches_hashlib_over_file_bytes(tmp_path: Path) -> None:
    """file_sha256 equals hashlib over the raw bytes and tracks content changes."""
    p = tmp_path / "blob.bin"
    p.write_bytes(b"alpha" * 1000)

    digest = file_sha256(p)
    assert digest == hashlib.sha256(p.read_bytes()).hexdigest()
    assert len(digest) == 64
    assert digest == digest.lower()
    assert all(c in "0123456789abcdef" for c in digest)

    # One changed byte must change the digest — presence is not enough.
    p.write_bytes(b"alpha" * 999 + b"alphb")
    assert file_sha256(p) != digest

    # Chunked reads must not affect the result.
    assert file_sha256(p, chunk_size=7) == file_sha256(p)


def test_sha256_missing_file_returns_none(tmp_path: Path) -> None:
    """A missing file hashes to None rather than raising (specs/28 §4)."""
    assert file_sha256(tmp_path / "nope.pt") is None


def test_write_sidecar_produces_valid_schema(tmp_path: Path) -> None:
    """write_sidecar round-trips the §2.5 schema and leaves no .tmp behind."""
    best_model = tmp_path / "best_model.pt"
    best_model.write_bytes(b"weights")
    ckpt = tmp_path / "pgexplainer_algorithm.pt"

    payload = build_sidecar_payload(
        best_model, pg_epochs=30, pg_lr=0.003, n_train=200, seed=42,
        schema_version=_TEST_SCHEMA_VERSION,
    )
    sidecar = write_sidecar(ckpt, payload)

    assert sidecar == sidecar_path_for(ckpt)
    data = json.loads(sidecar.read_text())

    for key in (
        "schema", "h_full_schema_version", "best_model_sha256",
        "best_model_path", "torch_geometric_version", "pg_epochs", "pg_lr",
        "n_train", "seed", "written_at",
    ):
        assert key in data, f"sidecar is missing key {key!r}"

    assert data["schema"] == SIDECAR_SCHEMA
    assert data["h_full_schema_version"] == _TEST_SCHEMA_VERSION
    assert isinstance(data["h_full_schema_version"], int)
    assert isinstance(data["best_model_sha256"], str)
    assert len(data["best_model_sha256"]) == 64
    assert data["best_model_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert data["best_model_path"] == str(best_model)
    assert data["pg_epochs"] == 30 and data["n_train"] == 200 and data["seed"] == 42
    assert data["written_at"].endswith("Z")

    # Atomic write: the temp file must not survive a successful write.
    assert not (sidecar.with_name(sidecar.name + ".tmp")).exists()
    assert [p.name for p in tmp_path.glob("*.tmp")] == []


def test_matching_sidecar_is_current(tmp_path: Path) -> None:
    """A freshly written sidecar with unchanged inputs is CURRENT (§5 rule 8)."""
    ckpt, best_model, _ = _make_pair(tmp_path)

    status = check_checkpoint(ckpt, best_model, schema_version=_TEST_SCHEMA_VERSION)

    assert status.status == STATUS_CURRENT
    assert status.is_current and not status.is_stale
    assert status.reason == ""
    assert status.detail == ""


@pytest.mark.parametrize("written,live", [(3, 4), (3, 2)])
def test_schema_version_bump_makes_checkpoint_stale(
    tmp_path: Path, written: int, live: int,
) -> None:
    """5a: an explicit live version differing from the sidecar's is STALE (§5 rule 5).

    Both directions are covered — a downgrade is as invalid as an upgrade.
    """
    ckpt, best_model, _ = _make_pair(tmp_path, schema_version=written)

    status = check_checkpoint(ckpt, best_model, schema_version=live)

    assert status.is_stale
    assert status.reason == "schema_version_mismatch"
    assert status.detail == f"sidecar h_full_schema_version={written}, live={live}"


def test_schema_version_read_from_live_adapter_at_call_time(
    tmp_path: Path, monkeypatch,
) -> None:
    """5b: the live constant is read at CALL time, not bound at import time.

    This is the only unit test that omits ``schema_version=`` and therefore the
    only one that imports ``adapter`` (and transitively torch).  It fails if
    ``_ckpt_meta`` ever does ``from src.baselines.adapter import
    H_FULL_SCHEMA_VERSION`` at module top (spec 29 §I.1.3).  Both halves are
    asserted so an implementation that always reports stale cannot pass.
    """
    pytest.importorskip("torch")
    import src.baselines.adapter as adapter

    ckpt, best_model, _ = _make_pair(tmp_path, schema_version=_TEST_SCHEMA_VERSION)

    monkeypatch.setattr(adapter, "H_FULL_SCHEMA_VERSION", 999)
    stale = check_checkpoint(ckpt, best_model)
    assert stale.is_stale
    assert stale.reason == "schema_version_mismatch"
    assert stale.detail == (
        f"sidecar h_full_schema_version={_TEST_SCHEMA_VERSION}, live=999"
    )

    monkeypatch.setattr(adapter, "H_FULL_SCHEMA_VERSION", _TEST_SCHEMA_VERSION)
    assert check_checkpoint(ckpt, best_model).is_current


def test_changed_model_hash_makes_checkpoint_stale(tmp_path: Path) -> None:
    """A different (or absent) best_model.pt makes the checkpoint STALE (§5 rule 6)."""
    ckpt, best_model, _ = _make_pair(tmp_path, model_bytes=b"model-v1")

    best_model.write_bytes(b"model-v2-retrained")
    status = check_checkpoint(ckpt, best_model, schema_version=_TEST_SCHEMA_VERSION)
    assert status.is_stale
    assert status.reason == "model_hash_mismatch"
    assert status.detail.startswith("sidecar best_model_sha256=")
    assert "<unreadable>" not in status.detail

    best_model.unlink()
    missing = check_checkpoint(ckpt, best_model, schema_version=_TEST_SCHEMA_VERSION)
    assert missing.is_stale
    assert missing.reason == "model_hash_mismatch"
    assert missing.detail.endswith("current=<unreadable>")


def test_missing_sidecar_with_present_checkpoint_is_stale(tmp_path: Path) -> None:
    """Checkpoint without a sidecar is STALE (§5 rule 1 — the crash-safety rule).

    This encodes specs/28 §2.4: a crash between ``torch.save`` and the sidecar
    write must degrade to a retrain, never to a silent load.
    """
    ckpt, best_model, sidecar = _make_pair(tmp_path)
    sidecar.unlink()

    status = check_checkpoint(ckpt, best_model, schema_version=_TEST_SCHEMA_VERSION)
    assert status.is_stale
    assert status.reason == "sidecar_missing"


@pytest.mark.parametrize(
    "raw,expected_reason",
    [
        ("{", "sidecar_unreadable"),                       # truncated JSON
        ("", "sidecar_unreadable"),                        # empty file
        ("null", "sidecar_unreadable"),                    # valid JSON, not a dict
        ("[]", "sidecar_unreadable"),                      # valid JSON, not a dict
        ('{"schema": "pgexplainer_algorithm.meta/1", '
         '"best_model_sha256": "ab"}', "sidecar_field_missing"),
        ('{"schema": "pgexplainer_algorithm.meta/1", '
         '"h_full_schema_version": 3}', "sidecar_field_missing"),
        ('{"schema": "pgexplainer_algorithm.meta/1", '
         '"h_full_schema_version": "3", '
         '"best_model_sha256": "ab"}', "sidecar_field_type"),
        ('{"schema": "pgexplainer_algorithm.meta/1", '
         '"h_full_schema_version": true, '
         '"best_model_sha256": "ab"}', "sidecar_field_type"),
        ('{"schema": "pgexplainer_algorithm.meta/1", '
         '"h_full_schema_version": 3, '
         '"best_model_sha256": 12345}', "sidecar_field_type"),
    ],
)
def test_corrupt_sidecar_is_stale_not_error(
    tmp_path: Path, raw: str, expected_reason: str,
) -> None:
    """Every corrupt/wrong-typed sidecar is STALE and never raises (§5 rules 2-3).

    The ``true`` case is load-bearing: ``isinstance(True, int)`` is True in
    Python and ``True == 1`` would otherwise let a corrupt sidecar validate
    against schema version 1.
    """
    ckpt, best_model, _ = _make_pair(tmp_path)
    _write_raw_sidecar(ckpt, raw)

    status = check_checkpoint(ckpt, best_model, schema_version=_TEST_SCHEMA_VERSION)
    assert status.is_stale
    assert status.reason == expected_reason


def test_boolean_schema_version_does_not_validate_against_version_one(
    tmp_path: Path,
) -> None:
    """``true`` must not satisfy ``schema_version=1`` via bool-is-int coercion."""
    ckpt, best_model, _ = _make_pair(tmp_path)
    real_hash = hashlib.sha256(best_model.read_bytes()).hexdigest()
    _write_raw_sidecar(ckpt, json.dumps({
        "schema": SIDECAR_SCHEMA,
        "h_full_schema_version": True,
        "best_model_sha256": real_hash,
    }))

    status = check_checkpoint(ckpt, best_model, schema_version=1)
    assert status.is_stale
    assert status.reason == "sidecar_field_type"


def test_null_model_hash_in_sidecar_is_stale_not_error(tmp_path: Path) -> None:
    """A sidecar written while best_model.pt was absent records null and is STALE.

    ``build_sidecar_payload`` uses ``file_sha256``, which returns ``None`` for a
    missing file, so ``best_model_sha256: null`` is a reachable on-disk state —
    not merely a synthetic corruption.  ``check_checkpoint`` must reject it on
    the type check before the ``sidecar_hash[:12]`` slice can raise.
    """
    ckpt = tmp_path / "pgexplainer_algorithm.pt"
    ckpt.write_bytes(b"ckpt")
    best_model = tmp_path / "best_model.pt"      # deliberately never created

    payload = build_sidecar_payload(
        best_model, pg_epochs=1, pg_lr=0.1, n_train=1, seed=0,
        schema_version=_TEST_SCHEMA_VERSION,
    )
    assert payload["best_model_sha256"] is None
    write_sidecar(ckpt, payload)

    status = check_checkpoint(ckpt, best_model, schema_version=_TEST_SCHEMA_VERSION)
    assert status.is_stale
    assert status.reason == "sidecar_field_type"


def test_unknown_schema_string_is_stale(tmp_path: Path) -> None:
    """An unrecognised sidecar ``schema`` string is STALE (§5 rule 4)."""
    ckpt, best_model, sidecar = _make_pair(tmp_path)
    data = json.loads(sidecar.read_text())
    data["schema"] = "something/2"
    sidecar.write_text(json.dumps(data))

    status = check_checkpoint(ckpt, best_model, schema_version=_TEST_SCHEMA_VERSION)
    assert status.is_stale
    assert status.reason == "sidecar_schema_unknown"


def test_absent_checkpoint_is_not_reported_as_stale(tmp_path: Path) -> None:
    """An absent checkpoint is UNCACHED, not STALE, sidecar present or not (§5 rule 7).

    Rule 7 must be evaluated first: otherwise a normal first run reports
    ``sidecar_missing`` and warns spuriously on every fresh pipeline.
    """
    ckpt, best_model, sidecar = _make_pair(tmp_path)
    ckpt.unlink()

    with_orphan = check_checkpoint(
        ckpt, best_model, schema_version=_TEST_SCHEMA_VERSION)
    assert with_orphan.status == STATUS_UNCACHED
    assert not with_orphan.is_stale and not with_orphan.is_current
    assert with_orphan.reason == "no_checkpoint"

    sidecar.unlink()
    without = check_checkpoint(ckpt, best_model, schema_version=_TEST_SCHEMA_VERSION)
    assert without.status == STATUS_UNCACHED
    assert without.reason == "no_checkpoint"


def test_move_aside_overwrites_existing_stale_file(tmp_path: Path) -> None:
    """move_aside overwrites pre-existing .stale destinations and is idempotent (§5.1)."""
    ckpt, _, sidecar = _make_pair(tmp_path)
    ckpt.write_bytes(b"new-checkpoint-bytes")
    new_sidecar_text = sidecar.read_text()

    stale_ckpt = ckpt.with_name(ckpt.name + STALE_SUFFIX)
    stale_sidecar = sidecar.with_name(sidecar.name + STALE_SUFFIX)
    stale_ckpt.write_bytes(b"OLD-stale-checkpoint")
    stale_sidecar.write_text("OLD-stale-sidecar")

    moved = move_aside(ckpt)

    assert set(moved) == {stale_ckpt, stale_sidecar}
    assert not ckpt.exists() and not sidecar.exists()
    assert stale_ckpt.read_bytes() == b"new-checkpoint-bytes"
    assert stale_sidecar.read_text() == new_sidecar_text

    # Sources already gone: no exception, nothing moved.
    assert move_aside(ckpt) == []
    assert stale_ckpt.read_bytes() == b"new-checkpoint-bytes"


def test_ckpt_meta_module_imports_no_heavy_deps() -> None:
    """Importing _ckpt_meta pulls in no torch / dgl / torch_geometric (A2).

    Run in a subprocess because the pytest session itself has torch imported by
    other test modules.
    """
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import src.baselines._ckpt_meta;"
        "heavy = {'torch', 'dgl', 'torch_geometric'} & set(sys.modules);"
        "assert not heavy, heavy"
    ) % str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr


def test_sidecar_path_for_derives_expected_name(tmp_path: Path) -> None:
    """The sidecar path is the checkpoint path with .pt replaced by .meta.json."""
    ckpt = tmp_path / "pgexplainer_algorithm.pt"
    assert sidecar_path_for(ckpt) == tmp_path / "pgexplainer_algorithm.meta.json"
    assert sidecar_path_for(ckpt).name.endswith(SIDECAR_SUFFIX)
    # Accepts a str as well as a Path, and stays in the same directory.
    assert sidecar_path_for(str(ckpt)).parent == tmp_path


# ══ Section 2 — integration: scripts/10_baselines.py wiring ═══════════════════
#
# These prove the predicate is actually consulted and actually gates the
# load-vs-retrain branch.  They import torch/dgl/torch_geometric (the script
# does at module level), so each is importorskip-guarded individually.

_BASELINES_MODULE = None


def _load_baselines_script():
    """Import ``scripts/10_baselines.py`` as a module (``scripts/`` is not a package).

    Returns:
        The loaded module object, cached across tests in this session.
    """
    global _BASELINES_MODULE
    if _BASELINES_MODULE is None:
        path = REPO_ROOT / "scripts" / "10_baselines.py"
        spec = importlib.util.spec_from_file_location("baselines_phase10", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _BASELINES_MODULE = module
    return _BASELINES_MODULE


class _RecordingTorch:
    """Delegating stand-in for the ``torch`` global inside 10_baselines.py.

    Records ``load``/``save`` calls while forwarding everything else to the
    real module.  Patched onto the *script module's* name so the real ``torch``
    module is never mutated for the rest of the pytest session.
    """

    def __init__(self, real) -> None:
        self._real = real
        self.load_calls: list[Path] = []
        self.save_calls: list[Path] = []

    def load(self, path, *args, **kwargs):
        """Record and forward a ``torch.load`` call."""
        self.load_calls.append(Path(path))
        return self._real.load(path, *args, **kwargs)

    def save(self, obj, path, *args, **kwargs):
        """Record and forward a ``torch.save`` call."""
        self.save_calls.append(Path(path))
        return self._real.save(obj, path, *args, **kwargs)

    def __getattr__(self, name):
        """Forward every other attribute to the real torch module."""
        return getattr(self._real, name)


class _FakeDGL:
    """Stand-in for the ``dgl`` global; ``load_graphs`` returns a dummy graph."""

    def load_graphs(self, path: str):
        """Return a ``([graph], {})`` pair without touching the filesystem."""
        return [object()], {}


def _install_pg_stubs(monkeypatch, tmp_path: Path, mlp):
    """Patch the three seams of ``_train_or_load_pgexplainer``.

    Replaces the script module's ``torch`` and ``dgl`` globals and the
    ``train_pgexplainer`` function at its source module (the script imports it
    function-locally, so patching the script's namespace would not take).

    Args:
        monkeypatch: pytest monkeypatch fixture.
        tmp_path: directory used as both artifacts_dir and graph dir.
        mlp: object assigned to the stub-trained algorithm's ``.mlp``.

    Returns:
        ``(module, fake_torch, calls)`` where ``calls`` is a list appended to
        once per stubbed ``train_pgexplainer`` invocation.
    """
    module = _load_baselines_script()
    fake_torch = _RecordingTorch(module.torch)
    monkeypatch.setattr(module, "torch", fake_torch)
    monkeypatch.setattr(module, "dgl", _FakeDGL())

    calls: list[dict] = []

    class _StubAlgorithm:
        """Minimal stand-in for a trained PGExplainer algorithm object."""

        def __init__(self) -> None:
            self.mlp = mlp

    def _stub_train(**kwargs):
        """Record the call and return a stub algorithm instead of training."""
        calls.append(kwargs)
        return _StubAlgorithm()

    monkeypatch.setattr(
        "src.baselines.pgexplainer_wrapper.train_pgexplainer", _stub_train)
    return module, fake_torch, calls


def _call_train_or_load(module, tmp_path: Path, *, skip_training: bool):
    """Invoke ``_train_or_load_pgexplainer`` with inert infrastructure args.

    Args:
        module: the loaded 10_baselines module.
        tmp_path: artifacts + graph directory.
        skip_training: value of the ``--skip-pg-train`` flag.

    Returns:
        Whatever ``_train_or_load_pgexplainer`` returns.
    """
    return module._train_or_load_pgexplainer(
        cfg={"graph": {"dir": str(tmp_path)}, "seed": 7},
        model=None, nsm=None, fs_train=None, sampler=None, device=None,
        artifacts_dir=tmp_path,
        n_train=5, pg_epochs=2, pg_lr=0.01,
        skip_training=skip_training,
    )


def test_integration_stale_checkpoint_is_moved_aside_and_retrained(
    tmp_path: Path, monkeypatch, caplog,
) -> None:
    """A stale cache is never torch.load-ed: it is moved aside, retrained, re-saved.

    Staleness is induced by removing the sidecar (rule 1), which is independent
    of the live H_FULL_SCHEMA_VERSION.  This single test covers the whole cycle:
    the guard is consulted, it gates the branch (``torch.load`` is NOT called),
    the pair moves to ``*.stale``, training runs, and the save path writes BOTH
    the checkpoint and a sidecar that validates as current.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    (tmp_path / "best_model.pt").write_bytes(b"trained-weights")
    ckpt = tmp_path / "pgexplainer_algorithm.pt"
    ckpt.write_bytes(b"STALE-CHECKPOINT")          # no sidecar beside it

    module, fake_torch, calls = _install_pg_stubs(
        monkeypatch, tmp_path, mlp=torch.nn.Linear(4, 1))

    with caplog.at_level(logging.DEBUG):
        result = _call_train_or_load(module, tmp_path, skip_training=True)

    # The guard gated the branch: the stale pickle was never opened.
    assert fake_torch.load_calls == []
    # It fell through to a real retrain.
    assert len(calls) == 1
    assert calls[0]["n_train"] == 5 and calls[0]["epochs"] == 2
    assert result is not None

    # The stale file was preserved, not deleted or reused.
    stale = tmp_path / ("pgexplainer_algorithm.pt" + STALE_SUFFIX)
    assert stale.read_bytes() == b"STALE-CHECKPOINT"

    # A loud warning naming the reason, not a silent info line.
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "stale cache must emit logger.warning"
    text = "\n".join(r.getMessage() for r in warnings)
    assert "STALE" in text and "sidecar_missing" in text and str(stale) in text

    # The save path wrote checkpoint AND sidecar, and the new pair is current.
    assert fake_torch.save_calls == [ckpt]
    sidecar = sidecar_path_for(ckpt)
    assert ckpt.exists() and sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["schema"] == SIDECAR_SCHEMA
    assert payload["seed"] == 7 and payload["pg_epochs"] == 2 and payload["n_train"] == 5
    assert payload["best_model_sha256"] == hashlib.sha256(b"trained-weights").hexdigest()
    assert check_checkpoint(ckpt, tmp_path / "best_model.pt").is_current


def test_integration_current_checkpoint_is_loaded_without_retraining(
    tmp_path: Path, monkeypatch, caplog,
) -> None:
    """A checkpoint with a matching sidecar still loads from cache — no retrain.

    Guards against the staleness check being over-eager and silently
    invalidating every cache (which would pass the stale test above vacuously).
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    best_model = tmp_path / "best_model.pt"
    best_model.write_bytes(b"trained-weights")
    ckpt = tmp_path / "pgexplainer_algorithm.pt"
    torch.save({"mlp": torch.nn.Linear(4, 1)}, ckpt)
    # Built with schema_version=None so it tracks the live constant, exactly as
    # the production save path does.
    write_sidecar(ckpt, build_sidecar_payload(
        best_model, pg_epochs=2, pg_lr=0.01, n_train=5, seed=7))

    module, fake_torch, calls = _install_pg_stubs(
        monkeypatch, tmp_path, mlp=torch.nn.Linear(4, 1))

    with caplog.at_level(logging.DEBUG):
        algorithm = _call_train_or_load(module, tmp_path, skip_training=True)

    assert calls == [], "a current checkpoint must not trigger a retrain"
    assert fake_torch.load_calls == [ckpt]
    assert fake_torch.save_calls == []
    assert algorithm.mlp is not None
    assert not (tmp_path / ("pgexplainer_algorithm.pt" + STALE_SUFFIX)).exists()
    assert not any(p.name.endswith(STALE_SUFFIX) for p in tmp_path.iterdir())
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_integration_no_sidecar_written_when_mlp_is_none(
    tmp_path: Path, monkeypatch,
) -> None:
    """The save path's ``if algorithm.mlp is not None`` guard skips BOTH writes.

    A checkpoint must never appear without an attempted sidecar, and neither
    must a sidecar appear without a checkpoint (specs/28 §2.4).
    """
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    (tmp_path / "best_model.pt").write_bytes(b"trained-weights")
    module, fake_torch, calls = _install_pg_stubs(monkeypatch, tmp_path, mlp=None)

    _call_train_or_load(module, tmp_path, skip_training=False)

    assert len(calls) == 1
    assert fake_torch.save_calls == []
    assert not (tmp_path / "pgexplainer_algorithm.pt").exists()
    assert not (tmp_path / "pgexplainer_algorithm.meta.json").exists()


def test_integration_orphan_sidecar_is_tidied_without_warning(
    tmp_path: Path, monkeypatch, caplog,
) -> None:
    """A sidecar with no checkpoint is UNCACHED: tidied at debug level, no warning."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    best_model = tmp_path / "best_model.pt"
    best_model.write_bytes(b"trained-weights")
    ckpt = tmp_path / "pgexplainer_algorithm.pt"
    write_sidecar(ckpt, build_sidecar_payload(
        best_model, pg_epochs=2, pg_lr=0.01, n_train=5, seed=7))
    assert not ckpt.exists()

    module, fake_torch, calls = _install_pg_stubs(
        monkeypatch, tmp_path, mlp=torch.nn.Linear(4, 1))

    with caplog.at_level(logging.DEBUG):
        _call_train_or_load(module, tmp_path, skip_training=True)

    assert fake_torch.load_calls == []
    assert len(calls) == 1, "an uncached checkpoint must retrain"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    # The orphan was moved aside, and the fresh save left a valid new pair.
    assert (tmp_path / ("pgexplainer_algorithm.meta.json" + STALE_SUFFIX)).exists()
    assert check_checkpoint(ckpt, best_model).is_current


def test_integration_check_is_skipped_entirely_without_skip_flag(
    tmp_path: Path, monkeypatch, caplog,
) -> None:
    """Without --skip-pg-train the cache is never consulted and is overwritten (§5 rule 9)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    best_model = tmp_path / "best_model.pt"
    best_model.write_bytes(b"trained-weights")
    ckpt = tmp_path / "pgexplainer_algorithm.pt"
    torch.save({"mlp": torch.nn.Linear(4, 1)}, ckpt)
    write_sidecar(ckpt, build_sidecar_payload(
        best_model, pg_epochs=2, pg_lr=0.01, n_train=5, seed=7))

    module, fake_torch, calls = _install_pg_stubs(
        monkeypatch, tmp_path, mlp=torch.nn.Linear(4, 1))

    with caplog.at_level(logging.DEBUG):
        _call_train_or_load(module, tmp_path, skip_training=False)

    assert fake_torch.load_calls == [], "cache must not be read when not skipping"
    assert len(calls) == 1
    assert fake_torch.save_calls == [ckpt]
    assert check_checkpoint(ckpt, best_model).is_current
