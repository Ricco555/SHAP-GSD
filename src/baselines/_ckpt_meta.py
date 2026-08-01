"""Provenance sidecar for the cached PGExplainer mask-MLP checkpoint.

scripts/10_baselines.py caches PGExplainer's trained mask-MLP at
<artifacts_dir>/pgexplainer_algorithm.pt so that --skip-pg-train can reuse it.
That MLP is fit on embeddings produced by src.baselines.adapter.build_h_full
using a specific trained model, and is silently invalid if either changes: the
MLP's input dimension does not change when build_h_full's semantics change, so
a stale checkpoint produces well-formed, plausible, WRONG edge masks that land
in Table 2 with no exception and no warning.

This module writes and checks a JSON sidecar recording what the checkpoint was
trained against, so staleness is detected BEFORE torch.load unpickles the
checkpoint (the payload is a pickled nn.Module loaded with weights_only=False,
which is exactly what you should not open to establish provenance).

Deliberately import-light: stdlib only (hashlib, json, os, pathlib, datetime,
dataclasses, importlib) PLUS src.model.selection.write_json_atomic, itself
stdlib-only (specs/34 §4.3) — the single shared tmp-file + os.replace atomic
write used everywhere in this project, so this module does not hand-maintain
its own second copy. No torch/dgl/torch_geometric import here, so the unit
tests need no GPU, no artifacts and no third-party explainer libraries
(see tests/test_pgexplainer_ckpt_meta.py). See specs/28 and specs/29.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.model.selection import write_json_atomic

SIDECAR_SCHEMA: str = "pgexplainer_algorithm.meta/1"
SIDECAR_SUFFIX: str = ".meta.json"
STALE_SUFFIX: str = ".stale"

STATUS_CURRENT: str = "current"
STATUS_STALE: str = "stale"
STATUS_UNCACHED: str = "uncached"


@dataclass(frozen=True)
class CheckpointStatus:
    """Verdict on a cached checkpoint + sidecar pair.

    Attributes:
        status: one of STATUS_CURRENT / STATUS_STALE / STATUS_UNCACHED.
        reason: machine-stable short key naming why; empty string when status
            is STATUS_CURRENT. One of ``no_checkpoint``, ``sidecar_missing``,
            ``sidecar_unreadable``, ``sidecar_field_missing``,
            ``sidecar_field_type``, ``sidecar_schema_unknown``,
            ``schema_version_mismatch``, ``model_hash_mismatch``.
        detail: log-line fragment; empty string when status is
            STATUS_CURRENT. For the two predicate mismatches the format is
            FIXED VERBATIM so tests assert on it rather than on prose:
              schema_version_mismatch:
                  "sidecar h_full_schema_version={s}, live={l}"
              model_hash_mismatch:
                  "sidecar best_model_sha256={s12}, current={c12}"
                  (12-char prefixes; ``current`` is the literal string
                  "<unreadable>" when best_model.pt is missing or unreadable)
            For all other reasons ``detail`` is free-form and no test asserts
            on it.
    """

    status: str
    reason: str = ""
    detail: str = ""

    @property
    def is_current(self) -> bool:
        """True iff the cached checkpoint may be loaded as-is."""
        return self.status == STATUS_CURRENT

    @property
    def is_stale(self) -> bool:
        """True iff a checkpoint exists but must not be reused."""
        return self.status == STATUS_STALE


def sidecar_path_for(ckpt_path: Path) -> Path:
    """Return the sidecar JSON path for a checkpoint path.

    ``.../pgexplainer_algorithm.pt`` -> ``.../pgexplainer_algorithm.meta.json``.
    Uses ``with_suffix("")`` + SIDECAR_SUFFIX so the two names are derived from
    one place and can never drift apart in a caller.

    Args:
        ckpt_path: Path of the checkpoint ``.pt`` file.

    Returns:
        The sidecar path beside it.
    """
    ckpt_path = Path(ckpt_path)
    return ckpt_path.with_name(ckpt_path.with_suffix("").name + SIDECAR_SUFFIX)


def file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str | None:
    """Streamed sha256 hex digest of a file's raw bytes.

    Bytes, not state_dict tensors: the point is to answer "is this the same
    model file" WITHOUT torch.load-ing anything (specs/28 §4). Streamed in
    1 MiB chunks; artifacts/best_model.pt is ~407 KB, ~1 ms.

    Args:
        path: File to hash.
        chunk_size: Read-chunk size in bytes.

    Returns:
        64-char lowercase hex digest, or ``None`` if the file does not exist or
        cannot be read -- never raises (specs/28 §4, §5 rule 6).
    """
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def live_h_full_schema_version() -> int:
    """Read H_FULL_SCHEMA_VERSION from src.baselines.adapter AT CALL TIME.

    Uses ``importlib.import_module("src.baselines.adapter")`` and
    ``getattr(mod, "H_FULL_SCHEMA_VERSION")`` rather than a module-level
    ``from ... import`` binding, so that
    ``monkeypatch.setattr(adapter, "H_FULL_SCHEMA_VERSION", ...)`` is observed
    here (specs/28 §6, §9 case 5). A ``from`` import would freeze the value at
    import time and make the headline regression test silently vacuous.

    This is the ONLY place in this stdlib-only module that touches adapter, and
    it is deliberately a *function-local* import so importing _ckpt_meta itself
    still pulls in no torch/dgl/PyG (acceptance criterion A2).

    Returns:
        The live ``H_FULL_SCHEMA_VERSION`` integer.
    """
    adapter = importlib.import_module("src.baselines.adapter")
    return int(getattr(adapter, "H_FULL_SCHEMA_VERSION"))


def build_sidecar_payload(
    best_model_path: Path,
    *,
    pg_epochs: int,
    pg_lr: float,
    n_train: int,
    seed: int,
    schema_version: int | None = None,
) -> dict:
    """Assemble the sidecar dict (specs/28 §2.5).

    Args:
        best_model_path: Path of the ``best_model.pt`` whose parameters
            produced the embeddings the MLP was fit on.
        pg_epochs: PGExplainer training epochs (informational).
        pg_lr: PGExplainer learning rate (informational).
        n_train: Number of training flows used (informational).
        seed: Seed used for the training run (informational).
        schema_version: ``build_h_full`` semantics version to record;
            ``None`` resolves via :func:`live_h_full_schema_version`.

    Returns:
        A JSON-serialisable dict. ``torch_geometric_version`` is best-effort:
        resolved via ``importlib.metadata.version("torch_geometric")`` inside a
        try/except and recorded as ``None`` when unavailable -- informational
        only, never in the predicate, and must never make the write fail.
        ``written_at`` is UTC ISO-8601 with a trailing ``Z``.
    """
    if schema_version is None:
        schema_version = live_h_full_schema_version()

    try:
        from importlib.metadata import version as _pkg_version

        pyg_version: str | None = _pkg_version("torch_geometric")
    except Exception:                               # pragma: no cover - env-dependent
        pyg_version = None

    return {
        "schema": SIDECAR_SCHEMA,
        "h_full_schema_version": int(schema_version),
        "best_model_sha256": file_sha256(Path(best_model_path)),
        "best_model_path": str(best_model_path),
        "torch_geometric_version": pyg_version,
        "pg_epochs": pg_epochs,
        "pg_lr": pg_lr,
        "n_train": n_train,
        "seed": seed,
        "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def write_sidecar(ckpt_path: Path, payload: dict) -> Path:
    """Atomically write the sidecar next to ``ckpt_path``.

    Delegates to ``src.model.selection.write_json_atomic`` -- the one
    shared tmp-file + os.replace atomic-write implementation this project
    uses everywhere, rather than a second hand-maintained copy -- with
    ``sort_keys=True`` for a stable on-disk key order. (The checkpoint and
    its sidecar are two files and CANNOT be written atomically with respect
    to each other; they do not need to be -- specs/28 §2.4 shows both crash
    orderings degrade to a retrain.)

    Args:
        ckpt_path: Path of the checkpoint the sidecar describes.
        payload: Dict produced by :func:`build_sidecar_payload`.

    Returns:
        The sidecar path written.
    """
    sidecar = sidecar_path_for(ckpt_path)
    write_json_atomic(sidecar, payload, sort_keys=True)
    return sidecar


def check_checkpoint(
    ckpt_path: Path,
    best_model_path: Path,
    *,
    schema_version: int | None = None,
) -> CheckpointStatus:
    """Decide whether a cached checkpoint may be loaded. Never raises.

    Implements spec 28 §5's truth table in order: rule 7 (no checkpoint) is
    checked first, then the sidecar's existence, readability, schema string,
    predicate-field presence and types, then the two predicate comparisons.
    Performs NO torch.load and imports no torch (acceptance criterion A7).

    Args:
        ckpt_path: Path of the cached ``pgexplainer_algorithm.pt``.
        best_model_path: Path of the ``best_model.pt`` in the same artifacts
            directory.
        schema_version: ``build_h_full`` semantics version to compare against;
            ``None`` resolves via :func:`live_h_full_schema_version`.

    Returns:
        A :class:`CheckpointStatus` with one of the three statuses.
    """
    ckpt_path = Path(ckpt_path)

    # Rule 7 first: an absent checkpoint is simply uncached, never "stale".
    if not ckpt_path.exists():
        return CheckpointStatus(
            STATUS_UNCACHED, "no_checkpoint",
            f"no cached checkpoint at {ckpt_path}",
        )

    sidecar = sidecar_path_for(ckpt_path)
    if not sidecar.exists():
        return CheckpointStatus(
            STATUS_STALE, "sidecar_missing",
            f"no provenance sidecar at {sidecar}",
        )

    try:
        with open(sidecar) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return CheckpointStatus(
            STATUS_STALE, "sidecar_unreadable",
            f"sidecar {sidecar} could not be parsed as JSON",
        )
    if not isinstance(data, dict):
        return CheckpointStatus(
            STATUS_STALE, "sidecar_unreadable",
            f"sidecar {sidecar} is not a JSON object",
        )

    if data.get("schema") != SIDECAR_SCHEMA:
        return CheckpointStatus(
            STATUS_STALE, "sidecar_schema_unknown",
            f"sidecar schema={data.get('schema')!r}, expected "
            f"{SIDECAR_SCHEMA!r}",
        )

    for key in ("h_full_schema_version", "best_model_sha256"):
        if key not in data:
            return CheckpointStatus(
                STATUS_STALE, "sidecar_field_missing",
                f"sidecar is missing predicate field {key!r}",
            )

    sidecar_version = data["h_full_schema_version"]
    if isinstance(sidecar_version, bool) or not isinstance(sidecar_version, int):
        # bool is an int subclass and True == 1 would let a corrupt sidecar
        # validate against version 1 -- reject it explicitly.
        return CheckpointStatus(
            STATUS_STALE, "sidecar_field_type",
            f"sidecar h_full_schema_version has type "
            f"{type(sidecar_version).__name__}, expected int",
        )

    sidecar_hash = data["best_model_sha256"]
    if not isinstance(sidecar_hash, str):
        return CheckpointStatus(
            STATUS_STALE, "sidecar_field_type",
            f"sidecar best_model_sha256 has type "
            f"{type(sidecar_hash).__name__}, expected str",
        )

    if schema_version is None:
        schema_version = live_h_full_schema_version()
    if sidecar_version != schema_version:
        return CheckpointStatus(
            STATUS_STALE, "schema_version_mismatch",
            f"sidecar h_full_schema_version={sidecar_version}, "
            f"live={schema_version}",
        )

    current_hash = file_sha256(Path(best_model_path))
    if current_hash != sidecar_hash:
        current_short = current_hash[:12] if current_hash else "<unreadable>"
        return CheckpointStatus(
            STATUS_STALE, "model_hash_mismatch",
            f"sidecar best_model_sha256={sidecar_hash[:12]}, "
            f"current={current_short}",
        )

    return CheckpointStatus(STATUS_CURRENT)


def move_aside(ckpt_path: Path) -> list[Path]:
    """Rename a stale checkpoint and its sidecar to ``*.stale``.

    Uses ``os.replace`` (not ``Path.rename``) because a ``.stale`` destination
    may already exist and must be overwritten silently rather than raising.
    Only one generation is kept; no rotation (specs/28 §7).

    Args:
        ckpt_path: Path of the checkpoint to move aside; its sidecar is
            derived via :func:`sidecar_path_for`.

    Returns:
        The destination paths actually written (0-2 entries), for the log line.
        Missing sources are skipped, not an error.
    """
    moved: list[Path] = []
    for src in (Path(ckpt_path), sidecar_path_for(ckpt_path)):
        if not src.exists():
            continue
        dst = src.with_name(src.name + STALE_SUFFIX)
        try:
            os.replace(src, dst)
        except OSError:                             # pragma: no cover - defensive
            continue
        moved.append(dst)
    return moved
