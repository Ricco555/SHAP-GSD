"""Dataset name -> canonical run-directory resolution for ``explore/evaluation/``.

Implements specs/64 Part II §13 in full. This is the single load-bearing
module of the package: every ``evalNN_*.py`` script turns a caller-supplied
short dataset name (``--dataset unsw``) into exactly one canonical run
directory through here, and a wrong resolution silently produces a paper
number from the wrong run.

What it reads
-------------
Only the *directory layout* of ``runs/`` (default) or ``--runs-root``: for
each candidate run directory it probes for the four artifacts in
``REQUIRED_ARTIFACTS``. No run artifact is ever opened here, and nothing
under ``runs/`` is ever written (specs/64 §10.2).

What it exports
---------------
- ``DatasetRun`` — the resolved (name, label, base id, run dir) record every
  downstream module carries.
- ``resolve_dataset()`` / ``resolve_datasets_from_args()`` — resolution.
- ``available_datasets()`` — the live dataset-key list, derived from disk.
- ``add_common_args()`` — the shared CLI surface of specs/64 §12.4.
- ``outputs_dir()`` / ``artifacts_dir()`` / ``feature_store_dir()`` /
  ``graphs_dir()`` / ``node_state_dir()`` — the only sanctioned way to compose
  a per-run path (§13.6); no module composes one by hand.

The three-layer resolution model (§13.2)
----------------------------------------
1. **name -> base run id** (§13.3). Candidate directory names are split into
   ``<base>[<variant>]`` by ``_RUN_ID_RE``; the base's *dataset key* is the
   base with the leading ``nf_`` and the trailing ``_v<N>`` stripped
   (``nf_unsw_nb15_v3`` -> ``unsw_nb15``). The caller's token is matched
   case-insensitively, exact-first then by unique prefix.
2. **base -> canonical run, by completeness** (§13.4). Among all candidates
   sharing a base, only those carrying all four ``REQUIRED_ARTIFACTS``
   survive. This — not the directory's name — is what separates a real
   dataset run from the incomplete stub, the binary-label experiment and the
   balancer sweep cells that sit beside them in ``runs/``.
3. **ambiguity is a hard error** (§13.5). More than one surviving candidate
   raises, naming every candidate and telling the caller to pass
   ``--run-dir NAME=PATH``. There is deliberately **no** tiebreak: not
   "prefer bare", not "prefer newest", not "prefer longest suffix". A
   "prefer bare" rule would have resolved UNSW to an empty stub as recently
   as 2026-08-26.

``SHAP_GSD_CONFIG`` is deliberately NOT consulted anywhere in this package
(§12.4): it selects a single active run, and these are cross-dataset tools.

No hardcoded dataset name, dataset key, or dataset count appears in this file
(CLAUDE.md CODING STANDARDS 4; specs/64 §13.3). The key set is derived live
from whatever complete runs exist under ``runs/`` at execution time, so a new
dataset appears with no code change.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config  # noqa: E402  [EXISTING, reused]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Package-wide constants (specs/64 §12.2, §12.4, §13.4)
# ---------------------------------------------------------------------------

#: Default output directory for every artifact this package produces.
#: Hardcoded, NOT resolved through ``explore._paths.paths()``/``SHAP_GSD_CONFIG``
#: (specs/64 D3): every ``configs/experiment_*.yaml`` sets a non-empty
#: ``run.dir``, so a config-driven resolution would file a *cross-dataset*
#: artifact inside one dataset's ``runs/`` subtree. Deliberately NOT created
#: at import time — only the ``evalNN_*.py`` entry points ``mkdir`` it, so
#: importing this module has no filesystem side effect.
DEFAULT_OUT_DIR: Path = REPO_ROOT / "outputs" / "figures" / "evaluation"

#: Default directory scanned for candidate run directories.
DEFAULT_RUNS_ROOT: Path = REPO_ROOT / "runs"

#: The completeness predicate of §13.4. All four conjuncts are load-bearing:
#: a 1-of-4 or 2-of-4 predicate lets the balancer sweep cells through, and
#: ``outputs/explanations/summary.csv`` is the single strongest discriminator
#: (no stub, experiment variant or sweep cell has one).
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "artifacts/evaluation/metrics.json",   # per-class P/R/F1/support (eval01, eval02)
    "artifacts/label_map.json",            # the class list (specs/64 D8)
    "outputs/metrics/summary.json",        # phi_F fidelity/stability (eval01)
    "outputs/explanations/summary.csv",    # per-flow explanations exist (eval05 onward)
)

#: Directory names under ``runs/`` that are never run candidates.
#: ``runs/archive/`` holds tarballs and a ``*_pre_rebuild_verify_*`` copy.
SKIP_DIR_NAMES: frozenset[str] = frozenset({"archive"})

#: Base/variant split for a pipeline run-directory name. Non-greedy, so it
#: matches at the *earliest* ``_v<N>``: ``nf_ton_iot_v3_r1_r01_A`` -> base
#: ``nf_ton_iot_v3``, variant ``_r1_r01_A``. The ``nf_<...>_v<N>`` shape is
#: established by ``scripts/run_dataset.py::derive_run_id`` +
#: ``materialize_config``, not invented here (§13.3).
_RUN_ID_RE: re.Pattern[str] = re.compile(r"^(?P<base>nf_.+?_v\d+)(?P<variant>_.+)?$")

#: Trailing dataset-version marker stripped from a base run id to obtain its
#: dataset key (``nf_unsw_nb15_v3`` -> ``unsw_nb15``).
_BASE_KEY_RE: re.Pattern[str] = re.compile(r"^nf_(?P<key>.+)_v\d+$")

#: Relative layout inside a run directory, matching what
#: ``configs/default.yaml`` declares. Used when no config was supplied.
_REL_OUTPUTS = "outputs"
_REL_ARTIFACTS = "artifacts"
_REL_FEATURE_STORE = "feature_store"
_REL_GRAPHS = "graphs"
_REL_NODE_STATE = "node_state_snapshots"


# ---------------------------------------------------------------------------
# scripts/run_dataset.py::derive_run_id — imported, never re-implemented
# ---------------------------------------------------------------------------

_derive_run_id_cache: Callable[[Path], str] | None = None


def _load_derive_run_id() -> Callable[[Path], str]:
    """Import ``derive_run_id`` from ``scripts/run_dataset.py``.

    ``scripts/`` is a CLI directory, not an importable package (it has no
    ``__init__.py``, and specs/64 §13.3 forbids adding one), so the module is
    loaded by file location with ``importlib``. Re-implementing the
    sanitisation here instead would let the pipeline's run-id grammar and this
    package's understanding of it drift apart.

    Returns:
        The pipeline's own ``derive_run_id(csv_path) -> str`` function.
    """
    global _derive_run_id_cache
    if _derive_run_id_cache is not None:
        return _derive_run_id_cache

    module_path = REPO_ROOT / "scripts" / "run_dataset.py"
    spec = importlib.util.spec_from_file_location("_shapgsd_run_dataset", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load derive_run_id from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _derive_run_id_cache = module.derive_run_id
    return _derive_run_id_cache


def normalise_token(token: str) -> str:
    """Sanitise a caller-supplied dataset token into run-id alphabet.

    Delegates to the pipeline's own ``derive_run_id`` so ``--dataset ToN-IoT``
    and ``--dataset ton_iot`` normalise identically, using exactly the rule
    that produced the run-directory names in the first place.

    Args:
        token: Raw ``--dataset`` token, e.g. ``"ToN-IoT"``.

    Returns:
        The lowercased, ``_``-collapsed form, e.g. ``"ton_iot"``.
    """
    derive_run_id = _load_derive_run_id()
    return derive_run_id(Path(f"{token}.csv"))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetRun:
    """One resolved dataset -> canonical run directory binding.

    Attributes:
        name: The caller's ``--dataset`` token (or, under ``--all-datasets``,
            the derived dataset key).
        label: Paper-facing display label (``--label NAME=TEXT``), defaulting
            to ``name``. This is what lands in every output's ``dataset``
            column.
        base_run_id: The run-id base this run belongs to, e.g.
            ``"nf_unsw_nb15_v3"``.
        run_dir: Absolute path to the canonical run directory.
        variant_suffix: ``""`` for a bare base run, else the variant suffix
            (``"_stub"``, ``"_binary"``, ``"_r1_r01_A"``, ...).
        config_path: The experiment YAML this run was resolved through, when
            ``--config NAME=PATH`` was used; otherwise ``None``.
        cfg: The merged ``load_config()`` result when ``config_path`` is set;
            otherwise ``None``.
    """

    name: str
    label: str
    base_run_id: str
    run_dir: Path
    variant_suffix: str
    # ``compare=False`` keeps both out of the generated ``__eq__``/``__hash__``:
    # ``cfg`` is a dict, which is unhashable, and a frozen dataclass otherwise
    # generates a ``__hash__`` that raises ``TypeError`` the moment a
    # ``--config``-resolved run lands in a set or a dict key. Two DatasetRuns
    # naming the same run dir are the same run regardless of which config was
    # used to reach it, so excluding them is also semantically right.
    config_path: Path | None = field(default=None, compare=False)
    cfg: dict | None = field(default=None, compare=False)


@dataclass(frozen=True)
class RunCandidate:
    """One directory under ``runs_root`` that parses as a pipeline run dir.

    Attributes:
        run_dir: Absolute path to the directory.
        base_run_id: The parsed base, e.g. ``"nf_ton_iot_v3"``.
        variant_suffix: ``""`` for a bare base run, else e.g. ``"_binary"``.
        missing_artifacts: The subset of ``REQUIRED_ARTIFACTS`` not present,
            in ``REQUIRED_ARTIFACTS`` order. Empty tuple => complete.
    """

    run_dir: Path
    base_run_id: str
    variant_suffix: str
    missing_artifacts: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        """Whether this candidate satisfies the §13.4 completeness predicate."""
        return not self.missing_artifacts


class DatasetResolutionError(RuntimeError):
    """Raised when a dataset name cannot be resolved to exactly one run dir.

    Covers all three failure modes of §13.3-§13.5: an unmatched or ambiguous
    dataset name, zero complete candidates for a base, and more than one
    complete candidate for a base. Never a silent pick.
    """


# ---------------------------------------------------------------------------
# Layer 0 — scanning runs/
# ---------------------------------------------------------------------------


def missing_required_artifacts(run_dir: Path) -> tuple[str, ...]:
    """Return the ``REQUIRED_ARTIFACTS`` entries absent from ``run_dir``.

    Args:
        run_dir: Candidate run directory.

    Returns:
        Missing relative paths, in ``REQUIRED_ARTIFACTS`` order. Empty when
        the directory satisfies the §13.4 completeness predicate.
    """
    return tuple(rel for rel in REQUIRED_ARTIFACTS if not (run_dir / rel).is_file())


def parse_run_dir_name(dir_name: str) -> tuple[str, str] | None:
    """Split a run-directory name into ``(base_run_id, variant_suffix)``.

    Args:
        dir_name: Bare directory name, e.g. ``"nf_ton_iot_v3_binary"``.

    Returns:
        ``(base, variant)`` with ``variant == ""`` for a bare base run, or
        ``None`` when the name is not a pipeline run-id shape at all.
    """
    match = _RUN_ID_RE.match(dir_name)
    if match is None:
        return None
    return match.group("base"), match.group("variant") or ""


def dataset_key_for_base(base_run_id: str) -> str:
    """Derive the caller-facing dataset key from a base run id.

    ``nf_unsw_nb15_v3`` -> ``unsw_nb15``; ``nf_ton_iot_v3`` -> ``ton_iot``.

    Args:
        base_run_id: A base run id as produced by :func:`parse_run_dir_name`.

    Returns:
        The dataset key (the base with ``nf_`` and ``_v<N>`` stripped).
    """
    match = _BASE_KEY_RE.match(base_run_id)
    if match is None:  # pragma: no cover - _RUN_ID_RE guarantees the shape
        return base_run_id
    return match.group("key")


#: Roots whose per-candidate accept/reject lines have already been logged in
#: this process. The filesystem is ALWAYS re-read (no result caching — a test
#: or a caller may add a run between calls); only the INFO chatter is
#: de-duplicated, so resolving four datasets does not print the same 17-line
#: inventory four times.
_logged_roots: set[Path] = set()


def scan_run_candidates(runs_root: Path) -> list[RunCandidate]:
    """Enumerate every directory under ``runs_root`` that parses as a run dir.

    Recursion is one level only (§13.4 rule 3); ``SKIP_DIR_NAMES`` entries are
    never candidates (rule 1). Every candidate is logged — accepted with its
    variant suffix, or rejected naming the first missing artifact (rule 4). A
    silent rejection is a defect: this is the line that explains "why is my
    dataset missing". The lines are emitted at INFO the first time a given
    root is scanned in this process and at DEBUG thereafter.

    Args:
        runs_root: Directory to scan (normally ``REPO_ROOT/runs``).

    Returns:
        One :class:`RunCandidate` per parseable directory, complete or not.
    """
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        log.warning("runs root does not exist or is not a directory: %s", runs_root)
        return []

    resolved_root = runs_root.resolve()
    level = logging.DEBUG if resolved_root in _logged_roots else logging.INFO
    _logged_roots.add(resolved_root)

    candidates: list[RunCandidate] = []
    for entry in sorted(runs_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in SKIP_DIR_NAMES:
            log.log(level, "skipping %s (in SKIP_DIR_NAMES)", entry.name)
            continue
        parsed = parse_run_dir_name(entry.name)
        if parsed is None:
            log.log(level, "skipping %s (not a pipeline run-id shape)", entry.name)
            continue
        base, variant = parsed
        missing = missing_required_artifacts(entry)
        candidates.append(
            RunCandidate(
                run_dir=entry.resolve(),
                base_run_id=base,
                variant_suffix=variant,
                missing_artifacts=missing,
            )
        )
        if missing:
            log.log(
                level,
                "rejected %s (base=%s variant=%r): missing %s",
                entry.name, base, variant, missing[0],
            )
        else:
            log.log(level, "accepted %s (base=%s variant=%r): 4/4 artifacts",
                    entry.name, base, variant)
    return candidates


def _complete_by_base(runs_root: Path) -> dict[str, list[RunCandidate]]:
    """Group the *complete* candidates under ``runs_root`` by base run id."""
    by_base: dict[str, list[RunCandidate]] = {}
    for cand in scan_run_candidates(runs_root):
        if cand.is_complete:
            by_base.setdefault(cand.base_run_id, []).append(cand)
    return by_base


def _all_by_base(runs_root: Path) -> dict[str, list[RunCandidate]]:
    """Group *every* parseable candidate under ``runs_root`` by base run id."""
    by_base: dict[str, list[RunCandidate]] = {}
    for cand in scan_run_candidates(runs_root):
        by_base.setdefault(cand.base_run_id, []).append(cand)
    return by_base


# ---------------------------------------------------------------------------
# Layer 1/2/3 — resolution
# ---------------------------------------------------------------------------


def available_datasets(runs_root: Path = DEFAULT_RUNS_ROOT) -> list[str]:
    """List the dataset keys that resolve to exactly one canonical run.

    Used by ``--all-datasets`` and by every error message. The key set is
    derived live from disk — no dataset name, key or count is written
    anywhere in this package (specs/64 §13.3, owner constraint 3).

    A base whose completeness predicate leaves more than one candidate is
    *excluded* here (so one future collision cannot break every script) but
    logged at WARNING naming the base and every surviving candidate, so the
    exclusion is never silent. ``resolve_dataset()`` on that same name still
    raises, per §13.5.

    A dataset key claimed by two different bases (e.g. a hypothetical
    ``nf_x_v3`` and ``nf_x_v4``) is likewise excluded with a WARNING; naming
    it explicitly raises.

    Args:
        runs_root: Directory scanned for candidate run dirs.

    Returns:
        Sorted dataset keys, e.g. ``["bot_iot", "cicids2018", ...]``.
    """
    by_base = _complete_by_base(runs_root)
    key_to_bases: dict[str, list[str]] = {}
    for base, cands in by_base.items():
        if len(cands) > 1:
            log.warning(
                "base %s has %d complete candidates (%s); excluded from "
                "--all-datasets. Pass --run-dir <name>=<path> to pick one.",
                base, len(cands),
                ", ".join(sorted(c.run_dir.name for c in cands)),
            )
            continue
        key_to_bases.setdefault(dataset_key_for_base(base), []).append(base)

    keys: list[str] = []
    for key, bases in key_to_bases.items():
        if len(bases) > 1:
            log.warning(
                "dataset key %r is claimed by %d base run ids (%s); excluded "
                "from --all-datasets. Pass --run-dir %s=<path> to pick one.",
                key, len(bases), ", ".join(sorted(bases)), key,
            )
            continue
        keys.append(key)
    return sorted(keys)


def _match_key(token: str, keys: Sequence[str]) -> str:
    """Match a caller token against known dataset keys: exact, then prefix.

    Args:
        token: Normalised caller token (see :func:`normalise_token`).
        keys: Known dataset keys.

    Returns:
        The single matching key.

    Raises:
        DatasetResolutionError: When zero or more than one key matches. A
            prefix matching two keys never silently picks one (§13.3).
    """
    if token in keys:
        return token
    prefixed = sorted(k for k in keys if k.startswith(token))
    if len(prefixed) == 1:
        return prefixed[0]
    if not prefixed:
        raise DatasetResolutionError(
            f"dataset name {token!r} matches no run directory under runs/. "
            f"Known dataset keys: {sorted(keys)!r}. "
            f"Pass --run-dir {token}=<path> to point at a run directory "
            f"explicitly."
        )
    raise DatasetResolutionError(
        f"dataset name {token!r} is an ambiguous prefix of {prefixed!r}. "
        f"Give the full key, or pass --run-dir {token}=<path>."
    )


def _describe_incomplete(cands: Iterable[RunCandidate]) -> str:
    """Render candidate directories with their first missing artifact."""
    parts = [
        f"{c.run_dir.name} (missing {c.missing_artifacts[0]})"
        if c.missing_artifacts else f"{c.run_dir.name} (complete)"
        for c in sorted(cands, key=lambda c: c.run_dir.name)
    ]
    return ", ".join(parts) if parts else "<none>"


def _dataset_run_from_dir(
    name: str,
    run_dir: Path,
    label: str | None,
    config_path: Path | None = None,
    cfg: dict | None = None,
) -> DatasetRun:
    """Build a :class:`DatasetRun` for an explicitly-pinned run directory.

    Used by ``--run-dir``/``--config``, which bypass layers 1-2 but still
    assert the four required artifacts (§13.5).

    Args:
        name: Caller's dataset token.
        run_dir: The pinned run directory.
        label: Display label, or ``None`` to default to ``name``.
        config_path: Experiment YAML, when resolved via ``--config``.
        cfg: Merged config, when resolved via ``--config``.

    Returns:
        The resolved :class:`DatasetRun`.

    Raises:
        DatasetResolutionError: When the directory is missing, or fails the
            completeness predicate.
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise DatasetResolutionError(
            f"pinned run directory for {name!r} does not exist: {run_dir}"
        )
    missing = missing_required_artifacts(run_dir)
    if missing:
        raise DatasetResolutionError(
            f"pinned run directory for {name!r} is not a complete run: "
            f"{run_dir} is missing {list(missing)!r} "
            f"(all of {list(REQUIRED_ARTIFACTS)!r} are required)."
        )
    parsed = parse_run_dir_name(run_dir.name)
    base, variant = parsed if parsed is not None else (run_dir.name, "")
    return DatasetRun(
        name=name,
        label=label if label else name,
        base_run_id=base,
        run_dir=run_dir,
        variant_suffix=variant,
        config_path=config_path,
        cfg=cfg,
    )


def resolve_dataset(
    name: str,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    overrides: dict[str, Path] | None = None,
    *,
    config_overrides: dict[str, Path] | None = None,
    labels: dict[str, str] | None = None,
) -> DatasetRun:
    """Resolve one dataset name to exactly one canonical run directory.

    Implements specs/64 §13.3-§13.5. Ambiguity is a hard error at every
    layer; no tiebreak is ever applied.

    Args:
        name: The caller's ``--dataset`` token, e.g. ``"unsw"``.
        runs_root: Directory scanned for candidate run dirs.
        overrides: ``--run-dir`` map, ``{name: path}``. An entry for ``name``
            bypasses layers 1-2 entirely (still completeness-checked).
        config_overrides: ``--config`` map, ``{name: experiment_yaml}``. The
            run directory comes from the merged config's ``run.dir``, and the
            merged config is carried on the result. Mutually exclusive with
            ``overrides`` for the same name.
        labels: ``--label`` map, ``{name: display_text}``.

    Returns:
        The resolved :class:`DatasetRun`.

    Raises:
        DatasetResolutionError: On an unmatched or ambiguous name, on zero
            complete candidates for the matched base, on more than one
            complete candidate, or when both ``--run-dir`` and ``--config``
            pin the same name.
    """
    overrides = overrides or {}
    config_overrides = config_overrides or {}
    labels = labels or {}
    label = labels.get(name)

    if name in overrides and name in config_overrides:
        raise DatasetResolutionError(
            f"dataset {name!r} is pinned by both --run-dir and --config; "
            f"pass exactly one."
        )

    if name in overrides:
        log.info("dataset %r pinned by --run-dir to %s", name, overrides[name])
        return _dataset_run_from_dir(name, Path(overrides[name]), label)

    if name in config_overrides:
        cfg_path = Path(config_overrides[name])
        if not cfg_path.is_absolute():
            cfg_path = REPO_ROOT / cfg_path
        cfg = load_config(cfg_path)
        run_dir_rel = cfg.get("run", {}).get("dir", "") or ""
        if not run_dir_rel:
            raise DatasetResolutionError(
                f"config {cfg_path} for dataset {name!r} sets no run.dir, so "
                f"it names no run directory; pass --run-dir {name}=<path> "
                f"instead."
            )
        run_dir = Path(run_dir_rel)
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        log.info("dataset %r pinned by --config %s -> %s", name, cfg_path, run_dir)
        return _dataset_run_from_dir(name, run_dir, label, config_path=cfg_path, cfg=cfg)

    # Layer 1 — name -> base run id.
    token = normalise_token(name)
    all_by_base = _all_by_base(runs_root)
    key_to_bases: dict[str, list[str]] = {}
    for base in all_by_base:
        key_to_bases.setdefault(dataset_key_for_base(base), []).append(base)
    if not key_to_bases:
        raise DatasetResolutionError(
            f"no pipeline run directories found under {Path(runs_root)!s}; "
            f"cannot resolve dataset {name!r}."
        )
    key = _match_key(token, sorted(key_to_bases))
    bases = sorted(key_to_bases[key])
    if len(bases) > 1:
        raise DatasetResolutionError(
            f"dataset key {key!r} (from {name!r}) is claimed by more than one "
            f"base run id: {bases!r}. No tiebreak is applied — pass "
            f"--run-dir {name}=<path> to name the run explicitly."
        )
    base = bases[0]

    # Layer 2 — base -> canonical run, by completeness.
    candidates = all_by_base[base]
    complete = [c for c in candidates if c.is_complete]

    # Layer 3 — ambiguity, and emptiness, are hard errors.
    if not complete:
        raise DatasetResolutionError(
            f"dataset {name!r} (base {base!r}) has no complete run yet. "
            f"Candidates examined: {_describe_incomplete(candidates)}. "
            f"A complete run carries all of {list(REQUIRED_ARTIFACTS)!r}. "
            f"If the pipeline has not finished for this dataset, that is a "
            f"normal state; pass --run-dir {name}=<path> to analyse a "
            f"specific directory anyway."
        )
    if len(complete) > 1:
        listed = ", ".join(
            f"{c.run_dir.name} (variant_suffix={c.variant_suffix!r})"
            for c in sorted(complete, key=lambda c: c.run_dir.name)
        )
        raise DatasetResolutionError(
            f"dataset {name!r} (base {base!r}) resolves to {len(complete)} "
            f"complete runs: {listed}. No tiebreak is applied — not 'prefer "
            f"bare', not 'prefer newest'. Pass --run-dir {name}=<path> to "
            f"choose one explicitly."
        )

    chosen = complete[0]
    log.info(
        "resolved dataset %r -> %s (base=%s variant=%r)",
        name, chosen.run_dir, chosen.base_run_id, chosen.variant_suffix,
    )
    return DatasetRun(
        name=name,
        label=label if label else name,
        base_run_id=chosen.base_run_id,
        run_dir=chosen.run_dir,
        variant_suffix=chosen.variant_suffix,
        config_path=None,
        cfg=None,
    )


# ---------------------------------------------------------------------------
# Per-run path accessors (§13.6) — no module composes a path by hand
# ---------------------------------------------------------------------------


def _run_path(run: DatasetRun, cfg_section: str, cfg_key: str, rel: str) -> Path:
    """Resolve one per-run directory, preferring the config when available.

    When the run was resolved via ``--config``, the path comes from the
    merged config (already ``run.dir``-prefixed by
    ``src.utils.config._apply_run_dir_prefix``), so config remains the single
    source of path truth. Otherwise it is composed from the run directory
    using the same relative names ``configs/default.yaml`` declares.

    Args:
        run: The resolved dataset run.
        cfg_section: Config section holding the key, e.g. ``"output"``.
        cfg_key: Config key, e.g. ``"outputs_dir"``.
        rel: Fallback relative directory name inside the run dir.

    Returns:
        An absolute directory path.
    """
    if run.cfg is not None:
        value = run.cfg.get(cfg_section, {}).get(cfg_key)
        if value:
            path = Path(value)
            return path if path.is_absolute() else REPO_ROOT / path
    return run.run_dir / rel


def outputs_dir(run: DatasetRun) -> Path:
    """Return the run's ``outputs/`` directory."""
    return _run_path(run, "output", "outputs_dir", _REL_OUTPUTS)


def artifacts_dir(run: DatasetRun) -> Path:
    """Return the run's ``artifacts/`` directory."""
    return _run_path(run, "output", "artifacts_dir", _REL_ARTIFACTS)


def feature_store_dir(run: DatasetRun) -> Path:
    """Return the run's ``feature_store/`` directory."""
    return _run_path(run, "output", "feature_store_dir", _REL_FEATURE_STORE)


def graphs_dir(run: DatasetRun) -> Path:
    """Return the run's DGL ``graphs/`` directory."""
    return _run_path(run, "graph", "dir", _REL_GRAPHS)


def node_state_dir(run: DatasetRun) -> Path:
    """Return the run's ``node_state_snapshots/`` directory."""
    return _run_path(run, "graph", "node_state_dir", _REL_NODE_STATE)


# ---------------------------------------------------------------------------
# Shared CLI surface (§12.4)
# ---------------------------------------------------------------------------

#: Significance threshold on *corrected* p-values (specs/64 Part I §5).
DEFAULT_ALPHA: float = 0.05

#: Default k for top-k rank/overlap agreement — matches the pipeline's own
#: ``scripts/08_metrics.py --top-k`` default and the ``top_k`` key recorded in
#: every ``summary.json``.
DEFAULT_TOP_K: int = 5


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the shared ``evalNN_*.py`` CLI surface of specs/64 §12.4.

    ``SHAP_GSD_CONFIG`` is deliberately not consulted: it selects a single
    active run, and these are cross-dataset tools.

    Args:
        parser: The module's own argument parser.

    Returns:
        The same parser, for chaining.
    """
    parser.add_argument(
        "--dataset", action="append", default=None, metavar="NAME",
        help="Dataset to analyse, by short name (repeatable). Resolved to a "
             "canonical run directory. Omit together with --all-datasets to "
             "get an error naming the available keys.",
    )
    parser.add_argument(
        "--all-datasets", action="store_true",
        help="Analyse every dataset name that resolves to exactly one "
             "canonical run. Mutually exclusive with --dataset.",
    )
    parser.add_argument(
        "--run-dir", action="append", default=None, metavar="NAME=PATH",
        help="Pin a dataset name to an explicit run directory, bypassing the "
             "completeness-based resolution (repeatable).",
    )
    parser.add_argument(
        "--config", action="append", default=None, metavar="NAME=PATH",
        help="Resolve a dataset through an experiment YAML's run.dir "
             "(repeatable). Mutually exclusive with --run-dir for one NAME.",
    )
    parser.add_argument(
        "--label", action="append", default=None, metavar="NAME=TEXT",
        # No example dataset name here on purpose: no dataset name, key or
        # count is written anywhere in this package (specs/64 §13.3).
        help="Paper-facing display label for a dataset, as NAME=TEXT "
             "(repeatable). Defaults to the dataset name itself.",
    )
    parser.add_argument(
        "--runs-root", type=Path, default=DEFAULT_RUNS_ROOT,
        help=f"Directory scanned for candidate run dirs (default: "
             f"{DEFAULT_RUNS_ROOT}).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help=f"Significance threshold on corrected p-values (default: "
             f"{DEFAULT_ALPHA}).",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"k for top-k rank/overlap agreement (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser


def parse_kv_pairs(values: Sequence[str] | None, flag: str) -> dict[str, str]:
    """Parse repeated ``NAME=VALUE`` CLI arguments into a dict.

    Args:
        values: The raw ``action="append"`` list, possibly ``None``.
        flag: The flag name, for error messages (e.g. ``"--run-dir"``).

    Returns:
        ``{name: value}``.

    Raises:
        DatasetResolutionError: On a malformed entry or a duplicated name.
    """
    out: dict[str, str] = {}
    for raw in values or []:
        if "=" not in raw:
            raise DatasetResolutionError(
                f"{flag} expects NAME=VALUE, got {raw!r}"
            )
        name, _, value = raw.partition("=")
        name, value = name.strip(), value.strip()
        if not name or not value:
            raise DatasetResolutionError(
                f"{flag} expects a non-empty NAME and VALUE, got {raw!r}"
            )
        if name in out:
            raise DatasetResolutionError(
                f"{flag} names {name!r} more than once"
            )
        out[name] = value
    return out


def resolve_datasets_from_args(args: argparse.Namespace) -> list[DatasetRun]:
    """Resolve every dataset selected by a parsed :func:`add_common_args` namespace.

    Names given by ``--run-dir``/``--config`` but not by ``--dataset`` are
    included too, so pinning a run is sufficient to analyse it.

    Args:
        args: Namespace produced by a parser carrying :func:`add_common_args`.

    Two names that resolve to the *same* run directory are a hard error, not a
    silent duplicate: every downstream table is keyed on the ``dataset``
    column, so one run entering it twice under two names would double-count
    that dataset in any cross-dataset mean.

    Returns:
        One :class:`DatasetRun` per selected dataset, in selection order
        (sorted key order under ``--all-datasets``).

    Raises:
        DatasetResolutionError: When the selection flags conflict, when
            neither selection flag is given, when two selected names resolve
            to one run directory, or when any name fails to resolve (§13.5).
    """
    run_dirs = {k: Path(v) for k, v in parse_kv_pairs(args.run_dir, "--run-dir").items()}
    configs = {k: Path(v) for k, v in parse_kv_pairs(args.config, "--config").items()}
    labels = parse_kv_pairs(args.label, "--label")
    runs_root = Path(args.runs_root)

    requested: list[str] = list(args.dataset or [])
    if args.all_datasets:
        if requested:
            raise DatasetResolutionError(
                "--all-datasets is mutually exclusive with --dataset"
            )
        requested = available_datasets(runs_root)
        if not requested:
            raise DatasetResolutionError(
                f"--all-datasets found no dataset with a complete run under "
                f"{runs_root}"
            )
    for pinned in list(run_dirs) + list(configs):
        if pinned not in requested:
            requested.append(pinned)

    # Deduplicate the token list, preserving selection order.
    deduped: list[str] = []
    for name in requested:
        if name in deduped:
            log.warning("dataset %r selected more than once; ignoring the "
                        "duplicate", name)
            continue
        deduped.append(name)
    requested = deduped

    if not requested:
        raise DatasetResolutionError(
            f"no dataset selected: pass --dataset NAME (available: "
            f"{available_datasets(runs_root)!r}), --all-datasets, or "
            f"--run-dir NAME=PATH"
        )

    resolved = [
        resolve_dataset(
            name, runs_root, run_dirs,
            config_overrides=configs, labels=labels,
        )
        for name in requested
    ]

    # Two different tokens reaching one run dir (e.g. --dataset unsw
    # --dataset unsw_nb15) would emit every row twice under two `dataset`
    # values and double-count that run in any cross-dataset aggregate.
    by_dir: dict[Path, str] = {}
    for run in resolved:
        if run.run_dir in by_dir:
            raise DatasetResolutionError(
                f"dataset names {by_dir[run.run_dir]!r} and {run.name!r} both "
                f"resolve to {run.run_dir}; that run would be counted twice. "
                f"Select it under one name only."
            )
        by_dir[run.run_dir] = run.name

    unused_labels = sorted(set(labels) - {run.name for run in resolved})
    if unused_labels:
        log.warning(
            "--label %s matched no selected dataset name (selected: %s); those "
            "labels were NOT applied — label keys must match the --dataset "
            "token, not the resolved dataset key",
            unused_labels, [run.name for run in resolved],
        )

    return resolved
