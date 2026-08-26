"""Per-dataset ingestion and schema normalisation for ``explore/evaluation/``.

Implements specs/64 Part II §14.1. This is the single ingestion point every
``evalNN_*.py`` uses to read one resolved :class:`~explore.evaluation._discover.DatasetRun`'s
metrics into a common shape, so no downstream module needs dataset-specific
knowledge of which classes exist or which optional keys a file carries.

What it reads (all read-only; specs/64 §10.2 — this package never writes into
``runs/``):
  - ``<artifacts_dir>/evaluation/metrics.json`` — accuracy, macro/weighted F1,
    per-class P/R/F1/support and the four coverage fields.
  - ``<artifacts_dir>/label_map.json`` — the canonical class list, in
    insertion order = integer-code order (``Benign=0``, rest alphabetical).
    Classes are NEVER hardcoded (specs/64 D8; CLAUDE.md "do NOT hardcode
    class names").
  - ``<outputs_dir>/metrics/summary.json`` / ``summary_temporal.json`` /
    ``summary_novelty.json`` — the phi_F / phi_T / phi_N coalition spaces.
  - ``<outputs_dir>/metrics/stability.csv`` — per-flow intra-run phi std
    (feature coalition space only).
  - ``<feature_store_dir>/train/labels.npy`` — train-split class support, for
    the closed-set denominator (§14.2). Read with ``mmap_mode="r"``
    (CLAUDE.md CODING STANDARDS 8).

Deliberately NOT read: ``outputs/metrics/table2*.txt``. Those are fixed-width
human-readable renderings carrying an em-dash sentinel for undefined cells;
parsing them is fragile and lossy. The machine-readable JSON/CSV siblings
carry the same numbers plus diagnostics (specs/64 D4).

Two live data hazards this module absorbs so no caller has to:

1. ``null`` in the JSON. ``summary_temporal.json``'s per-class dicts carry
   ``"fidelity_minus": null`` wherever the quantity is undefined (live today
   on ToN-IoT). ``float(None)`` raises, so every numeric read goes through
   :func:`as_float`, which maps ``None`` to ``NaN``.
2. Missing intra-run stability outside the feature space.
   ``summary_temporal.json`` / ``summary_novelty.json`` carry no ``stability``
   key at all, and ``stability.csv`` is feature-group phi only. Callers get
   ``NaN`` plus :data:`STABILITY_SCOPE_NOTE`, never a silently absent row
   (specs/64 D10, §14.1.2).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from explore.evaluation._discover import (
    DatasetRun,
    artifacts_dir,
    feature_store_dir,
    outputs_dir,
)

log = logging.getLogger(__name__)

#: The three coalition spaces, in canonical order (specs/64 Part I §4.1).
#: These are *data values* for the ``coalition_space`` column, not identifier
#: names — the column name itself disambiguates the two senses of "temporal"
#: that specs/64 Part I §2 legislates about (specs/64 D6).
COALITION_SPACES: tuple[str, ...] = ("feature", "temporal", "novelty")

#: coalition space -> the ``outputs/metrics/`` file carrying it.
COALITION_SUMMARY_FILES: dict[str, str] = {
    "feature": "summary.json",
    "temporal": "summary_temporal.json",
    "novelty": "summary_novelty.json",
}

#: Emitted in the ``notes`` column of every ``stability_intrarun_*`` row for a
#: non-feature coalition space (§14.1.2).
STABILITY_SCOPE_NOTE: str = (
    "intra-run stability is computed for the feature coalition space only "
    "(scripts/08_metrics.py compute_stability)"
)

#: Value of ``EvalMetrics.macro_f1_convention`` synthesised when the source
#: file predates ``src/model/evaluator.py``'s self-documenting field.
DERIVED_CONVENTION_NOTE: str = (
    "<derived; source file predates evaluator.py's self-documenting "
    "convention field — recomputed from per_class support>"
)


def as_float(value: Any) -> float:
    """Coerce a JSON scalar to ``float``, mapping ``None``/missing to ``NaN``.

    ``summary_temporal.json`` writes JSON ``null`` for undefined fidelity
    cells, which ``json.load`` yields as ``None``; ``float(None)`` raises.
    Every numeric read in this package goes through here.

    Args:
        value: A JSON scalar, ``None``, or anything non-numeric.

    Returns:
        The float value, or ``float("nan")`` when it is ``None`` or not
        convertible.
    """
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


@dataclass(frozen=True)
class EvalMetrics:
    """Normalised contents of one run's ``artifacts/evaluation/metrics.json``.

    Attributes:
        run: The dataset run this was loaded from.
        accuracy: Overall test accuracy as reported by the pipeline.
        macro_f1: Macro-F1 as reported by the pipeline (under its own
            convention — never used as a comparison input; ``eval02``
            recomputes all three denominators from ``per_class``).
        weighted_f1: Weighted F1 as reported by the pipeline.
        macro_f1_convention: The file's self-documenting convention string,
            or :data:`DERIVED_CONVENTION_NOTE` when the field was absent.
        n_classes_total: Size of the full label space.
        n_classes_present_in_test: Classes with non-zero test support.
        classes_absent_from_test: Class names with zero test support.
        n_test_edges: Test-split edge count, or ``None`` when absent.
        per_class: ``{class_name: {precision, recall, f1, support}}``.
        class_names: The canonical class list from ``label_map.json``, in
            integer-code order.
        schema_source: ``"native"`` when the four coverage fields were read
            from the file, ``"derived"`` when they were reconstructed.
        path: The ``metrics.json`` path actually read.
    """

    run: DatasetRun
    accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_f1_convention: str
    n_classes_total: int
    n_classes_present_in_test: int
    classes_absent_from_test: list[str]
    n_test_edges: int | None
    per_class: dict[str, dict[str, float]]
    class_names: list[str]
    schema_source: str
    path: Path

    def support(self, class_name: str) -> float:
        """Return a class's test support, 0.0 when it has no ``per_class`` row.

        Args:
            class_name: Class name as it appears in ``label_map.json``.

        Returns:
            The test-split support count.
        """
        return as_float(self.per_class.get(class_name, {}).get("support", 0.0))

    def is_present_in_test(self, class_name: str) -> bool:
        """Whether a class has non-zero test support (``class_present_in_test``).

        Args:
            class_name: Class name as it appears in ``label_map.json``.

        Returns:
            ``True`` when the class's test support is greater than zero.
        """
        value = self.support(class_name)
        return bool(value > 0)


@dataclass(frozen=True)
class CoalitionSummary:
    """Normalised contents of one ``outputs/metrics/summary*.json``.

    Attributes:
        space: One of :data:`COALITION_SPACES`.
        present: ``False`` when the file does not exist for this run.
        top_k: The ``top_k`` the pipeline used, or ``None``.
        per_class: ``{class_name: {...}}`` exactly as on disk (raw values,
            including ``None``; use :func:`as_float` when reading numbers).
        overall: The file's ``overall`` block.
        diagnostics: The file's ``diagnostics`` block, ``{}`` when absent.
        note: The file's ``note`` string, ``""`` when absent.
        path: The path probed (whether or not it exists).
    """

    space: str
    present: bool
    top_k: int | None
    per_class: dict[str, dict[str, Any]]
    overall: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    path: Path | None = None

    def has_stability(self) -> bool:
        """Whether this coalition space carries intra-run stability at all.

        Only the feature space does (§14.1.2); the other two are read as
        ``NaN`` with :data:`STABILITY_SCOPE_NOTE` rather than omitted.

        Returns:
            ``True`` when any per-class dict carries a ``stability`` key.
        """
        return any("stability" in v for v in self.per_class.values())


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from ``path``.

    Args:
        path: File to read.

    Returns:
        The decoded mapping.
    """
    with open(path) as fh:
        return json.load(fh)


def load_label_map(run: DatasetRun) -> dict[str, int]:
    """Load a run's ``artifacts/label_map.json``.

    Insertion order is the canonical integer-code order (``Benign=0``, rest
    alphabetical — CLAUDE.md). The class list is never hardcoded anywhere in
    this package (specs/64 D8).

    Args:
        run: The resolved dataset run.

    Returns:
        ``{class_name: int_code}`` in file order.

    Raises:
        FileNotFoundError: When the file is absent — impossible for a run that
            passed the §13.4 completeness predicate, so this signals that a
            ``--run-dir`` pin bypassed it or the run changed under us.
    """
    path = artifacts_dir(run) / "label_map.json"
    if not path.is_file():
        raise FileNotFoundError(f"label_map.json not found for {run.name!r}: {path}")
    return _read_json(path)


def load_eval_metrics(run: DatasetRun) -> EvalMetrics:
    """Load and normalise a run's ``artifacts/evaluation/metrics.json``.

    The four coverage fields (``n_classes_total``,
    ``n_classes_present_in_test``, ``classes_absent_from_test``,
    ``macro_f1_convention``) are read when present and **derived** when not,
    with ``schema_source`` recording which happened.

    Every dataset currently on disk carries the full 9-key schema, so the
    derive branch is defensive rather than exercised: those four fields are
    emitted by ``src/model/evaluator.py``, not guaranteed by any schema
    contract, and an older or externally-supplied run directory can still lack
    them. Keeping the branch is the difference between a clearly-marked
    derived value and a ``KeyError`` in the middle of a report.

    Args:
        run: The resolved dataset run.

    Returns:
        The normalised :class:`EvalMetrics`.

    Raises:
        FileNotFoundError: When ``metrics.json`` or ``label_map.json`` is
            absent.
    """
    path = artifacts_dir(run) / "evaluation" / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"metrics.json not found for {run.name!r}: {path}")
    raw = _read_json(path)
    class_names = list(load_label_map(run).keys())
    per_class: dict[str, dict[str, float]] = raw.get("per_class", {}) or {}

    coverage_keys = (
        "n_classes_total",
        "n_classes_present_in_test",
        "classes_absent_from_test",
        "macro_f1_convention",
    )
    native = all(key in raw for key in coverage_keys)

    if native:
        schema_source = "native"
        n_classes_total = int(raw["n_classes_total"])
        n_present = int(raw["n_classes_present_in_test"])
        absent = list(raw["classes_absent_from_test"])
        convention = str(raw["macro_f1_convention"])
    else:
        schema_source = "derived"
        missing = [key for key in coverage_keys if key not in raw]
        log.warning(
            "%s: metrics.json at %s lacks %s; deriving them from label_map.json "
            "+ per_class support (schema_source='derived')",
            run.name, path, missing,
        )
        n_classes_total = len(class_names)
        absent = [
            name for name in class_names
            if as_float(per_class.get(name, {}).get("support", 0.0)) == 0.0
        ]
        n_present = n_classes_total - len(absent)
        convention = str(raw.get("macro_f1_convention", DERIVED_CONVENTION_NOTE))
        if "macro_f1_convention" not in raw:
            convention = DERIVED_CONVENTION_NOTE

    # CLAUDE.md CODING STANDARDS 7 — assertions run in production code.
    assert n_classes_total >= 0, f"negative n_classes_total in {path}"
    assert 0 <= n_present <= n_classes_total, (
        f"n_classes_present_in_test={n_present} outside "
        f"[0, {n_classes_total}] in {path}"
    )

    n_test_edges_raw = raw.get("n_test_edges")
    return EvalMetrics(
        run=run,
        accuracy=as_float(raw.get("accuracy")),
        macro_f1=as_float(raw.get("macro_f1")),
        weighted_f1=as_float(raw.get("weighted_f1")),
        macro_f1_convention=convention,
        n_classes_total=n_classes_total,
        n_classes_present_in_test=n_present,
        classes_absent_from_test=absent,
        n_test_edges=int(n_test_edges_raw) if n_test_edges_raw is not None else None,
        per_class=per_class,
        class_names=class_names,
        schema_source=schema_source,
        path=path,
    )


def load_coalition_summary(run: DatasetRun, space: str) -> CoalitionSummary:
    """Load one coalition space's ``outputs/metrics/summary*.json``.

    An absent file is a normal state, not an error: the returned object has
    ``present=False`` and empty blocks, so the caller emits explicit ``NaN``
    rows rather than dropping the space (specs/64 D10).

    Args:
        run: The resolved dataset run.
        space: One of :data:`COALITION_SPACES`.

    Returns:
        The normalised :class:`CoalitionSummary`.

    Raises:
        ValueError: When ``space`` is not one of :data:`COALITION_SPACES`.
    """
    if space not in COALITION_SUMMARY_FILES:
        raise ValueError(
            f"unknown coalition space {space!r}; expected one of "
            f"{list(COALITION_SPACES)!r}"
        )
    path = outputs_dir(run) / "metrics" / COALITION_SUMMARY_FILES[space]
    if not path.is_file():
        log.warning(
            "%s: no %s coalition-space summary at %s; rows for this space will "
            "be emitted with NaN values",
            run.name, space, path,
        )
        return CoalitionSummary(
            space=space, present=False, top_k=None,
            per_class={}, overall={}, path=path,
        )
    raw = _read_json(path)
    top_k_raw = raw.get("top_k")
    return CoalitionSummary(
        space=space,
        present=True,
        top_k=int(top_k_raw) if top_k_raw is not None else None,
        per_class=raw.get("per_class", {}) or {},
        overall=raw.get("overall", {}) or {},
        diagnostics=raw.get("diagnostics", {}) or {},
        note=str(raw.get("note", "")),
        path=path,
    )


def load_all_coalition_summaries(run: DatasetRun) -> dict[str, CoalitionSummary]:
    """Load all three coalition-space summaries for one run.

    Args:
        run: The resolved dataset run.

    Returns:
        ``{space: CoalitionSummary}`` for every space in
        :data:`COALITION_SPACES`, present or not.
    """
    return {space: load_coalition_summary(run, space) for space in COALITION_SPACES}


def load_stability_frame(run: DatasetRun) -> pd.DataFrame | None:
    """Load a run's per-flow intra-run stability CSV.

    Columns on disk are ``class_name,edge_id,mean_phi_std,max_phi_std`` —
    feature-group phi only. There is no per-flow stability artifact for the
    temporal or novelty coalition spaces (§14.1.2).

    Args:
        run: The resolved dataset run.

    Returns:
        The dataframe, or ``None`` when the file is absent.
    """
    path = outputs_dir(run) / "metrics" / "stability.csv"
    if not path.is_file():
        log.warning("%s: no stability.csv at %s", run.name, path)
        return None
    return pd.read_csv(path)


def load_train_class_support(
    run: DatasetRun, class_names: list[str],
) -> np.ndarray | None:
    """Count train-split rows per class from the feature store.

    Train-split support appears in no metrics file — ``metrics.json`` reports
    test support only — so the closed-set denominator (§14.2) has to come from
    ``feature_store/train/labels.npy``. Read with ``mmap_mode="r"`` per
    CLAUDE.md CODING STANDARDS 8 (NetFlow is 1M+ edges).

    A genuinely absent train feature store is a real on-disk state and is
    reported as ``None`` so the caller can emit an explicit "not computable"
    row rather than silently falling back to another denominator.

    Args:
        run: The resolved dataset run.
        class_names: The canonical class list, in integer-code order; its
            length fixes the returned array's length.

    Returns:
        A ``len(class_names)``-long integer count array, or ``None`` when
        ``feature_store/train/labels.npy`` is absent.
    """
    path = feature_store_dir(run) / "train" / "labels.npy"
    if not path.is_file():
        log.warning(
            "%s: train feature store absent at %s; closed-set denominator is "
            "not computable for this dataset",
            run.name, path,
        )
        return None
    labels = np.load(path, mmap_mode="r")
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=len(class_names))
    # CLAUDE.md CODING STANDARDS 7 — a label outside the label map means the
    # label map and the feature store disagree, which must not pass silently.
    assert counts.shape[0] == len(class_names), (
        f"{run.name}: train labels carry {counts.shape[0]} distinct codes but "
        f"label_map.json declares {len(class_names)} classes ({path})"
    )
    return counts
