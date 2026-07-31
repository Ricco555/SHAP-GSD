"""Shard-aware selection, grid-enumeration and persistence helpers.

Shared by src/model/tuner.py (single-job and shard mode) and
scripts/promote_best.py. HARD CONTRACT: stdlib-only imports — this module
must be importable on a login node with no torch/dgl/numpy/yaml installed
(specs/34 §4.3). Pinned by tests/test_shard_selection.py.
"""

import hashlib
import itertools
import json
import os
import re
from pathlib import Path

#: Prefix shared by every per-shard results file; scripts/promote_best.py
#: globs on ``SHARD_FILE_PREFIX + "*.json"`` (the ``.json`` suffix excludes
#: in-flight ``.json.tmp`` files by construction — see write_json_atomic).
SHARD_FILE_PREFIX: str = "tuning_results_shard"

#: Version of the shard-file on-disk schema ({"header": ..., "results": ...}).
SHARD_SCHEMA_VERSION: int = 1

#: Axes that may never be chosen as shard-partition axes. ``batch_size`` is
#: the dominant-cost axis (measured 2.55-2.64x per-epoch wall-time spread,
#: specs/34 §1.2): a shard that does not sweep it in full is the actual
#: source of shard walltime imbalance.
FORBIDDEN_SHARD_AXES: frozenset[str] = frozenset({"batch_size"})


def enumerate_grid(search_space: dict) -> list[dict]:
    """Enumerate the full hyperparameter grid, one dict per trial.

    The single authority on grid-enumeration order: ``itertools.product``
    over ``search_space`` values in YAML key order. The list index of each
    returned dict IS the global trial index. Shared by
    ``HyperparameterTuner._configs`` and ``resolve_shard`` so the tuner and
    the shard resolver can never drift onto different enumeration orders.

    Args:
        search_space: dict of axis_name -> list of values, in grid key order.

    Returns:
        List of trial dicts keyed by axis name, in product order.
    """
    keys = list(search_space.keys())
    return [dict(zip(keys, combo))
            for combo in itertools.product(*(search_space[k] for k in keys))]


def select_best(results: list[dict]) -> dict | None:
    """Return the winning trial RECORD (never an index).

    Iterates in ascending record["trial"] order with a strict >
    comparison on record["best_val_macro_f1"], so ties resolve to the
    lowest GLOBAL trial index — byte-identical semantics to the loop this
    replaces (tuner.py:265-267 as of specs/33). Works identically on a
    dense results list (single-job) and a sparse one (shard / merged
    shards), because it never uses list position. Returns None on an
    empty list.
    """
    best: dict | None = None
    best_val = -1.0
    for record in sorted(results, key=lambda r: r["trial"]):
        if record["best_val_macro_f1"] > best_val:
            best_val = record["best_val_macro_f1"]
            best = record
    return best


def compute_grid_fingerprint(
    model_block: dict, search_space: dict, trial_block: dict
) -> str:
    """Fingerprint the effective tuning configuration (specs/34 §4.2, R6).

    Canonical-JSON sha256 over the ENTIRE effective post-merge ``model:``
    block, the live ``search_space``, and the int-coerced ``trial:`` budget
    — so shard files produced under different effective configs can never
    be silently merged. No ``default=`` hook: a non-JSON-native value
    raises ``TypeError`` — fail loud, never coerce.

    Args:
        model_block:  the effective, post-merge ``cfg["model"]`` dict at
            trial-launch time.
        search_space: ``grid_cfg["search_space"]`` as parsed.
        trial_block:  ``{"max_epochs": int, "patience": int}`` rebuilt from
            ``resolve_trial_settings``' int-coerced output.

    Returns:
        ``"sha256:" + <64 hex chars>``.
    """
    payload = {
        "model": model_block,
        "search_space": search_space,
        "trial": trial_block,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def resolve_shard(search_space: dict, shard_axes: str, shard: str) -> dict:
    """Resolve the ``--shard-axes``/``--shard`` CLI pair into a shard spec.

    Ownership is defined by axis VALUES, never by index arithmetic: shard
    ``k`` of ``N`` owns exactly the trials whose recorded values on the
    chosen axes equal the ``k``-th combination of the product enumeration
    over those axes (in ``search_space`` key order). Grid-shape-agnostic
    by construction (specs/34 §3.1, revision 3).

    Args:
        search_space: the live ``search_space`` dict (axis -> value list).
        shard_axes:   raw CLI string, comma-separated axis names.
        shard:        raw CLI string ``"K/N"``.

    Returns:
        dict with keys ``partition_scheme``, ``shard_axes``,
        ``owned_values``, ``owned_indices``, ``shard_index``,
        ``num_shards``, ``n_total``, ``filename``.

    Raises:
        ValueError: empty/duplicate/unknown axis names, a forbidden axis
            (``batch_size``), a malformed ``K/N``, ``N`` not equal to the
            product of the chosen axes' cardinalities, or ``k`` out of
            range.
    """
    axes = [a.strip() for a in shard_axes.split(",") if a.strip()]
    if not axes:
        raise ValueError(f"--shard-axes is empty: {shard_axes!r}")
    if len(axes) != len(set(axes)):
        raise ValueError(f"--shard-axes contains duplicate axis names: {axes}")

    unknown = [a for a in axes if a not in search_space]
    if unknown:
        raise ValueError(
            f"--shard-axes names unknown axis(es) {unknown} — not present in "
            f"the live search_space. Valid axes: {sorted(search_space)}."
        )

    forbidden = sorted(a for a in axes if a in FORBIDDEN_SHARD_AXES)
    if forbidden:
        raise ValueError(
            f"--shard-axes must not include {forbidden}: batch_size is the "
            "dominant-cost axis (measured 2.55-2.64x per-epoch wall-time "
            "spread) and every shard must sweep it in full. See specs/34 §1.2."
        )

    # Normalize axis order to search_space key order (the grid's own
    # itertools.product order), regardless of the order typed on the CLI.
    normalized_axes = [k for k in search_space if k in axes]

    match = re.fullmatch(r"(\d+)/(\d+)", shard)
    if match is None:
        raise ValueError(
            f"--shard must have the form K/N (e.g. 4/9), got {shard!r}"
        )
    k, n = int(match.group(1)), int(match.group(2))

    n_computed = 1
    for axis in normalized_axes:
        n_computed *= len(search_space[axis])
    if n != n_computed:
        raise ValueError(
            f"--shard N={n} does not match the product of the chosen axes' "
            f"cardinalities computed from the live search_space ({n_computed})."
        )
    if not 0 <= k < n:
        raise ValueError(f"--shard K={k} is out of range [0, {n}).")

    combos = list(itertools.product(*(search_space[a] for a in normalized_axes)))
    owned_values = dict(zip(normalized_axes, combos[k]))

    grid = enumerate_grid(search_space)
    owned_indices = [
        i for i, c in enumerate(grid)
        if all(c[a] == v for a, v in owned_values.items())
    ]

    def _slug(axis: str, value: object) -> str:
        return axis[:3] + re.sub(r"[^0-9A-Za-z]+", "-", str(value)).strip("-")

    filename = (
        SHARD_FILE_PREFIX
        + "_"
        + "_".join(_slug(a, owned_values[a]) for a in normalized_axes)
        + ".json"
    )

    return {
        "partition_scheme": "axis",
        "shard_axes": normalized_axes,
        "owned_values": owned_values,
        "owned_indices": owned_indices,
        "shard_index": k,
        "num_shards": n,
        "n_total": len(grid),
        "filename": filename,
    }


def write_json_atomic(path: Path, obj: object) -> None:
    """Write obj as JSON to path via <name>.tmp + os.replace (atomic).

    The temp file sits in the SAME directory as the target so os.replace
    never crosses a filesystem boundary (Lustre-safe). flush + fsync
    before replace, so a walltime kill / node failure / OOM can never
    leave a truncated file at the final name (specs/34 §3.5).
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
