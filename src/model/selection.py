"""Shard-aware selection, grid-enumeration and persistence helpers.

Shared by src/model/tuner.py (single-job and shard mode), src/baselines/
_ckpt_meta.py (write_json_atomic only) and scripts/promote_best.py, which
imports build_shard_header/PromoteError/validate_shards from here and stays
a thin CLI wrapper around them (this module is "the single enumeration
authority" for shard logic, specs/35 §V.4). HARD CONTRACT: stdlib-only
imports — this module must be importable on a login node with no
torch/dgl/numpy/yaml installed (specs/34 §4.3). Pinned by
tests/test_shard_selection.py.
"""

import hashlib
import itertools
import json
import os
import re
from collections import Counter
from datetime import datetime
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
        """Filesystem-safe slug for one (axis, value) pair.

        Injective in ``value`` (not merely "pretty"): a literal ``-``
        (a minus sign, e.g. in ``-1`` or ``-0.5``) is mapped to the
        explicit token ``"neg"`` BEFORE the generic non-alphanumeric-run
        collapse runs. Without this, ``1``/``-1`` and ``0.1``/``-0.1``
        would collapse to the IDENTICAL slug once the leading ``-`` is
        swallowed by the punctuation collapse and stripped from the
        string ends — two shards would then resolve to the same
        filename, and write_json_atomic()'s unconditional os.replace()
        would let whichever shard flushes last silently overwrite the
        other's completed trials. Today's real grid (fanouts/hidden_size,
        all positive ints/lists) never exercises this path, but the fix
        must not depend on that — a future signed axis value must still
        get a distinct slug.
        """
        text = str(value).replace("-", "neg")
        return axis[:3] + re.sub(r"[^0-9A-Za-z]+", "-", text).strip("-")

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


def build_shard_header(
    shard: dict,
    selection_metric_used: str,
    *,
    pbs_jobid: str | None = None,
    started_at: str | None = None,
) -> dict:
    """Build the on-disk shard-file header dict (specs/35 §III.5).

    Single source of truth for the 12-key header schema written by
    ``HyperparameterTuner.run()`` in shard mode. Both the tuner's real
    write path and any test fixture that needs to fabricate a shard file
    (e.g. ``tests/test_promote_best.py``) call this function instead of
    each maintaining its own copy of the field list, so a future header
    field cannot drift between the writer and a test fixture's
    hand-built header (the failure mode this function closes).

    Args:
        shard: shard spec from :func:`resolve_shard`, plus a
            ``"grid_fingerprint"`` key attached by the caller (e.g.
            ``scripts/03_tune.py``'s ``compute_grid_fingerprint`` call).
        selection_metric_used: the training-curve key this shard's trials
            are selected on (a ``SELECTION_METRIC_CURVE_KEY`` lookup
            result).
        pbs_jobid: overrides ``os.environ.get("PBS_JOBID", "none")`` when
            given — lets tests pin a deterministic value instead of
            depending on the environment.
        started_at: overrides ``datetime.now().astimezone().isoformat()``
            when given — same reason.

    Returns:
        The header dict, ``schema_version`` pinned to
        ``SHARD_SCHEMA_VERSION``.
    """
    if pbs_jobid is None:
        pbs_jobid = os.environ.get("PBS_JOBID", "none")
    if started_at is None:
        started_at = datetime.now().astimezone().isoformat()
    return {
        "schema_version": SHARD_SCHEMA_VERSION,
        "shard_index": shard["shard_index"],
        "num_shards": shard["num_shards"],
        "n_total": shard["n_total"],
        "partition_scheme": shard["partition_scheme"],
        "shard_axes": shard["shard_axes"],
        "owned_values": shard["owned_values"],
        "owned_indices": shard["owned_indices"],
        "selection_metric_used": selection_metric_used,
        "grid_fingerprint": shard["grid_fingerprint"],
        "pbs_jobid": pbs_jobid,
        "started_at": started_at,
    }


def write_json_atomic(path: Path, obj: object, *, sort_keys: bool = False) -> None:
    """Write obj as JSON to path via <name>.tmp + os.replace (atomic).

    The temp file sits in the SAME directory as the target so os.replace
    never crosses a filesystem boundary (Lustre-safe). flush + fsync
    before replace, so a walltime kill / node failure / OOM can never
    leave a truncated file at the final name (specs/34 §3.5).

    This is the single shared atomic-JSON-write primitive for the project:
    src.model.tuner, scripts.promote_best AND src.baselines._ckpt_meta
    (via ``from src.model.selection import write_json_atomic``) all call
    this one implementation rather than each hand-maintaining a
    tmp-file + os.replace copy that can silently drift.

    Args:
        path: destination path.
        obj: JSON-serialisable value.
        sort_keys: passed through to ``json.dump``. Default False matches
            every existing tuning-artifact write (tuning_results.json,
            best_params.json, shard files); callers that want a
            deterministic key order (e.g. the PGExplainer checkpoint
            sidecar) pass True.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=sort_keys)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class PromoteError(RuntimeError):
    """Raised with a fully-worded, operator-actionable message.

    Raised by :func:`validate_shards` (and by ``scripts/promote_best.py``'s
    ``gather_shard_files``/``load_shard_files``, which raise this same
    class for their own gate failures).
    """


#: Header fields that must be identical across every shard file for a merge
#: to be admissible (specs/35 §V.4 check 1). ``grid_fingerprint`` is the R6
#: mixed-grid guard: shard files produced under different effective configs
#: must never be silently merged.
_AGREEMENT_FIELDS: tuple[str, ...] = (
    "num_shards",
    "n_total",
    "partition_scheme",
    "shard_axes",
    "selection_metric_used",
    "grid_fingerprint",
)

#: Of _AGREEMENT_FIELDS, the subset the shard-index-cover check (3) and the
#: global-coverage check (5) below treat as load-bearing integers, not just
#: opaque values to compare for equality. A header where every file agrees
#: on the SAME missing/non-int value for one of these would pass the bare
#: equality check in the loop below and then silently disable checks 3/5
#: via their ``isinstance`` guards -- the exact "partial grid promoted
#: silently" failure this module exists to prevent. They are therefore
#: validated as present-and-int inside the agreement loop itself, so a
#: consistently-wrong header fails closed instead of skipping checks.
_REQUIRED_INT_FIELDS: frozenset[str] = frozenset({"num_shards", "n_total"})


def _canon(value: object) -> str:
    """Canonical JSON rendering of a header value, for equality grouping."""
    return json.dumps(value, sort_keys=True)


def validate_shards(shards: list[dict], num_shards: int | None = None) -> None:
    """Fail-closed completeness + consistency gate (specs/35 §V.4).

    All checks run; failures are accumulated and raised together in one
    :class:`PromoteError` so the operator sees the full picture in one
    pass. It must be impossible to promote from a partial grid silently
    (specs/34 §4.1) -- including when every shard file agrees on a
    missing or non-int ``num_shards``/``n_total`` header field: that case
    is caught by the presence-and-type check folded into check 1 below,
    rather than allowed to silently skip checks 3 and 5.

    Args:
        shards: loaded shard dicts, each ``{"file": <name>, "header":
            ..., "results": ...}`` (``scripts/promote_best.py``'s
            ``load_shard_files`` output shape).
        num_shards: optional operator-supplied cross-check; the headers
            remain the authority.

    Raises:
        PromoteError: naming every violated check.
    """
    failures: list[str] = []

    # 1. Header agreement across all files, PLUS presence-and-type
    #    validation for the two fields (num_shards, n_total) that checks
    #    3 and 5 below treat as load-bearing integers. A field that is
    #    missing or non-int must fail here, not be allowed to reach the
    #    downstream isinstance guards and disable those checks silently.
    for field in _AGREEMENT_FIELDS:
        by_value: dict[str, list[str]] = {}
        bad_type_files: list[str] = []
        for shard in shards:
            value = shard["header"].get(field)
            if field in _REQUIRED_INT_FIELDS and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                bad_type_files.append(
                    f"{shard['file']} (got {value!r}, type "
                    f"{type(value).__name__})"
                )
            by_value.setdefault(_canon(value), []).append(shard["file"])
        if bad_type_files:
            failures.append(
                f"header field {field!r} must be present and an int in "
                "every shard file -- required for the shard-index-cover "
                f"and global-coverage checks; refusing to promote from a "
                f"possibly partial grid: {bad_type_files}"
            )
        if len(by_value) > 1:
            detail = "; ".join(
                f"{value} in {files}" for value, files in sorted(by_value.items())
            )
            if field == "grid_fingerprint":
                failures.append(
                    "header field 'grid_fingerprint' disagrees — shard files "
                    "were produced under different effective configs — "
                    f"refusing to merge: {detail}"
                )
            else:
                failures.append(
                    f"header field {field!r} disagrees across shard files: "
                    f"{detail}"
                )

    # Authority values for the remaining checks: the first header's. If the
    # agreement/type check above already failed for these fields, that
    # failure is recorded; the remaining checks still run (guarded by
    # isinstance) so the report is complete, but validate_shards() as a
    # whole still raises because `failures` is already non-empty.
    header0 = shards[0]["header"]
    agreed_num_shards = header0.get("num_shards")
    agreed_n_total = header0.get("n_total")

    # 2. Operator --num-shards cross-check (headers are the authority).
    if num_shards is not None and num_shards != agreed_num_shards:
        failures.append(
            f"--num-shards {num_shards} does not match the shard headers' "
            f"num_shards {agreed_num_shards}"
        )

    # 3. Shard-index cover: exactly one file per k in range(num_shards).
    index_files: dict[object, list[str]] = {}
    for shard in shards:
        index_files.setdefault(shard["header"].get("shard_index"), []).append(
            shard["file"]
        )
    if isinstance(agreed_num_shards, int) and not isinstance(agreed_num_shards, bool):
        for k in range(agreed_num_shards):
            if k not in index_files:
                failures.append(
                    f"no shard file claims shard_index {k} of "
                    f"{agreed_num_shards}"
                )
    for k, files in sorted(index_files.items(), key=lambda kv: _canon(kv[0])):
        if len(files) > 1:
            failures.append(
                f"shard_index {k} is claimed by multiple files: {files}"
            )

    # 4. Per-file completeness against each file's OWN owned_indices.
    for shard in shards:
        name = shard["file"]
        owned = shard["header"].get("owned_indices") or []
        trials = [r["trial"] for r in shard["results"]]
        counts = Counter(trials)
        missing_trials = sorted(set(owned) - set(trials))
        foreign = sorted(set(trials) - set(owned))
        dupes = sorted(t for t, c in counts.items() if c > 1)
        if missing_trials:
            failures.append(
                f"{name} is incomplete: missing trials {missing_trials} — "
                "re-run that shard"
            )
        if foreign:
            failures.append(
                f"{name} contains trials it does not own per its own header: "
                f"{foreign}"
            )
        if dupes:
            failures.append(f"{name} contains duplicate trial records: {dupes}")

    # 5. Global exactness — asserted independently of checks 3+4: the union
    #    of all trial indices must be exactly range(n_total).
    trial_files: dict[object, list[str]] = {}
    for shard in shards:
        for record in shard["results"]:
            trial_files.setdefault(record["trial"], []).append(shard["file"])
    if isinstance(agreed_n_total, int) and not isinstance(agreed_n_total, bool):
        gaps = sorted(set(range(agreed_n_total)) - set(trial_files))
        if gaps:
            failures.append(
                f"merged results do not cover the full grid: missing trial "
                f"indices {gaps} of range({agreed_n_total})"
            )
    global_dupes = {
        t: files for t, files in trial_files.items() if len(files) > 1
    }
    for t in sorted(global_dupes, key=_canon):
        failures.append(
            f"trial index {t} appears in more than one shard file: "
            f"{global_dupes[t]}"
        )
    extraneous = sorted(
        t for t in trial_files
        if isinstance(agreed_n_total, int)
        and not isinstance(agreed_n_total, bool)
        and not (isinstance(t, int) and 0 <= t < agreed_n_total)
    )
    if extraneous:
        failures.append(
            f"trial indices outside range({agreed_n_total}): {extraneous}"
        )

    if failures:
        raise PromoteError(
            "refusing to promote — {} gate failure(s):\n  {}".format(
                len(failures), "\n  ".join(failures)
            )
        )
