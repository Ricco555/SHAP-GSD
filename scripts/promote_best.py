"""
Promote step for sharded Phase 3 (specs/34 §4.1, specs/35 Part V).

Merges the per-shard result files that N concurrent `scripts/03_tune.py
--shard-axes ... --shard K/N` jobs wrote into a shared
`runs/<run_id>/artifacts/tuning/` directory, runs a fail-closed
completeness + consistency gate over them, and — only if the gate passes —
emits the exact artifacts a single-job Phase 3 run produces:

  tuning_results.json   bare list of all trial records, sorted by global
                        trial index (same schema as a single-job run)
  best_params.json      {"best_params", "best_val_macro_f1",
                         "selection_metric_used", "best_trial"}
                        — the contract scripts/04_train.py consumes

HARD CONTRACT: stdlib-only import chain (plus src.model.selection, itself
stdlib-only) — this script must run on a login node with no
torch/dgl/numpy/yaml installed (specs/34 §4.3). Idempotent: a pure
function of the shard files; re-running overwrites both outputs with
identical bytes.

Usage:
  python scripts/promote_best.py --tuning-dir runs/<run_id>/artifacts/tuning \
      [--num-shards 9]

Exit status: 0 on success; 1 on any gate failure, with every failure named.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.selection import (
    SHARD_FILE_PREFIX,
    select_best,
    write_json_atomic,
)

logger = logging.getLogger(__name__)

#: Header fields that must be identical across every shard file for a merge
#: to be admissible (specs/35 §V.4 check 1). `grid_fingerprint` is the R6
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


class PromoteError(RuntimeError):
    """Raised with a fully-worded, operator-actionable message."""


def gather_shard_files(tuning_dir: Path) -> list[Path]:
    """Return every shard result file under tuning_dir, sorted by name.

    Globs ``SHARD_FILE_PREFIX + "*.json"`` — the ``.json`` suffix excludes
    in-flight ``.json.tmp`` files by construction (specs/35 §II.2.5).

    Args:
        tuning_dir: the shared ``artifacts/tuning`` directory all shards
            wrote into.

    Returns:
        Sorted list of shard file paths.

    Raises:
        PromoteError: if the directory does not exist or contains no shard
            files.
    """
    tuning_dir = Path(tuning_dir)
    pattern = SHARD_FILE_PREFIX + "*.json"
    if not tuning_dir.is_dir():
        raise PromoteError(
            f"tuning dir does not exist or is not a directory: {tuning_dir}"
        )
    paths = sorted(tuning_dir.glob(pattern))
    if not paths:
        raise PromoteError(
            f"no shard files matching {pattern!r} found in {tuning_dir} — "
            "nothing to promote. Have the shard jobs run?"
        )
    return paths


def load_shard_files(paths: list[Path]) -> list[dict]:
    """Parse every shard file; report all torn/unparseable files at once.

    Args:
        paths: shard file paths from :func:`gather_shard_files`.

    Returns:
        One dict per file: ``{"file": <basename>, "header": ..., "results":
        ...}``.

    Raises:
        PromoteError: listing EVERY offending file by name — a torn file is
            reported as such, never surfaced as a bare ``JSONDecodeError``
            (specs/34 §4.1 step 2).
    """
    shards: list[dict] = []
    torn: list[str] = []
    for path in paths:
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            torn.append(f"torn or unparseable shard file: {path.name}: {exc}")
            continue
        if not isinstance(data, dict):
            torn.append(
                f"torn or unparseable shard file: {path.name}: top-level "
                f"JSON value is {type(data).__name__}, expected object"
            )
            continue
        missing = [key for key in ("header", "results") if key not in data]
        if missing:
            torn.append(
                f"torn or unparseable shard file: {path.name}: missing "
                f"key(s) {missing}"
            )
            continue
        shards.append(
            {"file": path.name, "header": data["header"], "results": data["results"]}
        )
    if torn:
        raise PromoteError(
            "refusing to promote — {} unreadable shard file(s):\n  {}".format(
                len(torn), "\n  ".join(torn)
            )
        )
    return shards


def _canon(value: object) -> str:
    """Canonical JSON rendering of a header value, for equality grouping."""
    return json.dumps(value, sort_keys=True)


def validate_shards(shards: list[dict], num_shards: int | None = None) -> None:
    """Fail-closed completeness + consistency gate (specs/35 §V.4).

    All checks run; failures are accumulated and raised together in one
    ``PromoteError`` so the operator sees the full picture in one pass.
    It must be impossible to promote from a partial grid silently
    (specs/34 §4.1).

    Args:
        shards: loaded shard dicts from :func:`load_shard_files`.
        num_shards: optional operator-supplied cross-check; the headers
            remain the authority.

    Raises:
        PromoteError: naming every violated check.
    """
    failures: list[str] = []

    # 1. Header agreement across all files.
    for field in _AGREEMENT_FIELDS:
        by_value: dict[str, list[str]] = {}
        for shard in shards:
            by_value.setdefault(
                _canon(shard["header"].get(field)), []
            ).append(shard["file"])
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
    # agreement check above failed for these fields, that failure is already
    # recorded; the remaining checks still run so the report is complete.
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
    if isinstance(agreed_num_shards, int):
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
    if isinstance(agreed_n_total, int):
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


def promote(tuning_dir: Path, num_shards: int | None = None) -> dict:
    """Merge shard files, select the winner, write the merged artifacts.

    Idempotent: a pure function of the shard files; re-running overwrites
    ``tuning_results.json`` and ``best_params.json`` with identical bytes.

    Args:
        tuning_dir: the shared ``artifacts/tuning`` directory.
        num_shards: optional operator cross-check against the headers.

    Returns:
        The ``best_params.json`` payload that was written.

    Raises:
        PromoteError: on any gate failure (see :func:`validate_shards`,
            :func:`gather_shard_files`, :func:`load_shard_files`).
    """
    tuning_dir = Path(tuning_dir)
    paths = gather_shard_files(tuning_dir)
    logger.info("Found %d shard file(s) in %s", len(paths), tuning_dir)
    shards = load_shard_files(paths)
    validate_shards(shards, num_shards=num_shards)

    records = [r for s in shards for r in s["results"]]
    best = select_best(records)
    if best is None:
        raise PromoteError(
            f"shard files in {tuning_dir} contain no trial records"
        )

    results_path = tuning_dir / "tuning_results.json"
    if results_path.exists():
        logger.info(
            "Overwriting pre-existing %s with the merged shard results",
            results_path,
        )
    write_json_atomic(
        results_path, sorted(records, key=lambda r: r["trial"])
    )

    payload = {
        "best_params": best["params"],
        "best_val_macro_f1": best["best_val_macro_f1"],
        "selection_metric_used": best.get("selection_metric_used"),
        "best_trial": best["trial"],
    }
    write_json_atomic(tuning_dir / "best_params.json", payload)

    # Per-shard walltime summary (specs/34 §4.1 step 6 — the §3.2.5
    # dataset-invariance check, §7.10). Log lines only; no file output.
    shard_hours: list[float] = []
    for shard in shards:
        hours = sum(r.get("elapsed_s", 0.0) for r in shard["results"]) / 3600.0
        shard_hours.append(hours)
        logger.info(
            "Shard walltime: %s owned_values=%s trials=%d elapsed=%.2f h",
            shard["file"],
            shard["header"].get("owned_values"),
            len(shard["results"]),
            hours,
        )
    max_h, min_h = max(shard_hours), min(shard_hours)
    ratio = max_h / min_h if min_h > 0 else float("inf")
    logger.info(
        "Shard walltime balance: max=%.2f h, min=%.2f h, ratio=%.3f",
        max_h, min_h, ratio,
    )

    logger.info(
        "Promoted best trial %d: params=%s, %s=%.6f",
        best["trial"],
        best["params"],
        best.get("selection_metric_used"),
        best["best_val_macro_f1"],
    )
    logger.info("Wrote %s and %s", results_path, tuning_dir / "best_params.json")
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: argument list (defaults to ``sys.argv[1:]``).

    Returns:
        0 on success; 1 on any gate failure (every failure named — never a
        bare traceback for a foreseeable condition).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Merge sharded Phase-3 tuning results and emit "
            "tuning_results.json + best_params.json (specs/34, specs/35)."
        )
    )
    parser.add_argument(
        "--tuning-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="The shared runs/<run_id>/artifacts/tuning directory all "
             "shard jobs wrote into.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        metavar="N",
        help="Optional cross-check: must equal every shard header's "
             "num_shards. The headers remain the authority.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        promote(args.tuning_dir, num_shards=args.num_shards)
    except PromoteError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
