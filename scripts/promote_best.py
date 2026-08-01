"""
Promote step for sharded Phase 3 (specs/34 §4.1, specs/35 Part V).

Merges the per-shard result files that N concurrent `scripts/03_tune.py
--shard-axes ... --shard K/N` jobs wrote into a shared
`runs/<run_id>/artifacts/tuning/` directory, runs a fail-closed
completeness + consistency gate over them, and — only if the gate passes —
emits the exact artifacts a single-job Phase 3 run produces:

  tuning_results.json   bare list of all trial records, sorted by global
                        trial index (same schema as a single-job run)
  best_params.json      see src/model/selection.py's select_best_tie_aware
                        docstring and specs/38 §6.1 for the full
                        best_params.json schema — 4 legacy keys (still
                        authoritative for scripts/04_train.py's consumption)
                        plus 7 additive selection-provenance keys

This script is a thin CLI wrapper: file-gathering/parsing live here, but
the completeness/consistency gate (`validate_shards`, `PromoteError`) is
implemented once in `src.model.selection` — "the single enumeration
authority" for shard logic (specs/35 §V.4) — and imported from there, so
any future stdlib-only consumer of shard files does not have to duplicate
it or import this script via importlib.

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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.selection import (
    PromoteError,
    SHARD_FILE_PREFIX,
    select_best_tie_aware,
    validate_shards,
    write_json_atomic,
)

logger = logging.getLogger(__name__)


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
    if not records:
        raise PromoteError(
            f"shard files in {tuning_dir} contain no trial records"
        )

    # validate_shards() already confirmed every shard header agrees on
    # tie_band_pp/tie_break_axes/search_space (S1.4's _AGREEMENT_FIELDS) --
    # this is a presence check for the residual "all shards uniformly
    # missing the field" gap that check cannot close (S1.4), so a v1-schema
    # shard file set fails with a clear PromoteError, not a bare KeyError.
    header0 = shards[0]["header"]
    missing_selection_fields = [
        f for f in ("tie_band_pp", "tie_break_axes", "search_space")
        if f not in header0
    ]
    if missing_selection_fields:
        raise PromoteError(
            f"shard header(s) in {tuning_dir} are missing required "
            f"selection field(s) {missing_selection_fields} -- these shard "
            "files were produced under SHARD_SCHEMA_VERSION < 2 (before "
            "the Phase-3 selection-noise fix). Re-run the shard jobs under "
            "the current code."
        )
    tie_band_pp    = header0["tie_band_pp"]
    tie_break_axes = header0["tie_break_axes"]
    search_space   = header0["search_space"]
    result = select_best_tie_aware(records, tie_band_pp, tie_break_axes, search_space)

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
        "best_params":           result["winner"]["params"],
        "best_val_macro_f1":     result["winner"]["best_val_macro_f1"],
        "selection_metric_used": result["winner"].get("selection_metric_used"),
        "best_trial":            result["winner"]["trial"],
        "argmax_trial":          result["argmax"]["trial"],
        "argmax_val_macro_f1":   result["argmax"]["best_val_macro_f1"],
        "tie_band_pp":           result["tie_band_pp"],
        "tie_break_axes":        result["tie_break_axes"],
        "tie_set_trials":        result["tie_set_trials"],
        "tie_set_size":          result["tie_set_size"],
        "selection_method":      "tie_band_axis_priority",
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
        "Promoted best trial %d (argmax trial %d, tie set size %d): "
        "params=%s, %s=%.6f",
        result["winner"]["trial"], result["argmax"]["trial"],
        result["tie_set_size"], result["winner"]["params"],
        result["winner"].get("selection_metric_used"),
        result["winner"]["best_val_macro_f1"],
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
