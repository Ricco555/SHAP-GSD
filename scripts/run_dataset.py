"""Orchestrator: run the full SHAP-GSD pipeline against one or more NetFlow CSVs.

Usage
-----
    python scripts/run_dataset.py --csv data/dataset01.csv [data/dataset02.csv ...]
                                  [--from-phase 01] [--to-phase 14]
                                  [--skip-tests] [--skip-gateway-distance]
                                  [--config-template configs/experiment_unsw.yaml]
                                  [--force-config]

For each CSV the orchestrator
  1. Derives a ``run_id`` from the basename.
  2. Creates ``runs/<run_id>/``.
  3. Materialises ``configs/experiment_<run_id>.yaml`` (two-key overlay:
     ``data.csv_path`` and ``run.dir``).
  4. Invokes pipeline scripts 01–14 (or the requested sub-range) in order,
     streaming output directly to the console.
  5. Optionally runs the pytest gate after phase 02.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import yaml

# ---------------------------------------------------------------------------
# Repository root — one level up from this script.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Ordered pipeline: (number, script-stem, human label)
# ---------------------------------------------------------------------------
PIPELINE: list[tuple[int, str, str]] = [
    (1,  "01_preprocess.py",    "data pre-processing"),
    (2,  "02_build_graph.py",   "graph construction"),
    (3,  "03_tune.py",          "hyperparameter tuning"),
    (4,  "04_train.py",         "full training"),
    (5,  "05_evaluate.py",      "evaluation"),
    (6,  "06_explain.py",       "SHAP-GSD explanation"),
    (7,  "07_visualize.py",     "visualization"),
    (8,  "08_metrics.py",       "metrics"),
    (9,  "09_w_ablation.py",    "W-ablation"),
    (10, "10_baselines.py",     "baselines"),
    (11, "11_efficiency.py",    "efficiency audit"),
    (12, "12_novelty_audit.py", "node novelty audit"),
    (13, "13_ablations.py",     "model ablation study"),
    (14, "14_gateway_distance.py", "gateway distance measurement"),
]

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging() -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger("run_dataset")


log = _setup_logging()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RULE = "━" * 60  # thick horizontal rule


def _banner(msg: str) -> None:
    """Emit a clearly visible section header via the logger."""
    log.info(_RULE)
    log.info(msg)
    log.info(_RULE)


def derive_run_id(csv_path: Path) -> str:
    """Derive a clean identifier from a CSV basename.

    Steps:
      1. Strip the ``.csv`` suffix (case-insensitive).
      2. Lowercase.
      3. Replace every run of non-alphanumeric characters with a single ``_``.
      4. Strip leading/trailing underscores.

    Examples::

        NF-UNSW-NB15-v3.csv  →  nf_unsw_nb15_v3
        dataset 01.csv        →  dataset_01
    """
    stem = csv_path.stem  # removes final suffix only (.csv)
    run_id = stem.lower()
    run_id = re.sub(r"[^a-z0-9]+", "_", run_id)
    run_id = run_id.strip("_")
    return run_id


def materialize_config(
    csv_path: Path,
    run_dir: Path,
    run_id: str,
    force: bool,
) -> Path:
    """Write ``configs/experiment_<run_id>.yaml`` unless it already exists.

    Parameters
    ----------
    csv_path:
        Path to the input CSV (stored as-is in the YAML so it resolves from
        the repo root, matching how existing configs work).
    run_dir:
        Destination run directory (e.g. ``runs/nf_unsw_nb15_v3``).
    run_id:
        Sanitised dataset identifier.
    force:
        When *True*, overwrite an existing config file.

    Returns
    -------
    Path
        Absolute path to the written (or pre-existing) config file.
    """
    configs_dir = REPO_ROOT / "configs"
    cfg_path = configs_dir / f"experiment_{run_id}.yaml"

    # Make csv_path relative to repo root when possible (mirrors existing configs).
    # NOTE: Do NOT call .resolve() on csv_path — the repo's data/ directory is a
    # symlink and resolving it would produce a path outside REPO_ROOT, which then
    # cannot be made relative.  csv_path is already absolute at this point (the
    # caller constructs it as REPO_ROOT / user_arg before existence-checking).
    try:
        csv_rel = csv_path.relative_to(REPO_ROOT)
    except ValueError:
        # csv_path is outside the repo (e.g. absolute path to a network share).
        csv_rel = csv_path

    content = (
        "# Auto-generated by scripts/run_dataset.py"
        " — edit only if you know what you're doing.\n"
        f"data:\n"
        f'  csv_path: "{csv_rel}"\n'
        f"run:\n"
        f'  dir: "{run_dir}"\n'
    )

    if cfg_path.exists() and not force:
        log.warning(
            "Config already exists — skipping write to preserve user edits: %s"
            "  (pass --force-config to overwrite)",
            cfg_path,
        )
    else:
        if cfg_path.exists() and force:
            log.info("--force-config set; overwriting existing config: %s", cfg_path)
        cfg_path.write_text(content)
        log.info("Wrote config: %s", cfg_path)

    return cfg_path


def run_phase(
    script_name: str,
    label: str,
    phase_num: int,
    cfg_path: Path,
    env: dict[str, str] | None = None,
) -> bool:
    """Invoke a single pipeline script, streaming its output to the console.

    Parameters
    ----------
    script_name:
        Filename within ``scripts/`` (e.g. ``"01_preprocess.py"``).
    label:
        Human-readable description for log messages.
    phase_num:
        Integer phase number for the banner.
    cfg_path:
        Absolute path to the per-dataset config YAML.
    env:
        Optional extra environment variables merged over the current env.

    Returns
    -------
    bool
        *True* if the script exited cleanly (returncode 0), *False* otherwise.
    """
    _banner(f"Phase {phase_num:02d}: {label}")
    script = REPO_ROOT / "scripts" / script_name
    cmd = [sys.executable, str(script), "--config", str(cfg_path)]

    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)

    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=proc_env)
    if result.returncode != 0:
        log.error(
            "Phase %02d (%s) FAILED with exit code %d.",
            phase_num,
            label,
            result.returncode,
        )
        return False

    log.info("Phase %02d (%s) completed successfully.", phase_num, label)
    return True


def run_pytest_gate(cfg_path: Path) -> bool:
    """Run the pytest test suite (gate after phase 02).

    The ``SHAP_GSD_CONFIG`` environment variable is set so tests can locate the
    correct per-dataset config (Phase C convention).

    Returns
    -------
    bool
        *True* if all tests passed, *False* otherwise.
    """
    _banner("pytest gate (post-phase-02)")
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "-x"]
    proc_env = os.environ.copy()
    proc_env["SHAP_GSD_CONFIG"] = str(cfg_path)

    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=proc_env)
    if result.returncode != 0:
        log.error("pytest gate FAILED — aborting this dataset's run.")
        return False

    log.info("pytest gate passed.")
    return True


# ---------------------------------------------------------------------------
# Per-dataset orchestration
# ---------------------------------------------------------------------------

def run_dataset(
    csv_path: Path,
    dataset_index: int,
    total_datasets: int,
    from_phase: int,
    to_phase: int,
    skip_tests: bool,
    skip_gateway_distance: bool,
    force_config: bool,
) -> bool:
    """Run the pipeline for a single dataset CSV.

    Parameters
    ----------
    csv_path:
        Validated (existing) path to the dataset CSV.
    dataset_index:
        1-based index for display purposes.
    total_datasets:
        Total number of datasets being processed.
    from_phase, to_phase:
        Inclusive phase range to execute.
    skip_tests:
        When *True*, skip the pytest gate after phase 02.
    skip_gateway_distance:
        When *True*, force-skip phase 14 for this dataset regardless of config.
    force_config:
        When *True*, overwrite an existing per-dataset config.

    Returns
    -------
    bool
        *True* if this dataset completed without any phase failure.
    """
    run_id = derive_run_id(csv_path)
    run_dir = Path("runs") / run_id  # relative to repo root (matches run.dir semantics)

    _banner(
        f"Dataset {dataset_index}/{total_datasets}:"
        f" {csv_path}  →  {run_dir}"
    )

    # Create the run directory.
    abs_run_dir = REPO_ROOT / run_dir
    abs_run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Run directory: %s", abs_run_dir)

    # Materialise the per-dataset config.
    cfg_path = materialize_config(
        csv_path=csv_path,
        run_dir=run_dir,
        run_id=run_id,
        force=force_config,
    )

    # Extra env for sub-processes (pytest gate and scripts that honour it).
    sub_env = {"SHAP_GSD_CONFIG": str(cfg_path)}

    # Execute phases in order.
    phases_to_run = [
        (num, script, label)
        for num, script, label in PIPELINE
        if from_phase <= num <= to_phase
        and not (num == 14 and skip_gateway_distance)
    ]

    if not phases_to_run:
        log.info(
            "No phases in range [%02d, %02d] — nothing to run.", from_phase, to_phase
        )
        return True

    for phase_num, script_name, label in phases_to_run:
        ok = run_phase(
            script_name=script_name,
            label=label,
            phase_num=phase_num,
            cfg_path=cfg_path,
            env=sub_env,
        )
        if not ok:
            return False

        # Pytest gate fires once, immediately after phase 02 completes.
        if phase_num == 2 and not skip_tests:
            if not run_pytest_gate(cfg_path):
                return False

    _banner(f"Dataset {run_id}: all requested phases completed successfully.")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_dataset.py",
        description=(
            "Run the SHAP-GSD pipeline (phases 01–14) against one or more "
            "NetFlow CSV files. Each CSV gets its own isolated run directory "
            "and experiment config so results never collide."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples\n"
            "--------\n"
            "  # Full pipeline for one dataset:\n"
            "  python scripts/run_dataset.py --csv data/NF-UNSW-NB15-v3.csv\n\n"
            "  # Only pre-process and build the graph (phases 01–02):\n"
            "  python scripts/run_dataset.py --csv data/dataset01.csv "
            "--from-phase 1 --to-phase 2\n\n"
            "  # Multiple datasets, skip tests, force config overwrite:\n"
            "  python scripts/run_dataset.py --csv data/a.csv data/b.csv "
            "--skip-tests --force-config\n"
        ),
    )

    parser.add_argument(
        "--csv",
        metavar="CSV",
        nargs="+",
        required=True,
        help=(
            "One or more paths to NetFlow CSV files. "
            "Each file is processed independently."
        ),
    )
    parser.add_argument(
        "--from-phase",
        metavar="N",
        type=int,
        default=1,
        help="First phase to execute (1–14, inclusive). Default: 1.",
    )
    parser.add_argument(
        "--to-phase",
        metavar="N",
        type=int,
        default=14,
        help="Last phase to execute (1–14, inclusive). Default: 14.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        default=False,
        help=(
            "Skip the pytest gate that normally runs after phase 02 completes. "
            "Not recommended for production runs."
        ),
    )
    parser.add_argument(
        "--skip-gateway-distance",
        action="store_true",
        default=False,
        help=(
            "Skip phase 14 (gateway-distance measurement) for this run, "
            "regardless of the topology.gateway_distance.enabled config value."
        ),
    )
    parser.add_argument(
        "--config-template",
        metavar="YAML",
        default="configs/experiment_unsw.yaml",
        help=(
            "Path to an existing experiment config used as a reference template. "
            "Currently informational only — the generated per-dataset config is a "
            "minimal two-key overlay over configs/default.yaml. "
            "Default: configs/experiment_unsw.yaml."
        ),
    )
    parser.add_argument(
        "--force-config",
        action="store_true",
        default=False,
        help=(
            "Overwrite an existing per-dataset experiment config. "
            "By default, existing configs are preserved (with a warning) "
            "so that manual edits survive re-runs."
        ),
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    # Validate phase range — allow any positive integer; values beyond the
    # pipeline range (1–14) simply produce an empty phase selection (no-op).
    if args.from_phase < 1:
        log.error("--from-phase must be >= 1 (got %d).", args.from_phase)
        return 1
    if args.to_phase < 1:
        log.error("--to-phase must be >= 1 (got %d).", args.to_phase)
        return 1
    if args.from_phase > args.to_phase:
        log.warning(
            "--from-phase (%d) > --to-phase (%d): no phases will run.",
            args.from_phase,
            args.to_phase,
        )

    # Resolve and validate every CSV path up front (fail fast).
    csv_paths: list[Path] = []
    for raw in args.csv:
        p = Path(raw)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if not p.exists():
            log.error("CSV not found: %s", p)
            return 1
        csv_paths.append(p)

    total = len(csv_paths)
    any_failed = False

    for idx, csv_path in enumerate(csv_paths, start=1):
        ok = run_dataset(
            csv_path=csv_path,
            dataset_index=idx,
            total_datasets=total,
            from_phase=args.from_phase,
            to_phase=args.to_phase,
            skip_tests=args.skip_tests,
            skip_gateway_distance=args.skip_gateway_distance,
            force_config=args.force_config,
        )
        if not ok:
            log.error("Dataset %d/%d FAILED: %s", idx, total, csv_path)
            any_failed = True
            # Continue to the next dataset rather than aborting the whole run.

    if any_failed:
        log.error("One or more datasets failed. See messages above.")
        return 1

    log.info("All datasets completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
