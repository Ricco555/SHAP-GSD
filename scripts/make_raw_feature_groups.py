"""
Generate the raw-feature (ungrouped) coalition space for paper config (i).

Reads the run-scoped semantic ``feature_groups.json`` (K=48) and writes a
singleton grouping (K == d_e, one player per encoded edge-feature dimension)
to the path the config-(i) experiment YAML points at via
``output.feature_groups_path``.

This is a pure artifact generator: no model, no GPU, no data. It exists so the
config-(i) coalition space is derived deterministically from the run's own
authoritative grouping rather than hand-written, which is what guarantees
``d_e`` matches that run's feature store.

Usage:
  python scripts/make_raw_feature_groups.py \
      --config configs/experiment_nf_unsw_nb15_v3_r3_s2_config_i_full.yaml

  # explicit paths (source defaults to <artifacts_dir>/feature_groups.json)
  python scripts/make_raw_feature_groups.py --config <cfg> --source <path>
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.feature_groups import build_singleton_feature_groups

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Write the singleton feature_groups.json for the config-(i) run."""
    parser = argparse.ArgumentParser(
        description="Build the raw-feature (singleton) coalition space, config (i)"
    )
    parser.add_argument(
        "--config", required=True,
        help="Config-(i) experiment YAML; output.feature_groups_path is the target"
    )
    parser.add_argument(
        "--source", default=None,
        help="Semantic feature_groups.json to derive from "
             "(default: <output.artifacts_dir>/feature_groups.json)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite the target file if it already exists"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = Path(cfg["output"]["artifacts_dir"])
    out_path = Path(cfg["output"]["feature_groups_path"])
    src_path = Path(args.source) if args.source else artifacts_dir / "feature_groups.json"

    if out_path.resolve() == src_path.resolve():
        raise SystemExit(
            f"refusing to overwrite the source grouping in place: {src_path}. "
            f"The config-(i) YAML's output.feature_groups_path must name a "
            f"DIFFERENT file from the run's semantic feature_groups.json."
        )
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} already exists — pass --force to overwrite")

    with open(src_path) as f:
        source = json.load(f)
    logger.info(f"Source grouping: {src_path} (d_e={source['d_e']}, K={source['K']})")

    singleton = build_singleton_feature_groups(source)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(singleton, f, indent=2)
    logger.info(
        f"Singleton grouping written → {out_path} "
        f"(d_e={singleton['d_e']}, K={singleton['K']})"
    )


if __name__ == "__main__":
    main()
