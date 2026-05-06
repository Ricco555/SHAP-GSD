"""
Phase 5: Evaluate trained model on the test split.

Produces per-class F1/precision/recall, confusion matrix, and ROC curves.
Paper 1 minority-class baselines (Backdoor F1=0.071, DoS F1=0.26) are
printed for direct comparison — if either is not improved, revisit the
oversampling ratio or class weights before treating results as final.

Prerequisites:
  - Phase 4 complete: artifacts/best_model.pt exists

Usage:
  python scripts/05_evaluate.py --config configs/experiment_unsw.yaml
                                [--checkpoint artifacts/best_model.pt]
                                [--label-map  artifacts/label_map.json]

Outputs (in artifacts/evaluation/):
  metrics.json
  confusion_matrix.png
  roc_curves.png
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.feature_store import FeatureStore
from src.model.evaluator import DEFAULT_CLASS_NAMES, Evaluator
from src.model.node_state import NodeStateManager
from src.model.sage_model import EdgeAwareGraphSAGE
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(
    cfg: dict,
    checkpoint_path: Path | None = None,
    label_map_path: Path | None = None,
) -> None:
    repo_root = REPO_ROOT
    device    = torch.device(
        cfg["compute"]["device"] if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Device: {device}")

    # ── 1. Resolve checkpoint ──────────────────────────────────────────────────
    if checkpoint_path is None:
        checkpoint_path = repo_root / cfg["output"]["artifacts_dir"] / "best_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}\n"
            "Run scripts/04_train.py first."
        )

    # ── 2. Load best params to build model with correct architecture ───────────
    best_params_path = checkpoint_path.parent / "best_params.json"
    if best_params_path.exists():
        with open(best_params_path) as f:
            bp = json.load(f)
        best = bp.get("best_params", bp)
        cfg["model"].update(best)
        logger.info(f"Loaded best params: {best}")

    # Fix sampling seed so eval numbers are reproducible across runs.
    # TemporalNeighborSampler is stochastic (random fan-out when candidates > k).
    import torch, numpy as np
    _seed = cfg["reproducibility"]["model_seed"]
    torch.manual_seed(_seed)
    np.random.seed(_seed)

    # ── 3. Load test graph ─────────────────────────────────────────────────────
    import dgl
    graph_dir = repo_root / cfg["graph"]["dir"]
    g_test_list, _ = dgl.load_graphs(str(graph_dir / "test.bin"))
    g_test = g_test_list[0]
    logger.info(f"Test graph: {g_test.num_edges():,} edges, {g_test.num_nodes():,} nodes")

    # ── 4. Load feature store ──────────────────────────────────────────────────
    fs_test = FeatureStore(repo_root / cfg["output"]["feature_store_dir"] / "test")

    # ── 5. Load NodeStateManager ───────────────────────────────────────────────
    nsm = NodeStateManager.load(repo_root / cfg["graph"]["node_state_dir"])

    # ── 6. Build model and load checkpoint ────────────────────────────────────
    m = cfg["model"]
    model = EdgeAwareGraphSAGE(
        node_in_dim=m["node_state_dim"],
        edge_in_dim=fs_test.d_e,
        hidden_size=m["hidden_size"],
        num_classes=m["num_classes"],
        num_layers=m["num_layers"],
        dropout=m["dropout"],
        aggregator=m["aggregator"],
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    # ── 7. Evaluate ────────────────────────────────────────────────────────────
    output_dir = repo_root / cfg["output"]["artifacts_dir"] / "evaluation"
    evaluator  = Evaluator(
        model=model, g_test=g_test, fs_test=fs_test,
        nsm=nsm, cfg=cfg, device=device,
    )

    metrics = evaluator.evaluate(
        output_dir=output_dir,
        label_map_path=label_map_path,
    )

    # ── 8. Warn if Paper 1 targets not met ────────────────────────────────────
    comparison = metrics.get("paper1_comparison", {})
    not_improved = [k for k, v in comparison.items() if not v["improved"]]
    if not_improved:
        logger.warning(
            f"Paper 1 F1 target NOT met for: {not_improved}. "
            "Consider adjusting min_class_ratio or max_majority_ratio in balancer "
            "config before treating full training as final."
        )

    logger.info(
        f"Evaluation artefacts written to {output_dir}/\n"
        f"  metrics.json, confusion_matrix.png, roc_curves.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/experiment_unsw.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model checkpoint (default: artifacts/best_model.pt)")
    parser.add_argument("--label-map",  default=None,
                        help="Path to label_map.json  (default: use built-in class names)")
    args = parser.parse_args()

    cfg  = load_config(REPO_ROOT / args.config)
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    lmap = Path(args.label_map)  if args.label_map  else None
    main(cfg, ckpt, lmap)
