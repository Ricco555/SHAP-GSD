"""
Phase 4: Full training with best hyperparameters (50 epochs, patience 10).

Prerequisites:
  - Phase 3 complete: artifacts/tuning/best_params.json exists

Usage:
  python scripts/04_train.py --config configs/experiment_unsw.yaml
                             [--best-params artifacts/tuning/best_params.json]

Outputs:
  artifacts/best_model.pt         (best checkpoint by val macro-F1)
  artifacts/training_curves.json
  artifacts/best_params.json      (copy of training hyperparameters)
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
from src.model.node_state import NodeStateManager
from src.model.sage_model import EdgeAwareGraphSAGE
from src.model.trainer import Trainer
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(cfg: dict, best_params_path: Path | None = None) -> None:
    repo_root  = REPO_ROOT
    device = torch.device(
        cfg["compute"]["device"] if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Device: {device}")

    # ── 1. Load best params (override config model section) ───────────────────
    if best_params_path is None:
        best_params_path = repo_root / cfg["output"]["artifacts_dir"] / "tuning" / "best_params.json"

    if best_params_path.exists():
        with open(best_params_path) as f:
            bp = json.load(f)
        best = bp.get("best_params", bp)
        logger.info(f"Loading best params from {best_params_path}: {best}")
        cfg["model"].update(best)
    else:
        logger.warning(
            f"best_params.json not found at {best_params_path}. "
            "Using default model config. Run 03_tune.py first for optimal results."
        )

    # ── 2. Load graphs ─────────────────────────────────────────────────────────
    import dgl
    graph_dir = repo_root / cfg["graph"]["dir"]
    g_train_list, _ = dgl.load_graphs(str(graph_dir / "train.bin"))
    g_val_list,   _ = dgl.load_graphs(str(graph_dir / "val.bin"))
    g_train = g_train_list[0]
    g_val   = g_val_list[0]
    logger.info(
        f"Graphs: train={g_train.num_edges():,}, val={g_val.num_edges():,}"
    )

    # ── 3. Load feature stores ─────────────────────────────────────────────────
    fs_dir   = repo_root / cfg["output"]["feature_store_dir"]
    fs_train = FeatureStore(fs_dir / "train")
    fs_val   = FeatureStore(fs_dir / "val")

    # ── 4. Load NodeStateManager ───────────────────────────────────────────────
    nsm = NodeStateManager.load(repo_root / cfg["graph"]["node_state_dir"])

    # ── 5. Load balanced EIDs and class weights ────────────────────────────────
    balanced_eids = np.load(
        repo_root / cfg["output"]["balanced_train_indices_path"]
    )
    class_weights = torch.from_numpy(
        np.load(repo_root / cfg["output"]["class_weights_path"])
    ).float()
    train_labels = np.load(
        repo_root / cfg["output"]["feature_store_dir"] / "train" / "labels.npy"
    )
    train_label_counts = np.bincount(
        train_labels, minlength=cfg["model"]["num_classes"]
    ).astype(np.int64)

    # ── 6. Build model ─────────────────────────────────────────────────────────
    torch.manual_seed(cfg["reproducibility"]["model_seed"])
    m = cfg["model"]
    model = EdgeAwareGraphSAGE(
        node_in_dim=m["node_state_dim"],
        edge_in_dim=fs_train.d_e,
        hidden_size=m["hidden_size"],
        num_classes=m["num_classes"],
        num_layers=m["num_layers"],
        dropout=m["dropout"],
        aggregator=m["aggregator"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,}")

    # ── 7. Train ───────────────────────────────────────────────────────────────
    artifacts_dir = repo_root / cfg["output"]["artifacts_dir"]
    trainer = Trainer(
        model=model,
        g_train=g_train,
        g_val=g_val,
        fs_train=fs_train,
        fs_val=fs_val,
        nsm=nsm,
        cfg=cfg,
        device=device,
    )

    curves = trainer.train(
        balanced_train_eids=balanced_eids,
        class_weights=class_weights,
        output_dir=artifacts_dir,
        seed=cfg["reproducibility"]["model_seed"],
        train_label_counts=train_label_counts,
    )

    # ── 8. Save final config used ──────────────────────────────────────────────
    final_params = {k: cfg["model"][k] for k in (
        "num_layers", "hidden_size", "dropout", "aggregator",
        "fanouts", "batch_size", "learning_rate", "weight_decay",
        "max_epochs", "patience",
    ) if k in cfg["model"]}
    with open(artifacts_dir / "best_params.json", "w") as f:
        json.dump(final_params, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Training complete.")
    logger.info(f"  Best epoch:      {curves['best_epoch']}")
    logger.info(f"  Best val_macro_f1: {max(curves['val_macro_f1']):.4f}")
    logger.info(f"  Checkpoint:      {artifacts_dir / 'best_model.pt'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_unsw.yaml")
    parser.add_argument("--best-params", default=None,
                        help="Path to best_params.json from tuning (optional)")
    args = parser.parse_args()
    cfg  = load_config(REPO_ROOT / args.config)
    bp   = Path(args.best_params) if args.best_params else None
    main(cfg, bp)
