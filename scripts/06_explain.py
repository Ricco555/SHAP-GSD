"""
Phase 6 — SHAP-GSD explanations.

Runs explain_stratified(n_per_class=200) across the test split,
serializes ExplanationResult objects to:
  outputs/explanations/<class_name>/<global_eid>.json
  outputs/explanations/summary.csv          (one row per explained edge)

Usage:
  python scripts/06_explain.py --config configs/experiment_unsw.yaml
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import dgl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.feature_store import FeatureStore
from src.model.node_state import NodeStateManager
from src.model.sage_model import EdgeAwareGraphSAGE
from src.explainer.background import BackgroundDistributions
from src.explainer.shap_gsd import SHAPGSDExplainer, ExplanationResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("shap").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _load_model(cfg: dict, device: torch.device) -> EdgeAwareGraphSAGE:
    """Instantiate and load the best model checkpoint."""
    m = cfg["model"]
    artifacts_dir = Path(cfg["output"]["artifacts_dir"])

    best_params_path = artifacts_dir / "best_params.json"
    if best_params_path.exists():
        with open(best_params_path) as f:
            best_params = json.load(f)
        hidden_size = best_params.get("hidden_size", m["hidden_size"])
        num_layers  = best_params.get("num_layers",  m["num_layers"])
        dropout     = best_params.get("dropout",     m["dropout"])
        aggregator  = best_params.get("aggregator",  m["aggregator"])
    else:
        hidden_size = m["hidden_size"]
        num_layers  = m["num_layers"]
        dropout     = m["dropout"]
        aggregator  = m["aggregator"]

    fg_path = Path(cfg["output"]["feature_groups_path"])
    with open(fg_path) as f:
        fg = json.load(f)
    d_e = fg["d_e"]

    label_map_path = artifacts_dir / "label_map.json"
    with open(label_map_path) as f:
        label_map = json.load(f)
    num_classes = len(label_map)

    model = EdgeAwareGraphSAGE(
        node_in_dim=m["node_state_dim"],
        edge_in_dim=d_e,
        hidden_size=hidden_size,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        aggregator=aggregator,
    ).to(device)

    ckpt = artifacts_dir / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    logger.info(f"Model loaded from {ckpt}")
    return model


def _result_to_dict(result: ExplanationResult) -> dict:
    """Serialize ExplanationResult to a JSON-compatible dict."""
    return {
        "edge_id":            result.edge_id,
        "true_label":         result.true_label,
        "predicted_label":    result.predicted_label,
        "predicted_proba":    result.predicted_proba.tolist(),
        "feature_group_names": result.feature_group_names,
        "feature_group_shap": result.feature_group_shap.tolist(),
        "neighbor_edge_ids":  result.neighbor_edge_ids,
        "neighbor_timestamps": result.neighbor_timestamps,
        "neighbor_shap":      result.neighbor_shap.tolist(),
        "node_ids":           result.node_ids,
        "node_shap":          result.node_shap.tolist(),
        "src_novelty_shap":   result.src_novelty_shap,
        "dst_novelty_shap":   result.dst_novelty_shap,
        "subgraph_edge_ids":  result.subgraph_edge_ids,
        "subgraph_shap_weights": result.subgraph_shap_weights,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP-GSD Phase 6 — Explanations")
    parser.add_argument(
        "--config", required=True,
        help="Path to experiment YAML config (merged with default.yaml)"
    )
    parser.add_argument(
        "--n-per-class", type=int, default=200,
        help="Number of test edges to explain per class (default 200)"
    )
    parser.add_argument(
        "--recompute-background", action="store_true",
        help="Force recompute background distributions even if cached"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info("=== Phase 6: SHAP-GSD Explanations ===")

    artifacts_dir = Path(cfg["output"]["artifacts_dir"])
    fs_train_dir  = Path(cfg["output"]["feature_store_dir"]) / "train"
    fs_test_dir   = Path(cfg["output"]["feature_store_dir"]) / "test"
    graphs_dir    = Path(cfg["graph"]["dir"])
    nsm_dir       = Path(cfg["graph"]["node_state_dir"])
    output_dir    = Path("outputs") / "explanations"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        cfg.get("compute", {}).get("device", "cuda")
        if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Device: {device}")

    # --- Load graph, feature stores, NSM ---
    logger.info("Loading test graph …")
    g_test, _ = dgl.load_graphs(str(graphs_dir / "test.bin"))
    g_test = g_test[0]

    logger.info("Loading feature stores …")
    fs_test  = FeatureStore(fs_test_dir)
    fs_train = FeatureStore(fs_train_dir)

    logger.info("Loading NodeStateManager …")
    nsm = NodeStateManager.load(nsm_dir)

    # --- Background distributions ---
    bg_features_path = artifacts_dir / "background_features.npy"
    if bg_features_path.exists() and not args.recompute_background:
        logger.info("Loading cached background distributions …")
        background = BackgroundDistributions.load(artifacts_dir)
    else:
        logger.info("Computing background distributions from training split …")
        g_train, _ = dgl.load_graphs(str(graphs_dir / "train.bin"))
        g_train = g_train[0]

        with open(artifacts_dir / "label_map.json") as f:
            label_map = json.load(f)
        num_classes = len(label_map)

        background = BackgroundDistributions.compute(
            feature_store_train_dir=fs_train_dir,
            nsm=nsm,
            g_train=g_train,
            num_classes=num_classes,
            node_state_dim=cfg["model"]["node_state_dim"],
            rng_seed=cfg.get("reproducibility", {}).get("background_seed", 456),
        )
        background.save(artifacts_dir)
        del g_train  # free memory

    # --- Load model and feature groups ---
    logger.info("Loading model …")
    model = _load_model(cfg, device)

    fg_path = Path(cfg["output"]["feature_groups_path"])
    with open(fg_path) as f:
        feature_groups = json.load(f)

    # --- Build explainer ---
    explainer = SHAPGSDExplainer(
        model=model,
        g_split=g_test,
        fs=fs_test,
        nsm=nsm,
        background=background,
        feature_groups=feature_groups,
        cfg=cfg,
        device=device,
    )

    # --- Run stratified explanations ---
    seed = cfg.get("reproducibility", {}).get("coalition_seed", 123)
    logger.info(f"Running explain_stratified(n_per_class={args.n_per_class}) …")
    stratified = explainer.explain_stratified(
        n_per_class=args.n_per_class,
        rng_seed=seed,
    )

    # --- Serialize results ---
    with open(artifacts_dir / "label_map.json") as f:
        label_map = json.load(f)
    int_to_name = {v: k for k, v in label_map.items()}

    summary_rows: list[dict] = []
    total = 0

    for c, results in stratified.items():
        class_name = int_to_name.get(c, str(c))
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for result in results:
            out_path = class_dir / f"{result.edge_id}.json"
            with open(out_path, "w") as f:
                json.dump(_result_to_dict(result), f)

            # Summary row: one line per edge with top feature group and top neighbor
            top_feat_idx = int(np.abs(result.feature_group_shap).argmax()) \
                if len(result.feature_group_shap) > 0 else -1
            top_feat = (
                result.feature_group_names[top_feat_idx]
                if top_feat_idx >= 0 else ""
            )
            top_nbr_phi = (
                float(result.neighbor_shap[0])
                if len(result.neighbor_shap) > 0 else 0.0
            )
            summary_rows.append({
                "edge_id":           result.edge_id,
                "class_name":        class_name,
                "true_label":        result.true_label,
                "predicted_label":   result.predicted_label,
                "correct":           result.true_label == result.predicted_label,
                "top_feature_group": top_feat,
                "top_feature_shap":  (
                    float(result.feature_group_shap[top_feat_idx])
                    if top_feat_idx >= 0 else 0.0
                ),
                "sum_feature_shap":  float(result.feature_group_shap.sum()),
                "n_neighbors":       len(result.neighbor_edge_ids),
                "top_neighbor_shap": top_nbr_phi,
                "src_novelty_shap":  result.src_novelty_shap,
                "dst_novelty_shap":  result.dst_novelty_shap,
                "subgraph_size":     len(result.subgraph_edge_ids),
            })
            total += 1

    summary_path = output_dir / "summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    logger.info(
        f"Phase 6 complete: {total} edges explained, "
        f"results → {output_dir}, summary → {summary_path}"
    )


if __name__ == "__main__":
    main()
