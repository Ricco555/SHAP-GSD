"""
Phase 13 — Model ablation study.

Four controlled ablations vs the locked run (artifacts/best_model.pt,
macro-F1=0.508, weighted-F1=0.961):

  (a) constant_node_state    — 15-dim node state replaced with zeros (TE-G-SAGE-style constant)
  (b) behavioral_only        — 11-dim behavioral only; seasonal dims 11-14 dropped
  (c) no_balancing           — raw unbalanced training (balanced_train_indices unused)
  (d) teg_sage_defaults      — TE-G-SAGE hyperparameters: fanouts=[25,15], hidden=128,
                               dropout=0.3, lr=3e-4

Each ablation trains from scratch and evaluates on the test split.

Prerequisites:
  - Phases 1-5 complete: graphs/, feature_store/, node_state_snapshots/ exist
  - artifacts/class_weights.npy, artifacts/balanced_train_indices.npy exist

Outputs (per variant):
  artifacts/ablations/<name>/best_model.pt
  artifacts/ablations/<name>/best_params.json
  artifacts/ablations/<name>/training_curves.json
  artifacts/ablations/<name>/evaluation/metrics.json
  artifacts/ablations/comparison_table.{txt,json}

Usage:
  python scripts/13_ablations.py --config configs/experiment_unsw.yaml
  python scripts/13_ablations.py --config configs/experiment_unsw.yaml --variants a b c d
"""

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.feature_store import FeatureStore
from src.model.evaluator import Evaluator
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


# ── NSM wrappers ──────────────────────────────────────────────────────────────

class _ZeroNSM:
    """Returns all-zero node states (ablation a: constant/Paper-1-style)."""

    def __init__(self, dim: int = 15) -> None:
        self._dim = dim

    def get_batch_states(self, node_ids: np.ndarray, t: float) -> np.ndarray:
        return np.zeros((len(node_ids), self._dim), dtype=np.float32)

    def get_state_at_time(self, node_id: int, t: float) -> np.ndarray:
        return np.zeros(self._dim, dtype=np.float32)


class _SlicedNSM:
    """Returns only the first n_dims of each node state (ablation b: behavioral-only)."""

    def __init__(self, nsm: NodeStateManager, n_dims: int) -> None:
        self._nsm    = nsm
        self._n_dims = n_dims

    def get_batch_states(self, node_ids: np.ndarray, t: float) -> np.ndarray:
        return self._nsm.get_batch_states(node_ids, t)[:, : self._n_dims]

    def get_state_at_time(self, node_id: int, t: float) -> np.ndarray:
        return self._nsm.get_state_at_time(node_id, t)[: self._n_dims]


# ── Ablation definitions ──────────────────────────────────────────────────────

# Each entry: (short_name, description, cfg_overrides_fn, nsm_fn, balanced_eids_fn)
# - cfg_overrides_fn(cfg)  → modifies cfg in-place
# - nsm_fn(nsm)            → returns the NSM (or wrapper) to use
# - balanced_eids_fn(base_eids, g_train) → returns EID array for training

ABLATIONS: list[tuple[str, str, callable, callable, callable]] = [
    (
        "a_constant_node_state",
        "15-dim node state replaced with zeros (TE-G-SAGE-style constant)",
        lambda cfg: None,                          # no cfg change
        lambda nsm: _ZeroNSM(dim=15),
        lambda base, g: base,                      # keep balanced oversampling
    ),
    (
        "b_behavioral_only",
        "11-dim behavioral node state only (seasonal dims 11-14 dropped)",
        lambda cfg: cfg["model"].update({"node_state_dim": 11}),
        lambda nsm: _SlicedNSM(nsm, 11),
        lambda base, g: base,
    ),
    (
        "c_no_balancing",
        "Raw unbalanced training (temporal oversampling disabled)",
        lambda cfg: None,
        lambda nsm: nsm,
        lambda base, g: np.arange(g.num_edges(), dtype=np.int64),
    ),
    (
        "d_teg_sage_defaults",
        "TE-G-SAGE hyperparameters: fanouts=[25,15], hidden=128, dropout=0.3, lr=3e-4",
        lambda cfg: cfg["model"].update({
            "fanouts":       [25, 15],
            "hidden_size":   128,
            "dropout":       0.3,
            "learning_rate": 3e-4,
        }),
        lambda nsm: nsm,
        lambda base, g: base,
    ),
]


# ── Core train + eval ─────────────────────────────────────────────────────────

def _run_ablation(
    name: str,
    description: str,
    cfg_fn: callable,
    nsm_fn: callable,
    eids_fn: callable,
    base_cfg: dict,
    base_nsm: NodeStateManager,
    g_train,
    g_val,
    g_test,
    fs_train: FeatureStore,
    fs_val: FeatureStore,
    fs_test: FeatureStore,
    balanced_eids: np.ndarray,
    class_weights: torch.Tensor,
    train_label_counts: np.ndarray,
    device: torch.device,
    label_map_path: Path,
) -> dict:
    logger.info("=" * 70)
    logger.info("ABLATION %s: %s", name, description)
    logger.info("=" * 70)

    cfg = copy.deepcopy(base_cfg)
    cfg_fn(cfg)

    nsm        = nsm_fn(base_nsm)
    train_eids = eids_fn(balanced_eids, g_train)

    node_dim = cfg["model"]["node_state_dim"]
    m        = cfg["model"]

    torch.manual_seed(cfg["reproducibility"]["model_seed"])
    model = EdgeAwareGraphSAGE(
        node_in_dim  = node_dim,
        edge_in_dim  = fs_train.d_e,
        hidden_size  = m["hidden_size"],
        num_classes  = m["num_classes"],
        num_layers   = m["num_layers"],
        dropout      = m["dropout"],
        aggregator   = m["aggregator"],
    ).to(device)

    abl_dir = ROOT / "artifacts" / "ablations" / name
    abl_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model    = model,
        g_train  = g_train,
        g_val    = g_val,
        fs_train = fs_train,
        fs_val   = fs_val,
        nsm      = nsm,
        cfg      = cfg,
        device   = device,
    )

    curves = trainer.train(
        balanced_train_eids = train_eids,
        class_weights       = class_weights,
        output_dir          = abl_dir,
        seed                = cfg["reproducibility"]["model_seed"],
        train_label_counts  = train_label_counts,
    )

    # Save params used
    params_used = {k: cfg["model"][k] for k in (
        "num_layers", "hidden_size", "dropout", "aggregator",
        "fanouts", "batch_size", "learning_rate", "weight_decay",
        "max_epochs", "patience", "node_state_dim",
    ) if k in cfg["model"]}
    (abl_dir / "best_params.json").write_text(json.dumps(params_used, indent=2))

    # ── Evaluate ───────────────────────────────────────────────────────────────
    ckpt_path = abl_dir / "best_model.pt"
    torch.manual_seed(cfg["reproducibility"]["model_seed"])
    np.random.seed(cfg["reproducibility"]["model_seed"])

    eval_model = EdgeAwareGraphSAGE(
        node_in_dim  = node_dim,
        edge_in_dim  = fs_test.d_e,
        hidden_size  = m["hidden_size"],
        num_classes  = m["num_classes"],
        num_layers   = m["num_layers"],
        dropout      = m["dropout"],
        aggregator   = m["aggregator"],
    ).to(device)
    eval_model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    eval_model.eval()

    evaluator = Evaluator(
        model    = eval_model,
        g_test   = g_test,
        fs_test  = fs_test,
        nsm      = nsm,
        cfg      = cfg,
        device   = device,
    )
    metrics = evaluator.evaluate(
        output_dir     = abl_dir / "evaluation",
        label_map_path = label_map_path,
    )

    logger.info(
        "ABLATION %s DONE — macro_f1=%.4f  weighted_f1=%.4f  best_epoch=%d",
        name, metrics["macro_f1"], metrics["weighted_f1"], curves["best_epoch"],
    )
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config",   default=str(ROOT / "configs" / "experiment_unsw.yaml"))
    parser.add_argument(
        "--variants", nargs="+", default=["a", "b", "c", "d"],
        choices=["a", "b", "c", "d"],
        help="Which ablations to run (default: all four)",
    )
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device(
        cfg["compute"]["device"] if torch.cuda.is_available() else "cpu"
    )
    logger.info("Device: %s", device)

    import dgl

    # ── Common artifacts (loaded once) ─────────────────────────────────────────
    graph_dir = ROOT / cfg["graph"]["dir"]
    g_train_list, _ = dgl.load_graphs(str(graph_dir / "train.bin"))
    g_val_list,   _ = dgl.load_graphs(str(graph_dir / "val.bin"))
    g_test_list,  _ = dgl.load_graphs(str(graph_dir / "test.bin"))
    g_train = g_train_list[0]
    g_val   = g_val_list[0]
    g_test  = g_test_list[0]
    logger.info(
        "Graphs: train=%d  val=%d  test=%d",
        g_train.num_edges(), g_val.num_edges(), g_test.num_edges(),
    )

    fs_dir   = ROOT / cfg["output"]["feature_store_dir"]
    fs_train = FeatureStore(fs_dir / "train")
    fs_val   = FeatureStore(fs_dir / "val")
    fs_test  = FeatureStore(fs_dir / "test")

    nsm = NodeStateManager.load(ROOT / cfg["graph"]["node_state_dir"])

    balanced_eids = np.load(ROOT / cfg["output"]["balanced_train_indices_path"])
    class_weights = torch.from_numpy(
        np.load(ROOT / cfg["output"]["class_weights_path"])
    ).float()
    train_labels = np.load(fs_dir / "train" / "labels.npy")
    train_label_counts = np.bincount(
        train_labels, minlength=cfg["model"]["num_classes"]
    ).astype(np.int64)

    label_map_path = ROOT / cfg["output"]["artifacts_dir"] / "label_map.json"

    # ── Load best params into base cfg (same as 04_train.py does) ─────────────
    bp_path = ROOT / cfg["output"]["artifacts_dir"] / "best_params.json"
    if bp_path.exists():
        bp = json.loads(bp_path.read_text())
        cfg["model"].update(bp.get("best_params", bp))
        logger.info("Loaded best params: %s", {
            k: cfg["model"][k] for k in ("hidden_size", "fanouts", "dropout", "learning_rate")
        })

    # ── Filter requested variants ─────────────────────────────────────────────
    letter_to_idx = {"a": 0, "b": 1, "c": 2, "d": 3}
    selected = [ABLATIONS[letter_to_idx[v]] for v in sorted(set(args.variants))]

    # ── Run ablations ─────────────────────────────────────────────────────────
    results: dict[str, dict] = {}
    for name, description, cfg_fn, nsm_fn, eids_fn in selected:
        metrics = _run_ablation(
            name               = name,
            description        = description,
            cfg_fn             = cfg_fn,
            nsm_fn             = nsm_fn,
            eids_fn            = eids_fn,
            base_cfg           = cfg,
            base_nsm           = nsm,
            g_train            = g_train,
            g_val              = g_val,
            g_test             = g_test,
            fs_train           = fs_train,
            fs_val             = fs_val,
            fs_test            = fs_test,
            balanced_eids      = balanced_eids,
            class_weights      = class_weights,
            train_label_counts = train_label_counts,
            device             = device,
            label_map_path     = label_map_path,
        )
        results[name] = metrics

    if not results:
        return

    # ── Comparison table ──────────────────────────────────────────────────────
    locked_metrics_path = ROOT / "artifacts" / "evaluation" / "metrics.json"
    locked = json.loads(locked_metrics_path.read_text()) if locked_metrics_path.exists() else {}

    abl_dir = ROOT / "artifacts" / "ablations"
    abl_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "Model Ablation Comparison — SHAP-GSD / NF-UNSW-NB15-v3",
        "=" * 72,
        f"  {'variant':<35}  {'macro_f1':>9}  {'weighted_f1':>12}  {'Δmacro_f1':>10}",
        "-" * 72,
    ]

    locked_macro = locked.get("macro_f1", float("nan"))
    locked_wt    = locked.get("weighted_f1", float("nan"))
    lines.append(
        f"  {'locked_run (best_model.pt)':<35}  {locked_macro:>9.4f}  {locked_wt:>12.4f}  {'—':>10}"
    )

    comparison_json: dict = {
        "locked_run": {
            "macro_f1":    locked_macro,
            "weighted_f1": locked_wt,
        }
    }

    for name, metrics in results.items():
        mf1 = metrics["macro_f1"]
        wf1 = metrics["weighted_f1"]
        delta = mf1 - locked_macro if locked else float("nan")
        sign  = "+" if delta >= 0 else ""
        lines.append(
            f"  {name:<35}  {mf1:>9.4f}  {wf1:>12.4f}  {sign}{delta:>9.4f}"
        )
        comparison_json[name] = {
            "macro_f1":        mf1,
            "weighted_f1":     wf1,
            "delta_macro_f1":  delta,
        }

    lines += ["", "Per-class F1 (minority classes only):"]
    minority = ["Backdoor", "DoS", "Worms", "Analysis", "Shellcode"]
    header = f"  {'variant':<35}" + "".join(f"  {c:>10}" for c in minority)
    lines.append(header)
    lines.append("-" * (35 + 14 * len(minority) + 2))

    if locked:
        row = f"  {'locked_run':<35}"
        for c in minority:
            f1 = locked.get("per_class", {}).get(c, {}).get("f1", float("nan"))
            row += f"  {f1:>10.4f}"
        lines.append(row)

    for name, metrics in results.items():
        row = f"  {name:<35}"
        for c in minority:
            f1 = metrics.get("per_class", {}).get(c, {}).get("f1", float("nan"))
            row += f"  {f1:>10.4f}"
        lines.append(row)

    report = "\n".join(lines) + "\n"
    (abl_dir / "comparison_table.txt").write_text(report)
    (abl_dir / "comparison_table.json").write_text(json.dumps(comparison_json, indent=2))

    logger.info("\n%s", report)
    logger.info("Comparison table → %s", abl_dir / "comparison_table.txt")


if __name__ == "__main__":
    main()
