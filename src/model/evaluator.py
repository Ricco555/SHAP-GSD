"""
Test-set evaluation for EdgeAwareGraphSAGE.

Produces:
  metrics.json          per-class precision/recall/F1, overall accuracy/macro-F1
  confusion_matrix.png  row-normalised heatmap
  roc_curves.png        one-vs-rest ROC per class

TE-G-SAGE minority-class baselines to beat:
  Backdoor F1 = 0.071
  DoS      F1 = 0.26
These are printed alongside the new results for direct comparison.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

import dgl

from src.data.feature_store import FeatureStore
from src.model.node_state import NodeStateManager
from src.model.sage_model import EdgeAwareGraphSAGE, build_src_dst_pos
from src.model.temporal_sampler import TemporalNeighborSampler
from src.visualization.metrics_plots import plot_confusion_matrix, plot_roc_curves

logger = logging.getLogger(__name__)

# TE-G-SAGE per-class F1 baselines for NF-UNSW-NB15-v3.
# These are UNSW-specific and are only printed when the class names match
# this dataset (see Evaluator._compare_to_baselines).
TEG_SAGE_F1_BASELINES: dict[str, float] = {
    "Backdoor": 0.071,
    "DoS":      0.26,
}

# Class names that identify NF-UNSW-NB15-v3 — used to gate baseline comparison.
_UNSW_CLASS_NAMES: frozenset[str] = frozenset(TEG_SAGE_F1_BASELINES.keys()) | {
    "Benign", "Generic", "Exploits", "Fuzzers", "Recon",
    "Analysis", "Shellcode", "Worms",
}

# Default class names for NF-UNSW-NB15-v3 (Label 0-9).
# This is a FALLBACK for when no label_map.json is provided.
# All supported datasets should supply a label_map.json from 01_preprocess.py.
DEFAULT_CLASS_NAMES: list[str] = [
    "Benign",       # 0
    "Generic",      # 1
    "Exploits",     # 2
    "Fuzzers",      # 3
    "DoS",          # 4
    "Recon",        # 5
    "Analysis",     # 6
    "Backdoor",     # 7
    "Shellcode",    # 8
    "Worms",        # 9
]


def resolve_class_names(
    class_names: list[str] | None,
    label_map_path: Path | str | None,
) -> list[str]:
    """Resolve the ordered class-name list used to label per-class metrics.

    Precedence: an existing ``label_map.json`` (the dataset's own dynamically
    derived class map) > an explicit ``class_names`` list > DEFAULT_CLASS_NAMES.
    The final fallback is UNSW-NB15-specific and only correct for that label
    ordering — callers should always supply a label map.

    Args:
        class_names: explicit ordered class names (index = label int), or None.
        label_map_path: path to ``label_map.json`` ({"ClassName": int}), or None.

    Returns:
        Ordered class names where ``names[i]`` is the name of integer label ``i``.
    """
    if label_map_path and Path(label_map_path).exists():
        with open(label_map_path) as f:
            raw: dict[str, int] = json.load(f)
        names = [""] * (max(raw.values()) + 1)
        for name, idx in raw.items():
            names[idx] = name
        return names
    if class_names is not None:
        return class_names
    logger.warning(
        "No label_map.json available — falling back to DEFAULT_CLASS_NAMES "
        "(NF-UNSW-NB15-v3 ordering). Per-class names may not match this dataset."
    )
    return DEFAULT_CLASS_NAMES


class Evaluator:
    """Evaluate a trained EdgeAwareGraphSAGE on the test split."""

    def __init__(
        self,
        model: EdgeAwareGraphSAGE,
        g_test: dgl.DGLGraph,
        fs_test: FeatureStore,
        nsm: NodeStateManager,
        cfg: dict,
        device: torch.device,
    ) -> None:
        self.model   = model
        self.g_test  = g_test
        self.fs_test = fs_test
        self.nsm     = nsm
        self.cfg     = cfg
        self.device  = device

        self.sampler = TemporalNeighborSampler(
            fanouts=cfg["model"]["fanouts"]
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def evaluate(
        self,
        output_dir: Path | str,
        class_names: list[str] | None = None,
        label_map_path: Path | str | None = None,
    ) -> dict:
        """Run inference on the test split and produce all evaluation artefacts.

        Args:
            output_dir:      directory for metrics.json and PNG outputs.
            class_names:     ordered class name strings (index = label int).
                             Used only when no readable ``label_map_path`` is
                             given; falls back to DEFAULT_CLASS_NAMES if both
                             are absent (see ``resolve_class_names``).
            label_map_path:  path to label_map.json {"ClassName": int_label,
                             ...}; takes precedence over ``class_names``.

        Returns:
            metrics dict (also written to metrics.json).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        class_names = resolve_class_names(class_names, label_map_path)

        logger.info("Running inference on test split ...")
        y_true, y_pred, y_prob = self._run_inference()

        # ── Metrics ──────────────────────────────────────────────────────────
        accuracy   = float(accuracy_score(y_true, y_pred))
        macro_f1   = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        per_class = {
            name: {
                "precision": report[name]["precision"],
                "recall":    report[name]["recall"],
                "f1":        report[name]["f1-score"],
                "support":   int(report[name]["support"]),
            }
            for name in class_names
            if name in report
        }

        # ── TE-G-SAGE comparison ──────────────────────────────────────────────
        teg_sage_comparison = self._teg_sage_comparison(per_class)

        metrics = {
            "accuracy":      accuracy,
            "macro_f1":      macro_f1,
            "weighted_f1":   weighted_f1,
            "n_test_edges":  len(y_true),
            "per_class":     per_class,
            "teg_sage_comparison": teg_sage_comparison,
        }

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # ── Plots ─────────────────────────────────────────────────────────────
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        plot_confusion_matrix(
            cm, class_names,
            output_path=output_dir / "confusion_matrix.png",
        )
        plot_roc_curves(
            y_true, y_prob, class_names,
            output_path=output_dir / "roc_curves.png",
        )

        # ── Console summary ───────────────────────────────────────────────────
        self._print_summary(metrics, teg_sage_comparison, class_names)

        return metrics

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_inference(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (y_true, y_pred, y_prob) over all test edges."""
        self.model.eval()
        all_true:  list[int]  = []
        all_pred:  list[int]  = []
        all_prob:  list[list] = []

        batch_size = self.cfg["model"]["batch_size"]
        local_test_eids = np.arange(self.g_test.num_edges(), dtype=np.int64)

        loader = DataLoader(
            TensorDataset(torch.from_numpy(local_test_eids)),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        with torch.no_grad():
            for (batch_local_t,) in loader:
                batch_local = batch_local_t.long()

                input_nodes, seed_local, blocks = self.sampler.sample_blocks(
                    self.g_test, batch_local
                )
                blocks = [b.to(self.device) for b in blocks]

                batch_ts = float(
                    self.g_test.edata["timestamp"][batch_local].max().item()
                )
                node_feats = torch.from_numpy(
                    self.nsm.get_batch_states(input_nodes.numpy(), batch_ts)
                ).float().to(self.device)

                global_eids = self.g_test.edata[dgl.EID][seed_local].numpy()
                edge_feats = torch.from_numpy(
                    self.fs_test.get_batch(global_eids)
                ).float().to(self.device)

                seed_nodes = blocks[-1].dstdata[dgl.NID]
                src_pos, dst_pos = build_src_dst_pos(
                    self.g_test, seed_local, seed_nodes
                )
                src_pos = src_pos.to(self.device)
                dst_pos = dst_pos.to(self.device)

                logits = self.model(blocks, node_feats, edge_feats, src_pos, dst_pos)
                probs  = F.softmax(logits, dim=1).cpu().numpy()
                preds  = logits.argmax(dim=1).cpu().numpy()
                labels = self.fs_test.get_labels_batch(global_eids)

                all_true.extend(labels.tolist())
                all_pred.extend(preds.tolist())
                all_prob.extend(probs.tolist())

        return (
            np.array(all_true,  dtype=np.int64),
            np.array(all_pred,  dtype=np.int64),
            np.array(all_prob,  dtype=np.float32),
        )

    @staticmethod
    def _teg_sage_comparison(
        per_class: dict[str, dict],
    ) -> dict[str, dict]:
        """Build TE-G-SAGE vs SHAP-GSD delta table for NF-UNSW-NB15-v3 minority classes.

        Returns an empty dict when the class names do not match the UNSW dataset
        (i.e. when running on a different NetFlow dataset), so that no misleading
        comparison is printed.
        """
        # Only produce baseline comparison if this looks like the UNSW dataset.
        # We check that all baseline class names are present in per_class.
        if not all(cls in per_class for cls in TEG_SAGE_F1_BASELINES):
            logger.debug(
                "Skipping TE-G-SAGE baseline comparison: "
                "class names do not match NF-UNSW-NB15-v3 — "
                "expected %s, got %s",
                sorted(TEG_SAGE_F1_BASELINES.keys()),
                sorted(per_class.keys()),
            )
            return {}

        comparison: dict[str, dict] = {}
        for cls_name, baseline_f1 in TEG_SAGE_F1_BASELINES.items():
            shap_gsd_f1 = per_class.get(cls_name, {}).get("f1", None)
            if shap_gsd_f1 is not None:
                comparison[cls_name] = {
                    "teg_sage_f1": baseline_f1,
                    "shap_gsd_f1": shap_gsd_f1,
                    "delta":       round(shap_gsd_f1 - baseline_f1, 4),
                    "improved":    shap_gsd_f1 > baseline_f1,
                }
        return comparison

    @staticmethod
    def _print_summary(
        metrics: dict,
        teg_sage_comparison: dict,
        class_names: list[str],
    ) -> None:
        logger.info("=" * 65)
        logger.info("TEST EVALUATION SUMMARY")
        logger.info(f"  Accuracy:     {metrics['accuracy']:.4f}")
        logger.info(f"  Macro F1:     {metrics['macro_f1']:.4f}")
        logger.info(f"  Weighted F1:  {metrics['weighted_f1']:.4f}")
        logger.info(f"  Test edges:   {metrics['n_test_edges']:,}")
        logger.info("")
        logger.info("Per-class F1:")
        for name in class_names:
            pc = metrics["per_class"].get(name)
            if pc is None:
                continue
            marker = ""
            if name in teg_sage_comparison:
                p1 = teg_sage_comparison[name]["teg_sage_f1"]
                delta = teg_sage_comparison[name]["delta"]
                sign  = "+" if delta >= 0 else ""
                marker = f"  [TE-G-SAGE={p1:.3f}, Δ={sign}{delta:.3f}{'  ✓' if delta > 0 else '  ✗'}]"
            logger.info(
                f"  {name:<14s}  F1={pc['f1']:.4f}  "
                f"P={pc['precision']:.4f}  R={pc['recall']:.4f}  "
                f"n={pc['support']:,}{marker}"
            )
        if teg_sage_comparison:
            logger.info("")
            logger.info("TE-G-SAGE minority-class targets:")
            for name, row in teg_sage_comparison.items():
                status = "IMPROVED" if row["improved"] else "NOT MET"
                logger.info(
                    f"  {name}: TE-G-SAGE={row['teg_sage_f1']:.3f}  "
                    f"SHAP-GSD={row['shap_gsd_f1']:.4f}  [{status}]"
                )
        logger.info("=" * 65)
