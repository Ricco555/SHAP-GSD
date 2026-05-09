"""
Grid search over 108 hyperparameter configurations.

Search space (from configs/tuning_grid.yaml):
  fanouts:     [[15,10], [25,15], [35,25]]   3
  hidden_size: [64, 128, 256]                3
  dropout:     [0.1, 0.2, 0.3, 0.4]         4
  batch_size:  [512, 1024, 2048]             3
  Total: 3 × 3 × 4 × 3 = 108

Per trial: 20 epochs, patience 5, evaluated on val macro-F1.
Selection metric: val_macro_f1 (not accuracy — dominated by benign class).
"""

import copy
import itertools
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

import dgl

from src.data.feature_store import FeatureStore
from src.model.node_state import NodeStateManager
from src.model.sage_model import EdgeAwareGraphSAGE
from src.model.trainer import Trainer

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Grid search over the SHAP-GSD model hyperparameter space."""

    def __init__(
        self,
        search_space: dict,
        fixed_params: dict,
        max_epochs_per_trial: int = 20,
        patience: int = 5,
        selection_metric: str = "val_macro_f1",
    ) -> None:
        """
        Args:
            search_space:         dict of param_name → list of values.
            fixed_params:         dict of param_name → value (held constant).
            max_epochs_per_trial: training epochs per trial.
            patience:             early-stopping patience per trial.
            selection_metric:     metric to maximise ("val_macro_f1").
        """
        self.search_space          = search_space
        self.fixed_params          = fixed_params
        self.max_epochs_per_trial  = max_epochs_per_trial
        self.patience              = patience
        self.selection_metric      = selection_metric

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        g_train: dgl.DGLGraph,
        g_val: dgl.DGLGraph,
        fs_train: FeatureStore,
        fs_val: FeatureStore,
        nsm: NodeStateManager,
        balanced_train_eids: np.ndarray,
        class_weights: torch.Tensor,
        node_in_dim: int,
        edge_in_dim: int,
        num_classes: int,
        base_cfg: dict,
        device: torch.device,
        output_dir: Path,
        seed: int = 42,
        train_label_counts: np.ndarray | None = None,
    ) -> dict:
        """Run all hyperparameter trials and return the best config.

        Args:
            g_train / g_val:     split DGL graphs.
            fs_train / fs_val:   FeatureStore for each split.
            nsm:                 NodeStateManager.
            balanced_train_eids: sorted oversampled EIDs.
            class_weights:       float32 tensor from original distribution.
            train_label_counts:  int array (num_classes,) of original unbalanced
                                 training counts — passed to trainer for composite metric.
            node_in_dim:         node feature dimension (15).
            edge_in_dim:         edge feature dimension (d_e).
            num_classes:         number of output classes.
            base_cfg:            merged config dict; model sub-dict is overridden per trial.
            device:              torch device.
            output_dir:          directory to write tuning_results.json, best_params.json.
            seed:                base random seed (trial i uses seed+i).

        Returns:
            best_params dict (values for the tuned hyperparameters).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        configs = list(self._configs())
        n_total = len(configs)

        # ── Resume: load any previously completed trials ───────────────────────
        tuning_path = output_dir / "tuning_results.json"
        completed: dict[int, dict] = {}
        if tuning_path.exists():
            with open(tuning_path) as f:
                for r in json.load(f):
                    completed[r["trial"]] = r
            logger.info(
                f"Resuming: {len(completed)}/{n_total} trials already done, "
                f"{n_total - len(completed)} remaining."
            )
        else:
            logger.info(f"Starting grid search: {n_total} configurations")

        results: list[dict] = []
        best_val = -1.0
        best_idx = -1

        for i, trial_params in enumerate(configs):
            if i in completed:
                result = completed[i]
                results.append(result)
                f1 = result["best_val_macro_f1"]
                logger.info(f"[Trial {i+1}/{n_total}] SKIP (done, f1={f1:.4f})  {trial_params}")
                if f1 > best_val:
                    best_val = f1
                    best_idx = i
                continue

            trial_start = time.time()
            logger.info(f"\n[Trial {i+1}/{n_total}] {trial_params}")

            cfg_trial = copy.deepcopy(base_cfg)
            cfg_trial["model"].update(trial_params)
            cfg_trial["model"]["max_epochs"] = self.max_epochs_per_trial
            cfg_trial["model"]["patience"]   = self.patience

            model = EdgeAwareGraphSAGE(
                node_in_dim=node_in_dim,
                edge_in_dim=edge_in_dim,
                hidden_size=trial_params["hidden_size"],
                num_classes=num_classes,
                num_layers=self.fixed_params.get("num_layers", 2),
                dropout=trial_params["dropout"],
                aggregator=self.fixed_params.get("aggregator", "mean"),
            ).to(device)

            trainer = Trainer(
                model=model,
                g_train=g_train,
                g_val=g_val,
                fs_train=fs_train,
                fs_val=fs_val,
                nsm=nsm,
                cfg=cfg_trial,
                device=device,
            )

            trial_out = output_dir / f"trial_{i:04d}"
            curves = trainer.train(
                balanced_train_eids=balanced_train_eids,
                class_weights=class_weights,
                output_dir=trial_out,
                seed=seed + i,
                train_label_counts=train_label_counts,
            )

            # Use the same metric as early stopping for trial selection.
            stopping_metric = cfg_trial["model"].get("early_stopping_metric", "macro_f1")
            metric_key = {
                "composite":         "val_composite_f1",
                "minority_macro_f1": "val_minority_macro_f1",
            }.get(stopping_metric, "val_macro_f1")
            best_val_f1 = max(curves[metric_key]) if curves.get(metric_key) else 0.0
            elapsed = time.time() - trial_start

            result = {
                "trial": i,
                "params": trial_params,
                "best_val_macro_f1": best_val_f1,
                "best_epoch": curves["best_epoch"],
                "elapsed_s": elapsed,
            }
            results.append(result)

            logger.info(
                f"Trial {i+1}: best_val_macro_f1={best_val_f1:.4f}  ({elapsed:.1f}s)"
            )

            if best_val_f1 > best_val:
                best_val = best_val_f1
                best_idx = i

            # Flush after every trial so a Ctrl-C loses at most one trial's work.
            with open(tuning_path, "w") as f:
                json.dump(results, f, indent=2)

        best_params = results[best_idx]["params"]
        best_path   = output_dir / "best_params.json"
        with open(best_path, "w") as f:
            json.dump(
                {"best_params": best_params, "best_val_macro_f1": best_val,
                 "best_trial": best_idx},
                f, indent=2,
            )

        logger.info(
            f"\nGrid search complete. Best trial {best_idx}: "
            f"val_macro_f1={best_val:.4f}\n{best_params}"
        )
        return best_params

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _configs(self) -> list[dict]:
        """Enumerate all grid configurations."""
        keys   = list(self.search_space.keys())
        values = [self.search_space[k] for k in keys]
        configs = []
        for combo in itertools.product(*values):
            cfg = dict(zip(keys, combo))
            # Merge fixed params (fixed values are not overridden by grid)
            for k, v in self.fixed_params.items():
                cfg.setdefault(k, v)
            configs.append(cfg)
        return configs
