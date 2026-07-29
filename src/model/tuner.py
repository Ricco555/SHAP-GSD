"""
Grid search over 108 hyperparameter configurations.

Search space (from configs/tuning_grid.yaml):
  fanouts:     [[15,10], [25,15], [35,25]]   3
  hidden_size: [64, 128, 256]                3
  dropout:     [0.1, 0.2, 0.3, 0.4]         4
  batch_size:  [512, 1024, 2048]             3
  Total: 3 × 3 × 4 × 3 = 108

Per-trial budget (max_epochs, patience) comes from configs/tuning_grid.yaml's
trial: block — see resolve_trial_settings(). Shipped value: 40 epochs,
patience 20 (specs/33 §II.2).

Cross-trial selection uses the SAME metric as per-trial early stopping:
model.early_stopping_metric from the experiment config, mapped to a training
curve key ("composite" -> val_composite_f1, "minority_macro_f1" ->
val_minority_macro_f1, otherwise val_macro_f1). Each trial result records the
key actually maximised in its selection_metric_used field.
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

#: Required sub-keys of ``configs/tuning_grid.yaml``'s ``trial:`` block.
TRIAL_SETTING_KEYS: tuple[str, ...] = ("max_epochs", "patience")

#: Maps ``model.early_stopping_metric`` policy names to the training-curve key
#: they correspond to. Unrecognised policy names fall back to
#: ``"val_macro_f1"`` at the call site (``dict.get`` default).
SELECTION_METRIC_CURVE_KEY: dict[str, str] = {
    "composite":         "val_composite_f1",
    "minority_macro_f1": "val_minority_macro_f1",
    "macro_f1":          "val_macro_f1",
}


def resolve_trial_settings(grid_cfg: dict) -> dict:
    """Resolve the per-trial budget from ``configs/tuning_grid.yaml``.

    Reads ``grid_cfg["trial"]["max_epochs"]`` and
    ``grid_cfg["trial"]["patience"]`` and returns them under the keyword names
    ``HyperparameterTuner.__init__`` expects. Fails fast: a missing block or
    key raises ``KeyError`` rather than silently substituting a default, which
    is the exact bug class this function exists to remove (specs/32 §2.2).

    Unrecognised keys under ``trial:`` are ignored, not rejected — the
    ../SHAP-GSD-hpc mirror may still carry a ``selection_metric:`` key until it
    is synced (specs/32 §6), and that must stay harmless.

    Args:
        grid_cfg: the parsed ``configs/tuning_grid.yaml`` (already merged with
            ``configs/default.yaml`` by ``load_config``).

    Returns:
        ``{"max_epochs_per_trial": int, "patience": int}``.

    Raises:
        KeyError: if ``trial:`` or either required sub-key is absent.
    """
    trial = grid_cfg.get("trial")
    if not isinstance(trial, dict):
        raise KeyError(
            "configs/tuning_grid.yaml is missing the required 'trial:' block "
            "(expected keys: max_epochs, patience). Refusing to guess a per-trial "
            "budget — see specs/33."
        )

    for key in TRIAL_SETTING_KEYS:
        if key not in trial:
            raise KeyError(
                f"configs/tuning_grid.yaml: trial.{key} is required and was not found "
                f"(present keys: {sorted(trial)}). Refusing to substitute a default — "
                f"see specs/33."
            )

    max_epochs = int(trial["max_epochs"])
    patience   = int(trial["patience"])

    logger.info(
        "Per-trial budget from configs/tuning_grid.yaml [trial]: "
        "max_epochs=%d, patience=%d", max_epochs, patience
    )
    return {"max_epochs_per_trial": max_epochs, "patience": patience}


class HyperparameterTuner:
    """Grid search over the SHAP-GSD model hyperparameter space."""

    def __init__(
        self,
        search_space: dict,
        fixed_params: dict,
        max_epochs_per_trial: int,
        patience: int,
    ) -> None:
        """
        Args:
            search_space:         dict of param_name → list of values.
            fixed_params:         dict of param_name → value (held constant).
            max_epochs_per_trial: training epochs per trial.
            patience:             early-stopping patience per trial.

        Both ``max_epochs_per_trial`` and ``patience`` are required — the
        per-trial budget has exactly one authority, ``configs/tuning_grid.yaml``'s
        ``trial:`` block, resolved by ``resolve_trial_settings()``.
        """
        self.search_space          = search_space
        self.fixed_params          = fixed_params
        self.max_epochs_per_trial  = max_epochs_per_trial
        self.patience              = patience

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
            metric_key = SELECTION_METRIC_CURVE_KEY.get(stopping_metric, "val_macro_f1")
            best_val_f1 = max(curves[metric_key]) if curves.get(metric_key) else 0.0
            elapsed = time.time() - trial_start

            result = {
                "trial": i,
                "params": trial_params,
                "best_val_macro_f1": best_val_f1,
                "selection_metric_used": metric_key,
                "best_epoch": curves["best_epoch"],
                "elapsed_s": elapsed,
            }
            results.append(result)

            logger.info(
                f"Trial {i+1}: {metric_key}={best_val_f1:.4f}  ({elapsed:.1f}s)"
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
                 "selection_metric_used": results[best_idx].get("selection_metric_used"),
                 "best_trial": best_idx},
                f, indent=2,
            )

        logger.info(
            f"\nGrid search complete. Best trial {best_idx}: "
            f"{results[best_idx].get('selection_metric_used') or 'unrecorded metric'}"
            f"={best_val:.4f}\n{best_params}"
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
