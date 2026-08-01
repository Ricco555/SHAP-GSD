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

Shard mode (specs/34, specs/35): when run() receives a shard spec it OWNS
only the trials whose axis values match the spec's owned_values, writes its
results to a per-shard file in the same tuning directory, and NEVER writes
tuning_results.json or best_params.json — scripts/promote_best.py merges the
shard files and emits those two artifacts.
"""

import copy
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
from src.model.selection import (
    build_shard_header,
    enumerate_grid,
    select_best,
    write_json_atomic,
)
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
        shard: dict | None = None,
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
            shard:               Optional shard spec from
                                 ``src.model.selection.resolve_shard`` (plus a
                                 ``grid_fingerprint`` key added by the caller).
                                 When given, this process OWNS ONLY the trials
                                 whose axis values match ``shard["owned_values"]``;
                                 it reads/writes only
                                 ``output_dir / shard["filename"]`` and NEVER
                                 writes ``tuning_results.json`` or
                                 ``best_params.json`` (specs/34 §3.5). Default
                                 ``None`` = today's single-job behavior,
                                 unchanged.

        Returns:
            best_params dict (values for the tuned hyperparameters). In shard
            mode this is the SHARD-LOCAL best, not the grid-wide winner.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        configs = list(self._configs())
        n_total = len(configs)

        if shard is not None:
            # Guard against a tuning-grid edit between resolve_shard and run()
            # (both currently derive from the same in-memory dict, so this
            # cannot fire today — specs/35 §III.2).
            assert len(configs) == shard["n_total"], (
                f"grid size {len(configs)} != shard n_total "
                f"{shard['n_total']} — the tuning grid changed between shard "
                f"resolution and run()"
            )

        # Selection-metric curve key — same expression as the per-trial one.
        selection_metric_used = SELECTION_METRIC_CURVE_KEY.get(
            base_cfg["model"].get("early_stopping_metric", "macro_f1"),
            "val_macro_f1",
        )

        header: dict | None = None
        if shard is not None:
            # Single source of truth for the 12-key header schema
            # (src.model.selection.build_shard_header) — shared with any
            # test fixture that fabricates shard files, so the two cannot
            # drift apart (specs/35 §III.5).
            header = build_shard_header(shard, selection_metric_used)

        # ── Resume: load any previously completed trials ───────────────────────
        tuning_path = output_dir / (
            shard["filename"] if shard is not None else "tuning_results.json"
        )
        n_owned = len(shard["owned_indices"]) if shard is not None else n_total
        completed: dict[int, dict] = {}
        if tuning_path.exists():
            with open(tuning_path) as f:
                stored = json.load(f)
            if shard is not None:
                self._check_shard_resume_header(
                    stored["header"], header, tuning_path
                )
                records = stored["results"]
            else:
                records = stored
            for r in records:
                completed[r["trial"]] = r
            logger.info(
                f"Resuming: {len(completed)}/{n_owned} trials already done, "
                f"{n_owned - len(completed)} remaining."
            )
        else:
            logger.info(
                f"Starting grid search: {n_total} configurations"
                + (f" ({n_owned} owned by this shard)" if shard is not None else "")
            )

        results: list[dict] = []
        owned = shard["owned_values"] if shard is not None else None

        for i, trial_params in enumerate(configs):   # ALWAYS the FULL list — never
            if owned is not None and any(            # truncated, sliced or pre-filtered
                trial_params[a] != v for a, v in owned.items()
            ):
                continue
            if i in completed:
                result = completed[i]
                results.append(result)
                f1 = result["best_val_macro_f1"]
                logger.info(f"[Trial {i+1}/{n_total}] SKIP (done, f1={f1:.4f})  {trial_params}")
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

            # Flush after every trial so a Ctrl-C loses at most one trial's work.
            if shard is not None:
                write_json_atomic(
                    tuning_path, {"header": header, "results": results}
                )
            else:
                write_json_atomic(tuning_path, results)

        best = select_best(results)
        if best is None:
            raise RuntimeError("no trial results to select from")

        if shard is None:
            best_path = output_dir / "best_params.json"
            write_json_atomic(
                best_path,
                {"best_params": best["params"],
                 "best_val_macro_f1": best["best_val_macro_f1"],
                 "selection_metric_used": best.get("selection_metric_used"),
                 "best_trial": best["trial"]},
            )
            logger.info(
                f"\nGrid search complete. Best trial {best['trial']}: "
                f"{best.get('selection_metric_used') or 'unrecorded metric'}"
                f"={best['best_val_macro_f1']:.4f}\n{best['params']}"
            )
        else:
            logger.info(
                "\nShard %d/%d complete. Shard-local best trial %d: %s=%.4f "
                "(NOT promoted — run scripts/promote_best.py after all shards "
                "finish).\n%s",
                shard["shard_index"], shard["num_shards"], best["trial"],
                best.get("selection_metric_used") or "unrecorded metric",
                best["best_val_macro_f1"], best["params"],
            )
        return best["params"]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _check_shard_resume_header(
        stored: dict, current: dict, path: Path
    ) -> None:
        """Fail-closed shard-resume guard (specs/35 §III.5).

        The stored shard file's header must agree with the current
        invocation on every identity field — a ``tuning_grid.yaml`` or
        config edit between submission and resume must not silently
        continue (the sharded analogue of specs/28/29's staleness guard).
        A ``pbs_jobid`` mismatch is only a warning: a resubmission after a
        walltime kill is legitimate, and the header is rewritten with the
        current job's id on the next flush.

        Raises:
            RuntimeError: naming the field, the file, and both values, on
                any identity-field mismatch.
        """
        for field in (
            "shard_index", "num_shards", "n_total", "partition_scheme",
            "shard_axes", "owned_values", "grid_fingerprint",
        ):
            if stored.get(field) != current[field]:
                raise RuntimeError(
                    f"Shard resume refused: header field {field!r} in {path} "
                    f"is {stored.get(field)!r} but this invocation computed "
                    f"{current[field]!r}. The tuning grid or effective config "
                    f"changed between submission and resume — move the stale "
                    f"shard file aside to start this shard over."
                )
        if stored.get("pbs_jobid") != current["pbs_jobid"]:
            logger.warning(
                "Shard resume under a different PBS job id (%s -> %s) — "
                "legitimate after a walltime kill; continuing.",
                stored.get("pbs_jobid"), current["pbs_jobid"],
            )

    def _configs(self) -> list[dict]:
        """Enumerate all grid configurations.

        Delegates the base enumeration to
        ``src.model.selection.enumerate_grid`` — the single enumeration
        authority shared with ``resolve_shard`` — then merges fixed params.
        ``setdefault`` cannot alter axis values, so for every axis in
        ``search_space``, ``_configs()[i][axis] ==
        enumerate_grid(search_space)[i][axis]``.
        """
        configs = enumerate_grid(self.search_space)
        for cfg in configs:
            # Merge fixed params (fixed values are not overridden by grid)
            for k, v in self.fixed_params.items():
                cfg.setdefault(k, v)
        return configs
