"""
Training loop with temporal balancing and early stopping.

Key invariants:
- DataLoader iterates training EIDs sorted by timestamp (shuffle=False).
- TemporalNeighborSampler uses LOCAL edge IDs (DGL graph indices 0..n_edges-1).
- Edge features are fetched from FeatureStore using GLOBAL EIDs from g.edata[dgl.EID].
- NodeStateManager receives global node IDs from blocks[0].srcdata[dgl.NID].
- Early stopping on VALIDATION MACRO-F1 (not loss).
- Class weights from the ORIGINAL unbalanced distribution.

Local vs global EID note:
  Training split: local_eid == global_eid (both 0..n_train-1, since EIDs are
  assigned in chronological order and training is the first split).
  Validation split: local_eid ∈ [0, n_val-1], global_eid ∈ [n_train, n_train+n_val-1].
  Always convert local → global via g.edata[dgl.EID] before accessing FeatureStore.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

import dgl

from src.data.feature_store import FeatureStore
from src.model.node_state import NodeStateManager
from src.model.sage_model import EdgeAwareGraphSAGE, build_src_dst_pos
from src.model.temporal_sampler import TemporalNeighborSampler

logger = logging.getLogger(__name__)


class Trainer:
    """Full training loop for EdgeAwareGraphSAGE."""

    def __init__(
        self,
        model: EdgeAwareGraphSAGE,
        g_train: dgl.DGLGraph,
        g_val: dgl.DGLGraph,
        fs_train: FeatureStore,
        fs_val: FeatureStore,
        nsm: NodeStateManager,
        cfg: dict,
        device: torch.device,
    ) -> None:
        """
        Args:
            model:        EdgeAwareGraphSAGE instance (already on device).
            g_train:      training split DGL graph.
            g_val:        validation split DGL graph.
            fs_train:     FeatureStore for training split.
            fs_val:       FeatureStore for validation split.
            nsm:          NodeStateManager (pre-built snapshots).
            cfg:          merged config dict (uses cfg['model'] sub-dict).
            device:       torch device.
        """
        self.model    = model
        self.g_train  = g_train
        self.g_val    = g_val
        self.fs_train = fs_train
        self.fs_val   = fs_val
        self.nsm      = nsm
        self.cfg      = cfg
        self.device   = device

        m = cfg["model"]
        self.batch_size  = m["batch_size"]
        self.max_epochs  = m["max_epochs"]
        self.patience    = m["patience"]
        self.fanouts     = m["fanouts"]

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=m["learning_rate"],
            weight_decay=m.get("weight_decay", 1e-4),
        )
        self.sampler = TemporalNeighborSampler(fanouts=self.fanouts)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(
        self,
        balanced_train_eids: np.ndarray,
        class_weights: torch.Tensor,
        output_dir: Path,
        seed: int = 42,
    ) -> dict:
        """Run the training loop and return training curves.

        Args:
            balanced_train_eids: global EIDs for training (= local EIDs for
                the training split since it starts at EID 0).  Must be
                sortable by timestamp to satisfy shuffle=False invariant.
            class_weights:       float32 tensor (num_classes,) from original distribution.
            output_dir:          directory for best checkpoint and training_curves.json.
            seed:                random seed for reproducibility.

        Returns:
            dict with keys: 'train_loss', 'val_loss', 'val_macro_f1',
                            'val_per_class_f1', 'epoch_times', 'best_epoch'.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))

        # For training split: global_eid == local_eid (both 0-indexed from 0).
        # Sort ascending so TemporalNeighborSampler's non-decreasing assertion holds.
        balanced_sorted = np.sort(balanced_train_eids)

        curves: dict = {
            "train_loss":       [],
            "val_loss":         [],
            "val_macro_f1":     [],
            "val_per_class_f1": [],
            "epoch_times":      [],
            "best_epoch":       -1,
        }

        best_f1    = -1.0
        no_improve = 0

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()

            train_loss = self._run_epoch(
                g=self.g_train,
                fs=self.fs_train,
                local_eids=balanced_sorted,
                criterion=criterion,
                is_train=True,
            )

            val_loss, val_macro_f1, val_per_class = self._evaluate(criterion)

            elapsed = time.time() - t0
            curves["train_loss"].append(float(train_loss))
            curves["val_loss"].append(float(val_loss))
            curves["val_macro_f1"].append(float(val_macro_f1))
            curves["val_per_class_f1"].append(val_per_class)
            curves["epoch_times"].append(float(elapsed))

            logger.info(
                f"Epoch {epoch:3d}/{self.max_epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_macro_f1={val_macro_f1:.4f}  {elapsed:.1f}s"
            )

            if val_macro_f1 > best_f1 + 1e-5:
                best_f1    = val_macro_f1
                no_improve = 0
                curves["best_epoch"] = epoch
                ckpt = output_dir / "best_model.pt"
                torch.save(self.model.state_dict(), ckpt)
                logger.info(f"  → new best val_macro_f1={best_f1:.4f}, saved {ckpt}")
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {self.patience} epochs)"
                    )
                    break

        curves_path = output_dir / "training_curves.json"
        with open(curves_path, "w") as f:
            json.dump(curves, f, indent=2)
        logger.info(
            f"Training done. best_epoch={curves['best_epoch']}, "
            f"best_val_macro_f1={best_f1:.4f}"
        )
        return curves

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        g: dgl.DGLGraph,
        fs: FeatureStore,
        local_eids: np.ndarray,
        criterion: nn.CrossEntropyLoss,
        is_train: bool,
    ) -> float:
        """Run one epoch over local_eids (DGL local edge indices). Return mean loss."""
        self.model.train(is_train)
        total_loss = 0.0
        n_batches  = 0

        loader = DataLoader(
            TensorDataset(torch.from_numpy(local_eids)),
            batch_size=self.batch_size,
            shuffle=False,   # INVARIANT: must remain False
            drop_last=False,
        )

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for (batch_local_t,) in loader:
                batch_local = batch_local_t.long()

                input_nodes, seed_local, blocks = self.sampler.sample_blocks(
                    g, batch_local
                )
                blocks = [b.to(self.device) for b in blocks]

                # Node features from NodeStateManager (global node IDs)
                batch_ts = float(g.edata["timestamp"][batch_local].max().item())
                node_feats = torch.from_numpy(
                    self.nsm.get_batch_states(input_nodes.numpy(), batch_ts)
                ).float().to(self.device)

                # Edge features via GLOBAL EIDs (seed_local == seed_global for train)
                global_eids = g.edata[dgl.EID][seed_local].numpy()
                edge_feats = torch.from_numpy(
                    fs.get_batch(global_eids)
                ).float().to(self.device)

                # Positions in last-block output for edge classification
                seed_nodes = blocks[-1].dstdata[dgl.NID]
                src_pos, dst_pos = build_src_dst_pos(g, seed_local, seed_nodes)
                src_pos = src_pos.to(self.device)
                dst_pos = dst_pos.to(self.device)

                logits = self.model(blocks, node_feats, edge_feats, src_pos, dst_pos)

                labels = torch.from_numpy(
                    fs.get_labels_batch(global_eids)
                ).long().to(self.device)

                loss = criterion(logits, labels)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += float(loss.item())
                n_batches  += 1

        return total_loss / max(n_batches, 1)

    def _evaluate(
        self,
        criterion: nn.CrossEntropyLoss,
    ) -> tuple[float, float, list[float]]:
        """Evaluate on val split. Returns (val_loss, macro_f1, per_class_f1)."""
        self.model.eval()
        total_loss  = 0.0
        n_batches   = 0
        all_preds:  list[int] = []
        all_labels: list[int] = []

        # Use LOCAL edge IDs for the val graph (0..n_val-1)
        local_val_eids = np.arange(self.g_val.num_edges(), dtype=np.int64)

        loader = DataLoader(
            TensorDataset(torch.from_numpy(local_val_eids)),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        with torch.no_grad():
            for (batch_local_t,) in loader:
                batch_local = batch_local_t.long()

                input_nodes, seed_local, blocks = self.sampler.sample_blocks(
                    self.g_val, batch_local
                )
                blocks = [b.to(self.device) for b in blocks]

                batch_ts = float(
                    self.g_val.edata["timestamp"][batch_local].max().item()
                )
                node_feats = torch.from_numpy(
                    self.nsm.get_batch_states(input_nodes.numpy(), batch_ts)
                ).float().to(self.device)

                global_eids = self.g_val.edata[dgl.EID][seed_local].numpy()
                edge_feats = torch.from_numpy(
                    self.fs_val.get_batch(global_eids)
                ).float().to(self.device)

                seed_nodes = blocks[-1].dstdata[dgl.NID]
                src_pos, dst_pos = build_src_dst_pos(
                    self.g_val, seed_local, seed_nodes
                )
                src_pos = src_pos.to(self.device)
                dst_pos = dst_pos.to(self.device)

                logits = self.model(blocks, node_feats, edge_feats, src_pos, dst_pos)

                labels_np = self.fs_val.get_labels_batch(global_eids)
                labels = torch.from_numpy(labels_np).long().to(self.device)

                loss = criterion(logits, labels)
                total_loss += float(loss.item())
                n_batches  += 1

                all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                all_labels.extend(labels_np.tolist())

        val_loss   = total_loss / max(n_batches, 1)
        y_true     = np.array(all_labels)
        y_pred     = np.array(all_preds)
        macro_f1   = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        per_class  = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

        return val_loss, macro_f1, per_class
