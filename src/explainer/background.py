"""
Class-conditional background distributions. Training split only.

background_features[c]   = mean edge features for class c across training
background_node_state[c] = mean 15-dim node state for class c across training

Shapes:
  background_features.npy    (num_classes, d_e)
  background_node_state.npy  (num_classes, 15)

Used by all three SHAP granularities for absent-coalition replacement.
NEVER use val/test data here.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_BG_FEATURES_FILE = "background_features.npy"
_BG_NODE_STATE_FILE = "background_node_state.npy"


class BackgroundDistributions:
    """Class-conditional mean feature/node-state vectors from the training split."""

    def __init__(
        self,
        background_features: np.ndarray,
        background_node_state: np.ndarray,
    ) -> None:
        """
        Args:
            background_features:  float32 array (num_classes, d_e).
            background_node_state: float32 array (num_classes, node_state_dim).
        """
        assert background_features.shape[0] == background_node_state.shape[0], (
            "background_features and background_node_state must have the same num_classes"
        )
        self.background_features = background_features.astype(np.float32)
        self.background_node_state = background_node_state.astype(np.float32)
        self.num_classes = background_features.shape[0]

    @classmethod
    def compute(
        cls,
        feature_store_train_dir: "Path | str",
        nsm: "NodeStateManager",
        g_train: "dgl.DGLGraph",
        num_classes: int,
        node_state_dim: int = 15,
        sample_per_class: int = 1000,
        rng_seed: int = 456,
    ) -> "BackgroundDistributions":
        """Compute background distributions from the training split.

        For edge features: exact per-class mean over all training edges.
        For node states: per-class mean over a random sample (sample_per_class
        edges per class) to keep memory bounded.

        Args:
            feature_store_train_dir: path to feature_store/train/
            nsm:              NodeStateManager (loaded from node_state_snapshots/)
            g_train:          DGL training graph with edata['timestamp']
            num_classes:      number of output classes (e.g. 10)
            node_state_dim:   node state vector dimension (default 15)
            sample_per_class: max edges sampled per class for node state mean
            rng_seed:         numpy RNG seed for reproducible sampling
        """
        fs_dir = Path(feature_store_train_dir)
        labels = np.load(fs_dir / "labels.npy")
        edge_indices = np.load(fs_dir / "edge_indices.npy")
        timestamps = np.load(fs_dir / "timestamps.npy")

        n_train = len(edge_indices)
        feat_path = fs_dir / "features.dat"
        nbytes = feat_path.stat().st_size
        d_e = nbytes // (n_train * 4)  # float32 = 4 bytes
        features = np.memmap(feat_path, dtype=np.float32, mode="r", shape=(n_train, d_e))

        logger.info(
            f"Computing background distributions: {n_train:,} training edges, "
            f"d_e={d_e}, {num_classes} classes"
        )

        # --- Per-class mean edge features (chunked to stay memory-safe) ---
        bg_features = np.zeros((num_classes, d_e), dtype=np.float64)
        class_counts = np.zeros(num_classes, dtype=np.int64)
        chunk_size = 50_000
        for start in range(0, n_train, chunk_size):
            end = min(start + chunk_size, n_train)
            chunk_feat = features[start:end]
            chunk_labels = labels[start:end]
            for c in range(num_classes):
                mask = chunk_labels == c
                if mask.any():
                    bg_features[c] += chunk_feat[mask].sum(axis=0)
                    class_counts[c] += int(mask.sum())

        for c in range(num_classes):
            if class_counts[c] > 0:
                bg_features[c] /= class_counts[c]
            else:
                logger.warning(f"Class {c} has no training samples; edge feature background set to zeros")

        logger.info(f"Edge feature backgrounds computed. Class counts: {class_counts.tolist()}")

        # --- Per-class mean node states (sampled) ---
        src_nodes, dst_nodes = g_train.edges()
        src_nodes = src_nodes.numpy()
        dst_nodes = dst_nodes.numpy()

        rng = np.random.default_rng(rng_seed)
        bg_node_state = np.zeros((num_classes, node_state_dim), dtype=np.float64)
        ns_counts = np.zeros(num_classes, dtype=np.int64)

        for c in range(num_classes):
            class_idx = np.where(labels == c)[0]
            if len(class_idx) == 0:
                continue
            sample_idx = rng.choice(
                class_idx,
                size=min(sample_per_class, len(class_idx)),
                replace=False,
            )
            states: list[np.ndarray] = []
            for i in sample_idx:
                t_ms = float(timestamps[i])
                s_state = nsm.get_state_at_time(int(src_nodes[i]), t_ms)
                d_state = nsm.get_state_at_time(int(dst_nodes[i]), t_ms)
                # Average src and dst: background represents a "generic" node
                states.append((s_state + d_state) * 0.5)
            if states:
                bg_node_state[c] = np.mean(states, axis=0)
                ns_counts[c] = len(states)

        logger.info(f"Node state backgrounds computed. Samples per class: {ns_counts.tolist()}")

        return cls(
            background_features=bg_features.astype(np.float32),
            background_node_state=bg_node_state.astype(np.float32),
        )

    def save(self, artifacts_dir: "Path | str") -> None:
        """Save background arrays to artifacts_dir/."""
        out = Path(artifacts_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / _BG_FEATURES_FILE, self.background_features)
        np.save(out / _BG_NODE_STATE_FILE, self.background_node_state)
        logger.info(
            f"Background distributions saved → {out}  "
            f"(features {self.background_features.shape}, "
            f"node_state {self.background_node_state.shape})"
        )

    @classmethod
    def load(cls, artifacts_dir: "Path | str") -> "BackgroundDistributions":
        """Load pre-computed background distributions from artifacts_dir/."""
        out = Path(artifacts_dir)
        bg_feat = np.load(out / _BG_FEATURES_FILE)
        bg_ns = np.load(out / _BG_NODE_STATE_FILE)
        logger.info(
            f"Background distributions loaded: "
            f"features {bg_feat.shape}, node_state {bg_ns.shape}"
        )
        return cls(background_features=bg_feat, background_node_state=bg_ns)
