"""
Temporal-aware class balancing — training split ONLY.

PROBLEM: NF-UNSW-NB15-v3 is ~96% benign. Class-weighted loss alone
leaves minority-class precision low (Paper 1: Backdoor F1=0.071).

SOLUTION: Oversample minority classes while preserving temporal ordering.
Val/test splits are NEVER touched.

CLASS WEIGHTS: Computed from the ORIGINAL unbalanced distribution.
Use alongside balancing (addresses both sampling frequency and gradient magnitude).

STRATEGIES:
  oversample  — duplicate minority flows (with replacement) to min_class_ratio×majority
  undersample — drop majority flows to max_majority_ratio×total_attack_count
  hybrid      — undersample first, then oversample
"""

import logging
import math

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TemporalBalancer:
    """Balance the training split while preserving temporal edge ordering."""

    def __init__(
        self,
        strategy: str = "oversample",
        min_class_ratio: float = 0.1,
        max_majority_ratio: float = 5.0,
        seed: int = 42,
    ) -> None:
        """
        Args:
            strategy: "oversample" | "undersample" | "hybrid"
            min_class_ratio: target minority count = min_class_ratio × majority count
            max_majority_ratio: target majority count = max_majority_ratio × total_attack_count
            seed: RNG seed for reproducibility
        """
        assert strategy in ("oversample", "undersample", "hybrid"), (
            f"Unknown strategy: {strategy!r}"
        )
        self.strategy = strategy
        self.min_class_ratio = min_class_ratio
        self.max_majority_ratio = max_majority_ratio
        self.rng = np.random.default_rng(seed)

    def balance(
        self,
        edge_ids: np.ndarray,
        timestamps: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Return balanced EID array sorted by timestamp. May contain duplicates.

        Duplicated EIDs point to the same feature store rows — no data duplication.
        All original training EIDs are present in the output (oversample mode).

        Args:
            edge_ids:   global EIDs, shape (n,)
            timestamps: FLOW_START_MILLISECONDS, shape (n,)
            labels:     integer class labels, shape (n,)

        Returns:
            balanced_eids: global EIDs sorted by timestamp, shape (m,)
        """
        # Build O(1) numpy EID→timestamp lookup (avoids Python dict iteration)
        eid_max = int(edge_ids.max())
        ts_lookup = np.zeros(eid_max + 1, dtype=np.int64)
        ts_lookup[edge_ids] = timestamps
        lb_lookup = np.zeros(eid_max + 1, dtype=np.int64)
        lb_lookup[edge_ids] = labels

        if self.strategy == "oversample":
            result = self._oversample(edge_ids, ts_lookup, lb_lookup)
        elif self.strategy == "undersample":
            result = self._undersample(edge_ids, ts_lookup, lb_lookup)
        else:  # hybrid
            step1 = self._undersample(edge_ids, ts_lookup, lb_lookup)
            result = self._oversample(step1, ts_lookup, lb_lookup)

        # Final sort by timestamp (vectorized)
        result = result[np.argsort(ts_lookup[result])]

        logger.info(
            f"Balanced training set: {len(edge_ids):,} → {len(result):,} edges "
            f"(strategy={self.strategy})"
        )
        return result

    def get_class_weights(
        self,
        original_labels: np.ndarray,
        method: str = "effective_num",
        beta: float = 0.9999,
        max_clamp: float | None = None,
        log_weights: bool = True,
    ) -> torch.Tensor:
        """Compute class weights from the ORIGINAL (unbalanced) label distribution.

        Methods
        -------
        "effective_num"     Cui et al., CVPR 2019. E_n = (1 - β^n) / (1 - β).
                            weight_c = 1 / E_{n_c}, normalized so weights sum to
                            num_classes. Naturally bounded — safe across datasets.
        "sqrt_inverse_freq" weight_c = 1 / sqrt(n_c), normalized. Bounded
                            alternative, no citation needed.
        "inverse_freq"      weight_c = 1 / n_c, normalized. Retained for ablation
                            only — produces extreme values for rare classes.

        Parameters
        ----------
        original_labels : int labels from the ORIGINAL unbalanced training split.
                          Never pass balanced/resampled labels.
        method :          weighting method (default "effective_num").
        beta :            effective number β for method="effective_num".
                          β=0 → uniform; β→1 → inverse frequency.
        max_clamp :       if not None, clamp weights to this maximum after
                          normalization. Use only as last-resort — prefer
                          reducing β instead.
        log_weights :     log per-class weights and max/min ratio. Always True
                          for paper runs — reviewers will ask for these values.

        Returns
        -------
        torch.Tensor of shape (num_classes,), float32, on CPU.
        """
        classes = np.unique(original_labels)
        num_classes = len(classes)
        class_counts = np.array(
            [np.sum(original_labels == c) for c in classes], dtype=np.float64
        )

        if method == "effective_num":
            # Cui et al., CVPR 2019: E_n = (1 - β^n) / (1 - β)
            if beta == 1.0:
                raise ValueError(
                    "beta=1.0 causes division by zero in effective_num. "
                    "Use beta=0.9999 or switch to inverse_freq."
                )
            effective_num = (1.0 - np.power(beta, class_counts)) / (1.0 - beta)
            weights = 1.0 / effective_num
        elif method == "sqrt_inverse_freq":
            weights = 1.0 / np.sqrt(class_counts)
        elif method == "inverse_freq":
            weights = 1.0 / class_counts
        else:
            raise ValueError(
                f"Unknown class_weight_method '{method}'. "
                "Choose: 'effective_num', 'sqrt_inverse_freq', 'inverse_freq'."
            )

        # Normalize so weights sum to num_classes (keeps loss magnitude stable)
        weights = weights / weights.sum() * num_classes

        if max_clamp is not None:
            weights = np.clip(weights, a_min=None, a_max=float(max_clamp))

        if log_weights:
            logger.info("Class weights (%s, beta=%s):", method, beta)
            for c, w, n in zip(classes, weights, class_counts):
                logger.info("  class %d: weight=%.4f  n=%d", int(c), w, int(n))
            logger.info(
                "  weight ratio max/min: %.1f", weights.max() / weights.min()
            )

        return torch.tensor(weights, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _oversample(
        self,
        edge_ids: np.ndarray,
        ts_lookup: np.ndarray,
        lb_lookup: np.ndarray,
    ) -> np.ndarray:
        """Oversample minority classes; preserve all original EIDs.

        Args:
            edge_ids:  global EIDs in this split
            ts_lookup: numpy array where ts_lookup[eid] = timestamp
            lb_lookup: numpy array where lb_lookup[eid] = label
        """
        labels = lb_lookup[edge_ids]
        classes, counts = np.unique(labels, return_counts=True)
        majority_count = counts.max()
        target_count = math.ceil(majority_count * self.min_class_ratio)

        extra_eids: list[np.ndarray] = [edge_ids.copy()]
        for cls, cnt in zip(classes, counts):
            if cnt >= target_count:
                continue
            needed = target_count - cnt
            cls_eids = edge_ids[labels == cls]
            sampled = self.rng.choice(cls_eids, size=needed, replace=True)
            extra_eids.append(sampled)
            logger.info(
                f"  Oversampled class {cls}: {cnt:,} → {cnt + needed:,} (+{needed:,})"
            )

        return np.concatenate(extra_eids)

    def _undersample(
        self,
        edge_ids: np.ndarray,
        ts_lookup: np.ndarray,
        lb_lookup: np.ndarray,
    ) -> np.ndarray:
        """Undersample majority class.

        Args:
            edge_ids:  global EIDs in this split
            ts_lookup: numpy array where ts_lookup[eid] = timestamp
            lb_lookup: numpy array where lb_lookup[eid] = label
        """
        labels = lb_lookup[edge_ids]
        classes, counts = np.unique(labels, return_counts=True)
        majority_cls = classes[counts.argmax()]
        total_attack = len(labels) - counts.max()
        target_majority = int(self.max_majority_ratio * total_attack)

        majority_eids = edge_ids[labels == majority_cls]
        minority_eids = edge_ids[labels != majority_cls]

        if len(majority_eids) > target_majority:
            majority_eids = self.rng.choice(
                majority_eids, size=target_majority, replace=False
            )
            logger.info(
                f"  Undersampled class {majority_cls}: "
                f"{counts.max():,} → {target_majority:,}"
            )

        return np.concatenate([majority_eids, minority_eids])
