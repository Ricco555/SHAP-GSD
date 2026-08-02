"""
Temporal-aware class balancing — training split ONLY.

PROBLEM: NF-UNSW-NB15-v3 is ~96% benign. Class-weighted loss alone
leaves minority-class precision low (TE-G-SAGE: Backdoor F1=0.071).

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
        num_classes: int,
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
        num_classes :     the FULL class-index space (len(label_map), built
                          over the entire dataset before splitting —
                          Preprocessor.num_classes / len(label_map.json)).
                          MUST be >= the number of distinct values in
                          original_labels. A class with zero rows in
                          original_labels still gets a slot in the returned
                          vector, at index `c`, holding weight 0.0 (inert in
                          nn.CrossEntropyLoss — a class absent from training
                          never appears as a loss target). Required, no
                          default: the caller must supply the value it
                          already has in scope (preprocessor.num_classes at
                          the 01_preprocess.py call site; len(attack_to_int)
                          at the fix_labels.py call site) rather than have
                          this method infer a possibly-too-small value from
                          original_labels alone — that inference is exactly
                          the defect this parameter exists to close
                          (specs/39 §1).
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
        torch.Tensor of shape (num_classes,), float32, on CPU. Always has
        EXACTLY `num_classes` entries regardless of how many distinct
        classes appear in original_labels — a class with zero rows gets an
        explicit 0.0 slot at its correct index, never an omitted slot
        (specs/39 §3, specs/40 §1).
        """
        full_counts = np.bincount(
            original_labels.astype(np.int64), minlength=num_classes
        ).astype(np.float64)
        if full_counts.shape[0] > num_classes:
            raise ValueError(
                f"original_labels contains class index "
                f"{int(original_labels.max())}, which is >= num_classes "
                f"({num_classes}). num_classes must be the FULL class-index "
                "space from label_map.json — large enough to cover every "
                "label value that appears in original_labels."
            )

        supported = full_counts > 0
        n_supported = int(supported.sum())
        class_counts = full_counts[supported]   # nonzero-support classes only

        if method == "effective_num":
            # Cui et al., CVPR 2019: E_n = (1 - β^n) / (1 - β)
            if beta == 1.0:
                raise ValueError(
                    "beta=1.0 causes division by zero in effective_num. "
                    "Use beta=0.9999 or switch to inverse_freq."
                )
            effective_num = (1.0 - np.power(beta, class_counts)) / (1.0 - beta)
            weights_supported = 1.0 / effective_num
        elif method == "sqrt_inverse_freq":
            weights_supported = 1.0 / np.sqrt(class_counts)
        elif method == "inverse_freq":
            weights_supported = 1.0 / class_counts
        else:
            raise ValueError(
                f"Unknown class_weight_method '{method}'. "
                "Choose: 'effective_num', 'sqrt_inverse_freq', 'inverse_freq'."
            )

        # Normalize so SUPPORTED weights sum to n_supported (keeps loss
        # magnitude stable) -- computed over the supported subset only, so a
        # zero-support class's presence never perturbs the relative
        # weighting the supported classes receive. When every class has
        # support, n_supported == num_classes and this line is
        # byte-identical to the pre-fix `weights / weights.sum() *
        # num_classes` (specs/39 S3's no-op-when-fully-supported property).
        weights_supported = (
            weights_supported / weights_supported.sum() * n_supported
        )

        if max_clamp is not None:
            weights_supported = np.clip(
                weights_supported, a_min=None, a_max=float(max_clamp)
            )

        # Scatter supported weights back into the full num_classes-length
        # vector; zero-support slots keep their zero-initialized value
        # (specs/39 S3's "safe constant" == 0.0).
        weights = np.zeros(num_classes, dtype=np.float64)
        weights[supported] = weights_supported

        if log_weights:
            logger.info("Class weights (%s, beta=%s):", method, beta)
            for c in range(num_classes):
                n = int(full_counts[c])
                if n == 0:
                    logger.info(
                        "  class %d: weight=0.0000  n=0  (zero training "
                        "support -- inert in CrossEntropyLoss)", c
                    )
                else:
                    logger.info("  class %d: weight=%.4f  n=%d", c, weights[c], n)
            if n_supported > 0:
                logger.info(
                    "  weight ratio max/min (supported classes only): %.1f",
                    weights_supported.max() / weights_supported.min(),
                )
            if n_supported < num_classes:
                logger.info(
                    "  %d/%d classes have zero training-split support: %s",
                    num_classes - n_supported, num_classes,
                    sorted(int(c) for c in range(num_classes) if not supported[c]),
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
