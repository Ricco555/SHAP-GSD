"""
Semantic feature grouping for SHAP-GSD.

PROBLEM: Raw d_e-dim feature vector has one-hot blocks for each categorical
variable. KernelSHAP over d_e dims has high variance and redundant attributions.

SOLUTION: Group by original variable. K = 48 semantic groups (confirmed by preprocessing):
  - Each pruned numeric feature → one group (1 column each)
  - DST_PORT_GROUP (16-bin one-hot) → one group
  - SRC_PORT_IS_EPHEMERAL (1 binary) → one group
  - Each categorical variable's full one-hot block → one group

SHAP coalition vector z ∈ {0,1}^K:
  z[j]=1: group g_j columns pass through unchanged
  z[j]=0: group g_j columns replaced with class-conditional background

VALIDATION: union of all groups = {0, ..., d_e-1}, no overlaps.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FeatureGrouping:
    """Map between K semantic groups and d_e feature dimensions.

    Built from feature_groups.json written by the preprocessing pipeline.
    """

    def __init__(
        self,
        feature_names: list[str],
        groups: dict[str, dict[str, Any]],
    ) -> None:
        """
        Args:
            feature_names: ordered list of all d_e column names
            groups: dict mapping group_name → {"indices": [...], "type": str}
        """
        self.feature_names = feature_names
        self.d_e = len(feature_names)
        self.groups = groups  # name → {"indices": list[int], "type": str}
        self.group_names: list[str] = list(groups.keys())
        self.K: int = len(self.group_names)

        self._validate()

        # Lookup: group_name → column index array
        self._group_indices: dict[str, np.ndarray] = {
            name: np.array(info["indices"], dtype=np.int64)
            for name, info in groups.items()
        }

        logger.info(
            f"FeatureGrouping: K={self.K} groups, d_e={self.d_e}"
        )

    @classmethod
    def from_json(cls, path: Path | str) -> "FeatureGrouping":
        """Load from feature_groups.json."""
        with open(path) as f:
            data = json.load(f)
        return cls(data["feature_names"], data["groups"])

    @classmethod
    def build(
        cls,
        numeric_cols_kept: list[str],
        ohe_feature_names: list[str],
        categorical_cols: list[str],
    ) -> "FeatureGrouping":
        """Construct grouping from preprocessing outputs.

        Feature vector layout assumed:
          [numeric_cols_kept | DST_PORT_GROUP (16) | SRC_PORT_IS_EPHEMERAL | OHE_block]

        Args:
            numeric_cols_kept: pruned numeric column names (in order)
            ohe_feature_names: output of OneHotEncoder.get_feature_names_out()
            categorical_cols:  ordered list of categorical variable names passed to OHE
        """
        from src.data.preprocessor import DST_PORT_BIN_NAMES, N_DST_PORT_BINS

        groups: dict[str, dict[str, Any]] = {}
        offset = 0

        # Numeric groups (one column each)
        for col in numeric_cols_kept:
            groups[col] = {"indices": [offset], "type": "numeric"}
            offset += 1

        # DST_PORT_GROUP (16 bins)
        groups["DST_PORT_GROUP"] = {
            "indices": list(range(offset, offset + N_DST_PORT_BINS)),
            "type": "port_bin",
        }
        offset += N_DST_PORT_BINS

        # SRC_PORT_IS_EPHEMERAL
        groups["SRC_PORT_IS_EPHEMERAL"] = {"indices": [offset], "type": "port_bin"}
        offset += 1

        # Categorical one-hot blocks (each variable is one group)
        for cat_col in categorical_cols:
            prefix = cat_col + "_"
            col_indices = [
                offset + i
                for i, name in enumerate(ohe_feature_names)
                if name.startswith(prefix)
            ]
            if col_indices:
                groups[cat_col] = {"indices": col_indices, "type": "categorical"}

        # Build feature_names list
        feature_names = (
            numeric_cols_kept
            + DST_PORT_BIN_NAMES
            + ["SRC_PORT_IS_EPHEMERAL"]
            + list(ohe_feature_names)
        )

        return cls(feature_names, groups)

    def get_group_mask(self, coalition_vector: np.ndarray) -> np.ndarray:
        """Convert K-dim binary coalition vector to d_e-dim binary column mask.

        Args:
            coalition_vector: shape (K,), dtype bool or int, 1 = present

        Returns:
            mask: shape (d_e,), dtype bool — True = column passes through
        """
        assert len(coalition_vector) == self.K, (
            f"Coalition vector length {len(coalition_vector)} ≠ K={self.K}"
        )
        mask = np.zeros(self.d_e, dtype=bool)
        for j, name in enumerate(self.group_names):
            if coalition_vector[j]:
                mask[self._group_indices[name]] = True
        return mask

    def get_background_vector(
        self,
        class_label: int,
        class_backgrounds: dict[int, np.ndarray],
    ) -> np.ndarray:
        """Return class-conditional background for absent groups.

        Args:
            class_label:        integer class label
            class_backgrounds:  dict class_int → mean feature vector (d_e,)

        Returns:
            background: shape (d_e,) — training mean for this class
        """
        assert class_label in class_backgrounds, (
            f"No background for class {class_label}"
        )
        return class_backgrounds[class_label].copy()

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "d_e": self.d_e,
            "K": self.K,
            "feature_names": self.feature_names,
            "groups": {
                name: {"indices": info["indices"], "type": info["type"]}
                for name, info in self.groups.items()
            },
        }

    def save(self, path: Path | str) -> None:
        """Write feature_groups.json."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Feature groups saved to {path}  (K={self.K}, d_e={self.d_e})")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Assert groups partition {0, ..., d_e-1} exactly (no overlaps, full cover)."""
        seen: set[int] = set()
        for name, info in self.groups.items():
            for idx in info["indices"]:
                assert idx not in seen, (
                    f"Column {idx} appears in multiple groups (duplicate in group '{name}')"
                )
                seen.add(idx)
        all_expected = set(range(self.d_e))
        missing = all_expected - seen
        extra   = seen - all_expected
        assert not missing, f"Groups missing columns: {sorted(missing)}"
        assert not extra,   f"Groups reference out-of-range columns: {sorted(extra)}"
        logger.debug(f"FeatureGrouping validation passed: {self.K} groups cover all {self.d_e} columns")


# ---------------------------------------------------------------------------
# Raw-feature (ungrouped) coalition space — paper config (i)
# ---------------------------------------------------------------------------

def build_singleton_feature_groups(source: dict[str, Any]) -> dict[str, Any]:
    """Derive a one-player-per-raw-feature grouping from a semantic grouping.

    Produces the coalition space for the paper's SHAP-GSD config (i): raw,
    ungrouped KernelSHAP over the ``d_e`` encoded edge-feature dimensions,
    i.e. ``K == d_e`` singleton groups instead of the 48 semantic groups.
    The output has exactly the ``feature_groups.json`` schema that
    ``FeatureGroupSHAP``, ``FeatureGrouping`` and ``scripts/08_metrics.py``
    already consume, so no explainer or metric code has to change.

    The transform is deterministic and order-preserving: group ``i`` is named
    ``source["feature_names"][i]`` and owns exactly column ``i``. Each
    singleton inherits the ``type`` of the semantic group that contained its
    column, so downstream type-aware consumers keep working.

    Args:
        source: parsed ``feature_groups.json`` dict with keys ``d_e``, ``K``,
                ``feature_names`` and ``groups``. Must be the RUN-SCOPED file
                of the run being explained (its ``d_e`` must match that run's
                feature store), not a stale top-level copy.

    Returns:
        A new ``feature_groups.json``-shaped dict with ``K == d_e`` and one
        single-column group per raw feature. A ``singleton_source`` provenance
        key records the grouping it was derived from; no consumer reads it.

    Raises:
        ValueError: if the source is internally inconsistent (``K`` disagrees
            with ``groups``, ``feature_names`` length disagrees with ``d_e``,
            duplicate feature names, or the groups do not partition
            ``{0, ..., d_e-1}`` exactly).
    """
    required = ("d_e", "K", "feature_names", "groups")
    missing_keys = [k for k in required if k not in source]
    if missing_keys:
        raise ValueError(f"source feature_groups is missing keys: {missing_keys}")

    d_e: int = int(source["d_e"])
    feature_names: list[str] = list(source["feature_names"])
    src_groups: dict[str, Any] = source["groups"]

    if int(source["K"]) != len(src_groups):
        raise ValueError(
            f"source K={source['K']} disagrees with len(groups)={len(src_groups)}"
        )
    if len(feature_names) != d_e:
        raise ValueError(
            f"source has {len(feature_names)} feature_names but d_e={d_e}"
        )
    if len(set(feature_names)) != d_e:
        dupes = sorted({n for n in feature_names if feature_names.count(n) > 1})
        raise ValueError(
            f"feature_names are not unique — cannot key singleton groups by "
            f"name; duplicates: {dupes}"
        )

    # Column -> type of the semantic group owning it; doubles as the partition
    # check (every column claimed exactly once, nothing out of range).
    col_type: dict[int, str] = {}
    for name, info in src_groups.items():
        for idx in info["indices"]:
            if idx in col_type:
                raise ValueError(
                    f"column {idx} appears in more than one source group "
                    f"(duplicate found in '{name}')"
                )
            if not 0 <= idx < d_e:
                raise ValueError(
                    f"source group '{name}' references out-of-range column {idx} "
                    f"(d_e={d_e})"
                )
            col_type[idx] = info.get("type", "raw")

    uncovered = sorted(set(range(d_e)) - set(col_type))
    if uncovered:
        raise ValueError(f"source groups do not cover columns: {uncovered}")

    groups: dict[str, dict[str, Any]] = {
        feature_names[i]: {"indices": [i], "type": col_type[i]}
        for i in range(d_e)
    }

    logger.info(
        f"Singleton feature groups: K={d_e} (from K={source['K']} semantic "
        f"groups), d_e={d_e}"
    )
    return {
        "d_e": d_e,
        "K": d_e,
        "feature_names": feature_names,
        "groups": groups,
        "singleton_source": {
            "d_e": d_e,
            "K": int(source["K"]),
            "note": (
                "Derived by build_singleton_feature_groups() for paper config "
                "(i): raw-feature (ungrouped) KernelSHAP. One player per "
                "encoded edge-feature dimension."
            ),
        },
    }
