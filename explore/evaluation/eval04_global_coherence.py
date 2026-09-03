"""
eval04_global_coherence.py — global coherence vs. the proxy ground truth
=========================================================================
Implements specs/64 Part I §4.4a + §4.4b + §4.6 / Part II §14.5 + §14.6.

Two independent sub-metrics, deliberately kept separable so one missing input
never suppresses the other:

  (a) Rank correlation. Per (dataset, class), the SHAP-GSD feature-group
      ranking by mean |phi_F| is correlated (Spearman) against a
      literature-derived expected ranking. The expected ranking comes from an
      owner-authored mapping table, never from code: translating free-text
      literature prose into named feature groups is a methodological claim the
      manuscript has to defend, so nothing here generates it. When the mapping
      file is absent this half is skipped with a WARNING naming the exact path
      that was probed, the structural half still runs, and the process exits 0.

  (b) Structural coherence. Per (dataset, class), a THREE-way comparison
      between SHAP-GSD's own explanatory subgraph topology, the literature's
      expected topology, and the already-measured Stage-0 topology. It is
      three-way on purpose: cases where SHAP-GSD agrees with the Stage-0
      measurement while both disagree with the literature are a reportable
      outcome, not something to collapse into a pass/fail score.

Source 3 (non-circularity gate). ``--source3-locked`` is a researcher
assertion that the proxy-GT sources 1/2/4 are frozen for the datasets being
checked. Without it, the two CSVs are still written and a WARNING records that
no Source-3 artifact was emitted. With it, a markdown fragment of per-class
prose cells is written for the owner to paste into the proxy-GT table. This
script never opens or edits that table: it is hand-authored prose maintained
outside this repo, and a generated in-place edit of it would not be reversible
by this repo's git.

Files read (read-only):
  <out-dir>/per_class_explanation_summary.csv      — eval05's output (the ranking)
  <out-dir>/proxy_gt_feature_expectation_mapping.csv — owner-authored (§4.4a)
  <out-dir>/proxy_gt_structural_expectation_mapping.csv — owner-authored, optional
  <run>/outputs/explanations/<Class>/*.json        — subgraph_edge_ids
  <run>/feature_store/test/edge_indices.npy        — EID -> row alignment
  <run>/feature_store/test/edges_meta.parquet      — src_ip/dst_ip per row
  <run>/graphs/node_id_map.json                    — ip -> integer node id
  <run>/outputs/topology/gateway_distance.json     — Stage-0 per-class structure

Files output (under ``outputs/figures/evaluation/``):
  global_coherence_rank_correlation.csv
  global_coherence_structural.csv
  global_coherence_summary_<dataset>.{pdf,png,txt}
  source3_consistency_cells.md   (only with --source3-locked)

Output-location note: ``OUT_DIR`` is the hardcoded
``REPO_ROOT/outputs/figures/evaluation/``, NOT resolved through
``explore._paths.paths()``/``SHAP_GSD_CONFIG`` (specs/64 D3): that variable
names one active run, whereas this script both reads a cross-dataset input CSV
and writes cross-dataset outputs that belong in no single run's subtree.
Restated here rather than relied on by reference (specs/48 §1.1 pt 4).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # explore/AGENT.md mandatory convention

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from explore.evaluation._discover import (  # noqa: E402
    DEFAULT_OUT_DIR,
    DatasetRun,
    DatasetResolutionError,
    add_common_args,
    feature_store_dir,
    graphs_dir,
    outputs_dir,
    resolve_datasets_from_args,
)
from explore.evaluation._load import as_float, load_eval_metrics  # noqa: E402
from explore.evaluation._stats import apply_family, rank_agreement  # noqa: E402
from explore.evaluation.eval05_per_class_explanations import (  # noqa: E402
    CSV_NAME as PER_CLASS_CSV_NAME,
    parse_magnitudes,
    parse_ranking,
)

log = logging.getLogger("eval04_global_coherence")

OUT_DIR: Path = DEFAULT_OUT_DIR  # see module docstring for why this is hardcoded

RANK_CSV_NAME = "global_coherence_rank_correlation.csv"
STRUCTURAL_CSV_NAME = "global_coherence_structural.csv"
SOURCE3_MD_NAME = "source3_consistency_cells.md"
FIGURE_STEM_PREFIX = "global_coherence_summary_"

# SHAP-GSD samples its 200 explained flows per class by ground-truth label,
# regardless of the model's prediction — so a low-recall class's explained
# set is dominated by attributions of wrong decisions, not the attack
# signature Source 3 is meant to check consistency against. The manuscript's
# own Discussion §VI-C5 established recall >= 0.49 as the bar for "the
# trained model detects this class" and used exactly that cutoff to isolate
# a clean subset for its own analysis; Source 3 reuses that same,
# already-precedented threshold rather than inventing a new one. A ">0%"
# filter is not sufficient: UNSW DoS clears it at ~17% recall while the
# manuscript's own Discussion attributes its rho = -0.367 (the most negative
# in the benchmark) to this exact contamination mechanism, not a genuine
# literature mismatch.
SOURCE3_RECALL_THRESHOLD = 0.49

MAPPING_CSV_NAME = "proxy_gt_feature_expectation_mapping.csv"
STRUCTURAL_EXPECTATION_CSV_NAME = "proxy_gt_structural_expectation_mapping.csv"

#: Mean distinct-endpoints-per-endpoint above which a subgraph (or a Stage-0
#: per-class node-count pair) reads as a fan pattern. Named, never inline, and
#: exposed as ``--fan-threshold``; the same constant scores both the SHAP-GSD
#: leg and the Stage-0 leg so the two are compared on one scale.
FAN_THRESHOLD = 3.0

#: Fraction of a class's top-K subgraph edges the modal (src, dst) pair must
#: account for before the subgraph reads as a persistent edge. Exposed as
#: ``--persistent-frac``.
PERSISTENT_FRAC = 0.5

#: Accepted values of the mapping table's ``mapper_confidence`` column.
VALID_CONFIDENCES: frozenset[str] = frozenset({"high", "medium", "low"})

#: Topology vocabulary shared by all three legs of the comparison.
TOPOLOGY_FAN_OUT = "fan_out"
TOPOLOGY_FAN_IN = "fan_in"
TOPOLOGY_PERSISTENT = "persistent_edge"
TOPOLOGY_OTHER = "other"
TOPOLOGY_INDETERMINATE = "indeterminate"

#: Stage-0 leg value used when a dataset's per-class node counts are identical
#: across its classes, so the measurement cannot discriminate between them.
STAGE0_INDETERMINATE_IDENTICAL = "indeterminate_identical"

#: Value written where the owner has supplied no literature expectation.
LITERATURE_NOT_SUPPLIED = "not_supplied"

#: ``three_way_agreement`` vocabulary. The first five are Part I §4.4b's; the
#: sixth is specs/64 §14.5.2's addition for a non-discriminative Stage-0 leg;
#: the seventh covers the state where no literature expectation was supplied at
#: all, which would otherwise be mis-scored as a genuine disagreement.
AGREEMENT_FULL = "full"
AGREEMENT_SHAP_STAGE0 = "shap_stage0_only"
AGREEMENT_SHAP_LIT = "shap_literature_only"
AGREEMENT_STAGE0_LIT = "stage0_literature_only"
AGREEMENT_NONE = "none"
AGREEMENT_STAGE0_INDETERMINATE = "stage0_indeterminate"
AGREEMENT_LITERATURE_MISSING = "literature_not_supplied"

RANK_CSV_COLUMNS: tuple[str, ...] = (
    "dataset",
    "class_name",
    "spearman_rho_vs_expected",
    "p_raw",
    "p_corrected",
    "correction_family",
    "n_shap_features_ranked",
    "n_expected_features_mapped",
    "mapping_confidence",
    "rho_undefined_reason",
    "n_flows_aggregated",
    "notes",
)

STRUCTURAL_CSV_COLUMNS: tuple[str, ...] = (
    "dataset",
    "class_name",
    "shap_subgraph_topology",
    "literature_expected_topology",
    "stage0_measured_topology",
    "three_way_agreement",
    "n_subgraph_edges",
    "n_flows_aggregated",
    "stage0_d_gw_mean",
    "stage0_diameter",
    "stage0_n_src_nodes",
    "stage0_n_dst_nodes",
    "notes",
)


# ---------------------------------------------------------------------------
# §4.4a — rank correlation against the owner-authored expectation mapping
# ---------------------------------------------------------------------------


def load_per_class_summary(path: Path) -> pd.DataFrame | None:
    """Load eval05's per-class explanation summary.

    Read rather than recomputed so that this module and any other consumer
    provably score the same per-class ranking vector (specs/64 §14.5.1).

    Args:
        path: Path to ``per_class_explanation_summary.csv``.

    Returns:
        The frame, or ``None`` when it has not been generated yet.
    """
    if not path.is_file():
        log.error(
            "per-class explanation summary not found at %s — run "
            "explore/evaluation/eval05_per_class_explanations.py first "
            "(with the same --out-dir)", path,
        )
        return None
    return pd.read_csv(path)


def load_mapping_csv(path: Path, observed_groups: dict[str, set[str]]) -> pd.DataFrame | None:
    """Load and validate the owner-authored feature-expectation mapping.

    Args:
        path: Path to ``proxy_gt_feature_expectation_mapping.csv``.
        observed_groups: ``{dataset_label: {group names observed on disk}}``,
            used to reject a mistyped group name. A typo there would silently
            zero a correlation rather than fail.

    Returns:
        The validated frame, or ``None`` when the file does not exist (a normal
        state — the mapping is a manual deliverable).

    Raises:
        ValueError: When a required column is missing, a ``mapper_confidence``
            value is not one of high/medium/low, or a
            ``shap_gsd_feature_group`` is absent from that dataset's observed
            feature groups.
    """
    if not path.is_file():
        log.warning(
            "proxy-GT feature-expectation mapping not found at %s; skipping the "
            "rank-correlation half of global coherence. That file is authored "
            "by hand (it encodes a literature claim the manuscript must "
            "defend) and is never generated by this script. The structural "
            "half below is unaffected.", path,
        )
        return None
    frame = pd.read_csv(path)
    required = {
        "dataset", "class_name", "shap_gsd_feature_group",
        "expected_rank_or_direction", "source_citation", "mapper_confidence",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"mapping CSV {path} is missing required column(s) {missing}; "
            f"expected at least {sorted(required)}"
        )
    bad_conf = sorted(
        set(frame["mapper_confidence"].astype(str).str.strip().str.lower())
        - VALID_CONFIDENCES
    )
    if bad_conf:
        raise ValueError(
            f"mapping CSV {path} carries mapper_confidence value(s) {bad_conf}; "
            f"expected one of {sorted(VALID_CONFIDENCES)}"
        )
    for dataset_label, group in frame.groupby("dataset"):
        known = observed_groups.get(str(dataset_label))
        if known is None:
            log.warning(
                "mapping CSV names dataset %r, which matches none of the "
                "dataset labels this run produced (%s), so ALL %d of its "
                "mapping rows are dropped UNVALIDATED and NO class of that "
                "dataset will be scored. The `dataset` column must carry the "
                "display label the eval scripts wrote (the --label value, or "
                "the resolved dataset key under --all-datasets), not a "
                "paper-facing name chosen independently.",
                dataset_label, sorted(observed_groups), len(group),
            )
            continue
        unknown = sorted(set(group["shap_gsd_feature_group"].astype(str)) - known)
        if unknown:
            raise ValueError(
                f"mapping CSV {path} names feature group(s) {unknown} for "
                f"dataset {dataset_label!r} that do not appear in that "
                f"dataset's own feature_group_names. A mistyped group name "
                f"silently zeroes a rank correlation, so this is a hard error."
            )
    return frame


def _expected_rank_vector(
    ranking: list[str], mapping_rows: pd.DataFrame,
) -> tuple[np.ndarray, int, str]:
    """Build the literature-expected rank vector aligned to ``ranking``.

    A group named by the mapping takes its ``expected_rank_or_direction`` when
    that value parses as a number, and otherwise its 1-based position among
    that class's mapping rows. Every group the mapping does not name is placed
    at rank ``len(ranking) + 1`` — the same "outside the expected profile"
    convention the existing per-class literature comparison in
    ``explore/arguments/`` uses.

    Args:
        ranking: The dataset/class's observed feature groups, best-first.
        mapping_rows: The mapping rows for this (dataset, class).

    Returns:
        ``(expected_ranks, n_mapped, note)``.
    """
    unmapped_rank = float(len(ranking) + 1)
    expected = {name: unmapped_rank for name in ranking}
    note = ""
    non_numeric = 0
    for position, (_, row) in enumerate(mapping_rows.iterrows(), start=1):
        group = str(row["shap_gsd_feature_group"])
        raw = row["expected_rank_or_direction"]
        try:
            rank = float(raw)
        except (TypeError, ValueError):
            rank = float(position)
            non_numeric += 1
        if group in expected:
            expected[group] = rank
    if non_numeric:
        note = (
            f"{non_numeric} expected_rank_or_direction value(s) were not "
            f"numeric and were ranked by their row order in the mapping table"
        )
    return (
        np.asarray([expected[name] for name in ranking], dtype=float),
        int(len(mapping_rows)),
        note,
    )


def compute_rank_correlation(
    per_class: pd.DataFrame, mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Score every mapped (dataset, class) against its expected feature profile.

    Args:
        per_class: eval05's per-class summary frame.
        mapping: The validated owner-authored mapping frame.

    Returns:
        A frame carrying exactly :data:`RANK_CSV_COLUMNS`, BH-corrected per
        dataset.
    """
    rows: list[dict[str, object]] = []
    for (dataset_label, class_name), group in mapping.groupby(
        ["dataset", "class_name"], sort=False,
    ):
        match = per_class[
            (per_class["dataset"] == dataset_label)
            & (per_class["class_name"] == class_name)
        ]
        confidence = "|".join(
            sorted(set(group["mapper_confidence"].astype(str).str.lower()))
        )
        if match.empty:
            rows.append({
                "dataset": dataset_label, "class_name": class_name,
                "spearman_rho_vs_expected": float("nan"), "p_raw": float("nan"),
                "n_shap_features_ranked": 0, "n_expected_features_mapped": len(group),
                "mapping_confidence": confidence,
                "rho_undefined_reason": "no per-class explanation row",
                "n_flows_aggregated": 0,
                "notes": "class absent from per_class_explanation_summary.csv",
            })
            continue
        record = match.iloc[0]
        ranking = parse_ranking(record["full_feature_ranking"])
        magnitudes = parse_magnitudes(record["full_feature_mean_magnitudes"])
        n_flows = int(record["n_flows_aggregated"])
        if not ranking:
            rows.append({
                "dataset": dataset_label, "class_name": class_name,
                "spearman_rho_vs_expected": float("nan"), "p_raw": float("nan"),
                "n_shap_features_ranked": 0, "n_expected_features_mapped": len(group),
                "mapping_confidence": confidence,
                "rho_undefined_reason": "class has no explained flows",
                "n_flows_aggregated": n_flows,
                "notes": "no feature ranking to correlate",
            })
            continue

        expected_ranks, n_mapped, note = _expected_rank_vector(ranking, group)
        # Correlate on the SHAP magnitudes directly against NEGATED expected
        # ranks, so that "expected rank 1" lines up with "largest magnitude":
        # Spearman is rank-based, and negating turns the ascending rank scale
        # into the same descending-importance direction the magnitudes use.
        agreement = rank_agreement(
            np.asarray(magnitudes, dtype=float), -expected_ranks, len(ranking),
        )
        reason = agreement.note
        if reason and "constant" in reason:
            reason = (
                "expected-rank vector is constant: none of the mapped feature "
                "groups appears in this class's observed ranking, so rho is "
                "undefined (NOT rho = 0)"
            )
        rows.append({
            "dataset": dataset_label, "class_name": class_name,
            "spearman_rho_vs_expected": agreement.spearman_rho,
            "p_raw": agreement.p_raw,
            "n_shap_features_ranked": len(ranking),
            "n_expected_features_mapped": n_mapped,
            "mapping_confidence": confidence,
            "rho_undefined_reason": reason,
            "n_flows_aggregated": n_flows,
            "notes": note,
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=list(RANK_CSV_COLUMNS))
    corrected = [
        apply_family(part, f"dataset={label};module=eval04_rank_correlation")
        for label, part in frame.groupby("dataset", sort=False)
    ]
    return pd.concat(corrected, ignore_index=True)[list(RANK_CSV_COLUMNS)]


# ---------------------------------------------------------------------------
# §4.4b — structural coherence
# ---------------------------------------------------------------------------


def load_edge_endpoints(run: DatasetRun) -> tuple[np.ndarray, pd.DataFrame, dict[str, int]]:
    """Load the test split's EID -> (src node, dst node) mapping.

    The per-flow explanation records carry ``subgraph_edge_ids`` — global edge
    ids, not endpoints — so endpoint resolution has to come from the feature
    store, whose ``edges_meta.parquet`` rows are aligned to
    ``edge_indices.npy`` by the pipeline's EID-alignment invariant. IPs are
    mapped through the graph's own ``node_id_map.json`` where possible.

    Args:
        run: The resolved dataset run.

    Returns:
        ``(edge_indices, edges_meta, node_id_map)`` — the test split's ascending
        global edge ids, the row-aligned ``src_ip``/``dst_ip`` frame, and the
        ``{ip: integer node id}`` map (``{}`` when the graph artifact is
        absent). Pass the first two to :func:`resolve_edge_rows` and
        :func:`endpoints_for_rows` rather than indexing them by hand.

    Raises:
        FileNotFoundError: When either feature-store file is absent.
    """
    fs_test = feature_store_dir(run) / "test"
    eid_path = fs_test / "edge_indices.npy"
    meta_path = fs_test / "edges_meta.parquet"
    if not eid_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"{run.name}: structural coherence needs both {eid_path} and "
            f"{meta_path}"
        )
    edge_indices = np.load(eid_path, mmap_mode="r")
    meta = pd.read_parquet(meta_path, columns=["src_ip", "dst_ip"])
    # CLAUDE.md CRITICAL INVARIANT 1 — EID alignment, asserted in production.
    assert len(meta) == edge_indices.shape[0], (
        f"{run.name}: edges_meta.parquet has {len(meta)} rows but "
        f"edge_indices.npy has {edge_indices.shape[0]} — the EID-alignment "
        f"invariant is broken for {meta_path}"
    )
    node_map_path = graphs_dir(run) / "node_id_map.json"
    node_map: dict[str, int] = {}
    if node_map_path.is_file():
        node_map = json.loads(node_map_path.read_text())
    else:
        log.warning(
            "%s: no node_id_map.json at %s; endpoints stay as raw addresses",
            run.name, node_map_path,
        )
    return np.asarray(edge_indices), meta, node_map


def endpoints_for_rows(
    meta: pd.DataFrame, node_map: dict[str, int], rows: list[int],
) -> list[tuple[object, object]]:
    """Resolve only the requested feature-store rows to node-id endpoints.

    Only the rows a class's subgraphs actually name are materialised: the test
    split runs to millions of edges, and turning both address columns into
    Python objects for all of them would be gratuitous (CLAUDE.md CODING
    STANDARDS 8).

    Args:
        meta: The test split's ``edges_meta.parquet`` frame.
        node_map: ``{ip: integer node id}``; an unmapped address falls back to
            the address string itself.
        rows: Feature-store row indices.

    Returns:
        One ``(src_node, dst_node)`` pair per requested row, in input order.
    """
    if not rows:
        return []
    subset = meta.iloc[rows]
    return [
        (node_map.get(str(src), str(src)), node_map.get(str(dst), str(dst)))
        for src, dst in zip(subset["src_ip"], subset["dst_ip"])
    ]


def resolve_edge_rows(edge_indices: np.ndarray, edge_ids: list[int], run_name: str) -> list[int]:
    """Map global edge ids to their feature-store rows.

    Args:
        edge_indices: The test split's ``edge_indices.npy``, ascending.
        edge_ids: Global edge ids from an explanation record's subgraph.
        run_name: Dataset name, for the error message.

    Returns:
        One row index per edge id, in input order.

    Raises:
        KeyError: When an edge id is not in the test split. That would mean the
            EID-alignment invariant is broken, so it is a hard error rather
            than a skip.
    """
    rows: list[int] = []
    for eid in edge_ids:
        idx = int(np.searchsorted(edge_indices, int(eid)))
        if idx >= edge_indices.shape[0] or int(edge_indices[idx]) != int(eid):
            raise KeyError(
                f"{run_name}: subgraph edge id {eid} is not present in the test "
                f"split's edge_indices.npy; the EID-alignment invariant "
                f"(CLAUDE.md CRITICAL INVARIANT 1) does not hold for this run"
            )
        rows.append(idx)
    return rows


def classify_subgraph_topology(
    pairs: list[tuple[object, object]],
    fan_threshold: float = FAN_THRESHOLD,
    persistent_frac: float = PERSISTENT_FRAC,
) -> tuple[str, str]:
    """Classify an aggregated set of subgraph edges as a topology.

    Args:
        pairs: ``(src_node, dst_node)`` for every top-K subgraph edge of every
            flow in one class.
        fan_threshold: Mean distinct-endpoints-per-endpoint above which a fan
            pattern is declared.
        persistent_frac: Fraction of edges the modal pair must cover.

    Returns:
        ``(topology, evidence)`` — one of the topology constants plus a short
        human-readable justification for the ``notes`` column.
    """
    if not pairs:
        return TOPOLOGY_INDETERMINATE, "no subgraph edges"
    dst_per_src: dict[object, set[object]] = {}
    src_per_dst: dict[object, set[object]] = {}
    for src, dst in pairs:
        dst_per_src.setdefault(src, set()).add(dst)
        src_per_dst.setdefault(dst, set()).add(src)
    mean_out = float(np.mean([len(v) for v in dst_per_src.values()]))
    mean_in = float(np.mean([len(v) for v in src_per_dst.values()]))
    modal_pair, modal_count = Counter(pairs).most_common(1)[0]
    modal_share = modal_count / len(pairs)
    evidence = (
        f"mean distinct dst/src={mean_out:.2f}, src/dst={mean_in:.2f}, "
        f"modal-pair share={modal_share:.2f} over {len(pairs)} subgraph edges"
    )
    if mean_out >= fan_threshold and mean_out > mean_in:
        return TOPOLOGY_FAN_OUT, evidence
    if mean_in >= fan_threshold and mean_in > mean_out:
        return TOPOLOGY_FAN_IN, evidence
    if modal_share >= persistent_frac:
        return TOPOLOGY_PERSISTENT, evidence
    return TOPOLOGY_OTHER, evidence


def load_stage0(run: DatasetRun) -> dict[str, dict[str, object]]:
    """Load the Stage-0 per-class structural measurements.

    Args:
        run: The resolved dataset run.

    Returns:
        ``{class_name: {d_gw_mean, diameter, n_src_nodes, n_dst_nodes}}``;
        empty when the topology artifact is absent.
    """
    path = outputs_dir(run) / "topology" / "gateway_distance.json"
    if not path.is_file():
        log.warning("%s: no Stage-0 topology artifact at %s", run.name, path)
        return {}
    raw = json.loads(path.read_text())
    per_class = (raw.get("d_gw", {}) or {}).get("per_class", {}) or {}
    diameters = (raw.get("diameter", {}) or {}).get("per_class", {}) or {}
    out: dict[str, dict[str, object]] = {}
    for class_name, entry in per_class.items():
        out[class_name] = {
            # d_gw.mean is legitimately null on at least one resolvable
            # dataset, so it goes through as_float rather than float().
            "d_gw_mean": as_float(entry.get("mean")),
            "n_src_nodes": as_float(entry.get("n_src_nodes")),
            "n_dst_nodes": as_float(entry.get("n_dst_nodes")),
            "diameter": as_float((diameters.get(class_name, {}) or {}).get("diameter")),
        }
    return out


def classify_stage0_topology(
    entry: dict[str, object] | None,
    all_entries: dict[str, dict[str, object]],
    fan_threshold: float = FAN_THRESHOLD,
) -> tuple[str, str]:
    """Classify one class's Stage-0 measurement as a topology.

    Keyed on the per-class ``(n_src_nodes, n_dst_nodes)`` pair rather than on
    ``d_gw``: the gateway distance is near-degenerate (mean = max = 1.0 for
    almost every class) on more than one dataset, while the node counts
    separate a many-to-one concentration from a one-to-one pair unambiguously.
    The same ``fan_threshold`` as the SHAP-GSD leg is used, so the two legs are
    compared on one scale.

    Args:
        entry: This class's Stage-0 record, or ``None``.
        all_entries: Every class's Stage-0 record for this dataset, used to
            detect a dataset whose classes are structurally identical and whose
            measurement therefore cannot discriminate.
        fan_threshold: Ratio above which a fan pattern is declared.

    Returns:
        ``(topology, evidence)``.
    """
    if entry is None:
        return TOPOLOGY_INDETERMINATE, "no Stage-0 record for this class"
    n_src = float(entry.get("n_src_nodes", float("nan")))
    n_dst = float(entry.get("n_dst_nodes", float("nan")))
    if not (np.isfinite(n_src) and np.isfinite(n_dst)) or n_src <= 0 or n_dst <= 0:
        return TOPOLOGY_INDETERMINATE, "Stage-0 node counts missing or zero"

    distinct_pairs = {
        (e.get("n_src_nodes"), e.get("n_dst_nodes")) for e in all_entries.values()
    }
    if len(distinct_pairs) <= 1 and len(all_entries) > 1:
        return STAGE0_INDETERMINATE_IDENTICAL, (
            f"all {len(all_entries)} classes share the same "
            f"(n_src_nodes, n_dst_nodes) = ({n_src:.0f}, {n_dst:.0f}), so the "
            f"Stage-0 measurement cannot discriminate between them"
        )

    evidence = f"Stage-0 n_src_nodes={n_src:.0f}, n_dst_nodes={n_dst:.0f}"
    if n_src / n_dst >= fan_threshold:
        return TOPOLOGY_FAN_IN, evidence
    if n_dst / n_src >= fan_threshold:
        return TOPOLOGY_FAN_OUT, evidence
    if n_src == 1 and n_dst == 1:
        return TOPOLOGY_PERSISTENT, evidence
    return TOPOLOGY_OTHER, evidence


def three_way_agreement(shap: str, literature: str, stage0: str) -> str:
    """Categorise the three-way structural comparison.

    Args:
        shap: SHAP-GSD's own subgraph topology.
        literature: The literature's expected topology.
        stage0: The Stage-0 measured topology.

    Returns:
        One of the ``AGREEMENT_*`` constants.
    """
    if literature == LITERATURE_NOT_SUPPLIED:
        return AGREEMENT_LITERATURE_MISSING
    if stage0 in (TOPOLOGY_INDETERMINATE, STAGE0_INDETERMINATE_IDENTICAL):
        return AGREEMENT_STAGE0_INDETERMINATE
    shap_stage0 = shap == stage0
    shap_lit = shap == literature
    stage0_lit = stage0 == literature
    if shap_stage0 and shap_lit:
        return AGREEMENT_FULL
    if shap_stage0:
        return AGREEMENT_SHAP_STAGE0
    if shap_lit:
        return AGREEMENT_SHAP_LIT
    if stage0_lit:
        return AGREEMENT_STAGE0_LIT
    return AGREEMENT_NONE


def load_structural_expectations(path: Path) -> dict[tuple[str, str], str]:
    """Load the optional owner-authored literature structural expectations.

    Part I §4.4b's literature leg comes from the proxy-GT table's "Expected
    structural signals" column, which — like the feature-expectation mapping —
    is hand-authored prose, not something this script may generate. It is read
    from an optional CSV so that the leg can be populated once the owner
    authors it, and is honestly reported as not supplied until then.

    Args:
        path: Path to ``proxy_gt_structural_expectation_mapping.csv`` with
            columns ``dataset, class_name, literature_expected_topology``.

    Returns:
        ``{(dataset, class_name): topology}``; empty when the file is absent.
    """
    if not path.is_file():
        log.warning(
            "no literature structural-expectation table at %s; the literature "
            "leg of the three-way comparison is reported as %r rather than "
            "guessed", path, LITERATURE_NOT_SUPPLIED,
        )
        return {}
    frame = pd.read_csv(path)
    required = {"dataset", "class_name", "literature_expected_topology"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"structural-expectation CSV {path} is missing column(s) {missing}"
        )
    return {
        (str(row["dataset"]), str(row["class_name"])):
            str(row["literature_expected_topology"])
        for _, row in frame.iterrows()
    }


def compute_structural_coherence(
    runs: list[DatasetRun],
    per_class: pd.DataFrame,
    expectations: dict[tuple[str, str], str],
    fan_threshold: float = FAN_THRESHOLD,
    persistent_frac: float = PERSISTENT_FRAC,
) -> pd.DataFrame:
    """Score the three-way structural comparison for every (dataset, class).

    Args:
        runs: The resolved dataset runs.
        per_class: eval05's per-class summary (for ``n_flows_aggregated``).
        expectations: Literature expectations, possibly empty.
        fan_threshold: Fan-pattern threshold, shared by both measured legs.
        persistent_frac: Persistent-edge threshold.

    Returns:
        A frame carrying exactly :data:`STRUCTURAL_CSV_COLUMNS`.
    """
    from explore.evaluation.eval05_per_class_explanations import iter_class_records

    rows: list[dict[str, object]] = []
    for run in runs:
        metrics = load_eval_metrics(run)
        stage0 = load_stage0(run)
        edge_indices, meta, node_map = load_edge_endpoints(run)
        for class_name in metrics.class_names:
            records = iter_class_records(run, class_name)
            rows_needed: list[int] = []
            for rec in records:
                edge_ids = [int(e) for e in (rec.get("subgraph_edge_ids") or [])]
                rows_needed.extend(
                    resolve_edge_rows(edge_indices, edge_ids, run.name)
                )
            pairs = endpoints_for_rows(meta, node_map, rows_needed)

            shap_topology, shap_evidence = classify_subgraph_topology(
                pairs, fan_threshold, persistent_frac,
            )
            stage0_entry = stage0.get(class_name)
            stage0_topology, stage0_evidence = classify_stage0_topology(
                stage0_entry, stage0, fan_threshold,
            )
            literature = expectations.get(
                (run.label, class_name), LITERATURE_NOT_SUPPLIED,
            )
            per_class_match = per_class[
                (per_class["dataset"] == run.label)
                & (per_class["class_name"] == class_name)
            ]
            n_flows = (
                int(per_class_match.iloc[0]["n_flows_aggregated"])
                if not per_class_match.empty else len(records)
            )
            rows.append({
                "dataset": run.label,
                "class_name": class_name,
                "shap_subgraph_topology": shap_topology,
                "literature_expected_topology": literature,
                "stage0_measured_topology": stage0_topology,
                "three_way_agreement": three_way_agreement(
                    shap_topology, literature, stage0_topology,
                ),
                "n_subgraph_edges": len(pairs),
                "n_flows_aggregated": n_flows,
                "stage0_d_gw_mean": (
                    stage0_entry.get("d_gw_mean") if stage0_entry else float("nan")
                ),
                "stage0_diameter": (
                    stage0_entry.get("diameter") if stage0_entry else float("nan")
                ),
                "stage0_n_src_nodes": (
                    stage0_entry.get("n_src_nodes") if stage0_entry else float("nan")
                ),
                "stage0_n_dst_nodes": (
                    stage0_entry.get("n_dst_nodes") if stage0_entry else float("nan")
                ),
                "notes": f"SHAP-GSD leg: {shap_evidence}. Stage-0 leg: {stage0_evidence}.",
            })
        log.info("%s: scored %d classes for structural coherence",
                 run.label, metrics.n_classes_total)
    return pd.DataFrame(rows, columns=list(STRUCTURAL_CSV_COLUMNS))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

LABEL_FS = 9              # explore/AGENT.md §6 style constant
_SHAP_GSD = "#2a9d8f"     # teal  — a defined rank correlation
_ABSENCE = "#e76f51"      # coral — a negative rank correlation
_GRAY = "#888888"         # reference lines, and the "undefined" marker
# _PRESENCE/_GNN_EXP intentionally not used: this figure encodes rank
# correlation against a literature expectation and a categorical structural
# agreement, neither of which is a presence/absence-of-attribution axis or an
# explainer comparison (specs/47 §4 — do not force an unrelated constant in).


def write_dataset_figure(
    dataset_label: str,
    rank_rows: pd.DataFrame,
    structural_rows: pd.DataFrame,
    out_dir: Path,
    mapping_available: bool,
) -> None:
    """Write one dataset's global-coherence figure trio.

    Args:
        dataset_label: The dataset's display label.
        rank_rows: That dataset's rank-correlation rows (possibly empty).
        structural_rows: That dataset's structural rows.
        out_dir: Output directory.
        mapping_available: Whether the owner-authored mapping was loaded.
    """
    stem = f"{FIGURE_STEM_PREFIX}{dataset_label}"
    classes = list(structural_rows["class_name"])
    rho_by_class = {
        str(r["class_name"]): float(r["spearman_rho_vs_expected"])
        for _, r in rank_rows.iterrows()
    }
    values = [rho_by_class.get(c, float("nan")) for c in classes]
    y_positions = np.arange(len(classes))

    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.42 * len(classes) + 1.6)))
    for y, value in zip(y_positions, values):
        if not np.isfinite(value):
            # Drawn as a centred marker rather than a full-length bar: a
            # hatched bar spanning 0..1 would read as rho = 1.
            ax.text(
                0.0, y, "not scored", fontsize=LABEL_FS - 2, va="center",
                ha="center", color=_GRAY, style="italic",
            )
        else:
            ax.barh(
                y, value, height=0.6,
                color=_SHAP_GSD if value >= 0 else _ABSENCE,
            )
    ax.axvline(0.0, color=_GRAY, linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(classes, fontsize=LABEL_FS)
    ax.set_xlabel(
        "Spearman rho vs. literature-expected feature profile",
        fontsize=LABEL_FS,
    )
    ax.set_xlim(-1.05, 1.05)
    # Explicit, so the first and last rows are not flush against the axes
    # frame when no bar artist is present to give matplotlib a y-extent.
    ax.set_ylim(len(classes) - 0.4, -0.6)
    # Suppress the two categories that never carry a real three-way finding:
    # AGREEMENT_LITERATURE_MISSING means no literature leg was ever supplied
    # for this class (no comparison was possible), and AGREEMENT_NONE mostly
    # fires trivially when SHAP-GSD's own leg is "indeterminate" for lack of
    # explained flows, not from a genuine three-leg disagreement. Printing
    # either at the row's right edge reads as a finding when it is really an
    # absence of one; the remaining categories (full, stage0_indeterminate,
    # shap_stage0_only, shap_literature_only, stage0_literature_only) all
    # reflect an actual comparison outcome and stay labelled — EXCEPT that
    # stage0_indeterminate and stage0_literature_only can also fire when
    # SHAP-GSD's own leg is "indeterminate" (zero explained flows): in that
    # case the label is purely a literature-vs-Stage-0 comparison that never
    # involved SHAP-GSD at all, and printing it next to a "not scored" bar
    # reads as an explainer finding that does not exist. Suppress any label
    # whenever SHAP-GSD's own topology is indeterminate, regardless of which
    # category the comparison landed in.
    _SUPPRESSED_AGREEMENT_LABELS = {AGREEMENT_LITERATURE_MISSING, AGREEMENT_NONE}
    for y, (_, row) in zip(y_positions, structural_rows.iterrows()):
        agreement = str(row["three_way_agreement"])
        if agreement in _SUPPRESSED_AGREEMENT_LABELS:
            continue
        if str(row["shap_subgraph_topology"]) == TOPOLOGY_INDETERMINATE:
            continue
        ax.text(
            1.10, y, agreement,
            fontsize=LABEL_FS - 2, va="center", ha="left", color=_GRAY,
            clip_on=False,
        )
    ax.set_title(f"{dataset_label} — global coherence", fontsize=LABEL_FS + 1)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    agreement_counts = structural_rows["three_way_agreement"].value_counts().to_dict()
    shap_counts = structural_rows["shap_subgraph_topology"].value_counts().to_dict()
    scored = [v for v in values if np.isfinite(v)]
    findings = [
        f"  Classes plotted: {len(classes)}.",
        f"  Rank correlations actually scored: {len(scored)} "
        f"({'mapping table present' if mapping_available else 'mapping table absent — none scored'}).",
    ]
    if scored:
        findings.append(
            f"  Scored rho ranges {min(scored):.3f} to {max(scored):.3f}, "
            f"mean {float(np.mean(scored)):.3f}."
        )
    findings.append(f"  SHAP-GSD subgraph topologies: {shap_counts}.")
    findings.append(f"  Three-way agreement categories: {agreement_counts}.")

    reasoning = f"""Figure reasoning — {stem}
{'=' * (len(stem) + 20)}

WHAT THE FIGURE SHOWS
---------------------
One horizontal bar per class of {dataset_label}. Bar length is the Spearman
rank correlation between that class's SHAP-GSD feature-group importance
ranking (mean |phi_F| over its explained flows) and the literature-derived
expected feature profile from the owner-authored proxy-GT mapping table.
A row reading "not scored" at the zero line means the correlation was NOT
scored — either the mapping table does not cover that class or the class has
no explained flows. The row is kept rather than dropped, and is marked with
words rather than a bar, so that a missing score can be neither overlooked nor
mistaken for a zero (or a perfect) one. A grey label at the right of a row, when
present, is that class's three-way structural-agreement category, comparing
SHAP-GSD's own explanatory subgraph topology, the literature expectation, and
the Stage-0 measured topology. The label is omitted for the two categories
that never carry a real three-way finding: "literature_not_supplied" (no
literature expectation exists for this class, so no comparison was possible)
and "none" (which mostly fires trivially when SHAP-GSD's own leg is
"indeterminate" from zero explained flows, not from a genuine three-way
disagreement). A row with no label at its right means one of those two, not
that the comparison was skipped.

KEY FINDINGS
------------
{chr(10).join(findings)}

PAPER FRAMING
-------------
This is the global-coherence dimension: whether an explanation agrees with
what the intrusion-detection literature says the attack class should look
like, and whether it agrees with what the capture's own topology measurement
independently shows. The three-way framing is the point — a class where the
explanation matches the measured topology while both disagree with the
literature expectation is a substantive finding about the dataset, not a
failure of the explainer, and the category labels preserve that distinction
instead of collapsing it into a single score.

SUGGESTED FIGURE CAPTION
------------------------
Per-class global coherence for {dataset_label}: Spearman rank correlation
between SHAP-GSD's feature-group importance ranking and the literature-derived
expected profile. Where present, the right-hand label names which two of three
independently-derived topologies agreed — SHAP-GSD's own explanatory subgraph,
the literature-expected topology, and the Stage-0 topology measured directly
from the capture — e.g. "shap_stage0_only" means SHAP-GSD's subgraph matched
the independent Stage-0 measurement while both disagreed with the literature
expectation, and "full" means all three agreed. A label is omitted, not
merely left off, whenever it would not carry a finding that actually involves
SHAP-GSD's own explanation: when no literature expectation exists for the
class ("literature_not_supplied"), when none of the three pairwise
comparisons coincided ("none"), or when SHAP-GSD has no explained flows for
the class at all, in which case any apparent literature-vs-Stage-0 agreement
is unrelated to the explainer and would misleadingly read as one of its
findings. Classes for which no rank correlation was scored carry no bar and
are marked "not scored" at the zero line; a class with zero explained flows
is always both "not scored" and unlabelled, though the two omissions are not
otherwise the same condition — a class can be "not scored" purely because the
literature mapping omits it while still keeping a labelled subgraph, if it has
explained flows of its own.
"""
    (out_dir / f"{stem}.txt").write_text(reasoning)
    log.info("wrote %s.{pdf,png,txt}", out_dir / stem)


# ---------------------------------------------------------------------------
# §4.6 — Source-3 consistency cells
# ---------------------------------------------------------------------------


def write_source3_cells(
    rank_frame: pd.DataFrame,
    structural_frame: pd.DataFrame,
    runs: list[DatasetRun],
    path: Path,
) -> None:
    """Render the Source-3 prose cells for the owner to paste into the proxy-GT table.

    Args:
        rank_frame: The rank-correlation frame (possibly empty).
        structural_frame: The structural-coherence frame.
        runs: The resolved dataset runs, for per-class recall lookup — a
            class's explained flows are sampled by ground-truth label
            regardless of prediction, so a low-recall class's 200 explained
            flows are mostly attributions of wrong decisions. See
            ``SOURCE3_RECALL_THRESHOLD``.
        path: Destination ``.md`` path.
    """
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# Source 3 — SHAP-GSD consistency cells",
        "",
        f"Generated {stamp} by `explore/evaluation/eval04_global_coherence.py "
        "--source3-locked`.",
        "",
        "`--source3-locked` is a **researcher assertion** that proxy-GT sources "
        "1/2/4 were frozen for these datasets before this run — it is "
        "deliberately not auto-detectable, and it is recorded here with the "
        "invoking timestamp so a later reader can see who asserted the "
        "non-circularity precondition and when.",
        "",
        "Each cell below is the content of one proxy-GT table row's "
        "`Source 3 (SHAP-GSD consistency)` column. This script never opens or "
        "edits that table; paste the cells in by hand.",
        "",
        f"A class whose test recall is below {SOURCE3_RECALL_THRESHOLD} (the "
        "manuscript's own Discussion §VI-C5 detection-threshold "
        "precedent) gets no consistency claim: its 200 explained flows are "
        "sampled by ground-truth label regardless of prediction, so most of "
        "them would be attributions of a wrong decision, not the attack "
        "signature being checked.",
        "",
        "| dataset | class_name | Source 3 (SHAP-GSD consistency) |",
        "|---|---|---|",
    ]
    rho_lookup = {
        (str(r["dataset"]), str(r["class_name"])): r
        for _, r in rank_frame.iterrows()
    } if not rank_frame.empty else {}
    recall_lookup: dict[tuple[str, str], float] = {}
    for run in runs:
        metrics = load_eval_metrics(run)
        for class_name, class_metrics in metrics.per_class.items():
            recall = class_metrics.get("recall")
            if recall is not None:
                recall_lookup[(run.label, str(class_name))] = float(recall)

    for _, row in structural_frame.iterrows():
        key = (str(row["dataset"]), str(row["class_name"]))
        recall = recall_lookup.get(key)
        if recall is not None and recall < SOURCE3_RECALL_THRESHOLD:
            cell = (
                f"No consistency claim made: test recall for this class is "
                f"{recall:.3f}, below the {SOURCE3_RECALL_THRESHOLD} "
                "detection-threshold (Discussion §VI-C5). Roughly "
                f"{(1 - recall) * 100:.0f}% of its 200 explained flows are "
                "attributions of a misclassified instance, which would "
                "contaminate any rank-correlation or structural claim rather "
                "than test it."
            )
            lines.append(f"| {row['dataset']} | {row['class_name']} | {cell} |")
            continue
        rank_row = rho_lookup.get(key)
        if rank_row is not None and np.isfinite(float(rank_row["spearman_rho_vs_expected"])):
            rho_text = (
                f"rank correlation vs. expected feature profile "
                f"rho = {float(rank_row['spearman_rho_vs_expected']):.3f} "
                f"(corrected p = {float(rank_row['p_corrected']):.3g}, mapping "
                f"confidence {rank_row['mapping_confidence']})"
            )
        else:
            rho_text = "rank correlation not scored (no usable expectation mapping)"
        if int(row["n_flows_aggregated"]) == 0:
            interpretation = (
                "No explained flows exist for this class, so no consistency "
                "claim can be made either way."
            )
        elif row["three_way_agreement"] == AGREEMENT_FULL:
            interpretation = (
                "SHAP-GSD's explanatory subgraph, the literature expectation "
                "and the Stage-0 measurement all agree."
            )
        elif row["three_way_agreement"] == AGREEMENT_SHAP_STAGE0:
            interpretation = (
                "SHAP-GSD agrees with the capture's own measured topology "
                "while both diverge from the literature expectation."
            )
        elif row["three_way_agreement"] == AGREEMENT_STAGE0_INDETERMINATE:
            interpretation = (
                "The Stage-0 measurement cannot discriminate this class, so "
                "the structural leg rests on the SHAP-GSD/literature pair alone."
            )
        elif row["three_way_agreement"] == AGREEMENT_LITERATURE_MISSING:
            interpretation = (
                "No literature structural expectation has been authored for "
                "this class yet, so only the SHAP-GSD and Stage-0 legs are "
                "reported."
            )
        else:
            interpretation = (
                "The three legs disagree; see "
                "`global_coherence_structural.csv` for each leg's evidence."
            )
        cell = (
            f"{rho_text}; structural: SHAP-GSD `{row['shap_subgraph_topology']}` "
            f"vs. literature `{row['literature_expected_topology']}` vs. "
            f"Stage-0 `{row['stage0_measured_topology']}` "
            f"(`{row['three_way_agreement']}`). {interpretation}"
        )
        lines.append(f"| {row['dataset']} | {row['class_name']} | {cell} |")
    lines.append("")
    path.write_text("\n".join(lines))
    log.info("wrote %s", path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(
    runs: list[DatasetRun],
    out_dir: Path,
    mapping_csv: Path,
    structural_csv: Path,
    *,
    source3_locked: bool = False,
    fan_threshold: float = FAN_THRESHOLD,
    persistent_frac: float = PERSISTENT_FRAC,
    per_class_csv: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and write every eval04 artifact.

    Args:
        runs: The resolved dataset runs.
        out_dir: Output directory.
        mapping_csv: Owner-authored feature-expectation mapping.
        structural_csv: Owner-authored literature structural expectations.
        source3_locked: The researcher's non-circularity assertion.
        fan_threshold: Fan-pattern threshold.
        persistent_frac: Persistent-edge threshold.
        per_class_csv: eval05's summary; defaults to ``out_dir``'s copy.

    Returns:
        ``(rank_frame, structural_frame)``. ``rank_frame`` is empty when the
        mapping table is absent.

    Raises:
        FileNotFoundError: When eval05's per-class summary has not been
            generated yet — this module reads that ranking rather than
            recomputing it, so that both provably score the same vector.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    per_class_path = per_class_csv or (out_dir / PER_CLASS_CSV_NAME)
    per_class = load_per_class_summary(per_class_path)
    if per_class is None:
        raise FileNotFoundError(
            f"eval04 needs {per_class_path}; run "
            f"explore/evaluation/eval05_per_class_explanations.py first"
        )

    # Union of every feature group observed per dataset, used to validate the
    # owner-authored mapping table's group names.
    observed_groups: dict[str, set[str]] = {}
    for _, row in per_class.iterrows():
        observed_groups.setdefault(str(row["dataset"]), set()).update(
            parse_ranking(row["full_feature_ranking"])
        )

    mapping = load_mapping_csv(mapping_csv, observed_groups)
    if mapping is None:
        rank_frame = pd.DataFrame(columns=list(RANK_CSV_COLUMNS))
    else:
        rank_frame = compute_rank_correlation(per_class, mapping)
    rank_frame.to_csv(out_dir / RANK_CSV_NAME, index=False)
    log.info("wrote %s (%d rows)", out_dir / RANK_CSV_NAME, len(rank_frame))

    expectations = load_structural_expectations(structural_csv)
    structural_frame = compute_structural_coherence(
        runs, per_class, expectations, fan_threshold, persistent_frac,
    )
    structural_frame.to_csv(out_dir / STRUCTURAL_CSV_NAME, index=False)
    log.info("wrote %s (%d rows)", out_dir / STRUCTURAL_CSV_NAME,
             len(structural_frame))

    for run_obj in runs:
        rows = structural_frame[structural_frame["dataset"] == run_obj.label]
        rank_rows = (
            rank_frame[rank_frame["dataset"] == run_obj.label]
            if not rank_frame.empty else pd.DataFrame(columns=list(RANK_CSV_COLUMNS))
        )
        write_dataset_figure(
            run_obj.label, rank_rows, rows, out_dir, mapping is not None,
        )

    if source3_locked:
        write_source3_cells(rank_frame, structural_frame, runs, out_dir / SOURCE3_MD_NAME)
    else:
        log.warning(
            "Source 3 was NOT emitted: --source3-locked was not passed, so the "
            "sources-1/2/4-frozen (non-circularity) precondition has not been "
            "asserted for these datasets. The two coherence CSVs above are "
            "unaffected.",
        )
    return rank_frame, structural_frame


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector, or ``None`` to read ``sys.argv``.

    Returns:
        Process exit code (0 on success — including when the owner-authored
        mapping table is absent — 2 on a dataset-resolution failure, 3 when
        eval05's per-class summary has not been generated yet).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Score global coherence: SHAP-GSD's per-class feature ranking "
            "against a literature-derived expectation, and its explanatory "
            "subgraph topology against both the literature expectation and the "
            "Stage-0 measurement (specs/64 Part I §4.4, §4.6)."
        )
    )
    add_common_args(parser)
    parser.add_argument(
        "--mapping-csv", type=Path, default=None,
        help=f"Owner-authored feature-expectation mapping (default: "
             f"<out-dir>/{MAPPING_CSV_NAME}). Absent is a normal state: the "
             f"rank-correlation half is skipped with a warning.",
    )
    parser.add_argument(
        "--structural-expectations-csv", type=Path, default=None,
        help=f"Owner-authored literature structural expectations (default: "
             f"<out-dir>/{STRUCTURAL_EXPECTATION_CSV_NAME}). Absent means the "
             f"literature leg is reported as not supplied rather than guessed.",
    )
    parser.add_argument(
        "--per-class-csv", type=Path, default=None,
        help=f"eval05's per-class explanation summary (default: "
             f"<out-dir>/{PER_CLASS_CSV_NAME}).",
    )
    parser.add_argument(
        "--source3-locked", action="store_true",
        help="Researcher assertion that proxy-GT sources 1/2/4 are frozen for "
             "these datasets. Only with this flag is the Source-3 consistency "
             "markdown fragment written.",
    )
    parser.add_argument(
        "--fan-threshold", type=float, default=FAN_THRESHOLD,
        help=f"Mean distinct-endpoints-per-endpoint above which a topology "
             f"reads as a fan pattern (default: {FAN_THRESHOLD}).",
    )
    parser.add_argument(
        "--persistent-frac", type=float, default=PERSISTENT_FRAC,
        help=f"Share of a class's subgraph edges the modal (src, dst) pair must "
             f"cover for a persistent-edge classification (default: "
             f"{PERSISTENT_FRAC}).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        runs = resolve_datasets_from_args(args)
    except DatasetResolutionError as exc:
        log.error("%s", exc)
        return 2

    out_dir = Path(args.out_dir)
    mapping_csv = Path(args.mapping_csv) if args.mapping_csv else out_dir / MAPPING_CSV_NAME
    structural_csv = (
        Path(args.structural_expectations_csv) if args.structural_expectations_csv
        else out_dir / STRUCTURAL_EXPECTATION_CSV_NAME
    )
    try:
        rank_frame, structural_frame = run(
            runs, out_dir, mapping_csv, structural_csv,
            source3_locked=bool(args.source3_locked),
            fan_threshold=float(args.fan_threshold),
            persistent_frac=float(args.persistent_frac),
            per_class_csv=Path(args.per_class_csv) if args.per_class_csv else None,
        )
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 3

    log.info(
        "eval04 complete: %d rank-correlation rows, %d structural rows",
        len(rank_frame), len(structural_frame),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
