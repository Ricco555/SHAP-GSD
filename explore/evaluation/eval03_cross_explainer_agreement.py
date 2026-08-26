"""
eval03_cross_explainer_agreement.py — cross-explainer agreement, tested
=======================================================================
Implements specs/64 Part I §4.3 / Part II §14.3.

Why this exists:
  The pipeline already emits a side-by-side baseline comparison table. That
  table is a rendering, not a statistic: it says what each explainer's fidelity
  was, never whether two explainers actually agree about *which* features or
  nodes mattered. This script turns the per-explainer result CSVs into rank and
  overlap agreement statistics, per class, per dataset, with Benjamini-Hochberg
  correction inside an explicitly named comparison family.

Two agreement spaces, never mixed:
  ``feature_group``  — pairs among the explainers that attribute to feature
                       groups. Their per-flow vectors are aligned index-by-index
                       to the same group list, so a rank correlation between
                       them is meaningful.
  ``node_coalition`` — pairs among the explainers that attribute to nodes in a
                       k-hop subgraph. A cross-space pair is never computed:
                       correlating a feature-group ranking against a node
                       ranking is exactly the conflation the terminology rule
                       of this package exists to prevent.

  The node-coalition space currently yields no computable correlation, and this
  script reports that rather than papering over it — see
  :data:`NODE_ALIGNMENT_NOTE` and the evidence recorded per row.

Files read (read-only):
  <run>/outputs/baselines/<explainer>_results.csv   — per-flow baseline vectors
  <run>/artifacts/feature_groups.json               — the group list and its order
  <run>/outputs/explanations/<Class>/*.json         — SHAP-GSD's own vectors
  <run>/artifacts/evaluation/metrics.json           — the class list

Files output (under ``outputs/figures/evaluation/``):
  cross_explainer_agreement.csv
  cross_explainer_disagreement_flags.csv
  cross_explainer_agreement_heatmap_<dataset>_<space>.{pdf,png,txt}

Output-location note: ``OUT_DIR`` is the hardcoded
``REPO_ROOT/outputs/figures/evaluation/``, NOT resolved through
``explore._paths.paths()``/``SHAP_GSD_CONFIG`` (specs/64 D3): that variable
names one active run, and these tables and heatmaps span every resolved
dataset, so filing them under one run's subtree would misattribute them.
Restated here rather than relied on by reference (specs/48 §1.1 pt 4).

Currency caveat: this script consumes whatever baseline CSVs are on disk and
cannot detect staleness. The repo's standing "regenerate the baseline outputs
before citing them" item applies to these numbers unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import combinations
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
    artifacts_dir,
    outputs_dir,
    resolve_datasets_from_args,
)
from explore.evaluation._load import load_eval_metrics  # noqa: E402
from explore.evaluation._stats import apply_family, rank_agreement  # noqa: E402
from explore.evaluation.eval05_per_class_explanations import (  # noqa: E402
    iter_class_records,
)

log = logging.getLogger("eval03_cross_explainer_agreement")

OUT_DIR: Path = DEFAULT_OUT_DIR  # see module docstring for why this is hardcoded

AGREEMENT_CSV_NAME = "cross_explainer_agreement.csv"
FLAGS_CSV_NAME = "cross_explainer_disagreement_flags.csv"
HEATMAP_STEM_PREFIX = "cross_explainer_agreement_heatmap_"

#: This package's own name for the pipeline's Shapley explainer, used as one
#: side of every pair.
SHAP_GSD = "shap_gsd"

SPACE_FEATURE_GROUP = "feature_group"
SPACE_NODE_COALITION = "node_coalition"

#: Baseline result files, by explainer name, and which space each belongs to.
#: The mapping is derived from the column each CSV actually carries — a
#: ``group_scores`` column is a feature-group vector, a ``node_scores`` column
#: is a node-coalition vector — so a new baseline appears here by dropping its
#: CSV into ``outputs/baselines/``, with no code change.
BASELINE_SUFFIX = "_results.csv"
FEATURE_VECTOR_COLUMN = "group_scores"
NODE_VECTOR_COLUMN = "node_scores"

#: Threshold below which SHAP-GSD's median pairwise rho against the other
#: explainers in a space is flagged as a disagreement region. Named, never
#: inline, and exposed as ``--rho-threshold``.
DEFAULT_RHO_THRESHOLD = 0.3

#: Why no node-coalition correlation is computable against SHAP-GSD today.
#: This is deliberately specific rather than a generic "column missing": the
#: column is present, it just cannot be aligned.
NODE_ALIGNMENT_NOTE = (
    "node_scores is an unlabelled length-N vector with no accompanying node-id "
    "column, so it cannot be aligned element-by-element to any other "
    "explainer's node vector; a positional correlation between two independent "
    "explainers' unlabelled node vectors would not be a rank agreement between "
    "the same nodes"
)

AGREEMENT_CSV_COLUMNS: tuple[str, ...] = (
    "dataset",
    "class_name",
    "space",
    "explainer_a",
    "explainer_b",
    "spearman_rho",
    "kendall_tau",
    "p_raw",
    "p_corrected",
    "correction_family",
    "jaccard_topk",
    "k",
    "n_flows",
    "notes",
)

FLAGS_CSV_COLUMNS: tuple[str, ...] = (
    "dataset",
    "class_name",
    "space",
    "shap_gsd_rank_vs_majority_rho",
    "threshold",
    "flagged",
    "n_pairs",
    "notes",
)


def discover_baselines(run: DatasetRun) -> dict[str, tuple[Path, str]]:
    """Find a run's baseline explainer CSVs and classify each by space.

    Args:
        run: The resolved dataset run.

    Returns:
        ``{explainer_name: (csv_path, space)}``; empty when the run has no
        ``outputs/baselines/`` directory.
    """
    baselines_dir = outputs_dir(run) / "baselines"
    if not baselines_dir.is_dir():
        log.warning("%s: no baselines directory at %s", run.name, baselines_dir)
        return {}
    found: dict[str, tuple[Path, str]] = {}
    for path in sorted(baselines_dir.glob(f"*{BASELINE_SUFFIX}")):
        name = path.name[: -len(BASELINE_SUFFIX)]
        header = pd.read_csv(path, nrows=0).columns
        if FEATURE_VECTOR_COLUMN in header:
            found[name] = (path, SPACE_FEATURE_GROUP)
        elif NODE_VECTOR_COLUMN in header:
            found[name] = (path, SPACE_NODE_COALITION)
        else:
            log.warning(
                "%s: %s carries neither %r nor %r, so it exposes no per-flow "
                "ranking vector; skipped",
                run.name, path.name, FEATURE_VECTOR_COLUMN, NODE_VECTOR_COLUMN,
            )
    log.info("%s: baseline explainers found: %s", run.label, sorted(found))
    return found


def load_feature_group_names(run: DatasetRun) -> list[str]:
    """Load the run's feature-group names, in the order the vectors use.

    The count is never hardcoded — it is preprocessing-derived and differs per
    dataset.

    Args:
        run: The resolved dataset run.

    Returns:
        The ordered group names; empty when the artifact is absent.
    """
    path = artifacts_dir(run) / "feature_groups.json"
    if not path.is_file():
        log.warning("%s: no feature_groups.json at %s", run.name, path)
        return []
    raw = json.loads(path.read_text())
    groups = raw.get("groups", raw)
    if isinstance(groups, dict):
        return list(groups.keys())
    return [str(g.get("name", g)) for g in groups]


def shap_gsd_feature_vectors(run: DatasetRun, class_names: list[str]) -> dict[str, tuple[list[str], np.ndarray]]:
    """Aggregate SHAP-GSD's per-class mean |phi_F| vector, with its group order.

    Args:
        run: The resolved dataset run.
        class_names: Classes to aggregate, exactly as in ``label_map.json``.

    Returns:
        ``{class_name: (group_names, mean_abs_phi)}`` for classes with flows.
    """
    out: dict[str, tuple[list[str], np.ndarray]] = {}
    for class_name in class_names:
        records = iter_class_records(run, class_name)
        if not records:
            continue
        names = list(records[0].get("feature_group_names") or [])
        if not names:
            continue
        accum = np.zeros(len(names), dtype=float)
        n_used = 0
        for rec in records:
            values = rec.get("feature_group_shap") or []
            if len(values) != len(names):
                continue
            accum += np.abs(np.asarray(values, dtype=float))
            n_used += 1
        if n_used:
            out[class_name] = (names, accum / n_used)
    return out


def baseline_class_vectors(
    path: Path, column: str,
) -> dict[str, tuple[np.ndarray, int]]:
    """Aggregate one baseline explainer's per-class mean absolute score vector.

    Args:
        path: The explainer's ``*_results.csv``.
        column: ``group_scores`` or ``node_scores``.

    Returns:
        ``{class_name: (mean_abs_vector, n_flows)}``. Classes whose per-flow
        vectors differ in length are skipped with a warning — a mean over
        ragged vectors would be meaningless.
    """
    frame = pd.read_csv(path)
    if "_class_name" not in frame.columns or column not in frame.columns:
        log.warning("%s: expected columns %r and '_class_name'", path, column)
        return {}
    out: dict[str, tuple[np.ndarray, int]] = {}
    for class_name, group in frame.groupby("_class_name"):
        vectors: list[np.ndarray] = []
        for raw in group[column]:
            try:
                values = json.loads(raw)
            except (TypeError, ValueError):
                continue
            vectors.append(np.abs(np.asarray(values, dtype=float)))
        lengths = {v.shape[0] for v in vectors}
        if not vectors or len(lengths) != 1:
            log.info(
                "%s / %s: %d per-flow vectors with %d distinct lengths; no "
                "per-class mean vector computed",
                path.name, class_name, len(vectors), len(lengths),
            )
            continue
        out[str(class_name)] = (np.mean(np.vstack(vectors), axis=0), len(vectors))
    return out


def _node_length_evidence(run: DatasetRun, path: Path, class_names: list[str]) -> str:
    """Measure the per-flow node-vector length mismatch, for the notes column.

    The mismatch is what makes the node-coalition gap structural rather than a
    parsing miss, so it is measured live rather than asserted.

    Args:
        run: The resolved dataset run.
        path: The baseline explainer's CSV.
        class_names: Classes to sample.

    Returns:
        A short evidence string, or ``""`` when nothing could be compared.
    """
    frame = pd.read_csv(path, usecols=["edge_id", NODE_VECTOR_COLUMN, "_class_name"])
    lengths_by_eid = {}
    for _, row in frame.iterrows():
        try:
            lengths_by_eid[int(row["edge_id"])] = len(json.loads(row[NODE_VECTOR_COLUMN]))
        except (TypeError, ValueError):
            continue
    compared = 0
    matched = 0
    for class_name in class_names:
        for rec in iter_class_records(run, class_name):
            baseline_len = lengths_by_eid.get(int(rec.get("edge_id", -1)))
            if baseline_len is None:
                continue
            compared += 1
            if baseline_len == len(rec.get("node_ids") or []):
                matched += 1
        if compared >= 500:
            break
    if not compared:
        return ""
    return (
        f"measured on {compared} shared flows: the baseline node vector length "
        f"matches SHAP-GSD's node_ids length in {matched} of them"
    )


def compute_agreement(
    runs: list[DatasetRun], top_k: int,
) -> pd.DataFrame:
    """Compute pairwise agreement for every (dataset, class, space, pair).

    Args:
        runs: The resolved dataset runs.
        top_k: k for the top-k Jaccard overlap.

    Returns:
        A frame carrying exactly :data:`AGREEMENT_CSV_COLUMNS`, BH-corrected
        within each (dataset, space) family.
    """
    rows: list[dict[str, object]] = []
    for run in runs:
        metrics = load_eval_metrics(run)
        baselines = discover_baselines(run)
        group_names = load_feature_group_names(run)
        shap_vectors = shap_gsd_feature_vectors(run, metrics.class_names)

        feature_baselines = {
            name: path for name, (path, space) in baselines.items()
            if space == SPACE_FEATURE_GROUP
        }
        node_baselines = {
            name: path for name, (path, space) in baselines.items()
            if space == SPACE_NODE_COALITION
        }

        # --- feature-group space --------------------------------------------
        feature_vectors: dict[str, dict[str, tuple[np.ndarray, int]]] = {}
        for name, path in feature_baselines.items():
            feature_vectors[name] = baseline_class_vectors(path, FEATURE_VECTOR_COLUMN)
        for class_name, (names, vector) in shap_vectors.items():
            feature_vectors.setdefault(SHAP_GSD, {})[class_name] = (
                vector, len(iter_class_records(run, class_name)),
            )
            if group_names and names != group_names:
                log.warning(
                    "%s/%s: explanation group order differs from "
                    "feature_groups.json; the feature-group comparison assumes "
                    "index alignment", run.label, class_name,
                )

        for class_name in metrics.class_names:
            available = [
                name for name in sorted(feature_vectors)
                if class_name in feature_vectors[name]
            ]
            for a, b in combinations(available, 2):
                vec_a, n_a = feature_vectors[a][class_name]
                vec_b, n_b = feature_vectors[b][class_name]
                if vec_a.shape != vec_b.shape:
                    rows.append({
                        "dataset": run.label, "class_name": class_name,
                        "space": SPACE_FEATURE_GROUP, "explainer_a": a,
                        "explainer_b": b, "spearman_rho": float("nan"),
                        "kendall_tau": float("nan"), "p_raw": float("nan"),
                        "jaccard_topk": float("nan"), "k": top_k,
                        "n_flows": min(n_a, n_b),
                        "notes": (
                            f"vector lengths differ ({vec_a.shape[0]} vs "
                            f"{vec_b.shape[0]}); not alignable"
                        ),
                    })
                    continue
                agreement = rank_agreement(vec_a, vec_b, top_k)
                rows.append({
                    "dataset": run.label, "class_name": class_name,
                    "space": SPACE_FEATURE_GROUP, "explainer_a": a,
                    "explainer_b": b, "spearman_rho": agreement.spearman_rho,
                    "kendall_tau": agreement.kendall_tau,
                    "p_raw": agreement.p_raw,
                    "jaccard_topk": agreement.jaccard_topk, "k": agreement.k,
                    "n_flows": min(n_a, n_b), "notes": agreement.note,
                })

        # --- node-coalition space -------------------------------------------
        evidence = ""
        if node_baselines:
            first = sorted(node_baselines)[0]
            evidence = _node_length_evidence(
                run, node_baselines[first], metrics.class_names,
            )
        node_participants = sorted(node_baselines) + [SHAP_GSD]
        explained_counts = {
            class_name: len(iter_class_records(run, class_name))
            for class_name in metrics.class_names
        }
        for class_name in metrics.class_names:
            for a, b in combinations(node_participants, 2):
                note = NODE_ALIGNMENT_NOTE
                if evidence and SHAP_GSD in (a, b):
                    note = f"{NODE_ALIGNMENT_NOTE}; {evidence}"
                rows.append({
                    "dataset": run.label, "class_name": class_name,
                    "space": SPACE_NODE_COALITION, "explainer_a": a,
                    "explainer_b": b, "spearman_rho": float("nan"),
                    "kendall_tau": float("nan"), "p_raw": float("nan"),
                    "jaccard_topk": float("nan"), "k": top_k,
                    "n_flows": explained_counts.get(class_name, 0),
                    "notes": note,
                })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=list(AGREEMENT_CSV_COLUMNS))
    corrected = [
        apply_family(part, f"dataset={label};space={space}")
        for (label, space), part in frame.groupby(["dataset", "space"], sort=False)
    ]
    return pd.concat(corrected, ignore_index=True)[list(AGREEMENT_CSV_COLUMNS)]


def compute_disagreement_flags(
    agreement: pd.DataFrame, threshold: float,
) -> pd.DataFrame:
    """Flag classes where SHAP-GSD disagrees with the majority of baselines.

    Args:
        agreement: The pairwise agreement frame.
        threshold: rho below which the class is flagged.

    Returns:
        A frame carrying exactly :data:`FLAGS_CSV_COLUMNS` — every scored
        (dataset, class, space), flagged or not, so an unflagged class is
        visibly unflagged rather than absent.
    """
    rows: list[dict[str, object]] = []
    if agreement.empty:
        return pd.DataFrame(columns=list(FLAGS_CSV_COLUMNS))
    involved = agreement[
        (agreement["explainer_a"] == SHAP_GSD) | (agreement["explainer_b"] == SHAP_GSD)
    ]
    for (dataset_label, class_name, space), group in involved.groupby(
        ["dataset", "class_name", "space"], sort=False,
    ):
        values = group["spearman_rho"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        median = float(np.median(finite)) if finite.size else float("nan")
        flagged = bool(np.isfinite(median) and median < threshold)
        note = "" if finite.size else (
            "no computable rho against any baseline in this space, so no "
            "agreement claim is made either way"
        )
        rows.append({
            "dataset": dataset_label, "class_name": class_name, "space": space,
            "shap_gsd_rank_vs_majority_rho": median, "threshold": threshold,
            "flagged": flagged, "n_pairs": int(finite.size), "notes": note,
        })
    return pd.DataFrame(rows, columns=list(FLAGS_CSV_COLUMNS))


# --- figure ---------------------------------------------------------------

LABEL_FS = 9          # explore/AGENT.md §6 style constant
_GRAY = "#888888"     # reference lines and annotations
# _PRESENCE/_ABSENCE/_SHAP_GSD/_GNN_EXP intentionally not used here: an
# explainer x explainer agreement heatmap encodes a correlation, not a
# presence/absence-of-attribution sign and not a two-explainer series, so
# forcing those constants in would misrepresent the axis (specs/47 §4).


def write_heatmap(
    dataset_label: str,
    space: str,
    agreement: pd.DataFrame,
    out_dir: Path,
    threshold: float,
) -> None:
    """Write one (dataset, space) explainer x explainer heatmap trio.

    Args:
        dataset_label: The dataset's display label.
        space: ``feature_group`` or ``node_coalition``.
        agreement: That (dataset, space)'s agreement rows.
        out_dir: Output directory.
        threshold: The disagreement threshold, for the reasoning text.
    """
    stem = f"{HEATMAP_STEM_PREFIX}{dataset_label}_{space}"
    explainers = sorted(
        set(agreement["explainer_a"]) | set(agreement["explainer_b"])
    )
    size = len(explainers)
    matrix = np.full((size, size), np.nan, dtype=float)
    index = {name: i for i, name in enumerate(explainers)}
    # Each cell is the mean across that dataset's classes, written into both
    # triangles so the matrix reads symmetrically.
    means = (
        agreement.groupby(["explainer_a", "explainer_b"])["spearman_rho"]
        .mean().to_dict()
    )
    for (a, b), value in means.items():
        i, j = index[a], index[b]
        matrix[i, j] = value
        matrix[j, i] = value
    np.fill_diagonal(matrix, 1.0)

    fig, ax = plt.subplots(figsize=(1.2 * size + 3.0, 1.0 * size + 2.4))
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=-1.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03,
                 label="mean Spearman rho across classes")
    ax.set_xticks(range(size))
    ax.set_xticklabels(explainers, rotation=45, ha="right", fontsize=LABEL_FS - 1)
    ax.set_yticks(range(size))
    ax.set_yticklabels(explainers, fontsize=LABEL_FS - 1)
    for i in range(size):
        for j in range(size):
            value = matrix[i, j]
            ax.text(
                j, i, "n/a" if not np.isfinite(value) else f"{value:.2f}",
                ha="center", va="center", fontsize=LABEL_FS - 1,
                color=_GRAY if not np.isfinite(value) else "black",
            )
    ax.set_title(f"{dataset_label} — {space} agreement", fontsize=LABEL_FS + 1)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    n_scored = int(np.isfinite(agreement["spearman_rho"].to_numpy(dtype=float)).sum())
    findings = [
        f"  Explainers on this axis: {', '.join(explainers)}.",
        f"  Pairwise class-level correlations scored: {n_scored} of "
        f"{len(agreement)}.",
    ]
    if n_scored:
        finite = agreement["spearman_rho"].to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        findings.append(
            f"  Scored rho ranges {finite.min():.3f} to {finite.max():.3f}, "
            f"mean {finite.mean():.3f}; the disagreement threshold is "
            f"{threshold}."
        )
    else:
        findings.append(
            "  Every cell off the diagonal is n/a. This is the finding, not a "
            "rendering bug: see the notes column of "
            "cross_explainer_agreement.csv for the measured reason."
        )
    if size <= 2:
        findings.append(
            "  Only two explainers attribute in this space, so the matrix is "
            "2x2 by construction."
        )

    reasoning = f"""Figure reasoning — {stem}
{'=' * (len(stem) + 20)}

WHAT THE FIGURE SHOWS
---------------------
An explainer x explainer matrix for {dataset_label}, restricted to the
{space} attribution space. Each off-diagonal cell is the mean Spearman rank
correlation, across that dataset's classes, between the two explainers'
per-class importance vectors. The diagonal is 1.0 by definition. A cell
reading "n/a" is one where no correlation was computable at all; it is drawn
rather than left blank so that an uncomputable pair cannot be mistaken for a
weak one.

Explainer pairs are only ever formed within one attribution space. A
feature-group ranking and a node ranking answer different questions, and
correlating one against the other would produce a number with no defensible
interpretation.

KEY FINDINGS
------------
{chr(10).join(findings)}

PAPER FRAMING
-------------
Agreement between independent explainers is evidence that an attribution
reflects the model rather than the explainer's own inductive bias;
disagreement regions are themselves a reportable finding rather than a
failure. Correction is Benjamini-Hochberg FDR applied within one
(dataset, space) family, and both the raw and the corrected p-value are kept
in the CSV so neither can be mistaken for the other.

SUGGESTED FIGURE CAPTION
------------------------
Cross-explainer rank agreement on {dataset_label} in the {space} attribution
space: each cell is the mean Spearman correlation across classes between two
explainers' per-class importance vectors. Cells marked n/a were not
computable; see the accompanying CSV's notes column for the measured reason.
"""
    (out_dir / f"{stem}.txt").write_text(reasoning)
    log.info("wrote %s.{pdf,png,txt}", out_dir / stem)


def run(
    runs: list[DatasetRun],
    out_dir: Path,
    top_k: int,
    rho_threshold: float = DEFAULT_RHO_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and write every eval03 artifact.

    Args:
        runs: The resolved dataset runs.
        out_dir: Output directory.
        top_k: k for the top-k Jaccard overlap.
        rho_threshold: Disagreement-flag threshold.

    Returns:
        ``(agreement_frame, flags_frame)``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    agreement = compute_agreement(runs, top_k)
    agreement.to_csv(out_dir / AGREEMENT_CSV_NAME, index=False)
    log.info("wrote %s (%d rows)", out_dir / AGREEMENT_CSV_NAME, len(agreement))

    flags = compute_disagreement_flags(agreement, rho_threshold)
    flags.to_csv(out_dir / FLAGS_CSV_NAME, index=False)
    log.info("wrote %s (%d rows, %d flagged)", out_dir / FLAGS_CSV_NAME,
             len(flags), int(flags["flagged"].sum()) if not flags.empty else 0)

    if not agreement.empty:
        for (dataset_label, space), part in agreement.groupby(
            ["dataset", "space"], sort=False,
        ):
            write_heatmap(str(dataset_label), str(space), part, out_dir, rho_threshold)
    return agreement, flags


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector, or ``None`` to read ``sys.argv``.

    Returns:
        Process exit code (0 on success, 2 on a dataset-resolution failure).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Turn the per-explainer baseline result CSVs into rank and overlap "
            "agreement statistics per class per dataset, corrected within an "
            "explicitly named family (specs/64 Part I §4.3)."
        )
    )
    add_common_args(parser)
    parser.add_argument(
        "--rho-threshold", type=float, default=DEFAULT_RHO_THRESHOLD,
        help=f"SHAP-GSD median pairwise rho below which a class is flagged as "
             f"a disagreement region (default: {DEFAULT_RHO_THRESHOLD}).",
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

    agreement, flags = run(
        runs, Path(args.out_dir), int(args.top_k), float(args.rho_threshold),
    )
    log.info(
        "eval03 complete: %d agreement rows, %d flag rows",
        len(agreement), len(flags),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
