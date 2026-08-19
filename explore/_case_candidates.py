"""Single source of truth for the curated case-study / topology-panel flows.

`explore/case_studies.py` and `explore/graph/topology_panel.py` both render
one flow per attack class. Before this module existed they sampled
independently and disagreed on four classes (Analysis, Generic,
Reconnaissance, Worms) -- a reviewer comparing the case-study figure to the
topology figure for the "same" class would see two different flows. See
`final/review01/coder_instructions_figure_determinism.md` S4.

Selection is computed live from whichever run's `outputs/explanations/`
directory is currently resolved (via `explore._paths.paths()`), so it is
dataset-agnostic -- it never hardcodes a class name or edge_id. Every
consumer still reads neighbour counts, correctness, p(y_hat) and node-SHAP
magnitudes live from the run's own `outputs/explanations/<class>/<eid>.json`
at render time (S0 "definition of done"); this module only decides *which*
`(class_name, edge_id)` pair each script renders.

Selection rule ("curated" candidates -- one flagship flow per class):
  1. A flow is eligible only if `true_label == predicted_label` (correctly
     classified) AND its total attribution magnitude
     (sum of |feature_group_shap| + |neighbor_shap| + |node_shap| +
     |src_novelty_shap| + |dst_novelty_shap|) exceeds the project-wide
     1e-6 floor used for phi_N elsewhere (see `scripts/12_novelty_audit.py`).
  2. Among eligible flows, the flagship is the one with the highest
     confidence `predicted_proba[predicted_label]`; ties are broken by
     higher attribution magnitude, then by ascending edge_id for
     determinism. Confidence is the primary key, not blended into a single
     weighted score, so the rule stays auditable without needing to
     justify a weighting constant.
  3. A class with zero eligible flows (UNSW's Backdoor: 0/200 correctly
     classified) is excluded from the curated list entirely.

Classes excluded by rule 3 still get a *topology-only* fallback flow, since
the 2-hop topology panel is structural and does not depend on prediction
correctness (see topology_panel.py's own docstring): the flow with the
highest predicted-class probability across the WHOLE class, correct or not.
"""

import json
from pathlib import Path

_ATTRIBUTION_FLOOR = 1e-6


def _attribution_magnitude(rec: dict) -> float:
    """Sum of |phi| across all three attribution layers plus novelty."""
    return (
        sum(abs(v) for v in rec.get("feature_group_shap", []))
        + sum(abs(v) for v in rec.get("neighbor_shap", []))
        + sum(abs(v) for v in rec.get("node_shap", []))
        + abs(rec.get("src_novelty_shap", 0.0))
        + abs(rec.get("dst_novelty_shap", 0.0))
    )


def _confidence(rec: dict) -> float:
    """predicted_proba[predicted_label], i.e. p(y_hat) for the predicted class."""
    proba = rec.get("predicted_proba", [])
    pred  = rec.get("predicted_label")
    if pred is None or not (0 <= pred < len(proba)):
        return 0.0
    return float(proba[pred])


def _load_class_records(cls_dir: Path) -> list[tuple[int, dict]]:
    """Return [(edge_id, record), ...] for every parseable JSON in cls_dir."""
    records = []
    for jf in sorted(cls_dir.glob("*.json")):
        try:
            rec = json.loads(jf.read_text())
        except Exception:
            continue
        eid = rec.get("edge_id")
        if eid is None:
            continue
        records.append((int(eid), rec))
    return records


def select_curated_candidates(expl_dir: Path) -> list[tuple[str, int]]:
    """One flagship (correctly-classified, nonzero-attribution) flow per class.

    Classes with zero eligible flows are silently excluded (see module
    docstring rule 3) -- callers that need a fallback for those classes
    should use `select_topology_only_fallback`.
    """
    candidates: list[tuple[str, int]] = []
    for cls_dir in sorted(p for p in expl_dir.iterdir() if p.is_dir()):
        cls = cls_dir.name
        eligible = []
        for eid, rec in _load_class_records(cls_dir):
            if rec.get("true_label") != rec.get("predicted_label"):
                continue
            mag = _attribution_magnitude(rec)
            if mag <= _ATTRIBUTION_FLOOR:
                continue
            eligible.append((eid, _confidence(rec), mag))
        if not eligible:
            print(f"  SKIP {cls} — 0 flows pass the curated-selection filter "
                  f"(correctly classified + attribution > {_ATTRIBUTION_FLOOR:g})")
            continue
        # Primary key: confidence desc. Tiebreak: attribution magnitude desc.
        # Final tiebreak: edge_id asc, for determinism.
        eligible.sort(key=lambda t: (-t[1], -t[2], t[0]))
        best_eid = eligible[0][0]
        candidates.append((cls, best_eid))
    return candidates


def select_topology_only_fallback(
    expl_dir: Path,
    curated: list[tuple[str, int]] | None = None,
) -> dict[str, int]:
    """Structural-only fallback flow for classes with zero curated-eligible
    flows: the highest predicted-class-probability flow in the class,
    correctness ignored (structural topology does not depend on it).

    Pass `curated` (an already-computed `select_curated_candidates(expl_dir)`
    result) to avoid recomputing it and re-emitting its SKIP log lines.
    """
    if curated is None:
        curated = select_curated_candidates(expl_dir)
    curated_classes = {cls for cls, _ in curated}
    fallback: dict[str, int] = {}
    for cls_dir in sorted(p for p in expl_dir.iterdir() if p.is_dir()):
        cls = cls_dir.name
        if cls in curated_classes:
            continue
        best_eid, best_conf = None, -1.0
        for eid, rec in _load_class_records(cls_dir):
            conf = _confidence(rec)
            if conf > best_conf or (conf == best_conf and (best_eid is None or eid < best_eid)):
                best_eid, best_conf = eid, conf
        if best_eid is not None:
            fallback[cls] = best_eid
    return fallback
