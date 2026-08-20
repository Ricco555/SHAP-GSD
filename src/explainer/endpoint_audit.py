"""
Endpoint unseen-in-training vs. realized phi_N firing — cross-tab logic.

Pure, testable core for scripts/12_novelty_audit.py's Pass 4
(`--endpoint-audit`). This module performs no I/O and imports neither torch
nor dgl: it consumes plain Python/dict facts about a flow's two real
endpoints (recovered elsewhere via global-EID -> local-EID -> find_edges,
see scripts/12_novelty_audit.py's audit_endpoint_novelty_vs_firing) and
produces per-class aggregate counts.

Context: specs/57's per-class endpoint-unseen table was computed against
the wrong node population (`node_ids` in explanation JSONs is the
NON-target node set, not the flow's real (src, dst) — see
specs/61_endpoint_novelty_audit.md Section 1). This module is the corrected
replacement for that computation, joined against realized phi_N firing so
the two questions ("is the endpoint unseen?" and "did phi_N fire?") stay
distinguishable per class rather than assumed to co-occur.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def classify_endpoint_firing(
    src_unseen: bool,
    dst_unseen: bool,
    src_novelty_shap: float,
    dst_novelty_shap: float,
    fire_floor: float = 1e-6,
) -> dict[str, bool]:
    """One flow's endpoint-unseen / phi_N-firing facts, for cross-tab counting.

    Args:
        src_unseen: NodeStateManager dim-1 bit for the flow's real source
            node, i.e. get_state_at_time(src_nid, ts)[1] != 0, under
            novelty_mode="unseen_in_training".
        dst_unseen: same, for the destination node.
        src_novelty_shap: the src_novelty_shap scalar already written to the
            explanation JSON by scripts/06_explain.py.
        dst_novelty_shap: the dst_novelty_shap scalar already written to the
            explanation JSON by scripts/06_explain.py.
        fire_floor: noise floor matching scripts/12_novelty_audit.py's
            existing Pass-2 convention (1e-6) — do not hardcode a different
            value at any call site.

    Returns:
        dict with keys: any_endpoint_unseen (bool), both_endpoints_unseen
        (bool), fired (bool) — |src_novelty_shap| > fire_floor OR
        |dst_novelty_shap| > fire_floor.
    """
    any_endpoint_unseen = bool(src_unseen or dst_unseen)
    both_endpoints_unseen = bool(src_unseen and dst_unseen)
    fired = bool(abs(src_novelty_shap) > fire_floor or abs(dst_novelty_shap) > fire_floor)
    return {
        "any_endpoint_unseen": any_endpoint_unseen,
        "both_endpoints_unseen": both_endpoints_unseen,
        "fired": fired,
    }


def build_endpoint_firing_table(records: list[dict[str, Any]]) -> dict[str, dict]:
    """Aggregate classify_endpoint_firing(...) rows into the per-class table.

    Args:
        records: one dict per explained flow, each with keys class_name,
            src_unseen, dst_unseen, src_novelty_shap, dst_novelty_shap
            (i.e. the classify_endpoint_firing(...) input fields, plus
            class_name).

    Returns:
        dict keyed by class_name, each value:
        {n_flows, n_any_endpoint_unseen, frac_any_endpoint_unseen,
         n_both_endpoints_unseen, frac_both_endpoints_unseen,
         n_fired, frac_fired}
        plus an "_overall" key with the same shape pooled across all
        records. Never raises on an empty `records` list — all fractions
        are 0 and all counts are 0 in that case.
    """
    per_class: dict[str, list[dict]] = {}
    for rec in records:
        cls_name = rec["class_name"]
        facts = classify_endpoint_firing(
            src_unseen=rec["src_unseen"],
            dst_unseen=rec["dst_unseen"],
            src_novelty_shap=rec["src_novelty_shap"],
            dst_novelty_shap=rec["dst_novelty_shap"],
            fire_floor=rec.get("fire_floor", 1e-6),
        )
        per_class.setdefault(cls_name, []).append(facts)

    def _summarize(facts_list: list[dict]) -> dict:
        n_flows = len(facts_list)
        n_any = sum(1 for f in facts_list if f["any_endpoint_unseen"])
        n_both = sum(1 for f in facts_list if f["both_endpoints_unseen"])
        n_fired = sum(1 for f in facts_list if f["fired"])
        return {
            "n_flows": n_flows,
            "n_any_endpoint_unseen": n_any,
            "frac_any_endpoint_unseen": round(n_any / n_flows, 6) if n_flows else 0.0,
            "n_both_endpoints_unseen": n_both,
            "frac_both_endpoints_unseen": round(n_both / n_flows, 6) if n_flows else 0.0,
            "n_fired": n_fired,
            "frac_fired": round(n_fired / n_flows, 6) if n_flows else 0.0,
        }

    table: dict[str, dict] = {
        cls_name: _summarize(facts_list) for cls_name, facts_list in per_class.items()
    }

    all_facts = [f for facts_list in per_class.values() for f in facts_list]
    table["_overall"] = _summarize(all_facts)

    return table
