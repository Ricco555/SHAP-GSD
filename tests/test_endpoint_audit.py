"""
Tests for src/explainer/endpoint_audit.py — the pure cross-tab core for
scripts/12_novelty_audit.py's Pass 4 (--endpoint-audit), and for the
novelty_mode guard in scripts/12_novelty_audit.py's
audit_endpoint_novelty_vs_firing.

Follows tests/test_fidelity_novelty.py's style: pure-function unit tests,
no GPU/graph fixtures needed for T1-T7 since endpoint_audit.py is pure
Python/dict in, dict out. T9 uses a minimal unittest.mock stub for the
NodeStateManager (only `.novelty_mode` is read before the guard decides)
and a minimal DGL-graph stand-in, per specs/61_endpoint_novelty_audit.md
§4 T9.

Reference: specs/61_endpoint_novelty_audit.md §4 (T1-T9). T8 is a manual
verification requirement per the spec (no automated test possible without
a full graph+NSM fixture the spec does not otherwise require) and is not
implemented here — see the task report for the manual check result.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.explainer.endpoint_audit import (
    build_endpoint_firing_table,
    classify_endpoint_firing,
)

# ---------------------------------------------------------------------------
# Load scripts/12_novelty_audit.py as a module (scripts/ is not a package,
# and its filename starts with a digit so it cannot be `import`ed normally —
# same pattern as tests/test_pgexplainer_ckpt_meta.py's `_load_baselines_script`).
# ---------------------------------------------------------------------------

_NOVELTY_AUDIT_MODULE = None


def _load_novelty_audit_script():
    """Import scripts/12_novelty_audit.py as a module, cached across tests."""
    global _NOVELTY_AUDIT_MODULE
    if _NOVELTY_AUDIT_MODULE is None:
        path = REPO_ROOT / "scripts" / "12_novelty_audit.py"
        spec = importlib.util.spec_from_file_location("novelty_audit_phase12", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _NOVELTY_AUDIT_MODULE = module
    return _NOVELTY_AUDIT_MODULE


# ---------------------------------------------------------------------------
# T1 — basic firing classification
# ---------------------------------------------------------------------------


def test_basic_firing_classification():
    """Src unseen only, src fires: any=True, both=False, fired=True."""
    result = classify_endpoint_firing(True, False, 0.02, 0.0)
    assert result["any_endpoint_unseen"] is True
    assert result["both_endpoints_unseen"] is False
    assert result["fired"] is True


# ---------------------------------------------------------------------------
# T2 — noise floor respected
# ---------------------------------------------------------------------------


def test_noise_floor_respected():
    """Values below the 1e-6 floor must not count as fired (specs/59 §4.2b)."""
    result = classify_endpoint_firing(True, True, 5.6e-7, -3e-7)
    assert result["fired"] is False


# ---------------------------------------------------------------------------
# T3 — both-unseen vs any-unseen distinction
# ---------------------------------------------------------------------------


def test_both_vs_any_unseen_distinction():
    result = classify_endpoint_firing(True, True, 0.0, 0.0)
    assert result["any_endpoint_unseen"] is True
    assert result["both_endpoints_unseen"] is True
    assert result["fired"] is False


# ---------------------------------------------------------------------------
# T4 — neither endpoint unseen, still fires
# ---------------------------------------------------------------------------


def test_neither_unseen_still_fires():
    """Unseen and fired must not be conflated or assumed to co-occur."""
    result = classify_endpoint_firing(False, False, 0.05, 0.0)
    assert result["any_endpoint_unseen"] is False
    assert result["both_endpoints_unseen"] is False
    assert result["fired"] is True


# ---------------------------------------------------------------------------
# T5 — build_endpoint_firing_table per-class aggregation
# ---------------------------------------------------------------------------


def test_per_class_aggregation():
    records = [
        # class A: 2 flows, 1 any-unseen, 1 fired
        {"class_name": "A", "src_unseen": True, "dst_unseen": False,
         "src_novelty_shap": 0.0, "dst_novelty_shap": 0.0},
        {"class_name": "A", "src_unseen": False, "dst_unseen": False,
         "src_novelty_shap": 0.02, "dst_novelty_shap": 0.0},
        # class B: 2 flows, 2 any-unseen (1 both), 0 fired
        {"class_name": "B", "src_unseen": True, "dst_unseen": True,
         "src_novelty_shap": 0.0, "dst_novelty_shap": 0.0},
        {"class_name": "B", "src_unseen": True, "dst_unseen": False,
         "src_novelty_shap": 0.0, "dst_novelty_shap": 0.0},
    ]
    table = build_endpoint_firing_table(records)

    a = table["A"]
    assert a["n_flows"] == 2
    assert a["n_any_endpoint_unseen"] == 1
    assert a["frac_any_endpoint_unseen"] == pytest.approx(0.5)
    assert a["n_fired"] == 1
    assert a["frac_fired"] == pytest.approx(0.5)

    b = table["B"]
    assert b["n_flows"] == 2
    assert b["n_any_endpoint_unseen"] == 2
    assert b["frac_any_endpoint_unseen"] == pytest.approx(1.0)
    assert b["n_both_endpoints_unseen"] == 1
    assert b["frac_both_endpoints_unseen"] == pytest.approx(0.5)
    assert b["n_fired"] == 0
    assert b["frac_fired"] == pytest.approx(0.0)

    overall = table["_overall"]
    assert overall["n_flows"] == a["n_flows"] + b["n_flows"] == 4
    assert overall["n_any_endpoint_unseen"] == 3
    assert overall["n_fired"] == 1


# ---------------------------------------------------------------------------
# T6 — empty class list / empty records
# ---------------------------------------------------------------------------


def test_empty_records_no_raise():
    table = build_endpoint_firing_table([])
    assert table["_overall"]["n_flows"] == 0
    assert table["_overall"]["frac_any_endpoint_unseen"] == 0.0
    assert table["_overall"]["frac_fired"] == 0.0
    # No per-class keys beyond "_overall".
    assert list(table.keys()) == ["_overall"]


# ---------------------------------------------------------------------------
# T7 — reproduces the specs/57 anomaly shape on a constructed case
# ---------------------------------------------------------------------------


def test_anomaly_shape_high_unseen_zero_fired_vs_low_unseen_high_fired():
    """The table must structurally permit unseen-rate/fired-rate divergence."""
    records = []
    # "Backdoor"-like: mostly unseen, never fires.
    for _ in range(8):
        records.append({
            "class_name": "backdoor_like", "src_unseen": True, "dst_unseen": False,
            "src_novelty_shap": 0.0, "dst_novelty_shap": 0.0,
        })
    for _ in range(2):
        records.append({
            "class_name": "backdoor_like", "src_unseen": False, "dst_unseen": False,
            "src_novelty_shap": 0.0, "dst_novelty_shap": 0.0,
        })
    # "mitm"-like: rarely unseen, fires often.
    for _ in range(2):
        records.append({
            "class_name": "mitm_like", "src_unseen": True, "dst_unseen": False,
            "src_novelty_shap": 0.05, "dst_novelty_shap": 0.0,
        })
    for _ in range(8):
        records.append({
            "class_name": "mitm_like", "src_unseen": False, "dst_unseen": False,
            "src_novelty_shap": 0.05, "dst_novelty_shap": 0.0,
        })

    table = build_endpoint_firing_table(records)

    backdoor = table["backdoor_like"]
    assert backdoor["frac_any_endpoint_unseen"] == pytest.approx(0.8)
    assert backdoor["frac_fired"] == pytest.approx(0.0)

    mitm = table["mitm_like"]
    assert mitm["frac_any_endpoint_unseen"] == pytest.approx(0.2)
    assert mitm["frac_fired"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# T9 — novelty_mode guard fires
# ---------------------------------------------------------------------------


def test_guard_raises_on_wrong_novelty_mode():
    """recent_window mode must never silently produce a mislabeled table."""
    module = _load_novelty_audit_script()

    nsm = MagicMock()
    nsm.novelty_mode = "recent_window"

    # g_test/expl_dir are placeholders never touched, per step 0's ordering
    # (the guard checks nsm.novelty_mode before anything else is read).
    g_test_placeholder = object()
    expl_dir_placeholder = Path("/nonexistent/does/not/matter")

    with pytest.raises(ValueError) as excinfo:
        module.audit_endpoint_novelty_vs_firing(
            g_test_placeholder, nsm, expl_dir_placeholder
        )
    assert "unseen_in_training" in str(excinfo.value)


def test_guard_proceeds_under_correct_novelty_mode(tmp_path):
    """unseen_in_training mode proceeds and produces a self-describing, empty table."""
    module = _load_novelty_audit_script()

    import dgl
    import torch

    nsm = MagicMock()
    nsm.novelty_mode = "unseen_in_training"

    class _EmptyGraphStub:
        """Exposes only edata[dgl.EID].numpy() -> empty int64 array.

        find_edges/get_state_at_time are never reached because expl_dir has
        no class subdirectories, so the JSON walk (steps 2-6) yields zero
        files.
        """

        def __init__(self) -> None:
            self.edata = {dgl.EID: torch.zeros(0, dtype=torch.int64)}

    g_test = _EmptyGraphStub()
    empty_expl_dir = tmp_path  # no subdirectories -> iterdir() yields nothing

    result = module.audit_endpoint_novelty_vs_firing(g_test, nsm, empty_expl_dir)

    assert result["novelty_mode"] == "unseen_in_training"
    assert result["_overall"]["n_flows"] == 0
