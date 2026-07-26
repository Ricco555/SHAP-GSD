"""Unit tests for the opt-in per-batch timing helper (_BatchTimer).

These verify the profile_timing toggle in isolation, without constructing a
full Trainer (which would require model + graphs + feature stores + nsm).
The key contract: when disabled, no CUDA sync is invoked and no summary is
logged; when enabled, buckets accumulate and the summary fires.
"""

import logging

from src.model.trainer import _BatchTimer


def test_disabled_timer_is_a_noop_and_never_syncs() -> None:
    """Disabled: no cuda_sync calls, buckets stay zero, no summary emitted."""
    sync_calls = {"n": 0}

    def fake_sync() -> None:
        sync_calls["n"] += 1

    timer = _BatchTimer(enabled=False, cuda_sync=fake_sync)

    with timer.batch():
        with timer.section("sampling"):
            pass
        with timer.section("node_state"):
            pass
        with timer.gpu_section():
            pass

    assert sync_calls["n"] == 0, "cuda.synchronize must not run when disabled"
    assert timer.total_batch == 0.0
    assert all(v == 0.0 for v in timer.buckets.values())


def test_disabled_summary_logs_nothing(caplog) -> None:
    """Disabled: log_summary emits no records."""
    timer = _BatchTimer(enabled=False, cuda_sync=lambda: None)
    with caplog.at_level(logging.INFO, logger="src.model.trainer"):
        timer.log_summary("train", n_batches=3)
    assert caplog.records == []


def test_enabled_timer_syncs_and_accumulates() -> None:
    """Enabled: gpu_section brackets with two syncs; buckets accumulate."""
    sync_calls = {"n": 0}

    def fake_sync() -> None:
        sync_calls["n"] += 1

    timer = _BatchTimer(enabled=True, cuda_sync=fake_sync)

    with timer.batch():
        with timer.section("sampling"):
            pass
        with timer.section("node_state"):
            pass
        with timer.gpu_section():
            pass

    # gpu_section brackets the step with exactly two syncs.
    assert sync_calls["n"] == 2
    assert timer.total_batch >= 0.0
    for name in ("sampling", "node_state", "gpu_step"):
        assert timer.buckets[name] >= 0.0


def test_enabled_summary_logs_once(caplog) -> None:
    """Enabled: log_summary emits one INFO record with the label."""
    timer = _BatchTimer(enabled=True, cuda_sync=lambda: None)
    with caplog.at_level(logging.INFO, logger="src.model.trainer"):
        timer.log_summary("val", n_batches=7)
    assert len(caplog.records) == 1
    assert "val timing" in caplog.records[0].getMessage()


def test_unknown_bucket_name_raises() -> None:
    """A typo'd bucket name fails loudly rather than silently accumulating."""
    timer = _BatchTimer(enabled=True, cuda_sync=lambda: None)
    try:
        with timer.section("smapling"):  # deliberate typo
            pass
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unknown bucket name")
