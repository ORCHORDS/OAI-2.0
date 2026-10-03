"""Tests for oai2.runtime.residency.

These tests cover the deterministic residency and capacity-trend accountant
introduced for WI-QOS-002 (Refs #232). They exercise lifecycle transitions,
peak tracking, capacity trend windows and aggregate statistics with
hand-rolled fixtures, no time mocking, and no concurrency.
"""

from __future__ import annotations

import math

import pytest

from oai2.reasoning.modes import ReasoningMode
from oai2.runtime.admission import AdmissionAction, AdmissionReason
from oai2.runtime.residency import (
    CapacityTrendPoint,
    ResidencyAccountant,
    ResidencyOutcome,
    ResidencyRecord,
    ResidencySummary,
)


def _request(
    request_id: str,
    *,
    mode: ReasoningMode = ReasoningMode.NORMAL,
    swarm_lanes: int = 1,
    required_memory_gb: float = 1.0,
    enqueued_at_ms: float = 0.0,
) -> tuple[str, str, int, float, float]:
    return (
        request_id,
        mode.value,
        swarm_lanes,
        required_memory_gb,
        enqueued_at_ms,
    )


def _record_admit(
    accountant: ResidencyAccountant,
    request_id: str,
    *,
    enqueued_at_ms: float,
    admitted_at_ms: float,
) -> ResidencyRecord:
    accountant.record_decision(
        request_id=request_id,
        mode=ReasoningMode.NORMAL.value,
        swarm_lanes=1,
        required_memory_gb=1.0,
        decision=AdmissionAction.ADMIT,
        reason=AdmissionReason.CAPACITY_AVAILABLE,
        enqueued_at_ms=enqueued_at_ms,
        now_ms=admitted_at_ms,
    )
    return accountant.get_record(request_id)  # type: ignore[return-value]


def _record_queue(
    accountant: ResidencyAccountant,
    request_id: str,
    *,
    enqueued_at_ms: float,
    now_ms: float,
) -> ResidencyRecord:
    accountant.record_decision(
        request_id=request_id,
        mode=ReasoningMode.NORMAL.value,
        swarm_lanes=1,
        required_memory_gb=1.0,
        decision=AdmissionAction.QUEUE,
        reason=AdmissionReason.MEMORY_PRESSURE,
        enqueued_at_ms=enqueued_at_ms,
        now_ms=now_ms,
    )
    return accountant.get_record(request_id)  # type: ignore[return-value]


def _record_reject(
    accountant: ResidencyAccountant,
    request_id: str,
    *,
    enqueued_at_ms: float,
    now_ms: float,
) -> ResidencyRecord:
    accountant.record_decision(
        request_id=request_id,
        mode=ReasoningMode.NORMAL.value,
        swarm_lanes=1,
        required_memory_gb=1.0,
        decision=AdmissionAction.REJECT,
        reason=AdmissionReason.QUEUE_FULL,
        enqueued_at_ms=enqueued_at_ms,
        now_ms=now_ms,
    )
    return accountant.get_record(request_id)  # type: ignore[return-value]


def test_residency_accountant_records_admit_and_completes() -> None:
    accountant = ResidencyAccountant()
    _record_admit(accountant, "req-1", enqueued_at_ms=100.0, admitted_at_ms=150.0)
    accountant.complete("req-1", now_ms=400.0)

    record = accountant.get_record("req-1")
    assert record is not None
    assert record.outcome is ResidencyOutcome.COMPLETED
    assert record.wait_ms == pytest.approx(50.0)
    assert record.residency_ms == pytest.approx(300.0)
    assert record.run_ms == pytest.approx(250.0)


def test_residency_accountant_records_queue_then_complete() -> None:
    accountant = ResidencyAccountant()
    _record_queue(accountant, "req-2", enqueued_at_ms=200.0, now_ms=200.0)
    # Promote later
    accountant.record_decision(
        request_id="req-2",
        mode=ReasoningMode.NORMAL.value,
        swarm_lanes=1,
        required_memory_gb=1.0,
        decision=AdmissionAction.ADMIT,
        reason=AdmissionReason.CAPACITY_AVAILABLE,
        enqueued_at_ms=200.0,
        now_ms=600.0,
    )
    accountant.complete("req-2", now_ms=900.0)

    record = accountant.get_record("req-2")
    assert record is not None
    assert record.outcome is ResidencyOutcome.COMPLETED
    assert record.wait_ms == pytest.approx(400.0)
    assert record.residency_ms == pytest.approx(700.0)
    assert record.run_ms == pytest.approx(300.0)


def test_residency_accountant_records_rejected_lifecycle() -> None:
    accountant = ResidencyAccountant()
    _record_reject(accountant, "req-3", enqueued_at_ms=0.0, now_ms=10.0)
    record = accountant.get_record("req-3")
    assert record is not None
    assert record.outcome is ResidencyOutcome.REJECTED
    assert record.wait_ms is None
    assert record.residency_ms is None


def test_residency_accountant_cancel_marks_terminal() -> None:
    accountant = ResidencyAccountant()
    _record_queue(accountant, "req-4", enqueued_at_ms=0.0, now_ms=0.0)
    accountant.cancel("req-4", now_ms=125.0)

    record = accountant.get_record("req-4")
    assert record is not None
    assert record.outcome is ResidencyOutcome.CANCELLED
    assert record.residency_ms == pytest.approx(125.0)


def test_residency_accountant_complete_is_idempotent() -> None:
    accountant = ResidencyAccountant()
    _record_admit(accountant, "req-5", enqueued_at_ms=0.0, admitted_at_ms=10.0)
    first = accountant.complete("req-5", now_ms=100.0)
    second = accountant.complete("req-5", now_ms=200.0)
    assert first.completed_at_ms == pytest.approx(100.0)
    assert second.completed_at_ms == pytest.approx(100.0)


def test_residency_accountant_cancel_is_idempotent_after_terminal() -> None:
    accountant = ResidencyAccountant()
    _record_admit(accountant, "req-6", enqueued_at_ms=0.0, admitted_at_ms=10.0)
    accountant.complete("req-6", now_ms=100.0)
    # Cancelling a completed request should be a no-op
    after = accountant.cancel("req-6", now_ms=200.0)
    assert after.outcome is ResidencyOutcome.COMPLETED
    assert after.completed_at_ms == pytest.approx(100.0)


def test_residency_accountant_duplicate_request_id_raises() -> None:
    accountant = ResidencyAccountant()
    _record_admit(accountant, "req-7", enqueued_at_ms=0.0, admitted_at_ms=10.0)
    with pytest.raises(ValueError, match="request already tracked"):
        _record_admit(accountant, "req-7", enqueued_at_ms=0.0, admitted_at_ms=10.0)


def test_residency_accountant_complete_unknown_request_raises() -> None:
    accountant = ResidencyAccountant()
    with pytest.raises(ValueError, match="unknown request"):
        accountant.complete("missing", now_ms=10.0)


def test_residency_accountant_cancel_unknown_request_raises() -> None:
    accountant = ResidencyAccountant()
    with pytest.raises(ValueError, match="unknown request"):
        accountant.cancel("missing", now_ms=10.0)


def test_residency_accountant_complete_rejected_raises() -> None:
    accountant = ResidencyAccountant()
    _record_reject(accountant, "req-8", enqueued_at_ms=0.0, now_ms=10.0)
    with pytest.raises(ValueError, match="cannot complete"):
        accountant.complete("req-8", now_ms=20.0)


def test_residency_accountant_rejects_invalid_arguments() -> None:
    accountant = ResidencyAccountant()
    with pytest.raises(ValueError, match="non-empty string"):
        accountant.record_decision(
            request_id="",
            mode=ReasoningMode.NORMAL.value,
            swarm_lanes=1,
            required_memory_gb=1.0,
            decision=AdmissionAction.ADMIT,
            reason=AdmissionReason.CAPACITY_AVAILABLE,
            enqueued_at_ms=0.0,
            now_ms=0.0,
        )
    with pytest.raises(ValueError, match="non-empty string"):
        accountant.record_decision(
            request_id="req-x",
            mode="",
            swarm_lanes=1,
            required_memory_gb=1.0,
            decision=AdmissionAction.ADMIT,
            reason=AdmissionReason.CAPACITY_AVAILABLE,
            enqueued_at_ms=0.0,
            now_ms=0.0,
        )
    with pytest.raises(ValueError, match="positive integer"):
        accountant.record_decision(
            request_id="req-x",
            mode=ReasoningMode.NORMAL.value,
            swarm_lanes=0,
            required_memory_gb=1.0,
            decision=AdmissionAction.ADMIT,
            reason=AdmissionReason.CAPACITY_AVAILABLE,
            enqueued_at_ms=0.0,
            now_ms=0.0,
        )
    with pytest.raises(ValueError, match="non-negative number"):
        accountant.record_decision(
            request_id="req-x",
            mode=ReasoningMode.NORMAL.value,
            swarm_lanes=1,
            required_memory_gb=-1.0,
            decision=AdmissionAction.ADMIT,
            reason=AdmissionReason.CAPACITY_AVAILABLE,
            enqueued_at_ms=0.0,
            now_ms=0.0,
        )
    with pytest.raises(ValueError, match="non-negative number"):
        accountant.record_decision(
            request_id="req-x",
            mode=ReasoningMode.NORMAL.value,
            swarm_lanes=1,
            required_memory_gb=1.0,
            decision=AdmissionAction.ADMIT,
            reason=AdmissionReason.CAPACITY_AVAILABLE,
            enqueued_at_ms=-1.0,
            now_ms=0.0,
        )


def test_residency_accountant_rejects_invalid_capacity_window() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ResidencyAccountant(capacity_window=0)
    with pytest.raises(ValueError, match="positive integer"):
        ResidencyAccountant(capacity_window=-1)  # type: ignore[arg-type]


def test_residency_accountant_rejects_invalid_capacity_inputs() -> None:
    accountant = ResidencyAccountant()
    with pytest.raises(ValueError, match="non-negative number"):
        accountant.record_capacity(
            used_memory_gb=-1.0,
            available_memory_gb=10.0,
            active_tasks=0,
            queue_depth=0,
            now_ms=0.0,
        )
    with pytest.raises(ValueError, match="non-negative number"):
        accountant.record_capacity(
            used_memory_gb=1.0,
            available_memory_gb=-1.0,
            active_tasks=0,
            queue_depth=0,
            now_ms=0.0,
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        accountant.record_capacity(
            used_memory_gb=1.0,
            available_memory_gb=10.0,
            active_tasks=-1,
            queue_depth=0,
            now_ms=0.0,
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        accountant.record_capacity(
            used_memory_gb=1.0,
            available_memory_gb=10.0,
            active_tasks=0,
            queue_depth=-1,
            now_ms=0.0,
        )


def test_residency_accountant_capacity_trend_tracks_peak() -> None:
    accountant = ResidencyAccountant()
    accountant.record_capacity(
        used_memory_gb=2.0,
        available_memory_gb=10.0,
        active_tasks=1,
        queue_depth=0,
        now_ms=0.0,
    )
    accountant.record_capacity(
        used_memory_gb=7.0,
        available_memory_gb=5.0,
        active_tasks=4,
        queue_depth=2,
        now_ms=100.0,
    )
    accountant.record_capacity(
        used_memory_gb=3.0,
        available_memory_gb=9.0,
        active_tasks=2,
        queue_depth=0,
        now_ms=200.0,
    )

    trend = accountant.capacity_trend()
    assert len(trend) == 3
    assert trend[0].used_memory_gb == pytest.approx(2.0)
    assert trend[1].used_memory_gb == pytest.approx(7.0)
    assert trend[2].used_memory_gb == pytest.approx(3.0)
    summary = accountant.summary()
    assert summary.peak_used_memory_gb == pytest.approx(7.0)
    assert summary.peak_active_tasks == 4
    assert summary.peak_queue_depth == 2


def test_residency_accountant_capacity_trend_window_evicts_oldest() -> None:
    accountant = ResidencyAccountant(capacity_window=2)
    for index, used in enumerate([1.0, 2.0, 3.0]):
        accountant.record_capacity(
            used_memory_gb=used,
            available_memory_gb=10.0,
            active_tasks=0,
            queue_depth=0,
            now_ms=float(index * 10),
        )
    trend = accountant.capacity_trend()
    assert [point.used_memory_gb for point in trend] == [2.0, 3.0]


def test_residency_accountant_summary_percentiles_and_counts() -> None:
    accountant = ResidencyAccountant()
    # Six completed admits with known wait times: 10, 20, 30, 40, 50, 60
    for index, wait in enumerate([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]):
        request_id = f"req-{index}"
        _record_admit(
            accountant,
            request_id,
            enqueued_at_ms=0.0,
            admitted_at_ms=wait,
        )
        accountant.complete(request_id, now_ms=wait + 100.0)
    # One pending
    _record_queue(accountant, "req-pending", enqueued_at_ms=0.0, now_ms=0.0)
    # One rejected
    _record_reject(accountant, "req-rejected", enqueued_at_ms=0.0, now_ms=0.0)

    summary = accountant.summary()
    assert isinstance(summary, ResidencySummary)
    assert summary.total == 8
    assert summary.admitted == 6
    assert summary.rejected == 1
    assert summary.in_flight == 1
    assert summary.completed == 6
    assert summary.cancelled == 0
    assert summary.mean_wait_ms == pytest.approx(35.0)
    assert summary.max_wait_ms == pytest.approx(60.0)
    # p50 ~ 35, p95 ~ 57, p99 ~ 59.4 with our six samples
    assert math.isclose(summary.p50_wait_ms, 35.0, abs_tol=1.0)
    assert math.isclose(summary.p95_wait_ms, 57.0, abs_tol=5.0)
    assert math.isclose(summary.p99_wait_ms, 59.4, abs_tol=5.0)


def test_residency_accountant_summary_includes_residency_times() -> None:
    accountant = ResidencyAccountant()
    _record_admit(accountant, "req-a", enqueued_at_ms=0.0, admitted_at_ms=5.0)
    accountant.complete("req-a", now_ms=205.0)  # residency 205, run 200
    _record_admit(accountant, "req-b", enqueued_at_ms=10.0, admitted_at_ms=20.0)
    accountant.complete("req-b", now_ms=110.0)  # residency 100, run 90

    summary = accountant.summary()
    assert summary.completed == 2
    assert summary.max_residency_ms == pytest.approx(205.0)
    assert math.isclose(summary.mean_residency_ms, 152.5, abs_tol=0.01)


def test_residency_accountant_summary_empty() -> None:
    accountant = ResidencyAccountant()
    summary = accountant.summary()
    assert summary.total == 0
    assert summary.admitted == 0
    assert summary.rejected == 0
    assert summary.cancelled == 0
    assert summary.in_flight == 0
    assert summary.completed == 0
    # Latency statistics are None, not 0.0. Nothing was ever admitted, so
    # there is no queue-wait or residency distribution to summarise. This used
    # to assert a complete, flawless 0.0 profile -- mean, p50, p95, p99 and max
    # for both -- out of a tracker that had never seen a request, in a summary
    # that feeds promotion gates.
    #
    # `peak_used_memory_gb` stays 0.0 deliberately: that is a resource fact
    # ("nothing was resident"), not a distributional claim about observations
    # that were never taken.
    assert summary.mean_wait_ms is None
    assert summary.p50_wait_ms is None
    assert summary.p95_wait_ms is None
    assert summary.p99_wait_ms is None
    assert summary.max_wait_ms is None
    assert summary.mean_residency_ms is None
    assert summary.max_residency_ms is None
    assert summary.peak_used_memory_gb == 0.0
    assert summary.peak_active_tasks == 0
    assert summary.peak_queue_depth == 0


def test_residency_accountant_in_flight_count_excludes_terminal() -> None:
    accountant = ResidencyAccountant()
    _record_admit(accountant, "running-1", enqueued_at_ms=0.0, admitted_at_ms=10.0)
    _record_queue(accountant, "waiting-1", enqueued_at_ms=0.0, now_ms=0.0)
    _record_admit(accountant, "done-1", enqueued_at_ms=0.0, admitted_at_ms=10.0)
    accountant.complete("done-1", now_ms=100.0)
    _record_reject(accountant, "rejected-1", enqueued_at_ms=0.0, now_ms=0.0)

    assert accountant.in_flight_count == 2


def test_residency_accountant_capacity_trend_correlates_with_counters() -> None:
    accountant = ResidencyAccountant()
    _record_admit(accountant, "req-1", enqueued_at_ms=0.0, admitted_at_ms=10.0)
    accountant.complete("req-1", now_ms=110.0)
    _record_reject(accountant, "req-2", enqueued_at_ms=0.0, now_ms=20.0)
    point = accountant.record_capacity(
        used_memory_gb=2.0,
        available_memory_gb=10.0,
        active_tasks=1,
        queue_depth=0,
        now_ms=200.0,
    )
    assert isinstance(point, CapacityTrendPoint)
    assert point.admitted_total == 1
    assert point.rejected_total == 1
    assert point.queued_total == 0


def test_residency_accountant_get_record_returns_none_for_unknown() -> None:
    accountant = ResidencyAccountant()
    assert accountant.get_record("missing") is None


def test_residency_accountant_pending_then_admit_then_complete() -> None:
    accountant = ResidencyAccountant()
    # Initial queue decision at t=0
    _record_queue(accountant, "req-x", enqueued_at_ms=0.0, now_ms=0.0)
    # Promotion event at t=500 reuses the same request_id with a new ADMIT
    accountant.record_decision(
        request_id="req-x",
        mode=ReasoningMode.NORMAL.value,
        swarm_lanes=1,
        required_memory_gb=1.0,
        decision=AdmissionAction.ADMIT,
        reason=AdmissionReason.CAPACITY_AVAILABLE,
        enqueued_at_ms=0.0,
        now_ms=500.0,
    )
    accountant.complete("req-x", now_ms=900.0)
    record = accountant.get_record("req-x")
    assert record is not None
    assert record.outcome is ResidencyOutcome.COMPLETED
    assert record.wait_ms == pytest.approx(500.0)
    assert record.residency_ms == pytest.approx(900.0)
