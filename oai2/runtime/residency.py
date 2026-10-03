"""Residency and capacity-trend accounting for the deterministic admission pipeline.

This module records the full lifecycle of every admitted request (enqueue, admit,
complete, cancel, reject) and the live capacity trend (memory pressure, queue depth,
active tasks) so callers can compute wait-time statistics, admission-rate statistics
and tail pressure without leaking request/session identity. The data model is
deliberately scheduler-neutral: callers drive state transitions explicitly with
``now_ms`` and provide their own capacity snapshots.

The module is required by ``WI-QOS-002`` real residency/capacity telemetry and is
intentionally separate from ``admission_scheduler`` so it can be consumed by the
live service, the safe-batching scheduler, and the p95/p99/deadline-miss
promotion gate independently.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from .admission import AdmissionAction, AdmissionReason


class ResidencyOutcome(StrEnum):
    """Lifecycle outcome for a tracked admission request."""

    PENDING = "pending"
    ADMITTED = "admitted"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(slots=True, frozen=True)
class ResidencyRecord:
    """Public-safe residency record for a single request.

    Request/session identifiers are intentionally excluded from any summary
    shape so callers can expose this record in diagnostics without leaking
    user or workload identity.
    """

    request_id: str
    mode: str
    swarm_lanes: int
    required_memory_gb: float
    enqueued_at_ms: float
    decision: AdmissionAction
    decision_reason: AdmissionReason
    admitted_at_ms: float | None = None
    completed_at_ms: float | None = None
    outcome: ResidencyOutcome = ResidencyOutcome.PENDING

    @property
    def wait_ms(self) -> float | None:
        if self.admitted_at_ms is None:
            return None
        return max(self.admitted_at_ms - self.enqueued_at_ms, 0.0)

    @property
    def residency_ms(self) -> float | None:
        if self.completed_at_ms is None:
            return None
        return max(self.completed_at_ms - self.enqueued_at_ms, 0.0)

    @property
    def run_ms(self) -> float | None:
        if self.admitted_at_ms is None or self.completed_at_ms is None:
            return None
        return max(self.completed_at_ms - self.admitted_at_ms, 0.0)


@dataclass(slots=True, frozen=True)
class CapacityTrendPoint:
    """One point on the live capacity trend.

    Carries a snapshot of the residency counters at the same instant so the
    trend can be correlated with admission-rate changes without a second
    pass through the record store.
    """

    timestamp_ms: float
    used_memory_gb: float
    available_memory_gb: float
    active_tasks: int
    queue_depth: int
    admitted_total: int
    rejected_total: int
    queued_total: int


@dataclass(slots=True, frozen=True)
class ResidencySummary:
    """Aggregate residency/capacity statistics for reporting and promotion gates."""

    total: int
    admitted: int
    rejected: int
    cancelled: int
    in_flight: int
    completed: int
    # Latency statistics are None when NOTHING was ever admitted, not 0.0.
    #
    # This summary feeds promotion gates. Before this change an accountant
    # with total=0 reported mean/p50/p95/p99/max wait and residency all
    # exactly 0.0 -- a complete, flawless latency profile produced by a
    # tracker that had never observed a single request. A gate reading
    # `p95_wait_ms == 0.0` from that would conclude the residency system's
    # tail was perfect on the strength of an absence.
    #
    # This is the same failure closed in oai2/evals/qos.py, oai2/evals/
    # truth_runner.py, oai2/model/numerical_compare.py and
    # oai2/knowledge/evidence_package.py: an absent measurement must never
    # render as a clean one.
    mean_wait_ms: float | None
    p50_wait_ms: float | None
    p95_wait_ms: float | None
    p99_wait_ms: float | None
    max_wait_ms: float | None
    mean_residency_ms: float | None
    max_residency_ms: float | None
    peak_used_memory_gb: float
    peak_active_tasks: int
    peak_queue_depth: int


class ResidencyAccountant:
    """Deterministic residency and capacity-trend tracker.

    The accountant is scheduler-neutral and concurrency-free. The owner of the
    admission pipeline calls :meth:`record_decision` whenever a decision is
    made, :meth:`complete` / :meth:`cancel` whenever a tracked request exits
    the system, and :meth:`record_capacity` whenever a fresh capacity snapshot
    is available. Aggregates are produced by :meth:`summary`.
    """

    def __init__(self, *, capacity_window: int = 256) -> None:
        if isinstance(capacity_window, bool) or not isinstance(capacity_window, int):
            raise ValueError("capacity_window must be a positive integer")
        if capacity_window <= 0:
            raise ValueError("capacity_window must be a positive integer")
        self._records: dict[str, ResidencyRecord] = {}
        self._capacity_window = capacity_window
        self._capacity: deque[CapacityTrendPoint] = deque(maxlen=capacity_window)
        self._admitted_total = 0
        self._rejected_total = 0
        self._queued_total = 0
        self._completed_total = 0
        self._cancelled_total = 0
        self._peak_used_memory_gb = 0.0
        self._peak_active_tasks = 0
        self._peak_queue_depth = 0

    def record_decision(
        self,
        *,
        request_id: str,
        mode: str,
        swarm_lanes: int,
        required_memory_gb: float,
        decision: AdmissionAction,
        reason: AdmissionReason,
        enqueued_at_ms: float,
        now_ms: float,
    ) -> ResidencyRecord:
        """Record a freshly-made admission decision and start the request's lifecycle."""
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(mode, str) or not mode:
            raise ValueError("mode must be a non-empty string")
        if isinstance(swarm_lanes, bool) or not isinstance(swarm_lanes, int) or swarm_lanes <= 0:
            raise ValueError("swarm_lanes must be a positive integer")
        _non_negative_float(required_memory_gb, "required_memory_gb")
        _non_negative_float(enqueued_at_ms, "enqueued_at_ms")
        _non_negative_float(now_ms, "now_ms")
        existing = self._records.get(request_id)
        if existing is not None:
            # A queued request re-decided as ADMIT promotes in place; only
            # genuinely duplicated decisions are rejected.
            if existing.outcome is ResidencyOutcome.PENDING and decision is AdmissionAction.ADMIT:
                updated = ResidencyRecord(
                    request_id=existing.request_id,
                    mode=existing.mode,
                    swarm_lanes=existing.swarm_lanes,
                    required_memory_gb=existing.required_memory_gb,
                    enqueued_at_ms=existing.enqueued_at_ms,
                    decision=decision,
                    decision_reason=reason,
                    admitted_at_ms=float(now_ms),
                    outcome=ResidencyOutcome.ADMITTED,
                )
                self._records[request_id] = updated
                self._admitted_total += 1
                return updated
            raise ValueError(f"request already tracked: {request_id}")

        if decision is AdmissionAction.ADMIT:
            outcome = ResidencyOutcome.ADMITTED
            self._admitted_total += 1
            admitted_at = now_ms
        elif decision is AdmissionAction.QUEUE:
            outcome = ResidencyOutcome.PENDING
            self._queued_total += 1
            admitted_at = None
        else:
            outcome = ResidencyOutcome.REJECTED
            self._rejected_total += 1
            admitted_at = None

        record = ResidencyRecord(
            request_id=request_id,
            mode=mode,
            swarm_lanes=swarm_lanes,
            required_memory_gb=float(required_memory_gb),
            enqueued_at_ms=float(enqueued_at_ms),
            decision=decision,
            decision_reason=reason,
            admitted_at_ms=admitted_at,
            outcome=outcome,
        )
        self._records[request_id] = record
        return record

    def complete(self, request_id: str, *, now_ms: float) -> ResidencyRecord:
        """Mark a tracked request as completed and stamp the completion time."""
        _non_negative_float(now_ms, "now_ms")
        record = self._require_record(request_id)
        if record.outcome is ResidencyOutcome.COMPLETED:
            return record
        if record.outcome not in (ResidencyOutcome.ADMITTED, ResidencyOutcome.PENDING):
            raise ValueError(
                f"cannot complete request {request_id} from outcome {record.outcome}"
            )
        updated = ResidencyRecord(
            request_id=record.request_id,
            mode=record.mode,
            swarm_lanes=record.swarm_lanes,
            required_memory_gb=record.required_memory_gb,
            enqueued_at_ms=record.enqueued_at_ms,
            decision=record.decision,
            decision_reason=record.decision_reason,
            admitted_at_ms=record.admitted_at_ms,
            completed_at_ms=float(now_ms),
            outcome=ResidencyOutcome.COMPLETED,
        )
        self._records[request_id] = updated
        self._completed_total += 1
        return updated

    def cancel(self, request_id: str, *, now_ms: float) -> ResidencyRecord:
        """Mark a tracked request as cancelled and stamp the cancellation time."""
        _non_negative_float(now_ms, "now_ms")
        record = self._require_record(request_id)
        if record.outcome in (
            ResidencyOutcome.COMPLETED,
            ResidencyOutcome.CANCELLED,
            ResidencyOutcome.REJECTED,
        ):
            return record
        updated = ResidencyRecord(
            request_id=record.request_id,
            mode=record.mode,
            swarm_lanes=record.swarm_lanes,
            required_memory_gb=record.required_memory_gb,
            enqueued_at_ms=record.enqueued_at_ms,
            decision=record.decision,
            decision_reason=record.decision_reason,
            admitted_at_ms=record.admitted_at_ms,
            completed_at_ms=float(now_ms),
            outcome=ResidencyOutcome.CANCELLED,
        )
        self._records[request_id] = updated
        self._cancelled_total += 1
        return updated

    def record_capacity(
        self,
        *,
        used_memory_gb: float,
        available_memory_gb: float,
        active_tasks: int,
        queue_depth: int,
        now_ms: float,
    ) -> CapacityTrendPoint:
        """Record a capacity snapshot for the trend time series."""
        _non_negative_float(used_memory_gb, "used_memory_gb")
        _non_negative_float(available_memory_gb, "available_memory_gb")
        _non_negative_int(active_tasks, "active_tasks")
        _non_negative_int(queue_depth, "queue_depth")
        _non_negative_float(now_ms, "now_ms")
        point = CapacityTrendPoint(
            timestamp_ms=float(now_ms),
            used_memory_gb=float(used_memory_gb),
            available_memory_gb=float(available_memory_gb),
            active_tasks=int(active_tasks),
            queue_depth=int(queue_depth),
            admitted_total=self._admitted_total,
            rejected_total=self._rejected_total,
            queued_total=self._queued_total,
        )
        self._capacity.append(point)
        if point.used_memory_gb > self._peak_used_memory_gb:
            self._peak_used_memory_gb = point.used_memory_gb
        if point.active_tasks > self._peak_active_tasks:
            self._peak_active_tasks = point.active_tasks
        if point.queue_depth > self._peak_queue_depth:
            self._peak_queue_depth = point.queue_depth
        return point

    def get_record(self, request_id: str) -> ResidencyRecord | None:
        """Return the current residency record for a tracked request, if any."""
        return self._records.get(request_id)

    @property
    def admitted_total(self) -> int:
        return self._admitted_total

    @property
    def rejected_total(self) -> int:
        return self._rejected_total

    @property
    def queued_total(self) -> int:
        return self._queued_total

    @property
    def completed_total(self) -> int:
        return self._completed_total

    @property
    def cancelled_total(self) -> int:
        return self._cancelled_total

    @property
    def in_flight_count(self) -> int:
        return sum(
            1
            for record in self._records.values()
            if record.outcome in (ResidencyOutcome.PENDING, ResidencyOutcome.ADMITTED)
        )

    def capacity_trend(self) -> tuple[CapacityTrendPoint, ...]:
        """Return the recent capacity trend points in insertion order."""
        return tuple(self._capacity)

    def summary(self) -> ResidencySummary:
        """Compute aggregate wait-time, residency-time and peak-capacity statistics."""
        wait_times: list[float] = []
        residency_times: list[float] = []
        in_flight = 0
        completed = 0
        for record in self._records.values():
            if record.outcome is ResidencyOutcome.COMPLETED:
                completed += 1
                if record.residency_ms is not None:
                    residency_times.append(record.residency_ms)
                if record.wait_ms is not None:
                    wait_times.append(record.wait_ms)
            elif record.outcome in (ResidencyOutcome.PENDING, ResidencyOutcome.ADMITTED):
                in_flight += 1
        return ResidencySummary(
            total=len(self._records),
            admitted=self._admitted_total,
            rejected=self._rejected_total,
            cancelled=self._cancelled_total,
            in_flight=in_flight,
            completed=completed,
            mean_wait_ms=_mean(wait_times),
            p50_wait_ms=_percentile(wait_times, 50.0),
            p95_wait_ms=_percentile(wait_times, 95.0),
            p99_wait_ms=_percentile(wait_times, 99.0),
            max_wait_ms=max(wait_times) if wait_times else None,
            mean_residency_ms=_mean(residency_times),
            max_residency_ms=(
                max(residency_times) if residency_times else None
            ),
            peak_used_memory_gb=self._peak_used_memory_gb,
            peak_active_tasks=self._peak_active_tasks,
            peak_queue_depth=self._peak_queue_depth,
        )

    def _require_record(self, request_id: str) -> ResidencyRecord:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        record = self._records.get(request_id)
        if record is None:
            raise ValueError(f"unknown request: {request_id}")
        return record


def _non_negative_float(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _mean(values: list[float]) -> float | None:
    """Mean of the observations, or None when there were none.

    None, not 0.0. A 0.0 here is a claim that a queue wait was measured and
    took no time; an accountant that has admitted nothing has measured
    nothing. See the module note on absent measurements.
    """
    if not values:
        return None
    return float(statistics.fmean(values))


def _percentile(values: list[float], percentile: float) -> float | None:
    """Percentile of the observations, or None when there were none.

    The range check now runs BEFORE the emptiness check. It used to sit after
    it, so ``_percentile([], 500.0)`` returned 0.0 instead of raising -- the
    validation was silently skipped on exactly the path that returns a
    fabricated number.
    """
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


__all__ = [
    "ResidencyOutcome",
    "ResidencyRecord",
    "CapacityTrendPoint",
    "ResidencySummary",
    "ResidencyAccountant",
]
