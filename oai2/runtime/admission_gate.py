"""Live admission gate in front of a real inference runtime (WI-QOS-002).

Why this module exists
----------------------

The admission policy, the residency accountant and the safe-batching bridge all
existed on ``main`` before this module, and all three were exercised only by
tests. Nothing in the repository constructed a :class:`AdmissionPolicy`, fed it
a real capacity snapshot, waited on the resulting queue, or fed a decision to
:class:`ResidencyAccountant` outside a test file. The policy could therefore
be internally correct and still be unreachable: no request had ever passed
through it.

That gap is what REQ-QOS-021..028 and AC-QOS-021..025 need closed, so this
module is the binding layer. It is deliberately thin:

- :class:`AdmissionPolicy` still decides. Nothing here re-implements or
  re-orders a rule; every ADMIT/QUEUE/REJECT comes from ``policy.decide()``, so
  the policy that is unit-tested is the policy that runs.
- :class:`AdmissionQueue` still orders waiting work, including its bounded-wait
  anti-starvation promotion. This module never sorts the queue itself.
- :class:`ResidencyAccountant` still records the lifecycle and owns the
  wait/residency/peak statistics.
- :class:`AdmissionDecisionTrace` is still the public-safe decision shape.

What this module adds is the part none of them could own, because they are
documented as scheduler-neutral and concurrency-free: **a concurrency-safe
execution loop that holds a slot while a request runs and releases it on
completion, failure or cancellation.**

Scope, stated honestly
----------------------

This is a client-side admission gate over an existing runtime, not a new
gateway, scheduler or model server. It does not replace the serving path; it
decides whether a request is allowed to enter it.

The binding constraint on the llama.cpp layer is the **server slot count**,
which :class:`~oai2.runtime.llamacpp_runtime.LlamaServerRuntime` reports from
``/props`` as ``total_slots``. Memory is read from the host and still
participates in the policy, but the model is already resident and counted in
``used_memory_gb``, so ``max_active_tasks`` is usually what binds. A gate that
reported a memory-derived reason while actually gating on slots would be
misleading, so the trace always carries the ``max_active_tasks`` it used.

``per_lane_memory_gb`` is a **caller declaration, not a measurement**. The gate
cannot know a request's KV footprint before running it. The declaration is the
policy's input; the *evidence* that the declared limits held comes from sampling
the host during the run, which is the caller's job. Those two things are
reported separately and never conflated.

Cancellation and deadlines
--------------------------

:meth:`LlamaAdmissionGate.cancel` removes *queued* work and says so. A request
already executing inside a synchronous ``generate()`` cannot be recalled by
this module without owning the HTTP connection, so cancelling in-flight work
returns ``False`` instead of pretending it was cancelled. AC-QOS-023 is about
queue/capacity pressure, and only queued work is holding pressure this module
can release.

A deadline is checked when the policy decides, but a request can also *sit* in
the queue until its deadline passes while the policy is never re-consulted.
The gate re-checks that case on wake and rejects it as ``deadline_expired``
rather than admitting work that is already too late to be useful.

Status: IMPLEMENTED — offline under a fake runtime in
``tests/test_admission_gate.py``, live against the production NORMAL lane by
``scripts/admission_probe.py``.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from ..reasoning.modes import ReasoningMode
from .admission import (
    AdmissionAction,
    AdmissionDecision,
    AdmissionPolicy,
    AdmissionQueue,
    AdmissionReason,
    AdmissionRequest,
    CapacitySnapshot,
)
from .admission_scheduler import AdmissionDecisionTrace, AdmissionTelemetry
from .host_capacity import read_host_memory
from .inference import InferenceRequest, InferenceResponse, InferenceRuntime
from .residency import ResidencyAccountant

#: Declared incremental memory for one llama.cpp slot. See the module docstring:
#: a declaration that feeds the policy, not a measured KV footprint.
DEFAULT_LANE_MEMORY_GB = 0.25

#: Host memory held back from admission so a loaded machine can still breathe.
DEFAULT_RESERVED_MEMORY_GB = 2.0

#: Bound on waiting work. Past this, admission rejects rather than growing an
#: unbounded queue (REQ-QOS-022: explicit over-capacity behaviour, not
#: uncontrolled pressure).
DEFAULT_MAX_QUEUE_DEPTH = 64

#: A waiting request re-checks on this cadence. Promotion is normally signalled
#: explicitly on release, so this covers a missed signal and the deadline
#: re-check; it is not the normal wake-up path.
_POLL_SECONDS = 0.02


class GateOutcome(StrEnum):
    """What actually happened to a submitted request."""

    EXECUTED = "executed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class AdmissionGateError(RuntimeError):
    """Base class for admission outcomes that are not a completed request."""

    def __init__(self, message: str, trace: AdmissionDecisionTrace | None = None) -> None:
        super().__init__(message)
        self.trace = trace


class AdmissionRejected(AdmissionGateError):
    """The policy refused this request. Carries the public-safe trace."""


class AdmissionCancelled(AdmissionGateError):
    """The request was cancelled while queued, releasing its capacity."""


@dataclass(slots=True, frozen=True)
class GateCapacity:
    """Host facts supplied to the gate; the gate overlays its own live counters.

    Kept separate from :class:`CapacitySnapshot` because the host half is
    sampled outside the lock (it shells out to ``sysctl``/``vm_stat`` and costs
    milliseconds) while the gate's half must be read under the lock. Merging
    them would either hold the lock across a subprocess or decide against a
    stale in-flight count.
    """

    total_memory_gb: float
    used_memory_gb: float
    reserved_memory_gb: float = DEFAULT_RESERVED_MEMORY_GB
    max_active_tasks: int = 1
    max_queue_depth: int = DEFAULT_MAX_QUEUE_DEPTH

    def __post_init__(self) -> None:
        if not math.isfinite(self.total_memory_gb) or self.total_memory_gb <= 0.0:
            raise ValueError("total_memory_gb must be finite and > 0")
        if not math.isfinite(self.used_memory_gb) or self.used_memory_gb < 0.0:
            raise ValueError("used_memory_gb must be finite and >= 0")
        if self.used_memory_gb > self.total_memory_gb:
            raise ValueError("used_memory_gb cannot exceed total_memory_gb")
        if not math.isfinite(self.reserved_memory_gb) or self.reserved_memory_gb < 0.0:
            raise ValueError("reserved_memory_gb must be finite and >= 0")
        if self.reserved_memory_gb >= self.total_memory_gb:
            raise ValueError("reserved_memory_gb must be less than total_memory_gb")
        for name in ("max_active_tasks", "max_queue_depth"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def available_memory_gb(self) -> float:
        return max(
            self.total_memory_gb - self.reserved_memory_gb - self.used_memory_gb, 0.0
        )


@dataclass(slots=True, frozen=True)
class GateRequest:
    """One request to be admitted and executed.

    ``estimated_duration_ms`` and ``deadline_ms`` feed the existing
    :class:`AdmissionPolicy` and :class:`AdmissionQueue` ordering, so declaring
    them changes real scheduling order rather than decorating a record.
    """

    request_id: str
    inference: InferenceRequest
    mode: ReasoningMode = ReasoningMode.NORMAL
    per_lane_memory_gb: float = DEFAULT_LANE_MEMORY_GB
    estimated_duration_ms: float = 1_000.0
    deadline_ms: float | None = None
    swarm_lanes: int = 1

    def __post_init__(self) -> None:
        if not self.request_id or self.request_id != self.request_id.strip():
            raise ValueError("request_id must be a non-empty normalized string")
        if not math.isfinite(self.per_lane_memory_gb) or self.per_lane_memory_gb <= 0.0:
            raise ValueError("per_lane_memory_gb must be finite and > 0")
        if not math.isfinite(self.estimated_duration_ms) or self.estimated_duration_ms <= 0.0:
            raise ValueError("estimated_duration_ms must be finite and > 0")
        if self.deadline_ms is not None and (
            not math.isfinite(self.deadline_ms) or self.deadline_ms < 0.0
        ):
            raise ValueError("deadline_ms must be finite and >= 0")
        if isinstance(self.swarm_lanes, bool) or self.swarm_lanes <= 0:
            raise ValueError("swarm_lanes must be a positive integer")
        if self.mode is not ReasoningMode.SWARM and self.swarm_lanes != 1:
            raise ValueError("non-SWARM requests must use exactly one lane")


@dataclass(slots=True, frozen=True)
class GateResult:
    """Outcome plus the timings an admission decision is meant to protect."""

    request_id: str
    outcome: GateOutcome
    response: InferenceResponse | None
    decision: AdmissionDecision
    wait_ms: float
    run_ms: float
    trace: AdmissionDecisionTrace


@dataclass(slots=True)
class _Waiter:
    """Mutable per-request state shared between the caller thread and the gate."""

    request: GateRequest
    admission: AdmissionRequest
    event: threading.Event = field(default_factory=threading.Event)
    decision: AdmissionDecision | None = None
    trace: AdmissionDecisionTrace | None = None
    admitted: bool = False
    cancelled: bool = False


class LlamaAdmissionGate:
    """Concurrency-safe admission gate over an existing :class:`InferenceRuntime`.

    At most ``max_active_tasks`` requests execute at once. Everything else waits
    in the existing :class:`AdmissionQueue` under the existing anti-starvation
    ordering, or is rejected with an explicit reason.
    """

    def __init__(
        self,
        runtime: InferenceRuntime,
        *,
        capacity: GateCapacity,
        policy: AdmissionPolicy | None = None,
        accountant: ResidencyAccountant | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._runtime = runtime
        self._capacity_source = capacity
        self._policy = policy or AdmissionPolicy()
        self._queue = AdmissionQueue(starvation_after_ms=self._policy.starvation_after_ms)
        self._accountant = accountant or ResidencyAccountant()
        self._clock = clock or _monotonic_ms
        self._lock = threading.Lock()
        self._waiters: dict[str, _Waiter] = {}
        self._in_flight = 0
        self._last_trace: AdmissionDecisionTrace | None = None
        self._decision_counts: dict[AdmissionAction, int] = {
            action: 0 for action in AdmissionAction
        }
        self._reason_counts: dict[AdmissionReason, int] = {}

    # -- public surface ---------------------------------------------------

    @property
    def accountant(self) -> ResidencyAccountant:
        return self._accountant

    @property
    def max_active_tasks(self) -> int:
        """The concurrency limit actually in force, from the real host facts."""
        return self._capacity_source.max_active_tasks

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    @property
    def queue_depth(self) -> int:
        with self._lock:
            return max(len(self._waiters) - self._in_flight, 0)

    @property
    def telemetry(self) -> AdmissionTelemetry:
        with self._lock:
            return AdmissionTelemetry(
                decisions_total=sum(self._decision_counts.values()),
                admitted=self._decision_counts[AdmissionAction.ADMIT],
                queued=self._decision_counts[AdmissionAction.QUEUE],
                rejected=self._decision_counts[AdmissionAction.REJECT],
                cancelled=self._accountant.cancelled_total,
                reason_counts=tuple(
                    sorted((reason.value, count) for reason, count in self._reason_counts.items())
                ),
                last_decision=self._last_trace,
            )

    def capacity_snapshot(self) -> CapacitySnapshot:
        """Sample current host capacity and overlay the gate's live counters.

        Deliberately re-samples rather than reusing whatever justified the last
        decision: REQ-QOS-021 requires admission to consider *current* state,
        and a cached snapshot would decide against a machine that has moved on.
        """
        return self._snapshot(
            active_tasks=self.in_flight,
            queue_depth=self.queue_depth,
        )

    def submit(self, request: GateRequest) -> GateResult:
        """Admit, wait for, and execute one request.

        Raises :class:`AdmissionRejected` when the policy refuses it and
        :class:`AdmissionCancelled` when it is cancelled while queued. Returns a
        :class:`GateResult` only when the request actually ran.
        """
        waiter = self._register(request)
        if not waiter.admitted:
            raise self._build_failure(waiter)

        started = self._clock()
        try:
            response = self._runtime.generate(request.inference)
        finally:
            self._release(waiter)
        decision, trace = self._settled(waiter)
        return GateResult(
            request_id=request.request_id,
            outcome=GateOutcome.EXECUTED,
            response=response,
            decision=decision,
            wait_ms=max(started - waiter.admission.enqueued_at_ms, 0.0),
            run_ms=max(self._clock() - started, 0.0),
            trace=trace,
        )

    def cancel(self, request_id: str) -> bool:
        """Cancel *queued* work. ``True`` only if it really was queued.

        In-flight work is not cancelled: this module does not own the HTTP
        connection an executing request is using, and reporting a cancellation
        that stopped nothing would make the telemetry lie.
        """
        with self._lock:
            waiter = self._waiters.get(request_id)
            if waiter is None or waiter.cancelled or waiter.admitted:
                return False
            self._queue.cancel(request_id)
            # Drop the waiter here rather than waiting for its thread to wake:
            # AC-QOS-023 is about capacity pressure falling promptly, and a
            # cancelled request that still counted against queue depth until an
            # unrelated thread was scheduled would be exactly the lag the
            # requirement is about.
            self._waiters.pop(request_id, None)
            waiter.cancelled = True
            waiter.event.set()
            self._accountant.cancel(request_id, now_ms=self._clock())
            return True

    def host_memory(self) -> dict[str, float | None]:
        return read_host_memory()

    # -- registration and waiting -----------------------------------------

    def _register(self, request: GateRequest) -> _Waiter:
        now = self._clock()
        admission = AdmissionRequest(
            request_id=request.request_id,
            mode=request.mode,
            per_lane_memory_gb=request.per_lane_memory_gb,
            estimated_duration_ms=request.estimated_duration_ms,
            enqueued_at_ms=now,
            deadline_ms=request.deadline_ms,
            swarm_lanes=request.swarm_lanes,
        )
        waiter = _Waiter(request=request, admission=admission)

        with self._lock:
            if request.request_id in self._waiters:
                raise ValueError(f"request already in flight: {request.request_id}")
            # Decide *before* registering the waiter. The capacity snapshot's
            # queue depth is the number of requests already waiting, so a
            # request that counted itself would fill the last queue slot and
            # reject itself -- the same off-by-one the existing
            # AdmissionBatchController avoids by deciding against
            # ``len(self._pending)`` rather than the post-insert length.
            self._decide_locked(waiter)
            action = waiter.decision.action if waiter.decision else None
            if action is AdmissionAction.QUEUE:
                self._waiters[request.request_id] = waiter
                self._queue.enqueue(admission)
            elif action is AdmissionAction.ADMIT:
                self._waiters[request.request_id] = waiter
                self._in_flight += 1
                waiter.admitted = True
            # A REJECT is recorded by the accountant but occupies no gate
            # state, so it is deliberately not inserted into _waiters.

        if not waiter.admitted and waiter.decision is not None:
            # Only a QUEUE decision has anything to wait for. A REJECT is final
            # the moment the policy returns it, and blocking on it would turn a
            # fast, explicit refusal into an indefinite hang.
            if waiter.decision.action is AdmissionAction.QUEUE:
                self._await_promotion(waiter)
        return waiter

    def _await_promotion(self, waiter: _Waiter) -> None:
        """Block until a release promotes us, or we are cancelled, or we expire.

        Promotion is signalled explicitly on release, so the common wake-up is
        a set event rather than a timeout. The poll exists for two reasons: a
        deadline can pass *while queued*, which the policy only sees at decision
        time, and a signal can in principle be missed.
        """
        while True:
            waiter.event.wait(_POLL_SECONDS)
            if waiter.cancelled or waiter.admitted:
                return
            deadline = waiter.request.deadline_ms
            if deadline is not None and self._clock() >= deadline:
                with self._lock:
                    if waiter.cancelled or waiter.admitted:
                        return
                    self._queue.cancel(waiter.request.request_id)
                    waiter.decision = AdmissionDecision(
                        request_id=waiter.request.request_id,
                        action=AdmissionAction.REJECT,
                        reason=AdmissionReason.DEADLINE_EXPIRED,
                        required_memory_gb=waiter.admission.aggregate_memory_gb,
                        available_memory_gb=self._capacity_source.available_memory_gb,
                    )
                    waiter.trace = self._trace_locked(waiter, waiter.decision)
                    self._last_trace = waiter.trace
                    self._decision_counts[AdmissionAction.REJECT] += 1
                    self._reason_counts[AdmissionReason.DEADLINE_EXPIRED] = (
                        self._reason_counts.get(AdmissionReason.DEADLINE_EXPIRED, 0) + 1
                    )
                    self._accountant.cancel(
                        waiter.request.request_id, now_ms=self._clock()
                    )
                waiter.event.set()
                return

    # -- decisions --------------------------------------------------------

    def _decide_locked(
        self, waiter: _Waiter, admission: AdmissionRequest | None = None
    ) -> AdmissionDecision:
        """Ask the policy, and refuse any queueing that could never be satisfied.

        The policy can legitimately return QUEUE for a request that will never
        get in: it is a pure function of the capacity snapshot, and it cannot
        know whether any admitted work exists that could later complete and
        free capacity. With ``in_flight == 0`` no such completion can come, so
        honouring the QUEUE would block the caller forever. That is the
        unbounded-wait failure REQ-QOS-022 exists to prevent, so the gate
        converts it to an explicit rejection that keeps the policy's reason.
        """
        capacity = self._snapshot(
            active_tasks=self._in_flight,
            queue_depth=max(len(self._waiters) - self._in_flight, 0),
        )
        target = admission if admission is not None else waiter.admission
        decision = self._policy.decide(target, capacity, now_ms=self._clock())
        if decision.action is AdmissionAction.QUEUE and self._in_flight == 0:
            decision = AdmissionDecision(
                request_id=decision.request_id,
                action=AdmissionAction.REJECT,
                reason=decision.reason,
                required_memory_gb=decision.required_memory_gb,
                available_memory_gb=decision.available_memory_gb,
            )
        self._record_locked(waiter, decision, capacity)
        waiter.decision = decision
        return decision

    def _record_locked(
        self, waiter: _Waiter, decision: AdmissionDecision, capacity: CapacitySnapshot
    ) -> None:
        self._decision_counts[decision.action] += 1
        self._reason_counts[decision.reason] = (
            self._reason_counts.get(decision.reason, 0) + 1
        )
        self._accountant.record_decision(
            request_id=waiter.request.request_id,
            mode=waiter.request.mode.value,
            swarm_lanes=waiter.request.swarm_lanes,
            required_memory_gb=waiter.admission.aggregate_memory_gb,
            decision=decision.action,
            reason=decision.reason,
            enqueued_at_ms=waiter.admission.enqueued_at_ms,
            now_ms=self._clock(),
        )
        trace = self._trace_locked(waiter, decision, capacity=capacity)
        waiter.trace = trace
        self._last_trace = trace

    def _trace_locked(
        self,
        waiter: _Waiter,
        decision: AdmissionDecision,
        capacity: CapacitySnapshot | None = None,
    ) -> AdmissionDecisionTrace:
        source = self._capacity_source
        return AdmissionDecisionTrace(
            action=decision.action,
            reason=decision.reason,
            mode=waiter.request.mode.value,
            swarm_lanes=waiter.request.swarm_lanes,
            required_memory_gb=decision.required_memory_gb,
            available_memory_gb=decision.available_memory_gb,
            active_tasks=capacity.active_tasks if capacity else self._in_flight,
            max_active_tasks=source.max_active_tasks,
            pending_admission=max(len(self._waiters) - self._in_flight, 0),
            max_queue_depth=source.max_queue_depth,
            ready_queue_depth=max(len(self._waiters) - self._in_flight, 0),
            deadline_present=waiter.request.deadline_ms is not None,
        )

    def _snapshot(self, *, active_tasks: int, queue_depth: int) -> CapacitySnapshot:
        source = self._capacity_source
        return CapacitySnapshot(
            total_memory_gb=source.total_memory_gb,
            used_memory_gb=source.used_memory_gb,
            reserved_memory_gb=source.reserved_memory_gb,
            active_tasks=active_tasks,
            max_active_tasks=source.max_active_tasks,
            queue_depth=min(queue_depth, source.max_queue_depth),
            max_queue_depth=source.max_queue_depth,
        )

    # -- release and promotion --------------------------------------------

    def _release(self, waiter: _Waiter) -> None:
        """Free the slot, then promote waiting work in the existing queue order."""
        with self._lock:
            self._in_flight = max(self._in_flight - 1, 0)
            self._waiters.pop(waiter.request.request_id, None)
            self._accountant.complete(waiter.request.request_id, now_ms=self._clock())
            self._accountant.record_capacity(
                used_memory_gb=self._capacity_source.used_memory_gb,
                available_memory_gb=self._capacity_source.available_memory_gb,
                active_tasks=self._in_flight,
                queue_depth=max(len(self._waiters) - self._in_flight, 0),
                now_ms=self._clock(),
            )
            self._promote_locked()

    def _promote_locked(self) -> None:
        while self._in_flight < self._capacity_source.max_active_tasks:
            admission = self._queue.pop_next(now_ms=self._clock())
            if admission is None:
                return
            waiter = self._waiters.get(admission.request_id)
            if waiter is None or waiter.cancelled or waiter.admitted:
                # A cancelled waiter may still be tracked; skip it and keep
                # looking rather than promoting a request nobody is waiting for.
                continue

            decision = self._decide_locked(waiter, admission)
            if decision.action is AdmissionAction.ADMIT:
                self._in_flight += 1
                waiter.admitted = True
                waiter.event.set()
                return
            if decision.action is AdmissionAction.REJECT:
                # Recorded as a rejection in place, so the waiting caller is
                # told the real reason instead of being left to time out.
                waiter.event.set()
                return
            # Still QUEUE, which means admitted work exists that could complete
            # and free capacity. Re-queue in place so the existing
            # anti-starvation ordering is preserved rather than reordered here,
            # and stop: the next release re-tries.
            self._queue.enqueue(admission)
            return

    # -- failure ----------------------------------------------------------

    def _build_failure(self, waiter: _Waiter) -> AdmissionGateError:
        with self._lock:
            self._waiters.pop(waiter.request.request_id, None)
        decision, trace = self._settled(waiter)
        if waiter.cancelled and (
            decision is not None and decision.reason is not AdmissionReason.DEADLINE_EXPIRED
        ):
            return AdmissionCancelled(
                f"request cancelled while queued: {waiter.request.request_id}", trace=trace
            )
        if decision is not None and decision.action is AdmissionAction.REJECT:
            return AdmissionRejected(
                f"request rejected ({decision.reason.value}): {waiter.request.request_id}",
                trace=trace,
            )
        return AdmissionCancelled(
            f"request did not run: {waiter.request.request_id}", trace=trace
        )

    @staticmethod
    def _settled(waiter: _Waiter) -> tuple[AdmissionDecision, AdmissionDecisionTrace]:
        """The decision/trace pair for a finished waiter, or a safe placeholder.

        A caller reaching here without a decision would mean the gate lost
        track of its own request; raising would be better than inventing one, so
        the placeholder is only a last resort and is never silent in practice.
        """
        if waiter.decision is None or waiter.trace is None:
            raise RuntimeError(f"gate lost decision state for {waiter.request.request_id}")
        return waiter.decision, waiter.trace


def _monotonic_ms() -> float:
    return time.monotonic() * 1000.0


def gate_capacity_from_host(
    *,
    max_active_tasks: int,
    reserved_memory_gb: float = DEFAULT_RESERVED_MEMORY_GB,
    max_queue_depth: int = DEFAULT_MAX_QUEUE_DEPTH,
    memory: dict[str, float | None] | None = None,
) -> GateCapacity:
    """Build :class:`GateCapacity` from a real host reading.

    Raises ``RuntimeError`` when the host total is unreadable. Inventing a
    default physical-memory figure would let the policy reason about a machine
    it never observed, which is the same class of defect as a fabricated model
    identity.
    """
    reading = memory if memory is not None else read_host_memory()
    total = reading.get("total_gb")
    used = reading.get("used_gb")
    if total is None or used is None:
        raise RuntimeError("host memory is unreadable; refusing to invent a capacity figure")
    return GateCapacity(
        total_memory_gb=float(total),
        used_memory_gb=float(used),
        reserved_memory_gb=reserved_memory_gb,
        max_active_tasks=max_active_tasks,
        max_queue_depth=max_queue_depth,
    )


__all__ = [
    "DEFAULT_LANE_MEMORY_GB",
    "DEFAULT_MAX_QUEUE_DEPTH",
    "DEFAULT_RESERVED_MEMORY_GB",
    "AdmissionCancelled",
    "AdmissionGateError",
    "AdmissionRejected",
    "GateCapacity",
    "GateOutcome",
    "GateRequest",
    "GateResult",
    "LlamaAdmissionGate",
    "gate_capacity_from_host",
]
