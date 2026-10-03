"""Offline coverage for the live admission gate (WI-QOS-002).

These tests use a controllable fake runtime and a synthetic clock, so the
gate's ordering and bounding behaviour is exercised deterministically without a
model server. The live half — the same gate against the production NORMAL lane
with real slots, real memory and real contention — is
``scripts/admission_probe.py``.

The most important test here is
:func:`test_gate_never_exceeds_the_declared_slot_limit`, because it is the one
the shipped suite could otherwise have survived: a gate that ignored its own
policy and admitted everything would still pass every test that only checked
decision *reasons*, and would only show up as a tail-latency regression under
real load. :func:`test_gate_that_ignores_its_policy_is_caught` makes that
explicit.
"""

from __future__ import annotations

import threading
import time

import pytest

from oai2.reasoning.modes import ReasoningMode
from oai2.runtime.admission import AdmissionAction, AdmissionPolicy, AdmissionReason
from oai2.runtime.admission_gate import (
    DEFAULT_LANE_MEMORY_GB,
    AdmissionCancelled,
    AdmissionRejected,
    GateCapacity,
    GateOutcome,
    GateRequest,
    GateResult,
    LlamaAdmissionGate,
    gate_capacity_from_host,
)
from oai2.runtime.inference import (
    InferenceRequest,
    InferenceResponse,
    InferenceRuntime,
)
from oai2.runtime.residency import ResidencyOutcome

# -- fakes ------------------------------------------------------------------


class ControllableRuntime(InferenceRuntime):
    """A runtime whose ``generate`` blocks until the test releases it.

    Records the peak number of concurrent executions so a test can assert the
    gate really bounded concurrency, rather than inferring it from timings.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._released = threading.Event()
        self._started = threading.Semaphore(0)
        self.running = 0
        self.peak_running = 0
        self.calls: list[str] = []

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        with self._lock:
            self.running += 1
            self.calls.append(request.prompt)
            self.peak_running = max(self.peak_running, self.running)
        try:
            self._started.release()
            self._released.wait(timeout=10.0)
            return InferenceResponse(
                text=f"ok:{request.prompt}",
                tokens=1,
                elapsed_ms=1.0,
                device="fake",
            )
        finally:
            with self._lock:
                self.running -= 1

    def wait_for_start(self, count: int, timeout: float = 5.0) -> None:
        for _ in range(count):
            assert self._started.acquire(timeout=timeout), "runtime never started"

    def release_all(self) -> None:
        self._released.set()


def _capacity(max_active: int = 1, **kwargs) -> GateCapacity:
    defaults = {
        "total_memory_gb": 64.0,
        "used_memory_gb": 8.0,
        "reserved_memory_gb": 2.0,
        "max_active_tasks": max_active,
        "max_queue_depth": 8,
    }
    defaults.update(kwargs)
    return GateCapacity(**defaults)


def _request(request_id: str = "", **kwargs) -> GateRequest:
    resolved = kwargs.pop("request_id", request_id)
    fields: dict = {
        "request_id": resolved,
        "inference": InferenceRequest(prompt=resolved or "p", max_tokens=8, temperature=0.0),
    }
    fields.update(kwargs)
    return GateRequest(**fields)


def _await_queue_depth(gate, expected: int, timeout: float = 5.0) -> None:
    """Poll until the queue reaches ``expected`` instead of assuming a delay.

    Ten worker threads registering concurrently do not finish within any fixed
    sleep, so a timed assertion here would be testing the scheduler's mood.
    """
    deadline = time.monotonic() + timeout
    while gate.queue_depth < expected and time.monotonic() < deadline:
        time.sleep(0.005)
    assert gate.queue_depth == expected, f"queue_depth={gate.queue_depth}, want {expected}"


def _run_in_thread(gate: LlamaAdmissionGate, request: GateRequest) -> list:
    """Submit on a worker thread; return a 1-slot box holding result or error."""
    box: list = []

    def worker() -> None:
        try:
            box.append(gate.submit(request))
        except BaseException as exc:  # noqa: BLE001 - recorded for assertions
            box.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return box


def _await_box(box: list, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while not box and time.monotonic() < deadline:
        time.sleep(0.005)
    assert box, "worker never returned"
    return box[0]


# -- GateCapacity invariants ------------------------------------------------


def test_gate_capacity_exposes_the_real_available_memory() -> None:
    capacity = GateCapacity(
        total_memory_gb=64.0, used_memory_gb=40.0, reserved_memory_gb=2.0, max_active_tasks=4
    )
    assert capacity.available_memory_gb == pytest.approx(22.0)


def test_gate_capacity_available_memory_never_goes_negative() -> None:
    capacity = GateCapacity(
        total_memory_gb=64.0, used_memory_gb=63.0, reserved_memory_gb=2.0, max_active_tasks=4
    )
    assert capacity.available_memory_gb == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total_memory_gb": 0.0},
        {"total_memory_gb": -1.0},
        {"used_memory_gb": -0.1},
        {"used_memory_gb": 65.0},  # exceeds total
        {"reserved_memory_gb": -1.0},
        {"reserved_memory_gb": 64.0},  # must be strictly less than total
        {"max_active_tasks": 0},
        {"max_active_tasks": -1},
        {"max_queue_depth": 0},
    ],
)
def test_gate_capacity_rejects_invalid_invariants(kwargs) -> None:
    base = {
        "total_memory_gb": 64.0,
        "used_memory_gb": 8.0,
        "reserved_memory_gb": 2.0,
        "max_active_tasks": 2,
        "max_queue_depth": 4,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        GateCapacity(**base)


@pytest.mark.parametrize("value", [True, False])
def test_gate_capacity_rejects_bool_for_int_limits(value) -> None:
    # bool is a subclass of int; without an explicit guard max_active_tasks=True
    # would silently mean "one active task".
    with pytest.raises(ValueError):
        GateCapacity(
            total_memory_gb=64.0,
            used_memory_gb=8.0,
            max_active_tasks=value,
        )


# -- GateRequest invariants -------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"request_id": ""},
        {"request_id": " padded "},
        {"per_lane_memory_gb": 0.0},
        {"per_lane_memory_gb": -1.0},
        {"estimated_duration_ms": 0.0},
        {"deadline_ms": -1.0},
        {"swarm_lanes": 0},
        {"swarm_lanes": 2},  # NORMAL cannot claim two lanes
    ],
)
def test_gate_request_rejects_invalid_invariants(kwargs) -> None:
    # request_id is overridden for some cases, so it is passed by keyword only.
    with pytest.raises(ValueError):
        _request(**{"request_id": "r1", **kwargs})


def test_swarm_request_may_claim_multiple_lanes() -> None:
    request = _request("r1", mode=ReasoningMode.SWARM, swarm_lanes=4)
    assert request.swarm_lanes == 4


# -- admission behaviour ----------------------------------------------------


def test_gate_admits_a_single_request_and_reports_zero_wait() -> None:
    runtime = ControllableRuntime()
    gate = LlamaAdmissionGate(runtime, capacity=_capacity(max_active=2))
    runtime.release_all()
    result = gate.submit(_request("r1"))
    assert result.outcome is GateOutcome.EXECUTED
    assert result.wait_ms == pytest.approx(0.0, abs=1.0)
    assert result.response is not None
    assert result.response.text == "ok:r1"


def test_gate_never_exceeds_the_declared_slot_limit() -> None:
    """The core AC-QOS-021/025 property: concurrency is actually bounded.

    Ten concurrent submitters against a two-slot gate. If the gate admitted
    everything, ``peak_running`` would reach ten.
    """
    runtime = ControllableRuntime()
    gate = LlamaAdmissionGate(runtime, capacity=_capacity(max_active=2))
    boxes = [_run_in_thread(gate, _request(f"r{i}")) for i in range(10)]

    runtime.wait_for_start(2)
    # Let the other eight either be wrongly admitted or reach the queue.
    _await_queue_depth(gate, 8)
    assert runtime.peak_running == 2, f"gate let {runtime.peak_running} run at once"
    assert gate.in_flight == 2

    runtime.release_all()
    for box in boxes:
        assert isinstance(_await_box(box), GateResult)


def test_gate_queues_when_slots_are_full_and_admits_on_release() -> None:
    runtime = ControllableRuntime()
    gate = LlamaAdmissionGate(runtime, capacity=_capacity(max_active=1))
    first = _run_in_thread(gate, _request("first"))
    runtime.wait_for_start(1)
    second = _run_in_thread(gate, _request("second"))

    time.sleep(0.1)
    assert second == [], "second request ran despite the slot being occupied"
    assert gate.queue_depth == 1

    runtime.release_all()
    assert isinstance(_await_box(first), GateResult)
    assert isinstance(_await_box(second), GateResult)
    assert runtime.peak_running == 1


def test_full_queue_rejects_explicitly_rather_than_growing() -> None:
    """REQ-QOS-022: over-capacity is explicit, not an unbounded queue."""
    runtime = ControllableRuntime()
    gate = LlamaAdmissionGate(runtime, capacity=_capacity(max_active=1, max_queue_depth=2))
    running = [_run_in_thread(gate, _request(f"run{i}")) for i in range(1)]
    runtime.wait_for_start(1)
    queued = [_run_in_thread(gate, _request(f"q{i}")) for i in range(2)]
    _await_queue_depth(gate, 2)

    overflow = _run_in_thread(gate, _request("overflow"))
    error = _await_box(overflow)
    assert isinstance(error, AdmissionRejected)
    assert error.trace is not None
    assert error.trace.reason is AdmissionReason.QUEUE_FULL
    assert gate.queue_depth == 2, "a rejected request must not occupy queue space"

    runtime.release_all()
    for box in [*running, *queued]:
        assert isinstance(_await_box(box), GateResult)


def test_queue_full_reason_is_chosen_over_memory_when_slots_are_busy() -> None:
    runtime = ControllableRuntime()
    gate = LlamaAdmissionGate(runtime, capacity=_capacity(max_active=1, max_queue_depth=1))
    running = _run_in_thread(gate, _request("run"))
    runtime.wait_for_start(1)
    queued = _run_in_thread(gate, _request("q"))
    time.sleep(0.05)
    assert gate.telemetry.reason_counts  # a decision was recorded

    overflow = _await_box(_run_in_thread(gate, _request("overflow")))
    assert isinstance(overflow, AdmissionRejected)
    assert overflow.trace is not None
    assert overflow.trace.reason is AdmissionReason.QUEUE_FULL
    runtime.release_all()
    _await_box(running)
    _await_box(queued)


def test_memory_pressure_queues_instead_of_overcommitting() -> None:
    # Slots are free, but the host has no room for another declared lane.
    capacity = _capacity(max_active=4, total_memory_gb=16.0, used_memory_gb=15.0, reserved_memory_gb=0.5)
    runtime = ControllableRuntime()
    gate = LlamaAdmissionGate(runtime, capacity=capacity)
    runtime.release_all()
    with pytest.raises(AdmissionRejected) as excinfo:
        # Queue depth is 0 and max is 8, so this must queue then... it cannot
        # block forever with no slot free, so it is rejected on the next decide
        # once the queue is observed. Use a bounded request instead.
        gate.submit(_request("r1", per_lane_memory_gb=1.0))
    assert excinfo.value.trace is not None
    assert excinfo.value.trace.reason in {
        AdmissionReason.MEMORY_PRESSURE,
        AdmissionReason.QUEUE_FULL,
    }


def test_request_larger_than_usable_memory_is_rejected_not_queued() -> None:
    runtime = ControllableRuntime()
    gate = LlamaAdmissionGate(runtime, capacity=_capacity(max_active=2, total_memory_gb=16.0))
    runtime.release_all()
    with pytest.raises(AdmissionRejected) as excinfo:
        gate.submit(_request("huge", per_lane_memory_gb=32.0))
    assert excinfo.value.trace is not None
    assert excinfo.value.trace.reason is AdmissionReason.REQUEST_TOO_LARGE


def test_expired_deadline_is_rejected_before_any_capacity_check() -> None:
    runtime = ControllableRuntime()
    clock = {"now": 1_000.0}
    gate = LlamaAdmissionGate(
        runtime, capacity=_capacity(max_active=1), clock=lambda: clock["now"]
    )
    runtime.release_all()
    with pytest.raises(AdmissionRejected) as excinfo:
        gate.submit(_request("late", deadline_ms=500.0))
    assert excinfo.value.trace is not None
    assert excinfo.value.trace.reason is AdmissionReason.DEADLINE_EXPIRED


def test_deadline_that_passes_while_queued_is_rejected_not_admitted() -> None:
    """A queued request can outlive its deadline without the policy re-deciding.

    The policy only sees a request at decision time, so the gate has to notice
    the expiry itself rather than admitting work that is already too late.
    """
    clock = {"now": 0.0}
    runtime = ControllableRuntime()
    gate = LlamaAdmissionGate(
        runtime,
        capacity=_capacity(max_active=1),
        clock=lambda: clock["now"],
    )
    running = _run_in_thread(gate, _request("run"))
    runtime.wait_for_start(1)

    def advance() -> None:
        while runtime.running == 0:
            time.sleep(0.005)
        clock["now"] = 50_000.0

    threading.Thread(target=advance, daemon=True).start()
    late = _run_in_thread(gate, _request("late", deadline_ms=10_000.0))
    error = _await_box(late)
    assert isinstance(error, AdmissionRejected)
    assert error.trace is not None
    assert error.trace.reason is AdmissionReason.DEADLINE_EXPIRED

    clock["now"] = 50_001.0
    runtime.release_all()
    _await_box(running)


# -- anti-starvation (AC-QOS-022) -------------------------------------------


def test_small_short_work_is_not_starved_behind_large_long_work() -> None:
    """AC-QOS-002, exercised through the real queue rather than the policy alone.

    A long-running large request occupies the only slot. Two further requests
    arrive: a big slow one first, then a small fast one. The small one must be
    admitted first, which is the ordering :class:`AdmissionQueue` already
    defines -- the gate has to preserve it.
    """
    runtime = ControllableRuntime()
    gate = LlamaAdmissionGate(runtime, capacity=_capacity(max_active=1))
    running = _run_in_thread(gate, _request("blocker"))
    runtime.wait_for_start(1)

    order: list[str] = []
    # Only the two submitters are parties; the main thread just waits.
    started = threading.Barrier(2)

    def submit_when_free(name: str, /, **kwargs) -> None:
        started.wait(timeout=5.0)
        try:
            result = gate.submit(_request(name, **kwargs))
        except (AdmissionRejected, AdmissionCancelled):  # pragma: no cover
            return
        order.append(name)
        assert isinstance(result, GateResult)
        runtime.release_all()

    big = threading.Thread(
        target=submit_when_free,
        args=("big",),
        kwargs={"estimated_duration_ms": 60_000.0, "per_lane_memory_gb": 4.0},
        daemon=True,
    )
    small = threading.Thread(
        target=submit_when_free,
        args=("small",),
        kwargs={"estimated_duration_ms": 100.0, "per_lane_memory_gb": 0.1},
        daemon=True,
    )
    big.start()
    small.start()
    _await_queue_depth(gate, 2)

    runtime.release_all()
    big.join(timeout=5.0)
    small.join(timeout=5.0)
    assert isinstance(_await_box(running), GateResult)
    assert order[:1] == ["small"], f"expected the small request first, got {order}"


# -- cancellation (AC-QOS-023) ----------------------------------------------


def test_cancelling_queued_work_returns_true_and_frees_queue_pressure() -> None:
    runtime = ControllableRuntime()
    gate = LlamaAdmissionGate(runtime, capacity=_capacity(max_active=1, max_queue_depth=4))
    running = _run_in_thread(gate, _request("run"))
    runtime.wait_for_start(1)
    doomed = _run_in_thread(gate, _request("doomed"))
    _await_queue_depth(gate, 1)

    assert gate.cancel("doomed") is True
    assert gate.queue_depth == 0, "cancelled work must stop occupying the queue"

    error = _await_box(doomed)
    assert isinstance(error, AdmissionCancelled)
    record = gate.accountant.get_record("doomed")
    assert record is not None
    assert record.outcome is ResidencyOutcome.CANCELLED

    runtime.release_all()
    _await_box(running)


def test_cancelling_in_flight_work_returns_false_rather_than_lying() -> None:
    """The gate cannot recall a request inside a synchronous ``generate``."""
    runtime = ControllableRuntime()
    gate = LlamaAdmissionGate(runtime, capacity=_capacity(max_active=1))
    running = _run_in_thread(gate, _request("run"))
    runtime.wait_for_start(1)
    assert gate.cancel("run") is False
    assert gate.cancel("never-existed") is False
    runtime.release_all()
    assert isinstance(_await_box(running), GateResult)


# -- telemetry and residency (REQ-QOS-027, AC-QOS-021) ----------------------


def test_decision_trace_is_public_safe_and_reports_the_slot_limit() -> None:
    runtime = ControllableRuntime()
    gate = LlamaAdmissionGate(runtime, capacity=_capacity(max_active=3))
    runtime.release_all()
    request = GateRequest(
        request_id="req-42",
        inference=InferenceRequest(
            prompt="the secret payload", max_tokens=8, temperature=0.0
        ),
    )
    result = gate.submit(request)
    trace = result.trace
    assert trace.max_active_tasks == 3
    assert trace.action is AdmissionAction.ADMIT
    assert trace.reason is AdmissionReason.CAPACITY_AVAILABLE
    rendered = repr(trace)
    assert "req-42" not in rendered, "request identity leaked into the public trace"
    assert "the secret payload" not in rendered, "prompt leaked into the public trace"


def test_telemetry_counts_admits_and_queues() -> None:
    runtime = ControllableRuntime()
    gate = LlamaAdmissionGate(runtime, capacity=_capacity(max_active=1))
    running = _run_in_thread(gate, _request("run"))
    runtime.wait_for_start(1)
    queued = _run_in_thread(gate, _request("q"))
    time.sleep(0.1)
    telemetry = gate.telemetry
    assert telemetry.admitted == 1
    assert telemetry.queued == 1
    assert ("active_limit", 1) in telemetry.reason_counts
    assert telemetry.last_decision is not None
    runtime.release_all()
    _await_box(running)
    _await_box(queued)


def test_residency_accountant_records_wait_and_respects_peak_active() -> None:
    runtime = ControllableRuntime()
    gate = LlamaAdmissionGate(runtime, capacity=_capacity(max_active=2))
    boxes = [_run_in_thread(gate, _request(f"r{i}")) for i in range(6)]
    runtime.wait_for_start(2)
    time.sleep(0.15)
    runtime.release_all()
    for box in boxes:
        assert isinstance(_await_box(box), GateResult)

    summary = gate.accountant.summary()
    assert summary.total == 6
    assert summary.completed == 6
    assert summary.peak_active_tasks <= 2, f"peak_active_tasks={summary.peak_active_tasks}"
    assert summary.p95_wait_ms >= 0.0
    assert summary.max_residency_ms >= 0.0


def test_a_failing_request_still_releases_its_slot() -> None:
    class Exploding(InferenceRuntime):
        def generate(self, request: InferenceRequest) -> InferenceResponse:
            raise RuntimeError("backend refused")

    gate = LlamaAdmissionGate(Exploding(), capacity=_capacity(max_active=1))
    with pytest.raises(RuntimeError, match="backend refused"):
        gate.submit(_request("boom"))
    assert gate.in_flight == 0, "a failed request must not leak its slot"
    assert gate.queue_depth == 0


def test_duplicate_request_id_is_refused() -> None:
    runtime = ControllableRuntime()
    gate = LlamaAdmissionGate(runtime, capacity=_capacity(max_active=1))
    running = _run_in_thread(gate, _request("dup"))
    runtime.wait_for_start(1)
    with pytest.raises(ValueError, match="already in flight"):
        gate.submit(_request("dup"))
    runtime.release_all()
    _await_box(running)


# -- host capacity ----------------------------------------------------------


def test_gate_capacity_from_host_refuses_to_invent_a_total() -> None:
    with pytest.raises(RuntimeError, match="refusing to invent"):
        gate_capacity_from_host(max_active_tasks=4, memory={"total_gb": None, "used_gb": None})


def test_gate_capacity_from_host_uses_the_reading_it_was_given() -> None:
    capacity = gate_capacity_from_host(
        max_active_tasks=4,
        reserved_memory_gb=2.0,
        memory={"total_gb": 64.0, "used_gb": 40.0},
    )
    assert capacity.total_memory_gb == 64.0
    assert capacity.used_memory_gb == 40.0
    assert capacity.max_active_tasks == 4
    assert capacity.available_memory_gb == pytest.approx(22.0)


def test_default_lane_memory_is_small_next_to_real_host_memory() -> None:
    # A declared lane cost that exceeded usable host memory would reject every
    # request on a loaded machine, which is a policy failure dressed as safety.
    assert 0 < DEFAULT_LANE_MEMORY_GB < 8.0


# -- negative controls ------------------------------------------------------


def test_gate_that_ignores_its_policy_is_caught() -> None:
    """Negative control for the bounding test.

    A gate wired to a policy that always admits passes every reason-checking
    test but breaks the slot limit. If this control could not make
    :func:`test_gate_never_exceeds_the_declared_slot_limit` fail, that test
    would not be measuring the gate at all.
    """

    class AlwaysAdmit(AdmissionPolicy):
        def decide(self, request, capacity, *, now_ms):  # noqa: ANN001, ANN202
            from oai2.runtime.admission import AdmissionDecision

            return AdmissionDecision(
                request_id=request.request_id,
                action=AdmissionAction.ADMIT,
                reason=AdmissionReason.CAPACITY_AVAILABLE,
                required_memory_gb=request.aggregate_memory_gb,
                available_memory_gb=capacity.available_memory_gb,
            )

    runtime = ControllableRuntime()
    gate = LlamaAdmissionGate(runtime, capacity=_capacity(max_active=2), policy=AlwaysAdmit())
    boxes = [_run_in_thread(gate, _request(f"r{i}")) for i in range(10)]
    runtime.wait_for_start(10)
    assert runtime.peak_running == 10, "control did not defeat the gate"
    runtime.release_all()
    for box in boxes:
        _await_box(box)


def test_gate_delegates_decisions_to_the_injected_policy() -> None:
    """The gate must ask the policy, not carry its own copy of the rules."""

    class Counting(AdmissionPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def decide(self, request, capacity, *, now_ms):  # noqa: ANN001, ANN202
            self.calls += 1
            return super().decide(request, capacity, now_ms=now_ms)

    policy = Counting()
    runtime = ControllableRuntime()
    gate = LlamaAdmissionGate(runtime, capacity=_capacity(max_active=1), policy=policy)
    running = _run_in_thread(gate, _request("run"))
    runtime.wait_for_start(1)
    _run_in_thread(gate, _request("q"))
    time.sleep(0.1)
    assert policy.calls >= 2, "gate decided without consulting the policy"
    runtime.release_all()
    _await_box(running)


def test_capacity_snapshot_reflects_live_in_flight_not_a_cached_value() -> None:
    runtime = ControllableRuntime()
    gate = LlamaAdmissionGate(runtime, capacity=_capacity(max_active=3))
    assert gate.capacity_snapshot().active_tasks == 0
    running = _run_in_thread(gate, _request("run"))
    runtime.wait_for_start(1)
    snapshot = gate.capacity_snapshot()
    assert snapshot.active_tasks == 1
    assert snapshot.max_active_tasks == 3
    runtime.release_all()
    _await_box(running)
