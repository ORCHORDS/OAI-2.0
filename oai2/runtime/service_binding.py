"""Request-surface binding for the WI-INF-001 safe-batching scheduler.

This module owns the session/inference protocol routes: it composes the
health/readiness application from :mod:`oai2.runtime.local_service` with the
:class:`SafeBatchScheduler` and the :class:`IsolatedSessionRegistry` so that
concurrent client sessions can submit compatibility-keyed work, drain
exact-compatibility batches, cancel queued work without corrupting other
sessions, and observe batch-size / queue-depth / queue-latency telemetry
(REQ-INF-011 through REQ-INF-016).

Latency counters live in :class:`BatchSurfaceMetrics` rather than in
``SchedulerMetrics`` because the scheduler metric snapshot is pinned to its
five queue/scheduling fields; this binding layer is the component that
observes execution-side time.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import NoReturn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ..core import Status
from .inference import InferenceRequest, InferenceRuntime, PlaceholderRuntime
from .local_service import create_local_service_app
from .scheduler import SafeBatchScheduler, ScheduledRequest, SessionCompatibilityKey
from .service import ServiceCompatibility, ServiceLifecycle
from .service_security import AccessPolicy, IsolatedSessionRegistry


def _wall_clock_ms() -> float:
    return time.time() * 1000.0


def _parse_note_ms(notes: list[str], key: str) -> float | None:
    prefix = f"{key}="
    for note in notes:
        if note.startswith(prefix):
            try:
                return float(note[len(prefix):]) * 1000.0
            except ValueError:
                return None
    return None


def _parse_note_float(notes: list[str], key: str) -> float | None:
    prefix = f"{key}="
    for note in notes:
        if note.startswith(prefix):
            try:
                return float(note[len(prefix):])
            except ValueError:
                return None
    return None


def _parse_note_int(notes: list[str], key: str) -> int | None:
    prefix = f"{key}="
    for note in notes:
        if note.startswith(prefix):
            try:
                return int(note[len(prefix):])
            except ValueError:
                return None
    return None


@dataclass(slots=True, frozen=True)
class BatchSurfaceMetrics:
    """Batch, queue, and latency telemetry for the bound request surface."""

    queue_depth: int
    queued_sessions: int
    cancelled_requests: int
    batches_emitted: int
    requests_emitted: int
    last_batch_size: int
    # Averages are None when the divisor was zero -- no batch emitted, no
    # request executed -- because "the mean of no observations" is not 0.0,
    # it is undefined. Reporting 0.0 produced a surface that had never served
    # anything describing itself as a perfectly empty queue.
    mean_batch_size: float | None
    # None until the first request completes a queue wait. `last_*` fields
    # describe the most recent observation, and there has not been one.
    last_queue_wait_ms: float | None
    mean_queue_wait_ms: float | None
    # None until at least one request has waited.
    max_queue_wait_ms: float | None
    active_sessions: int


@dataclass(slots=True, frozen=True)
class BatchExecutionResult:
    """One executed request from a drained batch."""

    request_id: str
    session_id: str
    text: str
    queue_wait_ms: float
    status: Status = Status.IMPLEMENTED
    # Per-request timing fields populated when the runtime records them
    # in ``InferenceResponse.notes`` (the MLX hot runtime writes
    # ``prefill_seconds=...`` / ``decode_seconds=...`` / ``decode_tps=...``).
    # ``None`` means the runtime did not report them.
    prefill_ms: float | None = None
    decode_ms: float | None = None
    decode_tokens_per_second: float | None = None
    generated_tokens: int | None = None
    prompt_tokens: int | None = None


class BatchInferenceSurface:
    """Multi-session batching request surface bound to one lifecycle and scheduler.

    The surface enforces per-client session ownership, defaults the
    security-context component of the compatibility key to the client identity
    so distinct client identities never coalesce unsafely, and records
    queue-latency telemetry with an injected monotonic clock.
    """

    def __init__(
        self,
        *,
        lifecycle: ServiceLifecycle,
        scheduler: SafeBatchScheduler,
        registry: IsolatedSessionRegistry,
        runtime: InferenceRuntime | None = None,
        max_batch_size: int = 8,
        clock: Callable[[], float] = _wall_clock_ms,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        self._lifecycle = lifecycle
        self._scheduler = scheduler
        self._registry = registry
        self._runtime: InferenceRuntime = runtime if runtime is not None else PlaceholderRuntime()
        self._max_batch_size = max_batch_size
        self._clock = clock
        self._prompts: dict[str, str] = {}
        self._session_requests: dict[str, str] = {}
        self._request_max_tokens: dict[str, int] = {}
        self._last_batch_size = 0
        self._batches_executed = 0
        self._requests_executed = 0
        self._total_batch_size = 0
        self._last_queue_wait_ms: float | None = None
        self._total_queue_wait_ms = 0.0
        self._max_queue_wait_ms: float | None = None

    def open_session(self, *, client_id: str, session_id: str) -> None:
        """Register a session in the registry and the lifecycle, or roll back."""
        self._registry.create(client_id=client_id, session_id=session_id)
        try:
            self._lifecycle.session_started(session_id)
        except BaseException:
            self._registry.remove(client_id=client_id, session_id=session_id)
            raise

    def close_session(self, *, client_id: str, session_id: str) -> bool:
        """Cancel the session's queued work and retire the session."""
        self._registry.require_owned(client_id=client_id, session_id=session_id)
        cancelled = self._scheduler.cancel_session(session_id)
        request_id = self._session_requests.pop(session_id, None)
        if request_id is not None:
            self._prompts.pop(request_id, None)
            self._request_max_tokens.pop(request_id, None)
        self._registry.remove(client_id=client_id, session_id=session_id)
        self._lifecycle.session_finished(session_id)
        return cancelled

    def submit(
        self,
        *,
        client_id: str,
        session_id: str,
        request_id: str,
        prompt: str,
        model_id: str,
        tokenizer_version: str,
        prefix_digest: str,
        tool_schema_version: str,
        world_state_version: str,
        security_context: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Enqueue one request under its exact session-compatibility key."""
        self._registry.require_owned(client_id=client_id, session_id=session_id)
        key = SessionCompatibilityKey(
            model_id=model_id,
            tokenizer_version=tokenizer_version,
            prefix_digest=prefix_digest,
            tool_schema_version=tool_schema_version,
            world_state_version=world_state_version,
            security_context=security_context if security_context is not None else client_id,
        )
        request = ScheduledRequest(
            request_id=request_id,
            session_id=session_id,
            compatibility=key,
            enqueued_at_ms=self._clock(),
        )
        self._scheduler.enqueue(request)
        self._prompts[request_id] = prompt
        self._session_requests[session_id] = request_id
        if max_tokens is not None:
            self._request_max_tokens[request_id] = max_tokens

    def drain(self) -> tuple[BatchExecutionResult, ...]:
        """Pop one compatibility-exact batch, execute it, and record telemetry."""
        plan = self._scheduler.pop_batch(max_batch_size=self._max_batch_size)
        if plan is None:
            return ()
        now_ms = self._clock()
        results: list[BatchExecutionResult] = []
        for scheduled in plan.requests:
            prompt = self._prompts.pop(scheduled.request_id, "")
            max_tokens = self._request_max_tokens.pop(scheduled.request_id, None)
            if max_tokens is None:
                request = InferenceRequest(
                    prompt=prompt,
                    prefix_digest=scheduled.compatibility.prefix_digest,
                )
            else:
                request = InferenceRequest(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    prefix_digest=scheduled.compatibility.prefix_digest,
                )
            response = self._runtime.generate(request)
            notes = list(response.notes)
            results.append(
                BatchExecutionResult(
                    request_id=scheduled.request_id,
                    session_id=scheduled.session_id,
                    text=response.text,
                    queue_wait_ms=now_ms - scheduled.enqueued_at_ms,
                    status=Status.IMPLEMENTED,
                    prefill_ms=_parse_note_ms(notes, "prefill_seconds"),
                    decode_ms=_parse_note_ms(notes, "decode_seconds"),
                    decode_tokens_per_second=_parse_note_float(notes, "decode_tps"),
                    generated_tokens=response.tokens or None,
                    prompt_tokens=_parse_note_int(notes, "prompt_tokens"),
                )
            )
            if self._session_requests.get(scheduled.session_id) == scheduled.request_id:
                self._session_requests.pop(scheduled.session_id, None)
        batch_size = len(results)
        waits = [result.queue_wait_ms for result in results]
        self._last_batch_size = batch_size
        self._batches_executed += 1
        self._requests_executed += batch_size
        self._total_batch_size += batch_size
        self._total_queue_wait_ms += sum(waits)
        self._last_queue_wait_ms = waits[-1]
        # Seeded from the batch, not from a 0.0 sentinel. `max([None, *waits])`
        # would not compare, and folding a 0.0 in would have made the maximum
        # report 0.0 for a batch whose requests all waited -- the one value
        # guaranteed to be impossible.
        self._max_queue_wait_ms = (
            max(waits) if self._max_queue_wait_ms is None
            else max(self._max_queue_wait_ms, max(waits))
        )
        return tuple(results)

    def metrics(self) -> BatchSurfaceMetrics:
        """Compose scheduler queue telemetry with surface latency telemetry."""
        scheduler_metrics = self._scheduler.metrics
        batches = self._batches_executed
        executed = self._requests_executed
        return BatchSurfaceMetrics(
            queue_depth=scheduler_metrics.queue_depth,
            queued_sessions=scheduler_metrics.queued_sessions,
            cancelled_requests=scheduler_metrics.cancelled_requests,
            batches_emitted=scheduler_metrics.batches_emitted,
            requests_emitted=scheduler_metrics.requests_emitted,
            last_batch_size=self._last_batch_size,
            mean_batch_size=(self._total_batch_size / batches) if batches else None,
            last_queue_wait_ms=self._last_queue_wait_ms,
            mean_queue_wait_ms=(
                (self._total_queue_wait_ms / executed) if executed else None
            ),
            max_queue_wait_ms=self._max_queue_wait_ms,
            active_sessions=self._lifecycle.health.active_sessions,
        )


class SessionOpenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class InferenceSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    tokenizer_version: str = Field(min_length=1)
    prefix_digest: str = Field(min_length=1)
    tool_schema_version: str = Field(min_length=1)
    world_state_version: str = Field(min_length=1)
    security_context: str | None = None
    max_tokens: int | None = None


def _fail(exc: Exception) -> NoReturn:
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc


def create_batch_inference_app(
    *,
    lifecycle: ServiceLifecycle,
    actual_compatibility: ServiceCompatibility,
    access_policy: AccessPolicy,
    scheduler: SafeBatchScheduler,
    registry: IsolatedSessionRegistry,
    runtime: InferenceRuntime | None = None,
    max_batch_size: int = 8,
    clock: Callable[[], float] = _wall_clock_ms,
) -> FastAPI:
    """Bind the health/readiness slice to the batching scheduler and expose the routes.

    The returned application inherits the health application's global bearer
    dependency, so every protocol route below is protected by the same
    ``AccessPolicy`` instance.
    """
    app = create_local_service_app(
        lifecycle=lifecycle,
        actual_compatibility=actual_compatibility,
        access_policy=access_policy,
    )
    surface = BatchInferenceSurface(
        lifecycle=lifecycle,
        scheduler=scheduler,
        registry=registry,
        runtime=runtime,
        max_batch_size=max_batch_size,
        clock=clock,
    )
    app.state.batch_surface = surface

    @app.post("/v1/sessions", status_code=201)
    def open_session(body: SessionOpenRequest) -> dict[str, str]:
        try:
            surface.open_session(client_id=body.client_id, session_id=body.session_id)
        except Exception as exc:
            _fail(exc)
        return {"client_id": body.client_id, "session_id": body.session_id}

    @app.delete("/v1/sessions/{session_id}")
    def close_session(session_id: str, client_id: str) -> dict[str, bool]:
        try:
            closed = surface.close_session(client_id=client_id, session_id=session_id)
        except Exception as exc:
            _fail(exc)
        return {"closed": closed}

    @app.post("/v1/inference", status_code=202)
    def submit_request(body: InferenceSubmission) -> dict[str, str]:
        try:
            surface.submit(
                client_id=body.client_id,
                session_id=body.session_id,
                request_id=body.request_id,
                prompt=body.prompt,
                model_id=body.model_id,
                tokenizer_version=body.tokenizer_version,
                prefix_digest=body.prefix_digest,
                tool_schema_version=body.tool_schema_version,
                world_state_version=body.world_state_version,
                security_context=body.security_context,
                max_tokens=body.max_tokens,
            )
        except Exception as exc:
            _fail(exc)
        return {"request_id": body.request_id, "session_id": body.session_id}

    @app.post("/v1/inference/drain")
    def drain_batch() -> dict[str, list[dict[str, str | float | int | None]]]:
        results = surface.drain()
        return {
            "results": [
                {
                    "request_id": result.request_id,
                    "session_id": result.session_id,
                    "text": result.text,
                    "queue_wait_ms": result.queue_wait_ms,
                    "status": result.status.value,
                    "prefill_ms": result.prefill_ms,
                    "decode_ms": result.decode_ms,
                    "decode_tokens_per_second": result.decode_tokens_per_second,
                    "generated_tokens": result.generated_tokens,
                    "prompt_tokens": result.prompt_tokens,
                }
                for result in results
            ]
        }

    @app.get("/v1/batches/metrics")
    def batch_metrics() -> dict[str, int | float]:
        return asdict(surface.metrics())

    return app


__all__ = [
    "BatchExecutionResult",
    "BatchInferenceSurface",
    "BatchSurfaceMetrics",
    "create_batch_inference_app",
]
