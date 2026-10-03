"""Concurrent-session tests for the WI-INF-001 request-surface binding."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from oai2.core import Status
from oai2.runtime.inference import (
    InferenceRequest,
    InferenceResponse,
    InferenceRuntime,
    PlaceholderRuntime,
)
from oai2.runtime.scheduler import SafeBatchScheduler
from oai2.runtime.service import ServiceCompatibility, ServiceLifecycle, ServiceState
from oai2.runtime.service_binding import BatchInferenceSurface, create_batch_inference_app
from oai2.runtime.service_security import AccessPolicy, IsolatedSessionRegistry


def _compat() -> ServiceCompatibility:
    return ServiceCompatibility(
        config_digest="cfg-v1",
        model_id="model-a",
        schema_version="schema-v1",
    )


def _key_kwargs(**overrides: str) -> dict[str, str]:
    kwargs: dict[str, str] = {
        "model_id": "model-a",
        "tokenizer_version": "tok-v1",
        "prefix_digest": "prefix-v1",
        "tool_schema_version": "tools-v1",
        "world_state_version": "world-v1",
    }
    kwargs.update(overrides)
    return kwargs


class _ScriptClock:
    """Deterministic clock returning scripted values, repeating the last one."""

    def __init__(self, *values: float) -> None:
        self._values = list(values)
        self._index = 0

    def __call__(self) -> float:
        if not self._values:
            return 0.0
        value = self._values[min(self._index, len(self._values) - 1)]
        self._index += 1
        return value


def _surface(clock: _ScriptClock | None = None) -> BatchInferenceSurface:
    lifecycle = ServiceLifecycle(expected=_compat())
    lifecycle.start(_compat())
    return BatchInferenceSurface(
        lifecycle=lifecycle,
        scheduler=SafeBatchScheduler(),
        registry=IsolatedSessionRegistry(),
        clock=clock if clock is not None else _ScriptClock(),
    )


class _RecordingRuntime(InferenceRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[InferenceRequest] = []

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.requests.append(request)
        return InferenceResponse(text="ok", tokens=1, elapsed_ms=1.0, device="test")


def test_submission_max_tokens_flows_to_runtime() -> None:
    runtime = _RecordingRuntime()
    lifecycle = ServiceLifecycle(expected=_compat())
    lifecycle.start(_compat())
    surface = BatchInferenceSurface(
        lifecycle=lifecycle,
        scheduler=SafeBatchScheduler(),
        registry=IsolatedSessionRegistry(),
        runtime=runtime,
    )
    surface.open_session(client_id="c1", session_id="s-1")
    surface.submit(
        client_id="c1", session_id="s-1", request_id="r-1", prompt="p", max_tokens=5, **_key_kwargs()
    )
    surface.drain()
    assert runtime.requests[0].max_tokens == 5
    assert runtime.requests[0].prefix_digest == "prefix-v1"


def test_cross_client_access_is_rejected() -> None:
    surface = _surface()
    surface.open_session(client_id="alice", session_id="s-a1")
    surface.open_session(client_id="bob", session_id="s-b1")
    surface.submit(
        client_id="alice", session_id="s-a1", request_id="r-a1", prompt="hello", **_key_kwargs()
    )

    with pytest.raises(PermissionError):
        surface.submit(client_id="bob", session_id="s-a1", request_id="r-x", prompt="x", **_key_kwargs())
    with pytest.raises(PermissionError):
        surface.close_session(client_id="bob", session_id="s-a1")


def test_security_context_defaults_to_client_identity() -> None:
    surface = _surface()
    surface.open_session(client_id="alice", session_id="s-a")
    surface.open_session(client_id="bob", session_id="s-b")
    surface.submit(client_id="alice", session_id="s-a", request_id="r-a", prompt="a", **_key_kwargs())
    surface.submit(client_id="bob", session_id="s-b", request_id="r-b", prompt="bb", **_key_kwargs())

    first = surface.drain()
    assert len(first) == 1
    assert first[0].request_id == "r-a"
    assert surface.metrics().queue_depth == 1

    second = surface.drain()
    assert len(second) == 1
    assert second[0].request_id == "r-b"
    assert first[0].text != second[0].text
    metrics = surface.metrics()
    assert metrics.queue_depth == 0
    assert metrics.active_sessions == 2


def test_batching_coalesces_compatible_sessions() -> None:
    surface = _surface()
    surface.open_session(client_id="c1", session_id="s-1")
    surface.open_session(client_id="c2", session_id="s-2")
    for client, session, request in (("c1", "s-1", "r-1"), ("c2", "s-2", "r-2")):
        surface.submit(
            client_id=client,
            session_id=session,
            request_id=request,
            prompt=f"prompt-{request}",
            security_context="tenant-1",
            **_key_kwargs(),
        )

    results = surface.drain()
    assert len(results) == 2
    metrics = surface.metrics()
    assert metrics.batches_emitted == 1
    assert metrics.requests_emitted == 2
    assert metrics.last_batch_size == 2
    assert metrics.mean_batch_size == 2.0
    assert metrics.queue_depth == 0


def test_incompatible_security_contexts_do_not_coalesce() -> None:
    surface = _surface()
    surface.open_session(client_id="c1", session_id="s-1")
    surface.open_session(client_id="c2", session_id="s-2")
    surface.submit(
        client_id="c1",
        session_id="s-1",
        request_id="r-1",
        prompt="a",
        security_context="tenant-1",
        **_key_kwargs(),
    )
    surface.submit(
        client_id="c2",
        session_id="s-2",
        request_id="r-2",
        prompt="b",
        security_context="tenant-2",
        **_key_kwargs(),
    )

    first = surface.drain()
    assert len(first) == 1
    assert surface.metrics().queue_depth == 1
    second = surface.drain()
    assert len(second) == 1
    assert first[0].request_id != second[0].request_id


def test_cancellation_removes_only_target_session_work() -> None:
    surface = _surface()
    surface.open_session(client_id="c1", session_id="s-a")
    surface.open_session(client_id="c2", session_id="s-b")
    surface.open_session(client_id="c3", session_id="s-c")
    for client, session, request in (("c1", "s-a", "r-a"), ("c2", "s-b", "r-b"), ("c3", "s-c", "r-c")):
        surface.submit(
            client_id=client,
            session_id=session,
            request_id=request,
            prompt=f"p-{request}",
            security_context="shared",
            **_key_kwargs(),
        )

    assert surface.close_session(client_id="c2", session_id="s-b") is True
    metrics = surface.metrics()
    assert metrics.cancelled_requests == 1
    assert metrics.queue_depth == 2

    results = surface.drain()
    assert {result.request_id for result in results} == {"r-a", "r-c"}

    with pytest.raises(PermissionError):
        surface.close_session(client_id="c2", session_id="s-b")


def test_queue_latency_is_measured_deterministically() -> None:
    clock = _ScriptClock(1000.0, 1010.0, 1050.0)
    surface = _surface(clock)
    surface.open_session(client_id="c1", session_id="s-a")
    surface.open_session(client_id="c2", session_id="s-b")
    surface.submit(
        client_id="c1",
        session_id="s-a",
        request_id="r-a",
        prompt="a",
        security_context="shared",
        **_key_kwargs(),
    )
    surface.submit(
        client_id="c2",
        session_id="s-b",
        request_id="r-b",
        prompt="b",
        security_context="shared",
        **_key_kwargs(),
    )

    results = surface.drain()
    assert [result.queue_wait_ms for result in results] == [50.0, 40.0]
    metrics = surface.metrics()
    assert metrics.mean_queue_wait_ms == 45.0
    assert metrics.max_queue_wait_ms == 50.0
    assert metrics.last_queue_wait_ms == 40.0


def test_protocol_routes_require_bearer_and_bind_scheduler() -> None:
    lifecycle = ServiceLifecycle(expected=_compat())
    app = create_batch_inference_app(
        lifecycle=lifecycle,
        actual_compatibility=_compat(),
        access_policy=AccessPolicy(bind_host="0.0.0.0", bearer_token="secret-token"),
        scheduler=SafeBatchScheduler(),
        registry=IsolatedSessionRegistry(),
    )
    headers = {"Authorization": "Bearer secret-token"}

    with TestClient(app) as client:
        assert client.get("/healthz", headers=headers).status_code == 200
        assert client.post("/v1/sessions", json={"client_id": "c1", "session_id": "s-1"}).status_code == 401
        opened = client.post(
            "/v1/sessions", json={"client_id": "c1", "session_id": "s-1"}, headers=headers
        )
        assert opened.status_code == 201
        duplicate = client.post(
            "/v1/sessions", json={"client_id": "c2", "session_id": "s-1"}, headers=headers
        )
        assert duplicate.status_code == 409

        submission = {
            "client_id": "c1",
            "session_id": "s-1",
            "request_id": "r-1",
            "prompt": "hello",
            **_key_kwargs(),
        }
        missing = client.post("/v1/inference", json={**submission, "session_id": "s-none"}, headers=headers)
        assert missing.status_code == 403
        assert client.post("/v1/inference", json=submission, headers=headers).status_code == 202

        drained = client.post("/v1/inference/drain", headers=headers)
        assert drained.status_code == 200
        payload = drained.json()["results"]
        assert len(payload) == 1
        assert payload[0]["request_id"] == "r-1"
        # This used to assert "IMPLEMENTED", which was the defect: the
        # assertion two lines below proves the runtime is a PlaceholderRuntime
        # (its literal output format contains "prompt_len="), and no weights
        # were loaded. `drain()` hardcoded Status.IMPLEMENTED and discarded
        # `InferenceResponse.status`, so a placeholder result was reported to
        # the client as implemented. The two assertions together now say what
        # is true: this is a placeholder run, and it is labelled as one.
        assert payload[0]["status"] == "EXPERIMENTAL"
        assert "prompt_len=5" in payload[0]["text"]

        metrics = client.get("/v1/batches/metrics", headers=headers).json()
        assert metrics["batches_emitted"] == 1
        assert metrics["requests_emitted"] == 1

        queued = {**submission, "request_id": "r-2"}
        assert client.post("/v1/inference", json=queued, headers=headers).status_code == 202
        closed = client.delete("/v1/sessions/s-1", params={"client_id": "c1"}, headers=headers)
        assert closed.status_code == 200
        assert closed.json()["closed"] is True

    assert lifecycle.state is ServiceState.STOPPED


class TestDrainCarriesTheRuntimesOwnStatus:
    """`drain()` must forward `InferenceResponse.status`, not replace it.

    `drain()` hardcoded `Status.IMPLEMENTED`, discarding the field the
    runtime populated. The visible symptom was a `PlaceholderRuntime` -- the
    default when `runtime=None`, and the runtime that loads no weights --
    being reported to the client as `"implemented"`.

    Asserted with a runtime that reports a status *different from* the one the
    placeholder does, so this proves the value is carried through in both
    directions. A test using only the placeholder would still pass if
    `drain()` had swapped in a hardcoded EXPERIMENTAL, which would be the
    same defect wearing a truthful hat.
    """

    @dataclass
    class _StubRuntime(InferenceRuntime):
        reports: Status

        def generate(self, request: InferenceRequest) -> InferenceResponse:
            return InferenceResponse(
                text="stub output",
                tokens=2,
                elapsed_ms=1.0,
                device="cpu",
                status=self.reports,
            )

    @staticmethod
    def _drained_status(reports: Status) -> Status:
        lifecycle = ServiceLifecycle(expected=_compat())
        lifecycle.start(_compat())
        surface = BatchInferenceSurface(
            lifecycle=lifecycle,
            scheduler=SafeBatchScheduler(),
            registry=IsolatedSessionRegistry(),
            runtime=TestDrainCarriesTheRuntimesOwnStatus._StubRuntime(reports),
            clock=_ScriptClock(),
        )
        surface.open_session(client_id="c1", session_id="s-1")
        surface.submit(
            client_id="c1",
            session_id="s-1",
            request_id="r-1",
            prompt="hello",
            **_key_kwargs(),
        )
        results = surface.drain()
        assert len(results) == 1
        return results[0].status

    @pytest.mark.parametrize("status", [Status.PROPOSED, Status.EXPERIMENTAL, Status.IMPLEMENTED])
    def test_the_runtimes_status_is_forwarded_verbatim(self, status: Status) -> None:
        assert self._drained_status(status) is status

    def test_a_placeholder_is_not_reported_as_implemented(self) -> None:
        """The original defect, stated directly.

        `PlaceholderRuntime` is what `create_batch_inference_app` installs when
        no runtime is supplied, and it reports EXPERIMENTAL with the note
        "placeholder runtime -- no model loaded". Reporting that as
        IMPLEMENTED told the client a model had run when none had.
        """
        lifecycle = ServiceLifecycle(expected=_compat())
        lifecycle.start(_compat())
        surface = BatchInferenceSurface(
            lifecycle=lifecycle,
            scheduler=SafeBatchScheduler(),
            registry=IsolatedSessionRegistry(),
            clock=_ScriptClock(),
        )
        assert isinstance(surface._runtime, PlaceholderRuntime)  # noqa: SLF001
        surface.open_session(client_id="c1", session_id="s-1")
        surface.submit(
            client_id="c1",
            session_id="s-1",
            request_id="r-1",
            prompt="hello",
            **_key_kwargs(),
        )
        results = surface.drain()
        assert results[0].status is Status.EXPERIMENTAL
        assert results[0].status is not Status.IMPLEMENTED
