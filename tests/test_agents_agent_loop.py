"""Tests for the live agent loop with a stubbed runtime.

The :class:`AgentLoop` is exercised against a deterministic
:class:`StubRuntime` so we can verify the multi-step tool-use behavior
without touching the network. The stub records every request and
returns scripted responses (text or tool_calls) so the loop is fully
testable offline.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oai2.agents import (
    AgentLoop,
    default_dispatch_policy,
    default_system_prompt,
    default_tool_definitions,
)
from oai2.core import KnowledgeId, Status
from oai2.knowledge import (
    KnowledgeObject,
    KnowledgeStore,
    RetrievalRequest,
    RetrievalResult,
)
from oai2.protocols import ToolResult
from oai2.runtime import (
    InferenceRequest,
    InferenceResponse,
    InferenceRuntime,
    PlaceholderRuntime,
)
from oai2.tools import DispatchPolicy

# ---------------------------------------------------------------------------
# Stub runtime
# ---------------------------------------------------------------------------


@dataclass
class _Script:
    """One scripted response in the runtime stub."""

    text: str = ""
    tool_calls: tuple[dict[str, Any], ...] = ()
    finish_reason: str = "stop"


@dataclass
class _StubRuntime(InferenceRuntime):
    """Records every request and returns scripted responses in order."""

    script: list[_Script] = field(default_factory=list)
    requests: list[InferenceRequest] = field(default_factory=list)
    _cursor: int = 0

    STATUS = Status.EXPERIMENTAL

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.requests.append(request)
        if self._cursor >= len(self.script):
            return InferenceResponse(
                text="[stub exhausted]",
                tokens=1,
                elapsed_ms=0.0,
                device="stub",
                status=Status.EXPERIMENTAL,
                finish_reason="stop",
            )
        s = self.script[self._cursor]
        self._cursor += 1
        return InferenceResponse(
            text=s.text,
            tokens=max(1, len(s.text.split())),
            elapsed_ms=1.0,
            device="stub",
            status=Status.EXPERIMENTAL,
            finish_reason=s.finish_reason,
            tool_calls=s.tool_calls,
        )


def _openai_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Render an OpenAI-style ``tool_call`` dict (as the gateway returns it)."""
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_system_prompt_mentions_all_six_tools() -> None:
    s = default_system_prompt()
    for tool in ("Read", "Edit", "Write", "Bash", "Glob", "Grep"):
        assert tool in s


def test_default_dispatch_policy_admits_all_six_capabilities() -> None:
    p = default_dispatch_policy()
    for td in default_tool_definitions():
        assert td.capability in p.allow_capabilities or not p.allow_capabilities


def test_agent_loop_terminates_on_text_only_response() -> None:
    runtime = _StubRuntime(script=[_Script(text="all done", finish_reason="stop")])
    loop = AgentLoop(runtime=runtime, max_steps=4)
    run = loop.run("hi")
    assert run.final_text == "all done"
    assert run.finished_reason == "stop"
    assert run.total_tool_calls == 0
    assert len(run.steps) == 1


def test_agent_loop_executes_tool_call_and_loops_until_stop(tmp_path: Path) -> None:
    target = tmp_path / "data.txt"
    target.write_text("payload\n", encoding="utf-8")
    # The default policy scopes by EXACT path membership, so admit the
    # exact file path the agent will read.
    policy = DispatchPolicy(
        allow_capabilities=frozenset({"fs.read", "fs.write", "fs.list", "shell.exec"}),
        deny_capabilities=frozenset(),
        resource_scopes=frozenset({str(target)}),
        budget_calls=8,
        high_impact_approved=True,
    )
    runtime = _StubRuntime(
        script=[
            _Script(
                text="",
                tool_calls=(
                    _openai_call("c1", "Read", {"path": str(target)}),
                ),
                finish_reason="tool_calls",
            ),
            _Script(text="got the file", finish_reason="stop"),
        ]
    )
    loop = AgentLoop(runtime=runtime, max_steps=4, cwd=tmp_path, policy=policy)
    run = loop.run("read the file")
    assert run.final_text == "got the file"
    assert run.total_tool_calls == 1
    assert run.finished_reason == "stop"
    # Two model turns (one tool-call, one final).
    assert len(run.steps) == 2
    # The second request must carry the tool result message.
    second_request = runtime.requests[1]
    roles = [m.get("role") for m in second_request.messages]
    assert roles == ["system", "user", "assistant", "tool"]
    tool_msg = second_request.messages[3]
    assert tool_msg.get("tool_call_id") == "c1"
    assert "payload" in tool_msg.get("content", "")


def test_agent_loop_sends_tools_in_wire_shape(tmp_path: Path) -> None:
    runtime = _StubRuntime(script=[_Script(text="ok", finish_reason="stop")])
    loop = AgentLoop(runtime=runtime, cwd=tmp_path)
    loop.run("hi")
    request = runtime.requests[0]
    # ``tools`` must be a non-empty list of OpenAI-style function entries.
    assert isinstance(request.tools, list) and request.tools
    first = request.tools[0]
    assert first["type"] == "function"
    assert "function" in first
    # The system message should be the host-side default unless overridden.
    assert request.messages[0]["role"] == "system"
    assert "Read" in request.messages[0]["content"]


def test_agent_loop_tool_choice_auto_is_passed() -> None:
    runtime = _StubRuntime(script=[_Script(text="ok", finish_reason="stop")])
    AgentLoop(runtime=runtime).run("hi")
    assert runtime.requests[0].tool_choice == "auto"


def test_agent_loop_dispatch_failure_is_recorded_not_raised(tmp_path: Path) -> None:
    # The dispatcher only admits capabilities in the allow list. Mark
    # the policy empty so Read is denied; the loop should record the
    # dispatch failure as a tool_result error and continue.
    runtime = _StubRuntime(
        script=[
            _Script(
                text="",
                tool_calls=(
                    _openai_call("c1", "Read", {"path": str(tmp_path / "missing.txt")}),
                ),
                finish_reason="tool_calls",
            ),
            _Script(text="done", finish_reason="stop"),
        ]
    )
    policy = DispatchPolicy(
        allow_capabilities=frozenset(),
        deny_capabilities=frozenset(),
        resource_scopes=frozenset(),
        budget_calls=4,
        high_impact_approved=False,
    )
    loop = AgentLoop(
        runtime=runtime,
        cwd=tmp_path,
        policy=policy,
        max_steps=4,
    )
    run = loop.run("read")
    assert run.final_text == "done"
    assert run.total_tool_calls == 1
    last_step = run.steps[-2]
    assert last_step.tool_results[0].ok is False
    assert "dispatch" in (last_step.tool_results[0].error or "")


def test_agent_loop_respects_max_steps(tmp_path: Path) -> None:
    # Always returns a tool_call; loop must give up after max_steps.
    script = [
        _Script(
            text="",
            tool_calls=(
                _openai_call("c1", "Read", {"path": str(tmp_path / "x.txt")}),
            ),
            finish_reason="tool_calls",
        )
    ] * 6
    runtime = _StubRuntime(script=script)
    loop = AgentLoop(runtime=runtime, cwd=tmp_path, max_steps=3)
    run = loop.run("loop forever")
    # We expect max_steps turns even though none terminated.
    assert len(run.steps) == 3


def test_placeholder_runtime_falls_through_cleanly() -> None:
    """Offline path: placeholder should yield a text response with no tools."""
    loop = AgentLoop(runtime=PlaceholderRuntime(), max_steps=2)
    run = loop.run("hello")
    assert run.total_tool_calls == 0
    assert "[placeholder" in run.final_text


def test_agent_loop_messages_history_includes_assistant_tool_calls(tmp_path: Path) -> None:
    runtime = _StubRuntime(
        script=[
            _Script(
                text="",
                tool_calls=(
                    _openai_call("c1", "Glob", {"pattern": "*.py"}),
                ),
                finish_reason="tool_calls",
            ),
            _Script(text="ok", finish_reason="stop"),
        ]
    )
    loop = AgentLoop(runtime=runtime, cwd=tmp_path, max_steps=4)
    loop.run("list")
    second = runtime.requests[1]
    assistant_msg = second.messages[2]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["tool_calls"]  # non-empty
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "Glob"


# ---------------------------------------------------------------------------
# Gate 5 budget boundary
# ---------------------------------------------------------------------------


def _always_call_script(target: Path, count: int) -> list[_Script]:
    """``count`` turns that each emit one in-scope ``Read`` call."""
    return [
        _Script(
            text="",
            tool_calls=(_openai_call(f"c{i}", "Read", {"path": str(target)}),),
            finish_reason="tool_calls",
        )
        for i in range(count)
    ]


def _budget_run(tmp_path: Path, budget: int) -> tuple[int, list[str]]:
    """Run a loop with ``budget`` and report (calls executed, refusal errors)."""
    target = tmp_path / "data.txt"
    target.write_text("payload\n", encoding="utf-8")
    policy = DispatchPolicy(
        allow_capabilities=frozenset({"fs.read", "fs.write", "fs.list", "shell.exec"}),
        deny_capabilities=frozenset(),
        resource_scopes=frozenset({str(target)}),
        budget_calls=budget,
        high_impact_approved=True,
    )
    executed: list[str] = []

    def _executor(call: Any) -> ToolResult:
        executed.append(call.id)
        return ToolResult(call_id=call.id, ok=True, output="payload")

    # Far more turns than any budget under test, so the BUDGET is what stops
    # the loop rather than the script or max_steps running out.
    runtime = _StubRuntime(script=_always_call_script(target, 60))
    loop = AgentLoop(
        runtime=runtime,
        cwd=tmp_path,
        max_steps=60,
        policy=policy,
        executor=_executor,
    )
    run = loop.run("read repeatedly")
    refusals = [
        r.error or ""
        for step in run.steps
        for r in step.tool_results
        if not r.ok
    ]
    return len(executed), refusals


def test_agent_loop_budget_admits_exactly_budget_calls(tmp_path: Path) -> None:
    """A budget of N must admit N calls, not N-1.

    ``calls_used`` is the count already dispatched; gate 5 adds one for the
    call it is judging. Advancing the loop's counter before the check made the
    gate compare ``k + 1`` against the budget for the k-th call, so every
    policy was one call short.
    """
    for budget in (1, 2, 3, 4, 8):
        executed, _ = _budget_run(tmp_path, budget)
        assert executed == budget, f"budget_calls={budget} admitted {executed}"


def test_agent_loop_budget_of_one_still_executes_one_call(tmp_path: Path) -> None:
    """The sharp end of the off-by-one: a budget of 1 used to execute nothing."""
    executed, _ = _budget_run(tmp_path, 1)
    assert executed == 1


def test_agent_loop_over_budget_call_is_refused_not_executed(tmp_path: Path) -> None:
    """Exhausting the budget stops execution but still reports the refusal.

    The refused call must be visible as a failed tool result rather than
    silently dropped, so a caller can tell a refusal from a step that never
    happened.
    """
    executed, refusals = _budget_run(tmp_path, 3)
    assert executed == 3
    assert refusals, "the over-budget call must be recorded as a dispatch failure"
    assert all("budget exceeded" in r for r in refusals)


# ---------------------------------------------------------------------------
# An unreachable knowledge store must not read as an empty one
# ---------------------------------------------------------------------------


class _HealthyEmptyStore(KnowledgeStore):
    """A reachable store that genuinely holds nothing relevant."""

    def put(self, obj: KnowledgeObject) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def get(self, knowledge_id: KnowledgeId) -> KnowledgeObject | None:  # pragma: no cover
        raise NotImplementedError

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        return RetrievalResult(topic=request.topic, objects=())

    def all(self) -> Iterable[KnowledgeObject]:  # pragma: no cover - unused
        return ()


class _UnreachableStore(KnowledgeStore):
    """A store whose backend is down. Retrieval cannot answer at all."""

    def put(self, obj: KnowledgeObject) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def get(self, knowledge_id: KnowledgeId) -> KnowledgeObject | None:  # pragma: no cover
        raise NotImplementedError

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        raise ConnectionError("knowledge backend unreachable")

    def all(self) -> Iterable[KnowledgeObject]:  # pragma: no cover - unused
        return ()


class TestAFailedRetrievalIsNotAnEmptyRetrieval:
    """`_build_evidence_bodies` returned `()` for three different states.

        if self._knowledge_store is None:  return ()
        try:  ... retrieve ...
        except Exception:                  return ()      <- backend down
        if not result.objects:             return ()      <- genuinely nothing
        try:  ... build_evidence_package ...
        except Exception:                  return ()

    Only the middle one means "there is no knowledge about this topic". The
    first is a declared configuration, the second and third are faults. All
    three reached the model as the same thing: no evidence block at all.

    The consequence is specific. This method exists to ground the model in
    retrieved knowledge before it answers (REQ-PROMPT-023, and #226's
    REQ-TRUTH-005 "insufficient evidence shall support UNVERIFIED ...
    outcomes"). An unreachable backend produces a prompt with no evidence
    segment, which the model reads as *the topic has no stored knowledge* --
    so it answers from its own weights with nothing marking that the
    grounding it was given never arrived. That is the false-success shape,
    produced by a backend outage rather than by a model mistake.

    The swallow itself is NOT the defect and is not changed: REQ-LEARN-016
    requires that extraction failure must not affect task completion, so
    this must keep never raising. What was missing is any way for a reader
    to tell the three states apart afterwards.

    Same defect class as f309d81 in `agents/learning.py`, where
    `except Exception: return ()` in `_dedup_check` reported a failed
    store as "no conflicts". That one was fixed; this instance was missed.
    """

    def _loop(self, store: KnowledgeStore | None) -> AgentLoop:
        runtime = _StubRuntime(script=[_Script(text="done")])
        return AgentLoop(
            runtime=runtime,
            max_steps=2,
            knowledge_store=store,
        )

    def test_an_unreachable_backend_is_reported_as_unavailable(self) -> None:
        loop = self._loop(_UnreachableStore())
        loop.run("how do I configure the scheduler?")
        assert loop.last_evidence_status == "retrieval_failed", (
            "a knowledge backend that raised was recorded as if it had "
            "returned no results"
        )

    def test_a_healthy_empty_backend_is_reported_as_none_found(self) -> None:
        """The contrast case. Without it there is nothing to distinguish."""
        loop = self._loop(_HealthyEmptyStore())
        loop.run("how do I configure the scheduler?")
        assert loop.last_evidence_status == "none_found"

    def test_no_store_wired_is_its_own_declared_state(self) -> None:
        loop = self._loop(None)
        loop.run("how do I configure the scheduler?")
        assert loop.last_evidence_status == "not_wired"

    def test_the_run_still_completes_when_the_backend_is_down(self) -> None:
        """REQ-LEARN-016: extraction failure must not affect the task.

        The fix records the failure; it must not start propagating it.
        """
        loop = self._loop(_UnreachableStore())
        run = loop.run("how do I configure the scheduler?")
        assert run.final_text == "done"
        assert run.finished_reason == "stop"

    def test_the_distinction_reaches_the_composition_record(self) -> None:
        """It has to land in provenance, not only in a local variable.

        `CompositionProvenance.as_dict()` is the REQ-PROMPT-025 evidence
        form — what an operator or an artifact actually reads. A status
        that lives only on the loop instance is one refactor away from
        being unreadable, which is the shape f309d81 had.
        """
        loop = self._loop(_UnreachableStore())
        loop.run("how do I configure the scheduler?")
        record = loop.last_composition.provenance.as_dict()
        assert record["evidence_status"] == "retrieval_failed"

    def test_a_measured_evidence_status_survives_to_provenance(self) -> None:
        loop = self._loop(_HealthyEmptyStore())
        loop.run("how do I configure the scheduler?")
        record = loop.last_composition.provenance.as_dict()
        assert record["evidence_status"] == "none_found"
