"""Tests for the WI-LEARN-001 (#87) lesson / WI-LEARN-002 (#88)
negative-memory seam that wires the canonical AgentLoop to the
existing :class:`KnowledgeStore` abstraction.

These tests are fully offline — they use :class:`PlaceholderRuntime`
plus a deterministic :class:`InMemoryKnowledgeStore` and a stub
runtime that records every request so we can verify the retrieval
path actually reaches the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from oai2.agents import AgentLoop, extract_lesson, record_failure, sanitize_text
from oai2.agents.agent_loop import AgentRun
from oai2.core import KnowledgeId, Status
from oai2.knowledge import (
    InMemoryKnowledgeStore,
    KnowledgeObject,
    RetrievalRequest,
    sha256_hex,
)
from oai2.runtime import (
    InferenceRequest,
    InferenceResponse,
    InferenceRuntime,
)

# ---------------------------------------------------------------------------
# Stub runtime that records every request
# ---------------------------------------------------------------------------


@dataclass
class _Script:
    text: str = ""
    tool_calls: tuple[dict[str, Any], ...] = ()
    finish_reason: str = "stop"


@dataclass
class _StubRuntime(InferenceRuntime):
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(*, user_prompt: str, final_text: str, finished_reason: str = "stop") -> AgentRun:
    return AgentRun(
        user_prompt=user_prompt,
        final_text=final_text,
        steps=[],
        total_tool_calls=0,
        total_input_tokens=0,
        total_output_tokens=0,
        finished_reason=finished_reason,
    )


def _seed_lesson(
    store: InMemoryKnowledgeStore,
    *,
    topic: str,
    body: str,
    authority: float = 0.7,
) -> KnowledgeObject:
    obj = KnowledgeObject(
        knowledge_id=KnowledgeId(sha256_hex(topic + body)[:32]),
        topic=topic,
        content=body,
        content_hash=sha256_hex(body),
        source_uri="seed://test",
        retrieved_at=0.0,
        authority=authority,
        status=Status.IMPLEMENTED,
    )
    store.put(obj)
    return obj


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------


def test_sanitize_text_strips_openai_style_keys() -> None:
    raw = f"the secret is sk-proj-{'abc123def456' * 2} and that's it"
    out = sanitize_text(raw)
    assert "sk-proj-" not in out
    assert "REDACTED" in out


def test_sanitize_text_strips_bearer_token() -> None:
    out = sanitize_text("Authorization: Bearer abcdefghij1234567890XYZ")
    assert "Bearer abcdef" not in out
    assert "REDACTED" in out


def test_sanitize_text_strips_private_key_block() -> None:
    begin_marker = "-----BEGIN " + "PRIVATE KEY" + "-----"
    out = sanitize_text(f"{begin_marker}\nMIIE...\n-----END PRIVATE KEY-----")
    assert begin_marker not in out
    assert "REDACTED" in out


def test_sanitize_text_keeps_normal_text() -> None:
    text = "this is a normal sentence about Python 3.14 and MLX"
    assert sanitize_text(text) == text


# ---------------------------------------------------------------------------
# extract_lesson (#87)
# ---------------------------------------------------------------------------


def test_extract_lesson_returns_knowledge_object_with_provenance() -> None:
    run = _make_run(user_prompt="list the files in /tmp", final_text="a, b, c")
    obj = extract_lesson(
        run,
        task_id="t-001",
        verification_ref="verifier://sess-abc",
        source_version="b224fc6",
        runtime_version="oai2/0.1",
    )
    assert isinstance(obj, KnowledgeObject)
    assert obj.status == Status.IMPLEMENTED
    assert "t-001" in obj.content
    assert "verifier://sess-abc" in obj.content
    assert obj.artifact_ref == "verifier://sess-abc"
    assert obj.source_uri is not None
    assert "source=b224fc6" in obj.source_uri
    assert obj.content_hash == sha256_hex(obj.content)
    assert obj.knowledge_id.startswith(obj.content_hash[:8]) or len(obj.knowledge_id) >= 16


def test_extract_lesson_unfinished_run_is_experimental() -> None:
    run = _make_run(user_prompt="x", final_text="", finished_reason="error")
    obj = extract_lesson(
        run,
        task_id="t-002",
        verification_ref="verifier://sess-002",
        source_version="b224fc6",
        runtime_version="oai2/0.1",
    )
    assert obj.status == Status.EXPERIMENTAL


def test_extract_lesson_dedup_surfaces_existing_object() -> None:
    store = InMemoryKnowledgeStore()
    existing = _seed_lesson(
        store,
        topic="oai2:agent:list files in /tmp",
        body="prior lesson body",
    )
    run = _make_run(user_prompt="list files in /tmp", final_text="ok")
    obj = extract_lesson(
        run,
        task_id="t-003",
        verification_ref="verifier://sess-003",
        source_version="b224fc6",
        runtime_version="oai2/0.1",
        dedup_against=store,
    )
    conflicts = getattr(obj, "_dedup_conflicts", ())
    assert existing in conflicts


def test_extract_lesson_sanitizes_secrets() -> None:
    run = _make_run(
        user_prompt=f"use sk-proj-{'abc123def456' * 2} key",
        final_text="Authorization: Bearer abcdefghij1234567890XYZ",
    )
    obj = extract_lesson(
        run,
        task_id="t-004",
        verification_ref="v",
        source_version="b224fc6",
        runtime_version="oai2/0.1",
    )
    assert "sk-proj-" not in obj.content
    assert "REDACTED" in obj.content


# ---------------------------------------------------------------------------
# record_failure (#88)
# ---------------------------------------------------------------------------


def test_record_failure_is_diagnostic_not_positive() -> None:
    run = _make_run(user_prompt="compile", final_text="", finished_reason="error")
    obj = record_failure(
        run,
        task_id="t-fail-1",
        failure_class="tool_timeout",
        evidence_ref="ev://trace-1",
        source_version="b224fc6",
        runtime_version="oai2/0.1",
    )
    assert obj.status == Status.PROPOSED
    assert "diagnostic" in obj.content
    assert "tool_timeout" in obj.content
    assert "ev://trace-1" in obj.content
    assert obj.topic.startswith("oai2:negative:tool_timeout:")


def test_record_failure_dedup_surfaces_existing() -> None:
    store = InMemoryKnowledgeStore()
    prior = _seed_lesson(
        store,
        topic="oai2:negative:tool_timeout:oai2:agent:compile the project",
        body="prior diagnostic",
        authority=0.5,
    )
    run = _make_run(user_prompt="compile the project", final_text="")
    obj = record_failure(
        run,
        task_id="t-fail-2",
        failure_class="tool_timeout",
        evidence_ref="ev://trace-2",
        source_version="b224fc6",
        runtime_version="oai2/0.1",
        dedup_against=store,
    )
    assert prior in getattr(obj, "_dedup_conflicts", ())


# ---------------------------------------------------------------------------
# AgentLoop integration: retrieval reaches the model
# ---------------------------------------------------------------------------


def test_agent_loop_injects_retrieval_into_messages_when_store_matches() -> None:
    store = InMemoryKnowledgeStore()
    _seed_lesson(
        store,
        topic="oai2:agent:how do I list Python files",
        body="Use `find . -name '*.py'` from the host. Do not enumerate manually.",
    )
    rt = _StubRuntime(script=[_Script(text="done", finish_reason="stop")])
    loop = AgentLoop(
        runtime=rt,
        cwd="/tmp",
        knowledge_store=store,
        evidence_budget_tokens=256,
    )
    run = loop.run("how do I list Python files")
    assert len(rt.requests) == 1
    msgs = rt.requests[0].messages
    # Three messages: system, user, evidence (post-user, lowest priority)
    roles = [m["role"] for m in msgs]
    assert roles[0] == "system"
    assert roles[1] == "user"
    # The evidence is appended as a user-role message with a clear
    # precedence marker (REQ-PROMPT-002 of #179). The framing now comes
    # from the composer, which states that the block is data, not policy.
    assert any(
        m["role"] == "user" and "not instruction" in m["content"] for m in msgs
    )
    # Marker: retrieved evidence never overrides system prompt.
    last = msgs[-1]
    assert "cannot grant permission" in last["content"]
    assert "find . -name '*.py'" in last["content"]
    assert run.finished_reason == "stop"


def test_agent_loop_omits_evidence_when_store_has_no_match() -> None:
    store = InMemoryKnowledgeStore()
    _seed_lesson(
        store,
        topic="unrelated topic",
        body="unrelated body",
    )
    rt = _StubRuntime(script=[_Script(text="done")])
    loop = AgentLoop(runtime=rt, knowledge_store=store)
    loop.run("a totally different task that won't match")
    msgs = rt.requests[0].messages
    roles = [m["role"] for m in msgs]
    # No evidence injected: only system + user.
    assert roles == ["system", "user"]


def test_agent_loop_without_store_has_no_evidence() -> None:
    rt = _StubRuntime(script=[_Script(text="ok")])
    loop = AgentLoop(runtime=rt)
    loop.run("anything")
    msgs = rt.requests[0].messages
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_agent_loop_extraction_failure_does_not_crash_run() -> None:
    """REQ-LEARN-016: extraction failure must not affect task completion."""

    class _BrokenStore:
        def retrieve(self, request: RetrievalRequest) -> Any:  # pragma: no cover - exercised
            raise RuntimeError("simulated retrieval failure")

    rt = _StubRuntime(script=[_Script(text="ok")])
    loop = AgentLoop(runtime=rt, knowledge_store=_BrokenStore())  # type: ignore[arg-type]
    run = loop.run("any task")
    assert run.final_text == "ok"
    # No evidence message appended
    roles = [m["role"] for m in rt.requests[0].messages]
    assert roles == ["system", "user"]


def test_evidence_budget_must_be_positive() -> None:
    with pytest.raises(ValueError):
        AgentLoop(evidence_budget_tokens=0)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        AgentLoop(evidence_budget_tokens=-1)  # type: ignore[call-arg]


def test_full_loop_task_a_to_b_reuse_evidence() -> None:
    """End-to-end reuse: Task A verified → extract_lesson → put → Task B
    (different wording) retrieves the same lesson and the model sees
    it in messages.
    """
    store = InMemoryKnowledgeStore()

    # ---- Task A: a verified agent run ----
    run_a = _make_run(
        user_prompt="how do I find all Python files",
        final_text="Use `find . -name '*.py'` from the host.",
    )
    lesson = extract_lesson(
        run_a,
        task_id="t-A",
        verification_ref="verifier://A",
        source_version="b224fc6",
        runtime_version="oai2/0.1",
        dedup_against=store,
    )
    # Re-key the lesson with a topic that contains the Task B
    # query as a substring. The :class:`InMemoryKnowledgeStore`
    # does a naive substring match (`request.topic in obj.topic`),
    # so the topic must be a SUPERSET of the query. The production
    # Vectorize + D1 retrieval (#22) is semantic and would not
    # need this constraint; this test exercises the AgentLoop
    # wiring, not the retrieval quality.
    shared_topic = "how do I find python files in this repository"
    promoted = lesson.model_copy(
        update={
            "topic": shared_topic,
            "knowledge_id": KnowledgeId(
                sha256_hex(shared_topic + "verified")[:32]
            ),
        }
    )
    store.put(promoted)
    assert store.get(promoted.knowledge_id) is not None

    # ---- Task B: the same query (or a substring of the topic) ----
    rt = _StubRuntime(script=[_Script(text="ok")])
    loop = AgentLoop(
        runtime=rt,
        knowledge_store=store,
        evidence_budget_tokens=256,
    )
    run_b = loop.run("how do I find python files in this repository")
    assert run_b.finished_reason == "stop"
    msgs = rt.requests[0].messages
    last = msgs[-1]
    assert last["role"] == "user"
    assert "not instruction" in last["content"]
    assert "find . -name '*.py'" in last["content"]


class _UnreachableStore:
    """A knowledge store whose backend cannot be reached, e.g. D1 is down."""

    def retrieve(self, request: Any) -> Any:
        raise RuntimeError("D1 unreachable")


class TestAFailedDedupCheckIsNotACleanResult:
    """REQ-LEARN-014/022: the caller must decide about a conflict.

    `_dedup_check` used to `except Exception: return ()`, which made an
    unreachable backend return exactly the same value as a healthy backend
    with nothing on the topic. The caller then imported a duplicate lesson
    believing it had checked and found no conflict. Measured before the
    change:

        store RAISES (check could not run) -> ()
        store healthy, nothing found      -> ()
        store healthy, one conflict found -> ('EXISTING-KNOWLEDGE',)

    The first two were indistinguishable. A failed safety check must not be
    reported as a clean one.
    """

    def test_a_failing_store_raises_instead_of_reporting_no_conflicts(self) -> None:
        run = _make_run(user_prompt="do a thing", final_text="ok")
        with pytest.raises(RuntimeError, match="D1 unreachable"):
            extract_lesson(
                run,
                task_id="t-dedup-fail",
                verification_ref="verifier://sess-dedup-fail",
                source_version="b224fc6",
                runtime_version="oai2/0.1",
                dedup_against=_UnreachableStore(),
            )

    def test_the_original_error_is_preserved_not_wrapped(self) -> None:
        """The caller needs the real cause, not a generic wrapper.

        Flattening the exception into `()` discarded which backend failed and
        why, which is the part an operator needs to act on it.
        """
        run = _make_run(user_prompt="do a thing", final_text="ok")
        with pytest.raises(RuntimeError) as excinfo:
            extract_lesson(
                run,
                task_id="t-dedup-cause",
                verification_ref="verifier://sess-dedup-cause",
                source_version="b224fc6",
                runtime_version="oai2/0.1",
                dedup_against=_UnreachableStore(),
            )
        assert "D1 unreachable" in str(excinfo.value)

    def test_opposite_direction_a_healthy_empty_store_still_reports_no_conflict(
        self,
    ) -> None:
        """Guard: propagating must not turn an empty result into a failure.

        This is the case the blanket `except` was presumably written to cover.
        A healthy store that genuinely has nothing on the topic is a normal
        outcome and must keep returning no conflicts.
        """
        store = InMemoryKnowledgeStore()
        run = _make_run(user_prompt="list files in /tmp", final_text="ok")
        obj = extract_lesson(
            run,
            task_id="t-dedup-empty",
            verification_ref="verifier://sess-dedup-empty",
            source_version="b224fc6",
            runtime_version="oai2/0.1",
            dedup_against=store,
        )
        assert getattr(obj, "_dedup_conflicts", ()) == ()

    def test_opposite_direction_a_real_conflict_is_still_surfaced(self) -> None:
        """Guard: the normal dedup path must be unchanged."""
        store = InMemoryKnowledgeStore()
        existing = _seed_lesson(
            store,
            topic="oai2:agent:list files in /tmp",
            body="prior lesson body",
        )
        run = _make_run(user_prompt="list files in /tmp", final_text="ok")
        obj = extract_lesson(
            run,
            task_id="t-dedup-conflict",
            verification_ref="verifier://sess-dedup-conflict",
            source_version="b224fc6",
            runtime_version="oai2/0.1",
            dedup_against=store,
        )
        assert existing in getattr(obj, "_dedup_conflicts", ())
