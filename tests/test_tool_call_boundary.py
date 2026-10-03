"""The tool-call boundary: what must NOT become an executable call.

Advertising a tool does not mean every JSON example in the model's reply is an
intended call. Three of the failure modes below previously produced an
executable call:

* a **quoted example** — the model explains what a call looks like and the
  example is picked up;
* an **echoed lesson** — retrieved evidence is repeated, and the lesson body
  contains a JSON snippet shaped like a call;
* a **contract violation** — the model calls a function that was never
  advertised, or ignores a pinned ``tool_choice``, and the endpoint returned it
  in ``message.tool_calls`` anyway, where clients execute it.

The second half of this file checks containment through the *actual executor*
with real symlinks, because lexical ``..`` normalisation says nothing about
where an executor ultimately reads.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from oai2.core import ToolId
from oai2.protocols import ToolCall
from oai2.runtime.tool_calls import (
    parse_tool_calls,
    validate_tool_calls,
)
from oai2.tools import ToolDispatcher, execute_tool
from oai2.tools.registry import default_tool_definitions

# ---------------------------------------------------------------------------
# 1. Quoted examples and explanatory prose
# ---------------------------------------------------------------------------

_CALL_JSON = '```json\n{"name": "delete_all", "arguments": {"path": "/"}}\n```'
_AUDIT_JSON = '```json\n{"name": "run_audit", "arguments": {"scope": "all"}}\n```'

EXAMPLE_IN_PROSE = (
    "To call a tool you write an object with a name and arguments, for example "
    f"here is what that looks like: {_CALL_JSON} That is only an illustration of the format."
)

LESSON_ECHO = (
    f"The retrieved lesson says: when auditing, emit a record like {_AUDIT_JSON} "
    "and then compare it against the baseline before concluding anything."
)

EXPLANATORY_ONLY = (
    'The schema declares properties such as {"name": "field", "arguments": '
    '{"type": "string"}} which describe the request body, not a tool call.'
)


def test_quoted_example_is_not_a_call() -> None:
    assert parse_tool_calls(EXAMPLE_IN_PROSE, allow_bare_json=True) == ()


def test_echoed_lesson_content_is_not_a_call() -> None:
    """Evidence the model repeats must not become executable."""
    assert parse_tool_calls(LESSON_ECHO, allow_bare_json=True) == ()


def test_explanatory_json_schema_is_not_a_call() -> None:
    assert parse_tool_calls(EXPLANATORY_ONLY, allow_bare_json=True) == ()


def test_a_real_call_with_a_short_preamble_is_still_a_call() -> None:
    """The guard must not reject the legitimate short-answer case."""
    text = '```json\n{"name": "list_dir", "arguments": {"path": "src"}}\n```'
    assert len(parse_tool_calls(text, allow_bare_json=True)) == 1


def test_tagged_call_is_unaffected_by_the_prose_guard() -> None:
    """An explicit tag is unambiguous even inside a paragraph."""
    tagged = (
        "Here is my plan. "
        '<tool_call>{"name": "list_dir", "arguments": {"path": "src"}}</tool_call>'
        " Let me know if that is wrong."
    )
    assert len(parse_tool_calls(tagged, allow_bare_json=True)) == 1


# ---------------------------------------------------------------------------
# 2. Contract violations must not become executable success
# ---------------------------------------------------------------------------

DECLARED = [{"type": "function", "function": {"name": "list_dir", "parameters": {}}}]
UNDECLARED = '<tool_call>{"name": "delete_all", "arguments": {"path": "/"}}</tool_call>'


def test_undeclared_function_is_flagged() -> None:
    problems = validate_tool_calls(parse_tool_calls(UNDECLARED), declared=DECLARED, tool_choice="auto")
    assert any("undeclared" in p for p in problems)


def test_endpoint_does_not_return_a_violating_call_as_executable() -> None:
    from fastapi.testclient import TestClient

    from oai2.server import openai_compat_app as mod
    from tests.test_zcode_composer_integration import BASE_MESSAGES, _StubHotRuntime

    runtime = _StubHotRuntime(script=[
        {"text": "", "tool_calls": ({"id": "c1", "type": "function",
                                      "function": {"name": "delete_all", "arguments": "{}"}},),
         "finish_reason": "tool_calls"},
    ])

    real = mod.MLXHotRuntime

    class _Patched(_StubHotRuntime):
        def __init__(self, spec, *, model_id="", prefix_cache=None, gate_digest=False, **_):
            super().__init__(model_id=model_id or "stub")
            self._shared = runtime
            if prefix_cache is not None:
                self._prefix_cache = prefix_cache

        def generate(self, request):
            return self._shared.generate(request)

    mod.MLXHotRuntime = _Patched  # type: ignore[misc]
    try:
        client = TestClient(mod.create_app(model_id="stub", with_tools=True))
    finally:
        mod.MLXHotRuntime = real  # type: ignore[misc]

    tools = [{"type": "function", "function": {"name": "list_dir", "parameters": {}}}]
    resp = client.post("/v1/chat/completions", json={
        "model": "stub", "messages": BASE_MESSAGES, "tools": tools,
    }).json()
    choice = resp["choices"][0]
    # The violating call must NOT be in the executable array.
    assert "tool_calls" not in choice["message"]
    assert choice["finish_reason"] == "tool_contract_violation"
    # It must still be visible and auditable.
    assert resp["rejected_tool_calls"]
    assert any("undeclared" in p for p in resp["tool_contract_problems"])


def test_streaming_endpoint_does_not_emit_a_violating_call_as_executable() -> None:
    """Buffered SSE must enforce the same boundary as JSON responses."""
    import json

    from fastapi.testclient import TestClient

    from oai2.server import openai_compat_app as mod
    from tests.test_zcode_composer_integration import BASE_MESSAGES, _StubHotRuntime

    runtime = _StubHotRuntime(script=[
        {"text": "", "tool_calls": ({"id": "c1", "type": "function",
                                      "function": {"name": "delete_all", "arguments": "{}"}},),
         "finish_reason": "tool_calls"},
    ])
    real = mod.MLXHotRuntime

    class _Patched(_StubHotRuntime):
        def __init__(self, spec, *, model_id="", prefix_cache=None, gate_digest=False, **_):
            super().__init__(model_id=model_id or "stub")
            self._shared = runtime
            if prefix_cache is not None:
                self._prefix_cache = prefix_cache

        def generate(self, request):
            return self._shared.generate(request)

    mod.MLXHotRuntime = _Patched  # type: ignore[misc]
    try:
        client = TestClient(mod.create_app(model_id="stub", with_tools=True))
    finally:
        mod.MLXHotRuntime = real  # type: ignore[misc]

    tools = [{"type": "function", "function": {"name": "list_dir", "parameters": {}}}]
    response = client.post("/v1/chat/completions", json={
        "model": "stub", "messages": BASE_MESSAGES, "tools": tools, "stream": True,
    })
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    final = events[-1]
    choice = final["choices"][0]
    assert "tool_calls" not in choice["delta"]
    assert choice["finish_reason"] == "tool_contract_violation"
    assert final["rejected_tool_calls"]
    assert any("undeclared" in p for p in final["tool_contract_problems"])


def test_endpoint_returns_a_clean_call_normally() -> None:
    from fastapi.testclient import TestClient

    from oai2.server import openai_compat_app as mod
    from tests.test_zcode_composer_integration import BASE_MESSAGES, _StubHotRuntime

    runtime = _StubHotRuntime(script=[
        {"text": "", "tool_calls": ({"id": "c1", "type": "function",
                                      "function": {"name": "list_dir", "arguments": '{"path":"src"}'}},
                                     ), "finish_reason": "tool_calls"},
    ])
    real = mod.MLXHotRuntime

    class _Patched(_StubHotRuntime):
        def __init__(self, spec, *, model_id="", prefix_cache=None, gate_digest=False, **_):
            super().__init__(model_id=model_id or "stub")
            self._shared = runtime
            if prefix_cache is not None:
                self._prefix_cache = prefix_cache

        def generate(self, request):
            return self._shared.generate(request)

    mod.MLXHotRuntime = _Patched  # type: ignore[misc]
    try:
        client = TestClient(mod.create_app(model_id="stub", with_tools=True))
    finally:
        mod.MLXHotRuntime = real  # type: ignore[misc]

    tools = [{"type": "function", "function": {"name": "list_dir", "parameters": {}}}]
    resp = client.post("/v1/chat/completions", json={
        "model": "stub", "messages": BASE_MESSAGES, "tools": tools,
    }).json()
    choice = resp["choices"][0]
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "list_dir"
    assert choice["finish_reason"] == "tool_calls"
    assert resp["tool_contract_problems"] == []
    assert "rejected_tool_calls" not in resp


@pytest.mark.parametrize(
    ("tool_choice", "calls", "expected"),
    [
        ("none", '<tool_call>{"name":"list_dir","arguments":{}}</tool_call>', False),
        ("required", "I cannot do that.", True),
    ],
)
def test_tool_choice_contract_violations_are_flagged(tool_choice, calls, expected) -> None:
    problems = validate_tool_calls(
        parse_tool_calls(calls), declared=DECLARED, tool_choice=tool_choice
    )
    assert bool(problems) is expected


def test_forced_function_mismatch_is_flagged() -> None:
    calls = parse_tool_calls('<tool_call>{"name":"read_file","arguments":{}}</tool_call>')
    problems = validate_tool_calls(
        calls, declared=DECLARED,
        tool_choice={"type": "function", "function": {"name": "list_dir"}},
    )
    assert any("pinned" in p for p in problems)


# ---------------------------------------------------------------------------
# 3. Containment through the ACTUAL executor, with real symlinks
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path):
    """A cwd with one allowed subdirectory and one symlink escaping it."""
    inside = tmp_path / "work"
    inside.mkdir()
    (inside / "ok.txt").write_text("fine\n", encoding="utf-8")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOP SECRET\n", encoding="utf-8")

    os.symlink(outside, inside / "escape", target_is_directory=True)
    os.symlink(outside / "secret.txt", inside / "secret-link.txt")
    return tmp_path


def _dispatcher(cwd: Path, scopes: set[str]) -> ToolDispatcher:
    from oai2.agents.agent_loop import default_dispatch_policy

    policy = default_dispatch_policy(resource_scopes=scopes)
    return ToolDispatcher(registry=default_tool_definitions(), policy=policy, cwd=cwd)


def test_symlink_escape_is_denied_by_the_policy(workspace) -> None:
    """The lexical path is inside the scope; the resolved path is not."""
    inside = workspace / "work"
    d = _dispatcher(inside, {str(inside)})
    out = d.check(ToolCall(id="c1", tool_id=ToolId("Read"),
                           arguments={"path": str(inside / "escape" / "secret.txt")}),
                  calls_used=0)
    assert out.stage.value == "deny"
    assert "resource out of scope" in out.reason


def test_symlinked_file_escape_is_denied(workspace) -> None:
    inside = workspace / "work"
    d = _dispatcher(inside, {str(inside)})
    out = d.check(ToolCall(id="c1", tool_id=ToolId("Read"),
                           arguments={"path": str(inside / "secret-link.txt")}),
                  calls_used=0)
    assert out.stage.value == "deny"


def test_genuine_file_inside_the_scope_is_still_allowed(workspace) -> None:
    inside = workspace / "work"
    d = _dispatcher(inside, {str(inside)})
    out = d.check(ToolCall(id="c1", tool_id=ToolId("Read"),
                           arguments={"path": str(inside / "ok.txt")}),
                  calls_used=0)
    assert out.stage.value == "execute"


def test_executor_glob_cannot_list_through_a_symlinked_escape(workspace) -> None:
    """Policy AND executor must both refuse; the executor is checked here."""
    inside = workspace / "work"
    call = ToolCall(id="c1", tool_id=ToolId("Glob"),
                    arguments={"pattern": "*", "path": str(inside / "escape")})
    result = execute_tool(call, cwd=inside)
    # The executor refuses a root it cannot treat as a directory.
    assert result.ok is False or "TOP SECRET" not in (result.output or "")


def test_executor_glob_lists_the_authorized_root(workspace) -> None:
    inside = workspace / "work"
    call = ToolCall(id="c1", tool_id=ToolId("Glob"),
                    arguments={"pattern": "*.txt", "path": str(inside)})
    result = execute_tool(call, cwd=inside)
    assert result.ok is True
    assert "ok.txt" in (result.output or "")
    assert "secret-link.txt" not in (result.output or "")
    assert "TOP SECRET" not in (result.output or "")


def test_glob_scoped_tool_is_denied_when_the_root_is_out_of_scope(workspace) -> None:
    inside = workspace / "work"
    d = _dispatcher(inside, {str(inside)})
    out = d.check(ToolCall(id="c1", tool_id=ToolId("Glob"),
                           arguments={"pattern": "*", "path": str(workspace / "outside")}),
                  calls_used=0)
    assert out.stage.value == "deny"


def test_executor_grep_cannot_read_through_a_symlinked_escape(workspace) -> None:
    """Grep of an in-scope root must not surface content from outside it.

    The requested path (``inside``) is in scope, so the dispatcher's scope gate
    passes. The match lives behind a symlink whose name is inside the root, so
    only a resolved-containment check in the executor can stop the read.
    """
    inside = workspace / "work"
    call = ToolCall(id="c1", tool_id=ToolId("Grep"),
                    arguments={"pattern": "TOP SECRET", "path": str(inside)})
    result = execute_tool(call, cwd=inside)
    assert "TOP SECRET" not in (result.output or "")


def test_grep_naming_a_symlinked_file_escape_is_denied_by_the_policy(workspace) -> None:
    """Naming the escaping symlink directly is stopped at the scope gate.

    This is the dispatcher's job, not the executor's: a directly-named file is
    validated by the scope check before ``execute_tool`` is ever reached.
    """
    inside = workspace / "work"
    d = _dispatcher(inside, {str(inside)})
    out = d.check(ToolCall(id="c1", tool_id=ToolId("Grep"),
                           arguments={"pattern": "TOP SECRET",
                                      "path": str(inside / "secret-link.txt")}),
                  calls_used=0)
    assert out.stage.value == "deny"
    assert "resource out of scope" in out.reason


def test_executor_grep_still_finds_matches_in_the_authorized_root(workspace) -> None:
    """Containment must not over-block: real in-scope matches are still returned."""
    inside = workspace / "work"
    call = ToolCall(id="c1", tool_id=ToolId("Grep"),
                    arguments={"pattern": "fine", "path": str(inside)})
    result = execute_tool(call, cwd=inside)
    assert result.ok is True
    assert "ok.txt:1:" in (result.output or "")


def test_executor_grep_include_glob_still_filters(workspace) -> None:
    inside = workspace / "work"
    call = ToolCall(id="c1", tool_id=ToolId("Grep"),
                    arguments={"pattern": "fine", "path": str(inside),
                               "include_glob": "*.txt"})
    result = execute_tool(call, cwd=inside)
    assert result.ok is True
    assert "ok.txt:1:" in (result.output or "")


def test_executor_grep_on_a_single_in_scope_file_still_works(workspace) -> None:
    inside = workspace / "work"
    call = ToolCall(id="c1", tool_id=ToolId("Grep"),
                    arguments={"pattern": "fine", "path": str(inside / "ok.txt")})
    result = execute_tool(call, cwd=inside)
    assert result.ok is True
    assert "ok.txt:1:" in (result.output or "")


def test_executor_grep_no_match_is_reported_as_no_match(workspace) -> None:
    inside = workspace / "work"
    call = ToolCall(id="c1", tool_id=ToolId("Grep"),
                    arguments={"pattern": "nothing_matches_this", "path": str(inside)})
    result = execute_tool(call, cwd=inside)
    assert result.ok is True
    assert "ok.txt" not in (result.output or "")


def test_grep_scoped_tool_is_denied_when_the_root_is_out_of_scope(workspace) -> None:
    inside = workspace / "work"
    d = _dispatcher(inside, {str(inside)})
    out = d.check(ToolCall(id="c1", tool_id=ToolId("Grep"),
                           arguments={"pattern": "TOP SECRET",
                                      "path": str(workspace / "outside")}),
                  calls_used=0)
    assert out.stage.value == "deny"


class TestToolElapsedTimeIsMeasuredNotFabricated:
    """`elapsed_ms` must be a measurement, not a default nobody set.

    `ExecutionOutcome.elapsed_ms` defaulted to `0.0` and no handler ever
    assigned it, so `execute_tool` copied a fabricated `0.0` into
    `ToolResult.elapsed_ms` for every single call. A `bash` that slept for
    thirty seconds and a `true` that returned instantly were
    indistinguishable, and any latency statistic computed over tool results
    was a perfect zero -- the same value a never-measured system would
    report.

    `execute_tool` is the single funnel every tool passes through, so that is
    where the measurement is taken; handlers leave the field alone.
    """

    def test_a_slow_tool_call_reports_a_slow_elapsed_time(self, tmp_path) -> None:
        """The defect, stated as a contradiction that cannot both hold.

        Before the fix both calls returned exactly 0.0, so this assertion
        held only by accident of a zero default.
        """
        slow = execute_tool(
            ToolCall(id="c1", tool_id=ToolId("bash"), arguments={"command": "sleep 0.4"}),
            cwd=tmp_path,
        )
        assert slow.ok is True
        assert slow.elapsed_ms > 100, (
            f"a 400ms sleep reported elapsed_ms={slow.elapsed_ms}; the field "
            f"is a default, not a measurement"
        )

    def test_elapsed_time_distinguishes_a_slow_call_from_a_fast_one(
        self, tmp_path
    ) -> None:
        """Guard: a real distribution, not a constant.

        Asserting only "not zero" would also pass a hardcoded 1.0. Ordering
        is what a measurement guarantees and a constant cannot fake.
        """
        slow = execute_tool(
            ToolCall(id="c1", tool_id=ToolId("bash"), arguments={"command": "sleep 0.3"}),
            cwd=tmp_path,
        )
        fast = execute_tool(
            ToolCall(id="c2", tool_id=ToolId("bash"), arguments={"command": "true"}),
            cwd=tmp_path,
        )
        assert slow.elapsed_ms > fast.elapsed_ms
        assert fast.elapsed_ms < 100

    def test_a_failed_tool_call_is_also_timed(self, tmp_path) -> None:
        """The measurement must not depend on the tool having succeeded.

        Timing only the success path would leave the error path -- the one a
        latency SLO cares about most -- reporting a fabricated zero again.
        """
        failed = execute_tool(
            ToolCall(id="c1", tool_id=ToolId("read"), arguments={"path": "nope.py"}),
            cwd=tmp_path,
        )
        assert failed.ok is False
        assert failed.elapsed_ms > 0.0

    def test_an_unsupported_tool_is_also_timed(self, tmp_path) -> None:
        """The fallback branch goes through the same return, so it is timed."""
        unsupported = execute_tool(
            ToolCall(id="c1", tool_id=ToolId("nonexistent"), arguments={}),
            cwd=tmp_path,
        )
        assert unsupported.ok is False
        assert "unsupported tool" in (unsupported.error or "")
        assert unsupported.elapsed_ms > 0.0
