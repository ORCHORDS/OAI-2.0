"""Tests for the canonical tool registry and local execution."""

from __future__ import annotations

import json
from pathlib import Path

from oai2.core import ToolId
from oai2.protocols import ToolCall
from oai2.tools.registry import (
    default_tool_definitions,
    execute_tool,
    to_openai_wire,
)


def test_default_tool_definitions_has_six_tools() -> None:
    tools = default_tool_definitions()
    names = sorted(td.name for td in tools)
    assert names == ["Bash", "Edit", "Glob", "Grep", "Read", "Write"]


def test_to_openai_wire_shape() -> None:
    wire = to_openai_wire(default_tool_definitions())
    assert len(wire) == 6
    for entry in wire:
        assert entry["type"] == "function"
        fn = entry["function"]
        assert "name" in fn and "description" in fn and "parameters" in fn
        params = fn["parameters"]
        assert params["type"] == "object"
    names = {entry["function"]["name"] for entry in wire}
    assert "Read" in names


def test_to_openai_wire_marks_required_args() -> None:
    wire = to_openai_wire(default_tool_definitions())
    by_name = {entry["function"]["name"]: entry for entry in wire}
    read_required = by_name["Read"]["function"]["parameters"].get("required", [])
    assert "path" in read_required
    bash_required = by_name["Bash"]["function"]["parameters"].get("required", [])
    assert "command" in bash_required


def test_execute_read_returns_file_contents(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("hello\nworld\n", encoding="utf-8")
    call = ToolCall(
        id="c1", tool_id=ToolId("read"), arguments={"path": str(f)}
    )
    result = execute_tool(call, cwd=tmp_path)
    assert result.ok
    assert result.output == "hello\nworld\n"
    assert result.call_id == "c1"


def test_execute_read_missing_file(tmp_path: Path) -> None:
    call = ToolCall(
        id="c1",
        tool_id=ToolId("read"),
        arguments={"path": "nope.txt"},
    )
    result = execute_tool(call, cwd=tmp_path)
    assert not result.ok
    assert "not found" in (result.error or "")


def test_execute_edit_replaces_unique_string(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("alpha beta gamma\n", encoding="utf-8")
    call = ToolCall(
        id="c1",
        tool_id=ToolId("edit"),
        arguments={
            "path": str(f),
            "old_string": "beta",
            "new_string": "BETA",
        },
    )
    result = execute_tool(call, cwd=tmp_path)
    assert result.ok
    assert f.read_text(encoding="utf-8") == "alpha BETA gamma\n"


def test_execute_edit_rejects_ambiguous_string(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("a a a\n", encoding="utf-8")
    call = ToolCall(
        id="c1",
        tool_id=ToolId("edit"),
        arguments={
            "path": str(f),
            "old_string": "a",
            "new_string": "b",
        },
    )
    result = execute_tool(call, cwd=tmp_path)
    assert not result.ok
    assert "unique" in (result.error or "").lower()


def test_execute_write_creates_file(tmp_path: Path) -> None:
    f = tmp_path / "sub" / "x.txt"
    call = ToolCall(
        id="c1",
        tool_id=ToolId("write"),
        arguments={"path": str(f), "content": "hi"},
    )
    result = execute_tool(call, cwd=tmp_path)
    assert result.ok
    assert f.read_text(encoding="utf-8") == "hi"


def test_execute_bash_runs_command(tmp_path: Path) -> None:
    call = ToolCall(
        id="c1",
        tool_id=ToolId("bash"),
        arguments={"command": "echo hello-from-bash", "timeout_seconds": 5},
    )
    result = execute_tool(call, cwd=tmp_path)
    assert result.ok
    assert "hello-from-bash" in result.output


def test_execute_bash_surfaces_failure(tmp_path: Path) -> None:
    call = ToolCall(
        id="c1",
        tool_id=ToolId("bash"),
        arguments={"command": "exit 7", "timeout_seconds": 5},
    )
    result = execute_tool(call, cwd=tmp_path)
    assert not result.ok
    assert "7" in (result.error or "")


def test_execute_glob_lists_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "b.py").write_text("", encoding="utf-8")
    (tmp_path / "c.txt").write_text("", encoding="utf-8")
    call = ToolCall(
        id="c1",
        tool_id=ToolId("glob"),
        arguments={"pattern": "*.py"},
    )
    result = execute_tool(call, cwd=tmp_path)
    assert result.ok
    assert "a.py" in result.output
    assert "b.py" in result.output
    assert "c.txt" not in result.output


def test_execute_grep_finds_pattern(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("alpha\nbeta hit\ngamma\n", encoding="utf-8")
    call = ToolCall(
        id="c1",
        tool_id=ToolId("grep"),
        arguments={"pattern": r"hit", "path": str(tmp_path)},
    )
    result = execute_tool(call, cwd=tmp_path)
    assert result.ok
    assert "hit" in result.output
    assert "a.txt:2" in result.output


def test_execute_unknown_tool_returns_error(tmp_path: Path) -> None:
    call = ToolCall(
        id="c1",
        tool_id=ToolId("nope"),
        arguments={},
    )
    result = execute_tool(call, cwd=tmp_path)
    assert not result.ok
    assert "unsupported" in (result.error or "")


def test_to_openai_wire_is_json_serialisable() -> None:
    """The wire shape must be JSON-serialisable for the gateway POST body."""
    wire = to_openai_wire(default_tool_definitions())
    json.dumps(wire)  # must not raise


class TestGrepReportsIncompleteCoverage:
    """A search that skipped files is not a search that found nothing.

    `_grep` walks every candidate file under the requested root. When a file
    could not be read it was skipped, and the tool used to return a
    successful `(no matches)` — byte-identical to the result of a search that
    opened every file and genuinely found nothing.

    That is the difference between "this is not in the codebase" and "I could
    not look", and an agent acting on the first answer is acting on a claim
    the tool never earned.
    """

    def test_an_unreadable_file_makes_the_search_incomplete(self, tmp_path) -> None:
        import os

        from oai2.tools.registry import _grep

        (tmp_path / "clean.py").write_text("x = 1\n")
        locked = tmp_path / "locked.py"
        locked.write_text("def needle_function():\n    pass\n")
        os.chmod(locked, 0o000)
        try:
            outcome = _grep(
                "needle_function", path_str=str(tmp_path), include_glob=None, cwd=tmp_path
            )
        finally:
            os.chmod(locked, 0o644)

        assert outcome.ok is True, "a partial search is still a successful call"
        assert outcome.unreadable_count == 1
        assert outcome.coverage_complete is False
        # The absence is stated rather than implied.
        assert "no matches" in outcome.output
        assert "could not be read" in outcome.output
        assert "incomplete" in outcome.output

    def test_a_genuine_empty_search_is_unchanged(self, tmp_path) -> None:
        """The control: a complete search that found nothing stays clean.

        Without this, the fix could be satisfied by making every empty result
        look like a failure.
        """
        from oai2.tools.registry import _grep

        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("y = 2\n")
        outcome = _grep("needle", path_str=str(tmp_path), include_glob=None, cwd=tmp_path)

        assert outcome.ok is True
        assert outcome.output == "(no matches)"
        assert outcome.unreadable_count == 0
        assert outcome.coverage_complete is True

    def test_a_complete_search_with_matches_is_unchanged(self, tmp_path) -> None:
        from oai2.tools.registry import _grep

        (tmp_path / "a.py").write_text("x = 1\nneedle = 2\n")
        outcome = _grep("needle", path_str=str(tmp_path), include_glob=None, cwd=tmp_path)

        assert outcome.ok is True
        assert "a.py:2" in outcome.output
        assert "could not be read" not in outcome.output
        assert outcome.coverage_complete is True

    def test_a_match_is_reported_alongside_the_unreadable_count(self, tmp_path) -> None:
        """Partial results are still honest about what they missed."""
        import os

        from oai2.tools.registry import _grep

        (tmp_path / "hit.py").write_text("needle = 1\n")
        locked = tmp_path / "locked.py"
        locked.write_text("nothing here\n")
        os.chmod(locked, 0o000)
        try:
            outcome = _grep(
                "needle", path_str=str(tmp_path), include_glob=None, cwd=tmp_path
            )
        finally:
            os.chmod(locked, 0o644)

        assert outcome.ok is True
        assert "hit.py:1" in outcome.output
        assert "could not be read" in outcome.output
        assert outcome.coverage_complete is False

    def test_control_unreadable_discard_restored(self, tmp_path) -> None:
        """Restore the silent `continue` and show the record loses the fact.

        Source mutation rather than attribute patching: the skip is inside a
        loop body, so there is nothing to monkeypatch. The mutant is the
        pre-fix behaviour and must reproduce the indistinguishable record.
        """
        import importlib.util
        import os
        import sys

        source = (Path(__file__).parent.parent / "oai2" / "tools" / "registry.py").read_text()
        anchor = "                unreadable += 1\n                continue"
        assert anchor in source, "anchor moved; control is stale"
        target = tmp_path / "mutant_registry.py"
        target.write_text(source.replace(anchor, "                continue", 1))

        name = "oai2.tools._mutant_registry"
        spec = importlib.util.spec_from_file_location(name, target)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            del sys.modules[name]
            raise

        root = tmp_path / "corpus"
        root.mkdir()
        (root / "clean.py").write_text("x = 1\n")
        locked = root / "locked.py"
        locked.write_text("def needle_function():\n    pass\n")
        os.chmod(locked, 0o000)
        try:
            mutant = module._grep(
                "needle_function", path_str=str(root), include_glob=None, cwd=root
            )
        finally:
            os.chmod(locked, 0o644)

        # The defect: a clean, complete-looking record for a search that
        # never opened the only file that could have matched.
        assert mutant.ok is True
        assert mutant.output == "(no matches)"
        assert mutant.unreadable_count == 0
        assert mutant.coverage_complete is True
