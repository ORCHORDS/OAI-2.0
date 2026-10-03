"""Default tool registry for the OAI-2.0 agent loop.

This module owns the OpenAI-style tool definitions the agent advertises
to oai-2.0 via the gateway. It also owns the local tool execution
handlers so the agent loop can dispatch a model-emitted ``tool_call``
into a real side effect on the host.

The wire shape follows the OpenAI ``/v1/chat/completions`` tool spec::

    {
      "type": "function",
      "function": {
        "name": "read",
        "description": "...",
        "parameters": {
          "type": "object",
          "properties": {"path": {"type": "string"}},
          "required": ["path"],
        },
      },
    }

Status: EXPERIMENTAL (added for WI-AGT-001 / sess_10cbe33c-d83b-42ce-bf2c).
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core import ToolId
from ..protocols import (
    ToolArgument,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


def _function_tool(
    *,
    name: str,
    description: str,
    arguments: tuple[ToolArgument, ...],
    capability: str,
    scoped: bool = False,
    high_impact: bool = False,
) -> ToolDefinition:
    return ToolDefinition(
        id=ToolId(name),
        name=name,
        description=description,
        arguments=arguments,
        capability=capability,
        scoped=scoped,
        high_impact=high_impact,
    )


def default_tool_definitions() -> tuple[ToolDefinition, ...]:
    """Return the canonical agent tool set.

    Six tools, mirroring the standard agent/IDE surface so the model
    sees familiar names. ``Bash`` is high-impact; ``Edit``, ``Write``,
    ``Read`` and ``Grep`` are scoped. ``Bash`` and ``Glob`` are not
    scoped: ``Bash`` takes a command string and ``Glob`` takes only a
    pattern, so neither has a path for gate 4 to constrain — both need
    an explicit unscoped grant in the policy.
    """
    return (
        _function_tool(
            name="Read",
            description=(
                "Read the full contents of a file. Path is relative to "
                "the current working directory or absolute."
            ),
            arguments=(
                ToolArgument(name="path", type="path"),
                ToolArgument(name="offset", type="integer"),
                ToolArgument(name="limit", type="integer"),
            ),
            capability="fs.read",
            scoped=True,
        ),
        _function_tool(
            name="Edit",
            description=(
                "Edit a file by replacing a unique old_string with a new "
                "replacement. old_string must match exactly once."
            ),
            arguments=(
                ToolArgument(name="path", type="path"),
                ToolArgument(name="old_string", type="string"),
                ToolArgument(name="new_string", type="string"),
            ),
            capability="fs.write",
            scoped=True,
        ),
        _function_tool(
            name="Write",
            description=(
                "Write a new file (overwrites if it exists). Path is "
                "relative to cwd or absolute."
            ),
            arguments=(
                ToolArgument(name="path", type="path"),
                ToolArgument(name="content", type="string"),
            ),
            capability="fs.write",
            scoped=True,
        ),
        _function_tool(
            name="Bash",
            description=(
                "Run a shell command and return its combined stdout+stderr. "
                "Use for builds, tests, git, and any host command that "
                "does not have a dedicated tool."
            ),
            arguments=(
                ToolArgument(name="command", type="string"),
                ToolArgument(name="timeout_seconds", type="integer"),
            ),
            capability="shell.exec",
            high_impact=True,
        ),
        _function_tool(
            name="Glob",
            description=(
                "List paths matching a glob pattern under ``path`` "
                "(default: the working directory). Returns newline-separated "
                "paths relative to that directory."
            ),
            arguments=(
                ToolArgument(name="pattern", type="string"),
                ToolArgument(name="path", type="path"),
            ),
            capability="fs.list",
            # Scoped: it takes a ``path``, so gate 4 can hold the listing root
            # inside the host's declared scopes. Without it, directory listing
            # had no scope to be checked against and had to be granted
            # unscoped reach.
            scoped=True,
        ),
        _function_tool(
            name="Grep",
            description=(
                "Search a path for a regex. Returns matching lines in "
                "``path:lineno: line`` form."
            ),
            arguments=(
                ToolArgument(name="pattern", type="string"),
                ToolArgument(name="path", type="path"),
                ToolArgument(name="include_glob", type="string"),
            ),
            capability="fs.read",
            # Scoped: it takes a ``path``, so gate 4 can hold it inside the
            # host's declared scopes. Leaving this unscoped made Grep a way
            # to read anything Read was forbidden to read.
            scoped=True,
        ),
    )


def to_openai_wire(definitions: Iterable[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert internal :class:`ToolDefinition` to the OpenAI wire shape.

    The shape is what the gateway passes through to oai-2.0 inside the
    ``tools`` array of a chat-completions request.
    """
    wire: list[dict[str, Any]] = []
    for td in definitions:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for arg in td.arguments:
            properties[arg.name] = {"type": _openai_type(arg.type)}
            if arg.name in {"path", "command", "old_string", "new_string", "content", "pattern"}:
                required.append(arg.name)
        wire.append(
            {
                "type": "function",
                "function": {
                    "name": td.name,
                    "description": td.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )
    return wire


def _openai_type(internal: str) -> str:
    return {
        "string": "string",
        "integer": "integer",
        "number": "number",
        "boolean": "boolean",
        "array": "array",
        "object": "object",
        "null": "null",
        "uri": "string",
        "path": "string",
        "binary": "string",
    }.get(internal, "string")


# ---------------------------------------------------------------------------
# Local execution
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ExecutionOutcome:
    ok: bool
    output: str
    error: str | None = None
    # Measured by :func:`execute_tool`, which is the single funnel every tool
    # passes through. It used to default to 0.0 and no handler ever assigned
    # it, so every tool call -- including a `bash` that slept for thirty
    # seconds -- reported a fabricated 0.0 ms. A latency statistic over tool
    # results was a perfect zero, indistinguishable from never having
    # measured. Handlers leave it alone; the funnel fills it in.
    elapsed_ms: float = 0.0
    #: Number of candidate files whose contents this tool could NOT read.
    #: A non-zero value means the tool did not look at everything it set out
    #: to look at, so a successful outcome must not be read as "the whole
    #: corpus was searched and nothing was found". Distinct from an
    #: intentional exclusion (a path outside the authorised root), which is a
    #: deliberate scope decision rather than a failed read.
    unreadable_count: int = 0

    @property
    def coverage_complete(self) -> bool:
        """Whether every candidate file in scope was actually read."""
        return self.unreadable_count == 0


def _resolve(path_str: str, *, cwd: Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = cwd / p
    return p


def _read(path_str: str, *, offset: int | None, limit: int | None, cwd: Path) -> ExecutionOutcome:
    try:
        p = _resolve(path_str, cwd=cwd)
        if not p.exists():
            return ExecutionOutcome(False, "", f"file not found: {p}")
        if not p.is_file():
            return ExecutionOutcome(False, "", f"not a file: {p}")
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        start = max(0, int(offset or 0))
        end = start + int(limit) if limit else len(lines)
        return ExecutionOutcome(True, "".join(lines[start:end]))
    except Exception as exc:  # pragma: no cover - defensive
        return ExecutionOutcome(False, "", f"{type(exc).__name__}: {exc}")


def _edit(path_str: str, *, old_string: str, new_string: str, cwd: Path) -> ExecutionOutcome:
    try:
        p = _resolve(path_str, cwd=cwd)
        if not p.exists():
            return ExecutionOutcome(False, "", f"file not found: {p}")
        text = p.read_text(encoding="utf-8")
        if old_string not in text:
            return ExecutionOutcome(False, "", "old_string not found in file")
        occurrences = text.count(old_string)
        if occurrences > 1:
            return ExecutionOutcome(
                False, "", f"old_string matches {occurrences} locations; must be unique"
            )
        new_text = text.replace(old_string, new_string, 1)
        p.write_text(new_text, encoding="utf-8")
        return ExecutionOutcome(
            True, f"replaced 1 occurrence in {p}"
        )
    except Exception as exc:  # pragma: no cover
        return ExecutionOutcome(False, "", f"{type(exc).__name__}: {exc}")


def _write(path_str: str, *, content: str, cwd: Path) -> ExecutionOutcome:
    try:
        p = _resolve(path_str, cwd=cwd)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return ExecutionOutcome(True, f"wrote {len(content)} bytes to {p}")
    except Exception as exc:  # pragma: no cover
        return ExecutionOutcome(False, "", f"{type(exc).__name__}: {exc}")


def _bash(command: str, *, timeout_seconds: int | None, cwd: Path) -> ExecutionOutcome:
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds if timeout_seconds else 60,
        )
        out = proc.stdout + proc.stderr
        ok = proc.returncode == 0
        return ExecutionOutcome(
            ok,
            out if out else f"(exit {proc.returncode}, no output)",
            None if ok else f"exit {proc.returncode}",
        )
    except subprocess.TimeoutExpired:
        return ExecutionOutcome(False, "", "timeout")
    except Exception as exc:  # pragma: no cover
        return ExecutionOutcome(False, "", f"{type(exc).__name__}: {exc}")


def _glob(pattern: str, *, cwd: Path, root: str | None = None) -> ExecutionOutcome:
    """List paths matching ``pattern`` under ``root`` (default: ``cwd``).

    Every match is resolved and checked to be inside the search root before it
    is returned. A pattern can name an absolute path, and following one would
    walk the filesystem outside the root the host authorised — lexical checks
    on the *requested* path say nothing about where a match actually lives.
    """
    try:
        base = (cwd / root).resolve() if root else cwd.resolve()
        if not base.is_dir():
            return ExecutionOutcome(False, "", f"not a directory: {base}")
        matches: list[str] = []
        for path in base.rglob(pattern):
            try:
                resolved = path.resolve()
            except OSError:  # pragma: no cover - broken symlink
                continue
            # Containment is checked on the RESOLVED path, so a symlink
            # pointing outside the root cannot smuggle a match through.
            if resolved != base and base not in resolved.parents:
                continue
            matches.append(str(resolved.relative_to(base)))
        if not matches:
            return ExecutionOutcome(True, "(no matches)")
        return ExecutionOutcome(True, "\n".join(sorted(matches)[:200]))
    except Exception as exc:  # pragma: no cover
        return ExecutionOutcome(False, "", f"{type(exc).__name__}: {exc}")


def _grep(pattern: str, *, path_str: str, include_glob: str | None, cwd: Path) -> ExecutionOutcome:
    """Search file contents under ``path_str``.

    Every candidate is resolved and checked to be inside the search root before
    its contents are read. The requested path being in scope says nothing about
    where a match actually lives: a symlink whose *name* sits inside the root can
    point anywhere, and reading it would return content from outside the scope
    the host authorised. The dispatcher validates the requested path only, so the
    containment check belongs here too -- the same invariant ``_glob`` applies
    for the same reason.
    """
    try:
        rx = re.compile(pattern)
        root = _resolve(path_str, cwd=cwd)
        if not root.exists():
            return ExecutionOutcome(False, "", f"path not found: {root}")
        base = root.resolve()
        if root.is_file():
            # A single explicitly-targeted file: the caller already named it and
            # the dispatcher's scope gate validated it.
            files: Iterable[Path] = [root]
        else:
            # Walk the *unresolved* root so reported paths stay relative to the
            # caller's cwd exactly as before; only the containment decision uses
            # the resolved form.
            files = list(root.rglob(include_glob)) if include_glob else list(root.rglob("*"))
        lines: list[str] = []
        unreadable = 0
        for fp in files:
            if not fp.is_file():
                continue
            try:
                resolved = fp.resolve()
            except OSError:  # pragma: no cover - broken symlink
                continue
            if resolved != base and base not in resolved.parents:
                # Containment is checked on the RESOLVED path, so a symlink
                # pointing outside the root cannot smuggle its contents out.
                continue
            try:
                content = resolved.read_text(encoding="utf-8", errors="replace")
            except Exception:
                # A file we could not read is a file we did not search. The
                # exclusion just above is a deliberate scope decision; this is
                # a failed read, so it is counted rather than dropped and the
                # outcome can report that the sweep was incomplete.
                unreadable += 1
                continue
            for lineno, line in enumerate(content.splitlines(), start=1):
                if rx.search(line):
                    rel = fp.relative_to(cwd)
                    lines.append(f"{rel}:{lineno}: {line}")
                    if len(lines) >= 200:
                        break
            if len(lines) >= 200:
                lines.append("... (truncated)")
                break
        # A search that skipped files is not a search that found nothing.
        # Reporting "(no matches)" either way tells a reader the corpus was
        # searched when part of it was never opened, which is how a tool ends
        # up asserting that a symbol does not exist.
        note = (
            f"[{unreadable} file(s) could not be read; this search is incomplete]"
            if unreadable
            else ""
        )
        if not lines:
            return ExecutionOutcome(True, f"(no matches) {note}".strip(), unreadable_count=unreadable)
        if note:
            lines.append(note)
        return ExecutionOutcome(True, "\n".join(lines), unreadable_count=unreadable)
    except Exception as exc:  # pragma: no cover
        return ExecutionOutcome(False, "", f"{type(exc).__name__}: {exc}")


def execute_tool(
    call: ToolCall,
    *,
    cwd: Path | None = None,
) -> ToolResult:
    """Execute one tool call locally and return a :class:`ToolResult`.

    The ``call.id`` is preserved on the result so the agent loop can
    match it back to the model-emitted ``tool_call``. The function
    catches every exception so the agent loop never has to deal with
    raw Python errors.
    """
    working_dir = cwd or Path(os.getcwd())
    args = call.arguments
    # Tool ids are matched case-insensitively so the model-emitted
    # ``"Read"`` and the registry's ``"read"`` agree.
    name = str(call.tool_id).lower()
    started = time.perf_counter()
    if name == "read":
        outcome = _read(
            str(args.get("path", "")),
            offset=args.get("offset"),
            limit=args.get("limit"),
            cwd=working_dir,
        )
    elif name == "edit":
        outcome = _edit(
            str(args.get("path", "")),
            old_string=str(args.get("old_string", "")),
            new_string=str(args.get("new_string", "")),
            cwd=working_dir,
        )
    elif name == "write":
        outcome = _write(
            str(args.get("path", "")),
            content=str(args.get("content", "")),
            cwd=working_dir,
        )
    elif name == "bash":
        outcome = _bash(
            str(args.get("command", "")),
            timeout_seconds=args.get("timeout_seconds"),
            cwd=working_dir,
        )
    elif name == "glob":
        outcome = _glob(
            str(args.get("pattern", "*")),
            cwd=working_dir,
            root=str(args.get("path")) if args.get("path") else None,
        )
    elif name == "grep":
        outcome = _grep(
            str(args.get("pattern", "")),
            path_str=str(args.get("path", ".")),
            include_glob=args.get("include_glob"),
            cwd=working_dir,
        )
    else:
        outcome = ExecutionOutcome(False, "", f"unsupported tool: {call.tool_id}")
    return ToolResult(
        call_id=call.id,
        ok=outcome.ok,
        output=outcome.output,
        error=outcome.error,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


__all__ = [
    "default_tool_definitions",
    "execute_tool",
    "to_openai_wire",
]
