"""WI-LEARN-001 (#87) verified-lesson extraction and
WI-LEARN-002 (#88) negative-memory / conflict handling for the
canonical AgentLoop.

This module is the small bridge between the live :class:`AgentRun`
result and the existing :class:`KnowledgeStore` abstraction. It does
NOT open a Cloudflare connection itself, does NOT auto-mutate any
model weights, and does NOT promote anything into training data
without a separate, explicit gate.

Sanitization: secrets / bearer tokens / private endpoints / raw
terminal transcripts are stripped before any KnowledgeObject is
emitted. Per REQ-LEARN-013 of #87, raw source content is only copied
when explicitly provided and required.

Status: EXPERIMENTAL (added for #87 and #88).
"""

from __future__ import annotations

import re
from typing import Any

from ..core import KnowledgeId, Status
from ..knowledge import (
    KnowledgeObject,
    KnowledgeStore,
    RetrievalRequest,
    sha256_hex,
)

# Patterns that look like secrets or private credentials. Matched on
# the *value*, not the *key name*, to be robust to renamed fields.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),  # OpenAI-style
    re.compile(r"sk-ant-[A-Za-z0-9_-]{16,}"),  # Anthropic-style
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),  # Google API key
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),  # GitHub PAT
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),  # Slack
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"bearer\s+[A-Za-z0-9._-]{16,}", re.IGNORECASE),
)

_REDACTION = "[REDACTED]"


def _sanitize(text: str) -> str:
    """Strip obvious secret-shaped substrings. Best-effort, not a
    security boundary; the real boundary is the agent's host-side
    tool policy (oai2.tools.dispatch).
    """
    out = text
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub(_REDACTION, out)
    return out


def _topic_from_run(agent_run: Any, *, max_chars: int = 120) -> str:
    """Derive a stable, dedup-friendly topic from an AgentRun."""
    raw = (agent_run.user_prompt or "").strip()
    if not raw:
        raw = "oai2:agent_run"
    if len(raw) > max_chars:
        raw = raw[: max_chars - 3] + "..."
    return f"oai2:agent:{raw}"


def _dedup_check(
    store: KnowledgeStore,
    topic: str,
    *,
    limit: int = 4,
) -> tuple[KnowledgeObject, ...]:
    """Return existing knowledge objects with the same topic prefix.

    Conflict / dedup rule (REQ-LEARN-014 / REQ-LEARN-022): if a same-
    topic object already exists, the caller MUST decide whether to
    supersede, version, or keep both. We surface the candidates so
    the caller can do that explicitly.

    A store that raises is NOT treated as "no conflicts". It used to be:
    ``except Exception: return ()`` made an unreachable backend and a
    healthy backend with an empty topic return the identical value, so the
    caller imported a duplicate lesson believing it had checked and found
    nothing. The rule above says the caller must decide -- it cannot decide
    about a conflict it was never told about, so the failure is propagated
    with its original context instead of being flattened into a clean
    result.
    """
    result = store.retrieve(RetrievalRequest(topic=topic, limit=limit))
    return tuple(result.objects)


def extract_lesson(
    agent_run: Any,
    *,
    task_id: str,
    verification_ref: str,
    source_version: str,
    runtime_version: str,
    authority: float = 0.7,
    extra_provenance: dict[str, str] | None = None,
    dedup_against: KnowledgeStore | None = None,
) -> KnowledgeObject:
    """Convert a verified :class:`AgentRun` into a sanitized
    candidate :class:`KnowledgeObject` per #87 REQ-LEARN-011..016.

    This function NEVER calls ``store.put`` automatically. The caller
    is responsible for the final promotion step (sanitization /
    dedup-conflict / Cloudflare write), per #87 REQ-LEARN-014..016
    and the project's separation of "knowledge promotion" and
    "training-candidate promotion" (REQ-LEARN-023 of #88).

    Returns a :class:`KnowledgeObject` with:

    - ``knowledge_id`` derived from task_id + verification_ref (stable
      so dedup can recognize a re-extraction of the same lesson);
    - ``topic`` derived from the user_prompt;
    - ``content`` is a short, sanitized summary of the run — full
      transcripts are NOT copied (REQ-LEARN-013);
    - ``source_uri`` carries the verification_ref + source_version;
    - ``status = IMPLEMENTED`` only when the run completed; otherwise
      ``EXPERIMENTAL`` so the caller can re-verify before promoting.
    """
    # A run counts as finished only if it actually finished. "length" means
    # the model was truncated mid-generation; "tool_calls" is what is left
    # when the loop exhausts max_steps still asking for tools; None means the
    # reason was never recorded. All three are "not finished", and admitting
    # them meant an unfinished run produced an IMPLEMENTED lesson whose content
    # asserts "verified lesson from task_id=...". Measured on f309d81, a real
    # AgentLoop with max_steps=3 and a runtime that always requested a tool
    # call: 3 steps, 3 tool calls, finished_reason="tool_calls", and
    # status=IMPLEMENTED. The docstring above already says IMPLEMENTED is
    # "only when the run completed", so this makes the code match it.
    # Conservative in the right direction: an unverified run is EXPERIMENTAL
    # and the caller re-verifies, which is what that status is for.
    finished = bool(agent_run.final_text) and agent_run.finished_reason == "stop"
    steps = len(agent_run.steps) if hasattr(agent_run, "steps") else 0
    tool_calls = getattr(agent_run, "total_tool_calls", 0)
    content = (
        f"verified lesson from task_id={task_id}\n"
        f"finished_reason={agent_run.finished_reason!r}\n"
        f"steps={steps} tool_calls={tool_calls}\n"
        f"verification_ref={verification_ref}\n"
        f"runtime_version={runtime_version}\n"
        f"summary: {(agent_run.final_text or '').strip()[:280]}"
    )
    content = _sanitize(content)
    kid = KnowledgeId(sha256_hex(f"{task_id}|{verification_ref}|{source_version}")[:32])
    object_status = Status.IMPLEMENTED if finished else Status.EXPERIMENTAL
    obj = KnowledgeObject(
        knowledge_id=kid,
        topic=_topic_from_run(agent_run),
        content=content,
        content_hash=sha256_hex(content),
        source_uri=(
            f"agent_run://{task_id}?verify={verification_ref}"
            f"&source={source_version}&runtime={runtime_version}"
        ),
        retrieved_at=float(getattr(agent_run, "_retrieved_at", 0) or 0) or 0.0,
        authority=max(0.0, min(1.0, authority)),
        status=object_status,
        artifact_ref=verification_ref,
    )
    # Surface dedup candidates; the caller is the gate.
    obj_conflicts = _dedup_check(dedup_against, obj.topic) if dedup_against is not None else ()
    # Attach via __dict__ if dataclass-like, else via a side-channel; here
    # we use object.__setattr__ since KnowledgeObject is a pydantic
    # BaseModel and may forbid extra fields.
    if extra_provenance:
        # Best-effort metadata; ignore if KnowledgeObject schema forbids it.
        for key, value in extra_provenance.items():
            if key in obj.model_fields:
                try:
                    setattr(obj, key, value)
                except (ValueError, TypeError):
                    pass
    # We don't mutate obj for conflicts — caller decides. But we expose
    # them via a tuple attribute on the returned object so tests can
    # assert dedup behavior.
    object.__setattr__(obj, "_dedup_conflicts", obj_conflicts)
    return obj


def record_failure(
    agent_run: Any,
    *,
    task_id: str,
    failure_class: str,
    evidence_ref: str,
    source_version: str,
    runtime_version: str,
    extra_provenance: dict[str, str] | None = None,
    dedup_against: KnowledgeStore | None = None,
) -> KnowledgeObject:
    """Convert a verified failure into a NEGATIVE-memory
    :class:`KnowledgeObject` per #88 REQ-LEARN-021..026.

    Negative memory is intentionally distinguishable from positive
    lessons: ``status = Status.PROPOSED`` and the content uses a
    "diagnostic:" prefix so retrieval-side code can recognize and
    never surface it as positive instruction.

    This function also never calls ``store.put`` automatically.
    """
    steps = len(agent_run.steps) if hasattr(agent_run, "steps") else 0
    tool_calls = getattr(agent_run, "total_tool_calls", 0)
    content = (
        f"diagnostic: verified failure\n"
        f"task_id={task_id}\n"
        f"failure_class={failure_class}\n"
        f"evidence_ref={evidence_ref}\n"
        f"finished_reason={agent_run.finished_reason!r}\n"
        f"steps={steps} tool_calls={tool_calls}\n"
        f"runtime_version={runtime_version}\n"
        f"excerpt: {(agent_run.final_text or '').strip()[:240]}"
    )
    content = _sanitize(content)
    kid = KnowledgeId(sha256_hex(f"neg|{task_id}|{failure_class}|{evidence_ref}")[:32])
    obj = KnowledgeObject(
        knowledge_id=kid,
        topic=f"oai2:negative:{failure_class}:{_topic_from_run(agent_run)}",
        content=content,
        content_hash=sha256_hex(content),
        source_uri=(
            f"agent_run://{task_id}?verify={evidence_ref}"
            f"&source={source_version}&runtime={runtime_version}"
            f"&failure={failure_class}"
        ),
        retrieved_at=0.0,
        authority=0.5,
        status=Status.PROPOSED,
        artifact_ref=evidence_ref,
    )
    if extra_provenance:
        for key, value in extra_provenance.items():
            if key in obj.model_fields:
                try:
                    setattr(obj, key, value)
                except (ValueError, TypeError):
                    pass
    obj_conflicts = (
        _dedup_check(dedup_against, obj.topic) if dedup_against is not None else ()
    )
    object.__setattr__(obj, "_dedup_conflicts", obj_conflicts)
    return obj


__all__ = [
    "extract_lesson",
    "record_failure",
    "sanitize_text",
]


def sanitize_text(text: str) -> str:
    """Public re-export of the sanitization helper for tests / callers
    that want to pre-sanitize before they ever construct an object.
    """
    return _sanitize(text)
