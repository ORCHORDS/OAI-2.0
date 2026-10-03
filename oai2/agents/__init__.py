"""Multi-agent orchestration scaffold + live agent loop.

Public surface
--------------

- :class:`AgentLoop` — multi-step tool-use loop driven by any
  :class:`~oai2.runtime.InferenceRuntime` (defaults to the env-selected
  gateway runtime; falls back to :class:`PlaceholderRuntime` in CI).
  Optionally consults a :class:`~oai2.knowledge.KnowledgeStore` to
  inject a compact evidence package into the model context per #22/#23.
- :class:`AgentRun` / :class:`AgentStep` — result shapes.
- :func:`default_tool_definitions` — the canonical six tools
  (``Read`` / ``Edit`` / ``Write`` / ``Bash`` / ``Glob`` / ``Grep``).
- :func:`default_system_prompt` — host-side system prompt.
- :class:`AgentSpec` / :class:`Orchestrator` / :class:`OrchestratorContext`
  — SWARM-shaped scaffold (still PROPOSED; live spawn is future work).
- :func:`oai2.agents.learning.extract_lesson` / :func:`record_failure` —
  verified-lesson (#87) and negative-memory (#88) bridges into the
  canonical :class:`KnowledgeStore` abstraction.
- :func:`oai2.agents.instructions.resolve` / :class:`Instruction` /
  :class:`Precedence` — instruction precedence, trust classes and conflict
  resolution (#185 / ``WI-PROMPT-001``). Pure and deterministic; not yet
  wired into :class:`AgentLoop` (#186).
- :func:`oai2.agents.composer.compose` / :class:`PrefixSpec` /
  :class:`Composition` — stable versioned prefix, goal, optional state delta and
  fenced evidence, with per-segment provenance (#186 / ``WI-PROMPT-002``).
"""

from __future__ import annotations

from ..tools.registry import (
    default_tool_definitions,
    execute_tool,
    to_openai_wire,
)
from .agent_loop import (
    AgentLoop,
    AgentRun,
    AgentStep,
    EvidenceStatus,
    build_default_gateway,
    build_default_runtime,
    default_dispatch_policy,
    default_system_prompt,
)
from .composer import (
    COMPOSER_SCHEMA_VERSION,
    CompactRef,
    Composition,
    CompositionProvenance,
    PrefixSpec,
    RefResolver,
    Segment,
    SegmentKind,
    assert_template_safe,
    compose,
)
from .instructions import (
    INSTRUCTION_SCHEMA_VERSION,
    Conflict,
    Instruction,
    Precedence,
    Rejection,
    RejectionReason,
    ResolutionTrace,
    ResolvedInstructions,
    resolve,
    trusted_instruction,
    untrusted_content,
)
from .learning import extract_lesson, record_failure, sanitize_text
from .orchestration import AgentSpec, Orchestrator, OrchestratorContext

__all__ = [
    "COMPOSER_SCHEMA_VERSION",
    "AgentLoop",
    "AgentRun",
    "AgentStep",
    "AgentSpec",
    "CompactRef",
    "Composition",
    "CompositionProvenance",
    "Conflict",
    "EvidenceStatus",
    "INSTRUCTION_SCHEMA_VERSION",
    "Instruction",
    "PrefixSpec",
    "RefResolver",
    "Segment",
    "SegmentKind",
    "Orchestrator",
    "OrchestratorContext",
    "Precedence",
    "Rejection",
    "RejectionReason",
    "ResolvedInstructions",
    "ResolutionTrace",
    "assert_template_safe",
    "build_default_gateway",
    "build_default_runtime",
    "default_dispatch_policy",
    "default_system_prompt",
    "compose",
    "default_tool_definitions",
    "execute_tool",
    "extract_lesson",
    "record_failure",
    "resolve",
    "sanitize_text",
    "to_openai_wire",
    "trusted_instruction",
    "untrusted_content",
]
