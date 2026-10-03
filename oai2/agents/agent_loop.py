"""Live agent loop that drives oai-2.0 through real tool use.

The loop wires the existing pieces together:

- :class:`oai2.runtime.GatewayRuntime` (or any ``InferenceRuntime``)
  for the model call.
- :mod:`oai2.tools.registry` for the host-side tool definitions and
  the local execution handlers.
- :class:`oai2.tools.ToolDispatcher` to validate every model-emitted
  ``tool_call`` against the 6-gate policy pipeline (TOOL_CALLING.md).

The loop sends an OpenAI-style ``tools`` array on every request so the
model actually knows tools exist; without that block, oai-2.0 writes
prose instead of ``tool_calls`` — that is the root cause of
sess_10cbe33c-d83b-42ce-bf2c producing zero tool use across 34 model
turns.

Status: EXPERIMENTAL (added for WI-AGT-001 / sess_10cbe33c-d83b-42ce-bf2c).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core import Status, ToolId
from ..knowledge import (
    EvidencePackage,
    KnowledgeStore,
    RetrievalRequest,
    build_evidence_package,
)
from ..protocols import (
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from ..runtime import (
    GatewayRuntime,
    InferenceRequest,
    InferenceResponse,
    InferenceRuntime,
    select_runtime_from_env,
)
from ..tools import (
    DispatchPolicy,
    ToolDispatcher,
    default_tool_definitions,
    execute_tool,
    to_openai_wire,
)
from .composer import Composition, PrefixSpec, compose

#: Bumped when the built-in system prompt's meaning changes. It is part of
#: the prefix identity, so editing the prompt without bumping this would let
#: a prefix KV cache reuse state rendered under the previous text.
BUILTIN_POLICY_VERSION = "1"

#: Package version of the default tool set. Part of the prefix identity for
#: the same reason: the tool wire is rendered into the prompt by the chat
#: template, so a schema change is a prefix change.
BUILTIN_TOOL_SCHEMA_VERSION = "1"


@dataclass(slots=True, frozen=True)
class AgentStep:
    """One model turn + the tool call(s) it produced + their results."""

    step_index: int
    response: InferenceResponse
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()


@dataclass(slots=True)
class AgentRun:
    """End state of an agent loop invocation."""

    user_prompt: str
    final_text: str
    steps: list[AgentStep] = field(default_factory=list)
    total_tool_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    finished_reason: str | None = None


# ---------------------------------------------------------------------------
# Loaders / builders
# ---------------------------------------------------------------------------


def default_dispatch_policy(
    *,
    allow_capabilities: Iterable[str] | None = None,
    resource_scopes: Iterable[str] | None = None,
    allow_unscoped_capabilities: Iterable[str] | None = None,
    budget_calls: int = 64,
) -> DispatchPolicy:
    """Permissive policy that admits the six default tools.

    All ``default_tool_definitions`` capabilities are allowed by
    default; supply ``allow_capabilities`` to narrow the surface.

    ``allow_unscoped_capabilities`` names the capabilities permitted to act
    without a resource scope. Only ``Bash`` needs it: it takes a command
    string and so has no path for gate 4 to constrain. ``Glob`` and ``Grep``
    both take a ``path`` and are scoped like ``Read``/``Edit``/``Write``. A
    host that wants its scopes to bind everything it admits should omit this
    and accept that ``Bash`` is then refused.

    **The scopes do not bound what a shell can reach.** ``shell.exec`` is
    granted unscoped by default, and ``Bash("cat ~/.ssh/id_rsa")`` carries no
    path for the scope gate to test. ``resource_scopes`` constrains the
    *path-taking* tools; it is not a sandbox around the agent as a whole. A
    host that needs a real boundary should deny ``shell.exec``.

    The defaults are ``{"./", "/tmp"}`` — the working directory the host
    chose, and the shared temp area. They deliberately do **not** include the
    user's home directory. The home directory holds the credentials that
    matter most on a developer machine (``.ssh/id_rsa``,
    ``.aws/credentials``, ``.config/gh/hosts.yml``, cloud and registry
    tokens), and admitting it wholesale means the default policy authorises
    reading and overwriting every one of them on behalf of a model. That is
    not a permissive default, it is the absence of one.

    A host that genuinely wants home-directory access — an interactive coding
    assistant, say — can still ask for it, but it now has to be said out
    loud::

        default_dispatch_policy(resource_scopes=["./", "~/"])
    """
    if allow_capabilities is None:
        allow_capabilities = {"fs.read", "fs.write", "fs.list", "shell.exec"}
    if allow_unscoped_capabilities is None:
        allow_unscoped_capabilities = {"shell.exec"}
    return DispatchPolicy(
        allow_capabilities=frozenset(allow_capabilities),
        deny_capabilities=frozenset(),
        resource_scopes=frozenset(resource_scopes or {"./", "/tmp"}),
        allow_unscoped_capabilities=frozenset(allow_unscoped_capabilities),
        budget_calls=budget_calls,
        high_impact_approved=True,
    )


def default_system_prompt() -> str:
    """Host-side system prompt that primes oai-2.0 for tool use."""
    return (
        "You are OAI-2.0, a coding and engineering agent.\n"
        "When the user asks you to inspect or modify code, you MUST use the "
        "available tools (Read, Edit, Write, Bash, Glob, Grep) instead of "
        "guessing or writing essays. Do not invent file contents, do not "
        "claim to have done work you have not done. If a tool call returns "
        "an error, surface the error to the user verbatim before trying "
        "again. Prefer Read/Glob/Grep before Edit/Write. Use Bash for "
        "builds, tests, and git operations.\n"
    )


def build_default_runtime() -> InferenceRuntime:
    """Pick the best live runtime for the agent loop."""
    return select_runtime_from_env()


def build_default_gateway() -> GatewayRuntime:
    """Return a :class:`GatewayRuntime` for direct tool-call probing.

    Raises if no ``OAI2_GATEWAY_API_KEY`` is set in the environment.
    """
    from ..runtime.gateway_runtime import GatewayConfigError
    try:
        return GatewayRuntime.from_env()
    except GatewayConfigError as exc:
        raise RuntimeError(
            "OAI2_GATEWAY_API_KEY not set; cannot build default gateway"
        ) from exc


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


class AgentLoop:
    """Multi-step tool-use loop with policy-validated execution.

    Parameters
    ----------
    runtime:
        Any :class:`InferenceRuntime`. The default is the
        env-selected runtime (gateway when ``OAI2_GATEWAY_API_KEY``
        is set, otherwise the deterministic placeholder).
    tools:
        Host-side tool definitions. Defaults to the six canonical tools.
    policy:
        :class:`DispatchPolicy` controlling which tool capabilities are
        admitted. Defaults to a permissive policy that admits all six.
    cwd:
        Working directory for ``Read``/``Edit``/``Write``/``Bash``.
    max_steps:
        Maximum number of model turns before the loop gives up. Prevents
        runaway recursion in hostile or hallucinating states.
    model_id:
        Display name included in the system prompt and per-step notes.
    """

    def __init__(
        self,
        *,
        runtime: InferenceRuntime | None = None,
        tools: tuple[ToolDefinition, ...] | None = None,
        policy: DispatchPolicy | None = None,
        cwd: Path | None = None,
        max_steps: int = 8,
        model_id: str = "oai-2.0",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        executor: Callable[[ToolCall], ToolResult] | None = None,
        knowledge_store: KnowledgeStore | None = None,
        evidence_budget_tokens: int = 1024,
        token_counter: Callable[[str], int] | None = None,
        project_prompt: str = "",
        project_version: str = "0",
    ) -> None:
        self._runtime: InferenceRuntime = runtime or build_default_runtime()
        self._tools: tuple[ToolDefinition, ...] = tools or default_tool_definitions()
        self._cwd = cwd or Path.cwd()
        self._dispatcher = ToolDispatcher(
            registry=self._tools,
            policy=policy or default_dispatch_policy(),
            cwd=self._cwd,
        )
        self._max_steps = max_steps
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._executor: Callable[[ToolCall], ToolResult] = executor or (
            lambda call: execute_tool(call, cwd=self._cwd)
        )
        self._knowledge_store: KnowledgeStore | None = knowledge_store
        if (
            isinstance(evidence_budget_tokens, bool)
            or not isinstance(evidence_budget_tokens, int)
            or evidence_budget_tokens <= 0
        ):
            raise ValueError("evidence_budget_tokens must be a positive integer")
        self._evidence_budget_tokens = evidence_budget_tokens
        self._token_counter: Callable[[str], int] = token_counter or (
            lambda text: max(1, len(text.split()))
        )
        # Optional repository guidance (e.g. AGENTS.md). It sits in the
        # prefix region, so it is versioned with the rest of the prefix and
        # changing it invalidates reuse rather than silently re-using KV
        # state rendered under different guidance.
        self._project_prompt = project_prompt
        self._project_version = project_version
        #: The composition used for the most recent :meth:`run`, so a caller
        #: can audit what the model actually saw (and under which digest).
        self.last_composition: Composition | None = None

    @property
    def tools(self) -> tuple[ToolDefinition, ...]:
        return self._tools

    @property
    def dispatcher(self) -> ToolDispatcher:
        return self._dispatcher

    @property
    def knowledge_store(self) -> KnowledgeStore | None:
        return self._knowledge_store

    def _build_evidence_bodies(self, user_prompt: str) -> tuple[str, ...]:
        """Retrieve eligible knowledge and return bodies for the composer.

        Returns an empty tuple when no knowledge store is wired, retrieval
        yields no eligible candidates, or the evidence package is empty.
        Never raises (REQ-LEARN-016 of #87: extraction failure must not
        affect task completion).

        The bodies are handed to :func:`~oai2.agents.composer.compose`, which
        owns placement and the precedence framing. The loop deliberately does
        not build the evidence message itself: two owners of that block is
        exactly how evidence gets injected twice.
        """
        if self._knowledge_store is None:
            return ()
        try:
            # Per #88 (REQ-LEARN-021), negative memory is representable
            # as knowledge — include PROPOSED in the retrieval set. The
            # topic prefix `oai2:negative:` and the content prefix
            # `diagnostic:` are the markers that distinguish negative
            # memory from positive instruction; the model can recognize
            # both. Blocking PROPOSED here would silently swallow
            # negative memory, which is the opposite of what #88 wants.
            request = RetrievalRequest(
                topic=user_prompt,
                limit=8,
                include_status=(
                    Status.IMPLEMENTED,
                    Status.EXPERIMENTAL,
                    Status.PROPOSED,
                ),
            )
            result = self._knowledge_store.retrieve(request)
        except Exception:
            return ()
        if not result.objects:
            return ()
        try:
            package: EvidencePackage = build_evidence_package(
                result,
                token_budget=self._evidence_budget_tokens,
                token_counter=self._token_counter,
            )
        except Exception:
            return ()
        return tuple(entry.render() for entry in package.entries)

    def _prefix_spec(self, policy_text: str) -> PrefixSpec:
        """Derive the stable prefix identity this loop's requests carry."""
        tool_wire = tuple(to_openai_wire(self._tools))
        # Derive the schema version from the wire itself so a host that
        # passes a custom tool set still gets a correct identity.
        schema_version = hashlib.sha256(
            "|".join(json.dumps(t, sort_keys=True, default=str) for t in tool_wire).encode()
        ).hexdigest()[:12]
        return PrefixSpec(
            policy_text=policy_text,
            policy_version=BUILTIN_POLICY_VERSION,
            tool_wire=tool_wire,
            tool_schema_version=schema_version,
            project_text=self._project_prompt,
            project_version=self._project_version,
        )

    def run(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> AgentRun:
        """Drive one user prompt through the multi-step loop."""
        policy_text = system_prompt or default_system_prompt()
        # Composition happens ONCE per run, not once per step. Steps after
        # the first must carry the conversation and tool history forward
        # verbatim; re-composing them would drop the assistant/tool turns
        # that make a multi-step loop work.
        composition = compose(
            prefix=self._prefix_spec(policy_text),
            user_goal=user_prompt,
            evidence=self._build_evidence_bodies(user_prompt),
            evidence_budget_tokens=self._evidence_budget_tokens,
            token_counter=self._token_counter,
        )
        self.last_composition = composition
        messages: list[dict[str, Any]] = [dict(m) for m in composition.messages]
        # The digest is forwarded on every step so the runtime's prefix KV
        # cache keys on this composition rather than re-deriving identity.
        prefix_digest = composition.prefix_digest
        steps: list[AgentStep] = []
        total_tool_calls = 0
        total_input_tokens = 0
        total_output_tokens = 0
        final_text = ""
        finished_reason: str | None = None

        for step_index in range(self._max_steps):
            request = InferenceRequest(
                prompt=user_prompt,
                messages=[dict(m) for m in messages],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                tools=to_openai_wire(self._tools),
                tool_choice="auto",
                prefix_digest=prefix_digest,
            )
            response = self._runtime.generate(request)
            steps.append(AgentStep(step_index=step_index, response=response))
            total_input_tokens += response.tokens  # proxy: see note below

            content = response.text
            raw_tool_calls = response.tool_calls
            finished_reason = response.finish_reason

            if not raw_tool_calls:
                final_text = content
                break

            # Convert model-emitted dicts into ToolCall objects, dispatch
            # through the 6-gate policy pipeline, and execute locally.
            converted_calls = _raw_tool_calls_to_models(raw_tool_calls)
            tool_results: list[ToolResult] = []
            for _raw_call, model_call in zip(raw_tool_calls, converted_calls, strict=True):
                # ``calls_used`` is the count of calls ALREADY dispatched --
                # gate 5 adds one for the prospective call it is judging. The
                # counter is therefore advanced after the check, not before:
                # incrementing first makes the gate compare ``k + 1`` against
                # the budget for the k-th call, so a policy of N admits N - 1
                # and a budget of 1 executes nothing at all.
                decision = self._dispatcher.check(model_call, calls_used=total_tool_calls)
                total_tool_calls += 1
                if decision.stage.value != "execute":
                    tool_results.append(
                        ToolResult(
                            call_id=model_call.id,
                            ok=False,
                            output="",
                            error=f"dispatch:{decision.stage.value}:{decision.reason}",
                        )
                    )
                    continue
                tool_results.append(self._executor(model_call))

            steps[-1] = AgentStep(
                step_index=step_index,
                response=response,
                tool_calls=tuple(converted_calls),
                tool_results=tuple(tool_results),
            )

            # Append the assistant turn (with its tool_calls) and the
            # tool responses so the next model turn sees the results.
            messages.append(
                {
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": list(raw_tool_calls),
                }
            )
            for call, result in zip(raw_tool_calls, tool_results, strict=True):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call.get("id", "")),
                        "content": _format_tool_result(result),
                    }
                )
                total_output_tokens += _approx_tokens(_format_tool_result(result))

            if finished_reason == "stop":
                final_text = content
                break
        else:  # pragma: no cover - exhaustive
            final_text = steps[-1].response.text if steps else ""

        return AgentRun(
            user_prompt=user_prompt,
            final_text=final_text,
            steps=steps,
            total_tool_calls=total_tool_calls,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            finished_reason=finished_reason,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raw_tool_calls_to_models(raw_tool_calls: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> list[ToolCall]:
    """Translate OpenAI-style ``tool_calls`` dicts into :class:`ToolCall` objects."""
    converted: list[ToolCall] = []
    for raw in raw_tool_calls:
        call_id = str(raw.get("id") or uuid.uuid4().hex)
        function = raw.get("function") or {}
        name = str(function.get("name") or raw.get("name") or "")
        # ``arguments`` may arrive as a JSON string or a dict. We only
        # build a dict view here; validation happens in the dispatcher.
        args_raw = function.get("arguments") or raw.get("arguments") or {}
        if isinstance(args_raw, str):
            import json
            try:
                args_dict = json.loads(args_raw)
            except json.JSONDecodeError:
                args_dict = {"_raw": args_raw}
        else:
            args_dict = dict(args_raw)
        converted.append(
            ToolCall(
                id=call_id,
                tool_id=ToolId(name),
                arguments=args_dict,
            )
        )
    return converted


def _format_tool_result(result: ToolResult) -> str:
    """Render a :class:`ToolResult` as the ``content`` of a tool message."""
    if result.ok:
        return str(result.output)
    return f"ERROR: {result.error or 'unknown failure'}"


def _approx_tokens(text: str) -> int:
    return max(1, len(text.split()))


__all__ = [
    "AgentLoop",
    "AgentRun",
    "AgentStep",
    "build_default_gateway",
    "build_default_runtime",
    "default_dispatch_policy",
    "default_system_prompt",
    "default_tool_definitions",
]
