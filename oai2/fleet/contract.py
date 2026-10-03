"""The versioned worker contract (WI-FLEET-001, #242).

This module is the missing *definition* the rest of WP-77 builds on. It is a
wire contract and a capability vocabulary -- deliberately **not** a second
scheduler, gateway, knowledge system or agent protocol. Nothing here holds
authority:

- enrollment, lease and routing authority stay in the existing pipeline
  (REQ-FLEET-016);
- Cloudflare remains the sole knowledge authority (REQ-FLEET-015);
- tool execution stays client-owned (REQ-FLEET-015).

What it owns is the *shape* of what crosses the wire, so that ten downstream
issues stop inventing their own.

## Reuse, not duplication (RISK-FLEET-012)

Every identity in here is borrowed from the existing owner rather than
re-declared, so a change to `oai2.core` or `oai2.protocols` propagates instead
of drifting:

===========================  ==================================
field                        existing owner
===========================  ==================================
``task_id``                  :data:`oai2.core.TaskId`
``agent_id``                 :data:`oai2.core.AgentId`
``evidence_id``              :data:`oai2.core.EvidenceId`
``status``                   :data:`oai2.core.Status`
``tool_results``             :class:`oai2.protocols.ToolResult`
``workload``                 :class:`oai2.evals.qos.WorkloadClass`
===========================  ==================================

Only genuinely new *concepts* are declared here: the capability state
vocabulary, the resource envelope, and the version negotiation.

## The three readiness levels (REQ-FLEET-012)

A worker is not "ready". It is ready *for something*, and conflating the three
is how a placeholder gets promoted to inference:

- ``TRANSPORT_READY`` -- the node completed an authenticated handshake. Says
  nothing about compute.
- ``BACKEND_READY``   -- a named backend is present and executable here.
- ``WORKLOAD_QUALIFIED`` -- this node has actually run *this* workload class
  and is trusted for it.

A node may be transport-ready and workload-qualified for nothing at all. That
is the honest state, and the contract can express it; the previous vocabulary
could not.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..core import AgentId, EvidenceId, Status, TaskId
from ..evals.qos import WorkloadClass
from ..protocols import ToolResult

#: Version of the wire contract itself. Bumped only on a breaking change;
#: :func:`negotiate_version` is what lets a peer discover that rather than
#: discovering it through a silently mis-parsed field.
CONTRACT_VERSION = "1"

#: Contract versions this build speaks. A peer offering something outside this
#: set gets an explicit refusal, not a best-effort parse.
SUPPORTED_CONTRACT_VERSIONS = frozenset({"1"})


class ContractError(ValueError):
    """Raised when a message cannot be honoured under the contract."""


class CapabilityState(StrEnum):
    """What a node can be trusted to do (REQ-FLEET-014).

    The ordering is not a ranking -- ``UNSUPPORTED`` and ``UNAVAILABLE`` are
    both refusals and neither outranks the other. They are distinguished
    because they call for different responses: ``UNSUPPORTED`` means "this
    build will never do it", ``UNAVAILABLE`` means "it could, but not here,
    not now" (no GPU, model not downloaded, memory pressure).
    """

    DETECTED = "detected"
    """The hardware/feature is visible. Proves nothing about executability."""

    EXECUTABLE = "executable"
    """A named backend is present and can run. Not yet qualified."""

    QUALIFIED = "qualified"
    """Actually exercised on this node for this workload. The only state
    that may be used to admit production work."""

    UNSUPPORTED = "unsupported"
    """This build or platform will never provide it."""

    UNAVAILABLE = "unavailable"
    """Could work here, but cannot right now."""


class Readiness(StrEnum):
    """The three distinct questions about a node (REQ-FLEET-012)."""

    NOT_READY = "not_ready"
    """No authenticated handshake yet. The state every node starts in."""

    TRANSPORT_READY = "transport_ready"
    BACKEND_READY = "backend_ready"
    WORKLOAD_QUALIFIED = "workload_qualified"


#: States that may authorise work. Deliberately not derived from the enum's
#: declaration order -- it is an explicit allowlist so that adding a new state
#: cannot accidentally authorise production work by existing.
AUTHORISING_STATES = frozenset({CapabilityState.QUALIFIED})


class BackendVersion(BaseModel):
    """Identity of the artifact and backend that produced a result.

    A result that cannot name its backend and artifact version is not
    reproducible, so these are required rather than optional. ``provider`` and
    ``model`` are identity, never credentials (REQ-FLEET-011).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    artifact_version: str = Field(min_length=1)
    quantization: str | None = None
    revision: str | None = None


class WorkerCapability(BaseModel):
    """One capability claim, with the state that qualifies it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    state: CapabilityState
    detail: str = ""
    backend: BackendVersion | None = None

    @property
    def authorises_work(self) -> bool:
        """Whether this claim may be used to admit production work.

        Only ``QUALIFIED`` qualifies, and a ``QUALIFIED`` claim with no
        backend identity is refused: a node cannot be qualified for work it
        cannot say what ran.
        """
        return self.state in AUTHORISING_STATES and self.backend is not None

    def qualified_for(self, workload: WorkloadClass) -> bool:
        return self.authorises_work and self.name == workload.value


class ResourceEnvelope(BaseModel):
    """What the node claims it can be given, not what it has.

    This is the ceiling the pipeline may schedule against. Reporting it as an
    observation would let a node's self-description become its own admission.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_concurrency: int = Field(default=1, ge=1)
    max_ram_gb: float = Field(gt=0.0)
    max_vram_gb: float = Field(default=0.0, ge=0.0)
    cpu_count: int = Field(default=1, ge=1)

    @field_validator("max_ram_gb", "max_vram_gb")
    @classmethod
    def _finite(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("resource envelope must be finite")
        return value


class NodeIdentity(BaseModel):
    """Who is speaking, and under which contract version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    contract_version: str = CONTRACT_VERSION
    transport_ready: bool = False

    @property
    def readiness(self) -> Readiness:
        """The strongest readiness this identity can assert.

        Only transport readiness is derivable here. Backend and workload
        readiness are claims the node makes about specific backends and
        workloads -- see :class:`WorkerCapability` -- and inferring them from
        the mere fact that a node exists is how an unenrolled node comes to
        look qualified.
        """
        return (
            Readiness.TRANSPORT_READY if self.transport_ready else Readiness.NOT_READY
        )


class JobContract(BaseModel):
    """One unit of work, as the pipeline hands it to a worker.

    Cancellation is carried as data rather than as an out-of-band signal, so a
    cancelled job is a state a late result can be checked against instead of a
    race the worker wins or loses.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: TaskId
    attempt: int = Field(default=1, ge=1)
    agent_id: AgentId | None = None
    workload: WorkloadClass
    required_capabilities: tuple[str, ...] = ()
    envelope: ResourceEnvelope
    backend: BackendVersion
    cancelled: bool = False
    deadline_ms: float | None = Field(default=None, ge=0.0)
    contract_version: str = CONTRACT_VERSION


class ResultContract(BaseModel):
    """What the worker returns, reusing the existing result identities."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: TaskId
    attempt: int = Field(ge=1)
    status: Status
    output: str = ""
    error: str | None = None
    evidence_id: EvidenceId | None = None
    tool_results: tuple[ToolResult, ...] = ()
    backend: BackendVersion | None = None
    contract_version: str = CONTRACT_VERSION

    @field_validator("tool_results")
    @classmethod
    def _unique_tool_result_ids(cls, value: tuple[ToolResult, ...]) -> tuple[ToolResult, ...]:
        ids = [item.call_id for item in value]
        if len(set(ids)) != len(ids):
            raise ContractError("duplicate tool result call_id in one result")
        return value

    @property
    def call_ids(self) -> tuple[str, ...]:
        """Correlations back to the :class:`~oai2.protocols.ToolCall` ids.

        Note what is deliberately *not* here: tool identity. ``ToolResult``
        carries ``call_id`` only -- the tool name lives on the originating
        ``ToolCall``. Reusing the existing result type unchanged (RISK-FLEET-
        012) means the worker result cannot name which tool it ran on its own;
        a consumer that needs that must hold the calls and join on ``call_id``.
        Adding a ``tool_id`` to the result would duplicate a field the
        existing owner already defines, and the two would drift.
        """
        return tuple(item.call_id for item in self.tool_results)

    def accepted(self, job: JobContract) -> bool:
        """Whether the pipeline may accept this result for this job.

        Deliberately explicit rather than a caller convention: a result that
        arrives for a cancelled job, a superseded attempt, or a contract it
        does not speak must be refused by something, and this is it.
        """
        if self.task_id != job.task_id:
            return False
        if self.attempt != job.attempt:
            return False
        if job.cancelled:
            return False
        if self.contract_version != job.contract_version:
            return False
        return self.status is not Status.PROPOSED


# --------------------------------------------------------------------------
# Secrets (REQ-FLEET-011: "without embedding secrets")
# --------------------------------------------------------------------------

#: Field names that must never appear in a contract. The contract is
#: serialised onto the wire and stored in evidence, so a credential that
#: travels here is a credential in a log file.
_FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "password", "passwd", "secret", "token", "api_key", "apikey",
        "authorization", "auth", "credential", "credentials", "private_key",
        "session_token", "access_key", "bearer",
    }
)

#: Key-shaped patterns, for fields named something innocuous.
_SECRET_VALUE = re.compile(
    r"(?i)(sk-[A-Za-z0-9]{16,}|-----BEGIN[A-Z ]*PRIVATE KEY-----|ghp_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,})"
)


def assert_no_secrets(payload: object) -> None:
    """Refuse a contract payload that carries a credential.

    Checked on the serialised form rather than the model, so it also covers
    values that arrive nested inside a tool argument dictionary.
    """
    import json

    try:
        text = json.dumps(payload, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        raise ContractError("contract payload is not serialisable") from None

    for name in _iter_keys(payload):
        if name.lower() in _FORBIDDEN_FIELD_NAMES:
            raise ContractError(f"contract must not carry a credential field: {name!r}")
    hit = _SECRET_VALUE.search(text)
    if hit:
        # The value is never echoed; only the fact that one was present.
        raise ContractError("contract payload matched a credential-shaped value")


def _iter_keys(payload: object) -> list[str]:
    keys: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.append(str(key))
            keys.extend(_iter_keys(value))
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            keys.extend(_iter_keys(item))
    return keys


# --------------------------------------------------------------------------
# Version negotiation (REQ-FLEET-015: negotiate, never silently coerce)
# --------------------------------------------------------------------------


def negotiate_version(
    offered: str,
    supported: frozenset[str] = SUPPORTED_CONTRACT_VERSIONS,
) -> str:
    """Agree a contract version, or refuse.

    A peer offering an unknown version is refused outright. Guessing a
    compatible subset is how a field that moved gets read as the field that
    stayed, and the resulting corruption is invisible at the point of damage.
    """
    if offered not in supported:
        raise ContractError(
            f"contract version {offered!r} is not supported; this build speaks "
            f"{sorted(supported)}"
        )
    return offered


def check_compatible(message_version: str, required: str) -> str:
    """Assert a message's version matches what the sender requires."""
    if message_version != required:
        raise ContractError(
            f"message speaks contract {message_version!r}, expected {required!r}"
        )
    return message_version


__all__ = [
    "AUTHORISING_STATES",
    "BackendVersion",
    "CONTRACT_VERSION",
    "CapabilityState",
    "ContractError",
    "JobContract",
    "NodeIdentity",
    "Readiness",
    "ResourceEnvelope",
    "ResultContract",
    "SUPPORTED_CONTRACT_VERSIONS",
    "WorkerCapability",
    "assert_no_secrets",
    "check_compatible",
    "negotiate_version",
]
