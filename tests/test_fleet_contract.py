"""WI-FLEET-001: the versioned worker contract (Refs #242).

These exercise the contract as a *wire* format. AC-FLEET-012 asks for
round-tripping current structured requests, tool IDs/results, cancellation
and errors, and version mismatch behaviour, and that is what the
round-trip and negotiation tests below do.

Nothing here needs a worker, a GPU, or a network. That is deliberate: the
contract has to be checkable on any machine, or ten downstream issues will
each invent their own shape and assert it locally.
"""

from __future__ import annotations

import json

import pytest

from oai2.core import AgentId, EvidenceId, Status, TaskId
from oai2.evals.qos import WorkloadClass
from oai2.fleet import (
    CONTRACT_VERSION,
    SUPPORTED_CONTRACT_VERSIONS,
    BackendVersion,
    CapabilityState,
    ContractError,
    JobContract,
    NodeIdentity,
    Readiness,
    ResourceEnvelope,
    ResultContract,
    WorkerCapability,
    assert_no_secrets,
    check_compatible,
    negotiate_version,
)
from oai2.protocols import ToolCall, ToolResult

#: The artifact every backend identity in this file names. A promoted result
#: that cannot say what ran is not reproducible, so this is required rather
#: than defaulted.
BACKEND = BackendVersion(
    provider="mlx",
    model="qwen3-4b-thinking-2507",
    backend="mlx_lm",
    artifact_version="4b-thinking-2507-Q4_K_M",
    quantization="Q4_K_M",
)

ENVELOPE = ResourceEnvelope(max_concurrency=2, max_ram_gb=32.0, max_vram_gb=0.0, cpu_count=10)


def _job(**overrides: object) -> JobContract:
    payload: dict[str, object] = {
        "task_id": TaskId("task-1"),
        "workload": WorkloadClass.NORMAL,
        "envelope": ENVELOPE,
        "backend": BACKEND,
    }
    payload.update(overrides)
    return JobContract(**payload)  # type: ignore[arg-type]


def _result(**overrides: object) -> ResultContract:
    payload: dict[str, object] = {
        "task_id": TaskId("task-1"),
        "attempt": 1,
        "status": Status.IMPLEMENTED,
        "output": "done",
        "evidence_id": EvidenceId("ev-1"),
    }
    payload.update(overrides)
    return ResultContract(**payload)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# REQ-FLEET-011: the contract carries the required identities
# ---------------------------------------------------------------------------


def test_contract_declares_a_version() -> None:
    assert CONTRACT_VERSION == "1"
    assert CONTRACT_VERSION in SUPPORTED_CONTRACT_VERSIONS


def test_job_contract_carries_every_required_identity() -> None:
    job = _job(agent_id=AgentId("agent-7"), attempt=3, deadline_ms=1500.0)
    assert job.task_id == "task-1"
    assert job.agent_id == "agent-7"
    assert job.attempt == 3
    assert job.deadline_ms == 1500.0
    assert job.contract_version == CONTRACT_VERSION
    # Artifact/backend version is identity, never optional.
    assert job.backend.artifact_version == "4b-thinking-2507-Q4_K_M"
    assert job.backend.backend == "mlx_lm"


def test_resource_envelope_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        ResourceEnvelope(max_concurrency=0, max_ram_gb=1.0)
    with pytest.raises(ValueError):
        ResourceEnvelope(max_concurrency=1, max_ram_gb=0.0)
    with pytest.raises(ValueError):
        ResourceEnvelope(max_concurrency=1, max_ram_gb=float("inf"))
    with pytest.raises(ValueError):
        ResourceEnvelope(max_concurrency=1, max_ram_gb=1.0, max_vram_gb=-1.0)


def test_attempt_must_be_positive() -> None:
    with pytest.raises(ValueError):
        _job(attempt=0)


# ---------------------------------------------------------------------------
# REQ-FLEET-014: a placeholder must never qualify as inference
# ---------------------------------------------------------------------------


class TestCapabilityVocabulary:
    def test_detected_and_executable_never_authorise_work(self) -> None:
        """The two states a node reaches by existing and by having hardware.

        Neither is evidence that this node has run this workload, so neither
        may admit production work. `DETECTED` in particular is what a machine
        with the right GPU but no model and no exercise reports.
        """
        detected = WorkerCapability(name="NORMAL", state=CapabilityState.DETECTED)
        executable = WorkerCapability(
            name="NORMAL", state=CapabilityState.EXECUTABLE, backend=BACKEND
        )
        assert detected.authorises_work is False
        assert executable.authorises_work is False

    def test_qualified_without_a_backend_cannot_authorise_work(self) -> None:
        """A node cannot be qualified for work it cannot say what ran.

        This is the placeholder hole. Without it, a stub that reports
        `QUALIFIED` with no backend behind it would be admitted, and the
        promotion evidence would name nothing reproducible.
        """
        stub = WorkerCapability(name="NORMAL", state=CapabilityState.QUALIFIED)
        assert stub.state is CapabilityState.QUALIFIED
        assert stub.authorises_work is False

    def test_qualified_with_a_backend_authorises_work(self) -> None:
        real = WorkerCapability(
            name="NORMAL", state=CapabilityState.QUALIFIED, backend=BACKEND
        )
        assert real.authorises_work is True
        assert real.qualified_for(WorkloadClass.NORMAL) is True
        assert real.qualified_for(WorkloadClass.DEEP) is False

    def test_unsupported_and_unavailable_are_distinguished(self) -> None:
        """Both refuse, and they call for different responses.

        `UNSUPPORTED` means this build will never do it; `UNAVAILABLE` means
        it could, but not here, not now. Collapsing them loses the difference
        between "get another node" and "wait or fix this one".
        """
        assert CapabilityState.UNSUPPORTED is not CapabilityState.UNAVAILABLE
        for state in (CapabilityState.UNSUPPORTED, CapabilityState.UNAVAILABLE):
            cap = WorkerCapability(name="NORMAL", state=state, backend=BACKEND)
            assert cap.authorises_work is False

    def test_adding_a_new_state_does_not_authorise_work(self) -> None:
        """`AUTHORISING_STATES` is an allowlist, not a derivation.

        Anyone adding a state to the enum gets no authority by existing,
        which is the safe default and the reason it is written as a set.
        """
        from oai2.fleet import AUTHORISING_STATES

        assert AUTHORISING_STATES == {CapabilityState.QUALIFIED}
        assert CapabilityState.DETECTED not in AUTHORISING_STATES


class TestReadinessIsNotOneThing:
    def test_a_node_that_exists_is_not_workload_qualified(self) -> None:
        node = NodeIdentity(node_id="n1", session_id="s1", transport_ready=True)
        assert node.readiness is Readiness.TRANSPORT_READY
        assert node.readiness is not Readiness.WORKLOAD_QUALIFIED
        assert node.readiness is not Readiness.BACKEND_READY

    def test_a_node_that_has_not_connected_reads_as_not_ready(self) -> None:
        node = NodeIdentity(node_id="n1", session_id="s1")
        assert node.transport_ready is False
        assert node.readiness is Readiness.NOT_READY


# ---------------------------------------------------------------------------
# AC-FLEET-012: round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_job_survives_serialisation(self) -> None:
        job = _job(agent_id=AgentId("a"), attempt=2, required_capabilities=("NORMAL",))
        wire = job.model_dump_json()
        assert JobContract.model_validate_json(wire) == job

    def test_result_survives_serialisation(self) -> None:
        result = _result()
        assert ResultContract.model_validate_json(result.model_dump_json()) == result

    def test_tool_results_round_trip_and_keep_their_correlation(self) -> None:
        """Reuses `oai2.protocols.ToolResult` unchanged (RISK-FLEET-012).

        Tool identity is NOT in the result -- it lives on the originating
        `ToolCall` -- so the contract exposes `call_id` and documents that a
        consumer needing the tool name must join against the calls.
        """
        call = ToolCall(id="call-1", tool_id="Grep", arguments={"pattern": "x"})
        results = (
            ToolResult(call_id=call.id, ok=True, output="2 matches"),
            ToolResult(call_id="call-2", ok=False, error="exit 1"),
        )
        result = _result(tool_results=results)

        wire = result.model_dump_json()
        restored = ResultContract.model_validate_json(wire)
        assert restored == result
        assert restored.call_ids == ("call-1", "call-2")
        assert json.loads(wire)["tool_results"][1]["error"] == "exit 1"

    def test_duplicate_tool_result_ids_are_refused(self) -> None:
        """One result cannot answer the same call twice.

        Two results for one `call_id` means the consumer cannot tell which is
        authoritative, and the ambiguity is silent.
        """
        with pytest.raises(ValueError):
            _result(
                tool_results=(
                    ToolResult(call_id="c1", ok=True, output="a"),
                    ToolResult(call_id="c1", ok=False, error="b"),
                )
            )

    def test_unknown_fields_are_refused_not_ignored(self) -> None:
        """`extra="forbid"`: an unrecognised field must not be dropped.

        Silently dropping a field is how a moved field gets read as absent
        and the loss is invisible at the point of damage.
        """
        with pytest.raises(ValueError):
            ResultContract.model_validate(
                {
                    "task_id": "task-1",
                    "attempt": 1,
                    "status": "IMPLEMENTED",
                    "totally_new_field": 1,
                }
            )


class TestResultAcceptance:
    def test_a_matching_result_is_accepted(self) -> None:
        assert _result().accepted(_job()) is True

    def test_a_result_for_a_cancelled_job_is_refused(self) -> None:
        """Cancellation is data the result is checked against, not a race.

        A cancelled job whose worker was already mid-flight will still return
        something. Accepting it would let a slow node overwrite a decision the
        pipeline already made.
        """
        job = _job(cancelled=True)
        assert _result().accepted(job) is False

    def test_a_result_for_a_superseded_attempt_is_refused(self) -> None:
        assert _result(attempt=2).accepted(_job(attempt=3)) is False

    def test_a_result_for_another_task_is_refused(self) -> None:
        assert _result(task_id=TaskId("task-2")).accepted(_job()) is False

    def test_a_result_speaking_another_contract_version_is_refused(self) -> None:
        assert _result(contract_version="0").accepted(_job()) is False

    def test_a_result_that_is_only_proposed_is_refused(self) -> None:
        """`PROPOSED` means design only. Accepting it is the whole defect
        class this repository has been removing."""
        assert _result(status=Status.PROPOSED).accepted(_job()) is False


# ---------------------------------------------------------------------------
# REQ-FLEET-015: negotiate the version, never silently coerce
# ---------------------------------------------------------------------------


class TestVersionNegotiation:
    def test_a_supported_version_is_agreed(self) -> None:
        assert negotiate_version("1") == "1"

    def test_an_unknown_version_is_refused_not_guessed(self) -> None:
        """Guessing a compatible subset is how a moved field is read as the
        field that stayed, and the corruption is invisible where it matters."""
        for offered in ("0", "2", "", "1.0", "one"):
            with pytest.raises(ContractError):
                negotiate_version(offered)

    def test_version_mismatch_on_a_message_is_refused(self) -> None:
        assert check_compatible("1", "1") == "1"
        with pytest.raises(ContractError):
            check_compatible("2", "1")

    def test_a_job_carries_the_version_it_was_negotiated_under(self) -> None:
        job = _job(contract_version=negotiate_version("1"))
        assert job.contract_version == CONTRACT_VERSION


# ---------------------------------------------------------------------------
# REQ-FLEET-011: no secrets in a contract that is stored as evidence
# ---------------------------------------------------------------------------


class TestNoSecretsInTheContract:
    def test_a_clean_payload_passes(self) -> None:
        assert_no_secrets(json.loads(_job().model_dump_json()))

    def test_a_credential_shaped_field_name_is_refused(self) -> None:
        for key in ("password", "api_key", "token", "private_key", "authorization"):
            with pytest.raises(ContractError, match="credential field"):
                assert_no_secrets({"envelope": {"max_ram_gb": 1.0, key: "x"}})

    def test_a_credential_shaped_value_under_an_innocuous_name_is_refused(self) -> None:
        """Field names are not the only way a secret travels.

        The value is matched by shape, and the error never echoes it.
        """
        with pytest.raises(ContractError, match="credential-shaped"):
            assert_no_secrets({"note": "ghp_" + "a" * 36})

    def test_a_nested_secret_is_refused(self) -> None:
        with pytest.raises(ContractError):
            assert_no_secrets({"tool": {"args": {"auth": "Bearer x"}}})

    def test_the_error_never_echoes_the_value(self) -> None:
        secret = "ghp_" + "b" * 36
        with pytest.raises(ContractError) as excinfo:
            assert_no_secrets({"note": secret})
        assert secret not in str(excinfo.value)


# ---------------------------------------------------------------------------
# AC-FLEET-011: the common package imports without a GPU backend
# ---------------------------------------------------------------------------


class TestPortableImport:
    """REQ-FLEET-013: importing the contract must not need an unavailable
    backend.

    Checked by hard-blocking every GPU package at `__import__` and then
    actually importing and using the contract. A source grep cannot answer
    this -- `mlx_hot_runtime.py` still names `mlx_lm` at module scope, it is
    simply no longer on the import path -- so the property is proved by
    execution, on a machine where the imports would fail if they were eager.
    """

    _BLOCKED = ("mlx", "mlx_lm", "torch", "vllm", "onnxruntime", "psutil", "cuda")

    def test_the_contract_imports_and_works_with_no_gpu_backend(self) -> None:
        import builtins
        import importlib
        import sys

        real_import = builtins.__import__

        def guard(name, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            if name.split(".")[0] in self._BLOCKED:
                raise ImportError(f"blocked for this test: {name}")
            return real_import(name, *args, **kwargs)

        for name in list(sys.modules):
            if name == "oai2.fleet" or name.startswith("oai2.fleet."):
                del sys.modules[name]
        builtins.__import__ = guard
        try:
            fleet = importlib.import_module("oai2.fleet")
            job = fleet.JobContract(
                task_id=TaskId("t1"),
                workload=WorkloadClass.NORMAL,
                envelope=fleet.ResourceEnvelope(max_ram_gb=8.0),
                backend=fleet.BackendVersion(
                    provider="p", model="m", backend="b", artifact_version="v"
                ),
            )
        finally:
            builtins.__import__ = real_import
            for name in list(sys.modules):
                if name == "oai2.fleet" or name.startswith("oai2.fleet."):
                    del sys.modules[name]
            importlib.import_module("oai2.fleet")  # restore for other tests

        assert job.workload is WorkloadClass.NORMAL
        assert fleet.CONTRACT_VERSION == CONTRACT_VERSION

    def test_no_module_scope_backend_import_in_the_contract(self) -> None:
        """The same property, read off the source.

        Cheap, and it localises a regression to a line rather than to an
        ImportError somewhere downstream.
        """
        import ast
        from pathlib import Path

        source = Path(__file__).parent.parent / "oai2" / "fleet" / "contract.py"
        tree = ast.parse(source.read_text())
        top_level_lines = {
            sub.lineno
            for node in tree.body
            for sub in ast.walk(node)
            if isinstance(sub, (ast.Import, ast.ImportFrom))
        }
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)) and node.lineno in top_level_lines:
                names = [a.name for a in node.names] if isinstance(node, ast.ImportFrom) else []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                for name in names:
                    assert name.split(".")[0] not in self._BLOCKED, (
                        f"{name!r} is imported at module scope; it must be lazy or absent"
                    )
