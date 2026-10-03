"""Repository-wide: an absence must be distinguishable from a measurement.

This module exists because the same failure mode was found in six places
across four work packages, which is enough evidence to stop treating it as a
series of coincidences. The owner's reconciliation on Master #1 records the
pattern and asks for it to be handled as a review item:

> A self-declaration, an absence, and a leak all read exactly like a clean
> result -- unless something independent is required to disagree.

Each test below pins ONE already-closed instance, in the shape the defect
took: the "absent" case and the "measured and clean" case are constructed
side by side and asserted to be **distinguishable**. That is the property. A
guard that only checks the happy path would pass against every one of the
original defects, which is exactly what the per-issue tests did before.

The instances, by commit:

    3defdc0  numerical_compare      no_comparable_samples
    ce90245  numerical_compare      capability_measured
    1af62fb  evidence_package       RetrievalMetrics.measured / insufficient_evidence
    13ed05c  evals/__init__        HarnessReport.has_errors
    747e4f1  knowledge/gc_vector    strict inventory_complete
    f309d81  agents/learning        _dedup_check must not swallow
    29430da  evals/qos              evaluate_promotion correctness gate
    b5b7e83  agents/learning        extract_lesson finished_reason
    4b900c9  evals/qos              verified_actions_per_second
    8e0d2a1  tools/registry         _grep unreadable_count

Adding a new instance means adding a test here, so the next occurrence of
this class has somewhere to be recorded rather than becoming a new
coincidence.
"""

from __future__ import annotations

import dataclasses

import pytest

from oai2.agents.agent_loop import AgentRun
from oai2.agents.learning import _dedup_check, extract_lesson
from oai2.core import Status
from oai2.evals import HarnessReport, SuiteReport
from oai2.evals.qos import (
    BudgetKind,
    WorkloadBudget,
    WorkloadClass,
    WorkloadSample,
    evaluate_budget,
    evaluate_promotion,
    summarize_samples,
)
from oai2.knowledge import gc_vector
from oai2.knowledge.evidence_package import RetrievalMetrics
from oai2.model.numerical_compare import (
    NumericalArtifactIdentity,
    NumericalToleranceProfile,
    compare_numerical_paths,
)
from oai2.model.numerics import NumericalOperation
from oai2.model.tolerance_matrix import canonical_policy
from oai2.tools import registry


def _identity() -> NumericalArtifactIdentity:
    return NumericalArtifactIdentity(
        model_version="model-v1",
        runtime_version="runtime-v1",
        backend="optimized-backend",
        config_id="cfg-v1",
    )


def _profile(
    *,
    operation: NumericalOperation = NumericalOperation.LOSS,
    dtype: str = "fp32",
) -> NumericalToleranceProfile:
    """A profile that COMPLIES with the canonical tolerance policy.

    Since `e66630e`, a caller-declared tolerance is checked against
    `tolerance_matrix.canonical_policy` and a non-compliant profile fails
    with ``tolerance_policy``. Building the values by hand here would make
    the "clean" baseline fail for that reason and mask what these tests are
    actually about, so the ceilings come from the policy itself.
    """
    policy = canonical_policy(operation, dtype)
    return NumericalToleranceProfile(
        profile_id=policy.policy_id,
        operation=policy.operation,
        dtype=dtype,
        shape_class="scalar",
        max_abs_error=policy.max_abs_error,
        max_rel_error=policy.max_rel_error,
        max_capability_regression=policy.max_capability_regression,
        require_finite_state_match=policy.require_finite_state_match,
    )


# --------------------------------------------------------------------------
# 3defdc0 -- a comparison that never happened reported max_abs_error=0.0
# --------------------------------------------------------------------------


def test_a_comparison_with_no_comparable_samples_is_not_a_clean_one() -> None:
    profile = _profile()
    empty = compare_numerical_paths(
        [float("nan")],
        [float("nan")],
        profile=profile,
        identity=_identity(),
    )
    clean = compare_numerical_paths(
        [1.0, 2.0], [1.0, 2.0], profile=profile, identity=_identity()
    )

    # The defect: both reported max_abs_error 0.0 and passed=True.
    assert empty.max_abs_error == clean.max_abs_error == 0.0
    # The fix: the absent case is distinguishable.
    assert "no_comparable_samples" in empty.failures
    assert clean.failures == ()
    assert empty.passed is False
    assert clean.passed is True


# --------------------------------------------------------------------------
# ce90245 -- "capability not measured" was byte-identical to "measured, passed"
# --------------------------------------------------------------------------


def test_unmeasured_capability_is_distinguishable_from_a_measured_pass() -> None:
    profile = _profile()
    unmeasured = compare_numerical_paths(
        [1.0, 2.0], [1.0, 2.0], profile=profile, identity=_identity()
    )
    measured = compare_numerical_paths(
        [1.0, 2.0],
        [1.0, 2.0],
        profile=profile,
        identity=_identity(),
        reference_capability_score=0.9,
        optimized_capability_score=0.9,
    )

    # The defect: both read regression=0.0, passed=True, failures=().
    assert unmeasured.capability_regression == measured.capability_regression == 0.0
    assert unmeasured.failures == measured.failures == ()
    # The fix: the flag distinguishes them.
    assert unmeasured.capability_measured is False
    assert measured.capability_measured is True


# --------------------------------------------------------------------------
# 1af62fb -- a retrieval that retrieved nothing reported clean zeros
# --------------------------------------------------------------------------


def test_an_evidence_free_retrieval_is_distinguishable_from_a_clean_one() -> None:
    empty = RetrievalMetrics(
        k=0,
        precision_at_k=0.0,
        recall=0.0,
        irrelevant_context_rate=0.0,
        package_tokens=0,
        raw_source_tokens=0,
        compression_ratio=0.0,
        task_success_delta=0.0,
        insufficient_evidence=True,
    )
    real = RetrievalMetrics(
        k=2,
        precision_at_k=1.0,
        recall=1.0,
        irrelevant_context_rate=0.0,
        package_tokens=10,
        raw_source_tokens=20,
        compression_ratio=0.5,
        task_success_delta=0.1,
        insufficient_evidence=False,
    )

    # The defect: both reported irrelevant_context_rate == 0.0, which reads as
    # "this context introduced no irrelevant material" -- the most flattering
    # number a retrieval metric can carry.
    assert empty.irrelevant_context_rate == real.irrelevant_context_rate == 0.0
    # The fix.
    assert empty.measured is False
    assert real.measured is True
    assert empty.insufficient_evidence is True
    assert real.insufficient_evidence is False

    # And the lie is unconstructible, not merely unreported.
    with pytest.raises(ValueError, match="insufficient_evidence"):
        RetrievalMetrics(
            k=3,
            precision_at_k=1.0,
            recall=1.0,
            irrelevant_context_rate=0.0,
            package_tokens=10,
            raw_source_tokens=20,
            compression_ratio=0.5,
            task_success_delta=0.1,
            insufficient_evidence=True,
        )

    # The flag is serialized: an artifact is read by someone who is not the
    # writer, and a property alone would not appear in asdict.
    assert "insufficient_evidence" in dataclasses.asdict(empty)


# --------------------------------------------------------------------------
# 13ed05c -- a suite that raised vanished from pass_rate
# --------------------------------------------------------------------------


def _suite(suite_id: str, n_cases: int, n_passed: int, error: str | None = None):
    return SuiteReport(
        suite_id=suite_id,
        capability="demo",
        runtime="PlaceholderRuntime",
        n_cases=n_cases,
        n_passed=n_passed,
        error=error,
    )


def test_a_crashed_suite_is_distinguishable_from_a_fully_passing_one() -> None:
    clean = HarnessReport(
        runtime="PlaceholderRuntime", reports=(_suite("a", 12, 12),)
    )
    crashed = HarnessReport(
        runtime="PlaceholderRuntime",
        reports=(_suite("a", 12, 12), _suite("b", 0, 0, "ValueError: boom")),
    )

    # The defect: the headline numbers were identical, so a reader of
    # pass_rate could not tell "everything passed" from "a suite never ran".
    assert (clean.n_cases, clean.n_passed, clean.pass_rate) == (
        crashed.n_cases,
        crashed.n_passed,
        crashed.pass_rate,
    )
    # The fix: the aggregate now carries the signal.
    assert clean.has_errors is False
    assert crashed.has_errors is True
    assert crashed.errored_suite_ids == ("b",)


# --------------------------------------------------------------------------
# 747e4f1 -- the string "false" was coerced to a completed inventory
# --------------------------------------------------------------------------


def test_an_incomplete_inventory_is_distinguishable_from_a_complete_one() -> None:
    state = gc_vector.VectorGcReconciliationState.from_rows([], observed_at=1000.0)

    complete = state.to_snapshot()
    complete["pages_processed"] = 0
    complete["next_cursor"] = None
    complete["inventory_complete"] = True

    incomplete = dict(complete)
    incomplete["inventory_complete"] = False

    # The defect: `bool("false")` is True, so a snapshot declaring the scan
    # INCOMPLETE passed the `inventory scan is incomplete` guard in
    # build_report/recheck_grace -- the direction that authorises deletion.
    with pytest.raises(ValueError, match="inventory_complete must be a boolean"):
        lying = dict(complete)
        lying["inventory_complete"] = "false"
        gc_vector.VectorGcReconciliationState.from_snapshot(lying)

    # The fix: the two states are representable and the lie is not.
    assert (
        gc_vector.VectorGcReconciliationState.from_snapshot(
            complete
        ).inventory_complete
        is True
    )
    assert (
        gc_vector.VectorGcReconciliationState.from_snapshot(
            incomplete
        ).inventory_complete
        is False
    )


# --------------------------------------------------------------------------
# f309d81 -- a failed dedup check reported "no conflicts"
# --------------------------------------------------------------------------


class _UnreachableStore:
    def retrieve(self, request):  # noqa: ANN001, ANN201
        raise RuntimeError("D1 unreachable")


class _EmptyStore:
    def retrieve(self, request):  # noqa: ANN001, ANN201
        return type("R", (), {"objects": ()})()


def test_a_failed_dedup_check_is_distinguishable_from_no_conflicts() -> None:
    # The defect: both returned (), so a D1 outage looked like a clean bill
    # of health and the caller wrote a duplicate lesson.
    with pytest.raises(RuntimeError, match="D1 unreachable"):
        _dedup_check(_UnreachableStore(), "topic")
    # The fix, and the normal no-conflict case is preserved.
    assert _dedup_check(_EmptyStore(), "topic") == ()


# --------------------------------------------------------------------------
# 29430da -- the promotion gate could not see a false success
# --------------------------------------------------------------------------


def _sample(*, total: float, declared: bool, verified: bool) -> WorkloadSample:
    return WorkloadSample(
        workload=WorkloadClass.NORMAL,
        target_hardware="mac-studio-m5",
        config_id="normal-v1",
        ttft_ms=10.0,
        first_useful_action_ms=50.0,
        end_to_end_ms=total,
        declared_success=declared,
        verified_success=verified,
        decode_tokens_per_second=100.0,
    )


def test_a_false_success_candidate_is_not_promotable() -> None:
    good = summarize_samples(
        [_sample(total=100.0, declared=True, verified=True) for _ in range(2)]
    )
    awful = summarize_samples(
        [_sample(total=100.0, declared=True, verified=False) for _ in range(2)]
    )

    # The defect: identical latency, so the latency-only gate passed a
    # candidate that was never verified-correct.
    assert good.end_to_end_ms.p95 == awful.end_to_end_ms.p95
    promo = evaluate_promotion(awful, good)
    assert promo.passed is False
    assert "false_success_regression" in promo.failures

    # The fix: the two gates now agree about the same report.
    budget = WorkloadBudget(
        version="v1",
        workload=WorkloadClass.NORMAL,
        kind=BudgetKind.SERVICE_BUDGET,
        target_hardware="mac-studio-m5",
        config_id="normal-v1",
        first_useful_action_p95_ms=1000.0,
        end_to_end_p95_ms=1000.0,
        end_to_end_p99_ms=2000.0,
        max_false_success_rate=0.02,
        min_verified_success_rate=0.9,
    )
    assert evaluate_budget(awful, budget).passed is False
    # And a genuinely clean candidate is still promotable, so the gate is a
    # check and not a blanket refusal.
    assert evaluate_promotion(good, good).passed is True


# --------------------------------------------------------------------------
# b5b7e83 -- an unfinished run was promoted as a verified lesson
# --------------------------------------------------------------------------


def _run(*, final_text: str, finished_reason: str | None) -> AgentRun:
    return AgentRun(
        user_prompt="refactor the module",
        final_text=final_text,
        steps=[],
        total_tool_calls=0,
        total_input_tokens=0,
        total_output_tokens=0,
        finished_reason=finished_reason,
    )


def test_an_unfinished_run_is_distinguishable_from_a_verified_lesson() -> None:
    def lesson(run: AgentRun) -> Status:
        return extract_lesson(
            run,
            task_id="t-absence",
            verification_ref="verifier://sess",
            source_version="v1",
            runtime_version="oai2/0.1",
        ).status

    # The defect: "length" (truncated), "tool_calls" (loop exhausted
    # max_steps) and None (never recorded) all produced IMPLEMENTED, with
    # content asserting "verified lesson".
    for reason in ("length", "tool_calls", None):
        assert lesson(_run(final_text="partial answer", finished_reason=reason)) is (
            Status.EXPERIMENTAL
        ), f"finished_reason={reason!r} is not a completed run"

    # The fix, and a genuinely completed run is still promoted.
    assert lesson(_run(final_text="done", finished_reason="stop")) is (
        Status.IMPLEMENTED
    )


# --------------------------------------------------------------------------
# follow-on to b5b7e83 -- the status was corrected, the body still self-declared
# --------------------------------------------------------------------------


def test_a_corrected_status_is_not_undone_by_its_own_body() -> None:
    """b5b7e83 fixed `status`. It left the headline string alone.

    `extract_lesson` built its body with an unconditional
    `f"verified lesson from task_id={task_id}"` while `status` was
    computed separately and could be EXPERIMENTAL. So the record carried
    an EXPERIMENTAL status and a body asserting verification -- and the
    body is the part that propagates, because `content_hash` is
    `sha256(content)` and `R2BodyDescriptor.from_knowledge` enforces that
    equality before the body is stored.

    This is the cross-cutting form of the b5b7e83 defect: fixing one of
    the two disagreeing fields and not the other leaves the false
    assertion intact for every consumer that reads the body.
    """

    def body(*, final_text: str, finished_reason: str | None) -> str:
        return extract_lesson(
            _run(final_text=final_text, finished_reason=finished_reason),
            task_id="t-absence-body",
            verification_ref="verifier://sess",
            source_version="v1",
            runtime_version="oai2/0.1",
        ).content

    for reason in ("length", "tool_calls", None, "error", "cancelled"):
        content = body(final_text="partial answer", finished_reason=reason)
        assert "verified" not in content, (
            f"finished_reason={reason!r} produced a body still readable as "
            f"verified: {content.splitlines()[0]!r}"
        )

    # The honest fact is still present -- the claim was withdrawn, not the
    # evidence for withdrawing it.
    assert "finished_reason='length'" in body(
        final_text="partial answer", finished_reason="length"
    )

    # Opposite direction: a completed run is still labelled, byte-identically,
    # so no already-stored content_hash or R2 blob key changes.
    assert body(final_text="done", finished_reason="stop").startswith(
        "verified lesson from task_id=t-absence-body\n"
    )


# --------------------------------------------------------------------------
# The invariant itself
# --------------------------------------------------------------------------


def test_the_repository_actually_contains_these_instances() -> None:
    """A guard against this module quietly testing nothing.

    Every test above reaches into a real module, so if one of those modules
    is deleted or renamed the collection would fail loudly -- but a rename
    that silently redirected an import to a stub would not. This asserts the
    symbols under test come from the expected modules.
    """
    import oai2.agents.learning as learning
    import oai2.evals as evals
    import oai2.evals.qos as qos
    import oai2.knowledge.evidence_package as ep
    import oai2.knowledge.gc_vector as gv
    import oai2.model.numerical_compare as nc

    assert learning._dedup_check.__module__ == "oai2.agents.learning"
    assert learning.extract_lesson.__module__ == "oai2.agents.learning"
    assert evals.HarnessReport.__module__ == "oai2.evals"
    assert qos.evaluate_promotion.__module__ == "oai2.evals.qos"
    assert qos.evaluate_budget.__module__ == "oai2.evals.qos"
    assert ep.RetrievalMetrics.__module__ == "oai2.knowledge.evidence_package"
    assert gv.VectorGcReconciliationState.__module__ == "oai2.knowledge.gc_vector"
    assert nc.compare_numerical_paths.__module__ == "oai2.model.numerical_compare"

    # And every one of them still exposes the signal that makes the absence
    # visible. If a field is renamed away, this fails rather than the
    # module quietly testing less.
    for name in ("insufficient_evidence", "measured"):
        assert hasattr(ep.RetrievalMetrics, name), name
    for name in ("has_errors", "errored_suite_ids"):
        assert hasattr(evals.HarnessReport, name), name
    assert hasattr(nc.NumericalComparison, "capability_measured")
    assert hasattr(qos.WorkloadReport, "throughput_measured")
    assert hasattr(registry.ExecutionOutcome, "coverage_complete")


# --------------------------------------------------------------------------
# 8e0d2a1 -- a grep that could not read a file reported a completed search
# --------------------------------------------------------------------------


def test_a_grep_that_skipped_files_is_distinguishable_from_a_clean_one(
    tmp_path,
) -> None:
    """A file that could not be read is a file that was not searched.

    This instance is recorded here rather than only in the tool registry's
    own tests because the sibling scanner cannot see it: the AST sweep in
    ``test_absent_measurement_invariant`` looks for an empty *collection*
    being substituted with ``0.0``, and this defect is a swallowed exception
    inside a loop. Different mechanism, same question -- could a reader tell,
    from this record alone, whether the search covered what it looked at?
    """
    import os

    from oai2.tools.registry import _grep

    (tmp_path / "clean.py").write_text("x = 1\n")
    locked = tmp_path / "locked.py"
    locked.write_text("def needle_function():\n    pass\n")
    os.chmod(locked, 0o000)
    try:
        partial = _grep(
            "needle_function", path_str=str(tmp_path), include_glob=None, cwd=tmp_path
        )
    finally:
        os.chmod(locked, 0o644)

    complete_root = tmp_path / "complete"
    complete_root.mkdir()
    (complete_root / "clean.py").write_text("x = 1\n")
    complete = _grep(
        "needle_function", path_str=str(complete_root), include_glob=None, cwd=complete_root
    )

    # The defect: both were ok, and both said "(no matches)". Only one of
    # them had actually looked.
    assert partial.ok is complete.ok is True
    assert partial.output != complete.output
    assert partial.coverage_complete is False
    assert partial.unreadable_count == 1
    assert complete.coverage_complete is True
    assert complete.unreadable_count == 0
    assert complete.output == "(no matches)"
