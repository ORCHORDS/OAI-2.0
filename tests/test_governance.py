"""Coverage for eval revisioning and audit-before-promotion (WP-56 / WI-BENCH-002).

The central claim under test is that a changed eval or scorer produces a
*different result identity* and is therefore not silently comparable. That is
easy to assert and easy to get wrong, so the controls here are the point:

- ``test_unchanged_suite_yields_a_stable_revision`` pins determinism, and
  ``test_changing_one_prompt_changes_the_revision`` pins sensitivity. Without
  both, a revision id could be constant (nothing is ever comparable) or
  unstable (nothing ever is).
- ``test_absent_audit_blocks_promotion`` is the absence-as-clean control: a
  record with no audit must not pass, which is the whole reason ``AuditStatus``
  has an ``UNKNOWN`` member.
- ``test_governed_gate_refuses_the_comparison_the_raw_gate_accepted`` pins the
  exact gap that motivated this module: the pre-existing gate said ``passed``
  for two reports whose prompts were entirely different.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from oai2.evals import CapabilityCase, CapabilitySuite
from oai2.evals.contamination import ContaminationPolicy, CorpusEntry
from oai2.evals.governance import (
    AuditStatus,
    ContaminationAudit,
    EvalRevision,
    ResultRecord,
    audit_contamination,
    build_trend,
    compare_records,
    eval_revision,
    evaluate_governed_promotion,
)
from oai2.evals.regression import CapabilityRegressionThreshold, HeldOutPromotionBudget


def _suite(prompts: dict[str, str] | None = None, *, suite_id: str = "s") -> CapabilitySuite:
    prompts = prompts or {
        "a": "How many days are there in one week? Answer with a number.",
        "b": "Reverse the string 'orchords'. Output only the reversed string.",
    }
    return CapabilitySuite(
        suite_id=suite_id,
        capability="deterministic_hard",
        description="test",
        cases=[
            CapabilityCase(
                case_id=case_id,
                capability="deterministic_hard",
                prompt=prompt,
                expected_patterns=(r"\b7\b",),
            )
            for case_id, prompt in prompts.items()
        ],
    )


def _budget() -> HeldOutPromotionBudget:
    return HeldOutPromotionBudget(
        version="b1",
        thresholds=(
            CapabilityRegressionThreshold(
                capability="deterministic_hard",
                max_pass_rate_regression=1.0,
                max_mean_score_regression=1.0,
            ),
        ),
        min_abstention_accuracy=0.0,
        max_false_success_rate=1.0,
    )


def _clean_audit(revision: EvalRevision) -> ContaminationAudit:
    return ContaminationAudit(
        status=AuditStatus.CLEAN, audited_at_revision=revision.revision_id, finding_count=0
    )


# -- revisions --------------------------------------------------------------


def test_unchanged_suite_yields_a_stable_revision() -> None:
    a = eval_revision(_suite(), scorer_version="regex_or@1")
    b = eval_revision(_suite(), scorer_version="regex_or@1")
    assert a == b
    assert a.revision_id and a.case_count == 2


def test_changing_one_prompt_changes_the_revision() -> None:
    original = _suite()
    edited = _suite({"a": "How many days are there in one week? Answer with a number. ",
                     "b": "Reverse the string 'orchords'. Output only the reversed string."})
    assert eval_revision(original, scorer_version="1").revision_id != (
        eval_revision(edited, scorer_version="1").revision_id
    )


def test_case_order_does_not_change_the_revision() -> None:
    forward = _suite({"a": "alpha prompt here", "b": "beta prompt here"})
    backward = _suite({"b": "beta prompt here", "a": "alpha prompt here"})
    assert (
        eval_revision(forward, scorer_version="1").revision_id
        == eval_revision(backward, scorer_version="1").revision_id
    )


def test_scorer_change_changes_the_revision_even_with_identical_prompts() -> None:
    """REQ-BENCH-025: different scorer, different numbers, different identity."""
    suite = _suite()
    a = eval_revision(suite, scorer_version="regex_or@1")
    b = eval_revision(suite, scorer_version="regex_or@2")
    # The scorer version is *in* the digest: a different scorer means the two
    # sets of numbers were not produced by the same measurement.
    assert a.revision_id != b.revision_id
    check = compare_records(
        ResultRecord("x", a, "cand", _clean_audit(a), 1.0, 1.0),
        ResultRecord("y", b, "cand", _clean_audit(b), 1.0, 1.0),
    )
    assert "eval_revision_mismatch" in check.reasons
    assert "scorer_version_mismatch" in check.reasons
    assert not check.comparable


def test_scorer_name_is_part_of_the_revision() -> None:
    suite = _suite()
    a = eval_revision(suite, scorer="regex_or", scorer_version="1")
    b = eval_revision(suite, scorer="model_judge", scorer_version="1")
    assert a.revision_id != b.revision_id


def test_revision_requires_scorer_version() -> None:
    with pytest.raises(ValueError):
        eval_revision(_suite(), scorer_version="")
    with pytest.raises(ValueError):
        eval_revision(object(), scorer_version="1")
    with pytest.raises(ValueError):
        eval_revision(_suite(suite_id=""), scorer_version="1")


def test_revision_rejects_a_case_without_a_prompt() -> None:
    suite = CapabilitySuite(
        suite_id="s", capability="c", description="d",
        cases=[CapabilityCase(case_id="a", capability="c", prompt="ok", expected_patterns=())],
    )
    object.__setattr__(suite.cases[0], "prompt", None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="prompt"):
        eval_revision(suite, scorer_version="1")


def test_revision_rejects_invalid_fields() -> None:
    with pytest.raises(ValueError):
        EvalRevision("", "s", "c", "regex_or", "1", 1)
    with pytest.raises(ValueError):
        EvalRevision("r", "s", "c", "regex_or", "  ", 1)
    with pytest.raises(ValueError):
        EvalRevision("r", "s", "c", "regex_or", "1", -1)
    with pytest.raises(ValueError):
        EvalRevision("r", "s", "c", "regex_or", "1", True)


# -- audit status -----------------------------------------------------------


def test_absent_audit_is_unknown_not_clean() -> None:
    unknown = ContaminationAudit.unknown()
    assert unknown.status is AuditStatus.UNKNOWN
    assert unknown.finding_count is None, "an un-audited record has no finding count"
    assert unknown.blocks_promotion


def test_audit_status_is_first_class() -> None:
    assert {s.value for s in AuditStatus} == {"clean", "findings", "unknown"}


def test_audit_contamination_reports_findings() -> None:
    suite = _suite()
    burned = suite.cases[0].prompt
    corpus = [CorpusEntry(entry_id="leak", text=burned, provenance="lesson")]
    revision = eval_revision(suite, scorer_version="1")
    audit = audit_contamination(suite, corpus, revision=revision)
    assert audit.status is AuditStatus.FINDINGS
    assert audit.finding_count == 1
    assert audit.quarantined_case_ids == ("a",)
    assert audit.audited_at_revision == revision.revision_id
    assert audit.blocks_promotion


def test_audit_contamination_clean_on_unrelated_corpus() -> None:
    suite = _suite()
    audit = audit_contamination(
        suite, [CorpusEntry(entry_id="x", text="refactor the connection pool")],
        revision=eval_revision(suite, scorer_version="1"),
    )
    assert audit.status is AuditStatus.CLEAN
    assert audit.finding_count == 0
    assert not audit.blocks_promotion


def test_auditing_an_empty_suite_is_unknown_not_clean() -> None:
    empty = CapabilitySuite(suite_id="s", capability="c", description="d", cases=[])
    audit = audit_contamination(empty, [CorpusEntry(entry_id="x", text="anything at all")])
    assert audit.status is AuditStatus.UNKNOWN
    assert audit.blocks_promotion


# -- comparability ----------------------------------------------------------


def test_identical_records_are_comparable() -> None:
    revision = eval_revision(_suite(), scorer_version="1")
    a = ResultRecord("baseline", revision, "cand-1", _clean_audit(revision), 0.5, 0.5)
    b = ResultRecord("candidate", revision, "cand-1", _clean_audit(revision), 0.6, 0.6)
    check = compare_records(a, b)
    assert check.comparable, check.reasons
    assert check.reasons == ()


def test_all_mismatches_are_reported_not_just_the_first() -> None:
    rev_a = eval_revision(_suite(), scorer_version="1")
    rev_b = eval_revision(_suite({"a": "totally different prompt", "b": "other text here"}),
                          scorer_version="2")
    a = ResultRecord("x", rev_a, "cand-1", ContaminationAudit.unknown(), 0.5, 0.5)
    b = ResultRecord("y", rev_b, "cand-2", ContaminationAudit.unknown(), 0.6, 0.6)
    check = compare_records(a, b)
    assert set(check.reasons) == {
        "eval_revision_mismatch",
        "scorer_version_mismatch",
        "candidate_mismatch",
        "audit_not_clean",
    }


def test_record_rejects_invalid_numbers() -> None:
    revision = eval_revision(_suite(), scorer_version="1")
    with pytest.raises(ValueError):
        ResultRecord("x", revision, "c", _clean_audit(revision), 1.5, 0.5)
    with pytest.raises(ValueError):
        ResultRecord("", revision, "c", _clean_audit(revision), 0.5, 0.5)
    with pytest.raises(ValueError):
        ResultRecord("x", revision, "  ", _clean_audit(revision), 0.5, 0.5)


# -- trends -----------------------------------------------------------------


def test_identical_records_form_one_trend() -> None:
    revision = eval_revision(_suite(), scorer_version="1")
    records = [
        ResultRecord(f"run{i}", revision, "cand-1", _clean_audit(revision), 0.5 + i / 100, 0.5)
        for i in range(3)
    ]
    report = build_trend(records)
    assert report.is_single_trend
    assert not report.mixed
    assert report.reasons() == ()
    assert len(report.groups[0].labels) == 3


def test_different_revisions_do_not_merge_into_one_trend() -> None:
    """REQ-BENCH-023: the difference is marked, not averaged away."""
    rev_a = eval_revision(_suite(), scorer_version="1")
    rev_b = eval_revision(
        _suite({"a": "a different prompt entirely", "b": "yet another prompt here"}),
        scorer_version="1",
    )
    report = build_trend(
        [
            ResultRecord("old", rev_a, "cand-1", _clean_audit(rev_a), 0.1, 0.1),
            ResultRecord("new", rev_b, "cand-1", _clean_audit(rev_b), 0.9, 0.9),
        ]
    )
    assert report.mixed
    assert len(report.groups) == 2
    assert "2 non-comparable groups" in report.reasons()[0]
    # A single blended number is exactly what must not exist here.
    assert not hasattr(report, "mean_pass_rate")


def test_different_candidates_do_not_merge() -> None:
    revision = eval_revision(_suite(), scorer_version="1")
    report = build_trend(
        [
            ResultRecord("a", revision, "cand-1", _clean_audit(revision), 0.5, 0.5),
            ResultRecord("b", revision, "cand-2", _clean_audit(revision), 0.5, 0.5),
        ]
    )
    assert report.mixed


# -- governed promotion gate ------------------------------------------------


def test_governed_gate_refuses_the_comparison_the_raw_gate_accepted() -> None:
    """The gap that motivated this module, pinned as a regression.

    Two reports, same ``case_id``, completely different prompt behind it. The
    pre-existing gate reports ``passed=True``; the governed gate must refuse
    before any number is compared.
    """
    from oai2.evals import CapabilityScore, SuiteReport
    from oai2.evals.regression import evaluate_held_out_promotion

    def report(case_id: str, passed: bool) -> SuiteReport:
        r = SuiteReport(
            suite_id="deterministic_hard", capability="deterministic_hard",
            runtime="fake", n_cases=1, n_passed=int(passed),
        )
        r.scores = [
            CapabilityScore(
                case_id=case_id, capability="deterministic_hard", score=1.0,
                passed=passed, matched_pattern=None, forbidden_matched=(),
            )
        ]
        return r

    raw = evaluate_held_out_promotion(
        baseline_reports=[report("days_in_a_week", True)],
        candidate_reports=[report("days_in_a_week", True)],
        budget=_budget(), training_case_ids=set(),
        candidate_abstention_accuracy=1.0, candidate_false_success_rate=0.0,
    )
    assert raw.passed, "the raw gate is expected to accept this; that is the defect"

    rev_a = eval_revision(_suite(), scorer_version="1")
    rev_b = eval_revision(_suite({"a": "an entirely different prompt", "b": "and another"}),
                          scorer_version="1")
    governed = evaluate_governed_promotion(
        baseline_records=[ResultRecord("b", rev_a, "c", _clean_audit(rev_a), 1.0, 1.0)],
        candidate_records=[ResultRecord("c", rev_b, "c", _clean_audit(rev_b), 1.0, 1.0)],
        budget=_budget(), training_case_ids=set(),
        candidate_abstention_accuracy=1.0, candidate_false_success_rate=0.0,
    )
    assert not governed.passed
    assert "eval_revision_mismatch" in governed.failures
    assert governed.decided_by_governance
    assert governed.inner is None, "the numeric gate must not have run at all"


def test_unknown_audit_blocks_promotion() -> None:
    revision = eval_revision(_suite(), scorer_version="1")
    out = evaluate_governed_promotion(
        baseline_records=[ResultRecord("b", revision, "c", ContaminationAudit.unknown(), 1.0, 1.0)],
        candidate_records=[
            ResultRecord("c", revision, "c", ContaminationAudit.unknown(), 1.0, 1.0)
        ],
        budget=_budget(), training_case_ids=set(),
        candidate_abstention_accuracy=1.0, candidate_false_success_rate=0.0,
    )
    assert not out.passed
    assert "audit_not_clean" in out.failures
    assert out.decided_by_governance


def test_findings_audit_blocks_promotion() -> None:
    revision = eval_revision(_suite(), scorer_version="1")
    dirty = ContaminationAudit(
        status=AuditStatus.FINDINGS, audited_at_revision=revision.revision_id, finding_count=1
    )
    out = evaluate_governed_promotion(
        baseline_records=[ResultRecord("b", revision, "c", _clean_audit(revision), 1.0, 1.0)],
        candidate_records=[ResultRecord("c", revision, "c", dirty, 1.0, 1.0)],
        budget=_budget(), training_case_ids=set(),
        candidate_abstention_accuracy=1.0, candidate_false_success_rate=0.0,
    )
    assert not out.passed
    assert "audit_not_clean" in out.failures


def test_clean_and_matching_records_reach_the_numeric_gate() -> None:
    revision = eval_revision(_suite(), scorer_version="1")
    out = evaluate_governed_promotion(
        baseline_records=[ResultRecord("b", revision, "c", _clean_audit(revision), 1.0, 1.0)],
        candidate_records=[ResultRecord("c", revision, "c", _clean_audit(revision), 1.0, 1.0)],
        budget=_budget(), training_case_ids=set(),
        candidate_abstention_accuracy=1.0, candidate_false_success_rate=0.0,
    )
    assert not out.decided_by_governance
    assert out.inner is not None
    assert out.passed == out.inner.passed


def test_missing_records_are_refused() -> None:
    out = evaluate_governed_promotion(
        baseline_records=[], candidate_records=[],
        budget=_budget(), training_case_ids=set(),
        candidate_abstention_accuracy=1.0, candidate_false_success_rate=0.0,
    )
    assert not out.passed
    assert out.failures == ("missing_records",)


def test_inconsistent_revision_within_one_side_is_refused() -> None:
    rev_a = eval_revision(_suite(), scorer_version="1")
    rev_b = eval_revision(_suite({"a": "another prompt here", "b": "and one more"}),
                          scorer_version="1")
    out = evaluate_governed_promotion(
        baseline_records=[
            ResultRecord("b1", rev_a, "c", _clean_audit(rev_a), 1.0, 1.0),
            ResultRecord("b2", rev_b, "c", _clean_audit(rev_b), 1.0, 1.0),
        ],
        candidate_records=[ResultRecord("c", rev_a, "c", _clean_audit(rev_a), 1.0, 1.0)],
        budget=_budget(), training_case_ids=set(),
        candidate_abstention_accuracy=1.0, candidate_false_success_rate=0.0,
    )
    assert not out.passed
    assert "inconsistent_revision_within_side" in out.failures


# -- public safety ----------------------------------------------------------


def test_audit_and_record_serialize_without_prompt_text() -> None:
    suite = _suite()
    revision = eval_revision(suite, scorer_version="1")
    audit = audit_contamination(suite, [CorpusEntry(entry_id="e", text="unrelated")],
                                revision=revision)
    record = ResultRecord("run", revision, "cand-1", audit, 0.5, 0.5)
    rendered = json.dumps({"revision": asdict(revision), "audit": asdict(audit),
                           "record": asdict(record)}, default=str)
    assert "days are there" not in rendered
    assert "orchords" not in rendered


def test_negative_control_clean_suite_produces_a_clean_audit() -> None:
    """Proves the audit's clean verdict depends on it actually running."""
    suite = _suite()
    revision = eval_revision(suite, scorer_version="1")
    burned = [CorpusEntry(entry_id="e", text=suite.cases[0].prompt)]
    assert audit_contamination(suite, burned, revision=revision).status is AuditStatus.FINDINGS
    assert (
        audit_contamination(
            suite, [CorpusEntry(entry_id="e", text="completely unrelated text")], revision=revision
        ).status
        is AuditStatus.CLEAN
    )


def test_policy_is_carried_into_the_audit_record() -> None:
    suite = _suite()
    policy = ContaminationPolicy(near_overlap_threshold=0.9)
    audit = audit_contamination(suite, [CorpusEntry(entry_id="e", text="unrelated")],
                                policy=policy)
    assert audit.policy == policy
