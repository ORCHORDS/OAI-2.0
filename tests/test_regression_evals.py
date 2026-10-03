from __future__ import annotations

from types import SimpleNamespace

import pytest

from oai2.evals import CapabilityScore, SuiteReport
from oai2.evals.contamination import (
    ContaminationReport,
    CorpusEntry,
    check_contamination,
    fingerprint_cases,
)
from oai2.evals.regression import (
    CapabilityRegressionThreshold,
    HeldOutPromotionBudget,
    evaluate_held_out_promotion,
)


def _score(case_id: str, capability: str, *, passed: bool) -> CapabilityScore:
    return CapabilityScore(
        case_id=case_id,
        capability=capability,
        score=1.0 if passed else 0.0,
        passed=passed,
        matched_pattern="ok" if passed else None,
        forbidden_matched=(),
        notes="" if passed else "failed",
    )


def _report(
    suite_id: str,
    capability: str,
    outcomes: tuple[tuple[str, bool], ...],
) -> SuiteReport:
    scores = [_score(case_id, capability, passed=passed) for case_id, passed in outcomes]
    return SuiteReport(
        suite_id=suite_id,
        capability=capability,
        runtime="held-out-runtime",
        n_cases=len(scores),
        n_passed=sum(score.passed for score in scores),
        scores=scores,
    )


def _budget() -> HeldOutPromotionBudget:
    return HeldOutPromotionBudget(
        version="held-out-v1",
        thresholds=(
            CapabilityRegressionThreshold(
                capability="coding",
                max_pass_rate_regression=0.10,
                max_mean_score_regression=0.10,
            ),
            CapabilityRegressionThreshold(
                capability="verification",
                max_pass_rate_regression=0.0,
                max_mean_score_regression=0.0,
            ),
        ),
        min_abstention_accuracy=0.90,
        max_false_success_rate=0.05,
    )



def _clean_contamination() -> ContaminationReport:
    """A GENUINE clean audit, not a hand-assembled report object.

    These tests are about regression arithmetic, not contamination, so they
    need a corpus that passes the audit. Running the real check means the
    tests exercise the same path a caller would instead of trusting a report
    nobody produced -- and it means the required argument is a real audit
    rather than a formality that satisfies the signature.
    """
    cases = [
        SimpleNamespace(
            case_id=case_id,
            prompt=f"held-out case {case_id}: verify the retry backoff behaves",
            expected_patterns=(),
            forbidden_patterns=(),
        )
        for case_id in _POISON_IDS
    ]
    corpus = [
        CorpusEntry(
            entry_id="unrelated-training-corpus",
            text="quarterly ledger reconciliation notes for the warehouse ledger",
        )
    ]
    report = check_contamination(fingerprint_cases(cases), corpus)
    assert report.clean, "the unrelated corpus must not overlap the held-out prompts"
    return report

def test_material_critical_regression_is_not_hidden_by_other_gain() -> None:
    baseline = [
        _report("coding", "coding", (("c1", True), ("c2", False))),
        _report("verification", "verification", (("v1", True), ("v2", True))),
    ]
    candidate = [
        _report("coding", "coding", (("c1", True), ("c2", True))),
        _report("verification", "verification", (("v1", True), ("v2", False))),
    ]

    result = evaluate_held_out_promotion(
        baseline_reports=baseline,
        candidate_reports=candidate,
        budget=_budget(),
        training_case_ids={"train-1", "train-2"},
        candidate_abstention_accuracy=1.0,
        candidate_false_success_rate=0.0,
        contamination=_clean_contamination(),
    )

    assert result.passed is False
    assert "verification:pass_rate_regression" in result.failures
    assert "verification:mean_score_regression" in result.failures
    coding = next(item for item in result.capability_regressions if item.capability == "coding")
    assert coding.candidate_pass_rate > coding.baseline_pass_rate


def test_non_regressing_control_candidate_passes() -> None:
    baseline = [
        _report("coding", "coding", (("c1", True), ("c2", False))),
        _report("verification", "verification", (("v1", True), ("v2", True))),
    ]
    candidate = [
        _report("coding", "coding", (("c1", True), ("c2", True))),
        _report("verification", "verification", (("v1", True), ("v2", True))),
    ]

    result = evaluate_held_out_promotion(
        baseline_reports=baseline,
        candidate_reports=candidate,
        budget=_budget(),
        training_case_ids={"train-1"},
        candidate_abstention_accuracy=0.95,
        candidate_false_success_rate=0.0,
        contamination=_clean_contamination(),
    )
    assert result.passed is True
    assert result.failures == ()


def test_train_held_out_overlap_is_rejected_before_promotion() -> None:
    baseline = [
        _report("coding", "coding", (("c1", True),)),
        _report("verification", "verification", (("v1", True),)),
    ]
    candidate = [
        _report("coding", "coding", (("c1", True),)),
        _report("verification", "verification", (("v1", True),)),
    ]

    with pytest.raises(ValueError, match="overlap training/tuning"):
        evaluate_held_out_promotion(
            baseline_reports=baseline,
            candidate_reports=candidate,
            budget=_budget(),
            training_case_ids={"c1"},
            candidate_abstention_accuracy=1.0,
            candidate_false_success_rate=0.0,
        contamination=_clean_contamination(),
        )


@pytest.mark.parametrize(
    ("abstention", "false_success", "expected_failure"),
    [
        (0.80, 0.0, "abstention_accuracy"),
        (1.0, 0.10, "false_success_rate"),
    ],
)
def test_abstention_and_false_success_are_first_class_gates(
    abstention: float,
    false_success: float,
    expected_failure: str,
) -> None:
    baseline = [
        _report("coding", "coding", (("c1", True),)),
        _report("verification", "verification", (("v1", True),)),
    ]
    candidate = [
        _report("coding", "coding", (("c1", True),)),
        _report("verification", "verification", (("v1", True),)),
    ]

    result = evaluate_held_out_promotion(
        baseline_reports=baseline,
        candidate_reports=candidate,
        budget=_budget(),
        training_case_ids=set(),
        candidate_abstention_accuracy=abstention,
        candidate_false_success_rate=false_success,
        contamination=_clean_contamination(),
    )
    assert result.passed is False
    assert expected_failure in result.failures


def test_candidate_and_baseline_must_use_identical_held_out_cases() -> None:
    baseline = [
        _report("coding", "coding", (("c1", True),)),
        _report("verification", "verification", (("v1", True),)),
    ]
    candidate = [
        _report("coding", "coding", (("c2", True),)),
        _report("verification", "verification", (("v1", True),)),
    ]
    with pytest.raises(ValueError, match="identical held-out case IDs"):
        evaluate_held_out_promotion(
            baseline_reports=baseline,
            candidate_reports=candidate,
            budget=_budget(),
            training_case_ids=set(),
            candidate_abstention_accuracy=1.0,
            candidate_false_success_rate=0.0,
        contamination=_clean_contamination(),
        )


def test_every_candidate_capability_requires_explicit_threshold() -> None:
    budget = HeldOutPromotionBudget(
        version="partial",
        thresholds=(
            CapabilityRegressionThreshold(
                capability="coding",
                max_pass_rate_regression=0.0,
                max_mean_score_regression=0.0,
            ),
        ),
        min_abstention_accuracy=0.0,
        max_false_success_rate=1.0,
    )
    reports = [
        _report("coding", "coding", (("c1", True),)),
        _report("verification", "verification", (("v1", True),)),
    ]
    with pytest.raises(ValueError, match="missing capability thresholds: verification"):
        evaluate_held_out_promotion(
            baseline_reports=reports,
            candidate_reports=reports,
            budget=budget,
            training_case_ids=set(),
            candidate_abstention_accuracy=1.0,
            candidate_false_success_rate=0.0,
        contamination=_clean_contamination(),
        )



def test_held_out_regression_gate_exports_from_evals_package() -> None:
    from oai2.evals import CapabilityRegression as ExportedRegression
    from oai2.evals import CapabilityRegressionThreshold as ExportedThreshold
    from oai2.evals import HeldOutPromotionBudget as ExportedBudget
    from oai2.evals import HeldOutPromotionEvaluation as ExportedEvaluation
    from oai2.evals import evaluate_held_out_promotion as ExportedEvaluate
    from oai2.evals.regression import (
        CapabilityRegression,
        HeldOutPromotionEvaluation,
    )

    assert ExportedRegression is CapabilityRegression
    assert ExportedThreshold is CapabilityRegressionThreshold
    assert ExportedBudget is HeldOutPromotionBudget
    assert ExportedEvaluation is HeldOutPromotionEvaluation
    assert ExportedEvaluate is evaluate_held_out_promotion


# --- fail-closed reporting ---------------------------------------------------
#
# The promotion gate decides whether a candidate ships. It fails a capability
# only when the regression *exceeds* a tolerance, so a non-finite operand made
# every comparison False and silently disabled that check. Paired with a pass
# count larger than the case count, a candidate that failed every held-out
# case was promoted and the evidence recorded a pass rate of 9.9 and a NaN.
#
# These tests pin the fail-closed behaviour established in #239 and match the
# numeric validation sibling gates already perform in oai2.evals.qos and
# oai2.evals.truth.

_POISON_IDS = tuple(f"p{i}" for i in range(10))


def _poisoned_report(
    *,
    n_passed: int = 0,
    score_value: float = 0.0,
) -> SuiteReport:
    scores = [
        CapabilityScore(
            case_id=case_id,
            capability="verification",
            score=score_value,
            passed=False,
            matched_pattern=None,
            forbidden_matched=(),
        )
        for case_id in _POISON_IDS
    ]
    return SuiteReport(
        suite_id="verification",
        capability="verification",
        runtime="held-out-runtime",
        n_cases=len(scores),
        n_passed=n_passed,
        scores=scores,
    )


def _evaluate(candidate: SuiteReport):
    return evaluate_held_out_promotion(
        baseline_reports=[
            _report("verification", "verification", ((c, True) for c in _POISON_IDS)),
        ],
        candidate_reports=[candidate],
        budget=_budget(),
        training_case_ids=set(),
        candidate_abstention_accuracy=1.0,
        candidate_false_success_rate=0.0,
        contamination=_clean_contamination(),
    )


def test_failed_candidate_cannot_be_promoted_by_poisoned_report_numbers() -> None:
    """The exact bypass: 0/10 passed, yet n_passed=99 and a NaN mean."""
    poisoned = _poisoned_report(n_passed=99, score_value=float("nan"))
    with pytest.raises(ValueError, match="n_passed=99"):
        _evaluate(poisoned)


def test_nan_scores_are_rejected_before_ranking() -> None:
    with pytest.raises(ValueError, match="mean_score"):
        _evaluate(_poisoned_report(score_value=float("nan")))


def test_infinite_scores_are_rejected_before_ranking() -> None:
    with pytest.raises(ValueError, match="mean_score"):
        _evaluate(_poisoned_report(score_value=float("inf")))


def test_scores_outside_documented_range_are_rejected() -> None:
    with pytest.raises(ValueError, match="mean_score"):
        _evaluate(_poisoned_report(score_value=5.0))


def test_negative_case_count_is_rejected() -> None:
    report = _poisoned_report()
    report.n_cases = -1
    with pytest.raises(ValueError, match="invalid n_cases"):
        _evaluate(report)


def test_negative_pass_count_is_rejected() -> None:
    report = _poisoned_report(n_passed=0)
    report.n_passed = -3
    with pytest.raises(ValueError, match="invalid n_passed"):
        _evaluate(report)


def test_poisoned_baseline_is_also_rejected() -> None:
    """A corrupted baseline must not be usable to manufacture a pass either."""
    baseline = _poisoned_report(n_passed=99, score_value=float("nan"))
    with pytest.raises(ValueError, match="baseline"):
        evaluate_held_out_promotion(
            baseline_reports=[baseline],
            candidate_reports=[
                _report("verification", "verification", ((c, True) for c in _POISON_IDS)),
            ],
            budget=_budget(),
            training_case_ids=set(),
            candidate_abstention_accuracy=1.0,
            candidate_false_success_rate=0.0,
            contamination=_clean_contamination(),
        )


def test_honest_failing_candidate_still_fails_on_merit() -> None:
    """Guard against 'fixing' the bypass by over-rejecting valid reports."""
    result = _evaluate(_poisoned_report(n_passed=0, score_value=0.0))
    assert result.passed is False
    assert "verification:pass_rate_regression" in result.failures
    assert "verification:mean_score_regression" in result.failures


def test_evaluated_evidence_contains_no_non_finite_values() -> None:
    """Whatever the gate returns must survive strict JSON serialisation.

    The bypass also emitted a NaN into the evidence record, which
    ``json.dumps(..., allow_nan=False)`` rejects, so the artifact claiming to
    document the decision could not be written to an evidence store.
    """
    import dataclasses
    import json

    result = _evaluate(_poisoned_report(n_passed=0, score_value=0.0))
    for regression in result.capability_regressions:
        json.dumps(dataclasses.asdict(regression), allow_nan=False)


class TestHeldOutGateRefusesAContaminatedCorpus:
    """A case ID absent from `training_case_ids` is not evidence of isolation.

    `training_case_ids` is an identity check. Renaming a training case makes
    it "held out" while its text is byte-identical, and the identity check
    passes cleanly. REQ-BENCH-012/013 already measure this, and `e3f4c64`
    recorded the shipped `deterministic_hard` suite as fully contaminated
    across all 12 cases -- so an audit that the gate never consults is an
    audit whose result cannot change the decision.
    """

    _TRAINING_TEXT = (
        "Refactor the payment retry loop so transient gateway errors back off "
        "exponentially and the idempotency key survives a process restart."
    )

    @classmethod
    def _contaminated(cls) -> ContaminationReport:
        case = SimpleNamespace(
            case_id="held-hard-001",
            prompt=cls._TRAINING_TEXT,
            expected_patterns=(),
            forbidden_patterns=(),
        )
        corpus = [
            CorpusEntry(
                entry_id="train-077", text=cls._TRAINING_TEXT, provenance="synthetic-v3"
            )
        ]
        return check_contamination(fingerprint_cases([case]), corpus)

    def _evaluate(self, contamination: ContaminationReport):
        return evaluate_held_out_promotion(
            baseline_reports=[
                _report("verification", "verification", ((c, True) for c in _POISON_IDS)),
            ],
            candidate_reports=[
                _report("verification", "verification", ((c, True) for c in _POISON_IDS))
            ],
            budget=_budget(),
            training_case_ids=set(),
            candidate_abstention_accuracy=1.0,
            candidate_false_success_rate=0.0,
            contamination=contamination,
        )

    def test_the_audit_really_does_find_the_contamination(self) -> None:
        """The control on the control: the fixture must be genuinely dirty.

        Without this, a broken fixture would make the gate look like it is
        refusing for the right reason when it is not.
        """
        report = self._contaminated()
        assert report.clean is False
        assert report.quarantined_case_ids == ("held-hard-001",)
        assert "exact_prompt" in report.kinds()

    def test_a_contaminated_corpus_cannot_be_promoted(self) -> None:
        result = self._evaluate(self._contaminated())
        assert result.passed is False
        assert any(f.startswith("contamination:") for f in result.failures)
        # The record names the cases, so a caller can swap in clean ones
        # rather than re-deriving which were at fault.
        assert result.quarantined_case_ids == ("held-hard-001",)

    def test_no_regression_is_computed_over_a_contaminated_corpus(self) -> None:
        """Refuse BEFORE comparing numbers.

        A regression measured across a contaminated held-out set is not a
        regression, and emitting one would lend the invalid evidence a number
        that downstream readers could quote.
        """
        result = self._evaluate(self._contaminated())
        assert result.capability_regressions == ()

    def test_the_audit_argument_is_required_not_optional(self) -> None:
        """An optional audit is an audit that can be omitted.

        A default of `None` treated as "clean" would restore the original
        defect one indirection away, which is exactly how this was missed the
        first time.
        """
        with pytest.raises(TypeError, match="contamination"):
            evaluate_held_out_promotion(  # type: ignore[call-arg]
                baseline_reports=[
                    _report("verification", "verification", ((c, True) for c in _POISON_IDS))
                ],
                candidate_reports=[
                    _report("verification", "verification", ((c, True) for c in _POISON_IDS))
                ],
                budget=_budget(),
                training_case_ids=set(),
                candidate_abstention_accuracy=1.0,
                candidate_false_success_rate=0.0,
            )

    def test_a_clean_corpus_still_promotes(self) -> None:
        """The control that must pass in both worlds.

        Without this the fix could be satisfied by failing everything.
        """
        result = self._evaluate(_clean_contamination())
        assert result.passed is True
        assert result.quarantined_case_ids == ()
