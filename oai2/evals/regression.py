"""Held-out capability regression and promotion gates for WI-EVAL-002.

This layer compares baseline and candidate capability reports on the same
held-out case IDs, rejects train/held-out overlap, applies explicit tolerances
per capability class, and treats abstention/false-success as first-class gates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .contamination import ContaminationReport


class CapabilityScoreLike(Protocol):
    case_id: str
    # The promotion gate validates the per-case score, so a conforming score
    # must expose it. Declared here rather than relied on dynamically.
    score: float


class SuiteReportLike(Protocol):
    capability: str
    n_cases: int
    n_passed: int
    mean_score: float
    scores: list[CapabilityScoreLike]


@dataclass(slots=True, frozen=True)
class CapabilityRegressionThreshold:
    capability: str
    max_pass_rate_regression: float
    max_mean_score_regression: float

    def __post_init__(self) -> None:
        if not isinstance(self.capability, str) or not self.capability.strip():
            raise ValueError("capability must be a non-empty string")
        _rate(self.max_pass_rate_regression, "max_pass_rate_regression")
        _rate(self.max_mean_score_regression, "max_mean_score_regression")


@dataclass(slots=True, frozen=True)
class HeldOutPromotionBudget:
    version: str
    thresholds: tuple[CapabilityRegressionThreshold, ...]
    min_abstention_accuracy: float
    max_false_success_rate: float

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be a non-empty string")
        if not self.thresholds:
            raise ValueError("at least one capability threshold is required")
        capabilities = [item.capability for item in self.thresholds]
        if len(set(capabilities)) != len(capabilities):
            raise ValueError("capability thresholds must be unique")
        _rate(self.min_abstention_accuracy, "min_abstention_accuracy")
        _rate(self.max_false_success_rate, "max_false_success_rate")


@dataclass(slots=True, frozen=True)
class CapabilityRegression:
    capability: str
    baseline_pass_rate: float
    candidate_pass_rate: float
    pass_rate_regression: float
    baseline_mean_score: float
    candidate_mean_score: float
    mean_score_regression: float
    passed: bool
    failures: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class HeldOutPromotionEvaluation:
    budget_version: str
    passed: bool
    failures: tuple[str, ...]
    capability_regressions: tuple[CapabilityRegression, ...]
    #: Held-out case IDs the contamination audit quarantined. Non-empty means
    #: the corpus this decision rests on was measured as contaminated, so the
    #: record has to say which cases rather than only that it failed.
    quarantined_case_ids: tuple[str, ...] = ()


def evaluate_held_out_promotion(
    *,
    baseline_reports: tuple[SuiteReportLike, ...] | list[SuiteReportLike],
    candidate_reports: tuple[SuiteReportLike, ...] | list[SuiteReportLike],
    budget: HeldOutPromotionBudget,
    training_case_ids: set[str] | frozenset[str],
    candidate_abstention_accuracy: float,
    candidate_false_success_rate: float,
    contamination: ContaminationReport,
) -> HeldOutPromotionEvaluation:
    """Evaluate a candidate against baseline on isolated held-out cases.

    Isolation is established two ways, and both are required. `training_case_ids`
    is the cheap identity check; `contamination` is the content check that
    identity cannot substitute for. A caller must supply both — `contamination`
    is a required keyword rather than an optional one precisely because an
    optional audit is an audit that can be omitted, which puts this gate back
    where it started.

    A contaminated corpus fails the gate before any regression is computed.
    `quarantined_case_ids` on the result names the cases, so a caller can swap
    in clean ones (see :func:`~oai2.evals.contamination.select_clean_cases`) and
    re-run rather than having to re-derive which cases were at fault.
    """
    abstention = _rate(candidate_abstention_accuracy, "candidate_abstention_accuracy")
    false_success = _rate(candidate_false_success_rate, "candidate_false_success_rate")

    baseline = _reports_by_capability(baseline_reports, "baseline")
    candidate = _reports_by_capability(candidate_reports, "candidate")
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate capabilities must match")

    baseline_case_ids = _case_ids(baseline_reports)
    candidate_case_ids = _case_ids(candidate_reports)
    if baseline_case_ids != candidate_case_ids:
        raise ValueError("baseline and candidate must use identical held-out case IDs")
    overlap = candidate_case_ids.intersection(training_case_ids)
    if overlap:
        joined = ", ".join(sorted(overlap))
        raise ValueError(f"held-out cases overlap training/tuning inputs: {joined}")

    # A case ID being absent from `training_case_ids` says nothing about the
    # case being unseen. Renaming a training case makes it "held out" while
    # its text is byte-identical, and the ID check above passes cleanly. The
    # repository already measures this (REQ-BENCH-012/013) and has measured
    # the shipped deterministic_hard suite as fully contaminated, so an audit
    # that is not consulted here is an audit whose result cannot change the
    # promotion decision.
    #
    # Fails rather than raises, and refuses BEFORE any regression is computed:
    # a regression measured across a contaminated corpus is not a regression,
    # and reporting one would lend the invalid evidence a number.
    if not contamination.clean:
        kinds = contamination.kinds() or ("unknown",)
        return HeldOutPromotionEvaluation(
            budget_version=budget.version,
            passed=False,
            failures=tuple(f"contamination:{kind}" for kind in kinds),
            capability_regressions=(),
            quarantined_case_ids=contamination.quarantined_case_ids,
        )

    _validate_report_numbers(baseline_reports, "baseline")
    _validate_report_numbers(candidate_reports, "candidate")

    failures: list[str] = []
    regressions: list[CapabilityRegression] = []
    threshold_by_capability = {item.capability: item for item in budget.thresholds}

    missing_thresholds = set(candidate) - set(threshold_by_capability)
    if missing_thresholds:
        joined = ", ".join(sorted(missing_thresholds))
        raise ValueError(f"missing capability thresholds: {joined}")

    for capability in sorted(candidate):
        threshold = threshold_by_capability[capability]
        baseline_pass, baseline_mean = _aggregate_reports(baseline[capability])
        candidate_pass, candidate_mean = _aggregate_reports(candidate[capability])
        pass_regression = max(baseline_pass - candidate_pass, 0.0)
        mean_regression = max(baseline_mean - candidate_mean, 0.0)

        capability_failures: list[str] = []
        if pass_regression > threshold.max_pass_rate_regression:
            capability_failures.append("pass_rate_regression")
        if mean_regression > threshold.max_mean_score_regression:
            capability_failures.append("mean_score_regression")
        if capability_failures:
            failures.extend(
                f"{capability}:{failure}" for failure in capability_failures
            )

        regressions.append(
            CapabilityRegression(
                capability=capability,
                baseline_pass_rate=baseline_pass,
                candidate_pass_rate=candidate_pass,
                pass_rate_regression=pass_regression,
                baseline_mean_score=baseline_mean,
                candidate_mean_score=candidate_mean,
                mean_score_regression=mean_regression,
                passed=not capability_failures,
                failures=tuple(capability_failures),
            )
        )

    if abstention < budget.min_abstention_accuracy:
        failures.append("abstention_accuracy")
    if false_success > budget.max_false_success_rate:
        failures.append("false_success_rate")

    return HeldOutPromotionEvaluation(
        budget_version=budget.version,
        passed=not failures,
        failures=tuple(failures),
        capability_regressions=tuple(regressions),
    )


def _validate_report_numbers(
    reports: tuple[SuiteReportLike, ...] | list[SuiteReportLike],
    name: str,
) -> None:
    """Reject reports whose numbers cannot support a regression decision.

    The gate compares ``baseline - candidate`` and fails only when the result
    *exceeds* a tolerance. A non-finite operand makes that comparison False
    for every tolerance, so the corresponding check silently stops being a
    gate. Combined with a pass count larger than the case count, a candidate
    that failed every held-out case can be promoted.

    Sibling gates in this package already refuse non-finite and out-of-range
    inputs (``oai2.evals.qos``, ``oai2.evals.truth``); this brings the
    promotion gate in line, and follows the fail-closed rule established for
    evidence status in #239. An unusable report is a caller defect, so it is
    raised rather than recorded as a candidate failure.
    """
    for index, report in enumerate(reports):
        n_cases = report.n_cases
        n_passed = report.n_passed
        where = f"{name} report {index} ({report.capability})"
        if isinstance(n_cases, bool) or not isinstance(n_cases, int) or n_cases < 0:
            raise ValueError(f"{where} has invalid n_cases: {n_cases!r}")
        if isinstance(n_passed, bool) or not isinstance(n_passed, int) or n_passed < 0:
            raise ValueError(f"{where} has invalid n_passed: {n_passed!r}")
        if n_passed > n_cases:
            raise ValueError(
                f"{where} reports n_passed={n_passed} for n_cases={n_cases}; "
                "a pass rate above 1.0 is not evidence"
            )
        _rate(report.mean_score, f"{where} mean_score")
        for score in report.scores:
            _rate(score.score, f"{where} case {score.case_id!r} score")


def _reports_by_capability(
    reports: tuple[SuiteReportLike, ...] | list[SuiteReportLike],
    name: str,
) -> dict[str, list[SuiteReportLike]]:
    if not reports:
        raise ValueError(f"{name}_reports must not be empty")
    out: dict[str, list[SuiteReportLike]] = {}
    for report in reports:
        out.setdefault(report.capability, []).append(report)
    return out


def _case_ids(reports: tuple[SuiteReportLike, ...] | list[SuiteReportLike]) -> set[str]:
    ids: set[str] = set()
    for report in reports:
        for score in report.scores:
            if score.case_id in ids:
                raise ValueError(f"duplicate held-out case ID: {score.case_id}")
            ids.add(score.case_id)
    if not ids:
        raise ValueError("held-out reports must contain per-case scores")
    return ids


def _aggregate_reports(reports: list[SuiteReportLike]) -> tuple[float, float]:
    total_cases = sum(report.n_cases for report in reports)
    if total_cases <= 0:
        raise ValueError("capability reports must contain cases")
    passed = sum(report.n_passed for report in reports)
    weighted_score = sum(report.mean_score * report.n_cases for report in reports)
    return passed / total_cases, weighted_score / total_cases


def _rate(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be between 0 and 1")
    return float(value)


__all__ = [
    "CapabilityRegressionThreshold",
    "HeldOutPromotionBudget",
    "CapabilityRegression",
    "HeldOutPromotionEvaluation",
    "evaluate_held_out_promotion",
]
