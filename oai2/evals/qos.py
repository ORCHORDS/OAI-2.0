"""Deterministic end-to-end service-quality metrics.

The helpers in this module deliberately keep model throughput separate from
user-visible usefulness. They describe workload identity, collect per-run
latency/useful-work evidence, summarize tail distributions, and evaluate a
versioned caller-supplied budget without inventing hardware targets.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from enum import StrEnum


class WorkloadClass(StrEnum):
    """Canonical QoS workload classes owned by WP-76."""

    FURIOUS = "FURIOUS"
    NORMAL = "NORMAL"
    DEEP = "DEEP"
    SWARM = "SWARM"
    VISION = "VISION"
    RETRIEVAL_HEAVY = "RETRIEVAL_HEAVY"


class BudgetKind(StrEnum):
    """Keep aspirational research targets distinct from measured service gates."""

    RESEARCH_TARGET = "research_target"
    SERVICE_BUDGET = "service_budget"


@dataclass(slots=True, frozen=True)
class WorkloadBudget:
    """Versioned tail/useful-work limits for one hardware/config identity."""

    version: str
    workload: WorkloadClass
    kind: BudgetKind
    target_hardware: str
    config_id: str
    first_useful_action_p95_ms: float
    end_to_end_p95_ms: float
    end_to_end_p99_ms: float
    max_false_success_rate: float
    min_verified_success_rate: float
    max_deadline_miss_rate: float = 1.0

    def __post_init__(self) -> None:
        for name in ("version", "target_hardware", "config_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        for name in (
            "first_useful_action_p95_ms",
            "end_to_end_p95_ms",
            "end_to_end_p99_ms",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if self.end_to_end_p99_ms < self.end_to_end_p95_ms:
            raise ValueError("end_to_end_p99_ms must be >= end_to_end_p95_ms")
        for name in (
            "max_false_success_rate",
            "min_verified_success_rate",
            "max_deadline_miss_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(slots=True, frozen=True)
class WorkloadSample:
    """One replayed workload with end-to-end and useful-work evidence."""

    workload: WorkloadClass
    target_hardware: str
    config_id: str
    ttft_ms: float
    first_useful_action_ms: float
    end_to_end_ms: float
    declared_success: bool
    verified_success: bool
    verified_actions: int = 0
    generated_tokens: int = 0
    # Components that may not have OCCURRED in a given replay are `None`,
    # never 0.0.
    #
    # A 0.0 here is a measurement claim: the tool ran and took no time. The
    # default used to make it mean "no tool ran", and `summarize_samples`
    # then built a latency distribution out of that absence. Measured on
    # eec53ac, a 40-replay workload that never invoked a tool reported:
    #
    #     ttft_ms    count=40  mean=100.00  p95=100.00
    #     tool_ms    count=40  mean=  0.00  p95=  0.00
    #
    # `count=40` is not "unmeasured" -- it is false: zero tool invocations
    # occurred. And p95=0.00 is the best value any latency budget can be
    # given, produced entirely by absence, so a `tool_ms.p95 <= X` budget
    # passes trivially for a workload that never exercised the tool at all.
    # The two lines sit in the same report with the same shape, one honest
    # and one not, and nothing marked which was which.
    #
    # This is the same failure closed three times elsewhere in this
    # repository: `no_comparable_samples`, `capability_measured`, and
    # `RetrievalMetrics.insufficient_evidence`. An absence must never
    # render as a clean measurement.
    decode_tokens_per_second: float | None = None
    tool_ms: float | None = None
    retrieval_ms: float | None = None
    vision_ms: float | None = None
    build_test_ms: float | None = None
    deadline_missed: bool = False

    def __post_init__(self) -> None:
        for name in ("target_hardware", "config_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        for name in ("ttft_ms", "first_useful_action_ms", "end_to_end_ms"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        for name in (
            "decode_tokens_per_second",
            "tool_ms",
            "retrieval_ms",
            "vision_ms",
            "build_test_ms",
        ):
            value = getattr(self, name)
            # None means the component did not occur in this replay. 0.0 means
            # it did and took no measurable time. Both are legal; conflating
            # them is the defect.
            if value is None:
                continue
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be null or finite and >= 0")
        for name in ("verified_actions", "generated_tokens"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.generated_tokens > 0 and self.decode_tokens_per_second is None:
            raise ValueError(
                "decode_tokens_per_second is required when generated_tokens > 0: "
                "a replay that produced tokens always has a decode rate, and "
                "leaving it null would report generated output with no rate"
            )
        if self.ttft_ms > self.end_to_end_ms:
            raise ValueError("ttft_ms cannot exceed end_to_end_ms")
        if self.first_useful_action_ms > self.end_to_end_ms:
            raise ValueError("first_useful_action_ms cannot exceed end_to_end_ms")
        if not isinstance(self.deadline_missed, bool):
            raise ValueError("deadline_missed must be a boolean")

    @property
    def false_success(self) -> bool:
        return self.declared_success and not self.verified_success


@dataclass(slots=True, frozen=True)
class Distribution:
    """Latency distribution over the replays where a component OCCURRED.

    `mean`, `variance`, `p50` and `count` are defined for any non-empty
    sample set. `p95` and `p99` are not: they are None when `count < 2`.

    REQ-QOS-014 asks for percentiles "where repetition count supports tail
    estimates" and for sample count/variance beside them. One replay
    supports neither tail, and `_percentile` used to supply them anyway
    via a `len(ordered) == 1` special case that returns the single value
    for *every* requested percentile. One measurement therefore rendered
    as five agreeing statistics:

        Distribution(count=1, mean=100.0, variance=0.0,
                     p50=100.0, p95=100.0, p99=100.0)

    The percentiles are the defect, and they are a different kind of claim
    from the variance. The population variance of a singleton genuinely
    is 0.0 -- it is a correct answer to a defined question, and `count`
    travels beside it (reviewed and recorded in
    tests/test_absent_measurement_invariant.py). A p99 from one point is
    not a correct answer to anything: there is no tail, and the
    interpolation helper was manufacturing a rank that no observation
    occupied. A gate reading `end_to_end_ms.p99 <= X` then passed on a
    number that was never a tail.

    This is the same defect already closed four times in this repository
    on the *component* axis (`tool_ms` and friends,
    `verified_actions_per_second`, `capability_measured`,
    `RetrievalMetrics.insufficient_evidence`); here it appears on the
    *repetition-count* axis. Consumers must treat None as "not measured" --
    `evaluate_budget` and `evaluate_promotion` fail on it rather than
    skipping, so an unmeasured tail can never satisfy a limit.
    """

    count: int
    mean: float
    variance: float
    p50: float
    p95: float | None
    p99: float | None

    @property
    def tail_measured(self) -> bool:
        """Whether a tail estimate exists for this sample set."""
        return self.p95 is not None


@dataclass(slots=True, frozen=True)
class WorkloadReport:
    workload: WorkloadClass
    target_hardware: str
    config_id: str
    sample_count: int
    ttft_ms: Distribution
    first_useful_action_ms: Distribution
    end_to_end_ms: Distribution
    # Each is None when NO replay in the sample set exercised that
    # component. A Distribution here is always backed by its own `count`
    # real measurements, so the two can no longer be confused.
    decode_tokens_per_second: Distribution | None
    tool_ms: Distribution | None
    retrieval_ms: Distribution | None
    vision_ms: Distribution | None
    build_test_ms: Distribution | None
    verified_success_rate: float
    false_success_rate: float
    deadline_miss_rate: float
    # None when the sample set has no measurable elapsed time at all, i.e.
    # every replay recorded `end_to_end_ms == 0.0`. A rate is actions divided
    # by time; with no time there is no rate to report, so the figure is
    # UNDEFINED rather than zero. 0.0 remains the genuine measured value for a
    # run that spent real time and completed no verified actions.
    verified_actions_per_second: float | None

    @property
    def throughput_measured(self) -> bool:
        """Whether a throughput rate could be derived at all.

        False means no replay contributed measurable elapsed time, so
        `verified_actions_per_second` is None. This is distinct from a
        measured 0.0, which means time WAS measured and no action was
        verified inside it.
        """
        return self.verified_actions_per_second is not None


@dataclass(slots=True, frozen=True)
class BudgetEvaluation:
    budget_version: str
    budget_kind: BudgetKind
    passed: bool
    failures: tuple[str, ...]


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _distribution(values: list[float]) -> Distribution:
    if not values:
        raise ValueError("at least one value is required")
    # A single replay has no tail, so the percentiles are absent rather
    # than invented. See Distribution's docstring. The variance stays a
    # float: the population variance of a singleton really is 0.0, and
    # `count` travels with it.
    repeatable = len(values) >= 2
    return Distribution(
        count=len(values),
        mean=statistics.fmean(values),
        variance=statistics.pvariance(values) if repeatable else 0.0,
        p50=_percentile(values, 0.50),
        p95=_percentile(values, 0.95) if repeatable else None,
        p99=_percentile(values, 0.99) if repeatable else None,
    )


def _optional_distribution(values: list[float | None]) -> Distribution | None:
    """Build a distribution over the replays where the component OCCURRED.

    Returns None when it never occurred, so an unexercised component is not
    a distribution of zeros claiming a non-zero count.
    """
    measured = [float(value) for value in values if value is not None]
    return _distribution(measured) if measured else None


def summarize_samples(samples: list[WorkloadSample]) -> WorkloadReport:
    """Aggregate one homogeneous workload/config sample set."""
    if not samples:
        raise ValueError("at least one workload sample is required")

    first = samples[0]
    identity = (first.workload, first.target_hardware, first.config_id)
    for sample in samples[1:]:
        if (sample.workload, sample.target_hardware, sample.config_id) != identity:
            raise ValueError("all samples must share workload, hardware, and config identity")

    total_seconds = sum(sample.end_to_end_ms for sample in samples) / 1000.0
    verified_actions = sum(sample.verified_actions for sample in samples)
    n = len(samples)
    return WorkloadReport(
        workload=first.workload,
        target_hardware=first.target_hardware,
        config_id=first.config_id,
        sample_count=n,
        ttft_ms=_distribution([sample.ttft_ms for sample in samples]),
        first_useful_action_ms=_distribution(
            [sample.first_useful_action_ms for sample in samples]
        ),
        end_to_end_ms=_distribution([sample.end_to_end_ms for sample in samples]),
        decode_tokens_per_second=_optional_distribution(
            [sample.decode_tokens_per_second for sample in samples]
        ),
        tool_ms=_optional_distribution([sample.tool_ms for sample in samples]),
        retrieval_ms=_optional_distribution(
            [sample.retrieval_ms for sample in samples]
        ),
        vision_ms=_optional_distribution([sample.vision_ms for sample in samples]),
        build_test_ms=_optional_distribution(
            [sample.build_test_ms for sample in samples]
        ),
        verified_success_rate=sum(sample.verified_success for sample in samples) / n,
        false_success_rate=sum(sample.false_success for sample in samples) / n,
        deadline_miss_rate=sum(sample.deadline_missed for sample in samples) / n,
        # Division by zero would raise, and the reflexive repair was to report
        # 0.0. But 0.0 is a CLAIM: it says time was measured and no action was
        # verified within it. With total_seconds == 0.0 no elapsed time was
        # ever measured, so the rate does not exist. Reporting None makes the
        # absence visible instead of inventing a clean figure for it.
        verified_actions_per_second=(
            verified_actions / total_seconds if total_seconds > 0.0 else None
        ),
    )


def evaluate_budget(report: WorkloadReport, budget: WorkloadBudget) -> BudgetEvaluation:
    """Evaluate a report against the exact versioned budget identity."""
    if (
        report.workload != budget.workload
        or report.target_hardware != budget.target_hardware
        or report.config_id != budget.config_id
    ):
        raise ValueError("report and budget identity do not match")

    failures: list[str] = []
    # An absent tail is "unmeasurable", not "inside the limit". Marking it
    # absent and then comparing nothing would leave the gate passing on a
    # number that was never measured, so each unsupported tail is its own
    # failure and the corresponding *_over_budget check is skipped.
    checks: tuple[tuple[str, float | None, float, str], ...] = (
        (
            "first_useful_action_p95",
            report.first_useful_action_ms.p95,
            budget.first_useful_action_p95_ms,
            "first_useful_action_p95_unsupported",
        ),
        ("end_to_end_p95", report.end_to_end_ms.p95, budget.end_to_end_p95_ms, "end_to_end_p95_unsupported"),
        ("end_to_end_p99", report.end_to_end_ms.p99, budget.end_to_end_p99_ms, "end_to_end_p99_unsupported"),
    )
    for name, actual, maximum, unsupported in checks:
        if actual is None:
            failures.append(unsupported)
        elif actual > maximum:
            failures.append(name)
    if report.false_success_rate > budget.max_false_success_rate:
        failures.append("false_success_rate")
    if report.verified_success_rate < budget.min_verified_success_rate:
        failures.append("verified_success_rate")
    if report.deadline_miss_rate > budget.max_deadline_miss_rate:
        failures.append("deadline_miss_rate")
    return BudgetEvaluation(
        budget_version=budget.version,
        budget_kind=budget.kind,
        passed=not failures,
        failures=tuple(failures),
    )


@dataclass(slots=True, frozen=True)
class PromotionEvaluation:
    passed: bool
    failures: tuple[str, ...]


def evaluate_promotion(
    candidate: WorkloadReport,
    baseline: WorkloadReport,
    *,
    max_tail_regression_ratio: float = 1.0,
    max_deadline_miss_increase: float = 0.0,
) -> PromotionEvaluation:
    """Reject candidates whose tails/deadline misses regress versus baseline.

    Correctness is gated here too, and not as an optional extra. A promotion
    gate whose only checks are latency would promote a candidate that is
    exactly as fast as the baseline, never verified-correct, and reports a
    false success every time. Measured before this change, with a candidate
    at ``false_success_rate=1.0`` / ``verified_success_rate=0.0`` and
    latency identical to the baseline::

        evaluate_promotion -> passed=True  failures=()
        evaluate_budget    -> passed=False failures=('false_success_rate',
                                                     'verified_success_rate')

    so the gate that makes the promotion decision was the one gate that could
    not see it. The absolute limits live in :class:`WorkloadBudget` and stay
    there; what belongs here is the same no-regression rule the deadline
    check already applies via ``max_deadline_miss_increase=0.0``.
    """
    if (
        candidate.workload != baseline.workload
        or candidate.target_hardware != baseline.target_hardware
    ):
        raise ValueError("candidate and baseline workload/hardware must match")
    if not math.isfinite(max_tail_regression_ratio) or max_tail_regression_ratio < 1.0:
        raise ValueError("max_tail_regression_ratio must be finite and >= 1")
    if (
        not math.isfinite(max_deadline_miss_increase)
        or max_deadline_miss_increase < 0.0
        or max_deadline_miss_increase > 1.0
    ):
        raise ValueError("max_deadline_miss_increase must be between 0 and 1")

    failures: list[str] = []
    # Same rule as evaluate_budget: a tail that was never measured is a
    # promotion blocker, not a pass. Comparing `None > None` would raise,
    # and silently treating either side as absent-tolerant would let a
    # one-replay candidate replace a measured baseline.
    tail_checks: tuple[tuple[str, float | None, float | None, str], ...] = (
        (
            "p95_regression",
            candidate.end_to_end_ms.p95,
            baseline.end_to_end_ms.p95,
            "p95_unsupported",
        ),
        (
            "p99_regression",
            candidate.end_to_end_ms.p99,
            baseline.end_to_end_ms.p99,
            "p99_unsupported",
        ),
    )
    for name, candidate_tail, baseline_tail, unsupported in tail_checks:
        if candidate_tail is None or baseline_tail is None:
            failures.append(unsupported)
        elif candidate_tail > baseline_tail * max_tail_regression_ratio:
            failures.append(name)
    if (
        candidate.deadline_miss_rate
        > baseline.deadline_miss_rate + max_deadline_miss_increase
    ):
        failures.append("deadline_miss_regression")
    # A candidate may not be *more* wrong than the baseline it replaces, even
    # when it is no slower. These are regression checks, not the absolute
    # limits: a baseline that is already failing its budget is
    # evaluate_budget's problem, and re-deciding it here would make two gates
    # disagree about the same report for no additional safety.
    if candidate.false_success_rate > baseline.false_success_rate:
        failures.append("false_success_regression")
    if candidate.verified_success_rate < baseline.verified_success_rate:
        failures.append("verified_success_regression")
    return PromotionEvaluation(passed=not failures, failures=tuple(failures))


def useful_work_rank_key(report: WorkloadReport) -> tuple[float, float, float, float]:
    """Lower is better; usefulness/tail latency outrank raw token throughput.

    An absent tail ranks as infinity, not zero. A report that never
    measured a p95 has no useful-action latency to rank on, and scoring
    that absence as 0.0 would place it ahead of every report that did
    measure a fast tail -- the same absence-reads-as-clean defect one level
    up, where the reward is for lacking evidence.
    """
    return (
        report.false_success_rate,
        -report.verified_success_rate,
        report.first_useful_action_ms.p95
        if report.first_useful_action_ms.p95 is not None
        else math.inf,
        report.end_to_end_ms.p95 if report.end_to_end_ms.p95 is not None else math.inf,
    )


__all__ = [
    "BudgetEvaluation",
    "BudgetKind",
    "Distribution",
    "PromotionEvaluation",
    "WorkloadBudget",
    "WorkloadClass",
    "WorkloadReport",
    "WorkloadSample",
    "evaluate_budget",
    "evaluate_promotion",
    "summarize_samples",
    "useful_work_rank_key",
]
