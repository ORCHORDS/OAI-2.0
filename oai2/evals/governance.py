"""Evaluation revisioning, comparability and audit-before-promotion (WP-56 / WI-BENCH-002).

Why this module exists
----------------------

:func:`oai2.evals.regression.evaluate_held_out_promotion` is the existing
promotion gate and it checks two things about the eval set: that baseline and
candidate cover the same ``case_id`` set, and that none of those ids appear in
the training inputs. Both are necessary. Neither is sufficient.

Demonstrated on current ``main`` before this module existed: two
``SuiteReport`` objects carrying the same ``case_id`` but *entirely different
prompts behind it* compared as ``passed=True`` with no failures. The gate
cannot see the difference, because the prompt is not an input to the
comparison. A scorer change is likewise invisible — ``scorer`` never reaches
this function.

So a result set could be compared against a baseline, the score could move, and
nobody could say afterwards which eval revision produced either number. That is
REQ-BENCH-021, REQ-BENCH-022 and REQ-BENCH-025, and it is the failure mode
issue #172's engine-readiness audit describes when it says a result that cannot
be located cannot be governed.

Design points
-------------

- **A revision is a digest of content, not a label someone remembers to
  bump.** :func:`eval_revision` hashes the suite id, every case id, every
  prompt and the scorer identity, so editing one character of one prompt
  changes the revision. A hand-maintained version string cannot do that.
- **The scorer is part of the revision** (REQ-BENCH-025). A different scorer
  produces a different revision even when every prompt is byte-identical,
  because the numbers are not measuring the same thing.
- **An unknown contamination audit is not a clean one** (REQ-BENCH-024).
  :class:`AuditStatus` has a ``UNKNOWN`` member, and a record carrying it is
  refused for promotion rather than treated as passing. This is the same
  absence-as-clean failure the rest of the repository has been closing.
- **Non-comparable results are never silently averaged** (REQ-BENCH-023).
  :func:`build_trend` groups by comparability key and returns the groups; it
  does not return one blended number, because a blend across revisions is not
  a trend, it is two different experiments added together.
- **This does not replace the existing gate.** :func:`evaluate_governed_promotion`
  refuses on identity/audit grounds first, then delegates the numeric
  regression decision to ``evaluate_held_out_promotion`` unchanged. There is
  still exactly one promotion gate.

Status: IMPLEMENTED — unit-pinned in ``tests/test_governance.py`` and run over
the real registered suites by ``scripts/eval_governance_report.py``.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .contamination import (
    ContaminationPolicy,
    ContaminationReport,
    CorpusEntry,
    check_contamination,
    fingerprint_cases,
)
from .regression import (
    HeldOutPromotionBudget,
    HeldOutPromotionEvaluation,
    SuiteReportLike,
    evaluate_held_out_promotion,
)

__all__ = [
    "AuditStatus",
    "ComparabilityCheck",
    "ContaminationAudit",
    "EvalRevision",
    "GovernedPromotionEvaluation",
    "ResultRecord",
    "TrendGroup",
    "TrendReport",
    "audit_contamination",
    "build_trend",
    "compare_records",
    "eval_revision",
    "evaluate_governed_promotion",
]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


class AuditStatus(StrEnum):
    """Contamination-audit outcome for a result record.

    ``UNKNOWN`` is a first-class member, not an oversight. A record produced
    before audits existed carries no audit, and treating that as ``CLEAN``
    would let the oldest, least-governed numbers gate a promotion.
    """

    CLEAN = "clean"
    FINDINGS = "findings"
    UNKNOWN = "unknown"


@dataclass(slots=True, frozen=True)
class ContaminationAudit:
    """Audit outcome attached to a result record (REQ-BENCH-024).

    ``status`` is ``UNKNOWN`` when the record was not audited. The counts stay
    ``None`` in that case rather than reporting 0 findings, which would be the
    absence-as-clean defect.
    """

    status: AuditStatus
    audited_at_revision: str | None = None
    finding_count: int | None = None
    quarantined_case_ids: tuple[str, ...] = ()
    policy: ContaminationPolicy | None = None

    @property
    def blocks_promotion(self) -> bool:
        """A promotion needs a *clean* audit, not merely the absence of one."""
        return self.status is not AuditStatus.CLEAN

    @classmethod
    def unknown(cls) -> ContaminationAudit:
        return cls(status=AuditStatus.UNKNOWN)


@dataclass(slots=True, frozen=True)
class EvalRevision:
    """Content-addressed identity of an eval set plus its scorer.

    REQ-BENCH-021/022/025. ``revision_id`` changes when a prompt, a case id, the
    suite id, the scorer or the scorer version changes, and nothing else.
    """

    revision_id: str
    suite_id: str
    capability: str
    scorer: str
    scorer_version: str
    case_count: int

    def __post_init__(self) -> None:
        for name in ("revision_id", "suite_id", "capability", "scorer"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if not self.scorer_version.strip():
            raise ValueError("scorer_version must be non-empty")
        if isinstance(self.case_count, bool) or not isinstance(self.case_count, int):
            raise ValueError("case_count must be an integer")
        if self.case_count < 0:
            raise ValueError("case_count must be >= 0")

    def describe(self) -> str:
        return (
            f"{self.suite_id}/{self.capability}@{self.revision_id}"
            f"/{self.scorer}@{self.scorer_version}"
        )


def eval_revision(
    suite: object,
    *,
    scorer_version: str,
    scorer: str | None = None,
) -> EvalRevision:
    """Compute the content-addressed revision of a capability suite.

    Reads ``suite_id``, ``scorer``, and each case's ``case_id``/``prompt`` by
    name so it works on the existing :class:`~oai2.evals.CapabilitySuite`
    without a parallel type.

    ``scorer_version`` is explicit and required. A scorer with no version
    cannot be reasoned about: "the scorer changed" is not a reviewable claim
    when there is nothing to name the change against.
    """
    suite_id = getattr(suite, "suite_id", None)
    if not isinstance(suite_id, str) or not suite_id:
        raise ValueError("suite must expose a non-empty suite_id")
    resolved_scorer = scorer or getattr(suite, "scorer", None) or "unknown"
    if not isinstance(resolved_scorer, str) or not resolved_scorer:
        raise ValueError("scorer must be a non-empty string")
    if not isinstance(scorer_version, str) or not scorer_version.strip():
        raise ValueError("scorer_version must be a non-empty string")

    cases = list(getattr(suite, "cases", ()) or ())
    capability = getattr(suite, "capability", None) or suite_id
    if not isinstance(capability, str) or not capability:
        raise ValueError("capability must be a non-empty string")
    parts = [
        f"suite={suite_id}",
        f"capability={capability}",
        f"scorer={resolved_scorer}",
        f"scorer_version={scorer_version}",
    ]
    for case in sorted(cases, key=lambda c: str(getattr(c, "case_id", ""))):
        case_id = getattr(case, "case_id", None)
        prompt = getattr(case, "prompt", None)
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("every case must expose a non-empty case_id")
        if not isinstance(prompt, str):
            raise ValueError(f"case {case_id} must expose a string prompt")
        parts.append(f"{case_id}\x1f{prompt}")

    return EvalRevision(
        revision_id=_digest("\x1e".join(parts)),
        suite_id=suite_id,
        capability=capability,
        scorer=resolved_scorer,
        scorer_version=scorer_version,
        case_count=len(cases),
    )


def audit_contamination(
    suite: object,
    corpus: Sequence[CorpusEntry],
    *,
    policy: ContaminationPolicy | None = None,
    revision: EvalRevision | None = None,
) -> ContaminationAudit:
    """Run the #171 auditor over a suite and package the outcome for a record."""
    active = policy or ContaminationPolicy()
    cases = list(getattr(suite, "cases", ()) or ())
    if not cases:
        # No cases means nothing was audited. That is UNKNOWN, not CLEAN.
        return ContaminationAudit(status=AuditStatus.UNKNOWN, policy=active)
    report = check_contamination(fingerprint_cases(cases, policy=active), corpus, policy=active)
    status = AuditStatus.FINDINGS if report.findings else AuditStatus.CLEAN
    return ContaminationAudit(
        status=status,
        audited_at_revision=revision.revision_id if revision else None,
        finding_count=len(report.findings),
        quarantined_case_ids=report.quarantined_case_ids,
        policy=active,
    )


@dataclass(slots=True, frozen=True)
class ResultRecord:
    """One measured result, tied to the exact eval revision that produced it."""

    label: str
    revision: EvalRevision
    candidate_id: str
    audit: ContaminationAudit
    pass_rate: float
    mean_score: float
    recorded_at: str = ""

    def __post_init__(self) -> None:
        for name in ("label", "candidate_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        for name in ("pass_rate", "mean_score"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite value in [0, 1]")

    @property
    def comparability_key(self) -> tuple[str, str, str, str]:
        """What must match for two records to be one trend."""
        return (
            self.revision.revision_id,
            self.revision.scorer,
            self.revision.scorer_version,
            self.candidate_id,
        )

    @property
    def case_count(self) -> int:
        """Case count is a property of the revision, not of one run."""
        return self.revision.case_count


@dataclass(slots=True, frozen=True)
class ComparabilityCheck:
    comparable: bool
    reasons: tuple[str, ...]


def compare_records(
    baseline: ResultRecord, candidate: ResultRecord
) -> ComparabilityCheck:
    """Decide whether two records may be compared as one experiment.

    Every mismatch is reported, not just the first, so a caller can see the
    whole reason a pair is not comparable.
    """
    reasons: list[str] = []
    if baseline.revision.revision_id != candidate.revision.revision_id:
        reasons.append("eval_revision_mismatch")
    if baseline.revision.scorer != candidate.revision.scorer:
        reasons.append("scorer_mismatch")
    if baseline.revision.scorer_version != candidate.revision.scorer_version:
        reasons.append("scorer_version_mismatch")
    if baseline.candidate_id != candidate.candidate_id:
        reasons.append("candidate_mismatch")
    if baseline.audit.blocks_promotion or candidate.audit.blocks_promotion:
        reasons.append("audit_not_clean")
    return ComparabilityCheck(comparable=not reasons, reasons=tuple(reasons))


@dataclass(slots=True, frozen=True)
class TrendGroup:
    comparability_key: tuple[str, str, str, str]
    labels: tuple[str, ...]
    pass_rates: tuple[float, ...]
    mean_scores: tuple[float, ...]


@dataclass(slots=True, frozen=True)
class TrendReport:
    groups: tuple[TrendGroup, ...]

    @property
    def is_single_trend(self) -> bool:
        return len(self.groups) == 1

    @property
    def mixed(self) -> bool:
        """True when the inputs spanned more than one experiment.

        A caller that receives ``mixed=True`` must not present a single line
        through the data: the segments are not comparable with each other.
        """
        return len(self.groups) > 1

    def reasons(self) -> tuple[str, ...]:
        if not self.mixed:
            return ()
        return (
            f"records span {len(self.groups)} non-comparable groups: "
            + "; ".join(
                f"{g.comparability_key[0][:8]}/{g.comparability_key[1]}@{g.comparability_key[2]}"
                f"/{g.comparability_key[3]}"
                for g in self.groups
            ),
        )


def build_trend(records: Iterable[ResultRecord]) -> TrendReport:
    """Group records into comparable segments (REQ-BENCH-023).

    Deliberately does not return an average across groups. Blending a
    pre-eval-revision number with a post one produces a line that never existed
    in any run, and the only record of the change is a smooth curve.
    """
    grouped: dict[tuple[str, str, str, str], list[ResultRecord]] = {}
    for record in records:
        grouped.setdefault(record.comparability_key, []).append(record)
    groups = tuple(
        TrendGroup(
            comparability_key=key,
            labels=tuple(item.label for item in items),
            pass_rates=tuple(item.pass_rate for item in items),
            mean_scores=tuple(item.mean_score for item in items),
        )
        for key, items in sorted(grouped.items())
    )
    return TrendReport(groups=groups)


@dataclass(slots=True, frozen=True)
class GovernedPromotionEvaluation:
    passed: bool
    failures: tuple[str, ...]
    comparability: ComparabilityCheck
    inner: HeldOutPromotionEvaluation | None

    @property
    def decided_by_governance(self) -> bool:
        """True when the numeric gate never ran because identity/audit failed."""
        return self.inner is None


def evaluate_governed_promotion(
    *,
    baseline_records: Sequence[ResultRecord],
    candidate_records: Sequence[ResultRecord],
    budget: HeldOutPromotionBudget,
    training_case_ids: set[str] | frozenset[str],
    candidate_abstention_accuracy: float,
    candidate_false_success_rate: float,
    contamination: ContaminationReport,
) -> GovernedPromotionEvaluation:
    """Gate a promotion on comparability and audit status first (REQ-BENCH-024).

    The revision identity, scorer identity and audit status are checked before
    any number is compared. Only if all of them pass does the decision delegate
    to :func:`oai2.evals.regression.evaluate_held_out_promotion`, unchanged --
    there is still one promotion gate, not two.

    Corpus contamination is checked in the same pre-comparison pass, because it
    is the same kind of fact: a regression measured across a contaminated
    held-out set is not a regression, and comparing first would lend the
    invalid evidence a number before the refusal arrived.
    """
    failures: list[str] = []

    if not baseline_records or not candidate_records:
        return GovernedPromotionEvaluation(
            passed=False,
            failures=("missing_records",),
            comparability=ComparabilityCheck(False, ("missing_records",)),
            inner=None,
        )

    if not contamination.clean:
        failures.extend(f"contamination:{kind}" for kind in contamination.kinds())

    baseline_revisions = {record.revision.revision_id for record in baseline_records}
    candidate_revisions = {record.revision.revision_id for record in candidate_records}
    if len(baseline_revisions) != 1 or len(candidate_revisions) != 1:
        failures.append("inconsistent_revision_within_side")

    comparability = compare_records(baseline_records[0], candidate_records[0])
    failures.extend(comparability.reasons)

    if failures:
        # Refuse before comparing numbers. A regression computed across two
        # revisions is not a regression.
        return GovernedPromotionEvaluation(
            passed=False,
            failures=tuple(dict.fromkeys(failures)),
            comparability=comparability,
            inner=None,
        )

    inner = evaluate_held_out_promotion(
        baseline_reports=[record_payload(record) for record in baseline_records],
        candidate_reports=[record_payload(record) for record in candidate_records],
        budget=budget,
        training_case_ids=training_case_ids,
        candidate_abstention_accuracy=candidate_abstention_accuracy,
        candidate_false_success_rate=candidate_false_success_rate,
        contamination=contamination,
    )
    return GovernedPromotionEvaluation(
        passed=inner.passed,
        failures=inner.failures,
        comparability=comparability,
        inner=inner,
    )


def record_payload(record: ResultRecord) -> SuiteReportLike:
    """Adapt a :class:`ResultRecord` into the existing gate's report shape.

    The existing gate reads ``.capability``, ``.n_cases``, ``.n_passed`` and
    ``.scores``. Building a tiny stand-in here keeps that gate untouched; a
    ``ResultRecord`` deliberately does not subclass ``SuiteReport`` because its
    job is identity, not scoring.
    """

    @dataclass(slots=True)
    class _Score:
        case_id: str
        capability: str
        score: float
        passed: bool
        matched_pattern: str | None = None
        forbidden_matched: tuple[str, ...] = ()

    @dataclass(slots=True)
    class _Report:
        suite_id: str
        capability: str
        runtime: str
        n_cases: int
        n_passed: int
        mean_score: float
        scores: list

    passed_count = int(round(record.pass_rate * record.case_count))
    score = _Score(
        case_id=f"{record.revision.suite_id}::{record.revision.revision_id}",
        capability=record.revision.capability,
        score=record.mean_score,
        passed=passed_count > 0,
    )
    return _Report(
        suite_id=record.revision.suite_id,
        capability=record.revision.capability,
        runtime=record.candidate_id,
        n_cases=record.case_count,
        n_passed=passed_count,
        mean_score=record.mean_score,
        scores=[score],
    )
