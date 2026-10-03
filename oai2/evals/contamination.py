"""Held-out benchmark fingerprints and contamination detection (WP-56 / WI-BENCH-001).

Why this module exists
----------------------

A held-out evaluation set is only held out while its cases stay out of
everything that could teach the model. Nothing in this repository enforced
that. :class:`~oai2.evals.CapabilityCase` carries a prompt, the regex patterns
that decide whether the answer is right, and a ``case_id``; :mod:`oai2.agents.learning`
converts agent runs into sanitized text that can be stored as a lesson. Between
them there was no check that a lesson, a committed artifact or a generated
synthetic example did not simply *contain* a held-out case.

The first version of this issue's acceptance criterion is that deliberate
exact and near contamination is detected. The second is that contaminated cases
are excluded or replaced according to an explicit policy. This module owns
both, and owns the fingerprints they are computed from.

Design points
-------------

- **Fingerprints are digests, never text** (REQ-BENCH-016). A
  :class:`CaseFingerprint` can be written to a log, an artifact or a public
  report without exposing the case. It carries ``prompt_digest``,
  ``verifier_digest`` and a set of shingle digests, and the raw strings are not
  reachable from it.
- **Verifier content is fingerprinted separately** (REQ-BENCH-013). The
  patterns that decide correctness are ground truth, so they get their own
  digest and their own overlap kind, distinct from prompt overlap.
- **Near overlap is shingle *containment*, not Jaccard** (REQ-BENCH-012), on
  normalized tokens, so case/whitespace/punctuation edits cannot hide a copy.
  Containment is the correct direction for a needle-in-haystack question; see
  :func:`_overlap` for the measurement that forced the change.
- **Thresholds are an explicit policy object** (REQ-BENCH-014), not constants
  buried in a comparison, so a result can always be re-derived.
- **An unreadable corpus is not a clean corpus.** A check that returns "no
  contamination" because it had nothing to compare against is the same
  absence-as-clean failure this repository has been closing elsewhere, so an
  empty corpus is an error unless the caller opts out explicitly.

Status: IMPLEMENTED — unit-pinned in ``tests/test_contamination.py`` and run
against real repository data by ``scripts/contamination_audit.py``.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "DEFAULT_MIN_VERIFIER_TOKENS",
    "DEFAULT_MIN_SHARED_SHINGLES",
    "DEFAULT_SHINGLE_SIZE",
    "CaseFingerprint",
    "ContaminationFinding",
    "ContaminationPolicy",
    "ContaminationReport",
    "CorpusEntry",
    "OverlapKind",
    "check_contamination",
    "fingerprint_case",
    "fingerprint_cases",
    "normalize_tokens",
    "select_clean_cases",
    "shingle_digests",
]

#: Tokens per shingle.
#:
#: Measured on this repository's own short eval prompts (13-17 tokens): a
#: lightly-edited copy shares 0.86-0.92 of its trigrams with the original,
#: while an unrelated prompt and a same-topic-but-different prompt share 0.00.
#: Three is the smallest k that still separates the unrelated controls.
#:
#: The honest limit, pinned by a test: a *rewritten* short prompt is not
#: detected. Trigram overlap cannot recover a paraphrase that replaces half
#: the tokens of a 13-token prompt, and pretending otherwise would make the
#: threshold meaningless.
DEFAULT_SHINGLE_SIZE = 3

#: Shared shingles required in addition to the ratio. A three-token prompt has
#: a single trigram, so a ratio alone would let one coincidental trigram
#: report 100% overlap. The count makes a short case fail closed.
DEFAULT_MIN_SHARED_SHINGLES = 2

#: Verifier content shorter than this is too weak to accuse a corpus of a leak.
#: The suite legitimately contains single-token oracles such as ``\b7\b``; a
#: substring search for those matches ordinary prose everywhere and would make
#: every report a false positive.
DEFAULT_MIN_VERIFIER_TOKENS = 3

#: Characters treated as token separators after normalization. Kept as an
#: explicit class so underscore/identifier-style text still splits sensibly.
_TOKEN_SPLIT = re.compile(r"[^0-9a-z]+")

#: Regex syntax stripped from a verifier pattern before it is tokenized, so
#: ``\b7\b`` and ``answer 372`` are compared as words rather than as escapes.
_REGEX_NOISE = re.compile(r"\\[bswdWDSBZ]|[\^\$\.\*\+\?\(\)\[\]\{\}\|]")


def normalize_tokens(text: str) -> tuple[str, ...]:
    """Casefold, split on non-alphanumerics and drop empties.

    Two prompts that differ only in case, punctuation or spacing normalize to
    the same token sequence, which is what makes an exact copy detectable
    through a cosmetic edit.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return tuple(token for token in _TOKEN_SPLIT.split(text.casefold()) if token)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def shingle_digests(
    tokens: Sequence[str], *, shingle_size: int = DEFAULT_SHINGLE_SIZE
) -> frozenset[str]:
    """Hashed k-gram digests over ``tokens``.

    A text shorter than ``shingle_size`` yields a single digest over what it
    has, so a very short case is still fingerprinted rather than silently
    producing an empty set (an empty set would make Jaccard undefined and
    quietly skip the case).
    """
    if shingle_size <= 0:
        raise ValueError("shingle_size must be positive")
    if not tokens:
        return frozenset()
    if len(tokens) < shingle_size:
        return frozenset({_digest(" ".join(tokens))})
    return frozenset(
        _digest(" ".join(tokens[index : index + shingle_size]))
        for index in range(len(tokens) - shingle_size + 1)
    )


def _overlap(case: frozenset[str], entry: frozenset[str]) -> tuple[float, int]:
    """Fraction of the *case's* shingles that also occur in the entry.

    Containment, not Jaccard. The question is "is this held-out case present
    in this document", and the two sides are wildly different sizes: a 13-token
    case inside a 7 KB artifact. Jaccard divides by the union, so the
    document's unrelated content dilutes the score toward zero and a verbatim
    copy reads as clean.

    This was not a theoretical concern. Against this repository's own
    ``evidence/accuracy/deterministic_hard_12_a133744.json``, Jaccard reported
    0 findings for all 12 cases while containment reported 1.000 for all 12 --
    the file contains every held-out prompt verbatim. A detector that scores a
    verbatim copy as clean is worse than no detector, because it is trusted.

    Returns ``(ratio, shared_count)``. The count matters for short cases: a
    three-token prompt is one trigram, so a ratio with no floor would let a
    single coincidental match report 100%.
    """
    if not case or not entry:
        return 0.0, 0
    shared = len(case & entry)
    if shared == 0:
        return 0.0, 0
    return shared / len(case), shared


class OverlapKind(StrEnum):
    """How a corpus entry overlaps a held-out case."""

    EXACT_PROMPT = "exact_prompt"
    NEAR_PROMPT = "near_prompt"
    VERIFIER_LEAK = "verifier_leak"


@dataclass(slots=True, frozen=True)
class CaseFingerprint:
    """Public-safe identity for one held-out case.

    Carries no prompt text and no answer text (REQ-BENCH-016). ``case_id`` is
    retained because a quarantine decision has to name what it removed.
    """

    case_id: str
    prompt_digest: str
    verifier_digest: str
    shingle_digests: frozenset[str]
    verifier_shingles: frozenset[str]
    prompt_token_count: int

    @property
    def is_checkable(self) -> bool:
        """False when the case has no prompt tokens to compare."""
        return self.prompt_token_count > 0


@dataclass(slots=True, frozen=True)
class CorpusEntry:
    """One model-visible or training-side item to check against.

    ``provenance`` is free-form but should name where the text came from
    (``"lesson"``, ``"artifact"``, ``"doc"``, ``"synthetic"``) so a finding can
    be traced to a source without the report carrying the text itself.
    """

    entry_id: str
    text: str
    provenance: str = ""

    def __post_init__(self) -> None:
        if not self.entry_id or self.entry_id != self.entry_id.strip():
            raise ValueError("entry_id must be a non-empty normalized string")


@dataclass(slots=True, frozen=True)
class ContaminationPolicy:
    """Explicit thresholds. REQ-BENCH-014: the policy must be visible."""

    shingle_size: int = DEFAULT_SHINGLE_SIZE
    near_overlap_threshold: float = 0.6
    min_shared_shingles: int = DEFAULT_MIN_SHARED_SHINGLES
    min_verifier_tokens: int = DEFAULT_MIN_VERIFIER_TOKENS
    check_verifier_leak: bool = True
    allow_empty_corpus: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.shingle_size, int) or isinstance(self.shingle_size, bool):
            raise ValueError("shingle_size must be an integer")
        if self.shingle_size <= 0:
            raise ValueError("shingle_size must be positive")
        if not math.isfinite(self.near_overlap_threshold):
            raise ValueError("near_overlap_threshold must be finite")
        if not 0.0 < self.near_overlap_threshold <= 1.0:
            raise ValueError("near_overlap_threshold must be in (0, 1]")
        for name in ("min_shared_shingles", "min_verifier_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(slots=True, frozen=True)
class ContaminationFinding:
    """One detected overlap. Carries identities and a score, never text."""

    case_id: str
    entry_id: str
    kind: OverlapKind
    score: float
    provenance: str


@dataclass(slots=True, frozen=True)
class ContaminationReport:
    policy: ContaminationPolicy
    checked_cases: int
    checked_entries: int
    findings: tuple[ContaminationFinding, ...]

    @property
    def clean(self) -> bool:
        return not self.findings

    @property
    def quarantined_case_ids(self) -> tuple[str, ...]:
        """Cases to exclude or replace (REQ-BENCH-015). Sorted for determinism."""
        return tuple(sorted({finding.case_id for finding in self.findings}))

    def findings_for(self, case_id: str) -> tuple[ContaminationFinding, ...]:
        return tuple(f for f in self.findings if f.case_id == case_id)

    def kinds(self) -> tuple[str, ...]:
        return tuple(sorted({finding.kind.value for finding in self.findings}))


def fingerprint_case(
    case: object,
    *,
    policy: ContaminationPolicy | None = None,
) -> CaseFingerprint:
    """Fingerprint a :class:`~oai2.evals.CapabilityCase`-shaped object.

    Reads ``case_id``, ``prompt`` and the verifier fields
    (``expected_patterns``/``forbidden_patterns``/``must_contain_action_token``)
    by name, so it works on the existing case type without subclassing it or
    adding fields to it.
    """
    active = policy or ContaminationPolicy()
    case_id = getattr(case, "case_id", None)
    prompt = getattr(case, "prompt", None)
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case must expose a non-empty case_id")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("case must expose a non-empty prompt")

    verifier_parts: list[str] = []
    for field in ("expected_patterns", "forbidden_patterns"):
        value = getattr(case, field, ()) or ()
        if isinstance(value, str):
            value = (value,)
        verifier_parts.extend(str(item) for item in value)
    action_token = getattr(case, "must_contain_action_token", None)
    if isinstance(action_token, str) and action_token:
        verifier_parts.append(action_token)

    verifier_text = " ".join(verifier_parts)
    verifier_tokens = normalize_tokens(_REGEX_NOISE.sub(" ", verifier_text))

    return CaseFingerprint(
        case_id=case_id,
        prompt_digest=_digest(" ".join(normalize_tokens(prompt))),
        verifier_digest=_digest(" ".join(verifier_tokens)),
        shingle_digests=shingle_digests(
            normalize_tokens(prompt), shingle_size=active.shingle_size
        ),
        # A one-or-two-token oracle is not strong enough to accuse a corpus.
        verifier_shingles=(
            shingle_digests(verifier_tokens, shingle_size=active.shingle_size)
            if len(verifier_tokens) >= active.min_verifier_tokens
            else frozenset()
        ),
        prompt_token_count=len(normalize_tokens(prompt)),
    )


def fingerprint_cases(
    cases: Iterable[object],
    *,
    policy: ContaminationPolicy | None = None,
) -> tuple[CaseFingerprint, ...]:
    return tuple(fingerprint_case(case, policy=policy) for case in cases)


def check_contamination(
    fingerprints: Sequence[CaseFingerprint],
    corpus: Sequence[CorpusEntry],
    *,
    policy: ContaminationPolicy | None = None,
) -> ContaminationReport:
    """Check held-out fingerprints against model-visible/training text.

    REQ-BENCH-012 checks exact *and* near overlap; REQ-BENCH-013 checks that
    verifier content has not leaked into visible text. An exact prompt digest
    match is reported as :attr:`OverlapKind.EXACT_PROMPT` and always wins over
    a near match for the same pair, so a caller reading the report does not
    have to infer severity from a score.
    """
    active = policy or ContaminationPolicy()
    if not corpus and not active.allow_empty_corpus:
        # "No contamination" is not a conclusion you can reach by comparing
        # against nothing. Reporting clean here would be an absence presented
        # as a clean measurement.
        raise ValueError(
            "corpus is empty; pass allow_empty_corpus=True to assert there is "
            "genuinely nothing to compare against"
        )

    entry_tokens = {
        entry.entry_id: (normalize_tokens(entry.text), entry) for entry in corpus
    }
    findings: list[ContaminationFinding] = []

    for fingerprint in fingerprints:
        if not fingerprint.is_checkable:
            continue
        for entry_id, (tokens, entry) in entry_tokens.items():
            if not tokens:
                continue
            entry_shingles = shingle_digests(tokens, shingle_size=active.shingle_size)

            if fingerprint.prompt_digest == _digest(" ".join(tokens)):
                findings.append(
                    ContaminationFinding(
                        case_id=fingerprint.case_id,
                        entry_id=entry_id,
                        kind=OverlapKind.EXACT_PROMPT,
                        score=1.0,
                        provenance=entry.provenance,
                    )
                )
                continue

            score, shared = _overlap(fingerprint.shingle_digests, entry_shingles)
            if score >= active.near_overlap_threshold and shared >= active.min_shared_shingles:
                findings.append(
                    ContaminationFinding(
                        case_id=fingerprint.case_id,
                        entry_id=entry_id,
                        kind=OverlapKind.NEAR_PROMPT,
                        score=round(score, 4),
                        provenance=entry.provenance,
                    )
                )

            if (
                active.check_verifier_leak
                and fingerprint.verifier_shingles
                and fingerprint.verifier_shingles <= entry_shingles
            ):
                # Containment, not Jaccard: the verifier's shingles are a
                # contiguous signature and a much larger corpus should not
                # dilute the score just by containing unrelated text.
                findings.append(
                    ContaminationFinding(
                        case_id=fingerprint.case_id,
                        entry_id=entry_id,
                        kind=OverlapKind.VERIFIER_LEAK,
                        score=1.0,
                        provenance=entry.provenance,
                    )
                )

    findings.sort(key=lambda f: (f.case_id, f.entry_id, f.kind.value))
    return ContaminationReport(
        policy=active,
        checked_cases=len(fingerprints),
        checked_entries=len(corpus),
        findings=tuple(findings),
    )


def select_clean_cases(
    cases: Sequence[object],
    report: ContaminationReport,
    *,
    reserve: Sequence[object] = (),
) -> tuple[tuple[object, ...], tuple[object, ...], tuple[object, ...]]:
    """Split cases into survivors, quarantined, and replacements (REQ-BENCH-015).

    Returns ``(survivors, quarantined, replacements)``. A quarantined case is
    replaced from ``reserve`` in order when one is available; a reserve case is
    itself checked against the same report, so a contaminated reserve entry is
    never promoted into a held-out set.
    """
    quarantined_ids = set(report.quarantined_case_ids)
    survivors = tuple(case for case in cases if getattr(case, "case_id", None) not in quarantined_ids)
    quarantined = tuple(
        case for case in cases if getattr(case, "case_id", None) in quarantined_ids
    )
    replacements: list[object] = []
    for candidate in reserve:
        if len(replacements) >= len(quarantined):
            break
        if getattr(candidate, "case_id", None) in quarantined_ids:
            continue
        if getattr(candidate, "case_id", None) in {
            getattr(case, "case_id", None) for case in survivors
        }:
            continue
        replacements.append(candidate)
    return survivors, quarantined, tuple(replacements)
