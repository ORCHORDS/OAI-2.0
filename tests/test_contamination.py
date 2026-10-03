"""Coverage for held-out contamination detection (WP-56 / WI-BENCH-001).

The controls here matter more than the happy paths. A contamination detector
that always reports contamination is as useless as one that never does, and
both are easy to ship. So every detection claim is paired with a negative
control that the *clean* side must survive, and the sharpest control
(``test_detector_removed_from_the_pipeline_reports_clean``) proves the
pipeline's verdict actually depends on the detector running.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from oai2.evals import CapabilityCase
from oai2.evals.contamination import (
    CaseFingerprint,
    ContaminationPolicy,
    CorpusEntry,
    OverlapKind,
    check_contamination,
    fingerprint_case,
    fingerprint_cases,
    normalize_tokens,
    select_clean_cases,
    shingle_digests,
)

# -- normalization ----------------------------------------------------------


def test_normalization_folds_case_punctuation_and_spacing() -> None:
    a = normalize_tokens("How many days are there in ONE week?")
    b = normalize_tokens("  how   many, days; are there in one week ")
    assert a == b


def test_normalization_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        normalize_tokens(None)  # type: ignore[arg-type]


def test_short_text_still_produces_a_shingle() -> None:
    # An empty shingle set would make Jaccard undefined and silently skip the
    # case, which is the absence-as-clean failure again.
    assert shingle_digests(("hi",)) != frozenset()
    assert shingle_digests(()) == frozenset()
    with pytest.raises(ValueError):
        shingle_digests(("a", "b"), shingle_size=0)


# -- policy -----------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shingle_size": 0},
        {"shingle_size": -1},
        {"shingle_size": True},
        {"near_overlap_threshold": 0.0},
        {"near_overlap_threshold": 1.5},
        {"min_verifier_tokens": 0},
        {"min_verifier_tokens": -2},
    ],
)
def test_policy_rejects_invalid_thresholds(kwargs) -> None:
    with pytest.raises(ValueError):
        ContaminationPolicy(**kwargs)


# -- fingerprints -----------------------------------------------------------


def _case(case_id: str, prompt: str, patterns: tuple[str, ...] = (r"\b7\b",)) -> CapabilityCase:
    return CapabilityCase(
        case_id=case_id, capability="deterministic_hard", prompt=prompt, expected_patterns=patterns
    )


def test_fingerprint_exposes_no_prompt_or_answer_text() -> None:
    """REQ-BENCH-016: fingerprint metadata must stay public-safe."""
    case = _case("days", "How many days are there in one week?", (r"\b7\b", r"\bseven\b"))
    fingerprint = fingerprint_case(case)
    rendered = json.dumps(asdict(fingerprint), default=str)
    assert "days are there" not in rendered
    assert "seven" not in rendered
    assert case.prompt not in rendered
    assert fingerprint.case_id == "days"
    assert fingerprint.prompt_digest and fingerprint.verifier_digest


def test_fingerprint_requires_identity_fields() -> None:
    with pytest.raises(ValueError):
        fingerprint_case(object())
    with pytest.raises(ValueError):
        fingerprint_case(CapabilityCase(case_id="x", capability="c", prompt=""))
    with pytest.raises(ValueError):
        fingerprint_case(
            CapabilityCase(case_id="", capability="c", prompt="hello there")
        )


def test_verifier_digest_changes_with_the_answer() -> None:
    a = fingerprint_case(_case("c", "How many days are in a week?", (r"\b7\b",)))
    b = fingerprint_case(_case("c", "How many days are in a week?", (r"\b8\b",)))
    assert a.prompt_digest == b.prompt_digest
    assert a.verifier_digest != b.verifier_digest


def test_short_oracle_cannot_accuse_a_corpus() -> None:
    """A one-token oracle matches ordinary prose everywhere.

    ``\\b7\\b`` must never produce a verifier-leak finding, or every report
    against real text becomes a false positive.
    """
    case = _case("days", "How many days are in a week?", (r"\b7\b",))
    fingerprint = fingerprint_case(case)
    assert fingerprint.verifier_shingles == frozenset()
    corpus = [CorpusEntry(entry_id="note", text="we shipped 7 things on day 7 of week 7")]
    report = check_contamination([fingerprint], corpus)
    assert report.clean, report.findings


# -- exact overlap ----------------------------------------------------------


def test_exact_copy_is_detected_despite_a_cosmetic_edit() -> None:
    case = _case("arith", "Compute 17 * 23 - 19. Answer with a number only.", (r"\b372\b",))
    fingerprint = fingerprint_case(case)
    corpus = [
        CorpusEntry(
            entry_id="lesson-1",
            text="COMPUTE  17 * 23 - 19; answer with a NUMBER only.",
            provenance="lesson",
        )
    ]
    report = check_contamination([fingerprint], corpus)
    assert not report.clean
    assert report.findings[0].kind is OverlapKind.EXACT_PROMPT
    assert report.findings[0].score == 1.0
    assert report.quarantined_case_ids == ("arith",)


# -- near overlap -----------------------------------------------------------


def test_near_copy_with_a_light_edit_is_detected() -> None:
    """A lightly-edited copy is the realistic contamination case.

    Measured: 0.875 Jaccard at k=3, and stable at every k from 2 to 5, so this
    is not an artifact of the chosen shingle size.
    """
    original = "Compute 17 * 23 - 19 and report the resulting value as a bare number with no other text."
    edited = "Compute 17 * 23 - 19 and report the resulting value as a bare number, with no other words."
    fingerprint = fingerprint_case(_case("arith", original, (r"\b372\b",)))
    corpus = [CorpusEntry(entry_id="synthetic-1", text=edited, provenance="synthetic")]
    report = check_contamination([fingerprint], corpus)
    assert not report.clean
    finding = report.findings[0]
    assert finding.kind is OverlapKind.NEAR_PROMPT
    assert finding.score >= ContaminationPolicy().near_overlap_threshold


def test_heavy_rewrite_of_a_short_prompt_is_not_detected() -> None:
    """The honest limit, pinned so it cannot be quietly forgotten.

    A 13-token prompt whose wording is substantially replaced scores 0.211 by
    trigram Jaccard, below the 0.6 default. Trigram overlap cannot recover a
    paraphrase that replaces half the tokens, and a threshold low enough to
    catch it would also flag the unrelated controls. This is a real gap in
    trigram detection, not a passing test.
    """
    original = "Compute 17 * 23 - 19 and report the resulting value as a bare number."
    rewritten = "Multiply 17 by 23, subtract 19, then give just the resulting figure."
    fingerprint = fingerprint_case(_case("arith", original, (r"\b372\b",)))
    corpus = [CorpusEntry(entry_id="synthetic-1", text=rewritten, provenance="synthetic")]
    report = check_contamination([fingerprint], corpus)
    assert report.clean, "if this ever passes, the module docstring's stated limit is stale"


def test_unrelated_prompt_is_not_flagged() -> None:
    """Negative control for the near detector."""
    fingerprint = fingerprint_case(
        _case("arith", "Compute 17 * 23 - 19. Answer with a number only.", (r"\b372\b",))
    )
    corpus = [
        CorpusEntry(
            entry_id="unrelated",
            text="Refactor the connection pool to reuse sockets across retries.",
            provenance="lesson",
        ),
        CorpusEntry(
            entry_id="same-topic",
            text="A string is immutable, so concatenating in a loop is quadratic.",
            provenance="lesson",
        ),
    ]
    report = check_contamination([fingerprint], corpus)
    assert report.clean, report.findings


def test_threshold_is_load_bearing() -> None:
    """The same pair flips verdict with the threshold, so the policy is real."""
    original = "Compute 17 * 23 - 19 and report the resulting value as a bare number with no other text."
    edited = "Compute 17 * 23 - 19 and report the resulting value as a bare number, with no other words."
    fingerprint = fingerprint_case(_case("arith", original, (r"\b372\b",)))
    corpus = [CorpusEntry(entry_id="p", text=edited)]
    loose = check_contamination([fingerprint], corpus, policy=ContaminationPolicy(near_overlap_threshold=0.1))
    strict = check_contamination([fingerprint], corpus, policy=ContaminationPolicy(near_overlap_threshold=0.99))
    assert not loose.clean
    assert strict.clean


# -- verifier leak (REQ-BENCH-013) -----------------------------------------


def test_verifier_leak_is_detected_even_when_the_prompt_differs() -> None:
    """The answer key in visible text is contamination on its own.

    This is the shape of the defect fixed in 6f7f9fe, where a field named
    after the ground-truth condition reached the candidate.
    """
    case = _case(
        "reminder",
        "Summarise the release notes for the upcoming version.",
        (r"the staging deploy was rolled back after the migration lock",),
    )
    fingerprint = fingerprint_case(case)
    corpus = [
        CorpusEntry(
            entry_id="visible-note",
            text=(
                "Remember: the staging deploy was rolled back after the migration "
                "lock, so do not promote that build."
            ),
            provenance="doc",
        )
    ]
    report = check_contamination([fingerprint], corpus)
    assert not report.clean
    assert OverlapKind.VERIFIER_LEAK in {f.kind for f in report.findings}


def test_verifier_check_can_be_disabled() -> None:
    case = _case("r", "Summarise the release notes for the upcoming version.",
                 (r"the staging deploy was rolled back after the migration lock",))
    fingerprint = fingerprint_case(case)
    corpus = [CorpusEntry(entry_id="n", text=(
        "Remember: the staging deploy was rolled back after the migration lock today."
    ))]
    policy = ContaminationPolicy(check_verifier_leak=False, near_overlap_threshold=0.99)
    assert check_contamination([fingerprint], corpus, policy=policy).clean


# -- absence must not read as clean ----------------------------------------


def test_empty_corpus_is_refused_rather_than_reported_clean() -> None:
    fingerprint = fingerprint_case(_case("c", "Compute 17 * 23 - 19. Answer with a number."))
    with pytest.raises(ValueError, match="corpus is empty"):
        check_contamination([fingerprint], [])
    opted_in = check_contamination([fingerprint], [], policy=ContaminationPolicy(allow_empty_corpus=True))
    assert opted_in.clean and opted_in.checked_entries == 0


def test_report_counts_what_it_actually_checked() -> None:
    cases = [
        _case("a", "Compute 17 * 23 - 19. Answer with a number only.", (r"\b372\b",)),
        _case("b", "How many times does the letter a appear in banana?", (r"\b3\b",)),
    ]
    corpus = [CorpusEntry(entry_id="e1", text="a lesson about caching"), CorpusEntry(entry_id="e2", text="")]
    report = check_contamination(fingerprint_cases(cases), corpus)
    assert report.checked_cases == 2
    assert report.checked_entries == 2


# -- quarantine (REQ-BENCH-015) --------------------------------------------


def test_quarantine_excludes_and_replaces() -> None:
    burned = _case("burned", "Compute 17 * 23 - 19. Answer with a number only.", (r"\b372\b",))
    clean = _case("clean", "How many days are there in one week? Answer with a number.", (r"\b7\b",))
    reserve = _case("reserve", "Reverse the string 'orchords'. Output only the reversed string.", (r"sdrohcro",))
    corpus = [CorpusEntry(entry_id="leak", text=burned.prompt, provenance="lesson")]

    report = check_contamination(fingerprint_cases([burned, clean]), corpus)
    survivors, quarantined, replacements = select_clean_cases(
        [burned, clean], report, reserve=[reserve]
    )
    assert [c.case_id for c in quarantined] == ["burned"]
    assert [c.case_id for c in survivors] == ["clean"]
    assert [c.case_id for c in replacements] == ["reserve"]


def test_replacement_pool_is_not_unlimited() -> None:
    cases = [
        _case("a", "Compute 17 * 23 - 19. Answer with a number only.", (r"\b372\b",)),
        _case("b", "What is 98765 divided by 31? Answer with a number only.", (r"\b30\b",)),
    ]
    corpus = [CorpusEntry(entry_id="leak", text=cases[0].prompt)]
    report = check_contamination(fingerprint_cases(cases), corpus)
    survivors, quarantined, replacements = select_clean_cases(
        cases, report, reserve=[_case("r1", "Capital of Australia? Name only.", (r"Canberra",))]
    )
    assert len(quarantined) == 1
    assert len(replacements) == 1, "must not invent replacements beyond the reserve"


def test_a_contaminated_reserve_case_is_never_promoted() -> None:
    burned = _case("burned", "Compute 17 * 23 - 19. Answer with a number only.", (r"\b372\b",))
    also_burned = _case("also", "Compute 17 * 23 - 19. Answer with a number only.", (r"\b372\b",))
    clean = _case("clean", "How many days are there in one week? Answer with a number.", (r"\b7\b",))
    corpus = [CorpusEntry(entry_id="leak", text=burned.prompt)]
    report = check_contamination(fingerprint_cases([burned, clean, also_burned]), corpus)
    assert set(report.quarantined_case_ids) == {"burned", "also"}
    _survivors, _quarantined, replacements = select_clean_cases(
        [burned, clean, also_burned], report, reserve=[also_burned]
    )
    assert replacements == ()


# -- negative control on the pipeline itself -------------------------------


def test_detector_removed_from_the_pipeline_reports_clean() -> None:
    """Proves the verdict depends on the detector actually running.

    With every case replaced by an unrelated fingerprint, the same corpus
    produces a clean report. If it did not, the clean results elsewhere in
    this module would be meaningless.
    """
    corpus = [CorpusEntry(entry_id="e", text="Compute 17 * 23 - 19. Answer with a number only.")]
    real = fingerprint_case(
        _case("arith", "Compute 17 * 23 - 19. Answer with a number only.", (r"\b372\b",))
    )
    assert not check_contamination([real], corpus).clean

    neutral: CaseFingerprint = fingerprint_case(
        _case("other", "Explain how a write-ahead log protects durability.", (r"\bcommit\b",))
    )
    assert check_contamination([neutral], corpus).clean


def test_findings_never_carry_corpus_text() -> None:
    secret_phrase = "roll back the canary build before promoting"
    case = _case("r", "What should we do with the canary?", (secret_phrase,))
    corpus = [CorpusEntry(entry_id="e", text=f"Note: {secret_phrase} immediately.", provenance="doc")]
    report = check_contamination([fingerprint_case(case)], corpus)
    assert report.findings
    assert secret_phrase not in json.dumps([asdict(f) for f in report.findings])
