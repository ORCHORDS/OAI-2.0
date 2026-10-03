"""Coverage for the repository contamination audit (WP-56 / WI-BENCH-001).

The audit's value is that it runs against the real tree, so its own
enumeration behaviour is load-bearing. A filesystem walk would scan
``.venv`` and build output, which is both slow and a way for the verdict to
depend on local machine state -- the defect the public-safety guard in this
repository had already been bitten by.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_audit():
    spec = importlib.util.spec_from_file_location(
        "contamination_audit_mod", REPO / "scripts" / "contamination_audit.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["contamination_audit_mod"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules["contamination_audit_mod"]
    return module


audit = _load_audit()


def test_corpus_enumeration_uses_committed_files_only() -> None:
    """It must see exactly what is committed, not what is on disk."""
    source = (REPO / "scripts" / "contamination_audit.py").read_text()
    assert "git" in source and "ls-files" in source
    assert "rglob" not in source, "a filesystem walk would include .venv and build output"


def test_corpus_entries_are_well_formed() -> None:
    entries = list(audit.iter_corpus())
    assert entries, "the repository must have some committed corpus text"
    for entry in entries[:50]:
        assert entry.entry_id and entry.entry_id == entry.entry_id.strip()
        assert entry.text.strip()
        assert entry.provenance


def test_binary_and_oversized_files_are_skipped() -> None:
    suffixes = {Path(e.entry_id).suffix.lower() for e in audit.iter_corpus()}
    assert ".gguf" not in suffixes
    assert ".pyc" not in suffixes
    assert ".png" not in suffixes


def test_registered_suites_are_resolved_from_factories() -> None:
    """``_BUILTIN_SUITES`` holds factories; calling them wrong yields no cases."""
    suites = audit.registered_suites()
    assert suites
    ids = [suite_id for suite_id, _ in suites]
    assert "deterministic_hard" in ids
    for _suite_id, suite in suites:
        assert getattr(suite, "cases", None), "a suite with no cases should not be audited"


def test_audit_detects_a_planted_verbatim_copy(tmp_path: Path) -> None:
    """End-to-end control against the real detector and the real suite."""
    from oai2.evals.contamination import ContaminationPolicy, CorpusEntry, check_contamination
    from oai2.evals.deterministic_hard import deterministic_hard_suite

    suite = deterministic_hard_suite()
    policy = ContaminationPolicy()
    fingerprints = audit.fingerprint_cases(suite.cases, policy=policy)

    # Negative control first: the suite against itself is not contamination
    # unless the text is actually in a *separate* entry.
    clean = check_contamination(fingerprints, [CorpusEntry(entry_id="x", text="unrelated notes")])
    assert clean.clean

    planted = CorpusEntry(
        entry_id="planted.json", text=json_text_of(suite), provenance="planted"
    )
    dirty = check_contamination(fingerprints, [planted])
    assert not dirty.clean
    assert len(dirty.quarantined_case_ids) == 12


def json_text_of(suite) -> str:
    """Render a suite the way a committed artifact would."""
    import json

    return json.dumps(
        {
            "cases": [
                {"case_id": c.case_id, "prompt": c.prompt} for c in suite.cases
            ]
        },
        indent=2,
    )


def test_real_repository_artifact_is_flagged() -> None:
    """The finding this audit exists to produce, pinned against the real file.

    If someone redacts or removes the artifact, this test should fail and say
    so, rather than the audit quietly reporting a clean repository.
    """
    from oai2.evals.contamination import ContaminationPolicy, CorpusEntry, check_contamination
    from oai2.evals.deterministic_hard import deterministic_hard_suite

    artifact = REPO / "evidence" / "accuracy" / "deterministic_hard_12_a133744.json"
    if not artifact.exists():
        pytest.skip("accuracy artifact not present in this checkout")
    suite = deterministic_hard_suite()
    report = check_contamination(
        audit.fingerprint_cases(suite.cases, policy=ContaminationPolicy()),
        [CorpusEntry(entry_id=artifact.name, text=artifact.read_text(), provenance="evidence")],
    )
    assert not report.clean, (
        "the committed accuracy artifact no longer overlaps the held-out suite; "
        "if that is intentional, update this test and the #171 evidence"
    )
    assert set(report.quarantined_case_ids) == {c.case_id for c in suite.cases}
