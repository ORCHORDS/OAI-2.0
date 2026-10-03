"""Audit this repository's own held-out data for contamination (WP-56 / WI-BENCH-001).

Why this exists
---------------

A contamination detector that is only ever run against fixtures proves it can
recognise fixtures. This script points it at the actual repository: the
committed evidence artifacts, the registered eval suites, and the agent
learning/seeding text. Whatever it finds is a real property of this tree.

The specific thing worth asking is whether anything committed under
``evidence/`` or ``docs/`` contains a held-out prompt or its answer key. A
benchmark that ships its own cases *and* their expected values is not held
out from anyone who can read the repository, including the data pipeline.

Usage
-----
    .venv/bin/python scripts/contamination_audit.py
    .venv/bin/python scripts/contamination_audit.py --json out.json

Exit status is 0 when the audit ran, regardless of findings: a finding is a
result to report, not a tool failure. Use ``--fail-on-findings`` in a gate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oai2.evals.contamination import (  # noqa: E402
    ContaminationPolicy,
    CorpusEntry,
    check_contamination,
    fingerprint_cases,
    select_clean_cases,
)

REPO = Path(__file__).resolve().parents[1]

#: Directories that are plausibly model-visible or used to build training /
#: tuning data. Anything here is "the corpus" side of the check.
CORPUS_DIRS = ("evidence", "docs", "evals")

#: Binary/oversized files are skipped: they are not text a model would be
#: trained on, and reading them would only slow the audit.
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".gguf", ".bin", ".pyc", ".pdf", ".zip"}
MAX_BYTES = 2_000_000


def harness_identity() -> dict[str, str]:
    info = {"commit": "unavailable:OSError", "branch": "unknown", "dirty": "unknown"}
    try:
        head = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(  # noqa: S603
            ["git", "status", "--porcelain"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        branch = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        info = {"commit": head, "branch": branch, "dirty": "true" if status else "false"}
    except OSError, subprocess.SubprocessError:
        pass
    return info


def iter_corpus() -> Iterator[CorpusEntry]:
    """Every committed text file under the corpus directories.

    Uses ``git ls-files`` rather than a filesystem walk so the audit sees
    exactly what is committed. A walk would include build output and local
    scratch, and would let the verdict depend on whether a virtualenv exists
    -- the same defect the public-safety guard in this repository already had.
    """
    try:
        listing = subprocess.run(  # noqa: S603
            ["git", "ls-files", *CORPUS_DIRS],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"could not list corpus files: {exc}") from exc

    for rel in listing:
        path = REPO / rel
        if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
            continue
        try:
            if path.stat().st_size > MAX_BYTES:
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # Unreadable is skipped rather than treated as empty; an entry we
            # could not read must not be reported as a clean non-match.
            continue
        if text.strip():
            yield CorpusEntry(
                entry_id=rel, text=text, provenance=rel.split("/", 1)[0] or "repo"
            )


def registered_suites() -> list[tuple[str, object]]:
    """Every :class:`CapabilitySuite` the evals package registers.

    ``_BUILTIN_SUITES`` maps a name to a *factory*, not to a suite, so each
    entry has to be called to obtain the cases.
    """
    from oai2.evals import _BUILTIN_SUITES

    suites: list[tuple[str, object]] = []
    for suite_id, factory in sorted(_BUILTIN_SUITES.items()):
        suite = factory()
        if suite.cases:
            suites.append((suite_id, suite))
    return suites


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--fail-on-findings", action="store_true")
    args = parser.parse_args()

    policy = ContaminationPolicy()
    corpus = list(iter_corpus())
    suites = registered_suites()

    print(f"repo        : {REPO}")
    print(f"harness     : {harness_identity()}")
    print(f"policy      : {policy}")
    print(f"corpus      : {len(corpus)} committed text files from {CORPUS_DIRS}")
    print(f"suites      : {', '.join(suite_id for suite_id, _ in suites) or 'none'}")
    print()

    results = []
    total_findings = 0
    for suite_id, suite in suites:
        fingerprints = fingerprint_cases(suite.cases, policy=policy)
        report = check_contamination(fingerprints, corpus, policy=policy)
        survivors, quarantined, replacements = select_clean_cases(
            suite.cases, report
        )
        total_findings += len(report.findings)
        print(f"--- suite {suite_id}: {len(suite.cases)} held-out cases ---")
        print(f"    findings        : {len(report.findings)}")
        print(f"    kinds           : {report.kinds() or '(none)'}")
        print(f"    quarantined     : {[c.case_id for c in quarantined] or '(none)'}")
        print(f"    survivors       : {len(survivors)}")
        print(f"    replacements    : {len(replacements)} (no reserve pool configured)")
        for finding in report.findings:
            print(
                f"      - {finding.case_id} <- {finding.entry_id} "
                f"[{finding.kind.value} score={finding.score}]"
            )
        results.append(
            {
                "suite_id": suite_id,
                "case_count": len(suite.cases),
                "findings": [asdict(f) for f in report.findings],
                "kinds": list(report.kinds()),
                "quarantined_case_ids": [c.case_id for c in quarantined],
                "survivor_count": len(survivors),
                "replacement_count": len(replacements),
            }
        )
        print()

    artifact = {
        "issue": 171,
        "harness": harness_identity(),
        "policy": asdict(policy),
        "corpus_files": len(corpus),
        "corpus_dirs": list(CORPUS_DIRS),
        "suites": results,
        "total_findings": total_findings,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(artifact, indent=2))
        print(f"artifact: {args.json}")

    print(
        f"TOTAL: {total_findings} finding(s) across {len(suites)} suite(s) "
        f"against {len(corpus)} committed files"
    )
    if total_findings:
        print(
            "\nA finding is not automatically a bug: a held-out case quoted in\n"
            "documentation is not the same as one fed into tuning. What matters\n"
            "is that the overlap is now visible and can be adjudicated instead\n"
            "of being discovered after a score moves."
        )
    if args.fail_on_findings and total_findings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
