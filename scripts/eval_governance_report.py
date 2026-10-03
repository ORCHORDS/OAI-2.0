"""Report eval revision identity and audit status for the real suites (WP-56 / WI-BENCH-002).

Answers a question that unit tests cannot: for the suites this repository
actually registers, what is the content-addressed revision, is the
contamination audit clean, and are the results currently in evidence even
comparable with it?

Everything here is deterministic and local. No runner, no network, no model.

Usage
-----
    .venv/bin/python scripts/eval_governance_report.py
    .venv/bin/python scripts/eval_governance_report.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oai2.evals.contamination import ContaminationPolicy, CorpusEntry  # noqa: E402
from oai2.evals.governance import (  # noqa: E402
    ResultRecord,
    audit_contamination,
    build_trend,
    eval_revision,
)
from oai2.verification.reproducibility import source_identity  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CORPUS_DIRS = ("evidence", "docs", "evals")
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".gguf", ".bin", ".pyc", ".pdf", ".zip"}
MAX_BYTES = 2_000_000

#: Scorer versions are a governance declaration, not a guess. A suite scored by
#: ``regex_or`` and a suite scored by a model judge are different experiments,
#: so the version travels with the revision. Bumping either string is what
#: invalidates comparability, which is the point (REQ-BENCH-025).
SCORER_VERSION = "regex_or@2026-10-04"


def harness_identity() -> dict[str, str]:
    """Delegate to the single provenance owner in ``oai2.verification``.

    The previous private copy anchored git at the *current working directory*.
    Git walks up to find a repository, so a subdirectory was fine, but running
    this script from outside the tree recorded ``commit:
    "unavailable:OSError"`` for a healthy repository. The shared owner anchors
    at the repository root instead.
    """
    return source_identity(cwd=str(REPO)).as_dict()


def iter_corpus() -> list[CorpusEntry]:
    listing = subprocess.run(  # noqa: S603
        ["git", "ls-files", *CORPUS_DIRS],  # noqa: S607
        capture_output=True, text=True, check=True, cwd=REPO,
    ).stdout.splitlines()
    entries: list[CorpusEntry] = []
    for rel in listing:
        path = REPO / rel
        if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
            continue
        try:
            if path.stat().st_size > MAX_BYTES:
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if text.strip():
            entries.append(CorpusEntry(entry_id=rel, text=text, provenance=rel.split("/", 1)[0]))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    from oai2.evals import _BUILTIN_SUITES

    policy = ContaminationPolicy()
    corpus = iter_corpus()

    print(f"repo       : {REPO}")
    print(f"harness    : {harness_identity()}")
    print(f"scorer     : {SCORER_VERSION}")
    print(f"corpus     : {len(corpus)} committed files from {CORPUS_DIRS}")
    print()

    rows = []
    records: list[ResultRecord] = []
    print(f"{'suite':22} {'revision':34} {'cases':>5} {'audit':>9} {'findings':>8}")
    print("-" * 84)
    for suite_id, factory in sorted(_BUILTIN_SUITES.items()):
        suite = factory()
        if not suite.cases:
            continue
        revision = eval_revision(suite, scorer_version=SCORER_VERSION)
        audit = audit_contamination(suite, corpus, policy=policy, revision=revision)
        print(
            f"{suite_id:22} {revision.revision_id:34} {revision.case_count:5} "
            f"{audit.status.value:>9} {str(audit.finding_count):>8}"
        )
        rows.append(
            {
                "suite_id": suite_id,
                "revision": revision.revision_id,
                "suite_content": revision.suite_id,
                "capability": revision.capability,
                "scorer": revision.scorer,
                "scorer_version": revision.scorer_version,
                "case_count": revision.case_count,
                "audit_status": audit.status.value,
                "finding_count": audit.finding_count,
                "quarantined_case_ids": list(audit.quarantined_case_ids),
            }
        )
        records.append(
            ResultRecord(
                label=suite_id,
                revision=revision,
                candidate_id=SCORER_VERSION,
                audit=audit,
                pass_rate=0.0,
                mean_score=0.0,
            )
        )

    print()
    distinct = len({row["revision"] for row in rows})
    print(f"{len(rows)} suites, {distinct} distinct revisions")
    blocked = [r for r in rows if r["audit_status"] != "clean"]
    print(f"suites that cannot gate a promotion (audit not clean): {len(blocked)}")
    for row in blocked:
        print(
            f"  - {row['suite_id']}: {row['audit_status']} "
            f"({row['finding_count']} finding(s))"
        )

    trend = build_trend(records)
    print()
    print(f"trend groups: {len(trend.groups)} (records span distinct suites by design)")

    artifact = {
        "issue": 172,
        "harness": harness_identity(),
        "scorer_version": SCORER_VERSION,
        "policy": {"shingle_size": policy.shingle_size,
                   "near_overlap_threshold": policy.near_overlap_threshold},
        "corpus_files": len(corpus),
        "corpus_dirs": list(CORPUS_DIRS),
        "suites": rows,
        "distinct_revisions": distinct,
        "suites_blocked_by_audit": [r["suite_id"] for r in blocked],
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(artifact, indent=2))
        print(f"\nartifact: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
