"""Measure this repository's own evidence for reproduction metadata (WP-58 / WI-DET-002).

Why this exists
---------------

A measured number is only citable as reproducible if the record says enough to
rebuild the conditions that produced it. This repository publishes seven
evidence artifacts and, until now, no one had measured whether they say that.

An audit that only ever ran against fixtures would prove it can recognise
fixtures. This script points it at the real tree: every JSON under
``evidence/``, measured against :data:`oai2.verification.reproducibility.REPRODUCTION_FIELDS`
— the same field list the library uses, so "enough to reproduce" is one
definition rather than a claim in a script and another in a docstring.

The interesting output is the ``missing`` column. A field nobody records is the
finding; the script does not go and invent one.

Usage
-----
    .venv/bin/python scripts/reproducibility_audit.py
    .venv/bin/python scripts/reproducibility_audit.py --json evidence/reproducibility/audit.json
    .venv/bin/python scripts/reproducibility_audit.py --fail-on-gap

Exit status is 0 when the audit ran, regardless of findings: a gap is a result
to report, not a tool failure. Use ``--fail-on-gap`` in a gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oai2.verification.reproducibility import (  # noqa: E402
    REPRODUCTION_FIELDS,
    ArtifactCoverage,
    CoverageSummary,
    inspect_artifact,
    source_identity,
    summarise,
)

REPO = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = REPO / "evidence"


def _iter_artifacts(root: Path) -> list[Path]:
    return sorted(root.rglob("*.json"))


def _inspect(root: Path) -> CoverageSummary:
    """Inspect every artifact under ``root``.

    An artifact that fails to parse becomes an unreadable coverage entry rather
    than being dropped: a file we could not read must never be counted as a
    file we checked and found clean.
    """
    coverages: list[ArtifactCoverage] = []
    for path in _iter_artifacts(root):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            coverages.append(
                ArtifactCoverage(
                    label=str(path.relative_to(REPO)),
                    unreadable=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        coverages.append(inspect_artifact(str(path.relative_to(REPO)), document))
    return summarise(coverages)


def _report(summary: CoverageSummary, out: object) -> None:
    readable = summary.readable
    total = len(readable)

    print(f"artifacts   : {len(summary.coverages)} under evidence/")
    print(f"  readable  : {total}")
    print(f"  unreadable: {len(summary.unreadable)}")
    mean = summary.mean_coverage()
    print(f"mean coverage: {'n/a' if mean is None else f'{mean:.1%}'}")
    print(f"fully reproducible: {len(summary.fully_reproducible())}")
    print()

    width = max(len(name) for name, _ in summary.per_field_presence()) + 2
    print(f"{'field':<{width}} {'requirement':<14} {'present':>9}  missing from")
    print("-" * 100)
    for (name, present), spec in zip(
        summary.per_field_presence(), REPRODUCTION_FIELDS, strict=True
    ):
        offenders = summary.artifacts_missing(name)
        mark = "OK" if present == total and total else f"{present}/{total}"
        detail = "" if not offenders else ", ".join(offenders)
        print(f"{name:<{width}} {spec.requirement:<14} {mark:>9}  {detail}")
    print()

    for coverage in summary.coverages:
        if coverage.is_unreadable:
            print(f"  UNREADABLE {coverage.label}: {coverage.unreadable}")
            continue
        gaps = len(coverage.missing)
        status = "complete" if gaps == 0 else f"{gaps} missing"
        print(f"  {coverage.coverage_ratio:6.1%}  {coverage.label:<52} {status}")

    missing_everywhere = [
        name for name, present in summary.per_field_presence() if present == 0
    ]
    print()
    if not readable:
        print("no readable artifact: coverage is unknown, not zero")
    elif missing_everywhere:
        print(f"fields recorded by no artifact: {', '.join(missing_everywhere)}")
    else:
        print("every required field is recorded by at least one artifact")
    if out is not None:
        print(f"written     : {out}")


def _payload(summary: CoverageSummary, root: Path) -> dict[str, object]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "issue": 176,
        "harness": source_identity(cwd=str(REPO)).as_dict(),
        "audited_root": str(root.relative_to(REPO)),
        "required_fields": [
            {"name": spec.name, "requirement": spec.requirement, "probes": list(spec.probes)}
            for spec in REPRODUCTION_FIELDS
        ],
        "artifacts_scanned": len(summary.coverages),
        "artifacts_readable": len(summary.readable),
        "artifacts_unreadable": len(summary.unreadable),
        "mean_coverage": summary.mean_coverage(),
        "fully_reproducible": list(summary.fully_reproducible()),
        "per_field_presence": {
            name: {
                "present_in": present,
                "readable_total": len(summary.readable),
                "missing_from": list(summary.artifacts_missing(name)),
            }
            for name, present in summary.per_field_presence()
        },
        "artifacts": [
            {
                "label": coverage.label,
                "unreadable": coverage.unreadable,
                "coverage_ratio": coverage.coverage_ratio,
                "present": {name: path for name, path in coverage.present},
                "missing": list(coverage.missing),
            }
            for coverage in summary.coverages
        ],
        "scope": (
            "Measures whether each committed evidence artifact carries the metadata a "
            "reproduction needs (REQ-DET-021/022/023). It does not re-run any "
            "measurement, does not verify that a recorded value is true, and does not "
            "judge whether an artifact should exist."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=EVIDENCE_DIR,
        help="directory to audit (default: evidence/)",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the full report here")
    parser.add_argument(
        "--fail-on-gap",
        action="store_true",
        help="exit 1 when any readable artifact is missing a field",
    )
    args = parser.parse_args(argv)

    root = args.root if args.root.is_absolute() else REPO / args.root
    if not root.is_dir():
        print(f"FAIL: not a directory: {root}", file=sys.stderr)
        return 2

    summary = _inspect(root)
    _report(summary, args.json)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(_payload(summary, root), indent=2) + "\n", encoding="utf-8"
        )
        print(f"written     : {args.json}")

    if args.fail_on_gap and any(c.missing for c in summary.readable):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
