#!/usr/bin/env python3
"""Re-baseline deterministic accuracy against the real serving path.

#240 carried an accuracy figure of ``0/12`` that cited
``evidence/latency-20261003/summary.json`` -- a path that has never
existed in this repository, so the number could be neither reproduced nor
refuted. This script produces the artifact that was missing: a
machine-checkable accuracy baseline, tied to an exact source SHA, a proven
serving identity, and the raw model output for every case.

Run with::

    .venv/bin/python scripts/accuracy_baseline.py \
        --out evidence/accuracy/deterministic_hard_12.json

Design constraints
------------------

- **The oracle is a regex, not a model.** Scoring uses
  :func:`oai2.evals.run_suite`, so nothing grades the output except the
  patterns in the suite. An LLM judge would be exactly the unverifiable
  dependency this work exists to remove.
- **Serving identity is read from ``/props``**, so the record names the
  weights that produced it instead of the weights someone intended to
  load.
- **Sampling is pinned** (``temperature=0.0``, fixed seed). An accuracy
  number that changes between runs at the same SHA is not a baseline.
- **Raw output is preserved verbatim** for every case, including failures.
  A score without the text that produced it cannot be audited, and a
  failure taxonomy derived only from the score hides *why*.

Exit codes
----------
``0``
    Baseline recorded (this says nothing about the score; a 0/12 is a
    valid, useful result).
``1``
    Could not record a baseline -- the server was unreachable, or its
    identity could not be proven.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from oai2.evals import builtin_suite, run_suite  # noqa: E402
from oai2.evals.deterministic_hard import CASE_COUNT, DERIVED_ANSWERS  # noqa: E402
from oai2.runtime.inference import InferenceRequest  # noqa: E402
from oai2.runtime.llamacpp_runtime import (  # noqa: E402
    DEFAULT_SEED,
    LlamaServerError,
    LlamaServerRuntime,
)

SUITE_NAME = "deterministic_hard"

#: Input string for the reversal case, used to detect responses that are not
#: even a permutation of the source. A non-permutation is a different defect
#: from an imprecise reversal, and collapsing the two would hide it.
_REVERSAL_SOURCE = "orchords"


def classify_failure(case_id: str, expected: str, text: str) -> str:
    """Name *why* a case failed, from the shape of the answer.

    A bare score hides the interesting part: "0/12" and "12/12, all wrong the
    same way" are different systems. Categories are derived mechanically from
    the response so the taxonomy cannot be edited to suit a result.
    """
    stripped = text.strip()
    if not stripped:
        return "empty_response"
    if re.search(re.escape(expected), stripped, re.IGNORECASE):
        return "correct_but_unmatched"  # oracle gap, not a model error
    if case_id == "string_reversal":
        letters = {c for c in stripped.lower() if c.isalpha()}
        if not letters <= set(_REVERSAL_SOURCE):
            return "symbolic_corruption"  # emitted characters not in the input
        return "wrong_permutation"
    if expected.isdigit():
        numbers = re.findall(r"-?\d+", stripped)
        if numbers:
            return "arithmetic_error"
        return "non_numeric_answer"
    if expected in {"True", "False"}:
        return "logic_error"
    return "fact_error"


def git(*args: str) -> str | None:
    try:
        return subprocess.run(  # noqa: S603
            ["git", "-C", str(REPO), *args],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    # PEP 758, not Python 2: unparenthesised multiple exception types,
        # valid from Python 3.14 (`requires-python = ">=3.14"`). Only an
        # `as` clause still needs the parentheses. Left as-is deliberately.
    except OSError, subprocess.SubprocessError:
        return None


def _harness_identity() -> dict[str, str]:
    return {
        "source": "scripts/accuracy_baseline.py",
        "commit": git("rev-parse", "HEAD") or "unavailable",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "dirty": "true" if git("status", "--porcelain", "--", "oai2", "scripts") else "false",
    }


def _taxonomy(cases: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        if case["passed"]:
            continue
        key = str(case["failure_class"])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=None, help="llama-server base URL")
    parser.add_argument("--model", default=None, help="model alias to request")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--out", default=None, help="artifact path to write")
    args = parser.parse_args(argv)

    suite = builtin_suite(SUITE_NAME)
    if len(suite.cases) != CASE_COUNT:
        print(
            f"FAIL accuracy-baseline: suite has {len(suite.cases)} cases, expected {CASE_COUNT}",
            file=sys.stderr,
        )
        return 1

    runtime = LlamaServerRuntime(base_url=args.base_url, model=args.model)
    try:
        identity = runtime.serving_identity()
    except LlamaServerError as exc:
        print(f"FAIL accuracy-baseline: could not prove serving identity: {exc}", file=sys.stderr)
        return 1
    if not identity.get("model_path"):
        print("FAIL accuracy-baseline: /props returned no model_path", file=sys.stderr)
        return 1

    print(f"serving: {identity['model_path']}  slots={identity['total_slots']}")
    print(f"suite:   {suite.suite_id} ({len(suite.cases)} cases)")

    # The harness scores cases; the raw text is captured here so the
    # artifact carries the evidence, not only the verdict.
    raw: dict[str, str] = {}
    original_generate = runtime.generate

    def capturing_generate(request: InferenceRequest):  # noqa: ANN001, ANN202
        response = original_generate(request)
        raw[request.prompt] = response.text
        return response

    runtime.generate = capturing_generate  # type: ignore[method-assign]

    # The harness's default request factory leaves `temperature` at 0.7, so
    # two runs at the same SHA would disagree and the score would not be a
    # baseline. Pin it through the harness's own `request_factory` hook
    # rather than bypassing the harness.
    def pinned_request(case):  # noqa: ANN001, ANN202
        return InferenceRequest(
            prompt=case.prompt,
            max_tokens=args.max_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=DEFAULT_SEED,
        )

    report = run_suite(suite, runtime, request_factory=pinned_request, max_tokens=args.max_tokens)

    passed = [s for s in report.scores if s.passed]
    by_case = {c.case_id: c for c in suite.cases}

    cases_out = []
    for score in report.scores:
        case = by_case[score.case_id]
        text = raw.get(case.prompt, "")
        cases_out.append(
            {
                "case_id": score.case_id,
                "prompt": case.prompt,
                "expected_patterns": list(case.expected_patterns),
                "oracle_note": case.notes,
                "passed": score.passed,
                "matched_pattern": score.matched_pattern,
                "failure_reason": score.notes or None,
                "failure_class": classify_failure(
                    score.case_id, DERIVED_ANSWERS[score.case_id], text
                ),
                "expected_answer": DERIVED_ANSWERS[score.case_id],
                "raw_output": text,
            }
        )

    artifact = {
        "generated_at": datetime.now(UTC).isoformat(),
        "harness": _harness_identity(),
        "serving_identity": identity,
        "suite_id": suite.suite_id,
        "case_count": len(suite.cases),
        "scorer": suite.scorer,
        "sampling": {
            "temperature": 0.0,
            "seed": DEFAULT_SEED,
            "max_tokens": args.max_tokens,
            "note": "temperature 0.0 + fixed seed: the same SHA must reproduce this score",
        },
        "failure_taxonomy": _taxonomy(cases_out),
        "result": {
            "passed": len(passed),
            "total": len(report.scores),
            "pass_rate": round(len(passed) / len(report.scores), 4) if report.scores else 0.0,
        },
        "cases": cases_out,
    }

    print(f"\nresult: {len(passed)}/{len(report.scores)} verified-correct")
    for entry in cases_out:
        mark = "PASS" if entry["passed"] else "FAIL"
        print(f"  [{mark}] {entry['case_id']:<24} {entry['failure_reason'] or ''}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote: {out}")
    else:
        print(json.dumps(artifact, indent=2))

    runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
