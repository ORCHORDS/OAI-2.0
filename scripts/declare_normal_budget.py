#!/usr/bin/env python3
"""Declare the NORMAL service budget and evaluate the measured evidence.

AC-PERF-035 requires that promoted settings "pass capability/verification
gates and remain inside declared p95/p99/resource limits". No such limit has
ever been declared for the NORMAL lane, so nothing could pass or fail one.

This script closes the source-side half. It declares a
:class:`oai2.evals.qos.WorkloadBudget` for the NORMAL lane, feeds the
already-measured evidence into the **existing** QoS gate
(:func:`evaluate_budget` / :func:`evaluate_promotion`), and reports
whether the current production configuration passes.

It does **not** promote anything. Promotion changes what production
serves and is an owner decision; this makes that decision informed.

The budget is not reverse-engineered from the measurements. Its values come
from what an interactive service lane is for, and are stated here so a
reviewer can argue with them:

- ``first_useful_action_p95_ms = 1000`` — one second to a usable answer is
  the standard interactive budget, and this is the metric the issue's own
  promotion rule cares about, not TTFT.
- ``end_to_end_p95_ms = 5000`` / ``end_to_end_p99_ms = 10000`` — a 256-token
  generation at the measured ~100-180 tok/s is 1.5-2.5 s, so 5 s leaves
  roughly 2x headroom before it becomes the binding constraint.
- ``max_false_success_rate = 0.02`` — returning a confident wrong answer is
  the dangerous failure mode for an agent lane, not slowness.
- ``min_verified_success_rate = 0.90`` — a service lane that is right less
  than 90% of the time is not a service.

Censored observations
---------------------

Eight of the twelve accuracy cases are never answered correctly, so they have
**no finite** time to a useful action. Rather than dropping them (which would
report a 100% verified-success rate over the cases that happened to be easy)
or inventing a number, each is recorded at the attempt cap as a **lower
bound**, flagged ``deadline_missed=True``, with ``verified_success=False``
and ``declared_success=True`` -- which is exactly a false success. The
artifact states the censoring explicitly.

Run with::

    .venv/bin/python scripts/declare_normal_budget.py \
        --out evals/benchmarks/llamacpp_production_1ba5283/normal_budget.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from oai2.evals.qos import (  # noqa: E402
    BudgetKind,
    WorkloadBudget,
    WorkloadClass,
    WorkloadSample,
    evaluate_budget,
    summarize_samples,
)

BENCH = REPO / "evals/benchmarks/llamacpp_production_1ba5283"

TARGET_HARDWARE = "apple-silicon-m5-max-64gb"
CONFIG_ID = "normal-smollm2-1.7b-q4km-ctx8192-par4"

#: Declared NORMAL service budget. See module docstring for the derivation.
NORMAL_BUDGET = WorkloadBudget(
    version="normal-service-v1",
    workload=WorkloadClass.NORMAL,
    kind=BudgetKind.SERVICE_BUDGET,
    target_hardware=TARGET_HARDWARE,
    config_id=CONFIG_ID,
    first_useful_action_p95_ms=1000.0,
    end_to_end_p95_ms=5000.0,
    end_to_end_p99_ms=10000.0,
    max_false_success_rate=0.02,
    min_verified_success_rate=0.90,
)


def _load(name: str) -> list[dict]:
    path = BENCH / name
    if not path.exists():
        raise FileNotFoundError(f"missing measured evidence: {path}")
    return json.loads(path.read_text())


def build_samples(cap_seconds: float) -> tuple[list[WorkloadSample], list[dict]]:
    """Turn the measured accuracy probe into QoS samples.

    The accuracy probe already records, per case and per attempt, the TTFT
    and the wall time of the attempt that produced a correct answer. That is
    precisely the shape :class:`WorkloadSample` wants.
    """
    artifact = _load("first_useful_action.json")
    samples: list[WorkloadSample] = []
    notes: list[dict] = []

    for case in artifact["cases"]:
        attempts = case["attempts"]
        solved = case["verified_correct"]
        if solved:
            best = min(attempts, key=lambda a: a.get("total_seconds", 1e9))
            ttft_ms = float(best["ttft_seconds"]) * 1000.0
            useful_ms = float(case["seconds_to_first_correct"]) * 1000.0
            deadline_missed = False
            censored = False
        else:
            # No finite time to a correct answer. Record the attempt cap as a
            # lower bound rather than dropping the case or inventing a value.
            slowest = max((a.get("total_seconds", 0.0) for a in attempts), default=cap_seconds)
            ttft_ms = float(attempts[0].get("ttft_seconds", 0.0) or 0.0) * 1000.0
            useful_ms = slowest * 1000.0
            deadline_missed = True
            censored = True
        total_ms = max(
            useful_ms,
            sum(float(a.get("total_seconds", 0.0)) for a in attempts) * 1000.0,
        )
        samples.append(
            WorkloadSample(
                workload=WorkloadClass.NORMAL,
                target_hardware=TARGET_HARDWARE,
                config_id=CONFIG_ID,
                ttft_ms=ttft_ms,
                first_useful_action_ms=useful_ms,
                end_to_end_ms=total_ms,
                # The model returned a confident answer in every case, so
                # declared_success is uniformly True; only verification
                # separates the 4 that are right from the 8 that are not.
                declared_success=True,
                verified_success=bool(solved),
                # `generated_tokens` is deliberately NOT set from
                # `artifact["sampling"]["max_tokens"]`. That is the configured
                # generation ceiling, not a count of anything the model
                # produced: it was a 256-token budget whether a case answered
                # in 12 tokens or 256, and for a censored case it was never
                # reached at all. Reporting it as an observation overstated
                # generation for every case in the set.
                #
                # The attempt records carry no per-attempt token count, so
                # there is nothing measured to report. `generated_tokens=0`
                # means "not measured", which is the truth, and it keeps this
                # sample out of any decode-rate statistic -- a replay that
                # produced tokens always has a rate, and inventing one here
                # would have been the same defect in a different column.
                generated_tokens=0,
                deadline_missed=deadline_missed,
            )
        )
        if censored:
            notes.append(
                {
                    "case_id": case["case_id"],
                    "first_useful_action_is_lower_bound": True,
                    "reason": "no verified-correct answer within the attempt cap",
                }
            )
    return samples, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--attempt-cap-seconds", type=float, default=6.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    samples, censored = build_samples(args.attempt_cap_seconds)
    report = summarize_samples(samples)
    # AC-PERF-035 is "stay inside declared limits", which is exactly what
    # `evaluate_budget` checks. `evaluate_promotion` is a different gate --
    # it regresses a candidate against a *baseline report* -- and is
    # deliberately not called here, because no baseline WorkloadReport exists
    # for this lane yet. Saying so is better than passing a list to it.
    evaluation = evaluate_budget(report, NORMAL_BUDGET)

    print(f"budget:    {NORMAL_BUDGET.version} ({NORMAL_BUDGET.kind})")
    print(f"workload:  {NORMAL_BUDGET.workload}  config: {NORMAL_BUDGET.config_id}")
    print(f"samples:   {report.sample_count}\n")
    print("measured:")
    print(
        f"  verified_success_rate   {report.verified_success_rate:.3f}"
        f"   (budget min {NORMAL_BUDGET.min_verified_success_rate})"
    )
    print(
        f"  false_success_rate      {report.false_success_rate:.3f}"
        f"   (budget max {NORMAL_BUDGET.max_false_success_rate})"
    )
    print(
        f"  first_useful_action p95 {report.first_useful_action_ms.p95:.1f} ms"
        f"   (budget max {NORMAL_BUDGET.first_useful_action_p95_ms:.0f})"
    )
    print(
        f"  end_to_end p95          {report.end_to_end_ms.p95:.1f} ms"
        f"   (budget max {NORMAL_BUDGET.end_to_end_p95_ms:.0f})"
    )
    print(
        f"  end_to_end p99          {report.end_to_end_ms.p99:.1f} ms"
        f"   (budget max {NORMAL_BUDGET.end_to_end_p99_ms:.0f})"
    )
    print(f"  deadline_miss_rate      {report.deadline_miss_rate:.3f}")

    print(
        f"\nbudget evaluation vs {evaluation.budget_version}: "
        f"{'PASS' if evaluation.passed else 'FAIL'}"
    )
    for field in _CHECKED_FIELDS:
        failed = field in evaluation.failures
        print(f"  [{'FAIL' if failed else 'ok  '}] {field}")
    if evaluation.failures:
        print(f"\n  violated: {', '.join(evaluation.failures)}")

    if censored:
        print(
            f"\n{len(censored)} case(s) had no verified-correct answer within the cap and are "
            f"recorded as\nlower bounds with deadline_missed=True and verified_success=False."
        )

    artifact = {
        "generated_at": datetime.now(UTC).isoformat(),
        "requirement": "AC-PERF-035: promoted settings must stay inside declared p95/p99/resource limits",
        "gate_owner": "oai2.evals.qos (existing) — no parallel budget system was introduced",
        "budget": {
            "version": NORMAL_BUDGET.version,
            "workload": NORMAL_BUDGET.workload,
            "kind": NORMAL_BUDGET.kind,
            "target_hardware": NORMAL_BUDGET.target_hardware,
            "config_id": NORMAL_BUDGET.config_id,
            "first_useful_action_p95_ms": NORMAL_BUDGET.first_useful_action_p95_ms,
            "end_to_end_p95_ms": NORMAL_BUDGET.end_to_end_p95_ms,
            "end_to_end_p99_ms": NORMAL_BUDGET.end_to_end_p99_ms,
            "max_false_success_rate": NORMAL_BUDGET.max_false_success_rate,
            "min_verified_success_rate": NORMAL_BUDGET.min_verified_success_rate,
        },
        "measured": {
            "sample_count": report.sample_count,
            "verified_success_rate": report.verified_success_rate,
            "false_success_rate": report.false_success_rate,
            "deadline_miss_rate": report.deadline_miss_rate,
            "first_useful_action_ms_p95": report.first_useful_action_ms.p95,
            "end_to_end_ms_p95": report.end_to_end_ms.p95,
            "end_to_end_ms_p99": report.end_to_end_ms.p99,
            "ttft_ms_p95": report.ttft_ms.p95,
        },
        "evaluation": {
            "budget_version": evaluation.budget_version,
            "budget_kind": str(evaluation.budget_kind),
            "passed": evaluation.passed,
            "failures": list(evaluation.failures),
        },
        "promotion": {
            "attempted": False,
            "reason": (
                "promotion changes what production serves and is an owner decision; this "
                "script only declares the budget and evaluates the current configuration "
                "against it. evaluate_promotion is also the wrong gate here -- it regresses a "
                "candidate against a baseline WorkloadReport, and no baseline report exists "
                "for this lane yet."
            ),
        },
        "censored_observations": censored,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print(f"wrote: {out}")
    return 0


#: The budget fields `evaluate_budget` can report as failures. Listed so the
#: summary shows every checked field, not only the failing ones -- a gate that
#: prints only failures reads as though everything else was never checked.
_CHECKED_FIELDS = (
    "first_useful_action_p95",
    "end_to_end_p95",
    "end_to_end_p99",
    "false_success_rate",
    "verified_success_rate",
    "deadline_miss_rate",
)


if __name__ == "__main__":
    raise SystemExit(main())
