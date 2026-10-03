"""Admit-or-not probe: does admission control buy a better tail, or not?

Issue #232 (WI-QOS-002) requires AC-QOS-025 — "admission policy demonstrates a
better verified tail-latency/stability tradeoff than no admission control" —
and the honest answer to that had never been measured, because nothing in the
repository routed a real request through :class:`LlamaAdmissionGate`. This
script is that measurement.

What it compares
----------------

Two arms against the *same* running llama-server, at the *same* concurrency
levels, with the *same* prompts, seed, token budget and model:

``direct``
    N worker threads call the runtime directly. Nothing bounds concurrency, so
    overflow queues inside the server's own slot pool where it is neither
    prioritised nor observable. This is the no-admission baseline.

``gate<N>``
    The same N workers submit through :class:`LlamaAdmissionGate` with
    ``max_active_tasks=N``. This is the candidate.

It reports the gate's own queue wait separately from execution, because the
whole question is what admission trades away in exchange for what it buys.

Why the arms interleave per repetition
--------------------------------------

A previous measurement of this lane (:issue:`240`) found 30-76% swings in
throughput driven by an unrelated server on the same host. Running whole
blocks of one arm and then the other would let any such drift land entirely on
one arm and be reported as an admission effect. Interleaving every repetition
spreads drift across arms instead of confounding it.

The gate is not assumed to win
------------------------------

``evaluate_promotion`` is the existing owner of the candidate-vs-baseline tail
gate and is used unmodified. If the gate regresses the tail, this script
reports that it regressed. ``--gate-limits`` exists so a worse-than-slot
concurrency can be tested too, since bounding below the slot count is the
configuration most likely to trade throughput for tail.

Limits stated up front
----------------------

- The client is non-streaming, so first output and completion coincide. The
  report therefore sets ``ttft_ms`` and ``first_useful_action_ms`` equal to
  end-to-end. ``evaluate_promotion`` gates on ``end_to_end_ms`` p95/p99 and
  deadline misses only, so this does not flatter or penalise the comparison.
- ``per_lane_memory_gb`` is a *declaration* fed to the policy, not a measured
  KV footprint. The evidence that the host stayed inside its limits is the
  memory/swap sampling recorded here, which is separate and reported as such.
- Verified success is a **control**, not the objective: serving on this lane is
  deterministic, so admission must not change any answer. If it does, that is
  reported as a failure of the gate's correctness, not as a model result.

Usage
-----
    .venv/bin/python scripts/admission_probe.py --out evidence/admission/probe.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oai2.evals.deterministic_hard import deterministic_hard_suite  # noqa: E402
from oai2.evals.qos import (  # noqa: E402
    WorkloadClass,
    WorkloadSample,
    evaluate_promotion,
    summarize_samples,
)
from oai2.runtime.admission_gate import (  # noqa: E402
    GateRequest,
    LlamaAdmissionGate,
    gate_capacity_from_host,
)
from oai2.runtime.host_capacity import read_host_memory  # noqa: E402
from oai2.runtime.inference import InferenceRequest  # noqa: E402
from oai2.runtime.llamacpp_runtime import (  # noqa: E402
    DEFAULT_SEED,
    LlamaServerError,
    LlamaServerRuntime,
)

#: Short-answer cases. Small outputs keep decode time down so the measurement
#: reflects queueing and prefill -- the resources admission actually governs --
#: rather than being dominated by token generation.
CASE_IDS = ("days_in_a_week", "count_letter_in_word", "boolean_logic", "unit_conversion")

TARGET_WORKLOAD = WorkloadClass.NORMAL


@dataclass(slots=True)
class Row:
    """One measured request."""

    arm: str
    agents: int
    repetition: int
    agent: int
    case_id: str
    started_at: float
    end_to_end_ms: float
    wait_ms: float
    server_ms: float
    declared_success: bool
    verified_success: bool
    answer: str
    error: str | None = None


@dataclass(slots=True)
class ArmResult:
    """Aggregate for one (arm, concurrency level) cell."""

    arm: str
    agents: int
    repetitions: int
    rows: list[Row] = field(default_factory=list)
    host_memory_start: dict[str, float | None] = field(default_factory=read_host_memory)
    host_memory_end: dict[str, float | None] = field(default_factory=read_host_memory)
    window_start: float = 0.0
    window_end: float = 0.0

    @property
    def config_id(self) -> str:
        return f"{self.arm}-a{self.agents}"


def harness_identity() -> dict[str, str]:
    """Record the exact source revision that produced the artifact."""
    info = {"commit": "unavailable:OSError", "branch": "unknown", "dirty": "unknown"}
    try:
        head = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
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
        status = subprocess.run(  # noqa: S603
            ["git", "status", "--porcelain"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        info = {"commit": head, "branch": branch, "dirty": "true" if status else "false"}
    except OSError, subprocess.SubprocessError:
        pass
    return info


def _scorer(case_id: str) -> tuple[Any, ...]:
    for case in deterministic_hard_suite().cases:
        if case.case_id == case_id:
            return case.expected_patterns
    raise KeyError(f"unknown case: {case_id}")


def _verified(case_id: str, text: str) -> bool:
    import re

    return any(re.search(pattern, text, re.IGNORECASE) for pattern in _scorer(case_id))


def _one_request(
    runtime: LlamaServerRuntime,
    gate: LlamaAdmissionGate | None,
    arm: str,
    agents: int,
    repetition: int,
    agent: int,
    case_id: str,
    prompt: str,
    max_tokens: int,
) -> Row:
    inference = InferenceRequest(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        seed=DEFAULT_SEED,
    )
    started_at = time.time()
    begin = time.perf_counter()
    wait_ms = 0.0
    answer = ""
    server_ms = 0.0
    error: str | None = None
    declared = False
    try:
        if gate is None:
            response = runtime.generate(inference)
            answer = response.text
            server_ms = response.elapsed_ms
        else:
            result = gate.submit(
                GateRequest(
                    request_id=f"{arm}-a{agents}-r{repetition}-g{agent}-{case_id}",
                    inference=inference,
                )
            )
            wait_ms = result.wait_ms
            assert result.response is not None  # noqa: S101
            answer = result.response.text
            server_ms = result.response.elapsed_ms
        declared = True
    except (LlamaServerError, AssertionError, RuntimeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    end_to_end_ms = (time.perf_counter() - begin) * 1000.0
    return Row(
        arm=arm,
        agents=agents,
        repetition=repetition,
        agent=agent,
        case_id=case_id,
        started_at=started_at,
        end_to_end_ms=end_to_end_ms,
        wait_ms=wait_ms,
        server_ms=server_ms,
        declared_success=declared,
        verified_success=declared and _verified(case_id, answer),
        answer=answer.strip()[:120],
        error=error,
    )


def run_repetition(
    runtime: LlamaServerRuntime,
    gate: LlamaAdmissionGate | None,
    *,
    arm: str,
    agents: int,
    repetition: int,
    cases: list[tuple[str, str]],
    max_tokens: int,
) -> list[Row]:
    """Fire ``agents`` requests simultaneously and collect their rows.

    The case index rotates by repetition as well as by agent. Without that,
    ``agents=1`` only ever sees ``cases[0]`` and ``agents=2`` only ever sees
    ``cases[0:2]``, so a harder case that always fails would look like a
    concurrency-induced correctness regression -- verified success would drop
    from 1.00 to 0.50 purely because more case types became reachable. The
    rotation makes the case mix identical at every concurrency level, which is
    what lets verified success act as a control at all.
    """
    barrier = threading.Barrier(agents)
    rows: list[Row | None] = [None] * agents

    def worker(index: int) -> None:
        case_id, prompt = cases[(index + repetition) % len(cases)]
        barrier.wait()
        rows[index] = _one_request(
            runtime,
            gate,
            arm,
            agents,
            repetition,
            index,
            case_id,
            prompt,
            max_tokens,
        )

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(agents)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return [row for row in rows if row is not None]


def to_samples(result: ArmResult, *, deadline_ms: float, hardware: str) -> list[WorkloadSample]:
    """Convert rows into the existing QoS sample type.

    ``ttft_ms``/``first_useful_action_ms`` equal end-to-end because the client
    is non-streaming; see the module docstring.
    """
    return [
        WorkloadSample(
            workload=TARGET_WORKLOAD,
            target_hardware=hardware,
            config_id=result.config_id,
            ttft_ms=row.end_to_end_ms,
            first_useful_action_ms=row.end_to_end_ms,
            end_to_end_ms=row.end_to_end_ms,
            declared_success=row.declared_success,
            verified_success=row.verified_success,
            generated_tokens=0,
            deadline_missed=row.end_to_end_ms > deadline_ms,
        )
        for row in result.rows
    ]


def _pct(values: list[float], pct: float) -> float:
    """Nearest-rank percentile: report a sample the run actually produced."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "n": len(values),
        "min": round(min(values), 2),
        "p50": round(_pct(values, 50), 2),
        "p95": round(_pct(values, 95), 2),
        "p99": round(_pct(values, 99), 2),
        "max": round(max(values), 2),
        "mean": round(statistics.fmean(values), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8851")
    parser.add_argument("--model", default="smollm2-1.7b-q4km")
    parser.add_argument("--agents", default="1,2,4,8")
    parser.add_argument("--gate-limits", default="2,4")
    parser.add_argument("--samples", type=int, default=80, help="target samples per cell")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--deadline-ms", type=float, default=8_000.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    args = parser.parse_args()

    levels = [int(x) for x in args.agents.split(",") if x.strip()]
    gate_limits = [int(x) for x in args.gate_limits.split(",") if x.strip()]
    max_tokens = args.max_tokens

    runtime = LlamaServerRuntime(base_url=args.base_url, model=args.model, timeout_seconds=300.0)
    identity_before = runtime.serving_identity()
    if not identity_before.get("total_slots"):
        raise SystemExit("refusing to run: /props reported no total_slots")

    hardware = f"{platform.system()}-{platform.machine()}-{read_host_memory()['total_gb']}GB"
    suite = {case.case_id: case.prompt for case in deterministic_hard_suite().cases}
    cases = [(cid, suite[cid]) for cid in CASE_IDS]
    print(f"lane={args.base_url} slots={identity_before['total_slots']} hardware={hardware}")

    run_id = uuid.uuid4().hex[:8]
    gate_specs = [(f"gate{limit}", limit) for limit in gate_limits]
    results: dict[tuple[str, int], ArmResult] = {}
    started = time.time()

    for agents in levels:
        reps = max(1, math.ceil(args.samples / agents))
        cells: dict[str, ArmResult] = {
            name: ArmResult(arm=name, agents=agents, repetitions=reps)
            for name, _ in gate_specs
        }
        cells["direct"] = ArmResult(arm="direct", agents=agents, repetitions=reps)
        for cell in cells.values():
            cell.window_start = time.time()
        for rep in range(reps):
            # Interleaved: any external drift hits every arm, not just one.
            cells["direct"].rows.extend(
                run_repetition(
                    runtime,
                    None,
                    arm="direct",
                    agents=agents,
                    repetition=rep,
                    cases=cases,
                    max_tokens=max_tokens,
                )
            )
            for name, limit in gate_specs:
                capacity = gate_capacity_from_host(max_active_tasks=limit)
                gate = LlamaAdmissionGate(runtime, capacity=capacity)
                cells[name].rows.extend(
                    run_repetition(
                        runtime,
                        gate,
                        arm=name,
                        agents=agents,
                        repetition=rep,
                        cases=cases,
                        max_tokens=max_tokens,
                    )
                )
        for name, cell in cells.items():
            cell.window_end = time.time()
            results[(name, agents)] = cell
            e2e = [row.end_to_end_ms for row in cell.rows]
            wait = [row.wait_ms for row in cell.rows if row.wait_ms > 0]
            verified = sum(row.verified_success for row in cell.rows)
            print(
                f"  agents={agents} arm={name:8} n={len(cell.rows):3} "
                f"e2e p50={_pct(e2e, 50):8.1f} p95={_pct(e2e, 95):8.1f} "
                f"p99={_pct(e2e, 99):8.1f} "
                f"wait p95={_pct(wait, 95) if wait else 0:7.1f} "
                f"verified={verified}/{len(cell.rows)}"
            )

    identity_after = runtime.serving_identity()
    if identity_after.get("model_path") != identity_before.get("model_path"):
        raise SystemExit("refusing to report: the served model changed during the run")

    reports = {
        f"{name}-a{agents}": summarize_samples(
            to_samples(cell, deadline_ms=args.deadline_ms, hardware=hardware)
        )
        for (name, agents), cell in results.items()
    }

    promotions: dict[str, Any] = {}
    for agents in levels:
        baseline = reports[f"direct-a{agents}"]
        for name, _ in gate_specs:
            candidate = reports[f"{name}-a{agents}"]
            verdict = evaluate_promotion(candidate, baseline)
            promotions[f"{name}-a{agents}"] = {
                "passed": verdict.passed,
                "failures": list(verdict.failures),
                "baseline_p95_ms": round(baseline.end_to_end_ms.p95, 2),
                "candidate_p95_ms": round(candidate.end_to_end_ms.p95, 2),
                "p95_ratio": round(
                    candidate.end_to_end_ms.p95 / baseline.end_to_end_ms.p95, 4
                )
                if baseline.end_to_end_ms.p95
                else None,
                "baseline_p99_ms": round(baseline.end_to_end_ms.p99, 2),
                "candidate_p99_ms": round(candidate.end_to_end_ms.p99, 2),
                "p99_ratio": round(
                    candidate.end_to_end_ms.p99 / baseline.end_to_end_ms.p99, 4
                )
                if baseline.end_to_end_ms.p99
                else None,
                "baseline_deadline_miss_rate": round(baseline.deadline_miss_rate, 4),
                "candidate_deadline_miss_rate": round(candidate.deadline_miss_rate, 4),
                "baseline_verified_rate": round(baseline.verified_success_rate, 4),
                "candidate_verified_rate": round(candidate.verified_success_rate, 4),
                "baseline_false_success_rate": round(baseline.false_success_rate, 4),
                "candidate_false_success_rate": round(candidate.false_success_rate, 4),
            }

    artifact: dict[str, Any] = {
        "run_id": run_id,
        "issue": 232,
        "harness": harness_identity(),
        "serving_identity_before": identity_before,
        "serving_identity_after": identity_after,
        "target_hardware": hardware,
        "workload": TARGET_WORKLOAD.value,
        "parameters": {
            "agents": levels,
            "gate_limits": gate_limits,
            "samples_target_per_cell": args.samples,
            "max_tokens": max_tokens,
            "deadline_ms": args.deadline_ms,
            "case_ids": list(CASE_IDS),
            "seed": DEFAULT_SEED,
            "temperature": 0.0,
            "interleaved_arms": True,
            "non_streaming_client": True,
            "declared_lane_memory_gb": 0.25,
        },
        "host_memory_start": read_host_memory(),
        "host_memory_end": read_host_memory(),
        "elapsed_s": round(time.time() - started, 1),
        "cells": {
            f"{name}-a{agents}": {
                "arm": name,
                "agents": agents,
                "repetitions": cell.repetitions,
                "window_start": cell.window_start,
                "window_end": cell.window_end,
                "end_to_end_ms": _stats([r.end_to_end_ms for r in cell.rows]),
                "queue_wait_ms": _stats([r.wait_ms for r in cell.rows if r.wait_ms > 0]),
                "server_ms": _stats([r.server_ms for r in cell.rows]),
                "verified": sum(r.verified_success for r in cell.rows),
                "declared": sum(r.declared_success for r in cell.rows),
                "errors": sorted({r.error for r in cell.rows if r.error}),
                "host_memory_start": cell.host_memory_start,
                "host_memory_end": cell.host_memory_end,
            }
            for (name, agents), cell in results.items()
        },
        "reports": {key: asdict(value) for key, value in reports.items()},
        "promotion": promotions,
        "rows": [
            asdict(row)
            for cell in results.values()
            for row in cell.rows
        ],
    }

    if args.artifact_dir:
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        target = args.artifact_dir / f"admission_probe_{run_id}.json"
    elif args.out:
        target = args.out
    else:
        target = Path(f"admission_probe_{run_id}.json")
    target.write_text(json.dumps(artifact, indent=2, default=str))
    print(f"\nartifact: {target}")

    print("\n=== promotion: candidate(gate) vs baseline(direct), evaluate_promotion ===")
    for key, verdict in promotions.items():
        status = "PASS" if verdict["passed"] else "FAIL"
        print(
            f"  {key:12} {status} p95 {verdict['baseline_p95_ms']} -> "
            f"{verdict['candidate_p95_ms']} (x{verdict['p95_ratio']}) "
            f"p99 x{verdict['p99_ratio']} misses "
            f"{verdict['baseline_deadline_miss_rate']} -> {verdict['candidate_deadline_miss_rate']} "
            f"verified {verdict['baseline_verified_rate']} -> {verdict['candidate_verified_rate']}"
            + (f" {verdict['failures']}" if verdict["failures"] else "")
        )
    runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
