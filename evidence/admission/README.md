# Admission-control evidence — WI-QOS-002 (#232)

## What this is

`admission_probe_b7399ae.json` is the raw artifact behind the claim that
**admission control does not improve tail latency in front of a
`llama-server`, and the repository's own promotion gate rejects every gated
configuration measured.**

Produced by `scripts/admission_probe.py` at source SHA
`b7399ae3430bb7e49f5d62a373efe90582d99f10`, working tree clean
(`harness.dirty == "false"`).

## Identity and conditions

| Property | Value |
|---|---|
| Endpoint | `http://127.0.0.1:8851` (production NORMAL lane) |
| Model | `/Users/orchords/models/normal/SmolLM2-1.7B-Instruct-Q4_K_M.gguf` |
| Alias | `smollm2-1.7b-q4km` |
| `total_slots` (from `/props`) | **4**, unchanged start to end |
| `n_ctx` per slot | 8192 |
| Hardware | Darwin arm64, 64 GB |
| Host memory start / end | 55.51 GB used / **55.51 GB** |
| Host swap start / end | 3.94 GB used / **3.94 GB** |
| Seed / temperature | 20261004 / 0.0 |
| `max_tokens` | 16 |
| Declared deadline | 2000 ms |
| Requests | 6144 (8 concurrency levels x 3 arms x 256 samples) |

Model identity and slot count are read from `/props`, never inferred from a CLI
flag. The high ~55 GB host usage is the *idle* `:8854` lane holding ~19.5 GB
resident, not this workload — see the memory row below.

## Arms

- `direct` — N threads call the runtime directly. No bound on concurrency;
  overflow queues inside the server's own slot pool. **This is the
  no-admission baseline.**
- `gate4` — the same N threads submit through `LlamaAdmissionGate` with
  `max_active_tasks=4` (slot parity).
- `gate2` — same, with `max_active_tasks=2` (below slot count).

Arms are **interleaved per repetition**, so drift from the unrelated server on
this host is spread across arms instead of being attributed to admission.

## Result

| agents | arm | p50 ms | p95 ms | p99 ms | wait p95 ms | served | verified |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | direct | 59 | 134 | 151 | 0 | 256 | 128 |
| 1 | gate2 | 56 | 141 | 160 | 0 | 256 | 128 |
| 1 | gate4 | 58 | 138 | 158 | 0 | 256 | 128 |
| 2 | direct | 67 | 167 | 180 | 0 | 256 | 128 |
| 2 | gate2 | 68 | 164 | 175 | 0 | 256 | 128 |
| 2 | gate4 | 67 | 161 | 182 | 0 | 256 | 128 |
| 4 | direct | 89 | 204 | 216 | 0 | 256 | 128 |
| 4 | gate2 | 160 | 263 | 287 | 160 | 256 | 128 |
| 4 | gate4 | 87 | 202 | 218 | 0 | 256 | 128 |
| 8 | direct | 175 | 282 | 331 | 0 | 256 | 128 |
| 8 | gate2 | 238 | 449 | 479 | 361 | 256 | 128 |
| 8 | gate4 | 176 | 306 | 355 | 173 | 256 | 128 |
| 16 | direct | 298 | 561 | 603 | 0 | 256 | 128 |
| 16 | gate2 | 473 | 879 | 943 | 760 | 256 | 128 |
| 16 | gate4 | 326 | 628 | 694 | 476 | 256 | 128 |
| 32 | direct | 556 | 1042 | 1138 | 0 | 256 | 128 |
| 32 | gate2 | 856 | 1596 | 1759 | 1491 | 256 | 128 |
| 32 | gate4 | 609 | 1069 | 1145 | 960 | 256 | 128 |
| 64 | direct | 1075 | 2019 | 2104 | 0 | 256 | 128 |
| 64 | gate2 | 1837 | 3423 | 3556 | 3220 | 256 | 128 |
| 64 | gate4 | 1127 | 2125 | 2315 | 1976 | 256 | 128 |
| 128 | direct | 2127 | 3950 | 4117 | 0 | 256 | 128 |
| 128 | gate2 | 106 | 2984 | 3286 | 3001 | **132** | 66 |
| 128 | gate4 | 162 | 2209 | 2502 | 2209 | **136** | 69 |

`evaluate_promotion` (the existing WP-76 gate, used unmodified) verdict for
**every** candidate at **every** level and **both** limits: **FAIL**.

| candidate | p95 ratio | p99 ratio | deadline miss | verified | failures |
|---|---:|---:|---|---|---|
| gate4-a1 | 1.026 | 1.051 | 0.000 -> 0.000 | 0.500 -> 0.500 | p95, p99 |
| gate4-a2 | 0.962 | 1.009 | 0.000 -> 0.000 | 0.500 -> 0.500 | p99 |
| gate4-a4 | 0.991 | 1.013 | 0.000 -> 0.000 | 0.500 -> 0.500 | p99 |
| gate4-a8 | 1.085 | 1.093 | 0.000 -> 0.000 | 0.500 -> 0.500 | p95, p99 |
| gate4-a16 | 1.132 | 1.145 | 0.000 -> 0.000 | 0.500 -> 0.500 | p95, p99 |
| gate4-a32 | 1.038 | 1.012 | 0.000 -> 0.000 | 0.500 -> 0.500 | p95 |
| gate4-a64 | 1.053 | 1.105 | 0.059 -> 0.098 | 0.500 -> 0.500 | p95, p99, deadline |
| gate4-a128 | **0.559** | **0.608** | **0.539 -> 0.090** | 0.500 -> **0.270** | **verified_success_regression** |
| gate2-a128 | 0.754 | 0.787 | 0.539 -> 0.195 | 0.500 -> **0.258** | **verified_success_regression** |

## What the numbers say

**1. At or below slot count, admission is overhead.** At 1–2 agents the ratios
straddle 1.0 with the sign flipping between arms (gate2 p95 1.023 at 1 agent,
0.978 at 2). At 4–64 agents at slot parity it is 0.99–1.13, i.e. no benefit and
a mild cost.

**2. Bounding below the slot count is actively harmful.** `gate2` is worse at
every level: p95 x1.29 at 4 agents rising to x1.68 at 64. The mechanism is
visible in the wait column — a client holding 2 of 4 slots idle stops the
server from overlapping work across its own pool.

**3. At 32x oversubscription the tail does improve, and it is not a win.**
At 128 agents gate4 cuts p95 to 0.559x and p99 to 0.608x, and drops deadline
misses from 53.9% to 9.0%. But `served` drops from 256 to 136: **47% of requests
are rejected outright** with `AdmissionRejected(queue_full)`. The tail
improvement is the arithmetic consequence of refusing work instantly instead of
queueing it until it misses its deadline. The promotion gate's
`verified_success_regression` check fires and the candidate is rejected, which
is the correct outcome — this is a fail-fast policy, not a better one.

**4. The gate never changes an answer.** 5900 served requests across all 8
levels and 3 arms produced **exactly one distinct answer per case**, byte for
byte. Admission is not perturbing serving.

## Controls

**Correctness control.** Verified success is 0.500 in every arm at every level
up to 64 agents — identical. It is not 1.0 because two of the four cases are
deterministically wrong on this 1.7B model: `(True AND False) OR True` -> `False`
and "seconds in 3 hours" -> `7200`. Both fail 0/1474 and 0/1475 across every arm
and level. This is a model-capability result and is not attributed to admission.

**Stability control (AC-QOS-021).** Host memory and swap are **identical** at
the start and end of a run that pushed 128 concurrent clients at a 4-slot
server: 55.51 GB used, 3.94 GB swap used, both unchanged. `total_slots` stayed
4 and the model path never changed. Every error recorded was an explicit
`AdmissionRejected(queue_full)`; there were no server errors, timeouts or
restarts. Controlled overload stayed inside the declared limits.

**Negative control.** `tests/test_admission_gate.py` contains
`test_gate_that_ignores_its_policy_is_caught`. A gate whose policy always
admits was injected into the module and the shipped suite was re-run: **12
tests failed**, including `test_gate_never_exceeds_the_declared_slot_limit`
(`peak_active_tasks` 5 > 2) and the residency peak assertion. The suite
therefore detects an unbounded gate rather than merely describing one.

## A defect in this harness, found and corrected

The first full run reported verified success falling from 1.00 at 1–2 agents to
0.50 at 4–8 agents, which reads exactly like a concurrency-induced correctness
regression. It was not. The case index was `cases[index % len(cases)]`, so
`agents=1` could only ever reach `cases[0]`, `agents=2` only `cases[0:2]`, and the
two always-failing cases became reachable only at 4+ agents. The 0.50 was 100%
case-mix and 0% concurrency. Fixed to `cases[(index + repetition) % len(cases)]`
so the case mix is identical at every level, which is what lets verified success
act as a control at all. The first run was discarded rather than reported.

## Conclusion for AC-QOS-025

AC-QOS-025 asks whether admission policy demonstrates a better verified
tail-latency/stability tradeoff than no admission control. **On this layer, the
answer is no**, and the repository's own gate says so at every concurrency level
and both slot limits.

The `llama-server` slot pool *is* the admission control for a
`llama-server`-backed path. A client-side gate at slot parity adds a second
queue in front of the first, carrying no information the server lacks. Bounding
below the slot count prevents the server from overlapping work and is worse than
doing nothing. The only regime where the gate improves the tail is the one where
it refuses nearly half the requests, and that is a service-level trade, not a
tail improvement.

The gate's demonstrated value is elsewhere and is real: bounded queue depth with
explicit reasoned rejection (REQ-QOS-022), public-safe decision telemetry
(REQ-QOS-027), accurate wait/residency/peak accounting that an in-process MLX
path has no other source for, and prompt cancellation of queued work
(AC-QOS-023). Those are not throughput or tail claims and are not made here.

## Reproducing

```sh
.venv/bin/python scripts/admission_probe.py \
  --agents 1,2,4,8,16,32,64,128 --gate-limits 2,4 \
  --samples 256 --max-tokens 16 --deadline-ms 2000 \
  --artifact-dir <dir>
```

The probe fails closed if `/props` reports no `total_slots`, and aborts rather
than reporting if the served model path changes mid-run.
