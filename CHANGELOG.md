# Changelog

- Corrected model-architecture reconciliation: OAI-2.0 FURIOUS/NORMAL/DEEP/SWARM are active-compute profiles (~500M–1B / ~1–2B / ~2–4B / multiple ~1–2B+ lanes), not a single giant q-pipe model. The current q-pipe gateway wire identity remains separate from OAI-2.0's internal architecture. Small MLX artifacts in the repository remain benchmark/probe evidence unless separately promoted.

_Reconciliation baseline: `8520cdb5d8d3e49bc9e12dfd79ccf59e7422e44d` (source state before this documentation commit)._

All notable public changes to OAI-2.0 are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Live admission gate binding the existing WI-QOS-002 policy core to a real serving path: `oai2.runtime.LlamaAdmissionGate` composes `AdmissionPolicy` / `AdmissionQueue` / `ResidencyAccountant` / `AdmissionDecisionTrace` into a concurrency-safe execution loop that holds a server slot for the duration of a request and releases it on completion, failure or cancellation (Refs #232). Before this, nothing outside a test file ever constructed an `AdmissionPolicy` or fed a decision to the residency accountant, so the policy was internally correct and unreachable.
- Single owner for host memory/swap telemetry: `oai2.runtime.host_capacity.read_host_memory` (Refs #232). `scripts/bench.py::_host_memory` now delegates to it rather than keeping a second copy of a parser that once silently zeroed every swap figure.
- `scripts/admission_probe.py`: admit-or-not measurement comparing a no-admission baseline against gated candidates on the same lane, prompts, seed and token budget, with arms interleaved per repetition (Refs #232).

### Fixed

- `ResidencyAccountant` refused to record a request that was queued and then rejected, leaving it permanently PENDING and inflating in-flight counts. A PENDING request re-decided as REJECT is now refused in place, mirroring the existing PENDING-to-ADMIT promotion (Refs #232).

### Changed

- Repository metadata: the previous-cycle commit `86d1f04e1cfdab934ebe10a6a71001d93980cb3c` (chore: ruff nits in knowledge sweep async tests) was authored locally with the dev-shell identity `ZCode <zcode@local>` instead of the campaign identity `ORCHORDS.COM <72497645+ORCHORDS@users.noreply.github.com>` that the rest of the `main` history uses. Its content (one unused-import removal, one EOF newline) is correct and verified live (full local gate `ALL LOCAL CHECKS PASSED`); the SHA is preserved to avoid invalidating references already fetched by other agents, and this entry records the attribution regression so the trail is auditable.

### Added

- Versioned numerical safety policy, structured finite/range sentinels, extreme-value fixtures, and MLX smoke-probe integration for WP-71.

- Repository/tool/web claim-evidence path adapters consuming the versioned WP-75 evidence policy.
- Deterministic truthfulness metrics and a versioned promotion gate for false success, unsupported claims, stale claims, ignored contradictions, and unnecessary abstention.

- Authenticated framework-neutral Worker transport handler, bound Cloudflare component factory, public-safe Python Worker entrypoint, and deployment template for WP-02 source integration.
- Versioned claim-evidence policy primitives for WP-75 claim taxonomy/evidence binding work.
- Admission-to-safe-batching scheduler bridge for WP-76/WP-25 integration.
- Request-surface binding for the safe session-batching scheduler: `oai2.runtime.service_binding` composes the WP-39 health slice with session/inference protocol routes, client-owned session isolation, exact-compatibility batching, cancellation, and batch/queue/latency metrics (Refs #76).
- MLX hot runtime with digest-keyed prefix KV-state reuse for WI-PERF-003: `oai2.runtime.MLXHotRuntime` (resident weights, per-request prefill/decode notes, opt-in `PrefixKVCache` with common-prefix matching and hit/miss/invalidation metrics) behind the request surface, with live measured rows (~46× prefill reduction on exact repeats, ~28× on first-sight shared-base prompts) (Refs #240).

- `AsyncCloudflareKnowledgeRuntime` source core combining async D1 reader/writer, R2 content bodies, Vectorize semantic retrieval, and best-effort KV caching with integrity/revision checks.
- Focused async runtime tests covering source-level put/get/retrieve behavior for the composed Cloudflare knowledge path.

- Async D1 knowledge reader/query contracts and focused tests for the evolving live Cloudflare adapter surface.
- Safe session-batching scheduler core with exact-compatibility isolation tests for concurrent inference groundwork.

- Repository-wide Markdown reconciliation against current source/issues on 2026-10-02, including corrected closed/open status, Cloudflare progress, benchmark terminology, runner-free verification policy, and WP-76 mapping.
- Async R2 and KV binding wrappers plus D1 knowledge-table/schema export coverage for the evolving live Cloudflare adapter surface.

- Expanded standards-aligned work-package map through WP-76, including claim-level evidence enforcement/hallucination resistance and end-to-end service-quality budgets.
- D1-authoritative GC deletion-lease state contract with expected-revision acquisition, writer exclusion, expired-owner takeover/fencing, retryable failure, idempotent finalize/release, deleted-body tombstones and verified restore semantics.
- Reference-safe, non-destructive R2 liveness reconciliation with shared-body grouping, missing/orphan classification, byte/age metrics, resumable pagination and authoritative-reference fingerprinting.
- Conservative R2 orphan sweep decision core with grace windows, immediate pre-delete D1 recheck, dry-run default, explicit authorization/recovery gates, idempotent absence handling, bounded failure resume and integrity-checked checkpoints.
- Versioned public-safe Cloudflare knowledge transport schemas for request/response, normalized auth context, explicit errors, D1 metadata, R2 body descriptors, Vectorize metadata and KV cache envelopes.
- Regression tests for transport contract versioning, operation requirements, response invariants, content-addressed R2 integrity, Vectorize provenance and revisioned KV envelopes.
- q-pipe compatibility source revision/blob markers tied to the exact public q-pipe source used for importer-policy verification.
- Runner-free local preflight at `scripts/verify.py` covering Ruff, MyPy, Pytest, public-safety and Markdown-link checks.
- Standards-aligned master issue hierarchy and engineering issue/traceability nomenclature.
- Experimental OAI-2.0 Python package scaffold for protocols, reasoning, tools, agents, verification, vision abstractions, knowledge, runtime, and evaluation.
- Corrected MLX benchmark harness with separate load, compile/warm-up, prefill/TTFT, decode, end-to-end, and memory metrics.
- Offline capability-evaluation scaffolds for coding, tool use, bug diagnosis, reasoning, verification, vision, and orchestration.
- Application-level Cloudflare knowledge contract using logical D1/R2/Vectorize/KV roles plus deterministic mock bindings.
- Strict q-pipe import policy aligned to q-pipe's Cloudflare export gate.
- Synthetic 50-row q-pipe import round-trip test.
- Public architecture, branding, support, security, contribution, stars, and donation/sponsorship documentation.

### Changed

- WP-01 verification baseline and WI-VV-001/WI-VV-002 are now completed/closed; q-pipe compatibility WI-MIG-001 is also closed, while the real migration pilot remains open.
- R2 liveness reconciliation WI-GC-001 (#214) is closed; destructive sweep (#215) and D1 deletion-lease/live-concurrency work (#233) remain open.
- Benchmark documentation now calls the v0.2 decode-rate field **MLX-reported generation throughput** rather than overstating it as kernel-level “pure decode.”

- Master Issue #1 is the canonical global map through WP-76 and issue boundary #233, with parent WP issues owning detailed work-item registries.
- Runner-free preflight reports explicit Apple-Silicon PASS/SKIP state and verifies a real sibling/configured q-pipe checkout against pinned compatibility revision/blob hashes when available.
- Removed GitHub Actions runner-backed workflows; OAI-2.0 acceptance is local/manual-first and runner-free.
- Architecture target supersedes the original small-model-only concept: OAI-2.0 targets **10–30B+ total specialist capacity** with difficulty-dependent active compute.
- q-pipe knowledge imports default to **promoted, independently verified, quality-gated rows only**.
- Android curriculum imports require explicit opt-in.
- Cloudflare query caches are revision-keyed so writes invalidate earlier logical cache entries.
- Cloudflare documentation distinguishes application-level contracts from asynchronous live Worker binding APIs.
- Benchmark v0.1 numbers are historical bootstrap evidence, not pure decode evidence.

### Fixed

- Closed R2 sweep race windows by adding a second authoritative reference check immediately before deletion and retiring re-referenced candidates until a fresh dry-run/grace cycle.
- Added a D1 claim/writer-exclusion contract so live integration can fence the remaining cross-service D1-reference/R2-delete race rather than treating one pre-delete lookup as atomic.
- Sweep runtime rejects non-boolean dependency results instead of coercing them into destructive decisions.
- Sweep checkpoints fingerprint full state, including cursor/history/retired candidates, so tampered resume state is rejected.
- Dry-run GC page ingestion is atomic; snapshot booleans/cursors/inventory values are strictly validated; resumed scans can be checked against the current authoritative reference set.
- Repaired the local preflight hard-coded-secret regex and excluded generated environments/caches/build trees from repository scans.
- Removed stale claim that mock Cloudflare methods map 1:1 to live Worker APIs.
- Removed mock-internal assumptions from `KnowledgeStore.all()`.
- Fixed q-pipe dedupe ordering so an ineligible row cannot suppress a later eligible row with the same identity.
- Fixed q-pipe guidance hashing to match the verified exported guidance shape.
- Fixed the residency accountant rejecting the legitimate queue-to-admit transition: a pending request re-decided as ADMIT now promotes in place instead of raising a duplicate-request error (`1c84f2bb`).
- Fixed `oai2.runtime.local_service` returning 422 for authenticated requests: under postponed annotation evaluation FastAPI could not resolve the closure-local bearer scheme, so the dependency was silently dropped and `credentials` degraded to a required query parameter; the scheme is now a module-level `HTTPBearer(auto_error=False)` singleton (`f14c87f7`).
- Fixed `ServiceLifecycle.restart` failing from the FAILED state: restart now re-enters STARTING before delegating to `start`, whose guard only accepts fresh or cleanly-stopped services (`f6e62a77`).
- Restored the admission-scheduler public-surface pin to the deliberate five-name export set and added `httpx2>=2.0.0` to the dev extra so `starlette.testclient` imports cleanly under `-W error` (`07189ae4`, `c2c0700f`).
