"""OAI-2.0 benchmark harness — properly separated prefill / decode / TTFT.

Run with:    uv run python scripts/bench.py [args]
NOT with:    python3 scripts/bench.py   # ModuleNotFoundError: mlx_lm

This script imports :mod:`mlx_lm` (a project dependency declared in
``pyproject.toml``) for the ``mlx`` backend and :mod:`httpx` for the
``gateway`` backend. It runs under the project venv created by ``uv
sync``, i.e. via ``uv run python scripts/bench.py ...``. A plain
``python3`` invocation will fail with :class:`ModuleNotFoundError` for
the ``mlx`` backend because the system interpreter does not see
project-scoped dependencies. The :func:`run_one` guard below converts
that failure into an actionable one-line error.

This harness is the source of truth for measured performance on the
OAI-2.0 reference machines. It supersedes v0.1.0's ``scripts/bench.py``,
which incorrectly attributed prefill time to ``decode_seconds``.

Two backends are supported via ``--backend``:

- ``mlx`` (default) — Apple Silicon MLX, local weights, streaming
  token-by-token. Imports ``mlx_lm`` and reports MLX memory metrics.

- ``gateway`` — :class:`oai2.runtime.GatewayModelClient` driving
  ``https://api.orchords.com``. Requires ``OAI2_GATEWAY_API_KEY`` in
  the environment (or ``--gateway-api-key``). The gateway returns the
  full reply in a single HTTP response, so ``prefill_seconds ==
  end_to_end_seconds`` and ``decode_seconds`` is reported as ``0.0`` by
  convention. Memory metrics are ``None`` because the model runs in the
  cloud, not on this host.

Measured quantities (per run):

- ``load_seconds`` — model load + first-compile wall time (MLX only;
  ``None`` for gateway because there is no local load step).
- ``compile_seconds`` — warm-up run (compiled/uncached path).
- ``warm_run_seconds`` — second warm-up run.
- ``prompt_tokens`` — tokenized prompt size (MLX) or ``len(prompt.split())``
  approximation (gateway).
- ``prefill_seconds`` — wall time from prompt submitted to first token
  yielded. This is the *Time-To-First-Token* (TTFT).
- ``prefill_tokens_per_second`` — ``prompt_tokens / prefill_seconds``.
- ``decode_seconds`` — wall time from first token yielded to last token
  yielded. ``0.0`` for the gateway backend because it returns all at once.
- ``generation_tokens`` — actual number of tokens generated (excluding
  prefill).
- ``decode_tokens_per_second`` — ``generation_tokens / decode_seconds``.
- ``end_to_end_seconds`` — ``prefill_seconds + decode_seconds``.
- ``peak_memory_gb`` — peak unified-memory residency reported by MLX.
  ``None`` for the gateway backend.
- ``active_memory_gb`` — active (live) MLX memory after run.
- ``cache_memory_gb`` — MLX cache memory after run.

Per-model, per-config statistics over ``--repetitions`` runs:

- ``mean``, ``median``, ``min``, ``max``, ``stddev`` for every metric.
- Generated token counts and excerpt of the final output.

If a metric cannot be measured correctly it is reported as ``null``
(JSON) / ``None`` (Python). The harness NEVER invents zeros.
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
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from oai2.runtime.host_capacity import read_host_memory

if TYPE_CHECKING:  # pragma: no cover - typing only
    import httpx


@dataclass(slots=True)
class RunMetrics:
    run_id: str
    timestamp: float
    model: str
    quantization: str | None
    prompt_label: str
    prompt_tokens: int
    max_tokens: int

    load_seconds: float | None
    compile_seconds: float | None
    warm_run_seconds: float | None

    prefill_seconds: float | None  # TTFT
    prefill_tokens_per_second: float | None
    decode_seconds: float | None
    decode_tokens_per_second: float | None
    end_to_end_seconds: float | None
    generation_tokens: int | None

    peak_memory_gb: float | None
    active_memory_gb: float | None
    cache_memory_gb: float | None

    device: str
    sys_info: dict[str, str] = field(default_factory=dict)
    output_text: str = ""
    output_excerpt: str = ""
    notes: list[str] = field(default_factory=list)

    # ---- llama.cpp-server (remote/preloaded) observability ----
    # `prefill_seconds` and `decode_tokens_per_second` above are the
    # *server-reported* values for the llamacpp backend: llama-server
    # publishes its own `timings.prompt_ms` and `timings.predicted_per_second`.
    # HTTP wall-clock is never promoted to the decode rate; it is recorded
    # separately below so the two can never be confused.
    ttft_seconds: float | None = None
    client_decode_tokens_per_second: float | None = None
    server_prompt_tokens: int | None = None
    server_predicted_tokens: int | None = None
    cache_hit_tokens: int | None = None
    serving_identity: dict[str, object] | None = None
    agents: int = 1
    config_label: str = "hot"
    #: Index of the repetition this run belongs to. Concurrency aggregates are
    #: summed *within* a repetition and then reduced across repetitions; a
    #: flat sum over all runs would multiply throughput by the repetition
    #: count and by the agent count simultaneously.
    repetition: int = 0


@dataclass(slots=True)
class Stat:
    count: int
    mean: float | None
    median: float | None
    min: float | None
    max: float | None
    stddev: float | None
    #: Nearest-rank percentiles. AC-PERF-032/035 ask for tail behaviour and
    #: the aggregate previously carried min/median/max only, so a slow cell
    #: and a slow *tail* were indistinguishable from the same run.
    p95: float | None = None
    p99: float | None = None

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def _stat(values: Iterable[float | None]) -> Stat:
    nums = [v for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    if not nums:
        return Stat(count=0, mean=None, median=None, min=None, max=None, stddev=None)
    n = len(nums)
    ordered = sorted(nums)

    def _percentile(fraction: float) -> float:
        """Nearest-rank percentile: the smallest sample at or above ``fraction``.

        Nearest-rank is used rather than interpolation so the reported value
        is always an observed sample. An interpolated p99 of a 5-sample run
        would be a number no run produced.
        """
        index = max(0, math.ceil(fraction * n) - 1)
        return ordered[min(index, n - 1)]

    mean = statistics.fmean(nums)
    median = statistics.median(nums)
    sd = statistics.pstdev(nums) if n > 1 else 0.0
    return Stat(
        count=n,
        mean=mean,
        median=median,
        min=min(nums),
        max=max(nums),
        stddev=sd,
        p95=_percentile(0.95),
        p99=_percentile(0.99),
    )


def _system_info() -> dict[str, str]:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "processor": platform.processor() or "unknown",
    }
    try:
        import mlx.core as mx

        info["mlx_default_device"] = str(mx.default_device())
    except Exception as exc:  # pragma: no cover
        info["mlx_default_device"] = f"unavailable:{type(exc).__name__}"
    try:
        from importlib import metadata as _md

        info["mlx_lm_version"] = _md.version("mlx-lm")
    except Exception:
        info["mlx_lm_version"] = "unknown"
    return info


def _host_memory() -> dict[str, float | None]:
    """Host physical memory and swap in GB, read without extra dependencies.

    The comparison table requires a Memory/swap column. A remote-backend run
    cannot infer this from the HTTP response, so it is read from the OS.

    The parsing lives in :func:`oai2.runtime.host_capacity.read_host_memory`,
    which is the single owner of it. This wrapper stays so the three existing
    tests that pin swap-decimal parsing keep addressing ``bench`` directly; a
    second copy of that parser is what previously let a ``.``-split silently
    zero every swap figure.
    """
    return read_host_memory()


def _harness_identity() -> dict[str, str]:
    """Record the exact source revision that produced a summary artifact.

    #240's audit rejected prior artifacts because no revision could be tied to
    them. A throughput number without the harness SHA that produced it is not
    evidence, so every summary carries this block. Failure to read git is
    reported, never silently omitted.
    """
    info = {"source": "scripts/bench.py"}
    try:
        root = Path(__file__).resolve().parents[1]
        info["commit"] = (
            subprocess.run(  # noqa: S603
                ["git", "-C", str(root), "rev-parse", "HEAD"],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout.strip()
            or "unknown"
        )
        dirty = subprocess.run(  # noqa: S603
            ["git", "-C", str(root), "status", "--porcelain", "--", "scripts/bench.py"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        info["dirty"] = "true" if dirty else "false"
        info["branch"] = (
            subprocess.run(  # noqa: S603
                ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout.strip()
            or "unknown"
        )
    except (OSError, subprocess.SubprocessError) as exc:
        info["commit"] = f"unavailable:{type(exc).__name__}"
        info["dirty"] = "unknown"
    return info


def _build_prompt(target_tokens: int) -> tuple[str, int]:
    """Build a deterministic prompt of approximately ``target_tokens`` words.

    Returns (prompt, actual_word_count). Used as a controlled-length
    fill so we can hit approximate token budgets without depending on
    tokenizer-specific behavior.
    """
    if target_tokens <= 0:
        return "", 0
    base = (
        "Summarize the following Python module. Focus on public functions, "
        "their inputs and outputs, error conditions, and any side effects. "
        "Be concise and accurate. "
    )
    # 1 word ≈ 1.3 tokens for English prose; round generously.
    target_words = max(1, int(target_tokens / 1.3))
    words = (base * (target_words // len(base.split()) + 2)).split()
    text = " ".join(words[:target_words])
    return text, len(text.split())


def _measure_memory() -> tuple[float | None, float | None, float | None]:
    """Return (peak_gb, active_gb, cache_gb). Any unavailable → None."""
    try:
        import mlx.core as mx
    except Exception:
        return None, None, None
    peak = active = cache = None
    try:
        peak = mx.get_peak_memory() / 1e9
    except Exception:
        pass
    try:
        active = mx.get_active_memory() / 1e9
    except Exception:
        pass
    try:
        cache = mx.get_cache_memory() / 1e9
    except Exception:
        pass
    return peak, active, cache


def run_one(
    *,
    model_id: str,
    prompt: str,
    prompt_label: str,
    max_tokens: int,
    warm: bool,
) -> RunMetrics:
    """Load model + run a single measured generation."""
    try:
        from mlx_lm import load, stream_generate  # local import for --help speed.
    except ModuleNotFoundError as exc:
        # Python's import machinery sets `exc.name` only when the message
        # starts with "No module named ". Fall back to parsing the message
        # so the operator sees the *actual* missing module name.
        missing = exc.name or (
            exc.msg.split("'", 2)[1]
            if exc.msg.startswith("No module named '") and "'" in exc.msg[18:]
            else exc.msg
        )
        raise SystemExit(
            f"Python module {missing!r} is not installed in this interpreter. "
            f"bench.py requires project deps in the project venv. Run with "
            f"`uv run python scripts/bench.py ...`, or `uv sync` first."
        ) from exc

    run_id = f"bench_{uuid.uuid4().hex[:10]}"
    sys_info = _system_info()
    notes: list[str] = []

    # ---- Load + compile timing ----
    try:
        import mlx.core as mx

        mx.reset_peak_memory()
    except Exception:
        pass

    t0 = time.perf_counter()
    # mlx_lm.load() returns Union[(model, tokenizer), (model, tokenizer, config)];
    # default-arg path is always the 2-tuple but mypy can't statically narrow it.
    model, tokenizer = load(model_id)  # type: ignore[misc]
    load_seconds = time.perf_counter() - t0

    device = sys_info.get("mlx_default_device", "unknown")
    prompt_tokens = len(tokenizer.encode(prompt))

    # ---- Warm-up: forces MLX graph compile + KV cache allocation,
    #      excluded from measured runs below. ----
    compile_seconds: float | None = None
    warm_run_seconds: float | None = None

    if warm:
        try:
            import mlx.core as mx

            mx.reset_peak_memory()
            t_compile = time.perf_counter()
            for _ in stream_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=8,
            ):
                pass
            mx.eval(model.parameters())
            compile_seconds = time.perf_counter() - t_compile

            t_warm = time.perf_counter()
            for _ in stream_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=8,
            ):
                pass
            mx.eval(model.parameters())
            warm_run_seconds = time.perf_counter() - t_warm
        except Exception as exc:
            notes.append(f"warmup-failed: {type(exc).__name__}: {exc}")

    # ---- Measured generation ----
    prefill_seconds: float | None = None
    prefill_tps: float | None = None
    decode_seconds: float | None = None
    decode_tps: float | None = None
    end_to_end: float | None = None
    generation_tokens: int | None = None
    output_text = ""

    try:
        import mlx.core as mx

        mx.reset_peak_memory()
    except Exception:
        pass

    try:
        t_start = time.perf_counter()
        last_response = None
        for response in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        ):
            last_response = response
            # First yielded response: prompt_tps / generation_tokens=1 / etc.
            if prefill_seconds is None:
                prefill_seconds = time.perf_counter() - t_start
                if prompt_tokens > 0 and prefill_seconds > 0:
                    prefill_tps = prompt_tokens / prefill_seconds
            output_text += response.text

        t_end = time.perf_counter()
        end_to_end = t_end - t_start

        if prefill_seconds is not None:
            decode_seconds = max(t_end - t_start - prefill_seconds, 0.0)

        if last_response is not None:
            generation_tokens = last_response.generation_tokens
            decode_tps = last_response.generation_tps
        else:
            notes.append("stream_generate produced no responses")
            generation_tokens = 0
            decode_tps = None
    except Exception as exc:
        notes.append(f"generate-failed: {type(exc).__name__}: {exc}")

    peak, active, cache = _measure_memory()

    return RunMetrics(
        run_id=run_id,
        timestamp=time.time(),
        model=model_id,
        quantization=_extract_quantization(model_id),
        prompt_label=prompt_label,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        load_seconds=load_seconds,
        compile_seconds=compile_seconds,
        warm_run_seconds=warm_run_seconds,
        prefill_seconds=prefill_seconds,
        prefill_tokens_per_second=prefill_tps,
        decode_seconds=decode_seconds,
        decode_tokens_per_second=decode_tps,
        end_to_end_seconds=end_to_end,
        generation_tokens=generation_tokens,
        peak_memory_gb=peak,
        active_memory_gb=active,
        cache_memory_gb=cache,
        device=device,
        sys_info=sys_info,
        output_text=output_text,
        output_excerpt=output_text[:240],
        notes=notes,
    )


def _extract_quantization(model_id: str) -> str | None:
    lower = model_id.lower()
    for tag in ("4bit", "3bit", "2bit", "8bit", "bf16", "fp16", "f16"):
        if tag in lower:
            return tag
    return None


#: Value type of an aggregate entry. Most entries are a :meth:`Stat.as_dict`
#: mapping of scalars, but the llamacpp backend also stores nested objects
#: (the per-repetition aggregate and the per-agent distribution), so the
#: honest type is `object`, not `float | int | None`.
_AggregateValue = object


def _aggregate(runs: list[RunMetrics]) -> dict[str, _AggregateValue]:
    metric_names = [
        "load_seconds",
        "compile_seconds",
        "warm_run_seconds",
        "prefill_seconds",
        "prefill_tokens_per_second",
        "decode_seconds",
        "decode_tokens_per_second",
        "end_to_end_seconds",
        "peak_memory_gb",
        "active_memory_gb",
        "cache_memory_gb",
    ]
    out: dict[str, _AggregateValue] = {}
    for name in metric_names:
        values = [getattr(r, name) for r in runs]
        s = _stat(values)
        out[name] = s.as_dict()
    # Token counts are integers; report mean/median/min/max only.
    counts = [r.generation_tokens for r in runs if r.generation_tokens is not None]
    if counts:
        out["generation_tokens"] = {
            "count": len(counts),
            "mean": statistics.fmean(counts),
            "median": statistics.median(counts),
            "min": min(counts),
            "max": max(counts),
            "stddev": statistics.pstdev(counts) if len(counts) > 1 else 0.0,
        }
    else:
        out["generation_tokens"] = {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "stddev": None,
        }
    return out


def _print_run_table(model_id: str, prompt_label: str, runs: list[RunMetrics]) -> None:
    print(f"\n=== {model_id}  |  prompt={prompt_label}  |  runs={len(runs)} ===")
    # Per-run numbers
    for i, r in enumerate(runs, 1):
        print(
            f"  run {i:>2}: "
            f"load={_fmt(r.load_seconds)}s "
            f"compile={_fmt(r.compile_seconds)}s "
            f"warm={_fmt(r.warm_run_seconds)}s "
            f"prefill={_fmt(r.prefill_seconds)}s "
            f"decode={_fmt(r.decode_seconds)}s "
            f"gen={r.generation_tokens}tok "
            f"decode_tps={_fmt(r.decode_tokens_per_second)} "
            f"prefill_tps={_fmt(r.prefill_tokens_per_second)} "
            f"peak={_fmt(r.peak_memory_gb)}GB"
        )
    # Aggregates
    agg = _aggregate(runs)
    for name, s in agg.items():
        print(f"  AGG {name}: {s}")


def run_one_gateway(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    prompt_label: str,
    max_tokens: int,
    timeout_seconds: float,
    warm: bool,
) -> RunMetrics:
    """Single measured chat-completion round-trip through GatewayModelClient.

    The gateway returns the full reply in one HTTP response so the
    prefill/decode split collapses:

    - ``prefill_seconds`` == ``end_to_end_seconds`` (full round-trip).
    - ``decode_seconds`` is reported as ``0.0`` by convention because
      the gateway does not stream tokens incrementally.
    - All MLX memory fields are ``None`` (model runs in the cloud).
    - ``generation_tokens`` is a coarse ``max(1, len(text.split()))``
      approximation because the gateway does not return a usage block
      in every code path. ``notes`` records the source.
    """

    run_id = f"bench_{uuid.uuid4().hex[:10]}"
    sys_info = _system_info()
    notes: list[str] = []

    # Local import so --help speed is not gated on httpx.
    from oai2.runtime import GatewayConfig, GatewayModelClient, GatewayRuntime

    config = GatewayConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
    )

    prefill_seconds: float | None = None
    decode_seconds: float | None = None
    end_to_end: float | None = None
    generation_tokens: int | None = None
    output_text = ""

    warm_run_seconds: float | None = None
    compile_seconds: float | None = None

    with GatewayRuntime(config) as runtime:
        client = GatewayModelClient(runtime)
        # ---- Warm-up: forces gateway-side cache priming; excluded
        #      from measured runs below. ----
        if warm:
            try:
                t_warm = time.perf_counter()
                warm_reply = client.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=8,
                    temperature=0.0,
                )
                warm_run_seconds = time.perf_counter() - t_warm
                if not warm_reply.content:
                    notes.append("warmup-empty-reply")
            except Exception as exc:
                notes.append(f"warmup-failed: {type(exc).__name__}: {exc}")
                warm_run_seconds = None
                compile_seconds = None

        # ---- Measured round-trip ----
        try:
            t_start = time.perf_counter()
            reply = client.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            t_end = time.perf_counter()

            end_to_end = t_end - t_start
            # Gateway returns the full reply in one HTTP response, so
            # TTFT == end_to_end and decode is zero by convention.
            prefill_seconds = end_to_end
            decode_seconds = 0.0
            output_text = reply.content
            # The orchords gateway may not always include usage; fall
            # back to a coarse word-count approximation and record it.
            generation_tokens = max(1, len(output_text.split()))
            notes.append("gateway: prefill==end_to_end; decode=0.0 by convention")
        except Exception as exc:
            notes.append(f"generate-failed: {type(exc).__name__}: {exc}")
            prefill_seconds = None
            decode_seconds = None
            end_to_end = None
            generation_tokens = None

    prompt_tokens = len(prompt.split())
    prefill_tps = (
        prompt_tokens / prefill_seconds if prefill_seconds and prefill_seconds > 0 else None
    )
    decode_tps = None  # decode_seconds == 0 by convention; no tps to report

    return RunMetrics(
        run_id=run_id,
        timestamp=time.time(),
        model=f"gateway:{model}",
        quantization=None,
        prompt_label=prompt_label,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        load_seconds=None,  # No local load for the gateway backend.
        compile_seconds=compile_seconds,
        warm_run_seconds=warm_run_seconds,
        prefill_seconds=prefill_seconds,
        prefill_tokens_per_second=prefill_tps,
        decode_seconds=decode_seconds,
        decode_tokens_per_second=decode_tps,
        end_to_end_seconds=end_to_end,
        generation_tokens=generation_tokens,
        peak_memory_gb=None,  # Model runs in the cloud, not on this host.
        active_memory_gb=None,
        cache_memory_gb=None,
        device=f"gateway:{base_url}",
        sys_info=sys_info,
        output_text=output_text,
        output_excerpt=output_text[:240],
        notes=notes,
    )


def aggregate_per_repetition(runs: list[RunMetrics]) -> dict[str, object]:
    """Aggregate throughput summed *within* a repetition, reduced across reps.

    For the concurrency matrix each repetition produces ``agents`` independent
    runs. Summing all runs in the config would multiply the concurrency by the
    repetition count as well, so a flat ``sum`` would overstate real system
    throughput. Per-agent numbers stay on each :class:`RunMetrics`; this only
    produces the system-level figure the issue asks for.
    """
    by_rep: dict[int, list[float]] = {}
    for r in runs:
        if r.decode_tokens_per_second is None:
            continue
        by_rep.setdefault(r.repetition, []).append(r.decode_tokens_per_second)
    per_rep = [sum(v) for _, v in sorted(by_rep.items())]
    if not per_rep:
        return {"repetitions": 0, "median": None, "min": None, "max": None, "per_repetition": []}
    return {
        "repetitions": len(per_rep),
        "median": round(statistics.median(per_rep), 4),
        "min": round(min(per_rep), 4),
        "max": round(max(per_rep), 4),
        "per_repetition": [round(v, 4) for v in per_rep],
    }


def _fmt(v: float | None) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


# ---------------------------------------------------------------------------
# llama.cpp server backend (WI-PERF-003 / #240 configurations A/B/C + F)
# ---------------------------------------------------------------------------


def probe_llamacpp_identity(base_url: str, client: httpx.Client) -> dict[str, object]:
    """Read the serving identity straight from llama-server's ``/props``.

    Issue #240 and the repository model-identity rule require proving
    ``endpoint -> server -> model -> context/parallel`` rather than trusting a
    document or a 200 response. Anything the server does not report is left
    out of the mapping instead of being guessed.
    """
    identity: dict[str, object] = {"base_url": base_url}
    try:
        resp = client.get(f"{base_url}/props")
        resp.raise_for_status()
        props = resp.json()
    except Exception as exc:
        identity["props_error"] = f"{type(exc).__name__}: {exc}"
        return identity

    model_path = props.get("model_path")
    identity["model_path"] = model_path
    identity["model_basename"] = (
        model_path.rsplit("/", 1)[-1] if isinstance(model_path, str) else None
    )
    for key in ("default_generation_settings", "total_slots", "build_info"):
        if key in props:
            identity[key] = props[key]
    gen = props.get("default_generation_settings")
    if isinstance(gen, dict):
        for key in ("n_ctx", "n_parallel", "n_batch", "n_ubatch"):
            if key in gen:
                identity[key] = gen[key]
    return identity


def _sse_json(payload: str | bytes) -> dict[str, object] | None:
    """Decode one ``data:`` line of an OpenAI-compatible SSE stream.

    Accepts ``str`` or ``bytes`` because ``httpx.Response.iter_lines()``
    yields ``str`` while ``aiter_lines()``/raw iteration can yield ``bytes``.
    """
    if isinstance(payload, (bytes, bytearray)):
        text = bytes(payload).decode("utf-8", errors="replace").strip()
    else:
        text = payload.strip()
    if not text or not text.startswith("data:"):
        return None
    body = text[5:].strip()
    if body == "[DONE]":
        return None
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def run_one_llamacpp(
    *,
    base_url: str,
    model: str,
    prompt: str,
    prompt_label: str,
    max_tokens: int,
    timeout_seconds: float,
    client: httpx.Client,
    warm: bool,
    config_label: str,
) -> RunMetrics:
    """One measured streaming generation against a running llama-server.

    ``prefill_seconds`` is llama-server's own ``timings.prompt_ms`` and
    ``decode_tokens_per_second`` is its own ``timings.predicted_per_second``.
    The caller-observed time-to-first-token and the caller-observed token rate
    are recorded in ``ttft_seconds`` / ``client_decode_tokens_per_second`` so a
    transport-level number can never be reported as the model's decode rate.

    ``cache_hit_tokens`` is the server's ``timings.cache_n``. That is the
    observable hit/miss evidence REQ-PERF-032 asks for; when the build does
    not publish ``timings`` the fields stay ``None`` and a note records it.
    """
    run_id = f"bench_{uuid.uuid4().hex[:10]}"
    sys_info = _system_info()
    notes: list[str] = []

    payload = {
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }

    def _post():
        # Streaming is required for an honest TTFT: a non-streaming
        # httpx.Client.post() buffers the entire body before returning, which
        # would make the first observed line arrive at the *end* of the
        # response and silently turn TTFT into end-to-end latency.
        return client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

    warm_run_seconds: float | None = None
    if warm:
        # Discarded: primes the server-side prompt cache / GPU graphs so the
        # measured run below is not the process's first-request compile.
        try:
            t_warm = time.perf_counter()
            with _post() as warm_resp:
                warm_resp.raise_for_status()
                for line in warm_resp.iter_lines():
                    _sse_json(line)
            warm_run_seconds = time.perf_counter() - t_warm
        except Exception as exc:
            notes.append(f"warmup-failed: {type(exc).__name__}: {exc}")

    ttft: float | None = None
    first_token_at: float | None = None
    end_to_end: float | None = None
    chunks = 0
    output_text = ""
    timings: dict[str, object] | None = None

    try:
        t_start = time.perf_counter()
        last = t_start
        with _post() as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                decoded = _sse_json(raw)
                if decoded is None:
                    continue
                chunks += 1
                last = time.perf_counter()
                if first_token_at is None:
                    first_token_at = last - t_start
                if isinstance(decoded.get("timings"), dict):
                    timings = decoded["timings"]  # type: ignore[assignment]
                choices = decoded.get("choices")
                if isinstance(choices, list) and choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        delta = first.get("delta")
                        content = delta.get("content") if isinstance(delta, dict) else None
                        if isinstance(content, str) and content:
                            if ttft is None:
                                # TTFT is the first *content* token, not the
                                # first (possibly empty) role/frame chunk.
                                ttft = last - t_start
                            output_text += content
                        if first.get("finish_reason") is not None:
                            break
        end_to_end = last - t_start
        if ttft is None:
            # No content token was observed: the split is unmeasurable, so
            # leave it None instead of substituting the frame time.
            notes.append("no-content-token-observed")
    except Exception as exc:
        notes.append(f"generate-failed: {type(exc).__name__}: {exc}")

    server_prompt_seconds: float | None = None
    server_decode_tps: float | None = None
    server_prompt_tokens: int | None = None
    server_predicted_tokens: int | None = None
    cache_hit_tokens: int | None = None
    if timings is None:
        notes.append("server-published-no-timings")
    else:
        v_prompt_ms = timings.get("prompt_ms")
        if isinstance(v_prompt_ms, (int, float)):
            server_prompt_seconds = float(v_prompt_ms) / 1000.0
        v_predicted_ps = timings.get("predicted_per_second")
        if isinstance(v_predicted_ps, (int, float)):
            server_decode_tps = float(v_predicted_ps)
        v_prompt_n = timings.get("prompt_n")
        if isinstance(v_prompt_n, (int, float)):
            server_prompt_tokens = int(v_prompt_n)
        v_predicted_n = timings.get("predicted_n")
        if isinstance(v_predicted_n, (int, float)):
            server_predicted_tokens = int(v_predicted_n)
        v_cache_n = timings.get("cache_n")
        if isinstance(v_cache_n, (int, float)):
            cache_hit_tokens = int(v_cache_n)

    # Without server timings the prefill/decode split is genuinely unknown;
    # leave it None rather than back-filling a wall-clock estimate.
    prefill_seconds = server_prompt_seconds
    decode_tps = server_decode_tps
    client_decode_tps: float | None = None
    decode_seconds: float | None = None
    if end_to_end and ttft is not None and end_to_end > ttft:
        decode_seconds = end_to_end - ttft
        if chunks > 1:
            client_decode_tps = (chunks - 1) / decode_seconds

    # llama.cpp reports prompt_n as the tokens *processed on this call*; the
    # cached ones arrive separately as cache_n. The prompt the model actually
    # conditioned on is their sum, so that is what prompt_tokens means here.
    total_prompt_tokens: int | None = None
    if server_prompt_tokens is not None or cache_hit_tokens is not None:
        total_prompt_tokens = (server_prompt_tokens or 0) + (cache_hit_tokens or 0)
        notes.append(
            f"prompt_n={server_prompt_tokens} cache_n={cache_hit_tokens} "
            "(llama.cpp prompt_n excludes cached tokens; prompt_tokens is their sum)"
        )

    return RunMetrics(
        run_id=run_id,
        timestamp=time.time(),
        model=f"llamacpp:{model}",
        quantization=None,
        prompt_label=prompt_label,
        prompt_tokens=total_prompt_tokens
        if total_prompt_tokens is not None
        else len(prompt.split()),
        max_tokens=max_tokens,
        # The server owns model residency; this process loads nothing.
        load_seconds=None,
        compile_seconds=None,
        warm_run_seconds=warm_run_seconds,
        prefill_seconds=prefill_seconds,
        prefill_tokens_per_second=(
            server_prompt_tokens / server_prompt_seconds
            if server_prompt_tokens and server_prompt_seconds
            else None
        ),
        decode_seconds=decode_seconds,
        decode_tokens_per_second=decode_tps,
        end_to_end_seconds=end_to_end,
        generation_tokens=server_predicted_tokens
        if server_predicted_tokens is not None
        else chunks,
        peak_memory_gb=None,
        active_memory_gb=None,
        cache_memory_gb=None,
        device=f"llamacpp:{base_url}",
        sys_info=sys_info,
        output_text=output_text,
        output_excerpt=output_text[:240],
        notes=notes,
        ttft_seconds=ttft,
        client_decode_tokens_per_second=client_decode_tps,
        server_prompt_tokens=server_prompt_tokens,
        server_predicted_tokens=server_predicted_tokens,
        cache_hit_tokens=cache_hit_tokens,
        serving_identity=probe_llamacpp_identity(base_url, client),
        agents=1,
        config_label=config_label,
    )


def run_concurrent_llamacpp(
    *,
    base_url: str,
    model: str,
    prompt: str,
    prompt_label: str,
    max_tokens: int,
    timeout_seconds: float,
    client: httpx.Client,
    agents: int,
    warm: bool,
    config_label: str,
    repetition: int = 0,
) -> list[RunMetrics]:
    """Configuration F: run ``agents`` independent streams against one server.

    Each agent is an independent request; per-agent metrics stay in each
    :class:`RunMetrics` and the caller derives aggregate throughput. Aggregate
    is never substituted for a per-agent number.
    """
    if agents < 1:
        raise ValueError("agents must be >= 1")
    if warm:
        # Prime the shared cache once; every agent then races the same prefix.
        run_one_llamacpp(
            base_url=base_url,
            model=model,
            prompt=prompt,
            prompt_label=prompt_label,
            max_tokens=8,
            timeout_seconds=timeout_seconds,
            client=client,
            warm=False,
            config_label=config_label,
        )

    barrier = threading.Barrier(agents)
    results: list[RunMetrics | None] = [None] * agents
    errors: list[str] = []

    def _worker(index: int) -> None:
        try:
            barrier.wait()
            results[index] = run_one_llamacpp(
                base_url=base_url,
                model=model,
                prompt=prompt,
                prompt_label=prompt_label,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                client=client,
                warm=False,
                config_label=config_label,
            )
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"agent{index}: {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(agents)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for e in errors:
        print(f"WARN {e}", file=sys.stderr)
    done = [r for r in results if r is not None]
    for r in done:
        r.agents = agents
        r.repetition = repetition
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--backend",
        choices=("mlx", "gateway", "llamacpp"),
        default="mlx",
        help="Inference backend: mlx (local Apple Silicon weights), gateway "
        "(OAI-2.0 GatewayModelClient driving api.orchords.com), or llamacpp "
        "(a running llama-server OpenAI-compatible endpoint — the production "
        "serving path).",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/SmolLM-135M-Instruct-4bit",
        help="HuggingFace model id for the mlx backend, or model name for the gateway backend.",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        nargs="+",
        default=[128, 1024, 4096],
        help="One or more approximate prompt token budgets to benchmark.",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warm-up (default: warm up so measured numbers exclude compile).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "evals" / "benchmarks",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional label for the run; included in the summary file name.",
    )
    parser.add_argument(
        "--gateway-base-url",
        default=None,
        help="Gateway backend only. Override OAI2_GATEWAY_BASE_URL.",
    )
    parser.add_argument(
        "--gateway-api-key",
        default=None,
        help="Gateway backend only. Override OAI2_GATEWAY_API_KEY (never echoed).",
    )
    parser.add_argument(
        "--gateway-timeout-seconds",
        type=float,
        default=None,
        help="Gateway backend only. Override OAI2_GATEWAY_TIMEOUT_SECONDS.",
    )
    # ---- llamacpp backend (WI-PERF-003 / #240) ----
    parser.add_argument(
        "--llamacpp-base-url",
        default=None,
        help="llamacpp backend only. Base URL of a running llama-server, e.g. "
        "http://127.0.0.1:8851. Override OAI2_LLAMACPP_BASE_URL.",
    )
    parser.add_argument(
        "--llamacpp-model",
        default=None,
        help="llamacpp backend only. Model/alias the server accepts. Override "
        "OAI2_LLAMACPP_MODEL. Defaults to the alias reported by /props.",
    )
    parser.add_argument(
        "--llamacpp-timeout-seconds",
        type=float,
        default=900.0,
        help="llamacpp backend only. Per-request timeout.",
    )
    parser.add_argument(
        "--config",
        choices=("cold", "hot", "hot-prefix"),
        default="hot",
        help="Configuration label recorded in the artifact: 'cold' (B0, no "
        "warm-up credit), 'hot' (B1, server already resident) or 'hot-prefix' "
        "(B2/C, stable prefix repeated so the server prompt cache is hit).",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=1,
        help="llamacpp backend only. Concurrent independent streams (the #240 "
        "configuration-F concurrency matrix: 1/2/4/8).",
    )
    parser.add_argument(
        "--stable-prefix",
        default=None,
        help="llamacpp backend only. Prepended verbatim to every prompt so the "
        "server-side prompt cache is exercised and cache_n is observable.",
    )
    args = parser.parse_args(argv)

    # ---- Gateway-backend prerequisite resolution ----
    gateway_base_url: str | None = None
    gateway_api_key: str | None = None
    gateway_timeout_seconds: float | None = None
    if args.backend == "gateway":
        # Allow CLI flags to override env vars; fall back to env-only.
        import os

        from oai2.runtime import load_gateway_config_from_env

        env_base = args.gateway_base_url or os.environ.get("OAI2_GATEWAY_BASE_URL")
        env_key = args.gateway_api_key or os.environ.get("OAI2_GATEWAY_API_KEY")
        env_timeout_raw = args.gateway_timeout_seconds
        if env_timeout_raw is None:
            env_timeout_raw_str = os.environ.get("OAI2_GATEWAY_TIMEOUT_SECONDS")
            env_timeout_raw = float(env_timeout_raw_str) if env_timeout_raw_str else None

        config = load_gateway_config_from_env(
            {
                **os.environ,
                **({"OAI2_GATEWAY_API_KEY": env_key} if env_key else {}),
                **({"OAI2_GATEWAY_BASE_URL": env_base} if env_base else {}),
                **(
                    {"OAI2_GATEWAY_TIMEOUT_SECONDS": str(env_timeout_raw)}
                    if env_timeout_raw is not None
                    else {}
                ),
            }
        )
        if config is None:
            print(
                "SKIP gateway-bench: OAI2_GATEWAY_API_KEY not set. "
                "Set it in the environment or pass --gateway-api-key.",
                file=sys.stderr,
            )
            return 0
        gateway_base_url = config.base_url
        gateway_api_key = config.api_key
        gateway_timeout_seconds = config.timeout_seconds

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- llamacpp-backend prerequisite resolution ----
    llamacpp_client = None
    llamacpp_base_url: str | None = None
    llamacpp_model: str | None = None
    if args.backend == "llamacpp":
        import os

        import httpx

        llamacpp_base_url = args.llamacpp_base_url or os.environ.get("OAI2_LLAMACPP_BASE_URL")
        if not llamacpp_base_url:
            print(
                "SKIP llamacpp-bench: no base URL. Pass --llamacpp-base-url or set "
                "OAI2_LLAMACPP_BASE_URL (e.g. http://127.0.0.1:8851).",
                file=sys.stderr,
            )
            return 0
        llamacpp_base_url = llamacpp_base_url.rstrip("/")
        llamacpp_client = httpx.Client(timeout=args.llamacpp_timeout_seconds)
        identity = probe_llamacpp_identity(llamacpp_base_url, llamacpp_client)
        llamacpp_model = args.llamacpp_model or os.environ.get("OAI2_LLAMACPP_MODEL")
        if not llamacpp_model:
            # Take the alias the server actually reports rather than guessing.
            reported = identity.get("model_path")
            llamacpp_model = reported.rsplit("/", 1)[-1] if isinstance(reported, str) else None
        if not llamacpp_model:
            print(
                "SKIP llamacpp-bench: could not determine the served model. Pass "
                "--llamacpp-model or --llamacpp-timeout-seconds is too small for /props.",
                file=sys.stderr,
            )
            llamacpp_client.close()
            return 0

    if args.backend == "gateway":
        tag = args.tag or f"gateway-{args.model}"
    elif args.backend == "llamacpp":
        tag = args.tag or f"llamacpp-{args.config}-a{args.agents}-{llamacpp_model}"
    else:
        tag = args.tag or args.model.split("/")[-1]
    print(
        f"benchmark: backend={args.backend} model={args.model} "
        f"max_tokens={args.max_tokens} repetitions={args.repetitions} "
        f"warmup={not args.no_warmup} tag={tag}",
        file=sys.stderr,
    )

    summary: dict[str, object] = {
        "tag": tag,
        "backend": args.backend,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "repetitions": args.repetitions,
        "warmup": not args.no_warmup,
        "sys_info": _system_info(),
        "harness": _harness_identity(),
        "host_memory_start": _host_memory(),
        "configs": [],  # list[dict[str, object]], narrowed explicitly where appended
    }
    if args.backend == "llamacpp":
        summary["config_label"] = args.config
        summary["agents"] = args.agents
        summary["serving_identity"] = identity
        summary["llamacpp_model"] = llamacpp_model

    for budget in args.prompt_tokens:
        prompt, word_count = _build_prompt(budget)
        if args.stable_prefix:
            prompt = f"{args.stable_prefix}\n\n{prompt}"
            word_count = len(prompt.split())
        prompt_label = f"~{budget}tokens ({word_count}words)"
        runs: list[RunMetrics] = []
        for i in range(args.repetitions):
            print(f"--- {prompt_label} rep {i + 1}/{args.repetitions} ---", file=sys.stderr)
            if args.backend == "gateway":
                assert gateway_base_url is not None
                assert gateway_api_key is not None
                assert gateway_timeout_seconds is not None
                r = run_one_gateway(
                    base_url=gateway_base_url,
                    api_key=gateway_api_key,
                    model=args.model,
                    prompt=prompt,
                    prompt_label=prompt_label,
                    max_tokens=args.max_tokens,
                    timeout_seconds=gateway_timeout_seconds,
                    warm=not args.no_warmup,
                )
            elif args.backend == "llamacpp":
                assert llamacpp_client is not None
                assert llamacpp_base_url is not None
                assert llamacpp_model is not None
                if args.agents > 1:
                    for r in run_concurrent_llamacpp(
                        base_url=llamacpp_base_url,
                        model=llamacpp_model,
                        prompt=prompt,
                        prompt_label=prompt_label,
                        max_tokens=args.max_tokens,
                        timeout_seconds=args.llamacpp_timeout_seconds,
                        client=llamacpp_client,
                        agents=args.agents,
                        warm=not args.no_warmup,
                        config_label=args.config,
                        repetition=i,
                    ):
                        runs.append(r)
                        (out_dir / f"{r.run_id}.json").write_text(
                            json.dumps(asdict(r), indent=2, default=str)
                        )
                    continue
                r = run_one_llamacpp(
                    base_url=llamacpp_base_url,
                    model=llamacpp_model,
                    prompt=prompt,
                    prompt_label=prompt_label,
                    max_tokens=args.max_tokens,
                    timeout_seconds=args.llamacpp_timeout_seconds,
                    client=llamacpp_client,
                    # 'cold' is configuration B0: no warm-up credit at all.
                    warm=(not args.no_warmup) and args.config != "cold",
                    config_label=args.config,
                )
                r.repetition = i
            else:
                r = run_one(
                    model_id=args.model,
                    prompt=prompt,
                    prompt_label=prompt_label,
                    max_tokens=args.max_tokens,
                    warm=not args.no_warmup,
                )
            runs.append(r)
            run_file = out_dir / f"{r.run_id}.json"
            run_file.write_text(json.dumps(asdict(r), indent=2, default=str))
        if not runs:
            continue
        _print_run_table(args.model, prompt_label, runs)
        aggregate = _aggregate(runs)
        if args.backend == "llamacpp":
            for name in ("ttft_seconds", "client_decode_tokens_per_second", "cache_hit_tokens"):
                aggregate[name] = _stat([getattr(r, name) for r in runs]).as_dict()
            per_agent = [
                r.decode_tokens_per_second for r in runs if r.decode_tokens_per_second is not None
            ]
            # Aggregate is reported alongside, never instead of, per-agent, and
            # is summed *within* a repetition then reduced across repetitions —
            # a flat sum would multiply by the repetition count too.
            aggregate["aggregate_decode_tokens_per_second"] = aggregate_per_repetition(runs)
            aggregate["per_agent_decode_tokens_per_second"] = _stat(per_agent).as_dict()
        summary["configs"].append(  # type: ignore[attr-defined]
            {
                "prompt_label": prompt_label,
                "prompt_tokens": runs[0].prompt_tokens if runs else None,
                "runs": [asdict(r) for r in runs],
                "aggregate": aggregate,
            }
        )

    summary["host_memory_end"] = _host_memory()
    if llamacpp_client is not None:
        llamacpp_client.close()

    summary_file = out_dir / f"summary_{tag}.json"
    summary_file.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote summary: {summary_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
