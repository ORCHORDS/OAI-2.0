"""Integration tests for scripts/bench.py --backend=llamacpp.

WI-PERF-003 / #240: the harness previously could not measure the actual
production serving path (a running llama-server) at all, which is why the
B0/B1/B2 configurations and the 1/2/4/8 concurrency matrix had never been
produced. These tests pin the behaviour that makes those numbers honest:

- ``prefill_seconds`` and ``decode_tokens_per_second`` come from the
  server's own ``timings`` block, never from HTTP wall-clock;
- the caller-observed TTFT / token rate are recorded in *separate* fields so
  the two can never be conflated;
- ``cache_n`` is surfaced as ``cache_hit_tokens`` (REQ-PERF-032);
- when the server publishes no ``timings`` the split stays ``None`` and a note
  records it — the harness never invents a number;
- ``/props`` serving identity is captured rather than assumed;
- ``cold`` suppresses warm-up credit, ``hot`` does not;
- the concurrency matrix keeps per-agent numbers and reports aggregate beside
  them, never instead of them.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import scripts.bench as bench
from oai2.runtime import host_capacity
from oai2.verification import reproducibility

ROOT = Path(__file__).resolve().parents[1]

BASE = "http://127.0.0.1:8851"

#: Real macOS `vm.swapusage` output shape, and the unspaced variant some
#: BSD-derived tools emit. Both must parse identically.
SWAP_SPACED = "total = 5120.00M  used = 4129.94M  free = 990.06M  (encrypted)"
SWAP_TIGHT = "total=5120.00M used=4129.94M free=990.06M"


def _sse(chunks: list[dict[str, object]]) -> bytes:
    """Encode OpenAI-compatible SSE chunks, including a final [DONE]."""
    parts = []
    for c in chunks:
        parts.append(b"data: " + json.dumps(c).encode() + b"\n\n")
    parts.append(b"data: [DONE]\n\n")
    return b"".join(parts)


def _chunk(content: str | None = None) -> dict[str, object]:
    delta: dict[str, object] = {}
    if content is not None:
        delta["content"] = content
    return {"id": "x", "model": "m", "choices": [{"index": 0, "delta": delta}]}


def _handler(*, timings: dict[str, object] | None, props: dict[str, object] | None = None):
    props = (
        props
        if props is not None
        else {
            "model_path": "/Users/orchords/models/normal/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
            "default_generation_settings": {"n_ctx": 32768, "n_parallel": 4},
            "total_slots": 4,
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json=props)
        body = _sse(
            [
                _chunk("Hello"),
                _chunk(" world"),
                _chunk(None),
                {"id": "x", "model": "m", "choices": [], "timings": timings},
            ]
            if timings is not None
            else [_chunk("Hello"), _chunk(" world"), _chunk(None)]
        )
        return httpx.Response(200, content=body, headers={"Content-Type": "text/event-stream"})

    return handler


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# Server-reported timings are used; wall-clock is kept separate
# ---------------------------------------------------------------------------


def test_server_timings_drive_prefill_and_decode_not_wallclock() -> None:
    """decode tok/s must be the server's predicted_per_second, not our own."""
    c = _client(
        _handler(
            timings={
                "prompt_n": 1413,
                "cache_n": 0,
                "predicted_n": 256,
                "prompt_ms": 15.0,
                "predicted_per_second": 251.5,
            }
        )
    )
    try:
        r = bench.run_one_llamacpp(
            base_url=BASE,
            model="smollm2-1.7b-q4km",
            prompt="hi",
            prompt_label="p",
            max_tokens=256,
            timeout_seconds=5.0,
            client=c,
            warm=False,
            config_label="hot",
        )
    finally:
        c.close()

    assert r.decode_tokens_per_second == pytest.approx(251.5)
    assert r.prefill_seconds == pytest.approx(0.015)
    assert r.prefill_tokens_per_second == pytest.approx(1413 / 0.015)
    assert r.server_prompt_tokens == 1413
    assert r.server_predicted_tokens == 256
    assert r.ttft_seconds is not None


def test_ttft_is_measured_from_a_real_stream_not_a_buffered_body() -> None:
    """Regression: a buffered post() makes TTFT equal end-to-end latency.

    httpx.Client.post() reads the whole body before returning, so the first
    observed line arrives at the *end* of the generation. That silently turns
    TTFT into end-to-end time and produces an absurd client token rate. This
    test serves a genuinely slow stream and pins TTFT well below end-to-end.
    """
    import time as _time

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_path": "/m/model.gguf"})

        def body_iter():
            yield b"data: " + json.dumps(_chunk("first")).encode() + b"\n\n"
            _time.sleep(0.30)
            for i in range(20):
                yield b"data: " + json.dumps(_chunk(f" t{i}")).encode() + b"\n\n"
            yield (
                b"data: "
                + json.dumps(
                    {
                        "choices": [],
                        "timings": {
                            "prompt_n": 10,
                            "cache_n": 0,
                            "predicted_n": 21,
                            "prompt_ms": 1.0,
                            "predicted_per_second": 50.0,
                        },
                    }
                ).encode()
                + b"\n\n"
            )
            yield b"data: [DONE]\n\n"

        return httpx.Response(
            200,
            content=body_iter(),
            headers={"Content-Type": "text/event-stream"},
        )

    c = _client(handler)
    try:
        r = bench.run_one_llamacpp(
            base_url=BASE,
            model="m",
            prompt="hi",
            prompt_label="p",
            max_tokens=32,
            timeout_seconds=10.0,
            client=c,
            warm=False,
            config_label="hot",
        )
    finally:
        c.close()

    assert r.ttft_seconds is not None
    assert r.end_to_end_seconds is not None
    # The 0.30s gap is mid-stream: TTFT must land before it, not after it.
    assert r.ttft_seconds < 0.25, f"TTFT {r.ttft_seconds} looks like end-to-end"
    assert r.end_to_end_seconds >= 0.30
    assert r.decode_seconds is not None and r.decode_seconds > 0
    # A client rate in the tens of tok/s, not the 6-figure garbage a buffered
    # read produces.
    assert 0 < r.client_decode_tokens_per_second < 10_000


def test_prompt_tokens_is_the_sum_of_processed_and_cached() -> None:
    """llama.cpp prompt_n excludes cached tokens; the total is their sum."""
    c = _client(
        _handler(
            timings={
                "prompt_n": 1,
                "cache_n": 1105,
                "predicted_n": 256,
                "prompt_ms": 13.0,
                "predicted_per_second": 96.0,
            }
        )
    )
    try:
        r = bench.run_one_llamacpp(
            base_url=BASE,
            model="m",
            prompt="hi",
            prompt_label="p",
            max_tokens=256,
            timeout_seconds=5.0,
            client=c,
            warm=False,
            config_label="hot-prefix",
        )
    finally:
        c.close()
    assert r.server_prompt_tokens == 1
    assert r.cache_hit_tokens == 1105
    assert r.prompt_tokens == 1106
    assert any("prompt_n=1 cache_n=1105" in n for n in r.notes)


def test_cache_hit_tokens_surfaced_from_server_cache_n() -> None:
    """REQ-PERF-032: cache hit/miss must be observable from the artifact."""
    c = _client(
        _handler(
            timings={
                "prompt_n": 1413,
                "cache_n": 1200,
                "predicted_n": 256,
                "prompt_ms": 4.0,
                "predicted_per_second": 250.0,
            }
        )
    )
    try:
        r = bench.run_one_llamacpp(
            base_url=BASE,
            model="smollm2-1.7b-q4km",
            prompt="hi",
            prompt_label="p",
            max_tokens=256,
            timeout_seconds=5.0,
            client=c,
            warm=False,
            config_label="hot-prefix",
        )
    finally:
        c.close()
    assert r.cache_hit_tokens == 1200
    assert r.config_label == "hot-prefix"


def test_missing_timings_leaves_split_none_and_records_note() -> None:
    """Fail closed: an unmeasurable split stays None, never back-filled."""
    c = _client(_handler(timings=None))
    try:
        r = bench.run_one_llamacpp(
            base_url=BASE,
            model="m",
            prompt="hi",
            prompt_label="p",
            max_tokens=16,
            timeout_seconds=5.0,
            client=c,
            warm=False,
            config_label="hot",
        )
    finally:
        c.close()
    assert r.prefill_seconds is None
    assert r.decode_tokens_per_second is None
    assert "server-published-no-timings" in r.notes


def test_serving_identity_read_from_props_not_assumed() -> None:
    c = _client(_handler(timings={"prompt_ms": 1.0, "predicted_per_second": 1.0}))
    try:
        r = bench.run_one_llamacpp(
            base_url=BASE,
            model="m",
            prompt="hi",
            prompt_label="p",
            max_tokens=8,
            timeout_seconds=5.0,
            client=c,
            warm=False,
            config_label="hot",
        )
    finally:
        c.close()
    ident = r.serving_identity
    assert ident is not None
    assert ident["model_basename"] == "SmolLM2-1.7B-Instruct-Q4_K_M.gguf"
    assert ident["n_parallel"] == 4
    assert ident["total_slots"] == 4
    assert r.device == f"llamacpp:{BASE}"


def test_props_failure_is_recorded_not_fabricated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(500)
        return httpx.Response(
            200,
            content=_sse([_chunk("a"), _chunk(None)]),
            headers={"Content-Type": "text/event-stream"},
        )

    c = _client(handler)
    try:
        ident = bench.probe_llamacpp_identity(BASE, c)
    finally:
        c.close()
    assert "model_path" not in ident
    assert "props_error" in ident


# ---------------------------------------------------------------------------
# Configuration A/B/C: warm-up credit
# ---------------------------------------------------------------------------


def test_hot_config_performs_a_discarded_warmup() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_path": "/m/model.gguf"})
        calls.append(1)
        return httpx.Response(
            200,
            content=_sse([_chunk("a"), _chunk(None)]),
            headers={"Content-Type": "text/event-stream"},
        )

    c = _client(handler)
    try:
        r = bench.run_one_llamacpp(
            base_url=BASE,
            model="m",
            prompt="hi",
            prompt_label="p",
            max_tokens=8,
            timeout_seconds=5.0,
            client=c,
            warm=True,
            config_label="hot",
        )
    finally:
        c.close()
    assert len(calls) == 2  # warm-up + measured
    assert r.warm_run_seconds is not None


def test_cold_config_gets_no_warmup_credit() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_path": "/m/model.gguf"})
        calls.append(1)
        return httpx.Response(
            200,
            content=_sse([_chunk("a"), _chunk(None)]),
            headers={"Content-Type": "text/event-stream"},
        )

    c = _client(handler)
    try:
        r = bench.run_one_llamacpp(
            base_url=BASE,
            model="m",
            prompt="hi",
            prompt_label="p",
            max_tokens=8,
            timeout_seconds=5.0,
            client=c,
            warm=False,
            config_label="cold",
        )
    finally:
        c.close()
    assert len(calls) == 1
    assert r.warm_run_seconds is None


# ---------------------------------------------------------------------------
# Configuration F: concurrency matrix
# ---------------------------------------------------------------------------


def test_concurrency_returns_one_independent_run_per_agent() -> None:
    c = _client(
        _handler(
            timings={
                "prompt_n": 10,
                "cache_n": 0,
                "predicted_n": 8,
                "prompt_ms": 2.0,
                "predicted_per_second": 200.0,
            }
        )
    )
    try:
        runs = bench.run_concurrent_llamacpp(
            base_url=BASE,
            model="m",
            prompt="hi",
            prompt_label="p",
            max_tokens=8,
            timeout_seconds=5.0,
            client=c,
            agents=4,
            warm=False,
            config_label="hot",
        )
    finally:
        c.close()
    assert len(runs) == 4
    assert {r.agents for r in runs} == {4}
    # Per-agent decode stays a per-agent number; the caller derives the sum.
    assert all(r.decode_tokens_per_second == pytest.approx(200.0) for r in runs)
    assert len({r.run_id for r in runs}) == 4


def test_concurrency_rejects_zero_agents() -> None:
    c = _client(_handler(timings=None))
    try:
        with pytest.raises(ValueError):
            bench.run_concurrent_llamacpp(
                base_url=BASE,
                model="m",
                prompt="hi",
                prompt_label="p",
                max_tokens=8,
                timeout_seconds=5.0,
                client=c,
                agents=0,
                warm=False,
                config_label="hot",
            )
    finally:
        c.close()


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_main_skips_llamacpp_without_base_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OAI2_LLAMACPP_BASE_URL", raising=False)
    monkeypatch.delenv("OAI2_LLAMACPP_MODEL", raising=False)
    rc = bench.main(
        [
            "--backend",
            "llamacpp",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert "SKIP llamacpp-bench" in capsys.readouterr().err


def test_main_llamacpp_writes_summary_with_identity_and_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    handler = _handler(
        timings={
            "prompt_n": 100,
            "cache_n": 40,
            "predicted_n": 16,
            "prompt_ms": 3.0,
            "predicted_per_second": 123.0,
        }
    )
    real_client = httpx.Client

    def factory(**kwargs):
        kwargs.pop("timeout", None)
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)
    rc = bench.main(
        [
            "--backend",
            "llamacpp",
            "--llamacpp-base-url",
            BASE,
            "--llamacpp-model",
            "smollm2-1.7b-q4km",
            "--config",
            "hot-prefix",
            "--prompt-tokens",
            "128",
            "--repetitions",
            "2",
            "--max-tokens",
            "16",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    files = list(tmp_path.glob("summary_*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["backend"] == "llamacpp"
    assert data["config_label"] == "hot-prefix"
    assert data["llamacpp_model"] == "smollm2-1.7b-q4km"
    assert data["serving_identity"]["model_basename"] == ("SmolLM2-1.7B-Instruct-Q4_K_M.gguf")
    cfg = data["configs"][0]
    assert cfg["aggregate"]["cache_hit_tokens"]["median"] == 40
    # Aggregate sits beside the per-agent numbers, not instead of them, and is
    # summed within a repetition: 2 reps x 123 tok/s is 123 per repetition, not
    # a flat 246 that also multiplies by the repetition count.
    assert cfg["aggregate"]["aggregate_decode_tokens_per_second"]["per_repetition"] == [
        pytest.approx(123.0),
        pytest.approx(123.0),
    ]
    assert cfg["aggregate"]["aggregate_decode_tokens_per_second"]["median"] == pytest.approx(123.0)
    assert cfg["aggregate"]["decode_tokens_per_second"]["median"] == pytest.approx(123.0)
    assert "wrote summary" in capsys.readouterr().err


def test_main_llamacpp_agents_matrix_emits_one_run_per_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    handler = _handler(
        timings={
            "prompt_n": 100,
            "cache_n": 0,
            "predicted_n": 16,
            "prompt_ms": 3.0,
            "predicted_per_second": 100.0,
        }
    )
    real_client = httpx.Client

    def factory(**kwargs):
        kwargs.pop("timeout", None)
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)
    rc = bench.main(
        [
            "--backend",
            "llamacpp",
            "--llamacpp-base-url",
            BASE,
            "--llamacpp-model",
            "m",
            "--agents",
            "4",
            "--prompt-tokens",
            "128",
            "--repetitions",
            "2",
            "--max-tokens",
            "16",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    data = json.loads(next(tmp_path.glob("summary_*.json")).read_text())
    assert data["agents"] == 4
    assert data["config_label"] == "hot"
    runs = data["configs"][0]["runs"]
    assert len(runs) == 8  # 2 repetitions x 4 agents
    assert all(r["agents"] == 4 for r in runs)
    # 4 concurrent agents per repetition, reduced across the 2 repetitions.
    # The flat sum (2 x 4 x 100 = 800) is throughput the system never reached.
    assert data["configs"][0]["aggregate"]["aggregate_decode_tokens_per_second"][
        "per_repetition"
    ] == [pytest.approx(400.0), pytest.approx(400.0)]
    assert data["configs"][0]["aggregate"]["aggregate_decode_tokens_per_second"]["repetitions"] == 2
    # Per-agent rate is unchanged by the aggregation fix.
    assert data["configs"][0]["aggregate"]["decode_tokens_per_second"]["median"] == (
        pytest.approx(100.0)
    )


def test_stable_prefix_is_applied_verbatim_to_every_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_path": "/m/model.gguf"})
        if request.method == "POST":
            seen.append(json.loads(request.content)["messages"][0]["content"])
        return httpx.Response(
            200,
            content=_sse([_chunk("a"), _chunk(None)]),
            headers={"Content-Type": "text/event-stream"},
        )

    real_client = httpx.Client

    def factory(**kwargs):
        kwargs.pop("timeout", None)
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)
    rc = bench.main(
        [
            "--backend",
            "llamacpp",
            "--llamacpp-base-url",
            BASE,
            "--llamacpp-model",
            "m",
            "--stable-prefix",
            "STABLE-PREFIX-MARKER",
            "--prompt-tokens",
            "128",
            "--repetitions",
            "3",
            "--max-tokens",
            "8",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    measured = [s for s in seen]
    assert measured, "no request reached the server"
    assert all(s.startswith("STABLE-PREFIX-MARKER") for s in measured)
    # Identical prefix across repetitions is what makes a cache hit possible.
    assert len({s for s in measured}) == 1


def _run(run_id: str, *, repetition: int = 0, decode: float | None) -> bench.RunMetrics:
    """A minimal valid RunMetrics carrying only what aggregation reads."""
    return bench.RunMetrics(
        run_id=run_id,
        timestamp=0.0,
        model="m",
        quantization=None,
        prompt_label="p",
        prompt_tokens=10,
        max_tokens=8,
        load_seconds=None,
        compile_seconds=None,
        warm_run_seconds=None,
        prefill_seconds=None,
        prefill_tokens_per_second=None,
        decode_seconds=None,
        decode_tokens_per_second=decode,
        end_to_end_seconds=None,
        generation_tokens=None,
        peak_memory_gb=None,
        active_memory_gb=None,
        cache_memory_gb=None,
        device="llamacpp:test",
        repetition=repetition,
    )


def test_aggregate_sums_within_a_repetition_not_across_all_runs() -> None:
    """Regression: aggregate must not be a flat sum over every run.

    A flat sum multiplies throughput by the repetition count *and* the agent
    count, so a 2-agent x 3-repetition config reporting 100 tok/s per agent
    would claim 600 tok/s for a system that never exceeded 200.
    """
    runs = [
        _run(f"r{rep}a{agent}", repetition=rep, decode=100.0)
        for rep in range(3)
        for agent in range(2)
    ]
    agg = bench.aggregate_per_repetition(runs)
    assert agg["repetitions"] == 3
    assert agg["median"] == pytest.approx(200.0)  # 2 agents, not 6
    assert agg["per_repetition"] == [200.0, 200.0, 200.0]
    # The flat sum that this replaces.
    assert sum(r.decode_tokens_per_second for r in runs) == pytest.approx(600.0)


def test_aggregate_skips_runs_without_a_decode_rate() -> None:
    """A run with no server-reported rate must not be counted as zero."""
    agg = bench.aggregate_per_repetition(
        [_run("a", repetition=0, decode=100.0), _run("b", repetition=0, decode=None)]
    )
    assert agg["median"] == pytest.approx(100.0)


def test_aggregate_with_no_usable_runs_reports_zero_repetitions() -> None:
    agg = bench.aggregate_per_repetition([])
    assert agg["repetitions"] == 0
    assert agg["median"] is None


def test_main_stamps_repetition_index_on_every_measured_run(tmp_path, monkeypatch) -> None:
    """Repetition index must survive into the written artifacts.

    Without the stamp every run lands in repetition 0, the aggregate collapses
    to a single repetition, and the median over repetitions silently becomes a
    single-sample statistic.
    """
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body["messages"][0]["content"])
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_path": "/m.gguf", "total_slots": 4})
        return httpx.Response(
            200,
            content=_sse(
                [
                    _chunk("a"),
                    {
                        "id": "x",
                        "model": "m",
                        "choices": [{"index": 0, "delta": {}}],
                        "timings": {
                            "prompt_n": 10,
                            "cache_n": 10,
                            "prompt_ms": 10.0,
                            "predicted_n": 8,
                            "predicted_per_second": 100.0,
                        },
                    },
                ]
            ),
            headers={"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def factory(**kwargs):
        kwargs.pop("timeout", None)
        return real_client(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)
    rc = bench.main(
        [
            "--backend",
            "llamacpp",
            "--llamacpp-base-url",
            BASE,
            "--llamacpp-model",
            "m",
            "--prompt-tokens",
            "128",
            "--repetitions",
            "3",
            "--max-tokens",
            "8",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    summary = json.loads(next(tmp_path.glob("summary_*.json")).read_text())
    config = summary["configs"][0]
    assert [r["repetition"] for r in config["runs"]] == [0, 1, 2]
    assert config["aggregate"]["aggregate_decode_tokens_per_second"]["repetitions"] == 3


def test_summary_records_the_harness_revision_that_produced_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every artifact must name the exact source that produced it.

    #240's audit rejected earlier throughput artifacts precisely because no
    revision could be tied to them. A summary without a harness block cannot
    be re-derived or falsified against, so the field is mandatory.
    """
    handler = _handler(
        timings={
            "prompt_n": 10,
            "cache_n": 10,
            "predicted_n": 8,
            "prompt_ms": 1.0,
            "predicted_per_second": 100.0,
        }
    )
    real_client = httpx.Client

    def factory(**kwargs):
        kwargs.pop("timeout", None)
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)
    rc = bench.main(
        [
            "--backend",
            "llamacpp",
            "--llamacpp-base-url",
            BASE,
            "--llamacpp-model",
            "m",
            "--prompt-tokens",
            "128",
            "--repetitions",
            "1",
            "--max-tokens",
            "8",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    harness = json.loads(next(tmp_path.glob("summary_*.json")).read_text())["harness"]
    assert harness["source"] == "scripts/bench.py"
    # Either a real 40-hex SHA, or an explicit "unavailable:..." — never absent.
    commit = harness["commit"]
    assert commit.startswith("unavailable:") or (
        len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
    )
    assert harness["dirty"] in {"true", "false", "unknown"}


def test_harness_identity_reports_git_failure_instead_of_omitting(monkeypatch) -> None:
    """A broken git must degrade to an explicit marker, not a missing key.

    The marker changed shape when provenance moved to its single owner
    (``oai2.verification.reproducibility``). The old code interpolated the
    exception type into the commit, producing a commit-shaped string like
    ``unavailable:OSError`` that could be mistaken for a revision. The owner
    emits the unambiguous ``unavailable`` and keeps the reason separate.

    What this test is really for is unchanged and is the assertion that matters:
    ``dirty`` reads ``unknown``, never ``false``. A git that could not be read
    is not a verified clean tree.
    """

    def boom(*a, **k):
        raise OSError("git not found")

    monkeypatch.setattr(reproducibility.subprocess, "run", boom)
    info = bench._harness_identity()
    assert info["commit"] == "unavailable"
    assert info["dirty"] == "unknown"


def test_host_memory_parses_swap_without_losing_decimals(monkeypatch) -> None:
    """Regression: splitting swap output on '.' silently zeroed the value.

    `vm.swapusage` reports "total = 5120.00M  used = 4129.94M". A whitespace
    or "." split yields 5120 / 0 instead of 5.00 GB / 4.03 GB.
    """

    def fake_run(argv, **kwargs):
        joined = " ".join(argv)
        if joined == "sysctl -n hw.memsize":
            return SimpleNamespace(stdout="68719476736")
        if joined == "sysctl -n vm.swapusage":
            return SimpleNamespace(stdout=SWAP_SPACED)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(host_capacity.subprocess, "run", fake_run)
    m = bench._host_memory()
    assert m["total_gb"] == 64.0
    assert m["swap_total_gb"] == 5.0
    assert m["swap_used_gb"] == pytest.approx(4.03, abs=0.01)


def test_host_memory_parses_swap_regardless_of_spacing_around_equals(monkeypatch) -> None:
    """A whitespace split cannot parse `used=4129.94M`; only a real pattern can.

    This is the case that distinguishes a parser from a lucky `split()`. The
    spaced form is parseable by accident because the label happens to sit two
    tokens before the number; the unspaced form is not parseable that way at
    all, so it is the one that actually pins the behaviour.
    """
    monkeypatch.setattr(
        host_capacity.subprocess,
        "run",
        lambda argv, **k: SimpleNamespace(
            stdout=(
                "68719476736"
                if " ".join(argv) == "sysctl -n hw.memsize"
                else (SWAP_TIGHT if "swapusage" in " ".join(argv) else "")
            )
        ),
    )
    m = bench._host_memory()
    assert m["swap_total_gb"] == 5.0
    assert m["swap_used_gb"] == pytest.approx(4.03, abs=0.01)


def test_host_memory_degrades_to_none_when_sysctl_is_unavailable(monkeypatch) -> None:
    """No sysctl must produce None values, never a fabricated zero."""
    monkeypatch.setattr(
        host_capacity.subprocess, "run", lambda a, **k: (_ for _ in ()).throw(OSError("no sysctl"))
    )
    m = bench._host_memory()
    assert m["total_gb"] is None
    assert m["swap_used_gb"] is None
    assert m["used_gb"] is None


def test_summary_carries_memory_and_swap_columns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    handler = _handler(
        timings={
            "prompt_n": 10,
            "cache_n": 10,
            "predicted_n": 8,
            "prompt_ms": 1.0,
            "predicted_per_second": 100.0,
        }
    )
    real_client = httpx.Client

    def factory(**kwargs):
        kwargs.pop("timeout", None)
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)
    rc = bench.main(
        [
            "--backend",
            "llamacpp",
            "--llamacpp-base-url",
            BASE,
            "--llamacpp-model",
            "m",
            "--prompt-tokens",
            "128",
            "--repetitions",
            "1",
            "--max-tokens",
            "8",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    data = json.loads(next(tmp_path.glob("summary_*.json")).read_text())
    # The comparison table has a Memory/swap column; it must be present at both
    # ends of the run so peak pressure is bounded, not just a single sample.
    for key in ("host_memory_start", "host_memory_end"):
        assert key in data
        assert set(data[key]) >= {"total_gb", "used_gb", "free_gb", "swap_total_gb", "swap_used_gb"}


def test_percentiles_are_observed_samples_not_interpolations() -> None:
    """A reported p99 must be a value some repetition actually produced.

    Interpolated percentiles are conventional but would report numbers no
    run produced; over a 5-repetition cell that is most of the statistic.
    Nearest-rank keeps every reported percentile inside the sample set.
    """
    values = [10, 20, 30, 40, 50]
    stat = bench._stat(values)
    assert stat.p95 in values
    assert stat.p99 in values
    # Nearest-rank on 5 samples puts both at the top of the distribution.
    assert stat.p95 == 50
    assert stat.p99 == 50
    assert stat.p95 == stat.max
    assert stat.p99 == stat.max


def test_percentiles_track_the_tail_not_the_median() -> None:
    """A fast median with a slow tail must be visible as such.

    This is the case min/median/max alone hides: four quick samples and one
    pathological one still produce a good-looking median.
    """
    fast = bench._stat([100, 101, 102, 103, 104])
    tail = bench._stat([100, 101, 102, 103, 100_000])
    assert fast.median == tail.median == 102
    assert tail.p95 == 100_000
    assert tail.p99 == 100_000
    assert tail.p99 > tail.median * 100


def test_percentiles_are_none_when_there_are_no_samples() -> None:
    stat = bench._stat([None, None])
    assert stat.count == 0
    assert stat.p95 is None
    assert stat.p99 is None


def test_percentiles_handle_a_single_sample() -> None:
    stat = bench._stat([42.0])
    assert stat.p95 == 42.0
    assert stat.p99 == 42.0


def test_nearest_rank_p95_and_p99_need_twenty_samples_to_separate() -> None:
    """Documents why the issue's 5-repetition minimum cannot resolve a tail.

    AC-PERF-031 asks for "at least 5 samples per configuration" and that
    minimum is satisfied everywhere. It is still not enough to distinguish
    p95 from p99 under nearest rank: both indices round to the top of the
    sample set until N=20. Anyone reading a p95 from a 5-sample cell is
    reading the maximum, and should be told that rather than left to infer
    it.
    """
    import math

    def index(n: int, fraction: float) -> int:
        return math.ceil(fraction * n) - 1

    # Below 20 the two percentiles are the same observation.
    for n in (5, 10, 15, 19):
        assert index(n, 0.95) == index(n, 0.99), f"N={n} should not separate"

    # 20 is the first N where they can differ.
    assert index(20, 0.95) != index(20, 0.99)
    for n in (25, 40, 100):
        assert index(n, 0.95) <= index(n, 0.99)

    # And the practical consequence: at N=5 the reported p95 *is* the max.
    stat = bench._stat([10, 20, 30, 40, 50])
    assert stat.p95 == stat.max
