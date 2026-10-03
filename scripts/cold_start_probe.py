#!/usr/bin/env python3
"""Genuine cold-start measurement on an isolated replica (AC-PERF-031).

Config A of #240 requires "fresh model/runtime start; no reusable prompt/KV
state; no intentional warm-up credit". None of that is obtainable against a
long-lived server: its weights are already resident and its slot still holds
the previous request's prefix cache. The ``cold`` label in ``bench.py``
therefore means *no warm-up credit from the harness*, which is a weaker and
different claim.

This script closes that gap without touching production. For every sample it
starts a **new** ``llama-server`` process on its own port, waits for it to
become ready, issues exactly one request, records the server's own timings,
and terminates it. Nothing is reused between samples, so each one is a real
cold start.

Safety
------

- The replica runs on its own port and is never signalled by this script
  beyond its own PID.
- The replica is always terminated, including on failure, so a stray server
  cannot be left resident.
- Flags are copied **verbatim** from the production NORMAL lane so the cold
  figure is comparable to the hot figure; the only differences are ``--port``
  and ``--parallel 1`` (a single sample cannot use more than one slot, and
  fewer slots means a smaller KV allocation on a host that is already
  memory-constrained).

Run with::

    .venv/bin/python scripts/cold_start_probe.py --repetitions 5 \
        --out evals/benchmarks/llamacpp_production_1ba5283/cold_start.json
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

BINARY = "/Users/orchords/src/llama.cpp/build/bin/llama-server"
MODEL = "/Users/orchords/models/normal/SmolLM2-1.7B-Instruct-Q4_K_M.gguf"
ALIAS = "smollm2-1.7b-q4km"

# Copied verbatim from the production NORMAL lane (PID 99616) apart from the
# port and the slot count.
PRODUCTION_FLAGS = [
    "-ngl",
    "99",
    "--ctx-size",
    "32768",
    "--batch-size",
    "4096",
    "--ubatch-size",
    "1024",
    "--jinja",
    "--reasoning",
    "on",
    "--metrics",
    "--log-verbosity",
    "0",
]


def wait_ready(base_url: str, proc: subprocess.Popen[bytes], timeout: float) -> float:
    """Block until ``/props`` answers; return seconds from spawn to ready.

    This is the load time AC-PERF-031's "fresh model/runtime start" is
    really about. It is measured to the moment the server can serve, not to
    process spawn, because a process that exists but cannot answer has not
    started from the caller's point of view.
    """
    start = time.perf_counter()
    deadline = start + timeout
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"replica exited early with code {proc.returncode}")
        # PEP 758, not Python 2: unparenthesised multiple exception types,
        # valid from Python 3.14 (`requires-python = ">=3.14"`). Only an
        # `as` clause still needs the parentheses. Left as-is deliberately.
        try:
            with urllib.request.urlopen(f"{base_url}/props", timeout=2) as r:
                if r.status == 200:
                    return time.perf_counter() - start
        except urllib.error.URLError, TimeoutError, ConnectionError:
            time.sleep(0.25)
    raise TimeoutError(f"replica not ready within {timeout}s")


def one_request(base_url: str, prompt: str, max_tokens: int) -> dict[str, object]:
    """Issue exactly one request and return the server's reported timings."""
    body = json.dumps(
        {
            "model": ALIAS,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 20261004,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as r:
        payload = json.loads(r.read())
    e2e = time.perf_counter() - started

    # llama-server reports its own timings; HTTP wall-clock is kept separate
    # and is never promoted to a decode rate.
    timings = payload.get("timings") or {}
    text = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    return {
        "prompt_tokens_served": timings.get("prompt_n"),
        "cache_n": timings.get("cache_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "predicted_n": timings.get("predicted_n"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "client_e2e_seconds": round(e2e, 4),
        "output_chars": len(text),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--base-port", type=int, default=8951)
    parser.add_argument("--ready-timeout", type=float, default=180.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if not Path(BINARY).exists():
        print(f"FAIL cold-start: {BINARY} not found", file=sys.stderr)
        return 1

    prompt = (
        "You are a deterministic build assistant. You read files, report exact "
        "paths, and never speculate about code you have not read.\n\n"
        "List three concrete steps for verifying a change to a Python module."
    )

    samples: list[dict[str, object]] = []
    for i in range(args.repetitions):
        port = args.base_port + i
        base_url = f"http://127.0.0.1:{port}"
        argv_srv = [
            BINARY,
            "-m",
            MODEL,
            "--alias",
            ALIAS,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            *PRODUCTION_FLAGS,
            "--parallel",
            "1",
        ]
        proc = subprocess.Popen(  # noqa: S603
            argv_srv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        try:
            load_seconds = wait_ready(base_url, proc, args.ready_timeout)
            result = one_request(base_url, prompt, args.max_tokens)
            samples.append(
                {
                    "sample": i,
                    "port": port,
                    "load_seconds": round(load_seconds, 3),
                    **result,
                }
            )
            s = samples[-1]
            print(
                f"  sample {i}: load {load_seconds:6.2f}s  "
                f"prefill {s['prompt_ms']:.1f}ms  "
                f"decode {s['predicted_per_second']:.2f} tok/s  "
                f"cache_n={s['cache_n']}"
            )
        finally:
            # Always reap the replica. Leaving one resident would cost the
            # production lane real memory, which is the mistake this work
            # exists to avoid.
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

    def med(key: str) -> float | None:
        vals = sorted(v for v in (s.get(key) for s in samples) if isinstance(v, (int, float)))
        return round(vals[len(vals) // 2], 3) if vals else None

    artifact = {
        "generated_at": datetime.now(UTC).isoformat(),
        "requirement": "AC-PERF-031 / #240 config A: fresh model/runtime start, no reusable prompt/KV state",
        "method": (
            "one fresh llama-server process per sample on its own port; exactly one "
            "request per process; production NORMAL lane untouched throughout"
        ),
        "binary": BINARY,
        "model": MODEL,
        "alias": ALIAS,
        "flags": PRODUCTION_FLAGS + ["--parallel", "1"],
        "flags_note": "copied verbatim from the production lane apart from --port and --parallel 1",
        "repetitions": args.repetitions,
        "max_tokens": args.max_tokens,
        "median": {
            "load_seconds": med("load_seconds"),
            "prompt_ms": med("prompt_ms"),
            "predicted_per_second": med("predicted_per_second"),
        },
        "cache_n_all_zero": all(s.get("cache_n") == 0 for s in samples),
        "samples": samples,
    }
    print(
        f"\nmedian load: {artifact['median']['load_seconds']}s   "
        f"median prefill: {artifact['median']['prompt_ms']}ms   "
        f"median decode: {artifact['median']['predicted_per_second']} tok/s"
    )
    print(f"cache_n == 0 on every sample: {artifact['cache_n_all_zero']}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
