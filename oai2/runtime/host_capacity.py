"""Host physical-memory and swap telemetry, read without extra dependencies.

Why this module exists
----------------------

:mod:`oai2.runtime.admission` takes a :class:`~oai2.runtime.admission.CapacitySnapshot`
from its caller and never reads the machine itself, which is correct for a
deterministic policy core but means something in the process has to supply
real numbers. Before this module the only implementation lived in
``scripts/bench.py::_host_memory``, i.e. inside a measurement script. Anything
that needed the same figures at runtime would have had to either import a
script or copy the parsing, and a second copy of a parser that once silently
zeroed every swap value is a liability, not a convenience.

This module is the single owner of that parsing. ``scripts/bench.py`` now
delegates here so the benchmark and the admission gate cannot disagree about
what the host looked like during a run.

Design points
-------------

- **Fail to ``None``, never to zero.** A missing ``sysctl`` means the value is
  unknown, and an unknown memory figure reported as ``0`` would be read as
  "the host is full" or "the host is empty" depending on which field it was.
  Every unreadable value is ``None``.
- **Swap decimals are matched with a real pattern.** ``vm.swapusage`` reports
  ``total = 5120.00M  used = 4129.94M``; a naive split on whitespace or ``.``
  yields ``5120`` and ``0`` instead of ``5.00`` and ``4.03`` GB. The spacing
  around ``=`` also varies between releases, so the pattern accepts both
  ``used = 4129.94M`` and ``used=4129.94M``.
- **macOS-specific, and honestly so.** ``sysctl``/``vm_stat`` are the Darwin
  path. On another platform every field is ``None`` and the caller is expected
  to deal with that rather than receive a plausible-looking number.

Status: IMPLEMENTED — unit-pinned in ``tests/test_host_capacity.py`` and
exercised live on the production host by ``scripts/bench.py`` and
``scripts/admission_probe.py``.
"""

from __future__ import annotations

import re
import subprocess  # noqa: S404
from typing import Final

_BYTES_PER_GB: Final = 1024.0**3
_MB_PER_GB: Final = 1024.0

#: ``vm_stat`` page classes that count as available to a new allocation.
#: "Free" alone understates what macOS will actually hand out without swapping.
_AVAILABLE_PAGE_LABELS: Final = ("free", "speculative", "purgeable")

_COMMAND_TIMEOUT_SECONDS: Final = 5


def _run(argv: list[str]) -> str | None:
    """Run a read-only host query, returning stdout or ``None`` on any failure."""
    try:
        completed = subprocess.run(  # noqa: S603
            argv,  # noqa: S607
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT_SECONDS,
            check=True,
        )
    except OSError, subprocess.SubprocessError:
        return None
    return completed.stdout.strip()


def _sysctl(name: str) -> str | None:
    return _run(["sysctl", "-n", name])


def _mb(label: str, swap: str) -> float | None:
    """Extract one ``<label> = <number>M`` field from ``vm.swapusage`` output."""
    match = re.search(rf"{label}\s*=\s*([0-9]+(?:\.[0-9]+)?)M", swap)
    return float(match.group(1)) if match else None


def read_host_memory() -> dict[str, float | None]:
    """Return host physical memory and swap in GB, or ``None`` where unreadable.

    Keys are stable so callers can serialise the result directly as the
    ``Memory/swap`` column of a comparison table:

    ``total_gb``, ``used_gb``, ``free_gb``, ``swap_total_gb``, ``swap_used_gb``.
    """
    out: dict[str, float | None] = {
        "total_gb": None,
        "used_gb": None,
        "free_gb": None,
        "swap_total_gb": None,
        "swap_used_gb": None,
    }

    total_raw = _sysctl("hw.memsize")
    if not total_raw or not total_raw.isdigit():
        return out
    out["total_gb"] = round(int(total_raw) / _BYTES_PER_GB, 2)

    vm_stat = _run(["vm_stat"]) or ""
    page_match = re.search(r"page size of (\d+) bytes", vm_stat)
    if page_match:
        page_bytes = int(page_match.group(1))
        available_pages = 0
        for label in _AVAILABLE_PAGE_LABELS:
            label_match = re.search(rf"Pages {label}:\s+(\d+)", vm_stat)
            if label_match:
                available_pages += int(label_match.group(1))
        free_gb = round(available_pages * page_bytes / _BYTES_PER_GB, 2)
        out["free_gb"] = free_gb
        out["used_gb"] = round(int(total_raw) / _BYTES_PER_GB - free_gb, 2)

    swap = _sysctl("vm.swapusage")
    if swap:
        swap_total_mb = _mb("total", swap)
        swap_used_mb = _mb("used", swap)
        if swap_total_mb is not None:
            out["swap_total_gb"] = round(swap_total_mb / _MB_PER_GB, 2)
        if swap_used_mb is not None:
            out["swap_used_gb"] = round(swap_used_mb / _MB_PER_GB, 2)
    return out


__all__ = ["read_host_memory"]
