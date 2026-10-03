"""Unit coverage for the host memory/swap telemetry owner.

These tests pin the parsing, not the machine: every host query is stubbed, so
the suite is deterministic and does not depend on how loaded this host is.

The three cases that matter most are the ones that previously produced
*plausible but wrong* numbers rather than an obvious crash -- a lost swap
decimal, a spacing change between macOS releases, and an unreadable ``sysctl``
that must degrade to ``None`` instead of ``0``.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from oai2.runtime import host_capacity
from oai2.runtime.host_capacity import read_host_memory

MEMSIZE = "68719476736"  # 64 GiB
VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               100000.
Pages active:                            200000.
Pages inactive:                           50000.
Pages speculative:                        20000.
Pages purgeable:                          10000.
"""
SWAP_SPACED = "total = 5120.00M  used = 4129.94M  free = 990.06M  (encrypted)"
SWAP_TIGHT = "total = 5120.00M  used=4129.94M  free=990.06M  (encrypted)"


def _stub(monkeypatch, *, memsize=MEMSIZE, vm_stat=VM_STAT, swap=SWAP_SPACED) -> None:
    def fake_run(argv, **_kwargs):
        joined = " ".join(argv)
        if joined == "sysctl -n hw.memsize":
            return SimpleNamespace(stdout=memsize)
        if joined == "vm_stat":
            return SimpleNamespace(stdout=vm_stat)
        if joined == "sysctl -n vm.swapusage":
            return SimpleNamespace(stdout=swap)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_reads_total_used_free_and_swap(monkeypatch) -> None:
    _stub(monkeypatch)
    result = read_host_memory()
    assert result["total_gb"] == 64.0
    # 100000 free + 20000 speculative + 10000 purgeable = 130000 pages of 16384
    assert result["free_gb"] == pytest.approx(1.98, abs=0.01)
    assert result["used_gb"] == pytest.approx(62.02, abs=0.02)
    assert result["swap_total_gb"] == 5.0
    assert result["swap_used_gb"] == pytest.approx(4.03, abs=0.01)


def test_swap_decimals_survive_both_spacing_forms(monkeypatch) -> None:
    """`used = 4129.94M` and `used=4129.94M` must parse identically.

    Only a real pattern can do this. A whitespace split happens to work on the
    spaced form and silently yields 0 on the tight one.
    """
    _stub(monkeypatch, swap=SWAP_SPACED)
    spaced = read_host_memory()
    _stub(monkeypatch, swap=SWAP_TIGHT)
    tight = read_host_memory()
    assert spaced["swap_used_gb"] == tight["swap_used_gb"] == pytest.approx(4.03, abs=0.01)
    assert tight["swap_used_gb"] != 0.0


def test_unreadable_sysctl_yields_none_not_zero(monkeypatch) -> None:
    """A missing query is unknown. Reporting 0 would read as "the host is full"."""

    def boom(*_args, **_kwargs):
        raise OSError("no sysctl here")

    monkeypatch.setattr(subprocess, "run", boom)
    result = read_host_memory()
    assert result == {
        "total_gb": None,
        "used_gb": None,
        "free_gb": None,
        "swap_total_gb": None,
        "swap_used_gb": None,
    }


def test_non_numeric_memsize_is_not_treated_as_a_total(monkeypatch) -> None:
    _stub(monkeypatch, memsize="not-a-number")
    result = read_host_memory()
    assert result["total_gb"] is None
    assert result["used_gb"] is None


def test_vm_stat_without_a_page_size_leaves_used_unknown(monkeypatch) -> None:
    _stub(monkeypatch, vm_stat="Mach Virtual Memory Statistics:")
    result = read_host_memory()
    assert result["total_gb"] == 64.0
    assert result["free_gb"] is None
    assert result["used_gb"] is None


def test_missing_swap_output_keeps_memory_but_drops_swap(monkeypatch) -> None:
    _stub(monkeypatch, swap="")
    result = read_host_memory()
    assert result["total_gb"] == 64.0
    assert result["used_gb"] is not None
    assert result["swap_total_gb"] is None
    assert result["swap_used_gb"] is None


def test_result_keys_are_stable_for_table_serialisation(monkeypatch) -> None:
    """Callers serialise this straight into a Memory/swap column, so the keys
    must exist on every path, including total failure."""
    expected = {"total_gb", "used_gb", "free_gb", "swap_total_gb", "swap_used_gb"}
    _stub(monkeypatch)
    assert set(read_host_memory()) == expected

    def boom(*_args, **_kwargs):
        raise OSError("no sysctl here")

    monkeypatch.setattr(subprocess, "run", boom)
    assert set(read_host_memory()) == expected


def test_bench_delegates_to_the_single_owner(monkeypatch) -> None:
    """``scripts/bench.py`` must not keep its own copy of the parser.

    Compared under a stubbed host: two live reads a moment apart legitimately
    differ, and that would make this assert nothing about delegation.
    """
    import importlib.util
    import sys
    from pathlib import Path

    _stub(monkeypatch)
    repo = Path(host_capacity.__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("bench_mod", repo / "scripts" / "bench.py")
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    # dataclass resolution needs the module registered before execution.
    sys.modules["bench_mod"] = bench
    try:
        spec.loader.exec_module(bench)
    finally:
        del sys.modules["bench_mod"]

    assert bench._host_memory() == read_host_memory()
    # And the body really is a delegation, not a re-implementation.
    body = (repo / "scripts" / "bench.py").read_text()
    assert "return read_host_memory()" in body
    assert "vm.swapusage" not in body, "bench.py still parses swap itself"
