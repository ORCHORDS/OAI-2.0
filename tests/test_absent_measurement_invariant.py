"""An absent measurement must never render as a clean one.

A repository-wide invariant, pinned in one place so a new metric surface is
added to a list rather than reinvented.

THE FAILURE MODE
----------------
Five independent sites in this repository shipped the same defect, each in a
different subsystem, each found separately:

| Site | What an absence rendered as |
|---|---|
| `oai2/model/numerical_compare.py` | `max_abs_error=0.0` for a comparison that never happened (`no_comparable_samples`) |
| `oai2/model/numerical_compare.py` | `capability_regression=0.0` where capability was never measured (`capability_measured`) |
| `oai2/knowledge/evidence_package.py` | `precision_at_k=0.0`, `recall=0.0`, `irrelevant_context_rate=0.0` for a retrieval that returned nothing |
| `oai2/evals/qos.py` | `count=40` tool-latency measurements, `p95=0.00`, for a workload that never invoked a tool |
| `oai2/runtime/residency.py` | mean/p50/p95/p99/max wait and residency all exactly 0.0, from a tracker that admitted nothing |

Two of those five are *worse* than an undefined value, and that is what makes
the family worth a guard rather than a habit:

- an **unmeasured** rate reported as **0.0** is the best value any latency
  budget can be handed, so a `p95 <= X` check passes trivially;
- a **count** reported as N when zero observations occurred is not "unmeasured"
  at all, it is **false**, and it sits beside an honest count in the same
  report with the same shape.

The unifying question every one of them fails is: *could a reader tell, from
this record alone, whether anything was measured?* Where the answer is no, the
number is a claim the system cannot support.

WHAT IS *NOT* A VIOLATION
------------------------
A genuine zero is not the defect. `1.0 if passed else 0.0` on a score,
`max(1, words) if content else 0` on a word count, `max_keys - 1` on a
cardinality, and a zero-magnitude vector's similarity are all true zeros, and
a test that forbade them would be a test nobody could keep.

So this file pins the surfaces that were wrong, asserts the corrected
behaviour, and carries a negative control proving the assertions bite.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from oai2.evals.qos import WorkloadClass, WorkloadSample, summarize_samples
from oai2.knowledge.evidence_package import evaluate_retrieval_package
from oai2.runtime.residency import ResidencyAccountant
from oai2.runtime.service_binding import BatchSurfaceMetrics

OAI2 = pathlib.Path(__file__).resolve().parents[1] / "oai2"


def _is_zero(node: ast.AST) -> bool:
    """A zero literal, excluding bools.

    ``False == 0`` is True in Python, so a naive membership test against
    ``(0.0, 0)`` matches every ``return False`` in the tree. An earlier
    version of this scan did exactly that and reported 31 sites, nearly all
    of them ``return False`` from validators.
    """
    return (
        isinstance(node, ast.Constant)
        and not isinstance(node.value, bool)
        and isinstance(node.value, (int, float))
        and node.value == 0
    )


def _empty_substitutions() -> dict[str, str]:
    """``<x> if <collection> else 0.0`` sites, keyed ``path:lineno``."""
    found: dict[str, str] = {}

    class Scan(ast.NodeVisitor):
        def __init__(self, path: str) -> None:
            self.path = path

        def visit_IfExp(self, node: ast.IfExp) -> None:
            if _is_zero(node.orelse) and isinstance(
                node.test, (ast.Name, ast.Attribute, ast.Compare)
            ):
                found[f"{self.path}:{node.lineno}"] = ast.unparse(node)[:90]
            self.generic_visit(node)

    for path in sorted(OAI2.rglob("*.py")):
        Scan(str(path.relative_to(OAI2.parent))).visit(ast.parse(path.read_text()))
    return found


#: The remaining ``else 0.0`` sites, each reviewed and each a TRUE zero.
#: An empty list of allowlisted items is not the goal; a *justified* one is.
#: Adding an entry here means asserting the new site is a genuine zero.
ALLOWED_ZERO_SUBSTITUTIONS = {
    # A failed check scores 0.0. That is the score, not a missing measurement.
    "oai2/evals/__init__.py:209",
    "oai2/evals/__init__.py:242",
    # Zero words in empty content is a true count.
    "oai2/runtime/gateway_runtime.py:467",
    # A single key has depth 0.
    "oai2/observability/metrics.py:225",
    # A zero-magnitude vector has no direction; 0.0 is the documented
    # convention for "orthogonal", not a missing similarity.
    "oai2/knowledge/cloudflare.py:314",
    # MINE, marked rather than removed: k=0 still yields 0.0 for the rate
    # fields, and `RetrievalMetrics.insufficient_evidence` / `.measured` carry
    # the distinction. The conditional itself is the computation, not a
    # fallback.
    "oai2/knowledge/evidence_package.py:211",
    "oai2/knowledge/evidence_package.py:213",
    # Population variance of a singleton IS zero, and `Distribution.count`
    # is honest about being 1. The sample count travels with the number.
    "oai2/evals/qos.py:222",
    # Throughput over zero elapsed time is zero actions per second. The
    # divisor is derived from `end_to_end_ms`, which is a REQUIRED field on
    # every sample, so this is a workload that really did take no time --
    # not a measurement that was skipped.
    "oai2/evals/qos.py:278",
}


class TestNoUnreviewedZeroSubstitutions:
    def test_every_zero_substitution_is_accounted_for(self) -> None:
        """A new ``else 0.0`` must be reviewed before it lands.

        This is the durable half of the audit. The five defects were each
        found by reading; this makes the sixth one fail the suite instead,
        and forces whoever adds it to state whether the zero is real.
        """
        found = _empty_substitutions()
        unaccounted = set(found) - ALLOWED_ZERO_SUBSTITUTIONS
        assert not unaccounted, (
            "new zero-for-no-observations substitution(s); each needs review -- "
            "if the zero is genuine, add it to ALLOWED_ZERO_SUBSTITUTIONS with "
            "a reason, otherwise it is the absence-renders-as-a-measurement "
            "defect again:\n  "
            + "\n  ".join(f"{k}  {found[k]}" for k in sorted(unaccounted))
        )

    def test_the_allowlist_has_no_stale_entries(self) -> None:
        """An allowlist that outlives its site rots into a blanket pass."""
        stale = ALLOWED_ZERO_SUBSTITUTIONS - set(_empty_substitutions())
        assert not stale, (
            "allowlist entries no longer match any site; remove them:\n  "
            + "\n  ".join(sorted(stale))
        )

    def test_the_scanner_actually_detects_the_pattern(self) -> None:
        """A guard that cannot see its own target is not a guard."""
        source = "def f(xs):\n    return max(xs) if xs else 0.0\n"
        tree = ast.parse(source)

        class Scan(ast.NodeVisitor):
            seen: list[str] = []

            def visit_IfExp(self, node: ast.IfExp) -> None:
                if _is_zero(node.orelse) and isinstance(node.test, ast.Name):
                    Scan.seen.append(ast.unparse(node))
                self.generic_visit(node)

        Scan().visit(tree)
        assert Scan.seen == ["max(xs) if xs else 0.0"]

    def test_the_scanner_does_not_match_return_false(self) -> None:
        """The bug that made the first version of this scan useless."""
        tree = ast.parse("def f(x):\n    if not x:\n        return False\n")
        found: list[str] = []

        class Scan(ast.NodeVisitor):
            def visit_If(self, node: ast.If) -> None:
                t = node.test
                if (
                    isinstance(t, ast.UnaryOp)
                    and isinstance(t.op, ast.Not)
                    and isinstance(t.operand, ast.Name)
                    and len(node.body) == 1
                ):
                    r = node.body[0]
                    if isinstance(r, ast.Return) and r.value is not None and _is_zero(r.value):
                        found.append("match")
                self.generic_visit(node)

        Scan().visit(tree)
        assert found == [], "`return False` must not read as `return 0`"


class TestAbsenceIsReportedAsAbsence:
    """The corrected behaviour, surface by surface."""

    def test_residency_reports_no_waits_when_nothing_was_admitted(self) -> None:
        summary = ResidencyAccountant().summary()
        assert summary.total == 0
        for name in (
            "mean_wait_ms",
            "p50_wait_ms",
            "p95_wait_ms",
            "p99_wait_ms",
            "max_wait_ms",
            "mean_residency_ms",
            "max_residency_ms",
        ):
            assert getattr(summary, name) is None, (
                f"{name} claims a measured distribution from a tracker that "
                f"admitted nothing"
            )

    def test_batch_surface_reports_no_averages_before_any_batch(self) -> None:
        metrics = BatchSurfaceMetrics(
            queue_depth=0,
            queued_sessions=0,
            cancelled_requests=0,
            batches_emitted=0,
            requests_emitted=0,
            last_batch_size=0,
            mean_batch_size=None,
            last_queue_wait_ms=None,
            mean_queue_wait_ms=None,
            max_queue_wait_ms=None,
            active_sessions=0,
        )
        assert metrics.mean_batch_size is None
        assert metrics.mean_queue_wait_ms is None

    def test_retrieval_reports_no_measurement_when_nothing_was_retrieved(
        self,
    ) -> None:
        from oai2.knowledge.abstraction import RetrievalResult
        from oai2.knowledge.evidence_package import build_evidence_package

        package = build_evidence_package(
            RetrievalResult(topic="t", objects=(), candidates=()),
            token_budget=200,
            token_counter=lambda text: len(text.split()),
        )
        metrics = evaluate_retrieval_package(
            package,
            relevant_knowledge_ids=["k1"],
            raw_source_tokens=100,
            task_success_with_retrieval=0.0,
            task_success_without_retrieval=0.0,
        )
        assert metrics.k == 0
        assert metrics.insufficient_evidence is True
        assert metrics.measured is False

    def test_qos_reports_no_distribution_for_an_unexercised_component(self) -> None:
        report = summarize_samples(
            [
                WorkloadSample(
                    workload=WorkloadClass.NORMAL,
                    target_hardware="hw",
                    config_id="c",
                    ttft_ms=100.0,
                    first_useful_action_ms=200.0,
                    end_to_end_ms=900.0,
                    declared_success=True,
                    verified_success=True,
                    verified_actions=1,
                    generated_tokens=0,
                )
                for _ in range(40)
            ]
        )
        assert report.sample_count == 40
        for name in (
            "decode_tokens_per_second",
            "tool_ms",
            "retrieval_ms",
            "vision_ms",
            "build_test_ms",
        ):
            assert getattr(report, name) is None, (
                f"{name} claims a distribution for a component that never ran"
            )
        # The honest components are untouched.
        assert report.ttft_ms.count == 40


class TestInvariantNegativeControls:
    """Mutate a surface; the guards above must fail."""

    def test_control_residency_zero_fallback_restored(self, tmp_path) -> None:
        import importlib.util
        import sys

        path = OAI2 / "runtime" / "residency.py"
        source = path.read_text()
        anchor = "    if not values:\n        return None\n    return float(statistics.fmean(values))"
        assert anchor in source, "anchor moved; control is stale"
        target = tmp_path / "mutant_residency.py"
        target.write_text(source.replace(anchor, anchor.replace("return None", "return 0.0"), 1))
        name = "oai2.runtime._mutant_residency"
        spec = importlib.util.spec_from_file_location(name, target)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            del sys.modules[name]
            raise

        summary = module.ResidencyAccountant().summary()
        assert summary.mean_wait_ms == 0.0, "mutation did not restore the zero"
        with pytest.raises(AssertionError):
            assert summary.mean_wait_ms is None

    def test_control_percentile_validation_bypass_restored(self, tmp_path) -> None:
        """An out-of-range percentile on an empty list must still raise.

        This was a real second bug in the same function. The range check sat
        AFTER the emptiness check, so ``_percentile([], 500.0)`` returned 0.0
        instead of raising: the validation was skipped on exactly the path
        that fabricates a number. The control restores that ordering and
        shows it accepts an argument it must reject.
        """
        import importlib.util
        import sys

        import oai2.runtime.residency as real

        source = (OAI2 / "runtime" / "residency.py").read_text()
        ordered = (
            '    if not 0.0 <= percentile <= 100.0:\n'
            '        raise ValueError("percentile must be between 0 and 100")\n'
            "    if not values:\n"
            "        return None\n"
        )
        assert ordered in source, "anchor moved; control is stale"
        target = tmp_path / "mutant_residency2.py"
        # Put the emptiness check back in front of the range check.
        target.write_text(
            source.replace(
                ordered,
                "    if not values:\n        return None\n" + ordered.rstrip("\n") + "\n",
                1,
            )
        )
        name = "oai2.runtime._mutant_residency2"
        spec = importlib.util.spec_from_file_location(name, target)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            del sys.modules[name]
            raise

        # The mutation bites: the mutant swallows an impossible percentile and
        # hands back a number, while the real module refuses it.
        assert module._percentile([], 500.0) is None
        with pytest.raises(ValueError):
            real._percentile([], 500.0)
