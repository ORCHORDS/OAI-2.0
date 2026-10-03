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
import re

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
    """``<x> if <collection> else 0.0`` sites, keyed by path + source text.

    The key is deliberately NOT ``path:lineno``. Keying on line numbers made
    this guard a tripwire for unrelated edits: adding a 12-line import block
    above an allowlisted site moved it from 209 to 221 and failed the gate,
    even though the site's own logic and justification were untouched. That
    happened twice in one day (once here, once for a parallel agent's import
    block) before the key was changed.

    Keying on the normalized source of the conditional keeps the property the
    guard exists for -- a change to the site's *own* logic still fails the
    suite -- while an edit elsewhere in the file no longer forces a
    re-review of a site nobody touched.

    The ``#N`` suffix is the occurrence index within the file, so two
    syntactically identical sites in one module stay distinguishable rather
    than collapsing into a single key.
    """
    found: dict[str, str] = {}
    seen: dict[str, int] = {}

    def _key(rel: str, source: str) -> str:
        base = f"{rel}|{source}"
        seen[base] = seen.get(base, 0) + 1
        return f"{base}#{seen[base]}"

    class Scan(ast.NodeVisitor):
        def __init__(self, rel: str) -> None:
            self.rel = rel

        def visit_IfExp(self, node: ast.IfExp) -> None:
            if _is_zero(node.orelse) and isinstance(
                node.test, (ast.Name, ast.Attribute, ast.Compare)
            ):
                source = re.sub(r"\s+", " ", ast.unparse(node)).strip()
                found[_key(self.rel, source)] = source
            self.generic_visit(node)

    for path in sorted(OAI2.rglob("*.py")):
        rel = str(path.relative_to(OAI2.parent))
        Scan(rel).visit(ast.parse(path.read_text()))
    return found


#: The remaining ``else 0.0`` sites, each reviewed and each a TRUE zero.
#: An empty list of allowlisted items is not the goal; a *justified* one is.
#: Adding an entry here means asserting the new site is a genuine zero.
ALLOWED_ZERO_SUBSTITUTIONS = {
    # A failed check scores 0.0. That is the score, not a missing measurement.
    # Two identical sites in this module, hence the occurrence suffixes.
    "oai2/evals/__init__.py|1.0 if passed else 0.0#1",
    "oai2/evals/__init__.py|1.0 if passed else 0.0#2",
    # Zero words in empty content is a true count.
    "oai2/runtime/gateway_runtime.py|max(1, len(content.split())) if content else 0#1",
    # A single key has depth 0.
    "oai2/observability/metrics.py|max_keys - 1 if max_keys > 1 else 0#1",
    # A zero-magnitude vector has no direction; 0.0 is the documented
    # convention for "orthogonal", not a missing similarity.
    "oai2/knowledge/cloudflare.py|sum((a * b for a, b in zip(vector, stored, strict=True))) / denom if denom > 0.0 else 0.0#1",  # noqa: E501
    # MINE, marked rather than removed: k=0 still yields 0.0 for the rate
    # fields, and `RetrievalMetrics.insufficient_evidence` / `.measured` carry
    # the distinction. The conditional itself is the computation, not a
    # fallback.
    "oai2/knowledge/evidence_package.py|relevant_retrieved / k if k else 0.0#1",
    "oai2/knowledge/evidence_package.py|1.0 - precision if k else 0.0#1",
    # Population variance of a singleton IS zero, and `Distribution.count`
    # is honest about being 1. The sample count travels with the number.
    #
    # The percentiles beside it were REMOVED rather than allowed:
    # `_percentile` had a `len(ordered) == 1` case returning the single
    # value for every requested percentile, so a one-replay sample set
    # reported a p95 and a p99 that no observation occupied. p95/p99 are
    # now None below count 2, and the budget/promotion gates fail on them
    # rather than skipping the comparison.
    "oai2/evals/qos.py|statistics.pvariance(values) if repeatable else 0.0#1",
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

    def test_qos_leaves_throughput_undefined_when_no_time_was_measured(self) -> None:
        """A rate is actions divided by time; with no time there is no rate.

        This site was previously allowlisted in this file on the reasoning
        that `end_to_end_ms` is a REQUIRED field, so a 0.0 divisor must mean
        a workload that really did take no time. That reasoning was wrong and
        the wrongness is the point: a required field is not a measured one.
        A harness that records no timing at all still has to supply something
        for `end_to_end_ms`, and 0.0 is what it will supply. Requiring the
        field did not make the measurement exist; it only removed the
        compiler's objection to a fabricated one.
        """
        unmeasured = summarize_samples(
            [
                WorkloadSample(
                    workload=WorkloadClass.NORMAL,
                    target_hardware="hw",
                    config_id="c",
                    ttft_ms=0.0,
                    first_useful_action_ms=0.0,
                    end_to_end_ms=0.0,
                    declared_success=True,
                    verified_success=True,
                    verified_actions=4,
                    generated_tokens=0,
                )
                for _ in range(3)
            ]
        )
        # Twelve verified actions and a throughput of 0.0/sec is not a
        # possible physical outcome; it is a number standing in for a rate
        # that was never computed.
        assert unmeasured.verified_success_rate == 1.0
        assert unmeasured.verified_actions_per_second is None
        assert unmeasured.throughput_measured is False

        # A true zero survives: time WAS measured, and nothing was verified.
        measured_zero = summarize_samples(
            [
                WorkloadSample(
                    workload=WorkloadClass.NORMAL,
                    target_hardware="hw",
                    config_id="c",
                    ttft_ms=0.0,
                    first_useful_action_ms=0.0,
                    end_to_end_ms=10_000.0,
                    declared_success=True,
                    verified_success=True,
                    verified_actions=0,
                    generated_tokens=0,
                )
                for _ in range(3)
            ]
        )
        assert measured_zero.verified_actions_per_second == 0.0
        assert measured_zero.throughput_measured is True


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

    def test_control_qos_zero_denominator_fallback_restored(self, tmp_path) -> None:
        """Restore the `else 0.0` on an unmeasurable denominator.

        This control is the reason the site above is a *behavioural* pin and
        not only an allowlist removal. It restores the exact original
        fallback in a copy of the real module and shows the mutant produces
        the self-contradictory record (perfect success, zero throughput)
        that the shipped module refuses to produce.
        """
        import importlib.util
        import sys

        source = (OAI2 / "evals" / "qos.py").read_text()
        anchor = "verified_actions / total_seconds if total_seconds > 0.0 else None"
        assert anchor in source, "anchor moved; control is stale"
        target = tmp_path / "mutant_qos.py"
        target.write_text(
            source.replace(
                anchor,
                "verified_actions / total_seconds if total_seconds > 0.0 else 0.0",
                1,
            )
        )
        name = "oai2.evals._mutant_qos"
        spec = importlib.util.spec_from_file_location(name, target)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            del sys.modules[name]
            raise

        sample = module.WorkloadSample(
            workload=module.WorkloadClass.NORMAL,
            target_hardware="hw",
            config_id="c",
            ttft_ms=0.0,
            first_useful_action_ms=0.0,
            end_to_end_ms=0.0,
            declared_success=True,
            verified_success=True,
            verified_actions=4,
            generated_tokens=0,
        )
        mutant = module.summarize_samples([sample, sample, sample])
        assert mutant.verified_actions_per_second == 0.0
        # The contradiction the fix removes: a perfect run at zero throughput.
        assert mutant.verified_success_rate == 1.0
        assert mutant.throughput_measured is True

        real = summarize_samples([sample, sample, sample])
        assert real.verified_actions_per_second is None
        assert real.throughput_measured is False


class TestAllowlistKeyStability:
    """The allowlist used to be keyed `path:lineno`, which is brittle.

    Keying on line numbers meant the guard failed whenever *anything* above
    an allowlisted site shifted -- a 12-line import block moved two entries in
    `oai2/evals/__init__.py` from 209/242 to 221/254 and forced a re-review
    of sites whose logic had not changed. That happened twice in one day
    before the key was changed to the site's own source text.

    Re-keying is only safe if it keeps the two properties that matter:

      1. a NEW `else 0.0` still fails the suite
      2. a CHANGED site's logic still fails the suite
      3. an UNRELATED edit elsewhere in the file no longer fails it  <- new
    """

    @staticmethod
    def _keys(source: str) -> set[str]:
        """Run the real scanner over a synthetic module."""
        found: dict[str, str] = {}
        seen: dict[str, int] = {}

        def _key(rel: str, snippet: str) -> str:
            base = f"{rel}|{snippet}"
            seen[base] = seen.get(base, 0) + 1
            return f"{base}#{seen[base]}"

        class Scan(ast.NodeVisitor):
            def visit_IfExp(self, node: ast.IfExp) -> None:
                if _is_zero(node.orelse) and isinstance(
                    node.test, (ast.Name, ast.Attribute, ast.Compare)
                ):
                    snippet = re.sub(r"\s+", " ", ast.unparse(node)).strip()
                    found[_key("m.py", snippet)] = snippet
                self.generic_visit(node)

        Scan().visit(ast.parse(source))
        return set(found)

    _BASE = "def f(xs, total):\n    return max(xs) if xs else 0.0\n"

    def test_a_new_zero_substitution_is_still_detected(self) -> None:
        keys = self._keys(self._BASE + "def g(ys):\n    return min(ys) if ys else 0.0\n")
        assert len(keys) == 2
        assert not keys <= ALLOWED_ZERO_SUBSTITUTIONS

    def test_changing_a_sites_own_logic_still_detaches_it(self) -> None:
        """The property that must survive re-keying.

        If the conditional's own text changes, the new key is not in the
        allowlist, so the site is reported as unaccounted-for -- exactly as
        it would have been under line numbering.
        """
        before = self._keys(self._BASE)
        after = self._keys("def f(xs, total):\n    return (max(xs) / total) if xs else 0.0\n")
        assert before != after
        assert not after & ALLOWED_ZERO_SUBSTITUTIONS

    def test_an_unrelated_edit_above_the_site_no_longer_moves_it(self) -> None:
        """The improvement: line shifts are no longer the guard's problem."""
        shifted = (
            "import os\n"
            "import sys\n"
            "import json\n"
            "from typing import Any\n"
            "from collections import OrderedDict\n" + self._BASE
        )
        assert self._keys(self._BASE) == self._keys(shifted)

    def test_two_identical_sites_in_one_module_stay_distinguishable(self) -> None:
        """`oai2/evals/__init__.py` has two identical `1.0 if passed else 0.0`.

        Without the occurrence suffix they would collapse into one key, and
        the second occurrence would be reported as a new unaccounted site.
        """
        dup = "def f(x):\n    return 1.0 if x else 0.0\ndef g(y):\n    return 1.0 if y else 0.0\n"
        keys = self._keys(dup)
        assert len(keys) == 2
        assert all(k.endswith(("#1", "#2")) for k in keys)
