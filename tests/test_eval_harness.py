"""End-to-end tests for ``oai2.evals.run_eval_harness``.

The harness is the public entry point that drives the builtin
capability suites using :func:`oai2.runtime.select_runtime_from_env`.
These tests pin both the placeholder path (no API key) and the gateway
path (key set, MockTransport). They do not require a live token — every
test is offline.

Coverage:
* placeholder path runs every builtin suite
* placeholder path filters by ``suite_names``
* placeholder path raises ``KeyError`` on unknown suite name
* placeholder path aggregates ``pass_rate`` across all suites
* placeholder path forwards ``max_tokens`` to each per-suite run
* gateway path runs against a prebuilt MockTransport client end-to-end
  and reports ``runtime="GatewayRuntime"`` rather than the placeholder
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from oai2.evals import (
    BUILTIN_SUITES_NAMES,
    SCORERS,
    CapabilityCase,
    CapabilityScore,
    CapabilitySuite,
    HarnessReport,
    SuiteReport,
    builtin_suite,
    builtin_suites,
    run_eval_harness,
    run_suite,
)
from oai2.runtime import (
    GatewayConfig,
    GatewayRuntime,
    InferenceRequest,
    InferenceRuntime,
    PlaceholderRuntime,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patched_selector(monkeypatch: pytest.MonkeyPatch, runtime):
    """Replace ``oai2.runtime.inference.select_runtime_from_env`` with a fixed runtime.

    Using a stub selector (not real env vars) keeps the harness tests
    hermetic: no need to mutate ``OAI2_GATEWAY_API_KEY`` per case.
    The harness resolves the symbol lazily via a local import, so we
    patch the canonical source — ``oai2.runtime.inference`` — rather
    than ``oai2.evals`` (which has no module-level reference).
    """
    monkeypatch.setattr(
        "oai2.runtime.inference.select_runtime_from_env", lambda: runtime
    )


def _chat_completion(text: str = "ok") -> httpx.Response:
    """Canned OpenAI chat-completions response for MockTransport."""
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-harness",
            "object": "chat.completion",
            "created": 0,
            "model": "oai-2.0",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        },
    )


def _gateway_runtime_with(handler) -> GatewayRuntime:
    """Build a GatewayRuntime whose httpx.Client uses the supplied handler."""
    cfg = GatewayConfig(
        base_url="https://gateway.example.test",
        api_key="smoke-token-xyz",
        model="oai-2.0",
        timeout_seconds=5.0,
    )
    client = httpx.Client(
        base_url=cfg.base_url,
        transport=httpx.MockTransport(handler),
        headers={"Authorization": f"Bearer {cfg.api_key}"},
    )
    return GatewayRuntime(cfg, client=client)


# ---------------------------------------------------------------------------
# Placeholder path — no API key set
# ---------------------------------------------------------------------------


def test_run_eval_harness_with_placeholder_runs_all_suites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rt = PlaceholderRuntime()
    _patched_selector(monkeypatch, rt)

    report = run_eval_harness()

    assert isinstance(report, HarnessReport)
    assert report.runtime == "PlaceholderRuntime"
    # One SuiteReport per builtin suite (None => all).
    assert len(report.reports) == len(BUILTIN_SUITES_NAMES)
    # Each SuiteReport records the runtime class name.
    for sub in report.reports:
        assert sub.runtime == "PlaceholderRuntime"
        assert sub.n_cases >= 1


def test_run_eval_harness_with_placeholder_filters_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rt = PlaceholderRuntime()
    _patched_selector(monkeypatch, rt)

    report = run_eval_harness(suite_names=["coding"])

    assert len(report.reports) == 1
    assert report.reports[0].suite_id == "coding_basic"
    assert report.reports[0].runtime == "PlaceholderRuntime"


def test_run_eval_harness_with_placeholder_unknown_suite_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rt = PlaceholderRuntime()
    _patched_selector(monkeypatch, rt)

    with pytest.raises(KeyError):
        run_eval_harness(suite_names=["not-a-real-suite"])


def test_run_eval_harness_aggregates_pass_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rt = PlaceholderRuntime()
    _patched_selector(monkeypatch, rt)

    report = run_eval_harness()

    # The aggregator must mirror the sum of individual SuiteReports.
    expected_cases = sum(r.n_cases for r in report.reports)
    expected_passed = sum(r.n_passed for r in report.reports)
    assert report.n_cases == expected_cases
    assert report.n_passed == expected_passed
    if expected_cases == 0:
        assert report.pass_rate == 0.0
    else:
        assert report.pass_rate == expected_passed / expected_cases


def test_run_eval_harness_forwards_max_tokens_to_each_suite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``max_tokens`` must flow into the default request factory for every suite."""
    rt = PlaceholderRuntime()

    captured_max_tokens: list[int] = []

    real_run_suite = run_suite

    def patched_run_suite(suite, runtime, *, request_factory=None, max_tokens=256):
        factory = request_factory or (
            lambda case: InferenceRequest(prompt=case.prompt, max_tokens=max_tokens)
        )
        first_req = factory(suite.cases[0])
        captured_max_tokens.append(first_req.max_tokens)
        return real_run_suite(
            suite, runtime, request_factory=request_factory, max_tokens=max_tokens
        )

    monkeypatch.setattr("oai2.evals.run_suite", patched_run_suite)
    _patched_selector(monkeypatch, rt)

    run_eval_harness(suite_names=["coding", "reasoning"], max_tokens=64)

    # Every captured request used the harness-level max_tokens (64), not
    # run_suite's default of 256.
    assert captured_max_tokens
    assert all(mt == 64 for mt in captured_max_tokens)


# ---------------------------------------------------------------------------
# Gateway path — API key set, no live token used
# ---------------------------------------------------------------------------


def test_run_eval_harness_with_gateway_runs_against_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the selector returns a GatewayRuntime, the harness drives it.

    The harness's contract is "calls ``close()`` on a
    :class:`GatewayRuntime` before returning". We pin that contract with a
    spy that records the call without actually closing the (test-injected)
    client — :class:`GatewayRuntime`'s own ``close()`` is a no-op when the
    client was passed in, so the spy has to live elsewhere.
    """
    rt = _gateway_runtime_with(lambda req: _chat_completion("from-mock"))
    close_calls: list[bool] = []
    real_close = rt.close

    def spy_close() -> None:
        close_calls.append(True)
        real_close()

    rt.close = spy_close  # type: ignore[method-assign]
    _patched_selector(monkeypatch, rt)

    report = run_eval_harness(suite_names=["coding"])

    assert close_calls, "harness did not call close() on the GatewayRuntime"
    assert report.runtime == "GatewayRuntime"
    assert len(report.reports) == 1
    assert report.reports[0].suite_id == "coding_basic"
    assert report.reports[0].runtime == "GatewayRuntime"
    # The gateway returned "from-mock" for every prompt — that text
    # won't match the expected patterns, but the harness must still
    # produce a score per case.
    assert report.reports[0].n_cases >= 1


# ---------------------------------------------------------------------------
# Sanity: builtin_suite() is reachable through the same lens
# ---------------------------------------------------------------------------


def test_run_eval_harness_suite_names_iterable_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The suite_names parameter accepts any iterable, not just lists."""
    rt = PlaceholderRuntime()
    _patched_selector(monkeypatch, rt)

    report = run_eval_harness(suite_names=("coding", "vision"))

    assert len(report.reports) == 2
    suite_ids = {sub.suite_id for sub in report.reports}
    assert suite_ids == {"coding_basic", "vision_basic"}


def test_run_eval_harness_matches_individual_run_suite_pass_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The harness pass-rate must equal the sum of individual run_suite reports."""
    rt = PlaceholderRuntime()
    _patched_selector(monkeypatch, rt)

    harness_report = run_eval_harness(suite_names=["coding", "reasoning"])
    individual_reports = [
        run_suite(builtin_suite(name), rt) for name in ["coding", "reasoning"]
    ]

    individual_pass_rate = (
        0.0
        if sum(r.n_cases for r in individual_reports) == 0
        else sum(r.n_passed for r in individual_reports)
        / sum(r.n_cases for r in individual_reports)
    )
    assert harness_report.pass_rate == individual_pass_rate


# ---------------------------------------------------------------------------
# continue_on_error fault isolation (slice 14)
# ---------------------------------------------------------------------------


def _flaky_first_call(
    real_run_suite: Callable[..., SuiteReport], exc: Exception
) -> Callable[..., SuiteReport]:
    """Return a ``run_suite`` wrapper that raises ``exc`` only on the first call.

    Subsequent calls delegate to ``real_run_suite``. Used by the
    ``continue_on_error`` tests below to simulate one failing suite in
    a builtin suite list.
    """

    state = {"calls": 0}

    def wrapper(
        suite: CapabilitySuite,
        runtime: InferenceRuntime,
        *,
        request_factory: Callable[[CapabilityCase], InferenceRequest] | None = None,
        max_tokens: int = 256,
    ) -> SuiteReport:
        state["calls"] += 1
        if state["calls"] == 1:
            raise exc
        return real_run_suite(
            suite,
            runtime,
            request_factory=request_factory,
            max_tokens=max_tokens,
        )

    return wrapper


def test_run_eval_harness_continue_on_error_runs_all_suites_when_one_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``continue_on_error=True`` runs every builtin suite even when one raises."""
    rt = PlaceholderRuntime()
    _patched_selector(monkeypatch, rt)

    flaky = _flaky_first_call(run_suite, RuntimeError("first-suite boom"))
    monkeypatch.setattr("oai2.evals.run_suite", flaky)

    report = run_eval_harness(continue_on_error=True)

    assert len(report.reports) == len(BUILTIN_SUITES_NAMES)
    errored = [sub for sub in report.reports if sub.error is not None]
    successful = [sub for sub in report.reports if sub.error is None]
    assert len(errored) == 1
    assert len(successful) == len(BUILTIN_SUITES_NAMES) - 1


def test_run_eval_harness_continue_on_error_surfaces_error_in_subreport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The captured exception is recorded as ``f"{ExcType}: {msg}"`` on the SuiteReport."""
    rt = PlaceholderRuntime()
    _patched_selector(monkeypatch, rt)

    flaky = _flaky_first_call(run_suite, RuntimeError("first-suite boom"))
    monkeypatch.setattr("oai2.evals.run_suite", flaky)

    report = run_eval_harness(continue_on_error=True)
    errored = [sub for sub in report.reports if sub.error is not None]
    assert len(errored) == 1
    failing = errored[0]
    assert failing.error == "RuntimeError: first-suite boom"
    assert failing.n_cases == 0
    assert failing.n_passed == 0
    assert failing.pass_rate == 0.0
    assert failing.scores == []


def test_run_eval_harness_default_raises_on_first_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default behavior (``continue_on_error=False``) propagates the first per-suite error."""
    rt = PlaceholderRuntime()
    _patched_selector(monkeypatch, rt)

    flaky = _flaky_first_call(run_suite, RuntimeError("first-suite boom"))
    monkeypatch.setattr("oai2.evals.run_suite", flaky)

    with pytest.raises(RuntimeError, match="first-suite boom"):
        run_eval_harness()


def test_run_eval_harness_default_propagates_runtime_error_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A custom runtime-error subclass propagates as-is when ``continue_on_error`` is unset."""

    class BoomError(RuntimeError):
        pass

    rt = PlaceholderRuntime()
    _patched_selector(monkeypatch, rt)

    flaky = _flaky_first_call(run_suite, BoomError("specific subclass"))
    monkeypatch.setattr("oai2.evals.run_suite", flaky)

    with pytest.raises(BoomError, match="specific subclass"):
        run_eval_harness()


# ---------------------------------------------------------------------------
# Pass 9 — pin identity for every entry of ``oai2.evals.__all__``
# ---------------------------------------------------------------------------


def test_evals_full_package_identity() -> None:
    """Pin ``is`` identity for every entry of ``oai2.evals.__all__``.

    The same coverage-gap pattern that Pass 8 closed in ``oai2.model``
    (commit ``985d357`` for #216) was open here: only 8 of the 35
    entries were exercised via header imports, leaving
    :mod:`oai2.evals.regression`, :mod:`oai2.evals.truth`, and
    :mod:`oai2.evals.truth_runner` un-pinned. This test asserts ``is``
    identity for every entry of the package surface against its
    defining source module so a future drop or rename in either side
    trips the test.
    """
    from oai2.evals import BUILTIN_SUITES_NAMES as ExportedBUILTIN_SUITES_NAMES
    from oai2.evals import SCORERS as ExportedSCORERS
    from oai2.evals import CandidateRunner as ExportedCandidateRunner
    from oai2.evals import CandidateTruthInput as ExportedCandidateTruthInput
    from oai2.evals import CandidateTruthResponse as ExportedCandidateTruthResponse
    from oai2.evals import CapabilityCase as ExportedCapabilityCase
    from oai2.evals import CapabilityRegression as ExportedCapabilityRegression
    from oai2.evals import (
        CapabilityRegressionThreshold as ExportedCapabilityRegressionThreshold,
    )
    from oai2.evals import CapabilityScore as ExportedCapabilityScore
    from oai2.evals import CapabilitySuite as ExportedCapabilitySuite
    from oai2.evals import HarnessReport as ExportedHarnessReport
    from oai2.evals import (
        HeldOutPromotionBudget as ExportedHeldOutPromotionBudget,
    )
    from oai2.evals import (
        HeldOutPromotionEvaluation as ExportedHeldOutPromotionEvaluation,
    )
    from oai2.evals import HiddenVerifier as ExportedHiddenVerifier
    from oai2.evals import SuiteReport as ExportedSuiteReport
    from oai2.evals import (
        TruthCandidatePromotionEvaluation as ExportedTruthCandidatePromotionEvaluation,
    )
    from oai2.evals import TruthCase as ExportedTruthCase
    from oai2.evals import TruthCaseClass as ExportedTruthCaseClass
    from oai2.evals import TruthOutcome as ExportedTruthOutcome
    from oai2.evals import (
        TruthPromotionBudget as ExportedTruthPromotionBudget,
    )
    from oai2.evals import (
        TruthPromotionEvaluation as ExportedTruthPromotionEvaluation,
    )
    from oai2.evals import TruthReport as ExportedTruthReport
    from oai2.evals import TruthRunResult as ExportedTruthRunResult
    from oai2.evals import TruthSample as ExportedTruthSample
    from oai2.evals import TruthVerifierVerdict as ExportedTruthVerifierVerdict
    from oai2.evals import builtin_suite as ExportedBuiltinSuite
    from oai2.evals import builtin_suites as ExportedBuiltinSuites
    from oai2.evals import (
        classify_truth_outcome as ExportedClassifyTruthOutcome,
    )
    from oai2.evals import (
        evaluate_held_out_promotion as ExportedEvaluateHeldOutPromotion,
    )
    from oai2.evals import (
        evaluate_truth_candidate_promotion as ExportedEvaluateTruthCandidatePromotion,
    )
    from oai2.evals import (
        evaluate_truth_promotion as ExportedEvaluateTruthPromotion,
    )
    from oai2.evals import run_eval_harness as ExportedRunEvalHarness
    from oai2.evals import (
        run_held_out_truth_cases as ExportedRunHeldOutTruthCases,
    )
    from oai2.evals import run_suite as ExportedRunSuite
    from oai2.evals import summarize_truth as ExportedSummarizeTruth
    from oai2.evals.regression import (
        CapabilityRegression,
        CapabilityRegressionThreshold,
    )
    from oai2.evals.regression import (
        HeldOutPromotionBudget as _HeldOutPromotionBudget,
    )
    from oai2.evals.regression import (
        HeldOutPromotionEvaluation as _HeldOutPromotionEvaluation,
    )
    from oai2.evals.regression import (
        evaluate_held_out_promotion as _EvaluateHeldOutPromotion,
    )
    from oai2.evals.truth import (
        TruthCandidatePromotionEvaluation as _TruthCandidatePromotionEvaluation,
    )
    from oai2.evals.truth import TruthCaseClass as _TruthCaseClass
    from oai2.evals.truth import TruthOutcome as _TruthOutcome
    from oai2.evals.truth import TruthPromotionBudget as _TruthPromotionBudget
    from oai2.evals.truth import TruthPromotionEvaluation as _TruthPromotionEvaluation
    from oai2.evals.truth import TruthReport as _TruthReport
    from oai2.evals.truth import TruthSample as _TruthSample
    from oai2.evals.truth import (
        evaluate_truth_candidate_promotion as _EvaluateTruthCandidatePromotion,
    )
    from oai2.evals.truth import (
        evaluate_truth_promotion as _EvaluateTruthPromotion,
    )
    from oai2.evals.truth import summarize_truth as _SummarizeTruth
    from oai2.evals.truth_runner import (
        CandidateRunner as _CandidateRunner,
    )
    from oai2.evals.truth_runner import (
        CandidateTruthInput as _CandidateTruthInput,
    )
    from oai2.evals.truth_runner import (
        CandidateTruthResponse as _CandidateTruthResponse,
    )
    from oai2.evals.truth_runner import HiddenVerifier as _HiddenVerifier
    from oai2.evals.truth_runner import TruthCase as _TruthCase
    from oai2.evals.truth_runner import TruthRunResult as _TruthRunResult
    from oai2.evals.truth_runner import (
        TruthVerifierVerdict as _TruthVerifierVerdict,
    )
    from oai2.evals.truth_runner import (
        classify_truth_outcome as _ClassifyTruthOutcome,
    )
    from oai2.evals.truth_runner import (
        run_held_out_truth_cases as _RunHeldOutTruthCases,
    )

    # init-defined (11)
    assert ExportedBUILTIN_SUITES_NAMES is BUILTIN_SUITES_NAMES
    assert ExportedCapabilityCase is CapabilityCase
    assert ExportedCapabilityScore is CapabilityScore
    assert ExportedCapabilitySuite is CapabilitySuite
    assert ExportedHarnessReport is HarnessReport
    assert ExportedSCORERS is SCORERS
    assert ExportedSuiteReport is SuiteReport
    assert ExportedBuiltinSuite is builtin_suite
    assert ExportedBuiltinSuites is builtin_suites
    assert ExportedRunEvalHarness is run_eval_harness
    assert ExportedRunSuite is run_suite
    # regression slice (5)
    assert ExportedCapabilityRegression is CapabilityRegression
    assert ExportedCapabilityRegressionThreshold is CapabilityRegressionThreshold
    assert ExportedHeldOutPromotionBudget is _HeldOutPromotionBudget
    assert ExportedHeldOutPromotionEvaluation is _HeldOutPromotionEvaluation
    assert ExportedEvaluateHeldOutPromotion is _EvaluateHeldOutPromotion
    # truth slice (10) — TruthCase lives in truth_runner, not truth
    assert ExportedTruthCaseClass is _TruthCaseClass
    assert ExportedTruthOutcome is _TruthOutcome
    assert ExportedTruthPromotionBudget is _TruthPromotionBudget
    assert ExportedTruthPromotionEvaluation is _TruthPromotionEvaluation
    assert (
        ExportedTruthCandidatePromotionEvaluation
        is _TruthCandidatePromotionEvaluation
    )
    assert ExportedTruthReport is _TruthReport
    assert ExportedTruthSample is _TruthSample
    assert ExportedEvaluateTruthCandidatePromotion is _EvaluateTruthCandidatePromotion
    assert ExportedEvaluateTruthPromotion is _EvaluateTruthPromotion
    assert ExportedSummarizeTruth is _SummarizeTruth
    # truth_runner slice (8)
    assert ExportedCandidateRunner is _CandidateRunner
    assert ExportedCandidateTruthInput is _CandidateTruthInput
    assert ExportedCandidateTruthResponse is _CandidateTruthResponse
    assert ExportedHiddenVerifier is _HiddenVerifier
    assert ExportedTruthCase is _TruthCase
    assert ExportedTruthRunResult is _TruthRunResult
    assert ExportedTruthVerifierVerdict is _TruthVerifierVerdict
    assert ExportedClassifyTruthOutcome is _ClassifyTruthOutcome
    assert ExportedRunHeldOutTruthCases is _RunHeldOutTruthCases


class TestHarnessReportCannotHideAnErroredSuite:
    """A pass_rate of 1.0 must not be the only thing a caller can see.

    `run_eval_harness(continue_on_error=True)` records a suite that raised as
    `SuiteReport(n_cases=0, n_passed=0, error=...)`. That suite therefore
    contributes 0 to both `HarnessReport.n_cases` and `n_passed`, and the two
    aggregates are identical whether or not it ran:

        one clean suite (12/12)          -> n_cases=12 n_passed=12 pass_rate=1.0
        same suite + one that raised     -> n_cases=12 n_passed=12 pass_rate=1.0

    and `HarnessReport` exposed no aggregate error field, so a reader of the
    headline numbers could not tell "everything passed" from "a suite
    crashed". This is the same "absent reads as clean" family already closed
    for numerical error, capability measurement and input identity.
    """

    @staticmethod
    def _report(*, with_error: bool) -> HarnessReport:
        passing = SuiteReport(
            suite_id="all-good",
            capability="demo",
            runtime="PlaceholderRuntime",
            n_cases=12,
            n_passed=12,
        )
        if not with_error:
            return HarnessReport(runtime="PlaceholderRuntime", reports=(passing,))
        return HarnessReport(
            runtime="PlaceholderRuntime",
            reports=(
                passing,
                SuiteReport(
                    suite_id="exploded",
                    capability="demo",
                    runtime="PlaceholderRuntime",
                    n_cases=0,
                    n_passed=0,
                    error="ValueError: boom",
                ),
            ),
        )

    def test_an_errored_suite_is_reported_at_the_aggregate(self) -> None:
        report = self._report(with_error=True)
        assert report.has_errors is True
        assert report.errored_suite_ids == ("exploded",)

    def test_a_clean_run_reports_no_errors(self) -> None:
        """Opposite direction: the new fields must not be noisy.

        A `has_errors` that was always True would be as misleading as one
        that is always False, so the clean case is pinned too.
        """
        report = self._report(with_error=False)
        assert report.has_errors is False
        assert report.errored_suite_ids == ()

    def test_the_error_is_visible_even_though_pass_rate_cannot_show_it(self) -> None:
        """The distinction exists precisely because pass_rate cannot make it."""
        clean = self._report(with_error=False)
        errored = self._report(with_error=True)
        assert (clean.n_cases, clean.n_passed, clean.pass_rate) == (
            errored.n_cases,
            errored.n_passed,
            errored.pass_rate,
        )
        assert clean.has_errors is False
        assert errored.has_errors is True

    def test_pass_rate_semantics_are_unchanged(self) -> None:
        """Opposite direction: an errored suite must still not count as a failure.

        Folding errors into the denominator would change what `pass_rate`
        means for every existing caller. The fix is additive -- surface the
        error, do not silently redefine a published metric.
        """
        report = self._report(with_error=True)
        assert report.n_cases == 12
        assert report.n_passed == 12
        assert report.pass_rate == 1.0

    def test_empty_harness_reports_no_errors(self) -> None:
        """Opposite direction: no suites at all is not an error."""
        report = HarnessReport(runtime="PlaceholderRuntime")
        assert report.has_errors is False
        assert report.errored_suite_ids == ()
