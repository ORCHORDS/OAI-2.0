from __future__ import annotations

import pytest

from oai2.evals.qos import (
    BudgetKind,
    WorkloadBudget,
    WorkloadClass,
    WorkloadSample,
    evaluate_budget,
    evaluate_promotion,
    summarize_samples,
    useful_work_rank_key,
)


def _sample(
    *,
    ttft: float,
    useful: float,
    total: float,
    declared: bool = True,
    verified: bool = True,
    actions: int = 1,
    tps: float = 100.0,
    # None = the component did not occur in this replay. These defaults
    # used to be 0.0, which claimed every fixture exercised a tool, a
    # retrieval, a vision call and a build/test -- and reported a count of
    # N measurements for components that were never measured.
    tool_ms: float | None = None,
    retrieval_ms: float | None = None,
    vision_ms: float | None = None,
    build_test_ms: float | None = None,
    deadline_missed: bool = False,
) -> WorkloadSample:
    return WorkloadSample(
        workload=WorkloadClass.NORMAL,
        target_hardware="mac-studio-m5",
        config_id="normal-v1",
        ttft_ms=ttft,
        first_useful_action_ms=useful,
        end_to_end_ms=total,
        declared_success=declared,
        verified_success=verified,
        verified_actions=actions,
        generated_tokens=32,
        decode_tokens_per_second=tps,
        tool_ms=tool_ms,
        retrieval_ms=retrieval_ms,
        vision_ms=vision_ms,
        build_test_ms=build_test_ms,
        deadline_missed=deadline_missed,
    )


def test_canonical_workload_classes_cover_required_modes() -> None:
    assert {member.value for member in WorkloadClass} == {
        "FURIOUS",
        "NORMAL",
        "DEEP",
        "SWARM",
        "VISION",
        "RETRIEVAL_HEAVY",
    }


def test_summary_separates_ttft_from_first_useful_action_and_tail() -> None:
    report = summarize_samples(
        [
            _sample(ttft=10, useful=200, total=400, tps=300),
            _sample(ttft=12, useful=220, total=420, tps=310),
            _sample(ttft=11, useful=240, total=450, tps=320),
            _sample(ttft=9, useful=260, total=480, tps=330),
        ]
    )
    assert report.ttft_ms.p95 < 13
    assert report.first_useful_action_ms.p95 > 250
    assert report.end_to_end_ms.p99 > report.end_to_end_ms.p95
    assert report.decode_tokens_per_second.mean == pytest.approx(315.0)
    assert report.verified_success_rate == 1.0


def test_summary_reports_component_latency_decomposition() -> None:
    report = summarize_samples(
        [
            _sample(
                ttft=10,
                useful=100,
                total=400,
                tool_ms=40,
                retrieval_ms=80,
                vision_ms=0,
                build_test_ms=120,
            ),
            _sample(
                ttft=12,
                useful=120,
                total=500,
                tool_ms=60,
                retrieval_ms=120,
                vision_ms=20,
                build_test_ms=180,
            ),
        ]
    )
    assert report.tool_ms.mean == pytest.approx(50.0)
    assert report.retrieval_ms.mean == pytest.approx(100.0)
    assert report.vision_ms.mean == pytest.approx(10.0)
    assert report.build_test_ms.mean == pytest.approx(150.0)
    assert report.tool_ms.count == 2
    assert report.build_test_ms.p95 > 170


def test_false_success_and_verified_useful_work_are_reported() -> None:
    report = summarize_samples(
        [
            _sample(ttft=5, useful=20, total=100, declared=True, verified=False, actions=0),
            _sample(ttft=15, useful=25, total=100, declared=True, verified=True, actions=2),
        ]
    )
    assert report.false_success_rate == 0.5
    assert report.verified_success_rate == 0.5
    assert report.verified_actions_per_second == pytest.approx(10.0)


def test_useful_work_rank_penalizes_fast_talking_slow_action() -> None:
    fast_talking = summarize_samples([_sample(ttft=5, useful=500, total=700, tps=500)])
    useful_sooner = summarize_samples([_sample(ttft=30, useful=100, total=300, tps=100)])
    assert fast_talking.ttft_ms.p50 < useful_sooner.ttft_ms.p50
    assert useful_work_rank_key(useful_sooner) < useful_work_rank_key(fast_talking)


def test_budget_evaluation_is_versioned_and_kind_is_explicit() -> None:
    report = summarize_samples(
        [
            _sample(ttft=10, useful=100, total=200),
            _sample(ttft=10, useful=120, total=240, declared=True, verified=False),
        ]
    )
    budget = WorkloadBudget(
        version="normal-service-v1",
        workload=WorkloadClass.NORMAL,
        kind=BudgetKind.SERVICE_BUDGET,
        target_hardware="mac-studio-m5",
        config_id="normal-v1",
        first_useful_action_p95_ms=150,
        end_to_end_p95_ms=300,
        end_to_end_p99_ms=350,
        max_false_success_rate=0.1,
        min_verified_success_rate=0.9,
    )
    result = evaluate_budget(report, budget)
    assert result.budget_version == "normal-service-v1"
    assert result.budget_kind is BudgetKind.SERVICE_BUDGET
    assert result.passed is False
    assert result.failures == ("false_success_rate", "verified_success_rate")

    research = WorkloadBudget(
        version="normal-research-v1",
        workload=WorkloadClass.NORMAL,
        kind=BudgetKind.RESEARCH_TARGET,
        target_hardware="mac-studio-m5",
        config_id="normal-v1",
        first_useful_action_p95_ms=150,
        end_to_end_p95_ms=300,
        end_to_end_p99_ms=350,
        max_false_success_rate=0.5,
        min_verified_success_rate=0.5,
    )
    assert evaluate_budget(report, research).budget_kind is BudgetKind.RESEARCH_TARGET


def test_mixed_identity_and_invalid_budget_fail_closed() -> None:
    a = _sample(ttft=10, useful=20, total=30)
    b = WorkloadSample(
        workload=WorkloadClass.DEEP,
        target_hardware="mac-studio-m5",
        config_id="deep-v1",
        ttft_ms=10,
        first_useful_action_ms=20,
        end_to_end_ms=30,
        declared_success=True,
        verified_success=True,
    )
    with pytest.raises(ValueError, match="all samples must share"):
        summarize_samples([a, b])

    with pytest.raises(ValueError, match="p99"):
        WorkloadBudget(
            version="bad",
            workload=WorkloadClass.NORMAL,
            kind=BudgetKind.SERVICE_BUDGET,
            target_hardware="mac-studio-m5",
            config_id="normal-v1",
            first_useful_action_p95_ms=100,
            end_to_end_p95_ms=200,
            end_to_end_p99_ms=100,
            max_false_success_rate=0.1,
            min_verified_success_rate=0.9,
        )



def test_deadline_miss_rate_is_reported_and_budgeted() -> None:
    report = summarize_samples(
        [
            _sample(ttft=10, useful=50, total=100, deadline_missed=False),
            _sample(ttft=10, useful=60, total=120, deadline_missed=True),
        ]
    )
    assert report.deadline_miss_rate == 0.5

    budget = WorkloadBudget(
        version="deadline-v1",
        workload=WorkloadClass.NORMAL,
        kind=BudgetKind.SERVICE_BUDGET,
        target_hardware="mac-studio-m5",
        config_id="normal-v1",
        first_useful_action_p95_ms=200,
        end_to_end_p95_ms=300,
        end_to_end_p99_ms=350,
        max_false_success_rate=1.0,
        min_verified_success_rate=0.0,
        max_deadline_miss_rate=0.25,
    )
    result = evaluate_budget(report, budget)
    assert result.passed is False
    assert "deadline_miss_rate" in result.failures


def test_promotion_rejects_faster_mean_when_tail_regresses() -> None:
    baseline = summarize_samples(
        [
            _sample(ttft=20, useful=80, total=100, tps=100),
            _sample(ttft=20, useful=90, total=110, tps=100),
            _sample(ttft=20, useful=100, total=120, tps=100),
            _sample(ttft=20, useful=100, total=130, tps=100),
        ]
    )
    candidate = summarize_samples(
        [
            _sample(ttft=5, useful=50, total=60, tps=500),
            _sample(ttft=5, useful=50, total=60, tps=500),
            _sample(ttft=5, useful=50, total=60, tps=500),
            _sample(ttft=5, useful=50, total=250, tps=500),
        ]
    )
    assert candidate.decode_tokens_per_second.mean > baseline.decode_tokens_per_second.mean
    result = evaluate_promotion(candidate, baseline)
    assert result.passed is False
    assert "p95_regression" in result.failures or "p99_regression" in result.failures


def test_promotion_rejects_deadline_miss_regression() -> None:
    baseline = summarize_samples(
        [
            _sample(ttft=10, useful=50, total=100, deadline_missed=False),
            _sample(ttft=10, useful=50, total=100, deadline_missed=False),
        ]
    )
    candidate = summarize_samples(
        [
            _sample(ttft=10, useful=50, total=90, deadline_missed=False),
            _sample(ttft=10, useful=50, total=90, deadline_missed=True),
        ]
    )
    result = evaluate_promotion(candidate, baseline)
    assert result.passed is False
    assert result.failures == ("deadline_miss_regression",)


# ---------------------------------------------------------------------------
# A component that did not occur is not a component that took 0 ms.
#
# The gap: `tool_ms` / `retrieval_ms` / `vision_ms` / `build_test_ms` /
# `decode_tokens_per_second` defaulted to 0.0, and `summarize_samples` built a
# latency Distribution out of that default for every replay. A 40-replay
# workload that never invoked a tool reported:
#
#     ttft_ms    count=40  mean=100.00  p95=100.00     <- honest
#     tool_ms    count=40  mean=  0.00  p95=  0.00     <- false
#
# `count=40` is not "unmeasured", it is false: zero tool invocations occurred.
# And p95=0.00 is the best value any latency budget can be handed, produced
# entirely by absence -- so a `tool_ms.p95 <= X` budget passes trivially for
# a workload that never exercised the tool. Two lines, same report, same
# shape, one honest and one not, with nothing marking which.
#
# Fourth confirmed instance of one failure mode in this repository, after
# `no_comparable_samples`, `capability_measured` and
# `RetrievalMetrics.insufficient_evidence`.
# ---------------------------------------------------------------------------

_OPTIONAL = ("decode_tokens_per_second", "tool_ms", "retrieval_ms", "vision_ms",
             "build_test_ms")


def _plain(n: int = 40) -> list[WorkloadSample]:
    """Replays that exercised no optional component at all.

    Built directly rather than through ``_sample``, which always supplies a
    decode rate and 32 generated tokens -- itself a real measurement.
    """
    return [
        WorkloadSample(
            workload=WorkloadClass.NORMAL,
            target_hardware="mac-studio-m5",
            config_id="normal-v1",
            ttft_ms=100.0,
            first_useful_action_ms=200.0,
            end_to_end_ms=900.0,
            declared_success=True,
            verified_success=True,
            verified_actions=1,
            generated_tokens=0,
        )
        for _ in range(n)
    ]


class TestUnexercisedComponentIsNotZero:
    def test_never_exercised_component_is_none(self) -> None:
        report = summarize_samples(_plain())
        for name in _OPTIONAL:
            assert getattr(report, name) is None, (
                f"{name} claims a distribution for a component that never ran"
            )

    def test_the_honest_components_still_report_real_counts(self) -> None:
        """Guard the opposite failure: the fix must not empty real data."""
        report = summarize_samples(_plain())
        assert report.sample_count == 40
        assert report.ttft_ms.count == 40
        assert report.first_useful_action_ms.count == 40
        assert report.end_to_end_ms.count == 40
        assert report.ttft_ms.p95 == 100.0

    def test_exercised_component_reports_its_real_count(self) -> None:
        samples = [
            _sample(ttft=10, useful=100, total=400, tool_ms=40),
            _sample(ttft=12, useful=120, total=500, tool_ms=60),
        ]
        report = summarize_samples(samples)
        assert report.tool_ms is not None
        assert report.tool_ms.count == 2
        assert report.tool_ms.mean == 50.0

    def test_a_real_zero_is_preserved_and_counted(self) -> None:
        """The distinction that matters: 0.0 MEASURED, not unmeasured.

        A tool that ran and returned instantly is a real observation and must
        stay in the distribution. Collapsing it with "did not run" is the
        defect, and it cuts both ways.
        """
        samples = [
            _sample(ttft=10, useful=100, total=400, tool_ms=0.0),
            _sample(ttft=12, useful=120, total=500, tool_ms=60.0),
        ]
        report = summarize_samples(samples)
        assert report.tool_ms is not None
        assert report.tool_ms.count == 2, "a measured 0.0 ms was dropped"
        assert report.tool_ms.mean == 30.0
        # Linear interpolation between the two observed values. The point
        # is that BOTH are in the distribution, including the zero.
        assert report.tool_ms.p50 == 30.0
        assert report.tool_ms.p99 >= 0.0

    def test_mixed_samples_count_only_the_ones_that_ran(self) -> None:
        samples = [
            _sample(ttft=10, useful=100, total=400, tool_ms=40.0),
            _sample(ttft=12, useful=120, total=500),  # no tool
            _sample(ttft=14, useful=140, total=600, tool_ms=80.0),
        ]
        report = summarize_samples(samples)
        assert report.sample_count == 3
        assert report.tool_ms is not None
        assert report.tool_ms.count == 2, (
            "the replay that ran no tool contributed a fabricated measurement"
        )
        assert report.tool_ms.mean == 60.0
        assert report.retrieval_ms is None

    def test_decode_rate_is_required_when_tokens_were_generated(self) -> None:
        """Generated output with no decode rate is an incomplete measurement."""
        with pytest.raises(ValueError, match="decode_tokens_per_second"):
            WorkloadSample(
                workload=WorkloadClass.NORMAL,
                target_hardware="hw",
                config_id="c",
                ttft_ms=10.0,
                first_useful_action_ms=20.0,
                end_to_end_ms=100.0,
                declared_success=True,
                verified_success=True,
                generated_tokens=32,
            )

    def test_no_tokens_needs_no_decode_rate(self) -> None:
        """The converse must stay legal, or the check becomes a trap."""
        sample = WorkloadSample(
            workload=WorkloadClass.NORMAL,
            target_hardware="hw",
            config_id="c",
            ttft_ms=10.0,
            first_useful_action_ms=20.0,
            end_to_end_ms=100.0,
            declared_success=True,
            verified_success=False,
            generated_tokens=0,
        )
        assert sample.decode_tokens_per_second is None

    def test_negative_component_value_is_still_rejected(self) -> None:
        with pytest.raises(ValueError, match="tool_ms"):
            _sample(ttft=10, useful=100, total=400, tool_ms=-1.0)

    def test_budget_and_promotion_still_work_on_a_partial_report(self) -> None:
        """Optionality must not break the gates that do consume the report."""
        report = summarize_samples(_plain())
        budget = WorkloadBudget(
            version="v1",
            kind=BudgetKind.SERVICE_BUDGET,
            workload=WorkloadClass.NORMAL,
            target_hardware="mac-studio-m5",
            config_id="normal-v1",
            first_useful_action_p95_ms=1000.0,
            end_to_end_p95_ms=5000.0,
            end_to_end_p99_ms=8000.0,
            max_false_success_rate=0.0,
            min_verified_success_rate=1.0,
            max_deadline_miss_rate=0.0,
        )
        result = evaluate_budget(report, budget)
        assert result.passed is True
        assert evaluate_promotion(report, report).passed is True
        assert useful_work_rank_key(report)[2] == 200.0


class TestUnexercisedComponentNegativeControls:
    """Mutate the fix; the guards above must fail."""

    def test_control_zero_default_restored(self, tmp_path) -> None:
        """The original shape: 0.0 default aggregated into a distribution."""
        import importlib.util
        import pathlib
        import sys

        import oai2.evals.qos as qos_mod

        path = pathlib.Path(qos_mod.__file__)
        source = path.read_text()
        # Put the zero-default aggregation back.
        anchor = '''    measured = [float(value) for value in values if value is not None]
    return _distribution(measured) if measured else None
'''
        assert anchor in source, "helper not found; control is stale"
        target = tmp_path / "mutant_qos.py"
        target.write_text(
            source.replace(
                anchor,
                "    return _distribution([float(v or 0.0) for v in values])\n",
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

        report = module.summarize_samples(
            [
                module.WorkloadSample(
                    workload=module.WorkloadClass.NORMAL,
                    target_hardware="hw",
                    config_id="c",
                    ttft_ms=100.0,
                    first_useful_action_ms=200.0,
                    end_to_end_ms=900.0,
                    declared_success=True,
                    verified_success=True,
                    generated_tokens=0,
                )
                for _ in range(40)
            ]
        )
        assert report.tool_ms.count == 40, "mutation did not restore the fabrication"
        with pytest.raises(AssertionError):
            assert report.tool_ms is None

    def test_control_decode_rate_check_removed(self, tmp_path) -> None:
        """Without the check, generated output can carry no rate at all."""
        import importlib.util
        import pathlib
        import sys

        import oai2.evals.qos as qos_mod

        path = pathlib.Path(qos_mod.__file__)
        source = path.read_text()
        guard = '''        if self.generated_tokens > 0 and self.decode_tokens_per_second is None:
            raise ValueError(
                "decode_tokens_per_second is required when generated_tokens > 0: "
                "a replay that produced tokens always has a decode rate, and "
                "leaving it null would report generated output with no rate"
            )
'''
        assert guard in source, "guard not found; control is stale"
        target = tmp_path / "mutant_qos2.py"
        target.write_text(source.replace(guard, "", 1))
        name = "oai2.evals._mutant_qos2"
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
            ttft_ms=10.0,
            first_useful_action_ms=20.0,
            end_to_end_ms=100.0,
            declared_success=True,
            verified_success=True,
            generated_tokens=32,
        )
        assert sample.generated_tokens == 32
        assert sample.decode_tokens_per_second is None, "mutation did not bite"
        with pytest.raises(AssertionError):
            assert sample.decode_tokens_per_second is not None
