"""Tests for metrics collection and reporting."""

from __future__ import annotations

import pytest

from rwt_integration.convergence import ConvergenceResult
from rwt_integration.metrics import (
    ComparisonReport,
    IterationRecord,
    RunSummary,
)
from rwt_integration.providers import SelfAssessment


def _make_summary(
    task_id: str = "task1",
    strategy: str = "diff",
    iterations: int = 5,
    converged: bool = True,
    final_nu: float = 0.3,
) -> RunSummary:
    return RunSummary(
        task_id=task_id,
        strategy_name=strategy,
        iterations=iterations,
        converged=converged,
        convergence_reason="test",
        final_nu=final_nu,
        final_nu_raw=final_nu,
        nu_trajectory=tuple(1.0 - i * 0.1 for i in range(iterations)),
        nu_raw_trajectory=tuple(1.0 - i * 0.1 for i in range(iterations)),
        total_wall_time=1.0,
        total_tokens_estimate=1000,
        final_output="output",
    )


class TestRunSummary:
    def test_mean_nu_delta(self) -> None:
        s = _make_summary(iterations=5)
        assert s.mean_nu_delta == pytest.approx(-0.1)

    def test_mean_nu_delta_single(self) -> None:
        s = RunSummary(
            task_id="t", strategy_name="s", iterations=1, converged=True,
            convergence_reason="", final_nu=0.5, final_nu_raw=0.5,
            nu_trajectory=(0.5,), nu_raw_trajectory=(0.5,),
            total_wall_time=0.0, total_tokens_estimate=0, final_output="",
        )
        assert s.mean_nu_delta == 0.0


class TestComparisonReport:
    def test_iterations_comparison(self) -> None:
        report = ComparisonReport()
        report.add_run(_make_summary(strategy="diff", iterations=3))
        report.add_run(_make_summary(strategy="diff", iterations=5))
        report.add_run(_make_summary(strategy="hybrid", iterations=7))

        comp = report.iterations_comparison()
        assert comp["diff"] == pytest.approx(4.0)
        assert comp["hybrid"] == pytest.approx(7.0)

    def test_convergence_rate(self) -> None:
        report = ComparisonReport()
        report.add_run(_make_summary(strategy="diff", converged=True))
        report.add_run(_make_summary(strategy="diff", converged=False))
        report.add_run(_make_summary(strategy="hybrid", converged=True))

        rates = report.convergence_rate()
        assert rates["diff"] == pytest.approx(0.5)
        assert rates["hybrid"] == pytest.approx(1.0)

    def test_quality_comparison_no_scores(self) -> None:
        report = ComparisonReport()
        report.add_run(_make_summary(strategy="diff"))
        q = report.quality_comparison()
        assert q["diff"] is None

    def test_final_nu_comparison(self) -> None:
        report = ComparisonReport()
        report.add_run(_make_summary(strategy="diff", final_nu=0.3))
        report.add_run(_make_summary(strategy="diff", final_nu=0.5))
        report.add_run(_make_summary(strategy="hybrid", final_nu=0.2))

        comp = report.final_nu_comparison()
        assert comp["diff"] == pytest.approx(0.4)
        assert comp["hybrid"] == pytest.approx(0.2)

    def test_nu_trajectory_summary(self) -> None:
        report = ComparisonReport()
        report.add_run(_make_summary(strategy="diff", task_id="t1", iterations=3))
        summary = report.nu_trajectory_summary()
        assert "diff" in summary
        assert "t1" in summary["diff"]
        assert len(summary["diff"]["t1"]) == 1
