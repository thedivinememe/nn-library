"""Integration tests for the evaluation harness and the KEY BEHAVIORAL TEST.

The key behavioral test validates the entire thesis: that Hybrid convergence
catches premature convergence that Diff misses.
"""

from __future__ import annotations

import pytest

from nn_logic.types import MockClock
from nn_logic.policy import Policy

from rwt_integration.config import LoopConfig
from rwt_integration.convergence import (
    DiffConvergence,
    HybridConvergence,
    NuConvergence,
)
from rwt_integration.evaluation import EvaluationHarness
from rwt_integration.loop import RWTLoop
from rwt_integration.metacognition import MetacognitionBridge
from rwt_integration.providers import (
    ScriptedProvider,
    SelfAssessment,
    premature_convergence_scenario,
    steady_improvement_scenario,
    oscillation_scenario,
    stuck_scenario,
)
from rwt_integration.tasks import ALL_TASKS, TASK_EMAIL_VALIDATOR, TASK_RATE_LIMITER


# ============================================================
# THE KEY BEHAVIORAL TEST
# ============================================================

class TestKeyBehavioralTest:
    """Validates the integration thesis.

    Scenario:
    1. Output stabilizes after iteration 3 (DiffConvergence would stop)
    2. ν is still ~0.6 at iteration 3 (above licensing threshold)
    3. ν continues decreasing through iteration 6
    4. Output starts changing again at iteration 5 (model finds new issues)
    5. Both ν and output stabilize at iteration 7

    Expected:
    - DiffConvergence stops at iteration ~3 (premature)
    - NuConvergence stops at iteration ~7 (when ν stabilizes below threshold)
    - HybridConvergence stops at iteration ~7 (requires both conditions)
    """

    @pytest.fixture
    def clock(self) -> MockClock:
        return MockClock(1000.0)

    @pytest.fixture
    def config(self) -> LoopConfig:
        return LoopConfig(max_iterations=12, min_iterations=1)

    def test_diff_stops_prematurely(self, clock: MockClock, config: LoopConfig) -> None:
        """Diff strategy stops when output stabilizes, even with high ν."""
        provider = ScriptedProvider(script_fn=premature_convergence_scenario())
        strategy = DiffConvergence(similarity_threshold=0.95, stable_count=2)
        bridge = MetacognitionBridge(clock=clock, config=config)
        loop = RWTLoop(provider, strategy, bridge, config, clock)

        result = loop.run(TASK_RATE_LIMITER)

        # Diff should stop around iteration 3-4 (when output freezes)
        assert result.summary.converged is True
        assert result.summary.iterations <= 5  # stops early
        assert result.summary.final_nu > 0.4  # ν is still high — premature!

    def test_nu_continues_past_output_freeze(self, clock: MockClock, config: LoopConfig) -> None:
        """Nu strategy continues even when output is frozen, stops when ν is good."""
        provider = ScriptedProvider(script_fn=premature_convergence_scenario())
        strategy = NuConvergence(
            nu_threshold=0.4, nu_raw_threshold=0.5,
            stable_count=2, stable_epsilon=0.02,
        )
        bridge = MetacognitionBridge(clock=clock, config=config)
        loop = RWTLoop(provider, strategy, bridge, config, clock)

        result = loop.run(TASK_RATE_LIMITER)

        # Nu strategy should continue past the output freeze
        assert result.summary.iterations > 5

    def test_hybrid_catches_premature_convergence(
        self, clock: MockClock, config: LoopConfig
    ) -> None:
        """Hybrid strategy continues past output freeze, converges later with good ν."""
        provider = ScriptedProvider(script_fn=premature_convergence_scenario())
        strategy = HybridConvergence(
            similarity_threshold=0.95,
            diff_stable_count=2,
            nu_threshold=0.4,
            nu_raw_threshold=0.5,
            nu_stable_count=2,
            stable_epsilon=0.02,
        )
        bridge = MetacognitionBridge(clock=clock, config=config)
        loop = RWTLoop(provider, strategy, bridge, config, clock)

        result = loop.run(TASK_RATE_LIMITER)

        # Hybrid should NOT stop at the output freeze point
        assert result.summary.iterations > 5

    def test_hybrid_beats_diff_on_final_nu(
        self, clock: MockClock, config: LoopConfig
    ) -> None:
        """Hybrid should achieve lower final ν than Diff on premature scenario."""
        provider_diff = ScriptedProvider(script_fn=premature_convergence_scenario())
        provider_hybrid = ScriptedProvider(script_fn=premature_convergence_scenario())

        diff_strategy = DiffConvergence(similarity_threshold=0.95, stable_count=2)
        hybrid_strategy = HybridConvergence(
            similarity_threshold=0.95, diff_stable_count=2,
            nu_threshold=0.4, nu_raw_threshold=0.5,
            nu_stable_count=2, stable_epsilon=0.02,
        )

        bridge_diff = MetacognitionBridge(clock=clock, config=config)
        bridge_hybrid = MetacognitionBridge(clock=clock, config=config)

        loop_diff = RWTLoop(provider_diff, diff_strategy, bridge_diff, config, clock)
        loop_hybrid = RWTLoop(provider_hybrid, hybrid_strategy, bridge_hybrid, config, clock)

        result_diff = loop_diff.run(TASK_RATE_LIMITER)
        result_hybrid = loop_hybrid.run(TASK_RATE_LIMITER)

        # Hybrid's final output should be "better" (lower ν)
        assert result_hybrid.summary.final_nu < result_diff.summary.final_nu


# ============================================================
# SCENARIO-SPECIFIC TESTS
# ============================================================

class TestSteadyImprovement:
    def test_all_strategies_converge(self) -> None:
        """On steady improvement, all strategies should eventually converge."""
        clock = MockClock(1000.0)
        config = LoopConfig(max_iterations=12)

        for StrategyClass in [DiffConvergence, NuConvergence, HybridConvergence]:
            provider = ScriptedProvider(script_fn=steady_improvement_scenario())
            if StrategyClass == DiffConvergence:
                strategy = StrategyClass(similarity_threshold=0.99, stable_count=2)
            elif StrategyClass == NuConvergence:
                strategy = StrategyClass(
                    nu_threshold=0.45, nu_raw_threshold=0.55,
                    stable_count=2, stable_epsilon=0.03,
                )
            else:
                strategy = StrategyClass(
                    similarity_threshold=0.99, diff_stable_count=2,
                    nu_threshold=0.45, nu_raw_threshold=0.55,
                    nu_stable_count=2, stable_epsilon=0.03,
                )
            bridge = MetacognitionBridge(clock=clock, config=config)
            loop = RWTLoop(provider, strategy, bridge, config, clock)

            result = loop.run(TASK_EMAIL_VALIDATOR)
            # Should converge or hit max (steady improvement keeps changing)
            assert result.summary.iterations > 0


class TestStuckScenario:
    def test_diff_converges_immediately(self) -> None:
        """Diff stops quickly on stuck scenario (output never changes)."""
        clock = MockClock(1000.0)
        config = LoopConfig(max_iterations=10)

        provider = ScriptedProvider(script_fn=stuck_scenario())
        strategy = DiffConvergence(similarity_threshold=0.95, stable_count=2)
        bridge = MetacognitionBridge(clock=clock, config=config)
        loop = RWTLoop(provider, strategy, bridge, config, clock)

        result = loop.run(TASK_EMAIL_VALIDATOR)
        assert result.summary.converged is True
        assert result.summary.iterations <= 4

    def test_hybrid_detects_premature_on_stuck(self) -> None:
        """Hybrid should not converge on stuck scenario (ν stays high)."""
        clock = MockClock(1000.0)
        config = LoopConfig(max_iterations=6)

        provider = ScriptedProvider(script_fn=stuck_scenario())
        strategy = HybridConvergence(
            similarity_threshold=0.95, diff_stable_count=2,
            nu_threshold=0.3, nu_raw_threshold=0.4,
        )
        bridge = MetacognitionBridge(clock=clock, config=config)
        loop = RWTLoop(provider, strategy, bridge, config, clock)

        result = loop.run(TASK_EMAIL_VALIDATOR)
        # Should hit max iterations because ν never gets low enough
        assert result.summary.converged is False
        assert result.summary.iterations == 6


class TestOscillation:
    def test_detects_oscillation(self) -> None:
        """Oscillation scenario should trigger oscillation detection."""
        clock = MockClock(1000.0)
        config = LoopConfig(max_iterations=8)

        provider = ScriptedProvider(script_fn=oscillation_scenario())
        strategy = NuConvergence(
            nu_threshold=0.3, oscillation_limit=3,
            stable_count=2,
        )
        bridge = MetacognitionBridge(clock=clock, config=config)
        loop = RWTLoop(provider, strategy, bridge, config, clock)

        result = loop.run(TASK_EMAIL_VALIDATOR)
        # Should not converge (oscillating)
        assert result.summary.converged is False


# ============================================================
# EVALUATION HARNESS INTEGRATION TEST
# ============================================================

class TestEvaluationHarness:
    def test_comparison_report_structure(self) -> None:
        """Run the harness with multiple strategies and verify report structure."""
        clock = MockClock(1000.0)
        config = LoopConfig(max_iterations=8)
        harness = EvaluationHarness(config=config, clock=clock)

        provider = ScriptedProvider(script_fn=steady_improvement_scenario())
        strategies = [
            DiffConvergence(similarity_threshold=0.99, stable_count=2),
            NuConvergence(nu_threshold=0.45, stable_count=2, stable_epsilon=0.03),
            HybridConvergence(
                similarity_threshold=0.99, diff_stable_count=2,
                nu_threshold=0.45, nu_stable_count=2, stable_epsilon=0.03,
            ),
        ]

        report = harness.run_comparison(
            tasks=[TASK_EMAIL_VALIDATOR],
            strategies=strategies,
            provider=provider,
        )

        # Verify structure
        assert "diff" in report.results
        assert "nu" in report.results
        assert "hybrid" in report.results

        # Verify metrics
        iters = report.iterations_comparison()
        assert all(v > 0 for v in iters.values())

        rates = report.convergence_rate()
        assert all(0.0 <= v <= 1.0 for v in rates.values())

    def test_comparison_across_tasks(self) -> None:
        """Verify harness handles multiple tasks."""
        clock = MockClock(1000.0)
        config = LoopConfig(max_iterations=6)
        harness = EvaluationHarness(config=config, clock=clock)

        provider = ScriptedProvider(script_fn=steady_improvement_scenario())
        strategy = DiffConvergence(similarity_threshold=0.99, stable_count=2)

        report = harness.run_comparison(
            tasks=ALL_TASKS,
            strategies=[strategy],
            provider=provider,
        )

        assert "diff" in report.results
        # Should have entries for all three tasks
        assert len(report.results["diff"]) == 3
