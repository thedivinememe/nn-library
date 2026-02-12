"""Tests for RWTLoop — full loop execution with mock providers."""

from __future__ import annotations

import pytest

from nn_logic.types import MockClock
from nn_logic.policy import Policy

from rwt_integration.config import LoopConfig
from rwt_integration.convergence import DiffConvergence, HybridConvergence, NuConvergence
from rwt_integration.loop import RWTLoop
from rwt_integration.metacognition import MetacognitionBridge
from rwt_integration.providers import (
    MockProvider,
    ScriptedProvider,
    SelfAssessment,
    premature_convergence_scenario,
    steady_improvement_scenario,
    stuck_scenario,
)
from rwt_integration.tasks import TASK_EMAIL_VALIDATOR


@pytest.fixture
def clock() -> MockClock:
    return MockClock(1000.0)


@pytest.fixture
def config() -> LoopConfig:
    return LoopConfig(max_iterations=10, min_iterations=1)


@pytest.fixture
def policy() -> Policy:
    return Policy()


class TestRWTLoopBasic:
    def test_runs_to_completion(self, clock: MockClock, config: LoopConfig) -> None:
        provider = ScriptedProvider(script_fn=steady_improvement_scenario())
        strategy = DiffConvergence(similarity_threshold=0.99, stable_count=2)
        bridge = MetacognitionBridge(clock=clock, config=config)
        loop = RWTLoop(provider, strategy, bridge, config, clock)

        result = loop.run(TASK_EMAIL_VALIDATOR)
        assert result.summary.iterations > 0
        assert result.summary.final_output != ""
        assert len(result.summary.nu_trajectory) == result.summary.iterations

    def test_collects_iteration_records(self, clock: MockClock, config: LoopConfig) -> None:
        provider = ScriptedProvider(script_fn=steady_improvement_scenario())
        strategy = NuConvergence(nu_threshold=0.4, nu_raw_threshold=0.5, stable_count=2)
        bridge = MetacognitionBridge(clock=clock, config=config)
        loop = RWTLoop(provider, strategy, bridge, config, clock)

        result = loop.run(TASK_EMAIL_VALIDATOR)
        assert len(result.summary.iteration_records) == result.summary.iterations
        for rec in result.summary.iteration_records:
            assert rec.nu >= 0.0
            assert rec.nu <= 1.0

    def test_respects_max_iterations(self, clock: MockClock) -> None:
        config = LoopConfig(max_iterations=3)
        provider = ScriptedProvider(script_fn=stuck_scenario())
        strategy = NuConvergence(nu_threshold=0.1)  # unreachable
        bridge = MetacognitionBridge(clock=clock, config=config)
        loop = RWTLoop(provider, strategy, bridge, config, clock)

        result = loop.run(TASK_EMAIL_VALIDATOR)
        assert result.summary.iterations == 3
        assert result.summary.converged is False

    def test_snapshots_recorded(self, clock: MockClock, config: LoopConfig) -> None:
        provider = ScriptedProvider(script_fn=steady_improvement_scenario())
        strategy = DiffConvergence(similarity_threshold=0.99, stable_count=2)
        bridge = MetacognitionBridge(clock=clock, config=config)
        loop = RWTLoop(provider, strategy, bridge, config, clock)

        result = loop.run(TASK_EMAIL_VALIDATOR)
        assert len(result.snapshots) == result.summary.iterations

    def test_nu_trajectory_populated(self, clock: MockClock, config: LoopConfig) -> None:
        provider = ScriptedProvider(script_fn=steady_improvement_scenario())
        strategy = DiffConvergence(similarity_threshold=0.99, stable_count=2)
        bridge = MetacognitionBridge(clock=clock, config=config)
        loop = RWTLoop(provider, strategy, bridge, config, clock)

        result = loop.run(TASK_EMAIL_VALIDATOR)
        assert len(result.summary.nu_trajectory) > 0
        assert all(0.0 <= n <= 1.0 for n in result.summary.nu_trajectory)


class TestRWTLoopConvergence:
    def test_diff_converges_on_stable_output(self, clock: MockClock) -> None:
        """Diff strategy should converge when output stops changing."""
        script = [
            ("output A", SelfAssessment(definition_confidence=0.3, task_coverage=0.4)),
            ("output B", SelfAssessment(definition_confidence=0.4, task_coverage=0.5)),
            ("output B", SelfAssessment(definition_confidence=0.4, task_coverage=0.5)),
            ("output B", SelfAssessment(definition_confidence=0.4, task_coverage=0.5)),
        ]
        provider = MockProvider(script=script)
        config = LoopConfig(max_iterations=10)
        strategy = DiffConvergence(similarity_threshold=0.95, stable_count=2)
        bridge = MetacognitionBridge(clock=clock, config=config)
        loop = RWTLoop(provider, strategy, bridge, config, clock)

        result = loop.run(TASK_EMAIL_VALIDATOR)
        assert result.summary.converged is True
        assert result.summary.convergence_reason == "output_stable"

    def test_strategy_name_in_summary(self, clock: MockClock, config: LoopConfig) -> None:
        provider = ScriptedProvider(script_fn=steady_improvement_scenario())
        strategy = HybridConvergence()
        bridge = MetacognitionBridge(clock=clock, config=config)
        loop = RWTLoop(provider, strategy, bridge, config, clock)

        result = loop.run(TASK_EMAIL_VALIDATOR)
        assert result.summary.strategy_name == "hybrid"
