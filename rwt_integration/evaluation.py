"""Evaluation harness — run tasks across strategies and collect metrics."""

from __future__ import annotations

from typing import Optional

from nn_logic.types import Clock, MockClock
from nn_logic.policy import Policy

from rwt_integration.config import LoopConfig
from rwt_integration.convergence import ConvergenceStrategy
from rwt_integration.loop import RWTLoop, RunResult
from rwt_integration.metacognition import MetacognitionBridge
from rwt_integration.metrics import ComparisonReport, RunSummary
from rwt_integration.providers import ModelProvider
from rwt_integration.tasks import Task


class EvaluationHarness:
    """Run each task with each strategy and collect comparison metrics."""

    def __init__(
        self,
        config: LoopConfig = LoopConfig(),
        policy: Policy = Policy(),
        clock: Optional[Clock] = None,
    ) -> None:
        self._config = config
        self._policy = policy
        self._clock = clock or MockClock(0.0)

    def run_single(
        self,
        task: Task,
        strategy: ConvergenceStrategy,
        provider: ModelProvider,
    ) -> RunResult:
        """Run a single task with a single strategy."""
        bridge = MetacognitionBridge(
            policy=self._policy,
            config=self._config,
            clock=self._clock,
        )
        loop = RWTLoop(
            provider=provider,
            strategy=strategy,
            bridge=bridge,
            config=self._config,
            clock=self._clock,
        )
        return loop.run(task)

    def run_comparison(
        self,
        tasks: list[Task],
        strategies: list[ConvergenceStrategy],
        provider: ModelProvider,
        runs_per_config: int = 1,
    ) -> ComparisonReport:
        """Run each task with each strategy, collecting comparison metrics.

        Note: With deterministic mock providers, runs_per_config > 1 will
        produce identical results. Use runs_per_config > 1 with stochastic
        providers for statistical significance.
        """
        report = ComparisonReport()

        for strategy in strategies:
            for task in tasks:
                for run_idx in range(runs_per_config):
                    # Reset provider state if possible
                    if hasattr(provider, "reset"):
                        provider.reset()  # type: ignore[union-attr]

                    result = self.run_single(task, strategy, provider)
                    report.add_run(result.summary)

        return report
