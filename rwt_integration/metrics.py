"""Metrics collection: IterationRecord, RunSummary, ComparisonReport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from rwt_integration.convergence import ConvergenceResult
from rwt_integration.providers import SelfAssessment


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    output: str
    assessment: SelfAssessment
    nu: float
    nu_raw: float
    nu_penalty: float
    def_sem: float
    def_ep: float
    def_proc: float
    conflict_score: float
    output_diff_ratio: float  # 0 = identical to previous, 1 = completely different
    convergence_check: ConvergenceResult
    wall_time_seconds: float
    prompt_tokens_estimate: int


@dataclass(frozen=True)
class RunSummary:
    task_id: str
    strategy_name: str
    iterations: int
    converged: bool
    convergence_reason: str
    final_nu: float
    final_nu_raw: float
    nu_trajectory: tuple[float, ...]
    nu_raw_trajectory: tuple[float, ...]
    total_wall_time: float
    total_tokens_estimate: int
    final_output: str
    quality_score: Optional[float] = None
    iteration_records: tuple[IterationRecord, ...] = ()

    @property
    def mean_nu_delta(self) -> float:
        if len(self.nu_trajectory) < 2:
            return 0.0
        deltas = [
            self.nu_trajectory[i] - self.nu_trajectory[i - 1]
            for i in range(1, len(self.nu_trajectory))
        ]
        return sum(deltas) / len(deltas)


@dataclass
class ComparisonReport:
    """Results from running multiple strategies across multiple tasks."""

    results: dict[str, dict[str, list[RunSummary]]] = field(default_factory=dict)
    # strategy_name -> task_id -> list of run summaries

    def add_run(self, summary: RunSummary) -> None:
        strat = summary.strategy_name
        task = summary.task_id
        if strat not in self.results:
            self.results[strat] = {}
        if task not in self.results[strat]:
            self.results[strat][task] = []
        self.results[strat][task].append(summary)

    def iterations_comparison(self) -> dict[str, float]:
        """Average iterations to convergence per strategy."""
        result: dict[str, float] = {}
        for strat, tasks in self.results.items():
            all_iters: list[int] = []
            for task_runs in tasks.values():
                all_iters.extend(r.iterations for r in task_runs)
            result[strat] = sum(all_iters) / len(all_iters) if all_iters else 0.0
        return result

    def convergence_rate(self) -> dict[str, float]:
        """Fraction of runs that converged per strategy."""
        result: dict[str, float] = {}
        for strat, tasks in self.results.items():
            total = 0
            converged = 0
            for task_runs in tasks.values():
                for r in task_runs:
                    total += 1
                    if r.converged:
                        converged += 1
            result[strat] = converged / total if total > 0 else 0.0
        return result

    def quality_comparison(self) -> dict[str, Optional[float]]:
        """Average quality score per strategy (None if no scores available)."""
        result: dict[str, Optional[float]] = {}
        for strat, tasks in self.results.items():
            scores: list[float] = []
            for task_runs in tasks.values():
                for r in task_runs:
                    if r.quality_score is not None:
                        scores.append(r.quality_score)
            result[strat] = sum(scores) / len(scores) if scores else None
        return result

    def nu_trajectory_summary(self) -> dict[str, dict[str, list[tuple[float, ...]]]]:
        """ν trajectories per strategy per task."""
        result: dict[str, dict[str, list[tuple[float, ...]]]] = {}
        for strat, tasks in self.results.items():
            result[strat] = {}
            for task_id, runs in tasks.items():
                result[strat][task_id] = [r.nu_trajectory for r in runs]
        return result

    def final_nu_comparison(self) -> dict[str, float]:
        """Average final ν per strategy."""
        result: dict[str, float] = {}
        for strat, tasks in self.results.items():
            nus: list[float] = []
            for task_runs in tasks.values():
                nus.extend(r.final_nu for r in task_runs)
            result[strat] = sum(nus) / len(nus) if nus else 1.0
        return result
