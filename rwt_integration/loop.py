"""RWTLoop — the core iteration engine."""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

from nn_logic.types import (
    Clock,
    ContextID,
    MockClock,
    TargetID,
)
from nn_logic.policy import Policy

from rwt_integration.config import LoopConfig
from rwt_integration.convergence import (
    ConvergenceResult,
    ConvergenceStrategy,
    IterationSnapshot,
)
from rwt_integration.metacognition import MetacognitionBridge
from rwt_integration.metrics import IterationRecord, RunSummary
from rwt_integration.providers import ModelProvider, SelfAssessment
from rwt_integration.tasks import Task


@dataclass(frozen=True)
class RunResult:
    """Full result from an RWT loop execution."""

    summary: RunSummary
    snapshots: tuple[IterationSnapshot, ...] = ()


class RWTLoop:
    """Core RWT iteration engine with pluggable convergence strategy."""

    def __init__(
        self,
        provider: ModelProvider,
        strategy: ConvergenceStrategy,
        bridge: MetacognitionBridge,
        config: LoopConfig = LoopConfig(),
        clock: Optional[Clock] = None,
    ) -> None:
        self._provider = provider
        self._strategy = strategy
        self._bridge = bridge
        self._config = config
        self._clock = clock or MockClock(0.0)

    def run(self, task: Task) -> RunResult:
        """Execute the RWT loop on a task.

        1. Initialize N/N-N state for the task target
        2. Generate initial output
        3. Loop:
           a. Get self-assessment from model
           b. Update N/N-N state via bridge
           c. Check convergence strategy
           d. If not converged: generate refined output with ν diagnostics
           e. Record metrics
        4. Return final output + full trace
        """
        target_id = TargetID(task.id)
        context_id = ContextID(f"rwt_{task.id}")

        # Initialize
        self._bridge.reset()
        state = self._bridge.initialize_state(target_id, context_id)

        snapshots: list[IterationSnapshot] = []
        records: list[IterationRecord] = []
        nu_trajectory: list[float] = []
        nu_raw_trajectory: list[float] = []
        total_tokens = 0
        start_time = _time.monotonic()

        # Generate initial output
        initial_prompt = self._config.initial_prompt_template.format(
            task_specification=task.specification
        )
        current_output = self._provider.generate(initial_prompt)
        total_tokens += _estimate_tokens(initial_prompt + current_output)

        converged = False
        convergence_reason = "max_iterations"
        final_convergence = ConvergenceResult(
            converged=False, reason="max_iterations", confidence=0.0
        )

        for iteration in range(self._config.max_iterations):
            iter_start = _time.monotonic()

            # Step a: Get self-assessment
            assess_prompt = self._config.assessment_prompt_template.format(
                task_specification=task.specification,
                current_output=current_output,
            )
            assessment = self._provider.assess(task.specification, current_output, assess_prompt)
            total_tokens += _estimate_tokens(assess_prompt)

            # Step b: Update N/N-N state
            state, diagnostics = self._bridge.process_iteration(
                state, current_output, assessment, iteration
            )

            nu = diagnostics.nu
            nu_raw = diagnostics.nu_raw
            nu_trajectory.append(nu)
            nu_raw_trajectory.append(nu_raw)

            # Compute output diff
            if iteration > 0 and snapshots:
                diff_ratio = 1.0 - SequenceMatcher(
                    None, snapshots[-1].output, current_output
                ).ratio()
            else:
                diff_ratio = 1.0

            # Build snapshot for convergence strategy
            snapshot = IterationSnapshot(
                iteration=iteration,
                output=current_output,
                nu=nu,
                nu_raw=nu_raw,
                conflict=diagnostics.conflict,
                def_sem=diagnostics.def_sem,
                def_ep=diagnostics.def_ep,
                def_proc=diagnostics.def_proc,
            )
            snapshots.append(snapshot)

            # Step c: Check convergence
            convergence_check = self._strategy.should_stop(snapshots)

            iter_time = _time.monotonic() - iter_start

            # Record metrics
            record = IterationRecord(
                iteration=iteration,
                output=current_output,
                assessment=assessment,
                nu=nu,
                nu_raw=nu_raw,
                nu_penalty=diagnostics.nu_penalty,
                def_sem=diagnostics.def_sem,
                def_ep=diagnostics.def_ep,
                def_proc=diagnostics.def_proc,
                conflict_score=diagnostics.conflict,
                output_diff_ratio=diff_ratio,
                convergence_check=convergence_check,
                wall_time_seconds=iter_time,
                prompt_tokens_estimate=total_tokens,
            )
            records.append(record)

            if convergence_check.converged and iteration >= self._config.min_iterations - 1:
                converged = True
                convergence_reason = convergence_check.reason
                final_convergence = convergence_check
                break

            # Step d: Generate refined output
            suggestion = convergence_check.diagnostics.get("suggestion", "")
            guidance = diagnostics.refinement_guidance()
            if suggestion:
                guidance = f"{suggestion}. {guidance}"

            refine_prompt = self._config.refinement_prompt_template.format(
                task_specification=task.specification,
                current_output=current_output,
                assessment_summary=assessment.to_summary(),
                nu=nu,
                nu_raw=nu_raw,
                nu_penalty=diagnostics.nu_penalty,
                def_sem=diagnostics.def_sem,
                def_ep=diagnostics.def_ep,
                def_proc=diagnostics.def_proc,
                conflict=diagnostics.conflict,
                weakest_component=diagnostics.weakest_component,
                refinement_guidance=guidance,
                refinement_priority=assessment.refinement_priority,
            )
            current_output = self._provider.generate(refine_prompt)
            total_tokens += _estimate_tokens(refine_prompt + current_output)

        total_time = _time.monotonic() - start_time

        summary = RunSummary(
            task_id=task.id,
            strategy_name=self._strategy.name,
            iterations=len(records),
            converged=converged,
            convergence_reason=convergence_reason,
            final_nu=nu_trajectory[-1] if nu_trajectory else 1.0,
            final_nu_raw=nu_raw_trajectory[-1] if nu_raw_trajectory else 1.0,
            nu_trajectory=tuple(nu_trajectory),
            nu_raw_trajectory=tuple(nu_raw_trajectory),
            total_wall_time=total_time,
            total_tokens_estimate=total_tokens,
            final_output=current_output,
            iteration_records=tuple(records),
        )

        return RunResult(
            summary=summary,
            snapshots=tuple(snapshots),
        )


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4
