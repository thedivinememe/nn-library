"""Convergence strategies: DiffConvergence, NuConvergence, HybridConvergence."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class ConvergenceResult:
    converged: bool
    reason: str
    confidence: float  # 0-1
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IterationSnapshot:
    """Minimal view of one iteration, used by convergence strategies."""

    iteration: int
    output: str
    nu: float
    nu_raw: float
    conflict: float = 0.0
    def_sem: float = 0.0
    def_ep: float = 0.0
    def_proc: float = 0.0


@runtime_checkable
class ConvergenceStrategy(Protocol):
    def should_stop(self, history: list[IterationSnapshot]) -> ConvergenceResult: ...

    @property
    def name(self) -> str: ...


def _output_similarity(a: str, b: str) -> float:
    """Compute similarity ratio between two strings."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# ---------- DiffConvergence ----------


@dataclass
class DiffConvergence:
    """Baseline: stops when output stabilizes (text similarity above threshold).

    No awareness of output quality — only stability.
    """

    similarity_threshold: float = 0.95
    stable_count: int = 2  # consecutive stable iterations needed

    @property
    def name(self) -> str:
        return "diff"

    def should_stop(self, history: list[IterationSnapshot]) -> ConvergenceResult:
        if len(history) < 2:
            return ConvergenceResult(
                converged=False, reason="insufficient_history", confidence=0.0
            )

        # Check last N pairs for stability
        consecutive_stable = 0
        for i in range(len(history) - 1, 0, -1):
            sim = _output_similarity(history[i].output, history[i - 1].output)
            if sim >= self.similarity_threshold:
                consecutive_stable += 1
            else:
                break

        if consecutive_stable >= self.stable_count:
            last_sim = _output_similarity(
                history[-1].output, history[-2].output
            )
            return ConvergenceResult(
                converged=True,
                reason="output_stable",
                confidence=last_sim,
                diagnostics={
                    "consecutive_stable": consecutive_stable,
                    "last_similarity": last_sim,
                },
            )

        return ConvergenceResult(
            converged=False,
            reason="output_still_changing",
            confidence=0.0,
            diagnostics={"consecutive_stable": consecutive_stable},
        )


# ---------- NuConvergence ----------


@dataclass
class NuConvergence:
    """N/N-N trajectory-based convergence.

    Stops when:
    - ν ≤ licensing threshold AND
    - ν has been decreasing (or stable below threshold) for N iterations

    Detects:
    - Oscillation: ν increases after decreasing
    - Stalling: |Δν| < ε but ν > threshold
    """

    nu_threshold: float = 0.4
    nu_raw_threshold: float = 0.5
    stable_epsilon: float = 0.01
    stable_count: int = 2
    oscillation_limit: int = 3

    @property
    def name(self) -> str:
        return "nu"

    def should_stop(self, history: list[IterationSnapshot]) -> ConvergenceResult:
        if len(history) < 2:
            return ConvergenceResult(
                converged=False, reason="insufficient_history", confidence=0.0
            )

        current = history[-1]
        nus = [s.nu for s in history]

        # Check if licensed
        is_licensed = (
            current.nu_raw <= self.nu_raw_threshold
            and current.nu <= self.nu_threshold
        )

        # Check ν stability (below threshold)
        consecutive_stable_below = 0
        for i in range(len(nus) - 1, 0, -1):
            if nus[i] <= self.nu_threshold and abs(nus[i] - nus[i - 1]) < self.stable_epsilon:
                consecutive_stable_below += 1
            else:
                break

        if is_licensed and consecutive_stable_below >= self.stable_count:
            return ConvergenceResult(
                converged=True,
                reason="nu_licensed_and_stable",
                confidence=1.0 - current.nu,
                diagnostics={
                    "final_nu": current.nu,
                    "final_nu_raw": current.nu_raw,
                    "consecutive_stable": consecutive_stable_below,
                },
            )

        # Detect oscillation
        oscillation_count = 0
        for i in range(2, len(nus)):
            if (nus[i] - nus[i - 1]) * (nus[i - 1] - nus[i - 2]) < 0:
                oscillation_count += 1

        # Detect stalling above threshold
        is_stalled = False
        if len(nus) >= self.stable_count + 1:
            recent_deltas = [
                abs(nus[i] - nus[i - 1])
                for i in range(len(nus) - self.stable_count, len(nus))
            ]
            if all(d < self.stable_epsilon for d in recent_deltas) and current.nu > self.nu_threshold:
                is_stalled = True

        # Build diagnostic suggestion
        suggestion = ""
        if oscillation_count >= self.oscillation_limit:
            suggestion = "oscillation_detected: consider Recontextualize"
        elif is_stalled and current.conflict > 0.3:
            suggestion = "stuck_with_conflict: consider Split"
        elif is_stalled and current.def_sem < 0.5:
            suggestion = "stuck_low_def_sem: consider NegDefine"
        elif is_stalled:
            suggestion = "stalled: try different approach"

        return ConvergenceResult(
            converged=False,
            reason="nu_above_threshold" if not is_stalled else "stalled",
            confidence=max(0.0, 1.0 - current.nu),
            diagnostics={
                "current_nu": current.nu,
                "current_nu_raw": current.nu_raw,
                "is_licensed": is_licensed,
                "is_stalled": is_stalled,
                "oscillation_count": oscillation_count,
                "suggestion": suggestion,
            },
        )


# ---------- HybridConvergence ----------


@dataclass
class HybridConvergence:
    """Combined strategy: requires BOTH output stability AND ν below threshold.

    Uses ν trajectory to continue even when output is stable.
    Uses output diff to detect spinning (changes that don't improve ν).
    """

    similarity_threshold: float = 0.95
    diff_stable_count: int = 2
    nu_threshold: float = 0.4
    nu_raw_threshold: float = 0.5
    stable_epsilon: float = 0.01
    nu_stable_count: int = 2
    oscillation_limit: int = 3

    @property
    def name(self) -> str:
        return "hybrid"

    def should_stop(self, history: list[IterationSnapshot]) -> ConvergenceResult:
        if len(history) < 2:
            return ConvergenceResult(
                converged=False, reason="insufficient_history", confidence=0.0
            )

        current = history[-1]
        nus = [s.nu for s in history]

        # --- Output stability check ---
        consecutive_diff_stable = 0
        for i in range(len(history) - 1, 0, -1):
            sim = _output_similarity(history[i].output, history[i - 1].output)
            if sim >= self.similarity_threshold:
                consecutive_diff_stable += 1
            else:
                break
        output_is_stable = consecutive_diff_stable >= self.diff_stable_count

        # --- ν licensing check ---
        is_licensed = (
            current.nu_raw <= self.nu_raw_threshold
            and current.nu <= self.nu_threshold
        )

        # --- ν stability check ---
        consecutive_nu_stable = 0
        for i in range(len(nus) - 1, 0, -1):
            if abs(nus[i] - nus[i - 1]) < self.stable_epsilon:
                consecutive_nu_stable += 1
            else:
                break
        nu_is_stable = consecutive_nu_stable >= self.nu_stable_count

        # --- Convergence: both stable AND licensed ---
        if output_is_stable and is_licensed and nu_is_stable:
            sim = _output_similarity(history[-1].output, history[-2].output)
            return ConvergenceResult(
                converged=True,
                reason="output_stable_and_nu_licensed",
                confidence=(1.0 - current.nu) * sim,
                diagnostics={
                    "final_nu": current.nu,
                    "final_nu_raw": current.nu_raw,
                    "output_similarity": sim,
                    "diff_stable_count": consecutive_diff_stable,
                    "nu_stable_count": consecutive_nu_stable,
                },
            )

        # --- Detect spinning: output changes but ν not improving ---
        is_spinning = False
        if len(history) >= 3 and not output_is_stable:
            recent_nu_deltas = [
                nus[i] - nus[i - 1]
                for i in range(max(1, len(nus) - 3), len(nus))
            ]
            # Output changing but ν not improving (deltas positive or near zero)
            if all(d >= -self.stable_epsilon for d in recent_nu_deltas):
                is_spinning = True

        # --- Detect premature convergence: output stable but ν still high ---
        premature = output_is_stable and not is_licensed

        # --- Oscillation detection ---
        oscillation_count = 0
        for i in range(2, len(nus)):
            if (nus[i] - nus[i - 1]) * (nus[i - 1] - nus[i - 2]) < 0:
                oscillation_count += 1

        reason = "in_progress"
        suggestion = ""
        if premature:
            reason = "premature_convergence_detected"
            suggestion = "output stable but ν still high — push for deeper refinement"
        elif is_spinning:
            reason = "spinning_detected"
            suggestion = "output changing without ν improvement — try different approach"
        elif oscillation_count >= self.oscillation_limit:
            reason = "oscillation_detected"
            suggestion = "consider Recontextualize or Split"

        return ConvergenceResult(
            converged=False,
            reason=reason,
            confidence=max(0.0, 1.0 - current.nu),
            diagnostics={
                "current_nu": current.nu,
                "current_nu_raw": current.nu_raw,
                "output_is_stable": output_is_stable,
                "is_licensed": is_licensed,
                "nu_is_stable": nu_is_stable,
                "is_spinning": is_spinning,
                "premature_convergence": premature,
                "oscillation_count": oscillation_count,
                "suggestion": suggestion,
            },
        )
