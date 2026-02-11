"""Utility functions: clamp, compute_nu, recompute_ν, record_refinement."""

from __future__ import annotations

from nn_logic.types import (
    PenaltyMode,
    PenaltySource,
    RefinementRecord,
    State,
    TargetID,
    ContextID,
)
from typing import Any, Optional


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def compute_nu(
    nu_raw: float,
    nu_penalties: dict[PenaltySource, float],
    mode: PenaltyMode = PenaltyMode.MAX,
) -> float:
    if not nu_penalties:
        nu_penalty = 0.0
    elif mode == PenaltyMode.MAX:
        nu_penalty = max(nu_penalties.values())
    else:  # SUM
        nu_penalty = min(1.0, sum(nu_penalties.values()))
    return clamp(nu_raw + nu_penalty, 0.0, 1.0)


def recompute_nu(state: State, mode: PenaltyMode = PenaltyMode.MAX) -> float:
    return compute_nu(state.nu_raw, state.nu_penalties, mode)


def make_refinement_record(
    state_before: State,
    state_after: State,
    operator: str,
    timestamp: float,
    details: Optional[dict[str, Any]] = None,
) -> RefinementRecord:
    return RefinementRecord(
        target_id=state_after.target_id,
        context_id=state_after.context_id,
        operator=operator,
        nu_before=state_before.nu,
        nu_after=state_after.nu,
        nu_raw_before=state_before.nu_raw,
        nu_raw_after=state_after.nu_raw,
        penalties_before=dict(state_before.nu_penalties),
        penalties_after=dict(state_after.nu_penalties),
        timestamp=timestamp,
        details=details or {},
    )
