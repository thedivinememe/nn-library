"""Refinement velocity monitoring and SystemHealth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from nn_logic.types import RefinementRecord, TargetID, ContextID


def rv(records: list[RefinementRecord]) -> list[float]:
    """Compute per-step refinement velocities (Δν per step)."""
    if len(records) < 2:
        return []
    velocities: list[float] = []
    for i in range(1, len(records)):
        delta_nu = records[i].nu_after - records[i - 1].nu_after
        velocities.append(delta_nu)
    return velocities


def rv_from_records(records: list[RefinementRecord]) -> list[float]:
    """Compute per-step Δν_raw velocities."""
    if len(records) < 2:
        return []
    return [
        records[i].nu_raw_after - records[i - 1].nu_raw_after
        for i in range(1, len(records))
    ]


def rv_mean(records: list[RefinementRecord]) -> float:
    """Mean refinement velocity."""
    velocities = rv(records)
    if not velocities:
        return 0.0
    return sum(velocities) / len(velocities)


def rv_stuck_rate(
    records: list[RefinementRecord],
    threshold: float = 0.001,
) -> float:
    """Fraction of steps with velocity below threshold (stuck rate)."""
    velocities = rv(records)
    if not velocities:
        return 0.0
    stuck = sum(1 for v in velocities if abs(v) < threshold)
    return stuck / len(velocities)


@dataclass(frozen=True)
class SystemHealth:
    """Overall system health metrics."""
    total_targets: int
    licensed_count: int
    null_count: int
    mean_nu: float
    mean_nu_raw: float
    mean_velocity: float
    stuck_rate: float

    @property
    def licensed_fraction(self) -> float:
        if self.total_targets == 0:
            return 0.0
        return self.licensed_count / self.total_targets

    @property
    def health_score(self) -> float:
        """Simple health metric: lower is better."""
        return self.mean_nu
