"""Tests for velocity monitoring."""

from __future__ import annotations

import pytest

from nn_logic.types import (
    ContextID,
    PenaltySource,
    RefinementRecord,
    TargetID,
)
from nn_logic.velocity import rv, rv_mean, rv_stuck_rate, SystemHealth


def _record(nu_after: float, nu_raw_after: float = 0.5) -> RefinementRecord:
    return RefinementRecord(
        target_id=TargetID("t"),
        context_id=ContextID("c"),
        operator="test",
        nu_before=0.0,
        nu_after=nu_after,
        nu_raw_before=0.0,
        nu_raw_after=nu_raw_after,
        penalties_before={},
        penalties_after={},
        timestamp=0.0,
    )


class TestRV:
    def test_empty(self) -> None:
        assert rv([]) == []

    def test_single_record(self) -> None:
        assert rv([_record(0.5)]) == []

    def test_decreasing_nu(self) -> None:
        records = [_record(0.8), _record(0.6), _record(0.4)]
        velocities = rv(records)
        assert len(velocities) == 2
        assert all(v < 0 for v in velocities)

    def test_stable_nu(self) -> None:
        records = [_record(0.5), _record(0.5), _record(0.5)]
        velocities = rv(records)
        assert all(v == 0.0 for v in velocities)


class TestRVMean:
    def test_empty(self) -> None:
        assert rv_mean([]) == 0.0

    def test_decreasing(self) -> None:
        records = [_record(0.8), _record(0.6), _record(0.4)]
        assert rv_mean(records) == pytest.approx(-0.2)


class TestRVStuckRate:
    def test_all_stuck(self) -> None:
        records = [_record(0.5), _record(0.5), _record(0.5)]
        assert rv_stuck_rate(records) == pytest.approx(1.0)

    def test_none_stuck(self) -> None:
        records = [_record(0.8), _record(0.5), _record(0.2)]
        assert rv_stuck_rate(records) == pytest.approx(0.0)


class TestSystemHealth:
    def test_health_score(self) -> None:
        health = SystemHealth(
            total_targets=10,
            licensed_count=3,
            null_count=4,
            mean_nu=0.6,
            mean_nu_raw=0.5,
            mean_velocity=-0.1,
            stuck_rate=0.2,
        )
        assert health.licensed_fraction == pytest.approx(0.3)
        assert health.health_score == 0.6
