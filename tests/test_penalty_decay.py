"""Tests for PenaltyDecay operator."""

from __future__ import annotations

import pytest

from nn_logic.types import (
    ContextID,
    Metadata,
    MockClock,
    PenaltySource,
    State,
    TargetID,
)
from nn_logic.operators import penalty_decay
from nn_logic.policy import Policy


class TestPenaltyDecay:
    def test_disabled_mode(self) -> None:
        clock = MockClock(100.0)
        state = State(
            target_id=TargetID("t"),
            context_id=ContextID("c"),
            nu_raw=0.5,
            nu_penalties={PenaltySource.CONFLICT: 0.15},
        )
        policy = Policy(penalty_decay_enabled=False)
        new_state, _ = penalty_decay(state, policy, clock)
        # Penalties unchanged
        assert new_state.nu_penalties[PenaltySource.CONFLICT] == 0.15

    def test_conflict_decay_after_clear_window(self) -> None:
        clock = MockClock(100000.0)
        state = State(
            target_id=TargetID("t"),
            context_id=ContextID("c"),
            nu_raw=0.5,
            nu_penalties={PenaltySource.CONFLICT: 0.15},
            metadata=Metadata(
                penalty_clear_start=0.0,  # long ago
            ),
        )
        policy = Policy(
            penalty_decay_enabled=True,
            penalty_decay_factor=0.9,
            penalty_clear_window=86400.0,
        )
        new_state, _ = penalty_decay(state, policy, clock)
        assert new_state.nu_penalties[PenaltySource.CONFLICT] == pytest.approx(0.135)

    def test_conflict_no_decay_before_window(self) -> None:
        clock = MockClock(1000.0)
        state = State(
            target_id=TargetID("t"),
            context_id=ContextID("c"),
            nu_raw=0.5,
            nu_penalties={PenaltySource.CONFLICT: 0.15},
            metadata=Metadata(
                penalty_clear_start=500.0,  # only 500s ago
            ),
        )
        policy = Policy(
            penalty_decay_enabled=True,
            penalty_clear_window=86400.0,
        )
        new_state, _ = penalty_decay(state, policy, clock)
        # Conflict penalty unchanged (within clear window)
        assert new_state.nu_penalties[PenaltySource.CONFLICT] == 0.15

    def test_cleanup_below_threshold(self) -> None:
        clock = MockClock(100000.0)
        state = State(
            target_id=TargetID("t"),
            context_id=ContextID("c"),
            nu_raw=0.5,
            nu_penalties={PenaltySource.CONFLICT: 0.005},
            metadata=Metadata(penalty_clear_start=0.0),
        )
        policy = Policy(
            penalty_decay_enabled=True,
            penalty_decay_factor=0.9,
            penalty_cleanup_threshold=0.01,
            penalty_clear_window=86400.0,
        )
        new_state, _ = penalty_decay(state, policy, clock)
        assert PenaltySource.CONFLICT not in new_state.nu_penalties

    def test_non_conflict_penalty_decays(self) -> None:
        clock = MockClock(100.0)
        state = State(
            target_id=TargetID("t"),
            context_id=ContextID("c"),
            nu_raw=0.5,
            nu_penalties={PenaltySource.SCOPE_EXPANSION: 0.1},
        )
        policy = Policy(
            penalty_decay_enabled=True,
            penalty_decay_factor=0.9,
        )
        new_state, _ = penalty_decay(state, policy, clock)
        assert new_state.nu_penalties[PenaltySource.SCOPE_EXPANSION] == pytest.approx(0.09)
