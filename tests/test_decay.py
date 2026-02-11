"""Tests for Decay operator."""

from __future__ import annotations

import pytest

from nn_logic.types import (
    ContextID,
    MockClock,
    State,
    TargetID,
)
from nn_logic.operators import decay
from nn_logic.policy import Policy


class TestDecay:
    def test_decay_emits_record(self) -> None:
        clock = MockClock(100.0)
        state = State(
            target_id=TargetID("t"),
            context_id=ContextID("c"),
            nu_raw=0.5,
        )
        new_state, record = decay(state, clock=clock)
        assert record.operator == "decay"

    def test_history_updated(self) -> None:
        clock = MockClock(100.0)
        state = State(
            target_id=TargetID("t"),
            context_id=ContextID("c"),
            nu_raw=0.5,
        )
        new_state, _ = decay(state, clock=clock)
        assert "decay" in new_state.metadata.history
