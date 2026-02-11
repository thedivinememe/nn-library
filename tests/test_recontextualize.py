"""Tests for Recontextualize operator."""

from __future__ import annotations

import pytest

from nn_logic.types import (
    ContextID,
    MockClock,
    State,
    TargetID,
)
from nn_logic.operators import recontextualize
from nn_logic.policy import Policy


class TestRecontextualize:
    def test_context_change(self) -> None:
        clock = MockClock(100.0)
        state = State(
            target_id=TargetID("t"),
            context_id=ContextID("old_ctx"),
            nu_raw=0.5,
        )
        new_state, record = recontextualize(
            state, ContextID("new_ctx"), clock=clock,
        )
        assert new_state.context_id == ContextID("new_ctx")
        assert record.operator == "recontextualize"

    def test_crossing_recorded(self) -> None:
        clock = MockClock(100.0)
        state = State(
            target_id=TargetID("t"),
            context_id=ContextID("ctx_a"),
            nu_raw=0.5,
        )
        new_state, _ = recontextualize(
            state, ContextID("ctx_b"), clock=clock,
        )
        assert "ctx_a->ctx_b" in new_state.metadata.crossings

    def test_history_updated(self) -> None:
        clock = MockClock(100.0)
        state = State(
            target_id=TargetID("t"),
            context_id=ContextID("ctx_a"),
            nu_raw=0.5,
        )
        new_state, _ = recontextualize(
            state, ContextID("ctx_b"), clock=clock,
        )
        assert "recontextualize" in new_state.metadata.history
