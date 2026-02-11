"""Tests for NegDefine operator."""

from __future__ import annotations

import pytest

from nn_logic.types import (
    ContextID,
    EvidenceSet,
    MockClock,
    State,
    TargetID,
)
from nn_logic.operators import neg_define
from nn_logic.policy import Policy


class TestNegDefine:
    def test_constraint_addition(self) -> None:
        clock = MockClock(100.0)
        state = State(
            target_id=TargetID("t"),
            context_id=ContextID("c"),
            nu_raw=0.8,
            constraints=(),
        )
        new_state, record = neg_define(state, ["c1", "c2"], clock=clock)
        assert len(new_state.constraints) == 2
        assert "c1" in new_state.constraints
        assert "c2" in new_state.constraints

    def test_def_sem_improvement(self) -> None:
        clock = MockClock(100.0)
        state = State(
            target_id=TargetID("t"),
            context_id=ContextID("c"),
            nu_raw=0.8,
            constraints=(),
        )
        # More constraints → better Def_sem → lower ν_raw
        state1, _ = neg_define(state, ["c1"], clock=clock)
        state2, _ = neg_define(state, ["c1", "c2", "c3", "c4", "c5"], clock=clock)
        assert state2.nu_raw < state1.nu_raw

    def test_idempotence_with_same_constraints(self) -> None:
        clock = MockClock(100.0)
        state = State(
            target_id=TargetID("t"),
            context_id=ContextID("c"),
            nu_raw=0.8,
            constraints=(),
        )
        state1, _ = neg_define(state, ["c1", "c2"], clock=clock)
        # Apply same constraints again
        state2, _ = neg_define(state1, ["c1", "c2"], clock=clock)
        # Should be idempotent — same constraints, same ν_raw
        assert state2.nu_raw == pytest.approx(state1.nu_raw)
        assert len(state2.constraints) == 2

    def test_dedup_constraints(self) -> None:
        clock = MockClock(100.0)
        state = State(
            target_id=TargetID("t"),
            context_id=ContextID("c"),
            nu_raw=0.8,
            constraints=("c1",),
        )
        new_state, _ = neg_define(state, ["c1", "c2"], clock=clock)
        assert len(new_state.constraints) == 2  # c1 not duplicated

    def test_record_emitted(self) -> None:
        clock = MockClock(100.0)
        state = State(
            target_id=TargetID("t"),
            context_id=ContextID("c"),
            nu_raw=0.8,
            constraints=(),
        )
        _, record = neg_define(state, ["c1"], clock=clock)
        assert record.operator == "neg_define"

    def test_history_updated(self) -> None:
        clock = MockClock(100.0)
        state = State(
            target_id=TargetID("t"),
            context_id=ContextID("c"),
            nu_raw=0.8,
            constraints=(),
        )
        new_state, _ = neg_define(state, ["c1"], clock=clock)
        assert "neg_define" in new_state.metadata.history

    def test_injectable_def_sem(self) -> None:
        clock = MockClock(100.0)
        state = State(
            target_id=TargetID("t"),
            context_id=ContextID("c"),
            nu_raw=0.8,
            constraints=(),
        )

        def custom_sem(t: TargetID, ev: EvidenceSet, c: tuple[str, ...]) -> float:
            return 0.90

        def custom_ep(t: TargetID, ev: EvidenceSet) -> float:
            return 0.80

        def custom_proc(t: TargetID, ev: EvidenceSet) -> float:
            return 0.70

        policy = Policy(w_sem=0.4, w_ep=0.35, w_proc=0.25)
        new_state, _ = neg_define(
            state, ["c1"], policy=policy, clock=clock,
            def_sem_override=custom_sem,
            def_ep_override=custom_ep,
            def_proc_override=custom_proc,
        )
        # Def = 0.4*0.9 + 0.35*0.8 + 0.25*0.7 = 0.36 + 0.28 + 0.175 = 0.815
        # ν_raw = 1 - 0.815 = 0.185
        assert new_state.nu_raw == pytest.approx(0.185, abs=0.01)
