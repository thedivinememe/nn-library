"""Tests for Conflict operator."""

from __future__ import annotations

import pytest

from nn_logic.types import (
    AgentID,
    ContextID,
    Evidence,
    EvidenceID,
    EvidenceKind,
    EvidenceSet,
    Metadata,
    MockClock,
    PenaltyMode,
    PenaltySource,
    State,
    TargetID,
)
from nn_logic.operators import apply_conflict
from nn_logic.policy import Policy


def _state_with_evidence(
    pos_valence: float = 0.5,
    neg_valence: float = -0.5,
    clock_time: float = 1000.0,
) -> tuple[State, MockClock]:
    clock = MockClock(clock_time)
    e1 = Evidence(
        id=EvidenceID("pos"), kind=EvidenceKind.EPISTEMIC, claim="good",
        valence=pos_valence, src=AgentID("a"), time=clock_time,
    )
    e2 = Evidence(
        id=EvidenceID("neg"), kind=EvidenceKind.EPISTEMIC, claim="bad",
        valence=neg_valence, src=AgentID("b"), time=clock_time,
    )
    state = State(
        target_id=TargetID("t"),
        context_id=ContextID("c"),
        nu_raw=0.6,
        nu_penalties={},
        evidence=EvidenceSet(items=(e1, e2)),
        metadata=Metadata(creation=clock_time, last_modified=clock_time),
    )
    return state, clock


class TestConflict:
    def test_no_conflict_no_penalty(self) -> None:
        clock = MockClock(1000.0)
        e = Evidence(
            id=EvidenceID("e1"), kind=EvidenceKind.EPISTEMIC, claim="good",
            valence=0.5, src=AgentID("a"), time=1000.0,
        )
        state = State(
            target_id=TargetID("t"),
            context_id=ContextID("c"),
            nu_raw=0.6,
            evidence=EvidenceSet(items=(e,)),
            metadata=Metadata(creation=1000.0, last_modified=1000.0),
        )
        policy = Policy(theta_conflict=0.3)
        new_state, _, agg = apply_conflict(state, policy, clock)
        assert PenaltySource.CONFLICT not in new_state.nu_penalties
        assert agg.conflict == 0.0

    def test_conflict_applies_penalty(self) -> None:
        state, clock = _state_with_evidence(0.5, -0.5)
        policy = Policy(theta_conflict=0.3, max_conflict_penalty=0.2)
        new_state, _, agg = apply_conflict(state, policy, clock)
        assert agg.conflict == pytest.approx(1.0)
        assert PenaltySource.CONFLICT in new_state.nu_penalties
        assert new_state.nu_penalties[PenaltySource.CONFLICT] <= policy.max_conflict_penalty

    def test_cooldown_enforcement(self) -> None:
        state, clock = _state_with_evidence()
        policy = Policy(theta_conflict=0.3, max_conflict_penalty=0.2, conflict_cooldown=3600.0)

        # First application
        state, _, _ = apply_conflict(state, policy, clock)
        penalty1 = state.nu_penalties.get(PenaltySource.CONFLICT, 0.0)
        assert penalty1 > 0

        # Advance time within cooldown
        clock.advance(100.0)
        state, _, _ = apply_conflict(state, policy, clock)
        penalty2 = state.nu_penalties.get(PenaltySource.CONFLICT, 0.0)
        # Penalty should not change (cooldown active)
        assert penalty2 == pytest.approx(penalty1)

    def test_cooldown_expires(self) -> None:
        state, clock = _state_with_evidence()
        policy = Policy(theta_conflict=0.3, max_conflict_penalty=0.2, conflict_cooldown=3600.0)

        state, _, _ = apply_conflict(state, policy, clock)

        # Advance past cooldown
        clock.advance(4000.0)
        state, _, _ = apply_conflict(state, policy, clock)
        # Should have re-applied (conflict_last_applied updated)
        assert state.metadata.conflict_last_applied == pytest.approx(5000.0)

    def test_penalty_clear_start_on_resolve(self) -> None:
        clock = MockClock(1000.0)
        # Start with conflict penalty but no conflicting evidence
        e = Evidence(
            id=EvidenceID("e1"), kind=EvidenceKind.EPISTEMIC, claim="good",
            valence=0.5, src=AgentID("a"), time=1000.0,
        )
        state = State(
            target_id=TargetID("t"),
            context_id=ContextID("c"),
            nu_raw=0.6,
            nu_penalties={PenaltySource.CONFLICT: 0.1},
            evidence=EvidenceSet(items=(e,)),
            metadata=Metadata(creation=1000.0, last_modified=1000.0),
        )
        policy = Policy(theta_conflict_clear=0.15)
        new_state, _, agg = apply_conflict(state, policy, clock)
        assert agg.conflict == 0.0  # no conflict
        assert new_state.metadata.penalty_clear_start is not None
