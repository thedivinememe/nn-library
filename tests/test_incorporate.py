"""Tests for Incorporate operator."""

from __future__ import annotations

import pytest

from nn_logic.types import (
    AgentID,
    ContextID,
    Evidence,
    EvidenceID,
    EvidenceKind,
    EvidenceSet,
    MockClock,
    Role,
    State,
    TargetID,
)
from nn_logic.operators import incorporate
from nn_logic.policy import PI_DEFAULT, Policy
from nn_logic.state import make_initial_state


def _e(eid: str, valence: float = 0.5, kind: EvidenceKind = EvidenceKind.EPISTEMIC) -> Evidence:
    return Evidence(
        id=EvidenceID(eid), kind=kind, claim=f"claim_{eid}",
        valence=valence, src=AgentID("agent"), time=100.0,
    )


class TestIncorporate:
    def test_basic_add(self) -> None:
        clock = MockClock(100.0)
        state = make_initial_state(TargetID("t"), ContextID("c"), clock)
        e = _e("e1", 0.5, EvidenceKind.EPISTEMIC)
        new_state, record = incorporate(state, [e], clock=clock)
        assert len(new_state.evidence) == 1
        assert record.operator == "incorporate"

    def test_adds_multiple(self) -> None:
        clock = MockClock(100.0)
        state = make_initial_state(TargetID("t"), ContextID("c"), clock)
        ev = [_e("e1"), _e("e2"), _e("e3")]
        new_state, _ = incorporate(state, ev, clock=clock)
        assert len(new_state.evidence) == 3

    def test_dedup_strict(self) -> None:
        clock = MockClock(100.0)
        state = make_initial_state(TargetID("t"), ContextID("c"), clock)
        e = _e("e1")
        state, _ = incorporate(state, [e], clock=clock)
        # Incorporate same evidence again
        state, _ = incorporate(state, [e], clock=clock)
        assert len(state.evidence) == 1  # deduped

    def test_nu_raw_decreases_with_evidence(self) -> None:
        clock = MockClock(100.0)
        state = make_initial_state(TargetID("t"), ContextID("c"), clock)
        assert state.nu_raw == 1.0

        ev = [
            _e("e1", 0.5, EvidenceKind.EPISTEMIC),
            _e("e2", 0.3, EvidenceKind.DEFINITIONAL),
        ]
        new_state, _ = incorporate(state, ev, clock=clock)
        assert new_state.nu_raw < state.nu_raw

    def test_boundary_transform_integration(self) -> None:
        clock = MockClock(100.0)
        state = make_initial_state(TargetID("t"), ContextID("c"), clock)
        e = Evidence(
            id=EvidenceID("e_ext"), kind=EvidenceKind.EPISTEMIC, claim="ext claim",
            valence=0.5, src=AgentID("external"), time=100.0, trust=1.0,
        )
        roles = {AgentID("external"): Role.NOT_I}
        new_state, _ = incorporate(state, [e], roles=roles, clock=clock)
        # Trust should be reduced by not_i_trust_factor
        added = list(new_state.evidence)[0]
        assert added.trust == pytest.approx(PI_DEFAULT.not_i_trust_factor)

    def test_record_emitted(self) -> None:
        clock = MockClock(100.0)
        state = make_initial_state(TargetID("t"), ContextID("c"), clock)
        new_state, record = incorporate(state, [_e("e1")], clock=clock)
        assert record.nu_raw_before == 1.0
        assert record.nu_raw_after == new_state.nu_raw
        assert record.operator == "incorporate"

    def test_history_updated(self) -> None:
        clock = MockClock(100.0)
        state = make_initial_state(TargetID("t"), ContextID("c"), clock)
        new_state, _ = incorporate(state, [_e("e1")], clock=clock)
        assert "incorporate" in new_state.metadata.history
