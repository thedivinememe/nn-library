"""§13 Property tests — algebraic invariants for operators."""

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
    PenaltyMode,
    PenaltySource,
    State,
    TargetID,
)
from nn_logic.operators import (
    apply_conflict,
    decay,
    incorporate,
    neg_define,
    penalty_decay,
    split,
)
from nn_logic.helpers import clamp, compute_nu
from nn_logic.evidence import partition_by_kind
from nn_logic.policy import Policy


def _e(eid: str, valence: float = 0.5) -> Evidence:
    return Evidence(
        id=EvidenceID(eid), kind=EvidenceKind.EPISTEMIC, claim=f"claim_{eid}",
        valence=valence, src=AgentID("a"), time=100.0,
    )


def _state() -> State:
    return State(
        target_id=TargetID("t"),
        context_id=ContextID("c"),
        nu_raw=0.6,
        nu_penalties={PenaltySource.CONFLICT: 0.1},
        evidence=EvidenceSet(items=(_e("e1", 0.5), _e("e2", -0.3))),
    )


class TestInvariantI1:
    """I1: ν == clamp(ν_raw + ν_penalty, 0, 1) after every operation."""

    def _check_invariant(self, state: State) -> None:
        expected = compute_nu(state.nu_raw, state.nu_penalties, PenaltyMode.MAX)
        assert state.nu == pytest.approx(expected, abs=1e-9)

    def test_after_incorporate(self) -> None:
        clock = MockClock(100.0)
        state = _state()
        new_state, _ = incorporate(state, [_e("e3", 0.4)], clock=clock)
        self._check_invariant(new_state)

    def test_after_neg_define(self) -> None:
        clock = MockClock(100.0)
        state = _state()
        new_state, _ = neg_define(state, ["c1", "c2"], clock=clock)
        self._check_invariant(new_state)

    def test_after_conflict(self) -> None:
        clock = MockClock(100.0)
        state = _state()
        new_state, _, _ = apply_conflict(state, clock=clock)
        self._check_invariant(new_state)

    def test_after_split(self) -> None:
        clock = MockClock(100.0)
        state = _state()
        children, _ = split(state, [TargetID("c1"), TargetID("c2")], {}, clock=clock)
        for child in children:
            self._check_invariant(child)

    def test_after_penalty_decay(self) -> None:
        clock = MockClock(100.0)
        state = _state()
        new_state, _ = penalty_decay(state, clock=clock)
        self._check_invariant(new_state)

    def test_after_decay(self) -> None:
        clock = MockClock(100.0)
        state = _state()
        new_state, _ = decay(state, clock=clock)
        self._check_invariant(new_state)


class TestInvariantI2:
    """I2: Every ν change produces a RefinementRecord."""

    def test_incorporate_emits_record(self) -> None:
        clock = MockClock(100.0)
        state = _state()
        _, record = incorporate(state, [_e("e3")], clock=clock)
        assert record is not None
        assert record.operator == "incorporate"

    def test_neg_define_emits_record(self) -> None:
        clock = MockClock(100.0)
        state = _state()
        _, record = neg_define(state, ["c1"], clock=clock)
        assert record is not None

    def test_conflict_emits_record(self) -> None:
        clock = MockClock(100.0)
        state = _state()
        _, record, _ = apply_conflict(state, clock=clock)
        assert record is not None

    def test_split_emits_records(self) -> None:
        clock = MockClock(100.0)
        state = _state()
        _, records = split(state, [TargetID("c1")], {}, clock=clock)
        assert len(records) == 1


class TestInvariantI3:
    """I3: ν_penalties keys are always valid PenaltySource values."""

    def test_penalties_valid_after_conflict(self) -> None:
        clock = MockClock(100.0)
        state = State(
            target_id=TargetID("t"), context_id=ContextID("c"),
            nu_raw=0.6,
            evidence=EvidenceSet(items=(_e("pos", 0.5), _e("neg", -0.5))),
        )
        new_state, _, _ = apply_conflict(state, clock=clock)
        for key in new_state.nu_penalties:
            assert isinstance(key, PenaltySource)

    def test_penalties_valid_after_split(self) -> None:
        clock = MockClock(100.0)
        state = _state()
        children, _ = split(state, [TargetID("c1")], {}, clock=clock)
        for child in children:
            for key in child.nu_penalties:
                assert isinstance(key, PenaltySource)


class TestInvariantI4:
    """I4: Conflict penalty respects cooldown."""

    def test_cooldown_respected(self) -> None:
        clock = MockClock(1000.0)
        state = State(
            target_id=TargetID("t"), context_id=ContextID("c"),
            nu_raw=0.6,
            evidence=EvidenceSet(items=(_e("pos", 0.5), _e("neg", -0.5))),
        )
        policy = Policy(theta_conflict=0.3, conflict_cooldown=3600.0)

        # First application
        state, _, _ = apply_conflict(state, policy, clock)
        first_penalty = state.nu_penalties.get(PenaltySource.CONFLICT, 0.0)

        # Within cooldown
        clock.advance(100.0)
        state, _, _ = apply_conflict(state, policy, clock)
        second_penalty = state.nu_penalties.get(PenaltySource.CONFLICT, 0.0)

        assert second_penalty == pytest.approx(first_penalty)


class TestInvariantI5:
    """I5: Evidence partitions (epistemic, definitional, procedural) are disjoint."""

    def test_partitions_disjoint(self) -> None:
        e1 = Evidence(id=EvidenceID("ep"), kind=EvidenceKind.EPISTEMIC, claim="a",
                       valence=0.5, src=AgentID("a"), time=0.0)
        e2 = Evidence(id=EvidenceID("def"), kind=EvidenceKind.DEFINITIONAL, claim="b",
                       valence=0.5, src=AgentID("a"), time=0.0)
        e3 = Evidence(id=EvidenceID("proc"), kind=EvidenceKind.PROCEDURAL, claim="c",
                       valence=0.5, src=AgentID("a"), time=0.0)
        es = EvidenceSet(items=(e1, e2, e3))
        parts = partition_by_kind(es)

        all_ids: set[EvidenceID] = set()
        for kind in EvidenceKind:
            kind_ids = {e.id for e in parts[kind]}
            assert kind_ids.isdisjoint(all_ids)
            all_ids |= kind_ids
