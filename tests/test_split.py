"""Tests for Split operator."""

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
    PenaltySource,
    State,
    TargetID,
)
from nn_logic.operators import split
from nn_logic.policy import Policy


def _make_parent() -> State:
    e1 = Evidence(
        id=EvidenceID("e1"), kind=EvidenceKind.EPISTEMIC, claim="a",
        valence=0.5, src=AgentID("a"), time=100.0,
    )
    e2 = Evidence(
        id=EvidenceID("e2"), kind=EvidenceKind.EPISTEMIC, claim="b",
        valence=-0.3, src=AgentID("b"), time=100.0,
    )
    return State(
        target_id=TargetID("parent"),
        context_id=ContextID("c"),
        nu_raw=0.6,
        nu_penalties={PenaltySource.CONFLICT: 0.1},
        evidence=EvidenceSet(items=(e1, e2)),
        constraints=("c1", "c2"),
    )


class TestSplit:
    def test_evidence_copying_not_partitioning(self) -> None:
        parent = _make_parent()
        clock = MockClock(200.0)
        children, _ = split(
            parent,
            [TargetID("ch1"), TargetID("ch2")],
            {},
            clock=clock,
        )
        # Each child gets full copy of parent evidence
        for child in children:
            assert len(child.evidence) == len(parent.evidence)

    def test_fresh_penalties(self) -> None:
        parent = _make_parent()
        assert PenaltySource.CONFLICT in parent.nu_penalties
        clock = MockClock(200.0)
        children, _ = split(
            parent,
            [TargetID("ch1"), TargetID("ch2")],
            {},
            clock=clock,
        )
        for child in children:
            assert child.nu_penalties == {}

    def test_parent_tagging(self) -> None:
        parent = _make_parent()
        clock = MockClock(200.0)
        children, _ = split(
            parent,
            [TargetID("ch1")],
            {},
            clock=clock,
        )
        assert children[0].metadata.tags.get("parent") == TargetID("parent")

    def test_correct_child_count(self) -> None:
        parent = _make_parent()
        clock = MockClock(200.0)
        children, records = split(
            parent,
            [TargetID("c1"), TargetID("c2"), TargetID("c3")],
            {},
            clock=clock,
        )
        assert len(children) == 3
        assert len(records) == 3

    def test_relevance_override(self) -> None:
        parent = _make_parent()
        clock = MockClock(200.0)

        def only_positive(e: Evidence, t: TargetID, c: ContextID) -> float:
            return 1.0 if e.valence > 0 else 0.0

        children, _ = split(
            parent,
            [TargetID("pos_child"), TargetID("all_child")],
            {TargetID("pos_child"): only_positive},
            clock=clock,
        )
        assert children[0].metadata.tags.get("relevance_override") is True

    def test_constraints_inherited(self) -> None:
        parent = _make_parent()
        clock = MockClock(200.0)
        children, _ = split(
            parent,
            [TargetID("ch1")],
            {},
            clock=clock,
        )
        assert children[0].constraints == parent.constraints

    def test_records_have_child_details(self) -> None:
        parent = _make_parent()
        clock = MockClock(200.0)
        _, records = split(
            parent,
            [TargetID("ch1"), TargetID("ch2")],
            {},
            clock=clock,
        )
        assert records[0].details["child_id"] == TargetID("ch1")
        assert records[1].details["child_id"] == TargetID("ch2")
