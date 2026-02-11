"""Tests for Merge operator."""

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
from nn_logic.operators import merge
from nn_logic.policy import Policy


def _e(eid: str, valence: float = 0.5) -> Evidence:
    return Evidence(
        id=EvidenceID(eid), kind=EvidenceKind.EPISTEMIC, claim=f"claim_{eid}",
        valence=valence, src=AgentID("a"), time=100.0,
    )


class TestMerge:
    def test_evidence_union(self) -> None:
        s1 = State(
            target_id=TargetID("t1"), context_id=ContextID("c"),
            evidence=EvidenceSet(items=(_e("e1"),)),
        )
        s2 = State(
            target_id=TargetID("t2"), context_id=ContextID("c"),
            evidence=EvidenceSet(items=(_e("e2"),)),
        )
        clock = MockClock(200.0)
        result, _ = merge([s1, s2], TargetID("merged"), ContextID("c"), clock=clock)
        assert len(result.evidence) == 2

    def test_evidence_dedup_on_merge(self) -> None:
        s1 = State(
            target_id=TargetID("t1"), context_id=ContextID("c"),
            evidence=EvidenceSet(items=(_e("e1"),)),
        )
        s2 = State(
            target_id=TargetID("t2"), context_id=ContextID("c"),
            evidence=EvidenceSet(items=(_e("e1"),)),  # same evidence
        )
        clock = MockClock(200.0)
        result, _ = merge([s1, s2], TargetID("merged"), ContextID("c"), clock=clock)
        assert len(result.evidence) == 1  # deduped via union

    def test_conflict_detection_on_merge(self) -> None:
        s1 = State(
            target_id=TargetID("t1"), context_id=ContextID("c"),
            evidence=EvidenceSet(items=(_e("pos", 0.5),)),
        )
        s2 = State(
            target_id=TargetID("t2"), context_id=ContextID("c"),
            evidence=EvidenceSet(items=(_e("neg", -0.5),)),
        )
        clock = MockClock(200.0)
        policy = Policy(theta_conflict=0.3, max_conflict_penalty=0.2)
        result, _ = merge([s1, s2], TargetID("merged"), ContextID("c"), policy=policy, clock=clock)
        assert PenaltySource.MERGE_RUPTURE in result.nu_penalties

    def test_no_penalty_without_conflict(self) -> None:
        s1 = State(
            target_id=TargetID("t1"), context_id=ContextID("c"),
            evidence=EvidenceSet(items=(_e("e1", 0.5),)),
        )
        s2 = State(
            target_id=TargetID("t2"), context_id=ContextID("c"),
            evidence=EvidenceSet(items=(_e("e2", 0.3),)),
        )
        clock = MockClock(200.0)
        result, _ = merge([s1, s2], TargetID("merged"), ContextID("c"), clock=clock)
        assert PenaltySource.MERGE_RUPTURE not in result.nu_penalties

    def test_constraint_merge(self) -> None:
        s1 = State(
            target_id=TargetID("t1"), context_id=ContextID("c"),
            constraints=("c1", "c2"),
        )
        s2 = State(
            target_id=TargetID("t2"), context_id=ContextID("c"),
            constraints=("c2", "c3"),
        )
        clock = MockClock(200.0)
        result, _ = merge([s1, s2], TargetID("merged"), ContextID("c"), clock=clock)
        assert set(result.constraints) == {"c1", "c2", "c3"}

    def test_merge_tags(self) -> None:
        s1 = State(target_id=TargetID("t1"), context_id=ContextID("c"))
        s2 = State(target_id=TargetID("t2"), context_id=ContextID("c"))
        clock = MockClock(200.0)
        result, _ = merge([s1, s2], TargetID("merged"), ContextID("c"), clock=clock)
        assert "merged_from" in result.metadata.tags
