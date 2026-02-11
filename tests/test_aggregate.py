"""Tests for aggregate function and compute_conflict."""

from __future__ import annotations

import pytest

from nn_logic.types import (
    AgentID,
    ContextID,
    Evidence,
    EvidenceID,
    EvidenceKind,
    EvidenceSet,
    TargetID,
)
from nn_logic.aggregate import aggregate, compute_conflict


TID = TargetID("t")
CID = ContextID("c")


def _e(eid: str, valence: float, trust: float = 1.0, t: float = 0.0) -> Evidence:
    return Evidence(
        id=EvidenceID(eid),
        kind=EvidenceKind.EPISTEMIC,
        claim=f"claim_{eid}",
        valence=valence,
        src=AgentID("a"),
        time=t,
        trust=trust,
    )


class TestComputeConflict:
    def test_no_mass(self) -> None:
        assert compute_conflict(0.0, 0.0) == 0.0

    def test_only_positive(self) -> None:
        assert compute_conflict(1.0, 0.0) == 0.0

    def test_only_negative(self) -> None:
        assert compute_conflict(0.0, 1.0) == 0.0

    def test_balanced(self) -> None:
        # 2 * min(1, 1) / (1 + 1) = 1.0
        assert compute_conflict(1.0, 1.0) == pytest.approx(1.0)

    def test_asymmetric(self) -> None:
        # 2 * min(0.3, 0.7) / (0.3 + 0.7) = 0.6 / 1.0 = 0.6
        assert compute_conflict(0.3, 0.7) == pytest.approx(0.6)


class TestAggregate:
    def test_empty_evidence(self) -> None:
        result = aggregate(EvidenceSet.empty(), TID, CID)
        assert result.pos_mass == 0.0
        assert result.neg_mass == 0.0
        assert result.conflict == 0.0

    def test_all_positive(self) -> None:
        es = EvidenceSet(items=(_e("a", 0.5), _e("b", 0.3)))
        result = aggregate(es, TID, CID)
        assert result.pos_mass == pytest.approx(0.8)
        assert result.neg_mass == 0.0
        assert result.conflict == 0.0

    def test_all_negative(self) -> None:
        es = EvidenceSet(items=(_e("a", -0.5), _e("b", -0.3)))
        result = aggregate(es, TID, CID)
        assert result.pos_mass == 0.0
        assert result.neg_mass == pytest.approx(0.8)
        assert result.conflict == 0.0

    def test_single_item(self) -> None:
        es = EvidenceSet(items=(_e("a", 0.7),))
        result = aggregate(es, TID, CID)
        assert result.pos_mass == pytest.approx(0.7)
        assert result.neg_mass == 0.0

    def test_mixed_with_conflict(self) -> None:
        es = EvidenceSet(items=(_e("a", 0.5), _e("b", -0.5)))
        result = aggregate(es, TID, CID)
        assert result.pos_mass == pytest.approx(0.5)
        assert result.neg_mass == pytest.approx(0.5)
        assert result.conflict == pytest.approx(1.0)

    def test_relevance_filtering(self) -> None:
        es = EvidenceSet(items=(_e("a", 0.5), _e("b", -0.5)))

        def only_positive(e: Evidence, t: TargetID, c: ContextID) -> float:
            return 1.0 if e.valence > 0 else 0.0

        result = aggregate(es, TID, CID, relevance_fn=only_positive)
        assert result.pos_mass == pytest.approx(0.5)
        assert result.neg_mass == 0.0
        assert result.conflict == 0.0

    def test_decay_weighting(self) -> None:
        es = EvidenceSet(items=(_e("a", 0.5, t=0.0),))
        # With high decay rate and late "now", evidence should be discounted
        result = aggregate(es, TID, CID, now=1000.0, decay_rate=0.01)
        assert result.pos_mass < 0.5
        assert result.pos_mass > 0.0

    def test_trust_weighting(self) -> None:
        es = EvidenceSet(items=(_e("a", 0.5, trust=0.5),))
        result = aggregate(es, TID, CID)
        assert result.pos_mass == pytest.approx(0.25)
