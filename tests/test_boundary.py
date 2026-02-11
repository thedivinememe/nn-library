"""Tests for boundary transform."""

from __future__ import annotations

import pytest

from nn_logic.types import (
    AgentID,
    Evidence,
    EvidenceID,
    EvidenceKind,
    EvidenceSet,
    Role,
    TargetID,
)
from nn_logic.boundary import boundary_transform, boundary_transform_evidence
from nn_logic.policy import PI_DEFAULT, Policy


def _e(src: str = "agent_a", trust: float = 1.0) -> Evidence:
    return Evidence(
        id=EvidenceID("e1"),
        kind=EvidenceKind.EPISTEMIC,
        claim="test",
        valence=0.5,
        src=AgentID(src),
        time=0.0,
        trust=trust,
    )


class TestBoundaryTransformEvidence:
    def test_i_role_passthrough(self) -> None:
        e = _e(trust=0.8)
        result = boundary_transform_evidence(e, Role.I, PI_DEFAULT)
        assert result.trust == 0.8

    def test_not_i_trust_adjustment(self) -> None:
        e = _e(trust=1.0)
        result = boundary_transform_evidence(e, Role.NOT_I, PI_DEFAULT)
        assert result.trust == pytest.approx(PI_DEFAULT.not_i_trust_factor)

    def test_both_coalition_factor(self) -> None:
        e = _e(trust=1.0)
        result = boundary_transform_evidence(e, Role.BOTH, PI_DEFAULT)
        assert result.trust == pytest.approx(PI_DEFAULT.coalition_factor)

    def test_unknown_discount(self) -> None:
        e = _e(trust=1.0)
        result = boundary_transform_evidence(e, Role.UNKNOWN, PI_DEFAULT)
        assert result.trust == pytest.approx(PI_DEFAULT.unknown_trust_factor)

    def test_unknown_is_most_discounted(self) -> None:
        e = _e(trust=1.0)
        p = PI_DEFAULT
        u = boundary_transform_evidence(e, Role.UNKNOWN, p).trust
        n = boundary_transform_evidence(e, Role.NOT_I, p).trust
        assert u < n


class TestBoundaryTransform:
    def test_empty_set(self) -> None:
        result = boundary_transform(EvidenceSet.empty(), {}, PI_DEFAULT)
        assert len(result) == 0

    def test_role_based_transform(self) -> None:
        e1 = Evidence(
            id=EvidenceID("e1"), kind=EvidenceKind.EPISTEMIC, claim="a",
            valence=0.5, src=AgentID("internal"), time=0.0, trust=1.0,
        )
        e2 = Evidence(
            id=EvidenceID("e2"), kind=EvidenceKind.EPISTEMIC, claim="b",
            valence=0.5, src=AgentID("external"), time=0.0, trust=1.0,
        )
        es = EvidenceSet(items=(e1, e2))
        roles = {AgentID("internal"): Role.I, AgentID("external"): Role.NOT_I}

        result = boundary_transform(es, roles, PI_DEFAULT)
        items = list(result)
        assert items[0].trust == 1.0  # I role unchanged
        assert items[1].trust == pytest.approx(PI_DEFAULT.not_i_trust_factor)

    def test_unknown_agent_gets_unknown_role(self) -> None:
        e = _e(src="mystery_agent")
        es = EvidenceSet(items=(e,))
        result = boundary_transform(es, {}, PI_DEFAULT)
        items = list(result)
        assert items[0].trust == pytest.approx(PI_DEFAULT.unknown_trust_factor)
