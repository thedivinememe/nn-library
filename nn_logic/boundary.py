"""Boundary transform — trust adjustment based on agent roles."""

from __future__ import annotations

from typing import Optional

from nn_logic.types import (
    AgentID,
    Evidence,
    EvidenceSet,
    Role,
    TargetID,
)
from nn_logic.policy import Policy


def boundary_transform_evidence(
    e: Evidence,
    role: Role,
    policy: Policy,
) -> Evidence:
    """Apply boundary transform to a single evidence item based on source role."""
    if role == Role.I:
        return e
    elif role == Role.NOT_I:
        new_trust = e.trust * policy.not_i_trust_factor
        return Evidence(
            id=e.id,
            kind=e.kind,
            claim=e.claim,
            valence=e.valence,
            src=e.src,
            time=e.time,
            trust=new_trust,
            metadata=e.metadata,
        )
    elif role == Role.BOTH:
        new_trust = e.trust * policy.coalition_factor
        return Evidence(
            id=e.id,
            kind=e.kind,
            claim=e.claim,
            valence=e.valence,
            src=e.src,
            time=e.time,
            trust=new_trust,
            metadata=e.metadata,
        )
    else:  # UNKNOWN
        new_trust = e.trust * policy.unknown_trust_factor
        return Evidence(
            id=e.id,
            kind=e.kind,
            claim=e.claim,
            valence=e.valence,
            src=e.src,
            time=e.time,
            trust=new_trust,
            metadata=e.metadata,
        )


def boundary_transform(
    evidence: EvidenceSet,
    roles: dict[AgentID, Role],
    policy: Policy,
    scope: Optional[frozenset[TargetID]] = None,
) -> EvidenceSet:
    """Apply boundary transform to all evidence based on agent roles.

    If scope is provided, only evidence relevant to targets in scope is included.
    """
    transformed: list[Evidence] = []
    for e in evidence:
        role = roles.get(e.src, Role.UNKNOWN)
        t = boundary_transform_evidence(e, role, policy)
        transformed.append(t)

    if not transformed:
        return EvidenceSet.empty()

    return EvidenceSet(items=tuple(transformed))
