"""Shared fixtures for N/N-N Logic tests."""

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
from nn_logic.policy import PI_DEFAULT, Policy
from nn_logic.state import make_initial_state
from nn_logic.trace import Tracer


@pytest.fixture
def clock() -> MockClock:
    return MockClock(start=1000.0)


@pytest.fixture
def default_policy() -> Policy:
    return PI_DEFAULT


@pytest.fixture
def target_id() -> TargetID:
    return TargetID("feature_F")


@pytest.fixture
def context_id() -> ContextID:
    return ContextID("ctx_main")


@pytest.fixture
def initial_state(target_id: TargetID, context_id: ContextID, clock: MockClock) -> State:
    return make_initial_state(target_id, context_id, clock)


@pytest.fixture
def tracer() -> Tracer:
    return Tracer(enabled=True)


@pytest.fixture
def agent_internal() -> AgentID:
    return AgentID("agent_internal")


@pytest.fixture
def agent_external() -> AgentID:
    return AgentID("agent_external")


def make_test_evidence(
    kind: EvidenceKind = EvidenceKind.EPISTEMIC,
    claim: str = "test claim",
    valence: float = 0.5,
    src: str = "test_agent",
    t: float = 1000.0,
    trust: float = 1.0,
    eid: str = "e_test",
) -> Evidence:
    return Evidence(
        id=EvidenceID(eid),
        kind=kind,
        claim=claim,
        valence=valence,
        src=AgentID(src),
        time=t,
        trust=trust,
    )
