"""Tests for Query module."""

from __future__ import annotations

import pytest

from nn_logic.types import (
    ContextID,
    NullStatus,
    PenaltySource,
    State,
    TargetID,
)
from nn_logic.query import (
    DecisionQuery,
    null_status,
    query,
    QueryResponse,
)
from nn_logic.policy import Policy


POLICY = Policy(theta_eval=0.4, theta_eval_raw=0.5, theta_defined=0.3, theta_null=0.7)


class TestNullStatus:
    def test_not_null(self) -> None:
        state = State(target_id=TargetID("t"), context_id=ContextID("c"), nu_raw=0.2)
        assert null_status(state, POLICY) == NullStatus.NOT_NULL

    def test_null(self) -> None:
        state = State(target_id=TargetID("t"), context_id=ContextID("c"), nu_raw=0.8)
        assert null_status(state, POLICY) == NullStatus.NULL

    def test_indeterminate(self) -> None:
        state = State(target_id=TargetID("t"), context_id=ContextID("c"), nu_raw=0.5)
        assert null_status(state, POLICY) == NullStatus.INDETERMINATE


class TestQuery:
    def test_full_query_response(self) -> None:
        state = State(
            target_id=TargetID("t"), context_id=ContextID("c"),
            nu_raw=0.3, nu_penalties={},
        )
        resp = query(state, POLICY)
        assert isinstance(resp, QueryResponse)
        assert resp.licensed is True
        assert resp.nu_raw == 0.3

    def test_query_with_penalties(self) -> None:
        state = State(
            target_id=TargetID("t"), context_id=ContextID("c"),
            nu_raw=0.3, nu_penalties={PenaltySource.CONFLICT: 0.15},
        )
        resp = query(state, POLICY)
        assert resp.licensed is False
        assert resp.nu == pytest.approx(0.45)


class TestDecisionQuery:
    def test_best_option(self) -> None:
        dq = DecisionQuery(
            target_id=TargetID("t"),
            context_id=ContextID("c"),
            options=("a", "b", "c"),
            utility_scores={"a": 0.3, "b": 0.8, "c": 0.5},
        )
        assert dq.best_option() == "b"

    def test_no_scores(self) -> None:
        dq = DecisionQuery(
            target_id=TargetID("t"),
            context_id=ContextID("c"),
        )
        assert dq.best_option() is None
