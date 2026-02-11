"""Tests for licensing: is_licensed, determine_reason."""

from __future__ import annotations

import pytest

from nn_logic.types import (
    ContextID,
    PenaltySource,
    State,
    TargetID,
)
from nn_logic.query import is_licensed, determine_reason, Reason
from nn_logic.policy import Policy


POLICY = Policy(theta_eval=0.4, theta_eval_raw=0.5)


class TestIsLicensed:
    def test_both_conditions_met(self) -> None:
        state = State(
            target_id=TargetID("t"), context_id=ContextID("c"),
            nu_raw=0.3, nu_penalties={},
        )
        assert is_licensed(state, POLICY)

    def test_nu_raw_too_high(self) -> None:
        state = State(
            target_id=TargetID("t"), context_id=ContextID("c"),
            nu_raw=0.6, nu_penalties={},
        )
        assert not is_licensed(state, POLICY)

    def test_nu_total_too_high(self) -> None:
        state = State(
            target_id=TargetID("t"), context_id=ContextID("c"),
            nu_raw=0.3, nu_penalties={PenaltySource.CONFLICT: 0.15},
        )
        # ν_raw=0.3 ≤ 0.5 ✓, ν=0.45 > 0.4 ✗
        assert not is_licensed(state, POLICY)

    def test_both_conditions_failed(self) -> None:
        state = State(
            target_id=TargetID("t"), context_id=ContextID("c"),
            nu_raw=0.7, nu_penalties={PenaltySource.CONFLICT: 0.1},
        )
        assert not is_licensed(state, POLICY)

    def test_boundary_case_exactly_at_threshold(self) -> None:
        state = State(
            target_id=TargetID("t"), context_id=ContextID("c"),
            nu_raw=0.5, nu_penalties={},
        )
        # ν_raw = 0.5 ≤ 0.5 ✓ (<=, not <)
        # ν = 0.5 > 0.4 ✗
        assert not is_licensed(state, POLICY)


class TestDetermineReason:
    def test_licensed(self) -> None:
        state = State(
            target_id=TargetID("t"), context_id=ContextID("c"),
            nu_raw=0.3, nu_penalties={},
        )
        assert determine_reason(state, POLICY) == Reason.LICENSED

    def test_structurally_vague(self) -> None:
        state = State(
            target_id=TargetID("t"), context_id=ContextID("c"),
            nu_raw=0.7, nu_penalties={},
        )
        assert determine_reason(state, POLICY) == Reason.STRUCTURALLY_VAGUE

    def test_clear_but_penalized(self) -> None:
        state = State(
            target_id=TargetID("t"), context_id=ContextID("c"),
            nu_raw=0.3, nu_penalties={PenaltySource.CONFLICT: 0.15},
        )
        assert determine_reason(state, POLICY) == Reason.CLEAR_BUT_PENALIZED
