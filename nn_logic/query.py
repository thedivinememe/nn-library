"""Query, DecisionQuery, is_licensed, and determine_reason."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from nn_logic.types import (
    ContextID,
    NullStatus,
    PenaltyMode,
    PenaltySource,
    QueryResult,
    Reason,
    State,
    TargetID,
)
from nn_logic.helpers import compute_nu
from nn_logic.policy import PI_DEFAULT, Policy


def is_licensed(state: State, policy: Policy = PI_DEFAULT) -> bool:
    """Check if truth evaluation is licensed.

    Two conditions must BOTH hold:
        ν_raw ≤ θ_eval_raw  AND  ν ≤ θ_eval

    This distinguishes "structurally vague" from "clear but penalized."
    """
    nu = state.nu_with_mode(policy.penalty_mode)
    return state.nu_raw <= policy.theta_eval_raw and nu <= policy.theta_eval


def determine_reason(state: State, policy: Policy = PI_DEFAULT) -> Reason:
    """Determine why a state is or isn't licensed."""
    nu = state.nu_with_mode(policy.penalty_mode)

    if state.nu_raw <= policy.theta_eval_raw and nu <= policy.theta_eval:
        return Reason.LICENSED
    elif state.nu_raw <= policy.theta_eval_raw and nu > policy.theta_eval:
        return Reason.CLEAR_BUT_PENALIZED
    else:
        return Reason.STRUCTURALLY_VAGUE


def null_status(state: State, policy: Policy = PI_DEFAULT) -> NullStatus:
    """Determine the null status of a state."""
    nu = state.nu_with_mode(policy.penalty_mode)

    if nu <= policy.theta_defined:
        return NullStatus.NOT_NULL
    elif nu >= policy.theta_null:
        return NullStatus.NULL
    else:
        return NullStatus.INDETERMINATE


@dataclass(frozen=True)
class QueryResponse:
    target_id: TargetID
    context_id: ContextID
    licensed: bool
    reason: Reason
    status: NullStatus
    nu: float
    nu_raw: float
    penalties: dict[PenaltySource, float]


def query(state: State, policy: Policy = PI_DEFAULT) -> QueryResponse:
    """Full query response for a state."""
    nu = state.nu_with_mode(policy.penalty_mode)
    return QueryResponse(
        target_id=state.target_id,
        context_id=state.context_id,
        licensed=is_licensed(state, policy),
        reason=determine_reason(state, policy),
        status=null_status(state, policy),
        nu=nu,
        nu_raw=state.nu_raw,
        penalties=dict(state.nu_penalties),
    )


@dataclass(frozen=True)
class DecisionQuery:
    """Query for decision support (normative/utility extension stub)."""
    target_id: TargetID
    context_id: ContextID
    options: tuple[str, ...] = ()
    # Utility computation is stubbed per spec §4.4
    utility_scores: dict[str, float] = field(default_factory=dict)

    def best_option(self) -> Optional[str]:
        if not self.utility_scores:
            return None
        return max(self.utility_scores, key=self.utility_scores.get)  # type: ignore[arg-type]
