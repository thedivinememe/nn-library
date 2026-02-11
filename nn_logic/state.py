"""State and Σ (information state) management."""

from __future__ import annotations

from typing import Any, Optional

from nn_logic.types import (
    Clock,
    ContextID,
    EvidenceSet,
    Metadata,
    PenaltySource,
    State,
    SystemClock,
    TargetID,
)


def make_initial_state(
    target_id: TargetID,
    context_id: ContextID,
    clock: Optional[Clock] = None,
) -> State:
    now = (clock or SystemClock()).now()
    return State(
        target_id=target_id,
        context_id=context_id,
        truth_status=None,
        nu_raw=1.0,
        nu_penalties={},
        evidence=EvidenceSet.empty(),
        metadata=Metadata(
            creation=now,
            last_modified=now,
            history=(),
            crossings=(),
            conflict_last_applied=None,
            penalty_clear_start=None,
            tags={},
        ),
        constraints=(),
    )


class InformationState:
    """Σ: (TargetID, ContextID) → State mapping with default initialization."""

    def __init__(self, clock: Optional[Clock] = None) -> None:
        self._store: dict[tuple[TargetID, ContextID], State] = {}
        self._clock = clock or SystemClock()

    def get(self, target_id: TargetID, context_id: ContextID) -> State:
        key = (target_id, context_id)
        if key not in self._store:
            self._store[key] = make_initial_state(target_id, context_id, self._clock)
        return self._store[key]

    def set(self, state: State) -> InformationState:
        new_sigma = InformationState(self._clock)
        new_sigma._store = dict(self._store)
        new_sigma._store[(state.target_id, state.context_id)] = state
        return new_sigma

    def remove(self, target_id: TargetID, context_id: ContextID) -> InformationState:
        new_sigma = InformationState(self._clock)
        new_sigma._store = dict(self._store)
        new_sigma._store.pop((target_id, context_id), None)
        return new_sigma

    def keys(self) -> list[tuple[TargetID, ContextID]]:
        return list(self._store.keys())

    def states(self) -> list[State]:
        return list(self._store.values())

    def __contains__(self, key: tuple[TargetID, ContextID]) -> bool:
        return key in self._store
