"""Policy with π_default — refinement policy configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nn_logic.types import (
    DedupMode,
    Evidence,
    PenaltyMode,
    RelevanceFn,
    TargetID,
    ContextID,
    default_relevance_fn,
)


@dataclass(frozen=True)
class Policy:
    # Evaluation thresholds
    theta_eval: float = 0.4
    theta_eval_raw: float = 0.5
    theta_null: float = 0.7
    theta_defined: float = 0.3
    theta_conflict: float = 0.3
    theta_conflict_clear: float = 0.15
    theta_utility: float = 0.5

    # Definedness weights
    w_sem: float = 0.4
    w_ep: float = 0.35
    w_proc: float = 0.25

    # Conflict / penalty
    max_conflict_penalty: float = 0.2
    conflict_cooldown: float = 3600.0  # 1 hour in seconds
    penalty_mode: PenaltyMode = PenaltyMode.MAX
    penalty_decay_enabled: bool = True
    penalty_decay_factor: float = 0.9
    penalty_clear_window: float = 86400.0  # 24 hours in seconds
    penalty_cleanup_threshold: float = 0.01

    # Boundary transform
    not_i_trust_factor: float = 0.7
    unknown_trust_factor: float = 0.5
    coalition_factor: float = 0.85

    # Dedup
    dedup_mode: DedupMode = DedupMode.STRICT
    time_bucket_granularity: float = 60.0  # seconds

    # Decay
    decay_rate: float = 0.001  # per-second decay rate for evidence staleness
    decay_halflife: float = 86400.0  # 24 hours

    # Relevance
    relevance_fn: RelevanceFn = field(default=default_relevance_fn)

    def replace(self, **kwargs: Any) -> Policy:
        d: dict[str, Any] = {}
        for f in self.__dataclass_fields__:
            d[f] = getattr(self, f)
        d.update(kwargs)
        return Policy(**d)


# Default policy singleton
PI_DEFAULT = Policy()
