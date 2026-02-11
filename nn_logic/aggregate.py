"""Aggregate function and compute_conflict for N/N-N Logic."""

from __future__ import annotations

import math

from nn_logic.types import (
    AggregateResult,
    ContextID,
    Evidence,
    EvidenceSet,
    RelevanceFn,
    TargetID,
    default_relevance_fn,
)


def compute_conflict(pos_mass: float, neg_mass: float) -> float:
    total = pos_mass + neg_mass
    if total == 0.0:
        return 0.0
    return 2.0 * min(pos_mass, neg_mass) / total


def aggregate(
    evidence: EvidenceSet,
    target: TargetID,
    context: ContextID,
    relevance_fn: RelevanceFn = default_relevance_fn,
    now: float = 0.0,
    decay_rate: float = 0.0,
) -> AggregateResult:
    pos_mass = 0.0
    neg_mass = 0.0
    total_weight = 0.0

    for e in evidence:
        relevance = relevance_fn(e, target, context)
        if relevance <= 0.0:
            continue

        # Apply time decay
        age = max(0.0, now - e.time)
        decay = math.exp(-decay_rate * age) if decay_rate > 0.0 else 1.0

        weighted = abs(e.valence) * e.trust * relevance * decay

        if e.valence >= 0.0:
            pos_mass += weighted
        else:
            neg_mass += weighted

        total_weight += weighted

    conflict = compute_conflict(pos_mass, neg_mass)

    # Def_ep: epistemic definedness from evidence mass
    # More evidence → lower vagueness. Normalized by a soft cap.
    def_ep = min(1.0, total_weight / 2.0) if total_weight > 0.0 else 0.0

    return AggregateResult(
        pos_mass=pos_mass,
        neg_mass=neg_mass,
        conflict=conflict,
        def_ep=def_ep,
    )
