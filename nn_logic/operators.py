"""All refinement operators for N/N-N Logic.

Each operator is a pure function: (State, ...) → (State, RefinementRecord).
Operators never mutate their inputs.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from nn_logic.types import (
    AgentID,
    AggregateResult,
    Clock,
    ContextID,
    Evidence,
    EvidenceID,
    EvidenceKind,
    EvidenceSet,
    Metadata,
    MockClock,
    PenaltyMode,
    PenaltySource,
    RelevanceFn,
    RefinementRecord,
    Role,
    State,
    TargetID,
    default_relevance_fn,
)
from nn_logic.aggregate import aggregate, compute_conflict
from nn_logic.boundary import boundary_transform
from nn_logic.definedness import (
    DefEpFn,
    DefProcFn,
    DefSemFn,
    definedness,
    nu_raw_from_definedness,
    SemanticDefinednessProvider,
)
from nn_logic.evidence import add_evidence
from nn_logic.helpers import clamp, compute_nu, make_refinement_record
from nn_logic.policy import PI_DEFAULT, Policy


# ---------- Incorporate (§8.1) ----------

def incorporate(
    state: State,
    new_evidence: list[Evidence],
    policy: Policy = PI_DEFAULT,
    roles: Optional[dict[AgentID, Role]] = None,
    clock: Optional[Clock] = None,
    relevance_fn: Optional[RelevanceFn] = None,
    sem_provider: Optional[SemanticDefinednessProvider] = None,
    def_sem_override: Optional[DefSemFn] = None,
    def_ep_override: Optional[DefEpFn] = None,
    def_proc_override: Optional[DefProcFn] = None,
) -> tuple[State, RefinementRecord]:
    """Incorporate new evidence into a state.

    1. Apply boundary transform based on roles
    2. Dedup and add to evidence set
    3. Recompute definedness and ν_raw
    4. Emit refinement record
    """
    now = (clock or MockClock(0.0)).now()
    rfn = relevance_fn or policy.relevance_fn

    # Apply boundary transform if roles provided
    if roles:
        transformed = boundary_transform(
            EvidenceSet(items=tuple(new_evidence)),
            roles,
            policy,
        )
        to_add = list(transformed)
    else:
        to_add = new_evidence

    # Add evidence with dedup
    updated_evidence = state.evidence
    for e in to_add:
        updated_evidence = add_evidence(updated_evidence, e, policy.dedup_mode)

    # Recompute definedness
    def_value = definedness(
        state.target_id,
        updated_evidence,
        state.constraints,
        policy.w_sem,
        policy.w_ep,
        policy.w_proc,
        sem_provider,
        def_sem_override,
        def_ep_override,
        def_proc_override,
    )
    new_nu_raw = nu_raw_from_definedness(def_value)

    new_state = state.replace(
        nu_raw=new_nu_raw,
        evidence=updated_evidence,
        metadata=state.metadata.with_update(
            last_modified=now,
            history=state.metadata.history + ("incorporate",),
        ),
    )

    record = make_refinement_record(state, new_state, "incorporate", now)
    return new_state, record


# ---------- NegDefine (§8.2) ----------

def neg_define(
    state: State,
    constraints: list[str],
    policy: Policy = PI_DEFAULT,
    clock: Optional[Clock] = None,
    sem_provider: Optional[SemanticDefinednessProvider] = None,
    def_sem_override: Optional[DefSemFn] = None,
    def_ep_override: Optional[DefEpFn] = None,
    def_proc_override: Optional[DefProcFn] = None,
) -> tuple[State, RefinementRecord]:
    """Add negative/constraining definitions.

    NegDefine adds constraints that narrow what a concept IS by specifying
    what it IS NOT or what boundaries it must satisfy.
    """
    now = (clock or MockClock(0.0)).now()

    # Merge constraints (dedup)
    existing = set(state.constraints)
    new_constraints = tuple(state.constraints) + tuple(
        c for c in constraints if c not in existing
    )

    # Recompute definedness with new constraints
    def_value = definedness(
        state.target_id,
        state.evidence,
        new_constraints,
        policy.w_sem,
        policy.w_ep,
        policy.w_proc,
        sem_provider,
        def_sem_override,
        def_ep_override,
        def_proc_override,
    )
    new_nu_raw = nu_raw_from_definedness(def_value)

    new_state = state.replace(
        nu_raw=new_nu_raw,
        constraints=new_constraints,
        metadata=state.metadata.with_update(
            last_modified=now,
            history=state.metadata.history + ("neg_define",),
        ),
    )

    record = make_refinement_record(state, new_state, "neg_define", now)
    return new_state, record


# ---------- Merge (§8.3) ----------

def merge(
    states: list[State],
    target_id: TargetID,
    context_id: ContextID,
    policy: Policy = PI_DEFAULT,
    clock: Optional[Clock] = None,
    relevance_fn: Optional[RelevanceFn] = None,
    sem_provider: Optional[SemanticDefinednessProvider] = None,
    def_sem_override: Optional[DefSemFn] = None,
    def_ep_override: Optional[DefEpFn] = None,
    def_proc_override: Optional[DefProcFn] = None,
) -> tuple[State, RefinementRecord]:
    """Merge multiple states into one.

    Evidence is unioned. Constraints are merged. Conflict is checked,
    and merge_rupture penalty applied if conflict is high.
    """
    now = (clock or MockClock(0.0)).now()
    rfn = relevance_fn or policy.relevance_fn

    # Union all evidence
    merged_evidence = EvidenceSet.empty()
    for s in states:
        merged_evidence = merged_evidence.union(s.evidence)

    # Merge constraints
    all_constraints: list[str] = []
    seen: set[str] = set()
    for s in states:
        for c in s.constraints:
            if c not in seen:
                seen.add(c)
                all_constraints.append(c)

    # Compute aggregate to check conflict
    agg = aggregate(merged_evidence, target_id, context_id, rfn, now, policy.decay_rate)

    # Compute definedness
    def_value = definedness(
        target_id,
        merged_evidence,
        tuple(all_constraints),
        policy.w_sem,
        policy.w_ep,
        policy.w_proc,
        sem_provider,
        def_sem_override,
        def_ep_override,
        def_proc_override,
    )
    new_nu_raw = nu_raw_from_definedness(def_value)

    # Apply merge_rupture penalty if conflict is high
    penalties: dict[PenaltySource, float] = {}
    if agg.conflict > policy.theta_conflict:
        penalty = min(policy.max_conflict_penalty, agg.conflict * policy.max_conflict_penalty)
        penalties[PenaltySource.MERGE_RUPTURE] = penalty

    # Create a dummy "before" state for record
    before_state = State(
        target_id=target_id,
        context_id=context_id,
        nu_raw=1.0,
        nu_penalties={},
        evidence=EvidenceSet.empty(),
    )

    new_state = State(
        target_id=target_id,
        context_id=context_id,
        nu_raw=new_nu_raw,
        nu_penalties=penalties,
        evidence=merged_evidence,
        constraints=tuple(all_constraints),
        metadata=Metadata(
            creation=now,
            last_modified=now,
            history=("merge",),
            tags={"merged_from": [s.target_id for s in states]},
        ),
    )

    record = make_refinement_record(before_state, new_state, "merge", now)
    return new_state, record


# ---------- Split (§8.8) ----------

def split(
    state: State,
    child_ids: list[TargetID],
    relevance_map: dict[TargetID, RelevanceFn],
    policy: Policy = PI_DEFAULT,
    clock: Optional[Clock] = None,
    sem_provider: Optional[SemanticDefinednessProvider] = None,
    def_sem_override: Optional[DefSemFn] = None,
    def_ep_override: Optional[DefEpFn] = None,
    def_proc_override: Optional[DefProcFn] = None,
    # Per-child overrides for golden test calibration
    per_child_def_sem: Optional[dict[TargetID, DefSemFn]] = None,
    per_child_def_ep: Optional[dict[TargetID, DefEpFn]] = None,
    per_child_def_proc: Optional[dict[TargetID, DefProcFn]] = None,
) -> tuple[list[State], list[RefinementRecord]]:
    """Split a parent state into multiple child states.

    - Children get a COPY of parent's evidence (not a partition)
    - Children start with EMPTY penalties (fresh start)
    - Each child gets its own relevance_override from relevance_map
    - ν_raw is recomputed using the child's relevance function
    """
    now = (clock or MockClock(0.0)).now()
    children: list[State] = []
    records: list[RefinementRecord] = []

    for child_id in child_ids:
        child_rfn = relevance_map.get(child_id, policy.relevance_fn)

        # Per-child overrides
        child_def_sem = (per_child_def_sem or {}).get(child_id, def_sem_override)
        child_def_ep = (per_child_def_ep or {}).get(child_id, def_ep_override)
        child_def_proc = (per_child_def_proc or {}).get(child_id, def_proc_override)

        # Recompute definedness with child's relevance function
        def_value = definedness(
            child_id,
            state.evidence,  # copy of parent's evidence
            state.constraints,
            policy.w_sem,
            policy.w_ep,
            policy.w_proc,
            sem_provider,
            child_def_sem,
            child_def_ep,
            child_def_proc,
        )
        new_nu_raw = nu_raw_from_definedness(def_value)

        child_state = State(
            target_id=child_id,
            context_id=state.context_id,
            nu_raw=new_nu_raw,
            nu_penalties={},  # fresh start
            evidence=state.evidence,  # copy, not partition
            constraints=state.constraints,
            metadata=Metadata(
                creation=now,
                last_modified=now,
                history=("split",),
                tags={
                    "parent": state.target_id,
                    "relevance_override": True,
                },
            ),
        )

        record = make_refinement_record(state, child_state, "split", now, {
            "child_id": child_id,
            "parent_id": state.target_id,
        })
        children.append(child_state)
        records.append(record)

    return children, records


# ---------- Conflict (§8.6) ----------

def apply_conflict(
    state: State,
    policy: Policy = PI_DEFAULT,
    clock: Optional[Clock] = None,
    relevance_fn: Optional[RelevanceFn] = None,
) -> tuple[State, RefinementRecord, AggregateResult]:
    """Detect and apply conflict penalty.

    - Has a cooldown period — won't apply new penalty if within conflict_cooldown
    - Starts penalty decay timer when conflict drops below θ_conflict_clear
    - Resets decay timer when conflict re-triggers
    """
    now = (clock or MockClock(0.0)).now()
    rfn = relevance_fn or policy.relevance_fn

    agg = aggregate(state.evidence, state.target_id, state.context_id, rfn, now, policy.decay_rate)

    new_penalties = dict(state.nu_penalties)
    new_metadata = state.metadata

    if agg.conflict > policy.theta_conflict:
        # Check cooldown
        last_applied = state.metadata.conflict_last_applied
        if last_applied is not None and (now - last_applied) < policy.conflict_cooldown:
            # Within cooldown, don't apply new penalty
            pass
        else:
            # Apply conflict penalty
            penalty = min(policy.max_conflict_penalty, agg.conflict * policy.max_conflict_penalty)
            new_penalties[PenaltySource.CONFLICT] = penalty
            new_metadata = new_metadata.with_update(
                conflict_last_applied=now,
                penalty_clear_start=None,  # reset decay timer
                last_modified=now,
            )
    elif agg.conflict < policy.theta_conflict_clear:
        # Conflict resolved — start penalty decay timer if not already started
        if (PenaltySource.CONFLICT in new_penalties
                and new_metadata.penalty_clear_start is None):
            new_metadata = new_metadata.with_update(
                penalty_clear_start=now,
                last_modified=now,
            )

    new_state = state.replace(
        nu_penalties=new_penalties,
        metadata=new_metadata.with_update(
            history=state.metadata.history + ("conflict",),
        ),
    )

    record = make_refinement_record(state, new_state, "conflict", now, {
        "conflict": agg.conflict,
        "pos_mass": agg.pos_mass,
        "neg_mass": agg.neg_mass,
    })
    return new_state, record, agg


# ---------- Recontextualize (§8.5) ----------

def recontextualize(
    state: State,
    new_context_id: ContextID,
    policy: Policy = PI_DEFAULT,
    clock: Optional[Clock] = None,
    relevance_fn: Optional[RelevanceFn] = None,
    sem_provider: Optional[SemanticDefinednessProvider] = None,
    def_sem_override: Optional[DefSemFn] = None,
    def_ep_override: Optional[DefEpFn] = None,
    def_proc_override: Optional[DefProcFn] = None,
) -> tuple[State, RefinementRecord]:
    """Move a state to a new context, recomputing relevance."""
    now = (clock or MockClock(0.0)).now()
    rfn = relevance_fn or policy.relevance_fn

    # Recompute definedness in new context
    def_value = definedness(
        state.target_id,
        state.evidence,
        state.constraints,
        policy.w_sem,
        policy.w_ep,
        policy.w_proc,
        sem_provider,
        def_sem_override,
        def_ep_override,
        def_proc_override,
    )
    new_nu_raw = nu_raw_from_definedness(def_value)

    # Scope expansion penalty if context changes significantly
    penalties = dict(state.nu_penalties)

    new_state = state.replace(
        context_id=new_context_id,
        nu_raw=new_nu_raw,
        nu_penalties=penalties,
        metadata=state.metadata.with_update(
            last_modified=now,
            history=state.metadata.history + ("recontextualize",),
            crossings=state.metadata.crossings + (f"{state.context_id}->{new_context_id}",),
        ),
    )

    record = make_refinement_record(state, new_state, "recontextualize", now)
    return new_state, record


# ---------- Decay (§8.9) ----------

def decay(
    state: State,
    policy: Policy = PI_DEFAULT,
    clock: Optional[Clock] = None,
    relevance_fn: Optional[RelevanceFn] = None,
    sem_provider: Optional[SemanticDefinednessProvider] = None,
    def_sem_override: Optional[DefSemFn] = None,
    def_ep_override: Optional[DefEpFn] = None,
    def_proc_override: Optional[DefProcFn] = None,
) -> tuple[State, RefinementRecord]:
    """Apply time-based decay to a state.

    Evidence trust naturally decays over time, which affects aggregate
    and thus definedness/ν_raw.
    """
    now = (clock or MockClock(0.0)).now()

    # Recompute definedness (aggregate applies decay internally)
    def_value = definedness(
        state.target_id,
        state.evidence,
        state.constraints,
        policy.w_sem,
        policy.w_ep,
        policy.w_proc,
        sem_provider,
        def_sem_override,
        def_ep_override,
        def_proc_override,
    )
    new_nu_raw = nu_raw_from_definedness(def_value)

    new_state = state.replace(
        nu_raw=new_nu_raw,
        metadata=state.metadata.with_update(
            last_modified=now,
            history=state.metadata.history + ("decay",),
        ),
    )

    record = make_refinement_record(state, new_state, "decay", now)
    return new_state, record


# ---------- PenaltyDecay (§8.10) ----------

def penalty_decay(
    state: State,
    policy: Policy = PI_DEFAULT,
    clock: Optional[Clock] = None,
) -> tuple[State, RefinementRecord]:
    """Decay penalties over time.

    - Only runs if penalty_decay_enabled
    - Only decays penalties whose clear timer has expired
    - Removes penalties below cleanup threshold
    """
    now = (clock or MockClock(0.0)).now()

    if not policy.penalty_decay_enabled:
        record = make_refinement_record(state, state, "penalty_decay", now)
        return state, record

    new_penalties = dict(state.nu_penalties)
    new_metadata = state.metadata

    # Check if penalty clear window has elapsed for conflict penalty
    if (PenaltySource.CONFLICT in new_penalties
            and new_metadata.penalty_clear_start is not None):
        elapsed = now - new_metadata.penalty_clear_start
        if elapsed >= policy.penalty_clear_window:
            # Apply decay factor
            new_penalties[PenaltySource.CONFLICT] *= policy.penalty_decay_factor
            if new_penalties[PenaltySource.CONFLICT] < policy.penalty_cleanup_threshold:
                del new_penalties[PenaltySource.CONFLICT]
                new_metadata = new_metadata.with_update(penalty_clear_start=None)

    # Decay other penalty types (non-conflict) if they have been present
    for source in list(new_penalties.keys()):
        if source == PenaltySource.CONFLICT:
            continue
        new_penalties[source] *= policy.penalty_decay_factor
        if new_penalties[source] < policy.penalty_cleanup_threshold:
            del new_penalties[source]

    new_state = state.replace(
        nu_penalties=new_penalties,
        metadata=new_metadata.with_update(
            last_modified=now,
            history=state.metadata.history + ("penalty_decay",),
        ),
    )

    record = make_refinement_record(state, new_state, "penalty_decay", now)
    return new_state, record


# ---------- QueryNext (§8.7) ----------

def query_next(
    states: list[State],
    policy: Policy = PI_DEFAULT,
) -> Optional[State]:
    """Determine which target to refine next.

    Returns the state with highest ν_raw (most vague) that is above θ_null.
    If none are above θ_null, returns the one with highest ν_raw overall.
    Returns None if states is empty.
    """
    if not states:
        return None

    # Prefer targets above θ_null (most in need of refinement)
    above_null = [s for s in states if s.nu_raw > policy.theta_null]
    if above_null:
        return max(above_null, key=lambda s: s.nu_raw)

    # Otherwise, highest ν_raw
    return max(states, key=lambda s: s.nu_raw)
