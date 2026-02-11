"""Core types, enums, and dataclasses for N/N-N Logic."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    NewType,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)


# ---------- ID types ----------

TargetID = NewType("TargetID", str)
ContextID = NewType("ContextID", str)
EvidenceID = NewType("EvidenceID", str)
AgentID = NewType("AgentID", str)


# ---------- Enums ----------

class EvidenceKind(enum.Enum):
    EPISTEMIC = "epistemic"
    DEFINITIONAL = "definitional"
    PROCEDURAL = "procedural"


class Role(enum.Enum):
    I = "I"
    NOT_I = "NotI"
    BOTH = "Both"
    UNKNOWN = "Unknown"


class PenaltySource(enum.Enum):
    CONFLICT = "conflict"
    SCOPE_EXPANSION = "scope_expansion"
    MERGE_RUPTURE = "merge_rupture"
    CATEGORY_ERROR = "category_error"
    MANUAL = "manual"


class DedupMode(enum.Enum):
    STRICT = "strict"
    CORROBORATION = "corroboration"


class PenaltyMode(enum.Enum):
    MAX = "max"
    SUM = "sum"


class NullStatus(enum.Enum):
    NULL = "NULL"
    NOT_NULL = "NOT_NULL"
    INDETERMINATE = "INDETERMINATE"


class QueryResult(enum.Enum):
    LICENSED = "LICENSED"
    NOT_LICENSED = "NOT_LICENSED"


class Reason(enum.Enum):
    STRUCTURALLY_VAGUE = "structurally_vague"
    CLEAR_BUT_PENALIZED = "clear_but_penalized"
    LICENSED = "licensed"


# ---------- Clock protocol ----------

@runtime_checkable
class Clock(Protocol):
    def now(self) -> float: ...


class SystemClock:
    def now(self) -> float:
        return time.time()


class MockClock:
    def __init__(self, start: float = 0.0) -> None:
        self._time = start

    def now(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        self._time += seconds

    def set(self, t: float) -> None:
        self._time = t


# ---------- Core dataclasses ----------

@dataclass(frozen=True)
class Evidence:
    id: EvidenceID
    kind: EvidenceKind
    claim: str
    valence: float  # positive = supporting, negative = opposing
    src: AgentID
    time: float
    trust: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def weight(self) -> float:
        return abs(self.valence) * self.trust


@dataclass(frozen=True)
class EvidenceSet:
    items: tuple[Evidence, ...] = ()

    @staticmethod
    def empty() -> EvidenceSet:
        return EvidenceSet()

    def add(self, e: Evidence) -> EvidenceSet:
        return EvidenceSet(items=self.items + (e,))

    def union(self, other: EvidenceSet) -> EvidenceSet:
        seen: set[EvidenceID] = set()
        merged: list[Evidence] = []
        for item in self.items + other.items:
            if item.id not in seen:
                seen.add(item.id)
                merged.append(item)
        return EvidenceSet(items=tuple(merged))

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):  # type: ignore[override]
        return iter(self.items)


@dataclass(frozen=True)
class Metadata:
    creation: float = 0.0
    last_modified: float = 0.0
    history: tuple[str, ...] = ()
    crossings: tuple[str, ...] = ()
    conflict_last_applied: Optional[float] = None
    penalty_clear_start: Optional[float] = None
    tags: dict[str, Any] = field(default_factory=dict)

    def with_update(self, **kwargs: Any) -> Metadata:
        d = {
            "creation": self.creation,
            "last_modified": self.last_modified,
            "history": self.history,
            "crossings": self.crossings,
            "conflict_last_applied": self.conflict_last_applied,
            "penalty_clear_start": self.penalty_clear_start,
            "tags": dict(self.tags),
        }
        d.update(kwargs)
        return Metadata(**d)


@dataclass(frozen=True)
class State:
    """Per-target, per-context state."""
    target_id: TargetID
    context_id: ContextID
    truth_status: Optional[Any] = None  # pluggable truth semantics
    nu_raw: float = 1.0
    nu_penalties: dict[PenaltySource, float] = field(default_factory=dict)
    evidence: EvidenceSet = field(default_factory=EvidenceSet.empty)
    metadata: Metadata = field(default_factory=Metadata)
    constraints: tuple[str, ...] = ()  # definitional constraints for NegDefine

    @property
    def nu(self) -> float:
        from nn_logic.helpers import compute_nu
        return compute_nu(self.nu_raw, self.nu_penalties, PenaltyMode.MAX)

    def nu_with_mode(self, mode: PenaltyMode) -> float:
        from nn_logic.helpers import compute_nu
        return compute_nu(self.nu_raw, self.nu_penalties, mode)

    def replace(self, **kwargs: Any) -> State:
        d: dict[str, Any] = {
            "target_id": self.target_id,
            "context_id": self.context_id,
            "truth_status": self.truth_status,
            "nu_raw": self.nu_raw,
            "nu_penalties": dict(self.nu_penalties),
            "evidence": self.evidence,
            "metadata": self.metadata,
            "constraints": self.constraints,
        }
        d.update(kwargs)
        return State(**d)


@dataclass(frozen=True)
class Context:
    id: ContextID
    i_side: frozenset[AgentID] = field(default_factory=frozenset)
    not_i_side: frozenset[AgentID] = field(default_factory=frozenset)
    time_start: float = 0.0
    time_end: float = float("inf")
    scope: frozenset[TargetID] = field(default_factory=frozenset)
    roles: dict[AgentID, Role] = field(default_factory=dict)
    # policy is stored separately, referenced by context_id


# ---------- Semantic definedness provider protocol ----------

@runtime_checkable
class SemanticDefinednessProvider(Protocol):
    def ontology_coverage(self, target: TargetID, evidence: EvidenceSet, constraints: tuple[str, ...]) -> float: ...
    def ambiguity_score(self, target: TargetID, evidence: EvidenceSet, constraints: tuple[str, ...]) -> float: ...
    def constraint_coverage(self, target: TargetID, evidence: EvidenceSet, constraints: tuple[str, ...]) -> float: ...
    def boundary_precision(self, target: TargetID, evidence: EvidenceSet, constraints: tuple[str, ...]) -> float: ...


# ---------- Refinement record (for trace mode) ----------

@dataclass(frozen=True)
class RefinementRecord:
    target_id: TargetID
    context_id: ContextID
    operator: str
    nu_before: float
    nu_after: float
    nu_raw_before: float
    nu_raw_after: float
    penalties_before: dict[PenaltySource, float]
    penalties_after: dict[PenaltySource, float]
    timestamp: float
    details: dict[str, Any] = field(default_factory=dict)


# ---------- Aggregate result ----------

@dataclass(frozen=True)
class AggregateResult:
    pos_mass: float
    neg_mass: float
    conflict: float
    def_ep: float  # epistemic definedness from evidence


# ---------- Relevance function type ----------

RelevanceFn = Callable[["Evidence", TargetID, ContextID], float]


def default_relevance_fn(e: Evidence, target: TargetID, context: ContextID) -> float:
    return 1.0
