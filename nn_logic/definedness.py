"""Definedness computation: Def, Def_sem, Def_ep, Def_proc.

Def_sem uses a pluggable SemanticDefinednessProvider protocol.
Def_ep and Def_proc can also be overridden for testing.
"""

from __future__ import annotations

from typing import Callable, Optional

from nn_logic.types import (
    EvidenceKind,
    EvidenceSet,
    SemanticDefinednessProvider,
    TargetID,
)
from nn_logic.evidence import partition_by_kind
from nn_logic.helpers import clamp


class DefaultSemanticProvider:
    """Simple heuristic implementation of SemanticDefinednessProvider.

    Domain bindings should override this with richer semantics.
    """

    def ontology_coverage(
        self, target: TargetID, evidence: EvidenceSet, constraints: tuple[str, ...]
    ) -> float:
        definitional = [e for e in evidence if e.kind == EvidenceKind.DEFINITIONAL]
        base = min(1.0, len(definitional) * 0.15)
        constraint_bonus = min(0.5, len(constraints) * 0.1)
        return clamp(base + constraint_bonus)

    def ambiguity_score(
        self, target: TargetID, evidence: EvidenceSet, constraints: tuple[str, ...]
    ) -> float:
        # More constraints reduce ambiguity → higher score is less ambiguous
        if not constraints:
            return 0.0
        return clamp(len(constraints) * 0.12)

    def constraint_coverage(
        self, target: TargetID, evidence: EvidenceSet, constraints: tuple[str, ...]
    ) -> float:
        if not constraints:
            return 0.0
        return clamp(len(constraints) * 0.1)

    def boundary_precision(
        self, target: TargetID, evidence: EvidenceSet, constraints: tuple[str, ...]
    ) -> float:
        definitional = [e for e in evidence if e.kind == EvidenceKind.DEFINITIONAL]
        if not definitional and not constraints:
            return 0.0
        return clamp((len(definitional) * 0.1) + (len(constraints) * 0.08))


_DEFAULT_PROVIDER = DefaultSemanticProvider()


# Type aliases for override functions
DefSemFn = Callable[[TargetID, EvidenceSet, tuple[str, ...]], float]
DefEpFn = Callable[[TargetID, EvidenceSet], float]
DefProcFn = Callable[[TargetID, EvidenceSet], float]


def def_sem(
    target: TargetID,
    evidence: EvidenceSet,
    constraints: tuple[str, ...] = (),
    provider: Optional[SemanticDefinednessProvider] = None,
    override: Optional[DefSemFn] = None,
) -> float:
    """Semantic definedness.

    Def_sem = mean(ontology_coverage, 1 - ambiguity_score,
                   constraint_coverage, boundary_precision)
    """
    if override is not None:
        return override(target, evidence, constraints)

    p = provider or _DEFAULT_PROVIDER
    oc = p.ontology_coverage(target, evidence, constraints)
    amb = p.ambiguity_score(target, evidence, constraints)
    cc = p.constraint_coverage(target, evidence, constraints)
    bp = p.boundary_precision(target, evidence, constraints)

    return clamp((oc + (1.0 - amb) + cc + bp) / 4.0)


def def_ep(
    target: TargetID,
    evidence: EvidenceSet,
    override: Optional[DefEpFn] = None,
) -> float:
    """Epistemic definedness — how much evidence do we have?"""
    if override is not None:
        return override(target, evidence)

    epistemic = [e for e in evidence if e.kind == EvidenceKind.EPISTEMIC]
    if not epistemic:
        return 0.0
    total_weight = sum(abs(e.valence) * e.trust for e in epistemic)
    return clamp(total_weight / 2.0)


def def_proc(
    target: TargetID,
    evidence: EvidenceSet,
    override: Optional[DefProcFn] = None,
) -> float:
    """Procedural definedness — are procedures/methods specified?"""
    if override is not None:
        return override(target, evidence)

    procedural = [e for e in evidence if e.kind == EvidenceKind.PROCEDURAL]
    if not procedural:
        return 0.0
    total_weight = sum(abs(e.valence) * e.trust for e in procedural)
    return clamp(total_weight / 2.0)


def definedness(
    target: TargetID,
    evidence: EvidenceSet,
    constraints: tuple[str, ...] = (),
    w_sem: float = 0.4,
    w_ep: float = 0.35,
    w_proc: float = 0.25,
    sem_provider: Optional[SemanticDefinednessProvider] = None,
    def_sem_override: Optional[DefSemFn] = None,
    def_ep_override: Optional[DefEpFn] = None,
    def_proc_override: Optional[DefProcFn] = None,
) -> float:
    """Compute overall definedness Def = w_sem * Def_sem + w_ep * Def_ep + w_proc * Def_proc.

    Returns a value in [0, 1] where 1 = fully defined.
    """
    ds = def_sem(target, evidence, constraints, sem_provider, def_sem_override)
    de = def_ep(target, evidence, override=def_ep_override)
    dp = def_proc(target, evidence, override=def_proc_override)
    return clamp(w_sem * ds + w_ep * de + w_proc * dp)


def nu_raw_from_definedness(def_value: float) -> float:
    """ν_raw = 1 - Def. Higher definedness → lower vagueness."""
    return clamp(1.0 - def_value)
