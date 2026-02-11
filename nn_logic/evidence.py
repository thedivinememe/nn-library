"""Evidence, EvidenceSet operations, dedup, and identity computation."""

from __future__ import annotations

import hashlib
import math
from typing import Optional

from nn_logic.types import (
    AgentID,
    DedupMode,
    Evidence,
    EvidenceID,
    EvidenceKind,
    EvidenceSet,
)


def time_bucket(t: float, granularity: float) -> int:
    return int(math.floor(t / granularity))


def compute_evidence_id(
    kind: EvidenceKind,
    claim: str,
    src: AgentID,
    t: float,
    granularity: float = 60.0,
) -> EvidenceID:
    bucket = time_bucket(t, granularity)
    raw = f"{kind.value}:{claim}:{src}:{bucket}"
    return EvidenceID(hashlib.sha256(raw.encode()).hexdigest()[:16])


def should_add(
    new: Evidence,
    existing: EvidenceSet,
    mode: DedupMode,
) -> bool:
    for e in existing:
        if e.id == new.id:
            if mode == DedupMode.STRICT:
                return False
            if mode == DedupMode.CORROBORATION:
                if e.src == new.src:
                    return False
    return True


def add_evidence(
    evidence_set: EvidenceSet,
    new: Evidence,
    mode: DedupMode = DedupMode.STRICT,
) -> EvidenceSet:
    if should_add(new, evidence_set, mode):
        return evidence_set.add(new)
    return evidence_set


def make_evidence(
    kind: EvidenceKind,
    claim: str,
    valence: float,
    src: AgentID,
    t: float,
    trust: float = 1.0,
    granularity: float = 60.0,
    evidence_id: Optional[EvidenceID] = None,
    metadata: Optional[dict] = None,
) -> Evidence:
    eid = evidence_id or compute_evidence_id(kind, claim, src, t, granularity)
    return Evidence(
        id=eid,
        kind=kind,
        claim=claim,
        valence=valence,
        src=src,
        time=t,
        trust=trust,
        metadata=metadata or {},
    )


def partition_by_kind(es: EvidenceSet) -> dict[EvidenceKind, list[Evidence]]:
    result: dict[EvidenceKind, list[Evidence]] = {k: [] for k in EvidenceKind}
    for e in es:
        result[e.kind].append(e)
    return result
