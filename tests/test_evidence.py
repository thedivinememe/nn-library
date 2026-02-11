"""Tests for evidence module: dedup, identity, partitioning."""

from __future__ import annotations

import pytest

from nn_logic.types import (
    AgentID,
    DedupMode,
    Evidence,
    EvidenceID,
    EvidenceKind,
    EvidenceSet,
)
from nn_logic.evidence import (
    add_evidence,
    compute_evidence_id,
    make_evidence,
    partition_by_kind,
    should_add,
    time_bucket,
)


class TestTimeBucket:
    def test_same_bucket(self) -> None:
        assert time_bucket(10.0, 60.0) == time_bucket(50.0, 60.0)

    def test_different_bucket(self) -> None:
        assert time_bucket(10.0, 60.0) != time_bucket(70.0, 60.0)


class TestEvidenceIdentity:
    def test_same_inputs_same_id(self) -> None:
        id1 = compute_evidence_id(EvidenceKind.EPISTEMIC, "claim", AgentID("a"), 10.0)
        id2 = compute_evidence_id(EvidenceKind.EPISTEMIC, "claim", AgentID("a"), 10.0)
        assert id1 == id2

    def test_different_claim_different_id(self) -> None:
        id1 = compute_evidence_id(EvidenceKind.EPISTEMIC, "claim1", AgentID("a"), 10.0)
        id2 = compute_evidence_id(EvidenceKind.EPISTEMIC, "claim2", AgentID("a"), 10.0)
        assert id1 != id2

    def test_different_src_different_id(self) -> None:
        id1 = compute_evidence_id(EvidenceKind.EPISTEMIC, "claim", AgentID("a"), 10.0)
        id2 = compute_evidence_id(EvidenceKind.EPISTEMIC, "claim", AgentID("b"), 10.0)
        assert id1 != id2


class TestDedup:
    def test_strict_skips_duplicate(self) -> None:
        e1 = Evidence(
            id=EvidenceID("e1"), kind=EvidenceKind.EPISTEMIC, claim="x",
            valence=0.5, src=AgentID("a"), time=0.0,
        )
        e2 = Evidence(
            id=EvidenceID("e1"), kind=EvidenceKind.EPISTEMIC, claim="x",
            valence=0.5, src=AgentID("a"), time=0.0,
        )
        es = EvidenceSet(items=(e1,))
        assert not should_add(e2, es, DedupMode.STRICT)

    def test_strict_allows_different_id(self) -> None:
        e1 = Evidence(
            id=EvidenceID("e1"), kind=EvidenceKind.EPISTEMIC, claim="x",
            valence=0.5, src=AgentID("a"), time=0.0,
        )
        e2 = Evidence(
            id=EvidenceID("e2"), kind=EvidenceKind.EPISTEMIC, claim="y",
            valence=0.5, src=AgentID("a"), time=0.0,
        )
        es = EvidenceSet(items=(e1,))
        assert should_add(e2, es, DedupMode.STRICT)

    def test_corroboration_allows_same_claim_different_src(self) -> None:
        e1 = Evidence(
            id=EvidenceID("e1"), kind=EvidenceKind.EPISTEMIC, claim="x",
            valence=0.5, src=AgentID("a"), time=0.0,
        )
        e2 = Evidence(
            id=EvidenceID("e1"), kind=EvidenceKind.EPISTEMIC, claim="x",
            valence=0.5, src=AgentID("b"), time=0.0,
        )
        es = EvidenceSet(items=(e1,))
        assert should_add(e2, es, DedupMode.CORROBORATION)

    def test_corroboration_blocks_same_src(self) -> None:
        e1 = Evidence(
            id=EvidenceID("e1"), kind=EvidenceKind.EPISTEMIC, claim="x",
            valence=0.5, src=AgentID("a"), time=0.0,
        )
        e2 = Evidence(
            id=EvidenceID("e1"), kind=EvidenceKind.EPISTEMIC, claim="x",
            valence=0.5, src=AgentID("a"), time=0.0,
        )
        es = EvidenceSet(items=(e1,))
        assert not should_add(e2, es, DedupMode.CORROBORATION)


class TestPartitionByKind:
    def test_empty(self) -> None:
        result = partition_by_kind(EvidenceSet.empty())
        for kind in EvidenceKind:
            assert result[kind] == []

    def test_partitions_correctly(self) -> None:
        e1 = Evidence(id=EvidenceID("e1"), kind=EvidenceKind.EPISTEMIC, claim="a",
                       valence=0.5, src=AgentID("a"), time=0.0)
        e2 = Evidence(id=EvidenceID("e2"), kind=EvidenceKind.DEFINITIONAL, claim="b",
                       valence=0.5, src=AgentID("a"), time=0.0)
        e3 = Evidence(id=EvidenceID("e3"), kind=EvidenceKind.PROCEDURAL, claim="c",
                       valence=0.5, src=AgentID("a"), time=0.0)
        es = EvidenceSet(items=(e1, e2, e3))
        result = partition_by_kind(es)
        assert len(result[EvidenceKind.EPISTEMIC]) == 1
        assert len(result[EvidenceKind.DEFINITIONAL]) == 1
        assert len(result[EvidenceKind.PROCEDURAL]) == 1

    def test_partitions_are_disjoint(self) -> None:
        e1 = Evidence(id=EvidenceID("e1"), kind=EvidenceKind.EPISTEMIC, claim="a",
                       valence=0.5, src=AgentID("a"), time=0.0)
        e2 = Evidence(id=EvidenceID("e2"), kind=EvidenceKind.DEFINITIONAL, claim="b",
                       valence=0.5, src=AgentID("a"), time=0.0)
        es = EvidenceSet(items=(e1, e2))
        result = partition_by_kind(es)
        all_ids: list[EvidenceID] = []
        for kind in EvidenceKind:
            ids = [e.id for e in result[kind]]
            for eid in ids:
                assert eid not in all_ids
                all_ids.append(eid)
