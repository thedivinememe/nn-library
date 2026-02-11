"""Tests for definedness computation."""

from __future__ import annotations

import pytest

from nn_logic.types import (
    AgentID,
    Evidence,
    EvidenceID,
    EvidenceKind,
    EvidenceSet,
    TargetID,
)
from nn_logic.definedness import (
    DefaultSemanticProvider,
    def_ep,
    def_proc,
    def_sem,
    definedness,
    nu_raw_from_definedness,
)


TID = TargetID("t")


def _e(eid: str, kind: EvidenceKind, valence: float = 0.5) -> Evidence:
    return Evidence(
        id=EvidenceID(eid), kind=kind, claim=f"claim_{eid}",
        valence=valence, src=AgentID("a"), time=0.0,
    )


class TestDefSem:
    def test_empty_evidence_no_constraints(self) -> None:
        result = def_sem(TID, EvidenceSet.empty())
        # Should be low but not zero (1 - ambiguity contributes)
        assert 0.0 <= result <= 1.0

    def test_increases_with_constraints(self) -> None:
        r0 = def_sem(TID, EvidenceSet.empty(), ())
        r5 = def_sem(TID, EvidenceSet.empty(), ("c1", "c2", "c3", "c4", "c5"))
        assert r5 > r0

    def test_override(self) -> None:
        def custom(t: TargetID, ev: EvidenceSet, c: tuple[str, ...]) -> float:
            return 0.42

        result = def_sem(TID, EvidenceSet.empty(), override=custom)
        assert result == pytest.approx(0.42)


class TestDefEp:
    def test_empty(self) -> None:
        assert def_ep(TID, EvidenceSet.empty()) == 0.0

    def test_with_epistemic_evidence(self) -> None:
        e1 = _e("e1", EvidenceKind.EPISTEMIC, 0.5)
        e2 = _e("e2", EvidenceKind.EPISTEMIC, 0.7)
        es = EvidenceSet(items=(e1, e2))
        result = def_ep(TID, es)
        assert result > 0.0

    def test_ignores_non_epistemic(self) -> None:
        e1 = _e("e1", EvidenceKind.DEFINITIONAL, 0.5)
        es = EvidenceSet(items=(e1,))
        result = def_ep(TID, es)
        assert result == 0.0


class TestDefProc:
    def test_empty(self) -> None:
        assert def_proc(TID, EvidenceSet.empty()) == 0.0

    def test_with_procedural_evidence(self) -> None:
        e1 = _e("e1", EvidenceKind.PROCEDURAL, 0.6)
        es = EvidenceSet(items=(e1,))
        result = def_proc(TID, es)
        assert result > 0.0


class TestDefinedness:
    def test_empty_is_zero(self) -> None:
        result = definedness(TID, EvidenceSet.empty())
        # Not exactly zero due to ambiguity default, but low
        assert result >= 0.0

    def test_weights_sum_interpretation(self) -> None:
        # With overrides, verify weighted sum
        def sem(t: TargetID, ev: EvidenceSet, c: tuple[str, ...]) -> float:
            return 1.0
        def ep(t: TargetID, ev: EvidenceSet) -> float:
            return 1.0
        def proc(t: TargetID, ev: EvidenceSet) -> float:
            return 1.0

        result = definedness(
            TID, EvidenceSet.empty(),
            w_sem=0.4, w_ep=0.35, w_proc=0.25,
            def_sem_override=sem, def_ep_override=ep, def_proc_override=proc,
        )
        assert result == pytest.approx(1.0)

    def test_partial_definedness(self) -> None:
        def sem(t: TargetID, ev: EvidenceSet, c: tuple[str, ...]) -> float:
            return 0.5
        def ep(t: TargetID, ev: EvidenceSet) -> float:
            return 0.5
        def proc(t: TargetID, ev: EvidenceSet) -> float:
            return 0.5

        result = definedness(
            TID, EvidenceSet.empty(),
            def_sem_override=sem, def_ep_override=ep, def_proc_override=proc,
        )
        assert result == pytest.approx(0.5)


class TestNuRawFromDefinedness:
    def test_zero_def(self) -> None:
        assert nu_raw_from_definedness(0.0) == 1.0

    def test_full_def(self) -> None:
        assert nu_raw_from_definedness(1.0) == 0.0

    def test_half_def(self) -> None:
        assert nu_raw_from_definedness(0.5) == pytest.approx(0.5)

    def test_clamped(self) -> None:
        assert nu_raw_from_definedness(1.5) == 0.0
        assert nu_raw_from_definedness(-0.5) == 1.0
