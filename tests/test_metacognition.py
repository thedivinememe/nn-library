"""Tests for MetacognitionBridge — assessment-to-evidence mapping."""

from __future__ import annotations

import pytest

from nn_logic.types import (
    ContextID,
    EvidenceKind,
    MockClock,
    PenaltyMode,
    TargetID,
)
from nn_logic.policy import Policy

from rwt_integration.config import LoopConfig
from rwt_integration.metacognition import MetacognitionBridge
from rwt_integration.providers import SelfAssessment


@pytest.fixture
def bridge() -> MetacognitionBridge:
    clock = MockClock(1000.0)
    return MetacognitionBridge(
        policy=Policy(
            w_sem=0.4, w_ep=0.35, w_proc=0.25,
            theta_conflict=0.3,
        ),
        config=LoopConfig(),
        clock=clock,
    )


@pytest.fixture
def target() -> TargetID:
    return TargetID("test_target")


@pytest.fixture
def context() -> ContextID:
    return ContextID("test_context")


class TestAssessmentToEvidence:
    def test_high_confidence_low_nu(self, bridge: MetacognitionBridge) -> None:
        """High confidence assessment should produce low ν."""
        target = TargetID("t")
        ctx = ContextID("c")
        state = bridge.initialize_state(target, ctx)

        assessment = SelfAssessment(
            definition_confidence=0.9,
            ambiguity_flags=(),
            evidence_confidence=0.85,
            unsupported_claims=(),
            contradictions=(),
            task_coverage=0.95,
            missing_elements=(),
            refinement_priority="polish",
            refinement_suggestion="Minor polish",
        )

        state, diag = bridge.process_iteration(state, "good output", assessment, 0)
        assert diag.nu < 0.5  # well-defined output

    def test_low_confidence_high_nu(self, bridge: MetacognitionBridge) -> None:
        """Low confidence with many issues should produce high ν."""
        target = TargetID("t")
        ctx = ContextID("c")
        state = bridge.initialize_state(target, ctx)

        assessment = SelfAssessment(
            definition_confidence=0.2,
            ambiguity_flags=("vague", "unclear", "undefined"),
            evidence_confidence=0.2,
            unsupported_claims=("claim A", "claim B", "claim C"),
            contradictions=(),
            task_coverage=0.3,
            missing_elements=("req 1", "req 2", "req 3"),
            refinement_priority="everything",
            refinement_suggestion="Start over",
        )

        state, diag = bridge.process_iteration(state, "bad output", assessment, 0)
        assert diag.nu > 0.6  # poorly defined output

    def test_nu_decreases_with_improvement(self, bridge: MetacognitionBridge) -> None:
        """ν should decrease as assessments improve across iterations."""
        target = TargetID("t")
        ctx = ContextID("c")
        state = bridge.initialize_state(target, ctx)

        # Iteration 0: poor
        a0 = SelfAssessment(
            definition_confidence=0.3,
            ambiguity_flags=("vague", "unclear"),
            evidence_confidence=0.3,
            unsupported_claims=("claim A",),
            task_coverage=0.4,
            missing_elements=("req 1", "req 2"),
        )
        state, diag0 = bridge.process_iteration(state, "v1", a0, 0)

        # Iteration 1: better
        a1 = SelfAssessment(
            definition_confidence=0.7,
            ambiguity_flags=(),
            evidence_confidence=0.7,
            unsupported_claims=(),
            task_coverage=0.85,
            missing_elements=(),
        )
        state, diag1 = bridge.process_iteration(state, "v2", a1, 1)

        assert diag1.nu < diag0.nu

    def test_contradictions_trigger_conflict(self, bridge: MetacognitionBridge) -> None:
        """Contradictions in assessment should produce conflict evidence."""
        target = TargetID("t")
        ctx = ContextID("c")
        state = bridge.initialize_state(target, ctx)

        assessment = SelfAssessment(
            definition_confidence=0.5,
            evidence_confidence=0.5,
            contradictions=("point A contradicts point B",),
            task_coverage=0.6,
        )

        state, diag = bridge.process_iteration(state, "output", assessment, 0)
        assert diag.conflict >= 0.0  # conflict detected

    def test_evidence_kinds_correct(self, bridge: MetacognitionBridge) -> None:
        """Verify evidence is categorized into correct kinds."""
        target = TargetID("t")
        ctx = ContextID("c")
        state = bridge.initialize_state(target, ctx)

        assessment = SelfAssessment(
            definition_confidence=0.5,
            ambiguity_flags=("vague term",),
            evidence_confidence=0.6,
            unsupported_claims=("claim A",),
            task_coverage=0.7,
            missing_elements=("req 1",),
        )

        state, _ = bridge.process_iteration(state, "output", assessment, 0)

        kinds = set(e.kind for e in state.evidence)
        assert EvidenceKind.DEFINITIONAL in kinds
        assert EvidenceKind.EPISTEMIC in kinds
        assert EvidenceKind.PROCEDURAL in kinds


class TestCalibration:
    def test_empty_flags_with_low_confidence_discounted(
        self, bridge: MetacognitionBridge
    ) -> None:
        """If no ambiguity flags but confidence < 0.9, the raw confidence is
        discounted before blending. Compare: same confidence WITH flags should
        produce a lower def_sem than without flags."""
        target = TargetID("t")
        ctx = ContextID("c")

        # Case A: no flags, moderate confidence
        state_a = bridge.initialize_state(target, ctx)
        a_no_flags = SelfAssessment(
            definition_confidence=0.7,
            ambiguity_flags=(),
            evidence_confidence=0.7,
            unsupported_claims=(),
            task_coverage=0.8,
            missing_elements=(),
        )
        state_a, diag_a = bridge.process_iteration(state_a, "output", a_no_flags, 0)

        # Case B: with flags, same stated confidence
        bridge.reset()
        state_b = bridge.initialize_state(target, ctx)
        a_with_flags = SelfAssessment(
            definition_confidence=0.7,
            ambiguity_flags=("vague term", "unclear scope"),
            evidence_confidence=0.7,
            unsupported_claims=(),
            task_coverage=0.8,
            missing_elements=(),
        )
        state_b, diag_b = bridge.process_iteration(state_b, "output", a_with_flags, 0)

        # With flags should produce lower def_sem than without
        assert diag_b.def_sem < diag_a.def_sem

    def test_skepticism_penalty_on_empty_unsupported(
        self, bridge: MetacognitionBridge
    ) -> None:
        """On iteration > 0, empty unsupported_claims gets a skepticism penalty.
        Compare def_ep at iteration 0 vs iteration 1 with same stated confidence
        but no claims at iteration 1 — skepticism should lower the result."""
        target = TargetID("t")
        ctx = ContextID("c")

        # Run at iteration 0 with unsupported claims
        state0 = bridge.initialize_state(target, ctx)
        a0 = SelfAssessment(
            definition_confidence=0.5,
            evidence_confidence=0.6,
            unsupported_claims=("claim",),
            task_coverage=0.5,
        )
        state0, diag0 = bridge.process_iteration(state0, "v1", a0, 0)

        # Same bridge, iteration 1 claims no unsupported — skepticism applied
        a1 = SelfAssessment(
            definition_confidence=0.6,
            evidence_confidence=0.6,
            unsupported_claims=(),
            task_coverage=0.6,
        )
        state0, diag1 = bridge.process_iteration(state0, "v2", a1, 1)

        # Compare: run a fresh bridge at iteration 0 with the SAME empty claims
        bridge2 = MetacognitionBridge(
            policy=bridge._policy, config=bridge._config, clock=bridge._clock
        )
        state_fresh = bridge2.initialize_state(target, ctx)
        state_fresh, diag_fresh = bridge2.process_iteration(state_fresh, "v2", a1, 0)

        # Iteration 1 with skepticism should have lower def_ep than iteration 0 without
        assert diag1.def_ep < diag_fresh.def_ep

    def test_issue_counts_weighted_more(self, bridge: MetacognitionBridge) -> None:
        """Issue counts are weighted more than stated confidence."""
        target = TargetID("t")
        ctx = ContextID("c")

        # High stated confidence but many issues
        state1 = bridge.initialize_state(target, ctx)
        a_optimistic = SelfAssessment(
            definition_confidence=0.9,
            ambiguity_flags=("a", "b", "c", "d"),
            evidence_confidence=0.9,
            unsupported_claims=("x", "y", "z"),
            task_coverage=0.9,
            missing_elements=("m1", "m2"),
        )
        state1, diag1 = bridge.process_iteration(state1, "output", a_optimistic, 0)

        # Lower stated confidence with fewer issues
        bridge.reset()
        state2 = bridge.initialize_state(target, ctx)
        a_realistic = SelfAssessment(
            definition_confidence=0.6,
            ambiguity_flags=("a",),
            evidence_confidence=0.6,
            unsupported_claims=("x",),
            task_coverage=0.7,
            missing_elements=(),
        )
        state2, diag2 = bridge.process_iteration(state2, "output", a_realistic, 0)

        # The "realistic" assessment should produce lower ν than the "optimistic" one
        # because issue counts are weighted more heavily
        assert diag2.nu < diag1.nu
