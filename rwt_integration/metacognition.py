"""Metacognition bridge — maps LLM self-assessments to N/N-N state updates."""

from __future__ import annotations

from typing import Optional

from nn_logic.types import (
    AgentID,
    Clock,
    ContextID,
    Evidence,
    EvidenceID,
    EvidenceKind,
    EvidenceSet,
    MockClock,
    PenaltyMode,
    PenaltySource,
    RefinementRecord,
    State,
    TargetID,
)
from nn_logic.aggregate import aggregate, compute_conflict
from nn_logic.definedness import (
    DefEpFn,
    DefProcFn,
    DefSemFn,
    definedness,
    nu_raw_from_definedness,
)
from nn_logic.helpers import clamp, compute_nu, make_refinement_record
from nn_logic.operators import apply_conflict, incorporate
from nn_logic.policy import PI_DEFAULT, Policy
from nn_logic.state import make_initial_state

from rwt_integration.config import LoopConfig
from rwt_integration.providers import SelfAssessment


class MetacognitionBridge:
    """Maps LLM self-assessments into N/N-N state updates.

    Each iteration:
    1. Calibrate the raw assessment (apply skepticism countermeasures)
    2. Convert assessment fields to N/N-N evidence items
    3. Incorporate evidence into state
    4. Run conflict detection
    5. Return updated state with diagnostic info
    """

    def __init__(
        self,
        policy: Policy = PI_DEFAULT,
        config: LoopConfig = LoopConfig(),
        clock: Optional[Clock] = None,
    ) -> None:
        self._policy = policy
        self._config = config
        self._clock = clock or MockClock(0.0)
        self._iteration_assessments: list[SelfAssessment] = []

    def initialize_state(
        self, target_id: TargetID, context_id: ContextID
    ) -> State:
        """Create a fresh N/N-N state for a new task run."""
        return make_initial_state(target_id, context_id, self._clock)

    def process_iteration(
        self,
        state: State,
        output: str,
        assessment: SelfAssessment,
        iteration: int,
    ) -> tuple[State, IterationDiagnostics]:
        """Map a self-assessment into N/N-N state updates.

        Returns the updated state and diagnostics for this iteration.
        """
        self._iteration_assessments.append(assessment)

        # Step 1: Calibrate
        calibrated = self._calibrate(assessment, iteration)

        # Step 2: Convert to evidence
        evidence_items = self._assessment_to_evidence(calibrated, iteration)

        # Step 3: Create Def overrides from calibrated assessment
        def_sem_val = self._compute_def_sem(calibrated)
        def_ep_val = self._compute_def_ep(calibrated)
        def_proc_val = self._compute_def_proc(calibrated)

        def sem_override(t: TargetID, ev: EvidenceSet, c: tuple[str, ...]) -> float:
            return def_sem_val

        def ep_override(t: TargetID, ev: EvidenceSet) -> float:
            return def_ep_val

        def proc_override(t: TargetID, ev: EvidenceSet) -> float:
            return def_proc_val

        # Step 4: Incorporate evidence
        state, inc_record = incorporate(
            state,
            evidence_items,
            policy=self._policy,
            clock=self._clock,
            def_sem_override=sem_override,
            def_ep_override=ep_override,
            def_proc_override=proc_override,
        )

        # Step 5: Run conflict detection
        conflict_score = 0.0
        if calibrated.contradictions:
            state, conflict_record, agg_result = apply_conflict(
                state, policy=self._policy, clock=self._clock
            )
            conflict_score = agg_result.conflict

        # Compute aggregate for diagnostics
        agg = aggregate(
            state.evidence,
            state.target_id,
            state.context_id,
            self._policy.relevance_fn,
            (self._clock).now() if hasattr(self._clock, 'now') else 0.0,
            0.0,
        )

        nu = state.nu_with_mode(self._policy.penalty_mode)
        nu_penalty = nu - state.nu_raw if nu > state.nu_raw else 0.0

        # Determine weakest component
        components = {"def_sem": def_sem_val, "def_ep": def_ep_val, "def_proc": def_proc_val}
        weakest = min(components, key=components.get)  # type: ignore[arg-type]

        diagnostics = IterationDiagnostics(
            nu=nu,
            nu_raw=state.nu_raw,
            nu_penalty=nu_penalty,
            def_sem=def_sem_val,
            def_ep=def_ep_val,
            def_proc=def_proc_val,
            conflict=conflict_score,
            weakest_component=weakest,
            calibrated_assessment=calibrated,
        )

        return state, diagnostics

    def _calibrate(self, raw: SelfAssessment, iteration: int) -> SelfAssessment:
        """Apply calibration countermeasures to raw self-assessment.

        - If ambiguity_flags is empty but confidence < 0.9, discount confidence
        - If unsupported_claims is empty on iteration > 0, add skepticism penalty
        - Weight issue counts more than stated confidence
        """
        conf_weight = self._config.confidence_count_weight

        # Calibrated definition confidence
        def_conf = raw.definition_confidence
        if not raw.ambiguity_flags and def_conf < 0.9:
            def_conf *= 0.8  # discount: claims no issues but not highly confident
        # Blend stated confidence with issue-count-based estimate
        issue_based_def = clamp(1.0 - len(raw.ambiguity_flags) * 0.15)
        def_conf = (1 - conf_weight) * def_conf + conf_weight * issue_based_def

        # Calibrated evidence confidence
        ev_conf = raw.evidence_confidence
        if not raw.unsupported_claims and iteration > 0:
            ev_conf -= self._config.empty_flags_skepticism_penalty
        issue_based_ev = clamp(1.0 - len(raw.unsupported_claims) * 0.15)
        ev_conf = (1 - conf_weight) * ev_conf + conf_weight * issue_based_ev

        # Calibrated task coverage
        cov = raw.task_coverage
        issue_based_cov = clamp(1.0 - len(raw.missing_elements) * 0.12)
        cov = (1 - conf_weight) * cov + conf_weight * issue_based_cov

        # Confidence calibration across iterations
        if len(self._iteration_assessments) >= 2:
            prev = self._iteration_assessments[-2]
            # If confidence went up but issues didn't decrease, discount
            if (raw.definition_confidence > prev.definition_confidence
                    and len(raw.ambiguity_flags) >= len(prev.ambiguity_flags)):
                def_conf *= 0.9

        return SelfAssessment(
            definition_confidence=clamp(def_conf),
            ambiguity_flags=raw.ambiguity_flags,
            evidence_confidence=clamp(ev_conf),
            unsupported_claims=raw.unsupported_claims,
            contradictions=raw.contradictions,
            task_coverage=clamp(cov),
            missing_elements=raw.missing_elements,
            refinement_priority=raw.refinement_priority,
            refinement_suggestion=raw.refinement_suggestion,
        )

    def _assessment_to_evidence(
        self, assessment: SelfAssessment, iteration: int
    ) -> list[Evidence]:
        """Convert calibrated assessment into N/N-N evidence items."""
        now = self._clock.now() if hasattr(self._clock, 'now') else 0.0
        items: list[Evidence] = []
        src = AgentID("self_assessment")
        prefix = f"iter{iteration}"

        # Definitional evidence from definition_confidence
        items.append(Evidence(
            id=EvidenceID(f"{prefix}_def_conf"),
            kind=EvidenceKind.DEFINITIONAL,
            claim=f"Definition confidence at iteration {iteration}",
            valence=assessment.definition_confidence,
            src=src,
            time=now,
            trust=0.8,
        ))

        # Negative definitional evidence from ambiguity flags
        for i, flag in enumerate(assessment.ambiguity_flags):
            items.append(Evidence(
                id=EvidenceID(f"{prefix}_ambiguity_{i}"),
                kind=EvidenceKind.DEFINITIONAL,
                claim=f"Ambiguity: {flag}",
                valence=-0.3,
                src=src,
                time=now,
                trust=0.9,
            ))

        # Epistemic evidence from evidence_confidence
        items.append(Evidence(
            id=EvidenceID(f"{prefix}_ev_conf"),
            kind=EvidenceKind.EPISTEMIC,
            claim=f"Evidence confidence at iteration {iteration}",
            valence=assessment.evidence_confidence,
            src=src,
            time=now,
            trust=0.8,
        ))

        # Negative epistemic from unsupported claims
        for i, claim in enumerate(assessment.unsupported_claims):
            items.append(Evidence(
                id=EvidenceID(f"{prefix}_unsupported_{i}"),
                kind=EvidenceKind.EPISTEMIC,
                claim=f"Unsupported: {claim}",
                valence=-0.3,
                src=src,
                time=now,
                trust=0.9,
            ))

        # Contradiction evidence (triggers conflict detection)
        for i, contradiction in enumerate(assessment.contradictions):
            items.append(Evidence(
                id=EvidenceID(f"{prefix}_contradiction_{i}"),
                kind=EvidenceKind.EPISTEMIC,
                claim=f"Contradiction: {contradiction}",
                valence=-0.5,
                src=src,
                time=now,
                trust=1.0,
            ))

        # Procedural evidence from task_coverage
        items.append(Evidence(
            id=EvidenceID(f"{prefix}_coverage"),
            kind=EvidenceKind.PROCEDURAL,
            claim=f"Task coverage at iteration {iteration}",
            valence=assessment.task_coverage,
            src=src,
            time=now,
            trust=0.8,
        ))

        # Negative procedural from missing elements
        for i, elem in enumerate(assessment.missing_elements):
            items.append(Evidence(
                id=EvidenceID(f"{prefix}_missing_{i}"),
                kind=EvidenceKind.PROCEDURAL,
                claim=f"Missing: {elem}",
                valence=-0.25,
                src=src,
                time=now,
                trust=0.9,
            ))

        return items

    def _compute_def_sem(self, assessment: SelfAssessment) -> float:
        """Semantic definedness from definition confidence and ambiguity flags."""
        # Blend confidence with inverse of ambiguity count
        ambiguity_penalty = len(assessment.ambiguity_flags) * 0.1
        return clamp(assessment.definition_confidence - ambiguity_penalty)

    def _compute_def_ep(self, assessment: SelfAssessment) -> float:
        """Epistemic definedness from evidence confidence and unsupported claims."""
        unsupported_penalty = len(assessment.unsupported_claims) * 0.1
        contradiction_penalty = len(assessment.contradictions) * 0.15
        return clamp(
            assessment.evidence_confidence - unsupported_penalty - contradiction_penalty
        )

    def _compute_def_proc(self, assessment: SelfAssessment) -> float:
        """Procedural definedness from task coverage and missing elements."""
        missing_penalty = len(assessment.missing_elements) * 0.1
        return clamp(assessment.task_coverage - missing_penalty)

    def reset(self) -> None:
        """Reset bridge state for a new run."""
        self._iteration_assessments.clear()


class IterationDiagnostics:
    """Diagnostic information from one iteration's N/N-N processing."""

    __slots__ = (
        "nu", "nu_raw", "nu_penalty", "def_sem", "def_ep", "def_proc",
        "conflict", "weakest_component", "calibrated_assessment",
    )

    def __init__(
        self,
        nu: float,
        nu_raw: float,
        nu_penalty: float,
        def_sem: float,
        def_ep: float,
        def_proc: float,
        conflict: float,
        weakest_component: str,
        calibrated_assessment: SelfAssessment,
    ) -> None:
        self.nu = nu
        self.nu_raw = nu_raw
        self.nu_penalty = nu_penalty
        self.def_sem = def_sem
        self.def_ep = def_ep
        self.def_proc = def_proc
        self.conflict = conflict
        self.weakest_component = weakest_component
        self.calibrated_assessment = calibrated_assessment

    def refinement_guidance(self) -> str:
        """Generate human-readable guidance for the next iteration."""
        if self.weakest_component == "def_sem":
            return "Improve clarity and reduce ambiguity in your definitions and terms."
        elif self.weakest_component == "def_ep":
            return "Strengthen evidence and justification for your claims."
        else:
            return "Address missing requirements and improve procedural completeness."
