"""ModelProvider protocol and mock implementations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class SelfAssessment:
    """LLM's structured reflection on its own output."""

    # Semantic clarity
    definition_confidence: float = 0.5
    ambiguity_flags: tuple[str, ...] = ()

    # Epistemic grounding
    evidence_confidence: float = 0.5
    unsupported_claims: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()

    # Procedural completeness
    task_coverage: float = 0.5
    missing_elements: tuple[str, ...] = ()

    # Refinement direction
    refinement_priority: str = ""
    refinement_suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition_confidence": self.definition_confidence,
            "ambiguity_flags": list(self.ambiguity_flags),
            "evidence_confidence": self.evidence_confidence,
            "unsupported_claims": list(self.unsupported_claims),
            "contradictions": list(self.contradictions),
            "task_coverage": self.task_coverage,
            "missing_elements": list(self.missing_elements),
            "refinement_priority": self.refinement_priority,
            "refinement_suggestion": self.refinement_suggestion,
        }

    def to_summary(self) -> str:
        lines = [
            f"Definition confidence: {self.definition_confidence:.2f}",
            f"Ambiguity flags: {list(self.ambiguity_flags)}",
            f"Evidence confidence: {self.evidence_confidence:.2f}",
            f"Unsupported claims: {list(self.unsupported_claims)}",
            f"Contradictions: {list(self.contradictions)}",
            f"Task coverage: {self.task_coverage:.2f}",
            f"Missing elements: {list(self.missing_elements)}",
            f"Refinement priority: {self.refinement_priority}",
            f"Refinement suggestion: {self.refinement_suggestion}",
        ]
        return "\n".join(lines)


@runtime_checkable
class ModelProvider(Protocol):
    def generate(self, prompt: str) -> str: ...
    def assess(self, task: str, output: str, prompt: str) -> SelfAssessment: ...


@dataclass
class MockProvider:
    """Returns pre-scripted (output, assessment) pairs in sequence.

    If iterations exceed the script length, repeats the last entry.
    """

    script: list[tuple[str, SelfAssessment]] = field(default_factory=list)
    _call_index: int = field(default=0, init=False, repr=False)

    def generate(self, prompt: str) -> str:
        if not self.script:
            return ""
        idx = min(self._call_index, len(self.script) - 1)
        self._call_index += 1
        return self.script[idx][0]

    def assess(self, task: str, output: str, prompt: str) -> SelfAssessment:
        if not self.script:
            return SelfAssessment()
        # Assessment is paired with the previous generate call
        idx = min(self._call_index - 1, len(self.script) - 1)
        idx = max(0, idx)
        return self.script[idx][1]

    def reset(self) -> None:
        self._call_index = 0


# Type alias for scripted provider functions
IterationHistory = list[tuple[str, SelfAssessment]]
ScriptFn = Callable[[int, IterationHistory], tuple[str, SelfAssessment]]


@dataclass
class ScriptedProvider:
    """Flexible mock that uses a function to generate responses.

    The function receives the iteration number and full history,
    and returns the next (output, assessment) pair.
    """

    script_fn: ScriptFn = field(default=lambda i, h: ("", SelfAssessment()))
    _history: list[tuple[str, SelfAssessment]] = field(
        default_factory=list, init=False, repr=False
    )
    _iteration: int = field(default=0, init=False, repr=False)

    def generate(self, prompt: str) -> str:
        result = self.script_fn(self._iteration, list(self._history))
        self._history.append(result)
        self._iteration += 1
        return result[0]

    def assess(self, task: str, output: str, prompt: str) -> SelfAssessment:
        if not self._history:
            return SelfAssessment()
        return self._history[-1][1]

    def reset(self) -> None:
        self._history.clear()
        self._iteration = 0


# ---------- Pre-built scenarios ----------


def steady_improvement_scenario() -> ScriptFn:
    """LLM steadily improves: ν decreases each iteration, output changes."""

    def fn(iteration: int, history: IterationHistory) -> tuple[str, SelfAssessment]:
        # Each iteration: fewer issues, higher confidence, output changes
        n = iteration
        base_conf = min(0.3 + n * 0.15, 0.95)
        ambiguities = max(5 - n, 0)
        unsupported = max(4 - n, 0)
        missing = max(3 - n, 0)
        coverage = min(0.4 + n * 0.12, 0.98)

        output = f"Version {n + 1}: Refined output with {5 - ambiguities} clarifications applied."
        assessment = SelfAssessment(
            definition_confidence=base_conf,
            ambiguity_flags=tuple(f"ambiguity_{i}" for i in range(ambiguities)),
            evidence_confidence=base_conf,
            unsupported_claims=tuple(f"claim_{i}" for i in range(unsupported)),
            contradictions=(),
            task_coverage=coverage,
            missing_elements=tuple(f"element_{i}" for i in range(missing)),
            refinement_priority="clarity" if ambiguities > 0 else "polish",
            refinement_suggestion=f"Address remaining {ambiguities} ambiguities",
        )
        return output, assessment

    return fn


def premature_convergence_scenario() -> ScriptFn:
    """Output stabilizes early but ν remains high.

    Output freezes at iteration 3 but ν is still ~0.6.
    With continued prodding, output eventually changes again at iteration 5
    and genuinely converges at iteration 7.
    """

    outputs = [
        "Version 1: Initial rough draft with many gaps.",
        "Version 2: Added some structure, still vague.",
        "Version 3: Looks reasonable on the surface.",
        "Version 3: Looks reasonable on the surface.",  # frozen
        "Version 3: Looks reasonable on the surface.",  # frozen
        "Version 5: Actually found new issues to address.",  # unstuck
        "Version 6: Substantially improved with proper justification.",
        "Version 7: Fully refined, all requirements addressed.",
        "Version 7: Fully refined, all requirements addressed.",  # genuine convergence
    ]

    assessments = [
        SelfAssessment(
            definition_confidence=0.2, ambiguity_flags=("vague scope", "unclear terms", "missing context"),
            evidence_confidence=0.2, unsupported_claims=("claim A", "claim B", "claim C"),
            task_coverage=0.3, missing_elements=("req 1", "req 2", "req 3", "req 4"),
            refinement_priority="everything", refinement_suggestion="Start over with structure",
        ),
        SelfAssessment(
            definition_confidence=0.35, ambiguity_flags=("unclear terms", "missing context"),
            evidence_confidence=0.3, unsupported_claims=("claim A", "claim B"),
            task_coverage=0.45, missing_elements=("req 2", "req 3", "req 4"),
            refinement_priority="scope", refinement_suggestion="Clarify scope boundaries",
        ),
        SelfAssessment(
            definition_confidence=0.4, ambiguity_flags=("unclear terms",),
            evidence_confidence=0.35, unsupported_claims=("claim A", "claim B"),
            task_coverage=0.5, missing_elements=("req 3", "req 4"),
            refinement_priority="evidence", refinement_suggestion="Support claims",
        ),
        # Stuck — model says "looks fine" but issues remain
        SelfAssessment(
            definition_confidence=0.45, ambiguity_flags=("unclear terms",),
            evidence_confidence=0.35, unsupported_claims=("claim A",),
            task_coverage=0.55, missing_elements=("req 3", "req 4"),
            refinement_priority="evidence", refinement_suggestion="Support claims",
        ),
        SelfAssessment(
            definition_confidence=0.45, ambiguity_flags=("unclear terms",),
            evidence_confidence=0.38, unsupported_claims=("claim A",),
            task_coverage=0.55, missing_elements=("req 3",),
            refinement_priority="evidence", refinement_suggestion="Add justification",
        ),
        # Unstuck
        SelfAssessment(
            definition_confidence=0.7, ambiguity_flags=(),
            evidence_confidence=0.65, unsupported_claims=(),
            task_coverage=0.8, missing_elements=("req 3",),
            refinement_priority="completeness", refinement_suggestion="Address req 3",
        ),
        SelfAssessment(
            definition_confidence=0.85, ambiguity_flags=(),
            evidence_confidence=0.8, unsupported_claims=(),
            task_coverage=0.92, missing_elements=(),
            refinement_priority="polish", refinement_suggestion="Minor polish",
        ),
        SelfAssessment(
            definition_confidence=0.92, ambiguity_flags=(),
            evidence_confidence=0.88, unsupported_claims=(),
            task_coverage=0.97, missing_elements=(),
            refinement_priority="none", refinement_suggestion="Complete",
        ),
        SelfAssessment(
            definition_confidence=0.92, ambiguity_flags=(),
            evidence_confidence=0.88, unsupported_claims=(),
            task_coverage=0.97, missing_elements=(),
            refinement_priority="none", refinement_suggestion="Complete",
        ),
    ]

    def fn(iteration: int, history: IterationHistory) -> tuple[str, SelfAssessment]:
        idx = min(iteration, len(outputs) - 1)
        return outputs[idx], assessments[idx]

    return fn


def oscillation_scenario() -> ScriptFn:
    """ν bounces up and down — model keeps changing approach."""

    def fn(iteration: int, history: IterationHistory) -> tuple[str, SelfAssessment]:
        # Alternating good/bad: confidence oscillates
        is_good = iteration % 2 == 0
        conf = 0.7 if is_good else 0.35
        ambiguities = 1 if is_good else 3
        coverage = 0.75 if is_good else 0.5

        output = f"Approach {'A' if is_good else 'B'} — iteration {iteration + 1}"
        assessment = SelfAssessment(
            definition_confidence=conf,
            ambiguity_flags=tuple(f"ambiguity_{i}" for i in range(ambiguities)),
            evidence_confidence=conf,
            unsupported_claims=tuple(f"claim_{i}" for i in range(2 if is_good else 4)),
            contradictions=("approach conflict",) if not is_good else (),
            task_coverage=coverage,
            missing_elements=tuple(f"missing_{i}" for i in range(1 if is_good else 3)),
            refinement_priority="consistency",
            refinement_suggestion="Pick one approach and stick with it",
        )
        return output, assessment

    return fn


def stuck_scenario() -> ScriptFn:
    """Nothing changes — model is completely stuck."""

    def fn(iteration: int, history: IterationHistory) -> tuple[str, SelfAssessment]:
        return "The same mediocre output every time.", SelfAssessment(
            definition_confidence=0.4,
            ambiguity_flags=("vague_term",),
            evidence_confidence=0.35,
            unsupported_claims=("main_claim",),
            task_coverage=0.5,
            missing_elements=("key_requirement",),
            refinement_priority="everything",
            refinement_suggestion="Need a different approach entirely",
        )

    return fn
