"""Loop configuration and prompt templates."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LoopConfig:
    """Configuration for the RWT loop."""

    # Iteration limits
    max_iterations: int = 10
    min_iterations: int = 1

    # Convergence thresholds
    nu_licensing_threshold: float = 0.4
    nu_raw_licensing_threshold: float = 0.5
    diff_similarity_threshold: float = 0.95
    diff_stable_count: int = 2  # consecutive stable iterations to trigger Diff stop
    nu_stable_epsilon: float = 0.01  # Δν below this is "stable"
    nu_stable_count: int = 2  # consecutive stable ν iterations

    # Oscillation detection
    oscillation_limit: int = 3  # oscillation count before flagging

    # N/N-N policy weights
    w_sem: float = 0.4
    w_ep: float = 0.35
    w_proc: float = 0.25

    # Calibration
    empty_flags_skepticism_penalty: float = 0.1
    confidence_count_weight: float = 0.6  # weight issue counts vs stated confidence

    # Prompt templates
    assessment_prompt_template: str = field(default="""\
You have just produced the following output for a task. Assess your own work honestly.

TASK: {task_specification}

YOUR OUTPUT:
{current_output}

Provide a structured self-assessment in the following JSON format:
{{
    "definition_confidence": <0.0-1.0, how clearly defined and unambiguous is your output?>,
    "ambiguity_flags": [<list specific ambiguities, unclear terms, or vague statements>],
    "evidence_confidence": <0.0-1.0, how well-supported are the claims and decisions?>,
    "unsupported_claims": [<list claims that lack clear justification>],
    "contradictions": [<list any internal contradictions or inconsistencies>],
    "task_coverage": <0.0-1.0, what fraction of the task requirements are addressed?>,
    "missing_elements": [<list requirements not yet addressed>],
    "refinement_priority": "<which single aspect most needs improvement>",
    "refinement_suggestion": "<specific action to take next>"
}}

Be critical. Identify specific problems, not general impressions.
Rate yourself LOWER when you identify specific issues.
An output with zero ambiguity_flags and zero unsupported_claims should be rare.""")

    refinement_prompt_template: str = field(default="""\
You are refining your output for the following task.

TASK: {task_specification}

YOUR PREVIOUS OUTPUT:
{current_output}

SELF-ASSESSMENT:
{assessment_summary}

DEFINEDNESS DIAGNOSTICS:
- Overall vagueness (ν): {nu:.3f}
- Structural vagueness (ν_raw): {nu_raw:.3f}
- Penalty: {nu_penalty:.3f}
- Semantic definedness (Def_sem): {def_sem:.3f}
- Epistemic definedness (Def_ep): {def_ep:.3f}
- Procedural definedness (Def_proc): {def_proc:.3f}
- Conflict score: {conflict:.3f}
- Weakest component: {weakest_component}

GUIDANCE: {refinement_guidance}

Produce an improved version of your output that addresses the identified issues.
Focus especially on: {refinement_priority}""")

    initial_prompt_template: str = field(default="""\
{task_specification}

Produce a thorough, well-structured response.""")
