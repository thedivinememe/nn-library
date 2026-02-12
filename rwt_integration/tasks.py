"""Example task definitions for RWT evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    id: str
    description: str
    specification: str
    complexity: str  # "simple" | "moderate" | "complex"
    expected_challenges: tuple[str, ...] = ()
    quality_rubric: dict[str, str] = field(default_factory=dict)


TASK_EMAIL_VALIDATOR = Task(
    id="email_validator",
    description="Write a Python function that validates email addresses.",
    specification="""\
Write a Python function `validate_email(email: str) -> bool` that:
1. Accepts valid email addresses per RFC 5322 simplified rules
2. Rejects emails without @ symbol
3. Rejects emails with spaces
4. Requires at least one character before @
5. Requires a valid domain with at least one dot
6. Returns True for valid, False for invalid
Include docstring and type hints.""",
    complexity="simple",
    expected_challenges=("clear spec", "well-defined problem"),
    quality_rubric={
        "correctness": "Handles edge cases (empty string, multiple @, etc.)",
        "completeness": "All 6 requirements addressed",
        "code_quality": "Clean, readable, well-documented",
    },
)


TASK_RATE_LIMITER = Task(
    id="rate_limiter_spec",
    description="Write a technical specification for a rate limiter.",
    specification="""\
Write a technical specification for a rate limiter that handles:
1. Per-user rate limits (e.g., 100 requests/minute per user)
2. Global rate limits (e.g., 10,000 requests/minute total)
3. Backpressure signaling (how callers know they're being limited)
4. Burst handling (brief spikes above the limit)
5. Configuration (how limits are set and changed)

The spec should include:
- Data structures used
- Algorithm description (e.g., token bucket, sliding window)
- API surface (function signatures)
- Error handling and edge cases
- Monitoring/observability hooks""",
    complexity="moderate",
    expected_challenges=(
        "ambiguity in requirements",
        "design tradeoffs between algorithms",
        "balancing per-user and global limits",
    ),
    quality_rubric={
        "completeness": "All 5 functional requirements + 5 spec sections addressed",
        "specificity": "Concrete data structures and algorithms, not hand-waving",
        "tradeoffs": "Acknowledges and justifies design choices",
        "edge_cases": "Handles clock skew, distributed deployment, etc.",
    },
)


TASK_DATABASE_ANALYSIS = Task(
    id="database_analysis",
    description="Analyze database architecture tradeoffs for multi-tenant SaaS.",
    specification="""\
Analyze the tradeoffs between three database architectures for a multi-tenant
SaaS platform and recommend one with justified reasoning.

The three architectures to compare:
1. Shared database, shared schema (tenant_id column)
2. Shared database, separate schemas (one schema per tenant)
3. Separate databases (one database per tenant)

Evaluation criteria:
- Data isolation and security
- Performance at scale (100, 1000, 10000 tenants)
- Operational complexity (backup, migration, monitoring)
- Cost efficiency
- Customization flexibility per tenant
- Compliance requirements (GDPR, SOC2)

Your analysis should:
- Compare all three on each criterion
- Identify which scenarios favor which architecture
- Make a concrete recommendation with conditions
- Acknowledge uncertainty where it exists""",
    complexity="complex",
    expected_challenges=(
        "competing concerns with no single right answer",
        "requires balancing multiple evaluation criteria",
        "high potential for internal conflict between criteria",
    ),
    quality_rubric={
        "breadth": "All 3 architectures × 6 criteria covered",
        "depth": "Specific numbers, not just 'better/worse'",
        "nuance": "Recognizes context-dependence, not one-size-fits-all",
        "justification": "Recommendation backed by clear reasoning chain",
    },
)


ALL_TASKS = [TASK_EMAIL_VALIDATOR, TASK_RATE_LIMITER, TASK_DATABASE_ANALYSIS]
