# N/N-N Logic — Reference Implementation

Reference implementation of **Null/Not-Null (N/N-N) Logic v0.3.1**, a definedness calculus for managing knowledge states under uncertainty.

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from nn_logic.types import (
    AgentID, ContextID, Evidence, EvidenceID, EvidenceKind,
    EvidenceSet, MockClock, TargetID,
)
from nn_logic.state import make_initial_state
from nn_logic.operators import incorporate, neg_define, apply_conflict
from nn_logic.query import query, is_licensed
from nn_logic.policy import Policy
from nn_logic.trace import Tracer

# Create a clock and policy
clock = MockClock(start=1000.0)
policy = Policy()

# Initialize a target in a context
target = TargetID("feature_rollout")
ctx = ContextID("product_decision")
state = make_initial_state(target, ctx, clock)

# Incorporate evidence
e1 = Evidence(
    id=EvidenceID("e1"),
    kind=EvidenceKind.EPISTEMIC,
    claim="Strong user demand for feature",
    valence=0.7,
    src=AgentID("product_team"),
    time=clock.now(),
    trust=0.8,
)
state, record = incorporate(state, [e1], policy=policy, clock=clock)

# Add constraints via NegDefine
state, record = neg_define(
    state,
    ["must be feature-flaggable", "must not break existing API"],
    policy=policy,
    clock=clock,
)

# Check if truth evaluation is licensed
response = query(state, policy)
print(f"Licensed: {response.licensed}")
print(f"ν_raw: {response.nu_raw:.3f}, ν: {response.nu:.3f}")
print(f"Reason: {response.reason.value}")
```

## Architecture

All operators are **pure functions** that take a `State` and return a new `State` plus a `RefinementRecord`. No mutation, no side effects. Time is injected via a `Clock` protocol.

### Module Overview

| Module | Purpose |
|--------|---------|
| `types.py` | Core types, enums, dataclasses, protocols |
| `policy.py` | Policy configuration with defaults |
| `evidence.py` | Evidence creation, dedup, identity |
| `state.py` | State and Σ (information state) management |
| `aggregate.py` | Aggregate function, compute_conflict |
| `definedness.py` | Def, Def_sem, Def_ep, Def_proc (pluggable) |
| `boundary.py` | Trust adjustment based on agent roles |
| `operators.py` | All refinement operators |
| `query.py` | Query, DecisionQuery, is_licensed |
| `velocity.py` | Refinement velocity monitoring |
| `trace.py` | RefinementRecord collection for debugging |
| `helpers.py` | Utility functions (clamp, compute_nu) |

### Key Concepts

- **ν (nu)**: Vagueness score in [0, 1]. Lower = more defined.
- **ν_raw**: Structural vagueness (derived from definedness).
- **ν_penalties**: Situational penalties (conflict, merge rupture, etc.).
- **ν = clamp(ν_raw + max(penalties), 0, 1)** — always derived, never stored directly.
- **Licensed**: Truth evaluation is licensed when `ν_raw ≤ θ_eval_raw AND ν ≤ θ_eval`.

## Running Tests

```bash
pytest tests/ -v
```

The §12 worked example (`tests/test_worked_example.py`) is the golden test that validates the full operator pipeline.

## Spec

The formal specification is at `docs/spec-v0.3.1.md`.
