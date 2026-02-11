# Null/Not-Null Logic (N/N-N) — Formal Specification v0.3.1

---

## Changelog from v0.3

| Change | Rationale |
|--------|-----------|
| Split ν into ν_raw + ν_penalties | Resolves ambiguity about penalty persistence vs recomputation |
| Added relevance_fn to policy | Makes evidence weighting explicit for Split/Recontextualize |
| Aggregate signature now includes target | Required for relevance computation |
| Aggregate returns pos_mass/neg_mass | Useful for diagnostics and penalty decisions |
| Added penalty decay mechanism | Prevents permanent damage from transient conflicts |
| Added penalty source tracking | Enables targeted penalty clearing |
| Split operator now uses relevance mode | Cleaner than evidence partitioning |
| Clarified licensing with ν_raw consideration | Distinguishes "vague" from "penalized" |
| Fixed off-by-one loop bug in Split | `0..<n` not `0..|n|` |
| Added θ_conflict_clear threshold | Triggers penalty decay when conflict resolves |

---

## 0. Scope, Goal, and Positioning

### 0.1 Goal

Provide a **definedness calculus** for managing knowledge states under uncertainty through iterative refinement. N/N-N tracks:

- **Definedness**: How well-specified is a concept or proposition?
- **Provenance**: Where did information come from and how was it transformed?
- **Refinement**: How do knowledge states evolve toward greater definition?
- **Decision support** (optional): How do we evaluate options when action is required?

### 0.2 Non-Goals

N/N-N does **not** replace truth semantics. It is a **meta-layer** that:

- Governs when truth evaluation is licensed
- Manages how definitions evolve
- Tracks information provenance across identity boundaries

Any truth semantics (Boolean, fuzzy, probabilistic, paraconsistent) can plug into the truth-status slot.

### 0.3 Positioning Statement

| Framework | Primary Concern | N/N-N Relationship |
|-----------|----------------|-------------------|
| Classical Logic | Truth preservation | N/N-N licenses when truth evaluation applies |
| Fuzzy Logic | Degrees of truth | N/N-N tracks degrees of *definedness*, orthogonal to truth |
| Probabilistic Logic | Degrees of belief | N/N-N tracks *what we're uncertain about*, not just *how uncertain* |
| Epistemic Logic | Knowledge/belief modalities | N/N-N operationalizes refinement process, not just modal states |
| Paraconsistent Logic | Contradiction tolerance | N/N-N can host paraconsistent semantics; treats contradiction as refinement signal |

### 0.4 Design Principles

1. **Definedness before truth**: Don't evaluate truth until concepts are sufficiently defined
2. **Refinement over classification**: Process of becoming defined is primary; states are secondary
3. **Provenance is first-class**: All information carries source, time, and trust metadata
4. **Contradiction is signal**: Conflict triggers refinement, not explosion
5. **Uncertainty is explicit**: Both epistemic uncertainty (Def_ep) and moral uncertainty (ν_u) are tracked
6. **Penalties are separable**: Situational adjustments (conflict, scope expansion) are tracked separately from structural definedness

---

## 1. Foundational Axiom

### 1.1 Epistemic Asymmetry

For any finite agent with bounded resources operating on knowledge set S within universe U:

```
|Refinable(S)| > |Refined(S)|
```

Where:
- `Refined(S)` = concepts/propositions with ν ≤ θ_defined
- `Refinable(S)` = concepts/propositions with ν > θ_defined

**Interpretation**: The space of what remains to be defined always exceeds what has been defined. This is a resource-relative claim, not a set-theoretic one about infinite cardinalities.

### 1.2 Corollaries

**C1 (Asymptotic Completion)**: Complete definition (ν = 0 for all x) is asymptotically approachable but never achieved.

**C2 (Refinement Priority)**: Given bounded resources, agents must prioritize which targets to refine. This motivates the QueryNext operator.

**C3 (Humility Constraint)**: Any claim of complete knowledge (ν = 0) should be treated with suspicion; practical systems should enforce ν ≥ ε for some small ε > 0.

---

## 2. Core Sets and Types

### 2.1 Universe

| Symbol | Type | Meaning |
|--------|------|---------|
| C | Set | **Concept tokens** (e.g., `Hotel`, `Fairness`, `User`) |
| P | Set | **Proposition tokens** (e.g., `p := "Feature F benefits users"`) |
| K | Set | **Contexts** |
| A | Set | **Agents/actors** |
| T | Ordered set | **Time domain** (timestamps, with < ordering) |
| Σ | Function | **Information state** (the global knowledge store) |

A **target** is any element x ∈ C ∪ P.

### 2.2 Information State Σ

**Definition**: Σ is a partial function:

```
Σ: (C ∪ P) × K ⇀ State
```

**Default initialization**: When (x, k) ∉ dom(Σ), accessing Σ[x, k] returns:

```
State₀ = ⟨
    t: ?,
    ν_raw: 1.0,
    ν_penalties: {},
    e: EvidenceSet.empty(),
    m: ⟨creation: now(), last_modified: now(), history: [], crossings: [],
       conflict_last_applied: null, penalty_clear_start: null, tags: {}⟩
⟩
```

### 2.3 Context Structure

A context k ∈ K is a tuple:

```
k = ⟨id, A, B, τ, π, σ, ρ⟩
```

| Component | Type | Meaning |
|-----------|------|---------|
| id | ContextID | Unique identifier |
| A | ⊆ A | **I-side**: Agents/perspectives treated as "self" |
| B | ⊆ A | **Not-I-side**: External agents/perspectives |
| τ | Interval over T | **Time window**: When this context applies |
| π | Policy | **Refinement policy**: Trust rules, weights, thresholds |
| σ | ⊆ C ∪ P | **Scope**: Which targets this context governs |
| ρ | A → Role | **Role function**: Agent role assignments |

### 2.4 Role Function

```
Role = {I, NotI, Both, Unknown}
```

| Role | Meaning | Boundary Transform |
|------|---------|-------------------|
| I | Fully trusted internal agent | No transform |
| NotI | External agent | Full boundary transform |
| Both | Partially internal (e.g., coalition member) | Partial transform (trust × π.coalition_factor) |
| Unknown | Agent not classified | Conservative transform (treat as NotI with extra discount) |

### 2.5 Penalty Source Types

```
PenaltySource = {
    conflict,
    scope_expansion,
    merge_rupture,
    category_error,
    manual
}
```

### 2.6 Policy Structure

```
Policy = ⟨
    θ_eval: 0.4,          -- ν threshold for licensing truth evaluation
    θ_eval_raw: 0.5,      -- ν_raw threshold (structural vagueness)
    θ_null: 0.7,           -- above this, target is considered NULL
    θ_defined: 0.3,        -- below this, target is considered NOT_NULL
    θ_conflict: 0.3,       -- conflict level that triggers penalty
    θ_conflict_clear: 0.15,-- conflict level that starts penalty decay
    θ_utility: 0.5,        -- threshold for utility computation (stub)

    w_sem: 0.4,            -- weight for semantic definedness
    w_ep: 0.35,            -- weight for epistemic definedness
    w_proc: 0.25,          -- weight for procedural definedness

    max_conflict_penalty: 0.2,
    conflict_cooldown: 1_hour,
    penalty_mode: max,      -- how to combine penalties: max or sum
    penalty_decay_enabled: true,
    penalty_decay_factor: 0.9,
    penalty_clear_window: 24_hours,

    relevance_fn: (Evidence, Target, Context) → float  -- default returns 1.0
⟩
```

---

## 3. Evidence

### 3.1 Evidence Item

```
Evidence = ⟨id, kind, claim, valence, src, time, trust, metadata⟩
```

- **kind** ∈ {epistemic, definitional, procedural}
- **valence** ∈ [-1, 1]: positive = supporting, negative = opposing
- **trust** ∈ [0, 1]: confidence in this evidence

### 3.2 Evidence Kinds

| Kind | Meaning | Affects |
|------|---------|---------|
| epistemic | Observations, measurements, testimony | Def_ep |
| definitional | What something IS or IS NOT | Def_sem |
| procedural | How to verify, test, or apply | Def_proc |

### 3.3 Evidence Identity

```
evidence_id = hash(kind, claim, src, time_bucket(time, granularity))
```

- **strict** dedup: skip if same id exists
- **corroboration** mode: allow same claim from different sources

---

## 4. Vagueness (ν)

### 4.1 Structure

```
ν_raw ∈ [0, 1]         -- structural vagueness from definedness
ν_penalties: PenaltySource → float  -- situational penalties
ν ∈ [0, 1]             -- total vagueness (always derived)
```

### 4.2 Computation

```
ν_penalty = max(penalties.values()) if penalty_mode == "max"
            else min(1.0, sum(penalties.values()))
ν = clamp(ν_raw + ν_penalty, 0, 1)
```

ν is **always derived**, never stored. ν_raw and ν_penalties are stored.

### 4.3 Licensing

Two conditions must BOTH hold for truth evaluation:

```
ν_raw ≤ θ_eval_raw AND ν ≤ θ_eval
```

This distinguishes:
- **Structurally vague**: ν_raw > θ_eval_raw (not enough definition)
- **Clear but penalized**: ν_raw ≤ θ_eval_raw but ν > θ_eval (penalties block evaluation)

---

## 5. Definedness

```
Def = w_sem × Def_sem + w_ep × Def_ep + w_proc × Def_proc
ν_raw = 1 - Def
```

### 5.1 Def_sem (Semantic Definedness)

```
Def_sem = mean(ontology_coverage, 1 - ambiguity, constraint_coverage, boundary_precision)
```

Subfunctions are domain-specific and pluggable via SemanticDefinednessProvider protocol.

### 5.2 Def_ep (Epistemic Definedness)

Derived from epistemic evidence mass.

### 5.3 Def_proc (Procedural Definedness)

Derived from procedural evidence mass.

---

## 6. Aggregate

### 6.1 Aggregate Function

```
Aggregate(evidence, target, context, relevance_fn) → AggregateResult
```

- Relevance checked per evidence item, per target
- Decay applied at aggregation time
- Returns pos_mass, neg_mass, conflict, def_ep

### 6.2 Conflict

```
conflict = 2 × min(pos_mass, neg_mass) / (pos_mass + neg_mass)
```

---

## 7. Boundary Transform

Adjusts evidence trust based on agent roles:

| Role | Transform |
|------|-----------|
| I | No change |
| NotI | trust × not_i_trust_factor |
| Both | trust × coalition_factor |
| Unknown | trust × unknown_trust_factor |

---

## 8. Refinement Operators

All operators: (State, ...) → (State', RefinementRecord)

### 8.1 Incorporate
Add new evidence, recompute definedness.

### 8.2 NegDefine
Add constraints, improve Def_sem.

### 8.3 Merge
Union evidence from multiple states, detect conflict.

### 8.4 (reserved)

### 8.5 Recontextualize
Move state to new context, record crossing.

### 8.6 Conflict
Detect conflict in evidence, apply penalty with cooldown. Start penalty decay timer when conflict drops below θ_conflict_clear.

### 8.7 QueryNext
Select the most vague target for refinement priority.

### 8.8 Split
Split parent into children:
- Children get **copy** of parent evidence (not partition)
- Children start with **empty penalties** (fresh start)
- Each child gets its own relevance_override
- ν_raw recomputed per child

### 8.9 Decay
Apply time-based evidence staleness.

### 8.10 PenaltyDecay
Decay penalties over time when conflict resolves.

---

## 9-11. (Extended topics)

See implementation for details on velocity monitoring, system health, and trace mode.

---

## 12. Worked Example: Feature Rollout Decision

### Step 0: Initialize
Target: "Feature F rollout"
- ν_raw = 0.95, ν = 0.95

### Step 1: Incorporate E1(+0.7), E2(-0.5), E3(-0.6)
- Def_sem = 0.10, Def_ep = 0.40, Def_proc = 0.30
- Def = 0.4(0.10) + 0.35(0.40) + 0.25(0.30) = 0.255
- ν_raw = 1 - 0.255 = 0.745

### Step 1b: Conflict detection
- pos_mass = 0.532, neg_mass = 0.77
- conflict = 2 × min(0.532, 0.77) / (0.532 + 0.77) = 0.82
- penalty = min(0.2, 0.82 × 0.2) = 0.164
- ν = clamp(0.745 + 0.164) = 0.909

### Step 2: NegDefine (5 constraints)
- Def_sem improved to 0.55
- Def = 0.4(0.55) + 0.35(0.40) + 0.25(0.30) = 0.435
- ν_raw = 0.565
- ν = clamp(0.565 + 0.164) = 0.729

### Step 3: Split into segments
- p2 (power users): relevance filtering reduces neg_mass
- p2: Def_sem = 0.70, Def_ep = 0.55, Def_proc = 0.40
- p2: Def = 0.4(0.70) + 0.35(0.55) + 0.25(0.40) = 0.5725
- p2: ν_raw = 0.427, ν_penalty = 0 (fresh start), ν = 0.427

### Step 4: Final NegDefine on p2
- Def_sem = 0.80
- Def = 0.4(0.80) + 0.35(0.55) + 0.25(0.40) = 0.6125
- ν_raw = 0.387, ν = 0.387
- **Licensed**: ν_raw (0.387) ≤ θ_eval_raw (0.5) ✓ AND ν (0.387) ≤ θ_eval (0.4) ✓

---

## 13. Algebraic Invariants

- **I1**: ν = clamp(ν_raw + ν_penalty, 0, 1) after every operation
- **I2**: Every ν change produces a RefinementRecord
- **I3**: ν_penalties keys are always valid PenaltySource values
- **I4**: Conflict penalty respects cooldown
- **I5**: Evidence partitions (epistemic, definitional, procedural) are disjoint

---

## 14. Future Extensions (Out of Scope for v0.3.1)

### 14.1 Semantic Object
Vector embeddings, graph DB integration for richer Def_sem.

### 14.2 Spirit Loop
Async refinement loop that continuously processes targets.

### 14.3 Normative/Utility Extension
Decision support with utility computation (§4.4 stub).
