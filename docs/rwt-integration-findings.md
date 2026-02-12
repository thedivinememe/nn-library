# RWT × N/N-N Integration: Findings

## Thesis

Standard RWT (Ralph Wiggum Technique) convergence uses a **syntactic** signal — output diff. It can tell you *whether the output changed*, but not *whether it's good*. By integrating N/N-N Logic as a **metacognitive** convergence signal, we can track *how well-defined the output is* and make smarter stopping decisions.

We tested three convergence strategies:
- **Diff** — baseline, stops when output stabilizes
- **Nu** — stops when ν (vagueness) drops below licensing threshold and stabilizes
- **Hybrid** — requires both output stability AND ν below threshold

## Results

### Scenario 1: Premature Convergence

The critical test. The LLM's output freezes early while quality is still low, then eventually improves if pushed.

| Strategy | Iterations | Converged | Final ν | Outcome |
|----------|-----------|-----------|---------|---------|
| Diff     | 5         | Yes       | 0.405   | Stopped at output freeze — **premature** |
| Nu       | 11        | Yes       | 0.047   | Continued past freeze, converged properly |
| Hybrid   | 11        | Yes       | 0.047   | Continued past freeze, converged properly |

**ν trajectory (Diff):** 0.909 → 0.671 → 0.532 → 0.480 → **0.405 (stop)**

**ν trajectory (Hybrid):** 0.909 → 0.671 → 0.532 → 0.480 → 0.405 → 0.196 → 0.136 → 0.085 → **0.047 (stop)**

Diff stopped at iteration 5 with ν = 0.405 — right at the licensing boundary but with the output still carrying the problems it had when it froze at iteration 3. Hybrid and Nu pushed through 6 more iterations, driving ν down to 0.047 — genuinely well-defined output.

**The final ν gap is 8.6×** (0.405 vs 0.047). This is the core finding: metacognitive convergence signals prevent premature stopping.

### Scenario 2: Steady Improvement

The LLM improves consistently every iteration, with output changing each time.

| Strategy | Iterations | Converged | Final ν |
|----------|-----------|-----------|---------|
| Diff     | 12        | No        | 0.031   |
| Nu       | 8         | Yes       | 0.031   |
| Hybrid   | 12        | No        | 0.031   |

Nu converged earliest (iteration 8) because ν crossed the threshold and stabilized. Diff never converged because each iteration produced a slightly different output. Hybrid didn't converge because it requires *both* signals — the output kept changing even after ν stabilized.

**Observation:** On well-behaved tasks, Nu is the most efficient strategy. Hybrid is conservative — it won't stop until the model *also* stops changing its output, which can be wasteful when the changes are diminishing returns. This suggests Hybrid's output stability check could use a relaxed threshold for late iterations where ν is already low.

### Scenario 3: Stuck

The LLM produces identical output every iteration and the self-assessment never improves.

| Strategy | Iterations | Converged | Final ν |
|----------|-----------|-----------|---------|
| Diff     | 3         | Yes       | 0.422   |
| Hybrid   | 6 (max)   | No        | 0.422   |

Diff immediately declares convergence because the output is stable. Hybrid correctly identifies this as premature convergence (output stable but ν too high) and keeps trying until hitting max iterations. Neither strategy can *fix* the problem — the model is genuinely stuck — but Hybrid at least doesn't falsely declare success.

**Observation:** When a model is stuck, no convergence strategy can help. But Hybrid's refusal to converge gives the system a chance to trigger interventions (the strategy emits `premature_convergence_detected` diagnostics that could trigger a Split, Recontextualize, or different prompt strategy).

### Scenario 4: Oscillation

The LLM alternates between two approaches, never settling.

| Strategy | Iterations | Converged | ν trajectory |
|----------|-----------|-----------|-------------|
| Nu       | 8 (max)   | No        | 0.367 → 1.0 → 0.536 → 1.0 → 0.536 → ... |

The oscillation is clearly visible in the ν trajectory — it swings between ~0.4 and 1.0 on alternating iterations. Nu's oscillation detector flags this pattern and suggests "consider Recontextualize" — a directed intervention rather than blind continuation.

**Observation:** The ν trajectory is a powerful diagnostic even when it doesn't lead to convergence. A human or orchestrator watching the trajectory can spot oscillation patterns that are invisible in raw output diffs.

## Summary of Strategy Characteristics

| Property | Diff | Nu | Hybrid |
|----------|------|-----|--------|
| Catches premature convergence | No | Yes | Yes |
| Efficient on easy tasks | Yes | **Best** | Slightly wasteful |
| Handles stuck scenarios | Falsely converges | Correctly rejects | Correctly rejects |
| Detects oscillation | No | Yes | Yes |
| Provides directed guidance | No | Yes | Yes |
| False convergence risk | **High** | Low | **Lowest** |

## Key Insights

### 1. ν trajectory is more informative than output diff

Output diff is binary — changed or didn't. ν trajectory is a continuous signal that reveals:
- Rate of improvement (ν velocity)
- Stalling (flat ν above threshold)
- Oscillation (ν direction reversals)
- Which *component* is lagging (Def_sem vs Def_ep vs Def_proc)

### 2. Directed refinement > blind retry

When ν is high, the weakest component (semantic, epistemic, or procedural definedness) tells you *what to improve*, not just "try again." The refinement prompt includes diagnostics like "your semantic definedness is lowest — clarify your terms" instead of generic "please improve."

### 3. Calibration matters enormously

LLMs tend toward optimistic self-assessment. The calibration countermeasures (weighting issue counts over stated confidence, skepticism penalties for suspiciously clean self-reports) are essential. Without calibration, ν would track the model's confidence rather than actual quality.

### 4. Hybrid is the safest default, Nu is the most efficient

- Use **Hybrid** when false convergence is costly (production, high-stakes outputs)
- Use **Nu** when you want to minimize iterations (fast iteration, exploratory work)
- Use **Diff** only as a baseline for comparison — it has no quality awareness

### 5. The convergence gap is task-dependent

On simple, well-defined tasks, all strategies perform similarly. The gap widens with task complexity and ambiguity. The premature convergence scenario — where the model plateaus and then breaks through — is exactly the pattern that occurs with complex, multi-faceted tasks in practice.

## Limitations

- **Self-assessment accuracy:** The entire approach depends on the LLM producing calibrated self-assessments. Real models may be systematically biased in ways the calibration countermeasures don't catch.
- **Scripted scenarios:** These findings use deterministic scripted providers. Real LLM behavior is stochastic and may exhibit patterns not covered by the four scenarios tested.
- **No external quality judge:** We measure ν as a proxy for quality, but haven't validated that low ν correlates with externally-judged quality. The connection is theorized (well-defined + well-supported + complete → good), not yet empirically confirmed with real model outputs.
- **Computational cost:** Nu and Hybrid require an extra assessment call per iteration. Whether the improved convergence decisions justify the ~2× token cost per iteration depends on how expensive premature convergence is in your use case.

## Next Steps

1. **Wire in a real LLM provider** and run the evaluation harness on actual model outputs
2. **Add an external quality judge** (either human or a separate evaluator model) to validate that ν correlates with output quality
3. **Tune the Hybrid relaxation** — loosen the output stability requirement when ν is already well below threshold
4. **Implement Split/Recontextualize interventions** that trigger automatically from convergence diagnostics
5. **Explore ν-guided prompt engineering** — use the weakest Def component to dynamically adjust the refinement prompt style
