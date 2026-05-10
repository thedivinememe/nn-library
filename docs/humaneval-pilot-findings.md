# HumanEval Convergence Pilot: Findings

**Author:** Brandon Welner
**Status:** Pilot complete. Full study deferred pending re-scoping.
**Date:** May 2026
**Companion to:** [scope-convergence-experiment.md](./scope-convergence-experiment.md)
**Builds on:** [rwt-integration-findings.md](./rwt-integration-findings.md)

---

## Headline

The HumanEval pilot ran 5 problems × 3 strategies × 1 seed = 15 trajectories, each up to 12 iterations, on Claude Sonnet 4.6 via the existing `EvaluationHarness`. The integration works end-to-end (no fatal errors, no parse failures, ν trajectories produced as expected). The pre-registered convergence question — *does ν-based stopping beat diff-based stopping on HumanEval?* — could not be answered on this setup, because **no strategy ever fired**: all 15 trajectories ran to the max-iterations cap. The reason is that Sonnet's outputs do not converge on HumanEval-style refinement loops; they oscillate, and self-refinement most often *raises* ν rather than lowering it. The 14/15 test pass rate is misleadingly good — it reflects iteration-0 correctness, not refinement success. One trajectory shows a passing iteration-0 answer being corrupted into a failing one over subsequent iterations. ν correctly tracked that degradation.

This is the "self-refinement may hurt or no-op" risk that the scope doc pre-registered, surfacing in real data.

---

## What was run

| Parameter | Value |
|---|---|
| Benchmark | HumanEval, problems 0–4 |
| Model | Claude Sonnet 4.6 (`claude-sonnet-4-6`), temperature 0.7 |
| Strategies | Diff, Nu, Hybrid (LoopConfig defaults) |
| Seeds | 1 per (problem, strategy) |
| Max iterations | 12 |
| Total trajectories | 15 |
| Total API calls | ~360 (each iteration: 1 generate + 1 self-assessment) |
| Wall-clock | ~10 minutes at concurrency 4 |

Provider: `ClaudeProvider` (real API), JSON-parsed `SelfAssessment`.

---

## Results

### Integration verification — works

| Check | Result |
|---|---|
| Trajectories completed without fatal error | 15 / 15 |
| Self-assessment JSON parsing | 0 / 180 iterations show parse-failure default signature (`def_sem = def_ep = def_proc = 0.5`) |
| Definedness components vary across iterations | Yes — `def_sem`, `def_ep`, `def_proc` move meaningfully |
| ν trajectories produced | Yes — non-trivial range across all runs |

The new `ClaudeProvider`, the JSON parser, the HumanEval task adapter, and the test executor all behave as designed. The harness produces clean, complete trajectory records.

### Convergence detection — did not fire

| Strategy | Trajectories that converged before max-iter | Mean iterations | Median iterations |
|---|---|---|---|
| Diff | 0 / 5 | 12 | 12 |
| Nu | 0 / 5 | 12 | 12 |
| Hybrid | 0 / 5 | 12 | 12 |

**No strategy declared convergence on any trajectory.** All 15 hit the iteration cap.

The reason is in the volatility data — ν moves substantially every iteration:

| Trajectory | ν range | Mean \|Δν\| | Stable threshold |
|---|---|---|---|
| HumanEval/0 diff | 0.503 | 0.156 | 0.010 |
| HumanEval/0 hybrid | 0.515 | 0.190 | 0.010 |
| HumanEval/0 nu | 0.503 | 0.203 | 0.010 |
| HumanEval/2 diff | 0.728 | 0.202 | 0.010 |

ν is moving 15-20× the stability threshold per iteration. There is no ν-stable region for `NuConvergence` or `HybridConvergence` to detect. Output diff is similarly active — `output_diff_ratio` ranges 0.2–0.5 throughout, well above the 0.05 stability threshold for `DiffConvergence`. The model keeps rewriting its solution every iteration.

### Test pass rates — apparently strong, structurally misleading

| Strategy | Passed | Failed | Pass rate |
|---|---|---|---|
| Diff | 5 | 0 | 100% |
| Nu | 5 | 0 | 100% |
| Hybrid | 4 | 1 | 80% |

These look like Sonnet is doing well on HumanEval, and that's true — but the pass rate reflects the *initial iteration*, not the refinement loop. On most problems, the iteration-0 ν is already low (0.16–0.33), suggesting the model produced a likely-correct answer immediately. The 12 iterations that follow are not adding correctness; in many cases they reduce it.

### Self-refinement degraded ν

Across the 15 trajectories, comparing initial ν to final ν:

| Direction | Count |
|---|---|
| ν decreased (refinement helped) | 2 |
| ν approximately flat | 1 |
| ν increased (refinement hurt) | 12 |

The dominant pattern is *initial ν is the lowest ν of the trajectory.* The model produces its best-defined output on iteration 0, and subsequent self-refinement makes things less defined.

The single failed trajectory — **HumanEval/0 hybrid** — illustrates the worst case. Initial ν was 0.162 with a likely-correct answer. By iteration 3, ν was 0.662; by iteration 11, ν was 0.598 with a failing solution. The Hybrid strategy correctly emitted `spinning_detected` at iteration 2 and `oscillation_detected` from iteration 6 onward — but having no early-stopping signal that meets the licensing threshold, the loop continued, and the model corrupted its working answer.

### ν tracked the degradation

The interesting observation hiding in the failure: ν correctly distinguished trajectories where the answer was getting worse from trajectories that were stable. On HumanEval/0 hybrid (the one that broke a passing answer), ν climbed from 0.162 to 0.598 over the trajectory. On HumanEval/3 hybrid (passing throughout), ν stayed stable around 0.15. Output diff *cannot* distinguish these cases — both had similar diff_ratios. ν did.

---

## Interpretation

### The original convergence question can't be answered on this setup

The scope doc framed the experiment as a comparison of stopping methods. That comparison requires a phenomenon — *convergence* — for the methods to detect. Sonnet on HumanEval does not exhibit convergence in self-refinement loops; the model keeps rewriting and ν keeps moving. There is nothing for any of the strategies to stop on. Continuing to the full 1,476-trajectory study would not produce a different answer here, because the underlying behavior is the same.

This matches two pre-registered risks from the scope doc:

- *"Self-refinement may hurt or no-op"* (Huang et al.): the dominant pattern observed.
- *"HumanEval saturation"*: 14/15 trajectories passed because iteration-0 was already correct.

The scope doc explicitly said both outcomes are publishable. This is the negative outcome, with mechanism.

### A different question emerged from the data

The pilot suggests a more pointed question that the data already partially supports:

> **Does ν correctly distinguish self-refinement trajectories that preserve correctness from trajectories that degrade it — in cases where diff-based signals structurally cannot make that distinction?**

This is a different shape of question. It is not "which signal is a better convergence detector"; it is "does ν have information content that diff lacks." Diff measures whether the output changed; ν is claimed to track *definedness*. Trajectories where output keeps changing but the answer stays correct (HumanEval/3 hybrid: ν stable ~0.15) and trajectories where output keeps changing while the answer is corrupted (HumanEval/0 hybrid: ν climbing to 0.6) are indistinguishable to diff and clearly distinguishable to ν.

The reframed experiment would test:

- *Does final ν predict final test correctness across many trajectories?*
- *Does maximum-ν during a trajectory predict whether self-refinement degraded the output?*
- *Do ν-aware policies (e.g., return to the lowest-ν checkpoint, stop on first ν increase) outperform diff-based policies on final correctness?*

These map onto N/N-N's actual theoretical content (ν tracks definedness, not just stability) more directly than the convergence question does.

### What the pilot does and does not justify

It justifies:
- Confidence that the integration is sound.
- Belief that the convergence question as scoped cannot be answered on Sonnet + HumanEval.
- A plausible reframe toward degradation detection, supported by partial data.

It does not justify:
- Strong claims about ν as a degradation detector — the sample is 15 trajectories on 5 problems with one seed.
- Generalization to other models or other benchmarks.
- Vindication of the framework. The pre-registered hypothesis was not supported.

---

## Limitations

- **Sample size.** 5 problems × 1 seed is enough to falsify "convergence happens" but not enough to establish anything robust about the reframe. Any claim from this pilot needs more data.
- **One model.** Sonnet's iteration-0 capability on HumanEval is the main reason convergence doesn't fire. Less capable models might exhibit genuine refinement and convergence; this pilot says nothing about those cases.
- **One benchmark.** HumanEval may be uniquely saturated for Sonnet; harder benchmarks (HumanEval+, LiveCodeBench, APPS) might tell a different story.
- **The refinement prompt does not include test feedback.** The model self-assesses and refines without seeing whether tests pass. This was a deliberate scope choice (test the framework, not test-driven loops), but it likely contributes to the "self-correction hurts" pattern.
- **Temperature 0.7 introduces stochastic variance.** The single-seed results are not the population mean.

---

## Implications for the planned full study

The full 1,476-trajectory study as scoped should not run as-is. Spending ~$300 to establish that convergence doesn't happen at scale, when 5 problems already establish it doesn't happen at all, is not a useful experiment.

Three possible adjustments, in increasing order of scope change:

1. **Switch benchmark, keep the question.** Re-run the pilot on HumanEval+ or LiveCodeBench, where Sonnet is less saturated and genuine refinement may be needed. If those benchmarks exhibit actual convergence behavior (output stabilizing, ν decreasing), the original convergence question becomes answerable and the full study can proceed.
2. **Reframe to degradation detection.** Use the same infrastructure to ask whether ν predicts trajectory quality — this is the question the pilot data already partially supports. Same harness, different metrics, different statistical analysis. The published result is "ν as a degradation signal in self-refinement loops" instead of "ν as a convergence signal."
3. **Switch to a model where iteration-0 is not already correct.** Haiku 4.5 or an older model. Sonnet's capability is the proximate cause of the convergence-doesn't-happen result; a less capable model leaves more room for refinement.

Each of these is a coherent next step. The pilot does not by itself decide between them.

---

## Next steps

1. Decide between the three adjustment paths above.
2. Run a 5-problem pilot of whichever path is chosen, using the existing harness, before committing to a full study.
3. Once a setup is found where the chosen question has empirical bite, scale to the previously-scoped sample size.
4. If reframe is chosen: write the analysis script for ν-vs-correctness correlation across trajectories. The infrastructure to gather that data already exists.

---

## Reproducibility

All raw trajectory JSONs are in `runs/` — 15 files, one per (problem, strategy, seed). The pilot can be re-run with:

```bash
python -m rwt_integration.humaneval_runner --pilot
```

from the repo root. Total cost: roughly $2-5 in API calls.
