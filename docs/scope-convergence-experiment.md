# Scope: ν-Based Convergence Signals for Iterative Code Generation

**Author:** Brandon Welner
**Status:** Draft v0.1
**Date:** April 2026
**Target deliverable:** Blog post with reproducible code (GitHub), ~3-4 months
**Weekly budget:** 3-5 hours

---

## Claim

On iterative code generation with a real LLM, using N/N-N definedness trajectory (ν) as a convergence signal produces measurably different stopping behavior than diff-based convergence. Specifically, ν-based signals (Nu and Hybrid variants) are hypothesized to reduce the rate of *false convergence* — where a strategy declares the iteration done but the resulting code fails its tests — relative to diff-only convergence, with Nu and Hybrid exhibiting different quality/efficiency tradeoffs.

This is a hypothesis framed as a measurement, not a demonstration of a foregone conclusion. The experiment reports what actually happens on real model outputs, whatever that is.

---

## Prior work this builds on

The RWT × N/N-N integration work ([rwt-integration-findings.md](https://github.com/thedivinememe/nn-library/blob/main/docs/rwt-integration-findings.md)) demonstrated on scripted deterministic scenarios that ν-based convergence behaves as the theory predicts:

- Hybrid caught premature convergence that Diff missed (Scenario 1, 8.6× final-ν gap between Diff stopping at ν=0.405 and Hybrid pushing through to ν=0.047)
- Nu converged most efficiently on well-behaved tasks (Scenario 2, 8 iterations vs Diff never converging and Hybrid taking 12)
- Hybrid correctly rejected stuck states rather than falsely declaring success (Scenario 3)
- ν trajectory revealed oscillation patterns invisible to diff-based stopping (Scenario 4)

Those findings establish the mechanism works in controlled conditions. They do **not** establish that the mechanism produces better outcomes on real model outputs against externally-measured quality. This experiment is the first real-model test — execution of "Next Steps" #1 and #2 from the prior findings doc.

---

## Experimental setup

**Task:** HumanEval code generation problems (164 problems, standard benchmark).

**Model:** Claude Sonnet 4.6 via API, single model, fixed temperature 0.7, fixed system prompt. Chosen over Opus 4.6 for three reasons: (a) Sonnet leaves more headroom for iterative-refinement effects to be visible (Opus often converges correctly at iteration 1-2, which compresses the signal we're trying to measure), (b) Sonnet is the practitioner default for agentic workflows, so results are directly actionable for the target audience, and (c) cost headroom provides insurance for reruns and pilots. Opus replication is a natural v2, not a requirement for v1.

**Convergence strategies (three arms):**
- *Diff* — baseline, stops when consecutive outputs match (or are within edit-distance threshold; exact threshold set during pilot).
- *Nu* — stops when ν drops below licensing threshold and stabilizes. Parameters from nn-library defaults, documented in code.
- *Hybrid* — requires both output stability (Diff condition) AND ν below threshold.

**Max iterations cap:** 12 per problem per strategy (matches scripted experiments; prevents runaway costs).

**Refinement prompt:** Fixed template across strategies, asking the model to review its previous solution against the problem spec and revise. Same prompt for all three arms to isolate the stopping rule as the only variable.

**Samples:** 3 seeds per problem per strategy = 164 × 3 × 3 = 1,476 runs total. Adjust based on pilot variance estimates — commit to "pilot-informed sample size targeting 95% CI width of ≤X on false convergence rate" rather than a fixed n.

**Batch API:** Enabled from the start. 50% cost discount, 24-hour async window compatible with experimental design.

---

## Metrics

**Primary:** Test pass rate at declared convergence (did the final code pass HumanEval's tests?).

**Secondary:**
- Iterations to declared convergence (distribution per strategy).
- *False convergence rate*: % of runs where strategy declared converged AND tests failed.
- *Quality-adjusted iterations*: iterations per problem, conditional on eventual test pass.
- Per-strategy distribution of stopping iteration for pass vs. fail cases.

**Reported with:** Bootstrapped 95% confidence intervals. Per-problem variance also reported so readers can see the noise floor.

---

## Success criteria — and falsification criteria

**The claim is supported if:**
- False convergence rate for Nu and/or Hybrid is meaningfully lower than Diff (CIs don't overlap), AND
- Test pass rate at convergence is at least as high for ν-based strategies as for Diff, AND
- The pattern is interpretable — not just a statistical blip.

**The claim is falsified if:**
- False convergence rates are statistically indistinguishable across strategies, OR
- ν-based strategies show lower pass rates than Diff (signal is anti-predictive), OR
- Results are dominated by variance such that no strategy is distinguishable.

**Either outcome is publishable.** A "ν-based convergence did not outperform diff-only on HumanEval, here's why and what this means for the framework" post is genuinely valuable — to me, to the framework, and to other people considering similar approaches. The goal is honest measurement, not vindication.

---

## What this experiment does NOT claim

- Does not test output quality beyond test correctness (no code clarity, idiomaticity, or approach assessment — leaving LLM-as-judge evaluation for future work).
- Does not test on tasks beyond HumanEval (no MBPP, SWE-bench, or open-ended generation).
- Does not test across multiple models (single-model study — generalization is an open question).
- Does not validate N/N-N as a general framework — it tests one specific application of ν as a stopping signal.
- Does not claim N/N-N is superior to other uncertainty quantification methods (semantic entropy, verbalized confidence, etc.) — those are separate comparisons for future work.

---

## Milestones and rough timeline

| Weeks | Milestone | Deliverable |
|---|---|---|
| 1-2 | Literature scan on self-refinement and convergence (Self-Refine by Madaan et al., self-consistency, Huang et al. on self-correction limits, recent agentic convergence work) | One-paragraph positioning statement |
| 3-4 | Experimental harness — HumanEval loader, three convergence strategy implementations wired to existing nn-library code, Claude API integration, logging | Working harness + unit tests |
| 5-6 | Pilot run on ~10 problems per strategy. Catch bugs, calibrate thresholds, estimate required sample size | Pilot report + finalized sample size |
| 7-10 | Full experiment runs. Budget for reruns if bugs surface | Complete dataset |
| 11-13 | Analysis, writeup, code cleanup for public release | Draft blog post + clean repo |
| 14-16 | Blog post polish, final review, publish | Published artifact |

Cushion built in because life with a job, family, EM application, and MythReal will interrupt this. If a milestone slips a week, the plan survives. If it slips more than two weeks, the response is cutting scope (fewer samples, fewer secondary metrics), not extending the timeline.

---

## Known risks

- **API cost.** ~$270-400 on Sonnet 4.6 at standard rates, ~$135-200 with Batch API (50% discount). Budget for 1.5-2x that in case pilot reveals more iterations per problem or necessitates reruns. If pilot estimates come in higher than ~$500, reduce sample size before reducing scope elsewhere.
- **Threshold tuning for Diff baseline.** Exact output matching is brittle; edit-distance threshold needs to be set carefully or Diff will look artificially worse. Use the most charitable Diff configuration in the pilot so any measured advantage for ν-based strategies is honest.
- **Self-refinement may hurt or no-op.** Huang et al. and others have shown self-correction often fails to improve or actively hurts. If this is the dominant effect, all three strategies may converge to similar (poor) results. Report this honestly if it happens.
- **HumanEval saturation.** Modern frontier models score very high on HumanEval, which can compress the signal. If pilot reveals pass rates >95% across all strategies, the experiment loses discriminative power — consider shifting to MBPP or HumanEval+ (extended test cases) as a mid-course correction.
- **Scope creep pressure.** The pull toward adding MBPP, adding a second model, adding LLM-as-judge will be strong once early results come in. Resist. Ship the narrow version first.

---

## What success looks like (the artifact)

A blog post on a personal site or GitHub Pages, ~2-3K words, with:

- Clear statement of claim and findings (whichever direction they went).
- One or two key plots (false convergence rate by strategy; iterations-to-convergence distributions).
- Honest discussion of limitations and what the results do and don't mean.
- Linked public repo with reproducible code, including a `python run_experiment.py --strategy hybrid --seed 42` style entry point and the full results data.
- A short future-work section naming the obvious extensions (other benchmarks, LLM-as-judge, cross-model comparison, benchmarking against semantic entropy).

---

## Framing note for the eventual writeup

The scope doc is honest; the blog post can pick its framing once results are in. The artifact does not need to mention N/N-N as a foundational framework. It can treat ν as a technical signal, full stop. The framework lives in the linked nn-library repo for readers who want to go deep; it is not load-bearing for this artifact's credibility.

This is the track where the contribution stands on its technical merits. If ν works as a convergence signal, the framework gets credibility from the result. If it doesn't, the result is still useful and the framework can be refined or scoped more narrowly.

---

## Out of scope for this doc, noted for future work

- MBPP / SWE-bench-lite / HumanEval+ replication
- Opus 4.6 replication (does the effect strengthen or diminish at higher model capability?)
- LLM-as-judge evaluation of output quality beyond test correctness
- Benchmarking against semantic entropy (Farquhar et al. 2024), verbalized confidence (Kadavath et al.), and other uncertainty quantification methods
- Testing on non-code tasks (math reasoning, open-ended generation)
- Production deployment study (does ν-based convergence save real compute in deployed agentic systems?)

Each of these is a natural follow-up artifact. None of them belong in v1.