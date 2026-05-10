"""HumanEval convergence experiment runner.

Wires HumanEval into the existing rwt_integration infrastructure:
  - Loads HumanEval problems and converts them to Task objects
  - Uses ClaudeProvider as the real ModelProvider
  - Runs each (problem × strategy × seed) through EvaluationHarness
  - Executes the final output against HumanEval test cases
  - Saves one JSON per run, resumable

Setup:
    pip install -e .              # this repo
    pip install anthropic human-eval
    export ANTHROPIC_API_KEY=...

Run:
    python -m rwt_integration.humaneval_runner --pilot   # 5 problems × 1 seed × 3 strategies
    python -m rwt_integration.humaneval_runner            # full run

Outputs:
    runs/<task_id>__<strategy>__seed<S>.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from rwt_integration.claude_provider import ClaudeProvider
from rwt_integration.config import LoopConfig
from rwt_integration.convergence import (
    ConvergenceStrategy,
    DiffConvergence,
    HybridConvergence,
    NuConvergence,
)
from rwt_integration.evaluation import EvaluationHarness
from rwt_integration.tasks import Task

# ============================================================================
# CONFIG
# ============================================================================

MODEL              = "claude-sonnet-4-6"
TEMPERATURE        = 0.7
N_SEEDS            = 3
MAX_ITERATIONS     = 12
MAX_WORKERS        = 4         # concurrent runs (each makes ~12-24 API calls)
TEST_TIMEOUT_SEC   = 10
RUNS_DIR           = Path("runs")
PILOT_N_PROBLEMS   = 5
PILOT_N_SEEDS      = 1

# Code-generation system prompt (returned-only-the-function discipline)
CODE_SYSTEM = (
    "You are an expert Python programmer. When asked to write a function, "
    "return only valid Python code: the complete function definition with any "
    "imports it needs. Do not include test code, examples, or explanations."
)


# ============================================================================
# HUMANEVAL → TASK
# ============================================================================

def humaneval_problem_to_task(problem: dict) -> Task:
    """Convert a HumanEval problem to an rwt_integration Task.

    HumanEval problems include the function signature and docstring in
    `prompt`, the expected `entry_point`, the `test` code, and a
    `canonical_solution`. We use prompt + entry_point as the specification.
    """
    spec = problem["prompt"].rstrip()
    spec += (
        "\n\n# Implement the function above. "
        f"Function name: {problem['entry_point']}. "
        "Return only the complete function definition with any imports it needs; "
        "no test code or explanation."
    )
    return Task(
        id=problem["task_id"].replace("/", "_"),
        description=f"HumanEval problem: implement {problem['entry_point']}",
        specification=spec,
        complexity="moderate",
    )


# ============================================================================
# CODE EXTRACTION + TEST EXECUTION
# ============================================================================

_PY_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(response_text: str) -> str:
    if not response_text:
        return ""
    m = _PY_FENCE_RE.search(response_text)
    if m:
        return m.group(1).strip()
    return response_text.strip()


def run_humaneval_tests(code: str, problem: dict) -> dict:
    """Execute candidate code against HumanEval test cases in a subprocess."""
    program = (
        code
        + "\n\n"
        + problem["test"]
        + f"\ncheck({problem['entry_point']})\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            timeout=TEST_TIMEOUT_SEC,
            text=True,
        )
        stderr = proc.stderr or ""
        error_class = None
        if proc.returncode != 0:
            m = re.search(
                r"^([A-Za-z_][A-Za-z0-9_]*Error|AssertionError):",
                stderr, re.MULTILINE,
            )
            if m:
                error_class = m.group(1)
        return {
            "passed":      proc.returncode == 0,
            "exit_code":   proc.returncode,
            "stderr_tail": stderr[-800:],
            "timeout":     False,
            "error_class": error_class,
        }
    except subprocess.TimeoutExpired:
        return {
            "passed":      False,
            "exit_code":   None,
            "stderr_tail": "",
            "timeout":     True,
            "error_class": "Timeout",
        }


# ============================================================================
# RUN ORCHESTRATION
# ============================================================================

STRATEGY_FACTORIES = {
    "diff":   lambda cfg: DiffConvergence(
        similarity_threshold=cfg.diff_similarity_threshold,
        stable_count=cfg.diff_stable_count,
    ),
    "nu":     lambda cfg: NuConvergence(
        nu_threshold=cfg.nu_licensing_threshold,
        nu_raw_threshold=cfg.nu_raw_licensing_threshold,
        stable_epsilon=cfg.nu_stable_epsilon,
        stable_count=cfg.nu_stable_count,
        oscillation_limit=cfg.oscillation_limit,
    ),
    "hybrid": lambda cfg: HybridConvergence(
        similarity_threshold=cfg.diff_similarity_threshold,
        diff_stable_count=cfg.diff_stable_count,
        nu_threshold=cfg.nu_licensing_threshold,
        nu_raw_threshold=cfg.nu_raw_licensing_threshold,
        stable_epsilon=cfg.nu_stable_epsilon,
        nu_stable_count=cfg.nu_stable_count,
        oscillation_limit=cfg.oscillation_limit,
    ),
}


def run_path(task_id: str, strategy_name: str, seed: int) -> Path:
    return RUNS_DIR / f"{task_id}__{strategy_name}__seed{seed}.json"


def already_done(task_id: str, strategy_name: str, seed: int) -> bool:
    p = run_path(task_id, strategy_name, seed)
    if not p.exists():
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        return d.get("completed") is True
    except Exception:
        return False


def execute_one(
    problem: dict,
    strategy_name: str,
    seed: int,
    config: LoopConfig,
) -> dict:
    """Run one (problem × strategy × seed) trajectory and evaluate the final code."""
    task = humaneval_problem_to_task(problem)
    strategy = STRATEGY_FACTORIES[strategy_name](config)
    provider = ClaudeProvider(
        model=MODEL,
        temperature=TEMPERATURE,
        system=CODE_SYSTEM,
        max_tokens=2048,
    )
    harness = EvaluationHarness(config=config)

    try:
        result = harness.run_single(task, strategy, provider)
        summary = result.summary
        final_code = extract_code(summary.final_output)
        test_result = run_humaneval_tests(final_code, problem)
        run_record = {
            "task_id":             problem["task_id"],
            "entry_point":         problem["entry_point"],
            "strategy":            strategy_name,
            "seed":                seed,
            "model":               MODEL,
            "temperature":         TEMPERATURE,
            "completed":           True,
            "iterations":          summary.iterations,
            "converged":           summary.converged,
            "convergence_reason":  summary.convergence_reason,
            "final_nu":            summary.final_nu,
            "final_nu_raw":        summary.final_nu_raw,
            "nu_trajectory":       list(summary.nu_trajectory),
            "nu_raw_trajectory":   list(summary.nu_raw_trajectory),
            "wall_time_seconds":   summary.total_wall_time,
            "tokens_estimate":     summary.total_tokens_estimate,
            "final_output":        summary.final_output,
            "final_code":          final_code,
            "test_result":         test_result,
            # Per-iteration details — keep modest fields, full history is large
            "iteration_records": [
                {
                    "iteration":         r.iteration,
                    "nu":                r.nu,
                    "nu_raw":            r.nu_raw,
                    "nu_penalty":        r.nu_penalty,
                    "def_sem":           r.def_sem,
                    "def_ep":            r.def_ep,
                    "def_proc":          r.def_proc,
                    "conflict_score":    r.conflict_score,
                    "output_diff_ratio": r.output_diff_ratio,
                    "convergence_check": {
                        "converged":  r.convergence_check.converged,
                        "reason":     r.convergence_check.reason,
                        "confidence": r.convergence_check.confidence,
                    },
                    "wall_time_seconds": r.wall_time_seconds,
                }
                for r in summary.iteration_records
            ],
        }
    except Exception as e:
        run_record = {
            "task_id":     problem["task_id"],
            "strategy":    strategy_name,
            "seed":        seed,
            "model":       MODEL,
            "completed":   False,
            "fatal_error": f"{type(e).__name__}: {e}",
        }

    out_path = run_path(run_record.get("task_id", problem["task_id"]).replace("/", "_"),
                         strategy_name, seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(run_record, f, indent=2)
    return run_record


def load_humaneval(pilot: bool) -> list[dict]:
    try:
        from human_eval.data import read_problems
    except ImportError:
        raise SystemExit("Install human-eval: pip install human-eval")
    problems = list(read_problems().values())
    problems.sort(key=lambda p: int(p["task_id"].split("/")[1]))
    if pilot:
        problems = problems[:PILOT_N_PROBLEMS]
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true",
                        help=f"Pilot: {PILOT_N_PROBLEMS} problems, {PILOT_N_SEEDS} seed.")
    parser.add_argument("--strategies", nargs="+",
                        default=["diff", "nu", "hybrid"],
                        choices=["diff", "nu", "hybrid"])
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY in your environment.")

    problems = load_humaneval(pilot=args.pilot)
    n_seeds  = PILOT_N_SEEDS if args.pilot else N_SEEDS
    config   = LoopConfig(max_iterations=MAX_ITERATIONS)

    work = []
    for problem in problems:
        for strategy_name in args.strategies:
            for seed in range(1, n_seeds + 1):
                tid = problem["task_id"].replace("/", "_")
                if already_done(tid, strategy_name, seed):
                    continue
                work.append((problem, strategy_name, seed))

    print(f"Pending runs: {len(work)} "
          f"({len(problems)} problems × {len(args.strategies)} strategies × {n_seeds} seeds, "
          f"minus already-done) — workers={args.workers}")

    completed_count = 0
    pass_counts: dict[str, int] = {s: 0 for s in args.strategies}
    fail_counts: dict[str, int] = {s: 0 for s in args.strategies}
    err_counts:  dict[str, int] = {s: 0 for s in args.strategies}

    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(execute_one, problem, strategy_name, seed, config)
            for (problem, strategy_name, seed) in work
        ]
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except Exception as e:
                with lock:
                    completed_count += 1
                    print(f"  FATAL: {type(e).__name__}: {e}")
                continue
            s = rec.get("strategy", "?")
            with lock:
                completed_count += 1
                if not rec.get("completed"):
                    err_counts[s] = err_counts.get(s, 0) + 1
                elif rec.get("test_result", {}).get("passed"):
                    pass_counts[s] = pass_counts.get(s, 0) + 1
                else:
                    fail_counts[s] = fail_counts.get(s, 0) + 1
                if completed_count % 5 == 0 or completed_count == len(work):
                    print(f"  {completed_count}/{len(work)} done. "
                          f"pass={pass_counts} fail={fail_counts} err={err_counts}")

    print("\nFinal:")
    for s in args.strategies:
        total = pass_counts[s] + fail_counts[s] + err_counts[s]
        if total == 0:
            continue
        rate = pass_counts[s] / total
        print(f"  {s:<7} pass={pass_counts[s]} fail={fail_counts[s]} "
              f"err={err_counts[s]} (pass rate = {rate:.1%})")


if __name__ == "__main__":
    main()
