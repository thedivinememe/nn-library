"""Tests for convergence strategies against scripted scenarios."""

from __future__ import annotations

import pytest

from rwt_integration.convergence import (
    DiffConvergence,
    HybridConvergence,
    IterationSnapshot,
    NuConvergence,
)


def _snap(iteration: int, output: str, nu: float, nu_raw: float = 0.0,
          conflict: float = 0.0, def_sem: float = 0.5) -> IterationSnapshot:
    return IterationSnapshot(
        iteration=iteration, output=output, nu=nu, nu_raw=nu_raw or nu,
        conflict=conflict, def_sem=def_sem,
    )


# ---------- DiffConvergence ----------

class TestDiffConvergence:
    def test_stops_on_stable_output(self) -> None:
        strategy = DiffConvergence(similarity_threshold=0.95, stable_count=2)
        history = [
            _snap(0, "output A", nu=0.8),
            _snap(1, "output B", nu=0.7),
            _snap(2, "output B", nu=0.6),  # same as 1
            _snap(3, "output B", nu=0.5),  # same as 1 and 2
        ]
        result = strategy.should_stop(history)
        assert result.converged is True
        assert result.reason == "output_stable"

    def test_does_not_stop_when_changing(self) -> None:
        strategy = DiffConvergence(similarity_threshold=0.95, stable_count=2)
        history = [
            _snap(0, "output A", nu=0.8),
            _snap(1, "output B", nu=0.7),
            _snap(2, "output C", nu=0.6),
        ]
        result = strategy.should_stop(history)
        assert result.converged is False

    def test_ignores_nu_completely(self) -> None:
        """Diff strategy stops on stable output even when ν is high."""
        strategy = DiffConvergence(similarity_threshold=0.95, stable_count=2)
        history = [
            _snap(0, "bad output", nu=0.9),
            _snap(1, "bad output", nu=0.9),
            _snap(2, "bad output", nu=0.9),
        ]
        result = strategy.should_stop(history)
        assert result.converged is True  # stops despite high ν

    def test_insufficient_history(self) -> None:
        strategy = DiffConvergence()
        result = strategy.should_stop([_snap(0, "x", 0.5)])
        assert result.converged is False


# ---------- NuConvergence ----------

class TestNuConvergence:
    def test_stops_when_licensed_and_stable(self) -> None:
        strategy = NuConvergence(
            nu_threshold=0.4, nu_raw_threshold=0.5,
            stable_epsilon=0.01, stable_count=2,
        )
        history = [
            _snap(0, "v1", nu=0.8, nu_raw=0.8),
            _snap(1, "v2", nu=0.5, nu_raw=0.5),
            _snap(2, "v3", nu=0.35, nu_raw=0.35),
            _snap(3, "v4", nu=0.35, nu_raw=0.35),
            _snap(4, "v5", nu=0.349, nu_raw=0.349),
        ]
        result = strategy.should_stop(history)
        assert result.converged is True

    def test_continues_when_nu_high_even_if_stable(self) -> None:
        """This is the key: NuConvergence does NOT stop when ν is high."""
        strategy = NuConvergence(nu_threshold=0.4, stable_count=2)
        history = [
            _snap(0, "same", nu=0.6, nu_raw=0.6),
            _snap(1, "same", nu=0.6, nu_raw=0.6),
            _snap(2, "same", nu=0.6, nu_raw=0.6),
        ]
        result = strategy.should_stop(history)
        assert result.converged is False

    def test_detects_oscillation(self) -> None:
        strategy = NuConvergence(oscillation_limit=3)
        history = [
            _snap(0, "v1", nu=0.7),
            _snap(1, "v2", nu=0.5),
            _snap(2, "v3", nu=0.7),
            _snap(3, "v4", nu=0.5),
            _snap(4, "v5", nu=0.7),
        ]
        result = strategy.should_stop(history)
        assert result.converged is False
        assert result.diagnostics["oscillation_count"] >= 3

    def test_detects_stalling(self) -> None:
        strategy = NuConvergence(
            nu_threshold=0.4, stable_epsilon=0.01, stable_count=2,
        )
        history = [
            _snap(0, "v1", nu=0.6, nu_raw=0.6),
            _snap(1, "v2", nu=0.6, nu_raw=0.6),
            _snap(2, "v3", nu=0.6, nu_raw=0.6),
        ]
        result = strategy.should_stop(history)
        assert result.converged is False
        assert result.diagnostics.get("is_stalled") is True

    def test_suggests_split_on_conflict_stall(self) -> None:
        strategy = NuConvergence(
            nu_threshold=0.4, stable_epsilon=0.01, stable_count=2,
        )
        history = [
            _snap(0, "v1", nu=0.6, nu_raw=0.6, conflict=0.5),
            _snap(1, "v2", nu=0.6, nu_raw=0.6, conflict=0.5),
            _snap(2, "v3", nu=0.6, nu_raw=0.6, conflict=0.5),
        ]
        result = strategy.should_stop(history)
        assert "Split" in result.diagnostics.get("suggestion", "")


# ---------- HybridConvergence ----------

class TestHybridConvergence:
    def test_requires_both_stable_and_licensed(self) -> None:
        strategy = HybridConvergence(
            nu_threshold=0.4, nu_raw_threshold=0.5,
            similarity_threshold=0.95, diff_stable_count=2, nu_stable_count=2,
        )
        # Output stable AND ν licensed
        history = [
            _snap(0, "final output", nu=0.35, nu_raw=0.35),
            _snap(1, "final output", nu=0.35, nu_raw=0.35),
            _snap(2, "final output", nu=0.35, nu_raw=0.35),
        ]
        result = strategy.should_stop(history)
        assert result.converged is True

    def test_does_not_stop_when_only_output_stable(self) -> None:
        """Hybrid catches premature convergence that Diff misses."""
        strategy = HybridConvergence(
            nu_threshold=0.4, similarity_threshold=0.95, diff_stable_count=2,
        )
        history = [
            _snap(0, "bad output", nu=0.7, nu_raw=0.7),
            _snap(1, "bad output", nu=0.7, nu_raw=0.7),
            _snap(2, "bad output", nu=0.7, nu_raw=0.7),
        ]
        result = strategy.should_stop(history)
        assert result.converged is False
        assert result.diagnostics.get("premature_convergence") is True

    def test_does_not_stop_when_only_nu_good(self) -> None:
        strategy = HybridConvergence(
            nu_threshold=0.4, similarity_threshold=0.95,
            diff_stable_count=2, nu_stable_count=2,
        )
        # ν is good but output keeps changing
        history = [
            _snap(0, "output A", nu=0.35, nu_raw=0.35),
            _snap(1, "output B", nu=0.34, nu_raw=0.34),
            _snap(2, "output C", nu=0.33, nu_raw=0.33),
        ]
        result = strategy.should_stop(history)
        assert result.converged is False

    def test_detects_premature_convergence(self) -> None:
        strategy = HybridConvergence(
            nu_threshold=0.4, similarity_threshold=0.95, diff_stable_count=2,
        )
        history = [
            _snap(0, "mediocre", nu=0.6, nu_raw=0.6),
            _snap(1, "mediocre", nu=0.6, nu_raw=0.6),
            _snap(2, "mediocre", nu=0.6, nu_raw=0.6),
        ]
        result = strategy.should_stop(history)
        assert result.diagnostics.get("premature_convergence") is True
        assert result.reason == "premature_convergence_detected"

    def test_detects_spinning(self) -> None:
        strategy = HybridConvergence(
            nu_threshold=0.4, stable_epsilon=0.01,
        )
        # Output keeps changing but ν never improves (stays flat or worsens)
        history = [
            _snap(0, "version A", nu=0.6, nu_raw=0.6),
            _snap(1, "version B", nu=0.6, nu_raw=0.6),
            _snap(2, "version C", nu=0.61, nu_raw=0.61),
            _snap(3, "version D", nu=0.62, nu_raw=0.62),
        ]
        result = strategy.should_stop(history)
        assert result.diagnostics.get("is_spinning") is True
