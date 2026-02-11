"""Trace mode — RefinementRecord collection for debugging."""

from __future__ import annotations

from typing import Optional

from nn_logic.types import (
    ContextID,
    RefinementRecord,
    TargetID,
)


class Tracer:
    """Collects RefinementRecords for debugging and auditing.

    Usage:
        tracer = Tracer()
        state, record = incorporate(state, evidence, ...)
        tracer.record(record)
        # ... more operations ...
        tracer.dump()  # print all records
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._records: list[RefinementRecord] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def records(self) -> list[RefinementRecord]:
        return list(self._records)

    def record(self, rec: RefinementRecord) -> None:
        if self._enabled:
            self._records.append(rec)

    def record_all(self, recs: list[RefinementRecord]) -> None:
        if self._enabled:
            self._records.extend(recs)

    def for_target(
        self,
        target_id: TargetID,
        context_id: Optional[ContextID] = None,
    ) -> list[RefinementRecord]:
        result = [r for r in self._records if r.target_id == target_id]
        if context_id is not None:
            result = [r for r in result if r.context_id == context_id]
        return result

    def for_operator(self, operator: str) -> list[RefinementRecord]:
        return [r for r in self._records if r.operator == operator]

    def clear(self) -> None:
        self._records.clear()

    def dump(self) -> list[str]:
        """Return human-readable summary of all records."""
        lines: list[str] = []
        for i, r in enumerate(self._records):
            lines.append(
                f"[{i}] {r.operator} on {r.target_id}@{r.context_id}: "
                f"ν_raw {r.nu_raw_before:.3f}→{r.nu_raw_after:.3f}, "
                f"ν {r.nu_before:.3f}→{r.nu_after:.3f}"
            )
        return lines
