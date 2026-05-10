"""Real Claude provider implementing the ModelProvider protocol.

Drop this in `rwt_integration/claude_provider.py`. It implements:
  - generate(prompt) -> str  : calls the Anthropic API
  - assess(task, output, prompt) -> SelfAssessment
        : calls Claude with the assessment prompt, parses JSON response

Designed to be threadsafe so EvaluationHarness can be parallelized at the
(task, strategy, seed) level by an outer driver.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from anthropic import Anthropic, APIStatusError, RateLimitError

from rwt_integration.providers import SelfAssessment


@dataclass
class ClaudeProvider:
    """ModelProvider backed by the Anthropic API.

    Note: stochastic. Repeated calls with the same prompt return different
    outputs at temperature > 0. For paired-comparison experiments, run the
    same (task, strategy, seed) configuration once and rely on ν, output,
    and assessment determinism within that single trajectory.
    """

    model: str = "claude-sonnet-4-6"
    max_tokens: int = 2048
    temperature: float = 0.7
    system: Optional[str] = None
    api_key: Optional[str] = None  # falls back to ANTHROPIC_API_KEY env

    # Retry config for rate limits and transient errors
    max_retries: int = 6
    base_backoff: float = 4.0

    _client: Anthropic = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # max_retries=0 — we handle retries ourselves to control backoff
        self._client = Anthropic(api_key=self.api_key, max_retries=0)

    # ------------------------------------------------------------------
    # Internal: API call with backoff
    # ------------------------------------------------------------------

    def _call(self, prompt: str) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.system:
            kwargs["system"] = self.system

        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.messages.create(**kwargs)
                return response.content[0].text
            except RateLimitError as e:
                last = e
                time.sleep(self.base_backoff * (2 ** attempt))
            except APIStatusError as e:
                last = e
                if e.status_code in (500, 502, 503, 504, 529):
                    time.sleep(self.base_backoff * (2 ** attempt))
                else:
                    raise
        assert last is not None
        raise last

    # ------------------------------------------------------------------
    # ModelProvider protocol
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        return self._call(prompt)

    def assess(self, task: str, output: str, prompt: str) -> SelfAssessment:
        """Call the model with the assessment prompt, parse JSON to SelfAssessment.

        The `prompt` argument is the fully-rendered assessment prompt produced
        by LoopConfig.assessment_prompt_template; we pass it through.
        """
        raw = self._call(prompt)
        return _parse_self_assessment(raw)


# ----------------------------------------------------------------------
# JSON parsing for self-assessment responses
# ----------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def _extract_json_blob(text: str) -> Optional[str]:
    """Extract a JSON object string from a model response.

    Tries: fenced ```json block, then fenced ``` block, then the largest
    {...} substring. Returns None if nothing parseable is found.
    """
    if not text:
        return None
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # Fall back to the first balanced-looking {...} we can find
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def _f01(value, default: float = 0.5) -> float:
    """Coerce a value to a float in [0, 1]."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN
        return default
    return max(0.0, min(1.0, v))


def _str_tuple(value) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value if v is not None)
    return ()


def _parse_self_assessment(raw: str) -> SelfAssessment:
    """Parse a model response into a SelfAssessment.

    On parse failure, returns a maximally-uncertain default so the loop can
    continue without crashing. The metacognition bridge will see the
    high-uncertainty signal and act accordingly.
    """
    blob = _extract_json_blob(raw)
    if blob is None:
        return SelfAssessment(
            definition_confidence=0.3,
            ambiguity_flags=("assessment_parse_failed",),
            evidence_confidence=0.3,
            unsupported_claims=("assessment_parse_failed",),
            task_coverage=0.3,
            missing_elements=("assessment_parse_failed",),
            refinement_priority="reassess",
            refinement_suggestion="Re-issue the self-assessment in valid JSON.",
        )
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return SelfAssessment(
            definition_confidence=0.3,
            ambiguity_flags=("assessment_json_invalid",),
            evidence_confidence=0.3,
            unsupported_claims=("assessment_json_invalid",),
            task_coverage=0.3,
            missing_elements=("assessment_json_invalid",),
            refinement_priority="reassess",
            refinement_suggestion="Re-issue the self-assessment in valid JSON.",
        )

    return SelfAssessment(
        definition_confidence=_f01(data.get("definition_confidence")),
        ambiguity_flags=_str_tuple(data.get("ambiguity_flags")),
        evidence_confidence=_f01(data.get("evidence_confidence")),
        unsupported_claims=_str_tuple(data.get("unsupported_claims")),
        contradictions=_str_tuple(data.get("contradictions")),
        task_coverage=_f01(data.get("task_coverage")),
        missing_elements=_str_tuple(data.get("missing_elements")),
        refinement_priority=str(data.get("refinement_priority") or ""),
        refinement_suggestion=str(data.get("refinement_suggestion") or ""),
    )
