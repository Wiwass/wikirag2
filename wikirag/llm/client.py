"""LLMClient protocol — the provider contract (docs/llm.md).

This module is the one place where the interface is code rather than prose,
because every provider implementation and every stage depends on it.

Design notes:
- Structured output via json_schema (provider-native constrained decoding
  where available). Combined with the no-verbatim-echo rule this should
  make JSON repair dead weight; a thin parse fallback lives with the
  stages and logs every time it actually fires.
- File/PDF input is a provider capability flag, not part of the core call.
- truncated + usage are first-class: callers always see finish state and
  token counts (v1 behavior, kept).
- Providers should expose the model name they were configured with as a
  `model` attribute: MeteredClient reads it to label recorded calls.

Planned API
-----------
- class Usage: prompt/output/total/cached token counts
- class LLMResult: text, truncated, usage, cost_estimate
- class LLMClient(Protocol): generate(*, system, user, json_schema=None,
  max_output_tokens=None) -> LLMResult
- class FakeLLMClient: offline test double implementing the protocol
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


class AllKeysExhausted(Exception):
    """Every credential a provider had left is spent (quota or unusable).

    Provider-agnostic on purpose: the ingestion runner catches it to stop a
    batch cleanly — and to distinguish "come back when quota resets" from a
    real failure — without importing any provider module.
    """


@dataclass
class Usage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class LLMResult:
    text: str
    truncated: bool = False
    usage: Usage = field(default_factory=Usage)
    cost_estimate: float = 0.0  # from a per-model price table; 0 for local


class LLMClient(Protocol):
    """Minimal generation interface every provider must satisfy."""

    def generate(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResult: ...


class FakeLLMClient:
    """Deterministic offline LLMClient, shared by every module's tests.

    Returns the given results in call order; once they are exhausted the
    last one repeats, so a test that only cares about metering or wiring
    can pass a single result and call generate() any number of times.
    Every call is appended to `calls` as the keyword arguments it received.
    """

    def __init__(self, *results: LLMResult, model: str = "fake-model") -> None:
        self.results: Sequence[LLMResult] = results
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResult:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "json_schema": json_schema,
                "max_output_tokens": max_output_tokens,
            }
        )
        if not self.results:
            return LLMResult(text="")
        return self.results[min(len(self.calls) - 1, len(self.results) - 1)]
