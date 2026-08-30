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
"""

from dataclasses import dataclass, field
from typing import Protocol


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
