"""Gemini implementation of LLMClient.

Ports the production-hardened parts of v1's GeminiPipeline and nothing else:
- retry with backoff; 429/RESOURCE_EXHAUSTED quota detection;
- broken-key detection (401/403 UNAUTHENTICATED / PERMISSION_DENIED);
- key pool with linear rotation and optional carousel mode
  (wait-and-restart when the whole pool is exhausted);
- truncation detection via finish reason; usage extraction.

Deliberately NOT here (v1 mixed these in): logging setup, path resolution,
markdown output cleanup — those belong to callers or small text utils.
No prompts and no domain logic: this module only turns (system, user,
json_schema) into a google-genai request and the response into an LLMResult.
Callers always wrap it in MeteredClient (docs/llm.md).

PDF/file upload support sits behind the provider capability flag; only
needed if an ingest method chooses native-PDF calls for short documents.
Not implemented yet.

API
---
- PRICES: per-model {model: (input $/1M, output $/1M)}; see estimate_cost()
- estimate_cost(model, usage) -> float
- load_api_keys(keys_dir=None) -> list[str]: pool from *.txt files or env
- class GeminiClient(model, keys=None, *, temperature, max_output_tokens,
  carousel, carousel_wait_seconds, max_carousel_rounds, max_retries_per_key,
  base_delay): implements LLMClient; exposes `model` for MeteredClient
- AllKeysExhausted (defined in llm/client.py) when the whole pool is spent
"""

import os
import re
import time
from pathlib import Path
from typing import Any, Sequence

from google.genai import types

from wikirag.llm.client import AllKeysExhausted, LLMResult, Usage

PRICES: dict[str, tuple[float, float]] = {
    # (input, output) US$ per 1M tokens, paid tier, standard context.
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    # 3.5 family (verified 2026-08-31; 2.5 is retired for new API users).
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.5-flash-lite": (0.30, 2.50),
}
"""Price table for cost_estimate. A model absent from it costs 0.0 (see
estimate_cost): an unknown or local model must never fake a cost, and a
zero in a cost report is a visible signal that the table needs an entry."""

DEFAULT_KEYS_DIR = Path("private")
KEYS_DIR_ENV = "WIKIRAG_KEYS_DIR"
SINGLE_KEY_ENV = "GEMINI_API_KEY"


def estimate_cost(model: str, usage: Usage) -> float:
    """US$ estimate for one call, from PRICES; 0.0 for an unlisted model.

    Matching is exact first, then the longest PRICES key the model name
    starts with, so dated/preview aliases (gemini-2.5-flash-preview-09-2025)
    price as their family. Cached tokens are counted at the full input rate:
    Gemini bills them cheaper, so this over-estimates rather than under.
    """
    price = PRICES.get(model)
    if price is None:
        candidates = [k for k in PRICES if model.startswith(k)]
        if not candidates:
            return 0.0
        price = PRICES[max(candidates, key=len)]
    input_price, output_price = price
    return (usage.prompt_tokens * input_price + usage.output_tokens * output_price) / 1_000_000


def load_api_keys(keys_dir: str | Path | None = None) -> list[str]:
    """Load the API key pool, in order of precedence.

    Explicit `keys_dir`, else $WIKIRAG_KEYS_DIR, else ./private if it
    exists: every *.txt file in that directory (files sorted by name, one
    key per line, blank lines ignored) contributes its keys, so a pool is
    one file per key or one file with many. If no such directory yields
    keys, the single $GEMINI_API_KEY is used. Duplicates are dropped —
    rotating onto the same key twice would waste a round of the pool.
    Returns [] when nothing is configured; the caller decides when to fail.
    """
    directory = keys_dir or os.environ.get(KEYS_DIR_ENV)
    path = Path(directory) if directory else DEFAULT_KEYS_DIR
    keys: list[str] = []
    if path.is_dir():
        for file in sorted(path.glob("*.txt")):
            keys.extend(
                line.strip() for line in file.read_text(encoding="utf-8").splitlines() if line.strip()
            )
    if not keys:
        single = os.environ.get(SINGLE_KEY_ENV, "").strip()
        if single:
            keys = [single]
    return list(dict.fromkeys(keys))


def _make_genai_client(api_key: str) -> Any:
    """The single construction point for google-genai; tests patch this."""
    from google import genai

    return genai.Client(api_key=api_key)


def _sleep(seconds: float) -> None:
    """Indirection so tests can run backoff and carousel waits instantly."""
    time.sleep(seconds)


def _status_code(error: Exception) -> Any:
    return getattr(error, "code", getattr(error, "status_code", None))


def _is_quota_error(error: Exception) -> bool:
    text = str(error)
    return _status_code(error) == 429 or "429" in text or "RESOURCE_EXHAUSTED" in text


def _is_key_broken_error(error: Exception) -> bool:
    """The key itself (or its service account) is unusable, not rate-limited.

    Observed in v1 production: 401 UNAUTHENTICATED "The bound service account
    is deleted or disabled" — never a 429, so quota detection never rotated
    away from it. 403 PERMISSION_DENIED (API not enabled for that key's
    project) is the same class. Retrying the same key is pointless but another
    key of the pool may work, so it rotates like a 429 — and the key is marked
    broken so rotation never comes back to it. 400/404 are excluded: malformed
    request or missing resource, which no other key would fix.
    """
    text = str(error)
    return (
        _status_code(error) in (401, 403)
        or "UNAUTHENTICATED" in text
        or "PERMISSION_DENIED" in text
        or "ACCOUNT_STATE_INVALID" in text
    )


class _RotateKey(Exception):
    """Internal: this key is spent (quota or broken); try the next one."""


class GeminiClient:
    """LLMClient over google-genai with v1's key-pool hardening.

    Keys come from `keys` or, when that is None, from load_api_keys() at the
    first call — never at import, so a missing pool fails where it can be
    reported rather than at module load. Within one key: `max_retries_per_key`
    attempts with exponential backoff (a 429 waits the delay Google asks for
    when it states one). When they are all spent the client rotates to the
    next key; when the pool runs out it raises AllKeysExhausted, unless
    `carousel` is set, in which case it waits `carousel_wait_seconds` and
    restarts the pool up to `max_carousel_rounds` times — a quota that resets
    mid-run is then used instead of losing the rest of the batch. Keys marked
    broken are skipped by every later rotation, including carousel restarts.
    """

    def __init__(
        self,
        model: str,
        keys: Sequence[str] | None = None,
        *,
        temperature: float = 0.1,
        max_output_tokens: int = 8192,
        carousel: bool = False,
        carousel_wait_seconds: float = 70.0,
        max_carousel_rounds: int = 3,
        max_retries_per_key: int | None = None,
        base_delay: float = 2.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.carousel = carousel
        self.carousel_wait_seconds = carousel_wait_seconds
        self.max_carousel_rounds = max_carousel_rounds
        # In carousel mode fewer retries per key: moving on to another key is
        # more likely to pay off than insisting on an already-exhausted one.
        self.max_retries_per_key = max_retries_per_key or (3 if carousel else 5)
        self.base_delay = base_delay

        self._keys: list[str] | None = list(keys) if keys is not None else None
        self._key_index = 0
        self._broken: set[int] = set()
        self._client: Any = None

    def generate(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResult:
        keys = self._ensure_keys()
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            max_output_tokens=max_output_tokens or self.max_output_tokens,
            **(
                {"response_mime_type": "application/json", "response_schema": json_schema}
                if json_schema is not None
                else {}
            ),
        )

        rounds = 0
        while True:
            try:
                return self._generate_on_current_key(user, config)
            except _RotateKey as rotate:
                if self._rotate():
                    continue
                if self.carousel and rounds < self.max_carousel_rounds and self._restartable():
                    rounds += 1
                    _sleep(self.carousel_wait_seconds)
                    self._key_index = self._first_usable()
                    self._client = None
                    continue
                raise AllKeysExhausted(
                    f"all {len(keys)} Gemini API keys are out of quota or unusable "
                    f"({len(self._broken)} marked broken)"
                ) from rotate.__cause__

    def _generate_on_current_key(self, user: str, config: types.GenerateContentConfig) -> LLMResult:
        for attempt in range(1, self.max_retries_per_key + 1):
            try:
                response = self._genai_client().models.generate_content(
                    model=self.model, contents=user, config=config
                )
                return self._to_result(response)
            except Exception as error:
                if _is_key_broken_error(error):
                    self._broken.add(self._key_index)
                    raise _RotateKey() from error
                if _status_code(error) in (400, 404):
                    raise
                quota = _is_quota_error(error)
                if attempt == self.max_retries_per_key:
                    if quota:
                        raise _RotateKey() from error
                    raise
                _sleep(self._retry_delay(error, attempt, quota))
        raise AssertionError("unreachable: the loop returns or raises")

    def _retry_delay(self, error: Exception, attempt: int, quota: bool) -> float:
        if not quota:
            return self.base_delay * 2 ** (attempt - 1)
        # Google states the wait in the 429 body; respect it plus a margin.
        asked = re.search(r"Please retry in (\d+\.?\d*)s", str(error))
        return float(asked.group(1)) + 5.0 if asked else 30.0

    def _to_result(self, response: Any) -> LLMResult:
        candidates = getattr(response, "candidates", None) or []
        finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        finish = getattr(finish_reason, "name", None) or str(finish_reason or "UNKNOWN")

        metadata = getattr(response, "usage_metadata", None)
        usage = Usage(
            prompt_tokens=int(getattr(metadata, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(metadata, "candidates_token_count", 0) or 0),
            total_tokens=int(getattr(metadata, "total_token_count", 0) or 0),
            cached_tokens=int(getattr(metadata, "cached_content_token_count", 0) or 0),
        )
        return LLMResult(
            text=(getattr(response, "text", None) or "").strip(),
            truncated=finish == "MAX_TOKENS",
            usage=usage,
            cost_estimate=estimate_cost(self.model, usage),
        )

    def _ensure_keys(self) -> list[str]:
        if self._keys is None:
            self._keys = load_api_keys()
        if not self._keys:
            raise RuntimeError(
                f"no Gemini API keys: pass keys=..., set ${KEYS_DIR_ENV} to a directory of "
                f"*.txt key files (or create ./{DEFAULT_KEYS_DIR}), or set ${SINGLE_KEY_ENV}"
            )
        return self._keys

    def _genai_client(self) -> Any:
        if self._client is None:
            assert self._keys is not None
            self._client = _make_genai_client(self._keys[self._key_index])
        return self._client

    def _rotate(self) -> bool:
        """Move to the next usable key; False when the pool round is over."""
        assert self._keys is not None
        for index in range(self._key_index + 1, len(self._keys)):
            if index not in self._broken:
                self._key_index = index
                self._client = None
                return True
        return False

    def _restartable(self) -> bool:
        assert self._keys is not None
        return any(index not in self._broken for index in range(len(self._keys)))

    def _first_usable(self) -> int:
        assert self._keys is not None
        return next(i for i in range(len(self._keys)) if i not in self._broken)
