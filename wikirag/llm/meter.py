"""Cost metering: every LLM call recorded, every method comparison grounded.

Wraps any LLMClient and records (kb, stage, role, model, tokens, cost,
wall time, timestamp) into the KB's llm_calls table. This is core
infrastructure for the ingestion research program (docs/ingestion.md):
candidate methods are compared on measured build cost, not estimates.

Local models meter too (cost 0, wall time still recorded).

Failed calls are recorded as well, with zero tokens and zero cost but the
measured wall time, and the exception then propagates unchanged. Provider
hardening (retries, key rotation) lives inside the wrapped client, so a
failure's wall time is otherwise invisible — and it is exactly the number
that makes a flaky-but-cheap model look expensive.

Planned API
-----------
- class LLMCallStore(Protocol): the two store methods metering needs
- class MeteredClient(LLMClient): wraps a client with (store, stage, role)
- summarize(store, group_by=...) -> cost report rows for `wikirag costs`
"""

from time import perf_counter
from typing import Any, Protocol, Sequence

from wikirag.llm.client import LLMClient, LLMResult

GROUP_FIELDS = ("stage", "role", "model")
"""Row fields summarize() may group by; everything else is aggregated."""


class LLMCallStore(Protocol):
    """The slice of the KB store that metering depends on."""

    def record_llm_call(
        self,
        *,
        stage: str,
        role: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        cost: float,
        wall_ms: int,
    ) -> None: ...

    def get_llm_calls(self) -> list[dict]: ...


class MeteredClient:
    """An LLMClient that records every call it delegates.

    stage/role label the call site (an ingestion stage and the config role
    whose model is being used); the model name is read from the wrapped
    client's `model` attribute unless given explicitly.
    """

    def __init__(
        self,
        inner: LLMClient,
        *,
        store: LLMCallStore,
        stage: str,
        role: str,
        model: str | None = None,
    ) -> None:
        self.inner = inner
        self.store = store
        self.stage = stage
        self.role = role
        self.model = model if model is not None else str(getattr(inner, "model", "unknown"))

    def generate(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResult:
        started = perf_counter()
        try:
            result = self.inner.generate(
                system=system,
                user=user,
                json_schema=json_schema,
                max_output_tokens=max_output_tokens,
            )
        except Exception:
            self._record(0, 0, 0.0, self._elapsed_ms(started))
            raise
        self._record(
            result.usage.prompt_tokens,
            result.usage.output_tokens,
            result.cost_estimate,
            self._elapsed_ms(started),
        )
        return result

    def _record(self, tokens_in: int, tokens_out: int, cost: float, wall_ms: int) -> None:
        self.store.record_llm_call(
            stage=self.stage,
            role=self.role,
            model=self.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            wall_ms=wall_ms,
        )

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return max(0, round((perf_counter() - started) * 1000))


def summarize(
    store: LLMCallStore,
    *,
    group_by: Sequence[str] = ("stage", "role", "model"),
) -> list[dict[str, Any]]:
    """Aggregate recorded calls into cost report rows for `wikirag costs`.

    group_by is any subset of GROUP_FIELDS; an empty one yields a single
    total row. Each row carries the grouping fields plus calls, tokens_in,
    tokens_out, cost and wall_ms. Rows are sorted by their grouping values
    so reports are stable across runs.
    """
    fields = tuple(group_by)
    unknown = [f for f in fields if f not in GROUP_FIELDS]
    if unknown:
        raise ValueError(f"cannot group by {unknown}; allowed fields: {list(GROUP_FIELDS)}")
    if len(set(fields)) != len(fields):
        raise ValueError(f"duplicate fields in group_by: {list(fields)}")

    rows: dict[tuple[str, ...], dict[str, Any]] = {}
    for call in store.get_llm_calls():
        key = tuple(str(call[f]) for f in fields)
        row = rows.get(key)
        if row is None:
            row = dict(zip(fields, key))
            row.update(calls=0, tokens_in=0, tokens_out=0, cost=0.0, wall_ms=0)
            rows[key] = row
        row["calls"] += 1
        row["tokens_in"] += call["tokens_in"]
        row["tokens_out"] += call["tokens_out"]
        row["cost"] += call["cost"]
        row["wall_ms"] += call["wall_ms"]
    return [rows[key] for key in sorted(rows)]
