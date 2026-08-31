"""Offline tests for the metering chokepoint (wikirag/llm/meter.py).

Fake client + fake store only: no network, no real Store import (the store
dependency is the two-method LLMCallStore protocol).
"""

from typing import Any

import pytest

from wikirag.llm.client import FakeLLMClient, LLMResult, Usage
from wikirag.llm.meter import GROUP_FIELDS, MeteredClient, summarize


class FakeStore:
    """In-memory stand-in for the llm_calls table."""

    def __init__(self, calls: list[dict[str, Any]] | None = None) -> None:
        self._calls: list[dict[str, Any]] = list(calls or [])

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
    ) -> None:
        self._calls.append(
            {
                "id": len(self._calls) + 1,
                "ts": "2026-01-01T00:00:00",
                "stage": stage,
                "role": role,
                "model": model,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost": cost,
                "wall_ms": wall_ms,
            }
        )

    def get_llm_calls(self) -> list[dict]:
        return list(self._calls)


class RaisingClient:
    model = "boom-model"

    def generate(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResult:
        raise RuntimeError("provider exploded")


def _result(prompt_tokens: int = 11, output_tokens: int = 7, cost: float = 0.25) -> LLMResult:
    return LLMResult(
        text="hello",
        truncated=True,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=prompt_tokens + output_tokens,
            cached_tokens=3,
        ),
        cost_estimate=cost,
    )


def _canned_rows() -> list[dict[str, Any]]:
    return [
        row_dict(1, "enrich", "tagger", "flash", 100, 10, 0.01, 500),
        row_dict(2, "enrich", "tagger", "flash", 200, 20, 0.02, 700),
        row_dict(3, "enrich", "tagger", "local-8b", 300, 30, 0.0, 4000),
        row_dict(4, "synth", "writer", "pro", 400, 40, 0.5, 1500),
    ]


def row_dict(
    id_: int,
    stage: str,
    role: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    cost: float,
    wall_ms: int,
) -> dict[str, Any]:
    return {
        "id": id_,
        "ts": f"2026-01-0{id_}T00:00:00",
        "stage": stage,
        "role": role,
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost": cost,
        "wall_ms": wall_ms,
    }


def test_delegates_and_returns_inner_result_untouched() -> None:
    expected = _result()
    inner = FakeLLMClient(expected, model="flash")
    client = MeteredClient(inner, store=FakeStore(), stage="enrich", role="tagger")

    schema = {"type": "object"}
    got = client.generate(system="sys", user="usr", json_schema=schema, max_output_tokens=64)

    assert got is expected
    assert inner.calls == [
        {"system": "sys", "user": "usr", "json_schema": schema, "max_output_tokens": 64}
    ]


def test_records_stage_role_model_tokens_and_cost() -> None:
    store = FakeStore()
    client = MeteredClient(
        FakeLLMClient(_result(prompt_tokens=42, output_tokens=8, cost=0.75), model="flash"),
        store=store,
        stage="enrich",
        role="tagger",
    )

    client.generate(system="sys", user="usr")

    (call,) = store.get_llm_calls()
    assert call["stage"] == "enrich"
    assert call["role"] == "tagger"
    assert call["model"] == "flash"
    assert call["tokens_in"] == 42
    assert call["tokens_out"] == 8
    assert call["cost"] == 0.75
    assert isinstance(call["wall_ms"], int) and call["wall_ms"] >= 0


def test_explicit_model_overrides_inner_attribute() -> None:
    store = FakeStore()
    client = MeteredClient(
        FakeLLMClient(_result(), model="flash"),
        store=store,
        stage="synth",
        role="writer",
        model="gemini-2.5-pro",
    )

    client.generate(system="sys", user="usr")

    assert store.get_llm_calls()[0]["model"] == "gemini-2.5-pro"


def test_model_falls_back_to_unknown_without_inner_attribute() -> None:
    class Bare:
        def generate(self, **kwargs: Any) -> LLMResult:
            return _result()

    store = FakeStore()
    MeteredClient(Bare(), store=store, stage="s", role="r").generate(system="", user="")

    assert store.get_llm_calls()[0]["model"] == "unknown"


def test_local_model_meters_with_zero_cost() -> None:
    store = FakeStore()
    free = LLMResult(text="x", usage=Usage(prompt_tokens=5, output_tokens=2), cost_estimate=0.0)
    client = MeteredClient(
        FakeLLMClient(free, model="local-8b"), store=store, stage="enrich", role="tagger"
    )

    client.generate(system="sys", user="usr")

    (call,) = store.get_llm_calls()
    assert call["cost"] == 0.0
    assert call["tokens_in"] == 5
    assert call["wall_ms"] >= 0


def test_every_call_is_recorded() -> None:
    store = FakeStore()
    client = MeteredClient(
        FakeLLMClient(_result(), model="flash"), store=store, stage="enrich", role="tagger"
    )

    for _ in range(3):
        client.generate(system="sys", user="usr")

    assert len(store.get_llm_calls()) == 3


def test_failure_propagates_and_is_recorded_with_zeros() -> None:
    store = FakeStore()
    client = MeteredClient(RaisingClient(), store=store, stage="enrich", role="tagger")

    with pytest.raises(RuntimeError, match="provider exploded"):
        client.generate(system="sys", user="usr")

    (call,) = store.get_llm_calls()
    assert call["model"] == "boom-model"
    assert (call["tokens_in"], call["tokens_out"], call["cost"]) == (0, 0, 0.0)
    assert call["wall_ms"] >= 0


def test_summarize_groups_by_stage_and_model() -> None:
    rows = summarize(FakeStore(_canned_rows()), group_by=("stage", "model"))

    assert rows == [
        {
            "stage": "enrich",
            "model": "flash",
            "calls": 2,
            "tokens_in": 300,
            "tokens_out": 30,
            "cost": pytest.approx(0.03),
            "wall_ms": 1200,
        },
        {
            "stage": "enrich",
            "model": "local-8b",
            "calls": 1,
            "tokens_in": 300,
            "tokens_out": 30,
            "cost": 0.0,
            "wall_ms": 4000,
        },
        {
            "stage": "synth",
            "model": "pro",
            "calls": 1,
            "tokens_in": 400,
            "tokens_out": 40,
            "cost": pytest.approx(0.5),
            "wall_ms": 1500,
        },
    ]


def test_summarize_groups_by_single_field() -> None:
    rows = summarize(FakeStore(_canned_rows()), group_by=("stage",))

    assert [r["stage"] for r in rows] == ["enrich", "synth"]
    assert rows[0]["calls"] == 3
    assert rows[0]["tokens_in"] == 600
    assert rows[0]["tokens_out"] == 60
    assert rows[0]["cost"] == pytest.approx(0.03)
    assert rows[0]["wall_ms"] == 5200


def test_summarize_empty_group_by_gives_one_total_row() -> None:
    (row,) = summarize(FakeStore(_canned_rows()), group_by=())

    assert row == {
        "calls": 4,
        "tokens_in": 1000,
        "tokens_out": 100,
        "cost": pytest.approx(0.53),
        "wall_ms": 6700,
    }


def test_summarize_default_grouping_is_all_fields() -> None:
    rows = summarize(FakeStore(_canned_rows()))

    assert [(r["stage"], r["role"], r["model"]) for r in rows] == [
        ("enrich", "tagger", "flash"),
        ("enrich", "tagger", "local-8b"),
        ("synth", "writer", "pro"),
    ]
    assert set(GROUP_FIELDS) <= set(rows[0])


def test_summarize_on_empty_store_returns_no_rows() -> None:
    assert summarize(FakeStore()) == []


@pytest.mark.parametrize("group_by", [("kb",), ("stage", "stage")])
def test_summarize_rejects_bad_group_by(group_by: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        summarize(FakeStore(_canned_rows()), group_by=group_by)


def test_metered_client_round_trips_through_the_store_protocol() -> None:
    """End-to-end: two stages metered, then summarized as `wikirag costs` would."""
    store = FakeStore()
    enrich = MeteredClient(
        FakeLLMClient(_result(prompt_tokens=10, output_tokens=1, cost=0.1), model="flash"),
        store=store,
        stage="enrich",
        role="tagger",
    )
    synth = MeteredClient(
        FakeLLMClient(_result(prompt_tokens=100, output_tokens=50, cost=1.0), model="pro"),
        store=store,
        stage="synth",
        role="writer",
    )

    enrich.generate(system="s", user="u")
    enrich.generate(system="s", user="u")
    synth.generate(system="s", user="u")

    rows = summarize(store, group_by=("stage",))
    assert [(r["stage"], r["calls"], r["tokens_in"]) for r in rows] == [
        ("enrich", 2, 20),
        ("synth", 1, 100),
    ]
