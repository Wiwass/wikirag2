"""Verify tests: coverage judge (R2 repair loop) and the consolidation trigger.

Offline by construction: a canned FakeLLMClient for the judge, a temp KB
under WIKIRAG_HOME for the store side. No network, no models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest

from wikirag.config import KnowledgeBase, create_kb
from wikirag.llm.client import AllKeysExhausted, FakeLLMClient, LLMResult
from wikirag.retrieval.assembly import Context, ContextEntry
from wikirag.retrieval.verify import (
    COVERAGE_RESPONSE_SCHEMA,
    COVERAGE_SYSTEM_PROMPT,
    DEFAULT_MAX_ROUNDS,
    MAX_REPAIR_QUERIES,
    Coverage,
    _validate_coverage,
    check_coverage,
    trigger_consolidation,
)


def entry(ref: int, title: str, text: str) -> ContextEntry:
    return ContextEntry(
        ref=ref,
        owner_kind="chunk",
        owner_id=f"doc1#{ref}",
        doc_id="doc1",
        section_id="doc1:s1",
        section_title="Vis",
        doc_title=title,
        chunk_ids=[f"doc1#{ref}"],
        score=1.0,
        text=text,
    )


def context() -> Context:
    entries = [
        entry(1, "Grimoire", "Vis is raw magical energy."),
        entry(2, "Covenants", "Aura strength sets the vis yield."),
    ]
    text = "\n\n".join(f"[{e.ref}] {e.doc_title} — {e.section_title}\n{e.text}" for e in entries)
    return Context(text=text, entries=entries, token_count=40)


def coverage_json(
    ok: bool, missing: list[Any] | None = None, queries: list[Any] | None = None
) -> str:
    return json.dumps(
        {"ok": ok, "missing_aspects": missing or [], "repair_queries": queries or []}
    )


# --------------------------------------------------------------- the judge


def test_coverage_ok_path_returns_no_repair_queries() -> None:
    client = FakeLLMClient(LLMResult(text=coverage_json(True)))
    result = check_coverage(client, "What is vis?", context())

    assert result == Coverage(ok=True, missing_aspects=[], repair_queries=[])
    assert result.fallback is False
    assert len(client.calls) == 1


def test_coverage_not_ok_returns_missing_aspects_and_queries() -> None:
    client = FakeLLMClient(
        LLMResult(
            text=coverage_json(
                False,
                ["no rules for extracting vis"],
                ["vis extraction ritual", "vis source seasons"],
            )
        )
    )
    result = check_coverage(client, "How is vis extracted?", context())

    assert result.ok is False
    assert result.missing_aspects == ["no rules for extracting vis"]
    assert result.repair_queries == ["vis extraction ritual", "vis source seasons"]
    assert result.fallback is False


def test_repair_queries_are_capped() -> None:
    client = FakeLLMClient(
        LLMResult(text=coverage_json(False, ["gap"], ["one", "two", "three", "four"]))
    )
    result = check_coverage(client, "q", context())

    assert result.repair_queries == ["one", "two"]
    assert len(result.repair_queries) == MAX_REPAIR_QUERIES


def test_ok_true_forces_empty_repair_queries() -> None:
    client = FakeLLMClient(LLMResult(text=coverage_json(True, [], ["pointless extra round"])))
    assert check_coverage(client, "q", context()).repair_queries == []


def test_validate_coverage_normalizes_a_sloppy_payload() -> None:
    result = _validate_coverage(
        {
            "ok": 0,  # falsy, not a bool
            "missing_aspects": ["  spacing  ", "", 7, None, "second gap"],
            "repair_queries": ["  first  ", "", None, "second", "third"],
        }
    )

    assert result.ok is False
    assert result.missing_aspects == ["spacing", "second gap"]
    assert result.repair_queries == ["first", "second"]
    assert result.fallback is False


def test_validate_coverage_survives_missing_and_wrong_typed_fields() -> None:
    assert _validate_coverage({}) == Coverage(ok=False)
    assert _validate_coverage({"ok": "yes", "missing_aspects": "a string"}) == Coverage(ok=True)


def test_coverage_retries_malformed_json_then_succeeds() -> None:
    client = FakeLLMClient(
        LLMResult(text="not json at all"),
        LLMResult(text=coverage_json(False, ["gap"], ["repair me"])),
    )
    result = check_coverage(client, "q", context())

    assert len(client.calls) == 2
    assert result.repair_queries == ["repair me"]
    assert result.fallback is False


def test_coverage_strips_markdown_fences() -> None:
    client = FakeLLMClient(LLMResult(text=f"```json\n{coverage_json(True)}\n```"))
    assert check_coverage(client, "q", context()).ok is True


def test_coverage_falls_back_to_ok_after_max_attempts() -> None:
    client = FakeLLMClient(LLMResult(text="still not json"))
    result = check_coverage(client, "q", context(), max_attempts=2)

    assert len(client.calls) == 2
    assert result == Coverage(ok=True, missing_aspects=[], repair_queries=[], fallback=True)


def test_coverage_falls_back_on_a_non_object_payload() -> None:
    client = FakeLLMClient(LLMResult(text="[1, 2, 3]"))
    result = check_coverage(client, "q", context())

    assert len(client.calls) == 2
    assert result.ok is True
    assert result.fallback is True


def test_coverage_degrades_on_a_provider_error() -> None:
    class BoomClient:
        model = "boom"

        def generate(self, **kwargs: Any) -> LLMResult:
            raise RuntimeError("500 from provider")

    result = check_coverage(BoomClient(), "q", context())

    assert result.ok is True
    assert result.fallback is True
    assert result.repair_queries == []


def test_coverage_propagates_exhausted_keys() -> None:
    class ExhaustedClient:
        model = "spent"

        def generate(self, **kwargs: Any) -> LLMResult:
            raise AllKeysExhausted("quota")

    with pytest.raises(AllKeysExhausted):
        check_coverage(ExhaustedClient(), "q", context())


def test_coverage_call_uses_the_json_schema_and_shows_numbered_entries() -> None:
    ctx = context()
    client = FakeLLMClient(LLMResult(text=coverage_json(True)))
    check_coverage(client, "What is vis?", ctx)

    call = client.calls[0]
    assert call["json_schema"] == COVERAGE_RESPONSE_SCHEMA
    assert call["system"] == COVERAGE_SYSTEM_PROMPT
    assert "What is vis?" in call["user"]
    assert ctx.text in call["user"]
    for ref in ("[1]", "[2]"):
        assert ref in call["user"]


def test_coverage_prompt_never_asks_for_verbatim_echo() -> None:
    client = FakeLLMClient(LLMResult(text=coverage_json(True)))
    check_coverage(client, "q", context())
    prompt = client.calls[0]["system"].lower()

    for forbidden in ("verbatim", "excerpt", "transcribe", "word for word", "copy the text"):
        assert forbidden not in prompt
    assert "do not copy or quote text" in prompt
    assert "by their number only" in prompt


def test_default_max_rounds_is_a_single_repair_round() -> None:
    assert DEFAULT_MAX_ROUNDS == 1


def test_empty_context_still_produces_a_prompt() -> None:
    client = FakeLLMClient(LLMResult(text=coverage_json(False, ["nothing retrieved"], ["vis"])))
    result = check_coverage(client, "q", Context(text="", entries=[], token_count=0))

    assert "(no evidence was retrieved)" in client.calls[0]["user"]
    assert result.repair_queries == ["vis"]


# ------------------------------------------------- the consolidation trigger


@pytest.fixture()
def kb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[KnowledgeBase]:
    monkeypatch.setenv("WIKIRAG_HOME", str(tmp_path))
    handle = create_kb("test")
    handle.store.add_document(doc_id="doc1", filename="doc1.pdf", title="Grimoire")
    handle.store.add_chunks(
        [{"id": "doc1#0", "doc_id": "doc1", "section_id": None, "ordinal": 0, "text": "vis"}]
    )
    yield handle
    handle.close()


def queued_ids(kb: KnowledgeBase) -> list[int]:
    return [row["keyword_id"] for row in kb.store.consolidation_demand()]


def test_trigger_queues_only_keywords_without_a_fresh_note(kb: KnowledgeBase) -> None:
    store = kb.store
    fresh = store.upsert_keyword("Vis")
    stale = store.upsert_keyword("Aura")
    bare = store.upsert_keyword("Certamen")
    store.upsert_note(kind="keyword", keyword_id=fresh, text="vis note", sources=["doc1#0"])
    stale_note = store.upsert_note(
        kind="keyword", keyword_id=stale, text="aura note", sources=["doc1#0"]
    )
    store.set_note_stale(stale_note)

    assert trigger_consolidation(kb, [fresh, stale, bare]) == 2
    assert queued_ids(kb) == sorted([stale, bare])
    assert {row["reason"] for row in store.consolidation_demand()} == {"query"}


def test_trigger_is_idempotent(kb: KnowledgeBase) -> None:
    keyword_id = kb.store.upsert_keyword("Vis")

    trigger_consolidation(kb, [keyword_id])
    trigger_consolidation(kb, [keyword_id])

    assert queued_ids(kb) == [keyword_id]


def test_trigger_deduplicates_ids_within_one_call(kb: KnowledgeBase) -> None:
    keyword_id = kb.store.upsert_keyword("Vis")

    assert trigger_consolidation(kb, [keyword_id, keyword_id, keyword_id]) == 1
    assert queued_ids(kb) == [keyword_id]


def test_trigger_records_the_given_reason(kb: KnowledgeBase) -> None:
    keyword_id = kb.store.upsert_keyword("Vis")
    trigger_consolidation(kb, [keyword_id], reason="vault")

    demand = kb.store.consolidation_demand()
    assert [(row["keyword_id"], row["reason"]) for row in demand] == [(keyword_id, "vault")]


def test_trigger_returns_zero_on_an_s0_kb_with_no_keywords(kb: KnowledgeBase) -> None:
    hits = kb.store.keywords_for_chunk("doc1#0")

    assert kb.store.list_keywords() == []
    assert hits == []
    assert trigger_consolidation(kb, [k["id"] for k in hits]) == 0
    assert kb.store.consolidation_demand() == []
