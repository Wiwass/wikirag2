"""Tests for the two Wave 4 modules with no dedicated suite: enrich, atlas.

Offline throughout: FakeLLMClient, FakeEmbedder, a temp KB. The properties
under test are the ones that cost v1 dearly — a tagger that invents
keywords outside its shortlist, and a synthesis pass that duplicates its
own output on every re-run.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from wikirag.config import WIKIRAG_HOME_ENV, KnowledgeBase, create_kb
from wikirag.llm.client import AllKeysExhausted, FakeLLMClient, LLMResult
from wikirag.llm.embeddings import FakeEmbedder
from wikirag.synthesis import atlas
from wikirag.synthesis.atlas import OVERVIEW_KIND, build_overview, curate_links
from wikirag.synthesis.notes import NOTE_KIND
from wikirag.synthesis.summaries import SUMMARY_KIND
from wikirag import ingestion
from wikirag.ingestion import enrich


@pytest.fixture()
def kb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[KnowledgeBase]:
    monkeypatch.setenv(WIKIRAG_HOME_ENV, str(tmp_path / "kbs"))
    with create_kb("w4") as handle:
        handle.store.add_document(doc_id="d1", filename="d1.pdf", title="Dragons")
        handle.store.add_sections(
            "d1",
            [{"id": "d1#1", "doc_id": "d1", "parent_id": None, "ordinal": 0,
              "title": "Hoards", "level": 1, "page_start": 1, "page_end": 9}],
        )
        handle.store.add_chunks(
            [{"id": f"d1#1#{i}", "doc_id": "d1", "section_id": "d1#1", "ordinal": i,
              "text": f"Dragons guard hoards of silver, passage {i}.", "token_count": 10}
             for i in range(3)]
        )
        yield handle


def tag_reply(*per_chunk: dict) -> FakeLLMClient:
    return FakeLLMClient(LLMResult(text=json.dumps({"chunks": list(per_chunk)})))


# ------------------------------------------------------------------- enrich


def test_tagging_writes_pairs_and_creates_keywords(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(enrich, "_client_factory", lambda kb: tag_reply(
        {"chunk": 1, "keywords": [], "new": ["Hoard"]},
        {"chunk": 2, "keywords": [], "new": ["Hoard"]},
        {"chunk": 3, "keywords": [], "new": ["Silver"]},
    ))
    monkeypatch.setattr(ingestion, "_embedder_factory", lambda *a, **k: FakeEmbedder())

    enrich.enrich_stage(kb, "d1")

    names = {k["name"] for k in kb.store.list_keywords()}
    assert "Hoard" in names
    # Reused across chunks rather than created twice.
    assert len([k for k in kb.store.list_keywords() if k["name"] == "Hoard"]) == 1
    tagged = kb.store.keywords_for_chunk("d1#1#0")
    assert tagged, "the first chunk must carry its tag"


def test_a_keyword_outside_the_shortlist_is_rejected(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1's runaway-filter failure: the model must not tag with anything it likes."""
    existing = kb.store.upsert_keyword("Hoard")
    kb.store.tag_chunks([("d1#1#2", existing)])

    # The model answers with a keyword that is neither shortlisted nor declared new.
    monkeypatch.setattr(enrich, "_client_factory", lambda kb: tag_reply(
        {"chunk": 1, "keywords": ["Definitely Not Shortlisted"], "new": []},
    ))
    monkeypatch.setattr(ingestion, "_embedder_factory", lambda *a, **k: FakeEmbedder())

    enrich.enrich_stage(kb, "d1")

    names = {k["name"] for k in kb.store.list_keywords()}
    assert "Definitely Not Shortlisted" not in names


def test_enrich_is_idempotent(kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(enrich, "_client_factory", lambda kb: tag_reply(
        {"chunk": 1, "keywords": [], "new": ["Hoard"]},
    ))
    monkeypatch.setattr(ingestion, "_embedder_factory", lambda *a, **k: FakeEmbedder())

    enrich.enrich_stage(kb, "d1")
    first = len(kb.store.keywords_for_chunk("d1#1#0"))
    enrich.enrich_stage(kb, "d1")

    assert len(kb.store.keywords_for_chunk("d1#1#0")) == first


def test_enrich_quota_exhaustion_propagates(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Exhausted:
        model = "spent"
        def generate(self, **kwargs: object) -> LLMResult:
            raise AllKeysExhausted("pool spent")

    monkeypatch.setattr(enrich, "_client_factory", lambda kb: Exhausted())
    monkeypatch.setattr(ingestion, "_embedder_factory", lambda *a, **k: FakeEmbedder())

    with pytest.raises(AllKeysExhausted):
        enrich.enrich_stage(kb, "d1")


def test_the_method_registry_holds_the_default(kb: KnowledgeBase) -> None:
    assert enrich.DEFAULT_METHOD in enrich.METHODS


# -------------------------------------------------------------------- atlas


def overview_reply(text: str = "This base covers dragons and their hoards.") -> FakeLLMClient:
    return FakeLLMClient(LLMResult(text=json.dumps({"text": text, "used": [1]})))


def test_overview_returns_none_without_any_summaries(kb: KnowledgeBase) -> None:
    """No L1 material means the atlas has nothing to map: degrade quietly."""
    assert build_overview(kb, client=overview_reply()) is None
    assert kb.store.get_note(kind=OVERVIEW_KIND) is None


def test_overview_is_built_and_replaced_not_duplicated(kb: KnowledgeBase) -> None:
    kb.store.upsert_note(kind=SUMMARY_KIND, doc_id="d1", text="Dragons and hoards.")

    first = build_overview(kb, client=overview_reply())
    assert first is not None
    stored = kb.store.get_note(kind=OVERVIEW_KIND)
    assert stored is not None and "dragons" in stored["text"].lower()

    second = build_overview(kb, client=overview_reply("A revised map of the base."))
    assert second is not None
    again = kb.store.get_note(kind=OVERVIEW_KIND)
    assert again is not None
    assert "revised" in again["text"].lower(), "a rebuild must replace the overview"


def test_overview_quota_exhaustion_keeps_the_previous_overview(
    kb: KnowledgeBase,
) -> None:
    """Documented contract: quota returns None; half an overview is not stored."""
    class Exhausted:
        model = "spent"
        def generate(self, **kwargs: object) -> LLMResult:
            raise AllKeysExhausted("pool spent")

    kb.store.upsert_note(kind=SUMMARY_KIND, doc_id="d1", text="Dragons and hoards.")
    build_overview(kb, client=overview_reply("The original map."))

    assert build_overview(kb, client=Exhausted()) is None

    kept = kb.store.get_note(kind=OVERVIEW_KIND)
    assert kept is not None and "original" in kept["text"].lower()


def test_curate_links_reports_without_mutating_under_dry_run(kb: KnowledgeBase) -> None:
    """Merging concepts is destructive, so the default must only report."""
    a_id = kb.store.upsert_keyword("Hoard")
    b_id = kb.store.upsert_keyword("Hoards")
    kb.store.tag_chunks([("d1#1#0", a_id), ("d1#1#1", b_id)])
    before = {k["id"]: k["name"] for k in kb.store.list_keywords()}

    report = curate_links(kb, embedder=FakeEmbedder(), dry_run=True)

    assert report.dry_run is True
    assert {k["id"]: k["name"] for k in kb.store.list_keywords()} == before
