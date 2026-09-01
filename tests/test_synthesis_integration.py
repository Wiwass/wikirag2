"""Wave 4 cross-module integration: the seams where this layer fails quietly.

Each synthesis module writes notes that another, already-written module
reads by `kind`. A wrong kind string passes every unit test and leaves the
consumer blind, so those joins are asserted here against the real
consumers (orient, verify, vault), not against mocks.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from wikirag.config import WIKIRAG_HOME_ENV, KnowledgeBase, create_kb
from wikirag.llm.client import FakeLLMClient, LLMResult
from wikirag.llm.embeddings import FakeEmbedder
from wikirag.retrieval.orient import orient
from wikirag.retrieval.verify import trigger_consolidation
from wikirag.storage.vault import export_vault
from wikirag.synthesis import consolidation, notes
from wikirag.synthesis.atlas import OVERVIEW_KIND
from wikirag.synthesis.notes import NOTE_KIND
from wikirag.synthesis.summaries import SUMMARY_KIND


@pytest.fixture()
def kb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[KnowledgeBase]:
    monkeypatch.setenv(WIKIRAG_HOME_ENV, str(tmp_path / "kbs"))
    with create_kb("synth") as handle:
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


def note_reply(text: str = "A hoard is a dragon's accumulated treasure.",
               links: list[dict] | None = None) -> FakeLLMClient:
    payload = {"text": text, "sources": [1], "links": links or []}
    return FakeLLMClient(LLMResult(text=json.dumps(payload)))


# --------------------------------------------------------------- verify join


def test_a_keyword_note_clears_the_r2_consolidation_trigger(kb: KnowledgeBase) -> None:
    """verify.trigger_consolidation must stop queueing once a note exists."""
    keyword_id = kb.store.upsert_keyword("Hoard")
    kb.store.tag_chunks([("d1#1#0", keyword_id)])

    # Before synthesis the concept is demand.
    assert trigger_consolidation(kb, [keyword_id]) == 1

    notes.synthesize_note(kb, keyword_id, client=note_reply(), embedder=FakeEmbedder())

    stored = kb.store.get_note(kind=NOTE_KIND, keyword_id=keyword_id)
    assert stored is not None, "notes must use the kind verify.py queries"
    assert not stored["stale"]
    # After synthesis it is not demand any more.
    assert trigger_consolidation(kb, [keyword_id]) == 0


def test_a_stale_note_becomes_demand_again(kb: KnowledgeBase) -> None:
    keyword_id = kb.store.upsert_keyword("Hoard")
    kb.store.tag_chunks([("d1#1#0", keyword_id)])
    notes.synthesize_note(kb, keyword_id, client=note_reply(), embedder=FakeEmbedder())
    note = kb.store.get_note(kind=NOTE_KIND, keyword_id=keyword_id)
    assert note is not None

    kb.store.set_note_stale(int(note["id"]), True)

    assert trigger_consolidation(kb, [keyword_id]) == 1


# --------------------------------------------------------------- orient join


def test_a_document_summary_is_visible_to_orientation(kb: KnowledgeBase) -> None:
    """An S0 KB orients to nothing; an L1 summary must switch the channel on."""
    assert orient(kb, "dragon hoards").is_empty()

    kb.store.upsert_note(
        kind=SUMMARY_KIND,
        doc_id="d1",
        text="This book covers dragon hoards and how wyrms guard silver.",
        sources=["d1#1#0"],
    )

    regions = orient(kb, "dragon hoards", embedder=FakeEmbedder())
    assert not regions.is_empty(), "orient must see notes of kind 'summary'"
    assert "d1" in regions.doc_ids


def test_an_overview_is_visible_to_orientation(kb: KnowledgeBase) -> None:
    kb.store.upsert_note(
        kind=OVERVIEW_KIND,
        text="The knowledge base covers dragons, their hoards and their flight.",
    )
    # The overview has no doc_id, so it raises no document region; what matters
    # is that the pass runs and does not crash on a doc_id-less note.
    regions = orient(kb, "what does this knowledge base cover?", embedder=FakeEmbedder())
    assert regions.doc_ids == {}


# ---------------------------------------------------------------- vault join


def test_the_vault_renders_every_note_kind(kb: KnowledgeBase) -> None:
    keyword_id = kb.store.upsert_keyword("Hoard")
    kb.store.tag_chunks([("d1#1#0", keyword_id)])
    kb.store.upsert_note(kind=NOTE_KIND, keyword_id=keyword_id,
                         text="A hoard is treasure.", sources=["d1#1#0"])
    kb.store.upsert_note(kind=SUMMARY_KIND, doc_id="d1", text="About dragons.")
    kb.store.upsert_note(kind=OVERVIEW_KIND, text="Everything about dragons.")

    report = export_vault(kb)

    written = {p.name for p in report.written}
    assert "overview.md" in written
    exported = list(kb.vault_dir.rglob("*.md"))
    assert len(exported) >= 3, exported
    # Deleting the vault must lose nothing: a re-export reproduces it.
    for path in exported:
        path.unlink()
    again = export_vault(kb)
    assert len(again.written) == len(report.written)


# -------------------------------------------------------- consolidation loop


def test_consolidation_drains_the_queue_and_reports(kb: KnowledgeBase) -> None:
    hot = kb.store.upsert_keyword("Hoard")
    cold = kb.store.upsert_keyword("Ledger")
    kb.store.tag_chunks([("d1#1#0", hot), ("d1#1#1", cold)])
    kb.store.queue_consolidation(hot, "query")

    pending = consolidation.demand(kb)
    assert pending[0].keyword_id == hot and pending[0].queued

    report = consolidation.consolidate_pending(
        kb, client=note_reply(), embedder=FakeEmbedder(),
        budget=consolidation.ConsolidationBudget(max_notes=1),
    )

    assert report.built == 1 and report.failed == 0
    assert kb.store.get_note(kind=NOTE_KIND, keyword_id=hot) is not None
    # Dequeued only after the note existed.
    assert [r["keyword_id"] for r in kb.store.consolidation_demand()] == []


def test_consolidation_budget_caps_the_run(kb: KnowledgeBase) -> None:
    for name in ("Hoard", "Flight", "Silver"):
        keyword_id = kb.store.upsert_keyword(name)
        kb.store.tag_chunks([("d1#1#0", keyword_id)])
        kb.store.queue_consolidation(keyword_id, "query")

    report = consolidation.consolidate_pending(
        kb, client=note_reply(), embedder=FakeEmbedder(),
        budget=consolidation.ConsolidationBudget(max_notes=2),
    )

    assert report.total == 2
    assert len(kb.store.consolidation_demand()) == 1  # the rest stays queued


def test_a_failing_note_does_not_abort_the_batch(kb: KnowledgeBase) -> None:
    class Flaky:
        model = "flaky"
        def __init__(self) -> None:
            self.n = 0
        def generate(self, **kwargs: object) -> LLMResult:
            self.n += 1
            if self.n == 1:
                raise RuntimeError("model hiccup")
            return LLMResult(text=json.dumps({"text": "ok", "sources": [1], "links": []}))

    for name in ("Hoard", "Flight"):
        keyword_id = kb.store.upsert_keyword(name)
        kb.store.tag_chunks([("d1#1#0", keyword_id)])
        kb.store.queue_consolidation(keyword_id, "query")

    report = consolidation.consolidate_pending(
        kb, client=Flaky(), embedder=FakeEmbedder(),
        budget=consolidation.ConsolidationBudget(max_notes=5),
    )

    assert report.failed == 1 and report.built == 1
    assert len(report.errors) == 1


def test_quota_exhaustion_stops_consolidation_cleanly(kb: KnowledgeBase) -> None:
    from wikirag.llm.client import AllKeysExhausted

    class Exhausted:
        model = "spent"
        def generate(self, **kwargs: object) -> LLMResult:
            raise AllKeysExhausted("pool spent")

    keyword_id = kb.store.upsert_keyword("Hoard")
    kb.store.tag_chunks([("d1#1#0", keyword_id)])
    kb.store.queue_consolidation(keyword_id, "query")

    report = consolidation.consolidate_pending(
        kb, client=Exhausted(), embedder=FakeEmbedder()
    )

    assert report.stopped_early and report.total == 0
    # The request survives for the next run.
    assert len(kb.store.consolidation_demand()) == 1


def test_demand_prefers_hot_and_central_concepts(kb: KnowledgeBase) -> None:
    hot = kb.store.upsert_keyword("Hoard")
    cold = kb.store.upsert_keyword("Ledger")
    kb.store.tag_chunks([("d1#1#0", hot), ("d1#1#1", hot), ("d1#1#2", cold)])
    kb.store.log_query(query="dragon hoards?", depth="R2", hit_keywords=[hot])

    ranked = consolidation.demand(kb)

    assert [d.keyword_id for d in ranked][0] == hot
    assert ranked[0].reason == "hot"
