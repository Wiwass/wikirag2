"""Tests for wikirag.synthesis.summaries — offline: FakeLLMClient, temp KB.

The integration that matters comes first: a summary note written here must
be findable by retrieval/orient.py, because a gist the orientation pass
cannot see is a gist nobody reads.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from wikirag.config import WIKIRAG_HOME_ENV, KnowledgeBase, create_kb
from wikirag.llm.client import AllKeysExhausted, FakeLLMClient, LLMResult, Usage
from wikirag.llm.meter import MeteredClient
from wikirag.retrieval.orient import orient
from wikirag.synthesis import summaries
from wikirag.synthesis.summaries import (
    MIN_DIGEST_TOKENS,
    SUMMARY_KIND,
    summarize_document,
    summarize_stage,
)

DOC = "doc-bestiary"
OTHER = "doc-ledgers"

# Two sections per document, each with its own opening chunk, so a digest and
# a "covers" reference are unambiguous.
SECTIONS: tuple[tuple[str, str, int, int, int], ...] = (
    ("s-hoard", "Dragon hoard", 0, 3, 20),
    ("s-flight", "Flight and wings", 1, 21, 40),
)
CHUNKS: tuple[tuple[str, str, str], ...] = (
    ("c-hoard-1", "s-hoard", "The wyrm counts its coins in the deep vault."),
    ("c-hoard-2", "s-hoard", "More coins, more counting, deeper vaults."),
    ("c-flight-1", "s-flight", "Wings beat above the crags at dawn."),
)

SUMMARY_TEXT = (
    "A bestiary of dragons: their hoards and their flight, organised as two "
    "chapters, the first on treasure, the second on movement."
)


def response(summary: str = SUMMARY_TEXT, covers: list[int] | None = None) -> LLMResult:
    payload = {"summary": summary, "covers": [1] if covers is None else covers}
    return LLMResult(
        text=json.dumps(payload),
        usage=Usage(prompt_tokens=100, output_tokens=30, total_tokens=130),
    )


@pytest.fixture()
def kb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[KnowledgeBase]:
    """An S0 KB with one sectioned document (plus a second, unsummarized one)."""
    monkeypatch.setenv(WIKIRAG_HOME_ENV, str(tmp_path))
    knowledge_base = create_kb("kb")
    store = knowledge_base.store
    store.add_document(doc_id=DOC, filename="bestiary.pdf", title="Bestiary of Wyrms")
    store.add_sections(
        DOC,
        [
            {
                "id": sid,
                "ordinal": ordinal,
                "title": title,
                "level": 0,
                "page_start": start,
                "page_end": end,
            }
            for sid, title, ordinal, start, end in SECTIONS
        ],
    )
    store.add_chunks(
        [
            {
                "id": cid,
                "doc_id": DOC,
                "section_id": sid,
                "ordinal": ordinal,
                "text": text,
            }
            for ordinal, (cid, sid, text) in enumerate(CHUNKS)
        ]
    )
    store.add_document(doc_id=OTHER, filename="ledgers.pdf", title="Accounts")
    store.add_chunks(
        [
            {
                "id": "c-tax",
                "doc_id": OTHER,
                "section_id": None,
                "ordinal": 0,
                "text": "Quarterly filings and receipts.",
            }
        ]
    )
    with knowledge_base:
        yield knowledge_base


def metered(kb: KnowledgeBase, *results: LLMResult) -> MeteredClient:
    """A client shaped like the one the stage factory returns."""
    return MeteredClient(
        FakeLLMClient(*results), store=kb.store, stage="summarize", role="synth"
    )


# ------------------------------------------------------- the note, and orient


def test_summary_note_is_stored_where_orient_looks(kb: KnowledgeBase) -> None:
    note_id = summarize_document(kb, DOC, client=metered(kb, response()))

    note = kb.store.get_note(kind=SUMMARY_KIND, doc_id=DOC)
    assert note is not None
    assert note["id"] == note_id
    assert note["doc_id"] == DOC
    assert note["kind"] == "summary"
    assert SUMMARY_TEXT in note["text"]
    assert not note["stale"]


def test_orient_finds_the_summary_and_raises_its_document_region(
    kb: KnowledgeBase,
) -> None:
    summarize_document(kb, DOC, client=metered(kb, response()))

    regions = orient(kb, "dragon hoards and flight")

    assert not regions.is_empty()
    assert regions.doc_ids.get(DOC) == pytest.approx(1.0)
    assert OTHER not in regions.doc_ids


def test_grounding_records_the_referenced_chunks(kb: KnowledgeBase) -> None:
    note_id = summarize_document(kb, DOC, client=metered(kb, response(covers=[2])))

    # Entry [2] is the second section; its evidence is that section's opener.
    assert kb.store.note_sources(note_id) == ["c-flight-1"]


def test_unusable_covers_grounds_the_note_in_everything_shown(
    kb: KnowledgeBase,
) -> None:
    note_id = summarize_document(
        kb, DOC, client=metered(kb, response(covers=[99]))
    )

    assert kb.store.note_sources(note_id) == ["c-flight-1", "c-hoard-1"]


def test_rerun_replaces_the_note_instead_of_duplicating_it(kb: KnowledgeBase) -> None:
    first = summarize_document(kb, DOC, client=metered(kb, response()))
    second = summarize_document(
        kb, DOC, client=metered(kb, response("A rewritten gist of the same book."))
    )

    assert first == second
    note = kb.store.get_note(note_id=first)
    assert note is not None
    assert "rewritten gist" in note["text"]
    with kb.store.transaction() as db:
        count = db.execute(
            "SELECT COUNT(*) FROM notes WHERE kind = 'summary' AND doc_id = ?", (DOC,)
        ).fetchone()[0]
    assert count == 1


# ------------------------------------------------------------------ the prompt


def test_prompt_carries_a_numbered_outline_and_forbids_verbatim_echo(
    kb: KnowledgeBase,
) -> None:
    client = metered(kb, response())
    summarize_document(kb, DOC, client=client)

    call = client.inner.calls[0]  # type: ignore[attr-defined]
    user, system = call["user"], call["system"]

    assert "Bestiary of Wyrms" in user
    assert "[1] Dragon hoard (pp. 3-20)" in user
    assert "[2] Flight and wings (pp. 21-40)" in user

    # No-echo (docs/ingestion.md rule 1): the model is told to write new prose
    # and to refer to the document by entry number, never to reproduce text.
    assert "Never copy, quote or reproduce sentences" in system
    assert "in your own words" in system
    assert "You never receive the full text" in system
    assert call["json_schema"] is not None


def test_whole_chunks_are_never_sent_only_bounded_digests(kb: KnowledgeBase) -> None:
    long_text = " ".join(f"word{i}" for i in range(4000))
    kb.store.add_chunks(
        [
            {
                "id": "c-long",
                "doc_id": DOC,
                "section_id": "s-flight",
                "ordinal": 99,
                "text": long_text,
            }
        ]
    )
    client = metered(kb, response())
    summarize_document(kb, DOC, client=client, budget_tokens=300)

    user = client.inner.calls[0]["user"]  # type: ignore[attr-defined]
    assert long_text not in user
    assert len(user) < len(long_text)


def test_oversized_document_is_truncated_by_budget(kb: KnowledgeBase) -> None:
    """A book-sized outline is cut to the spine, and only its chapters remain."""
    store = kb.store
    store.add_sections(
        DOC,
        [
            {
                "id": f"s-extra-{i}",
                "ordinal": 10 + i,
                "title": f"Subsection number {i} of the appendix material",
                "level": 2,
                "page_start": 100 + i,
                "page_end": 100 + i,
            }
            for i in range(400)
        ],
    )
    store.add_chunks(
        [
            {
                "id": f"c-extra-{i}",
                "doc_id": DOC,
                "section_id": f"s-extra-{i}",
                "ordinal": 100 + i,
                "text": f"Appendix text number {i}, at some length, repeated often.",
            }
            for i in range(400)
        ]
    )
    client = metered(kb, response())
    summarize_document(kb, DOC, client=client, budget_tokens=600)

    user = client.inner.calls[0]["user"]  # type: ignore[attr-defined]
    from wikirag.ingestion.chunking import estimate_tokens

    assert estimate_tokens(user) < 600
    # The level-0 chapters survive the cut; level-2 subsections are the surplus.
    assert "Dragon hoard" in user and "Flight and wings" in user
    assert user.count("Subsection number") < 400


def test_max_sections_caps_the_outline(kb: KnowledgeBase) -> None:
    client = metered(kb, response())
    summarize_document(kb, DOC, client=client, max_sections=1)

    user = client.inner.calls[0]["user"]  # type: ignore[attr-defined]
    assert "[1] Dragon hoard" in user
    assert "Flight and wings" not in user


def test_digest_budget_is_respected_per_entry(kb: KnowledgeBase) -> None:
    client = metered(kb, response())
    summarize_document(kb, DOC, client=client)

    user = client.inner.calls[0]["user"]  # type: ignore[attr-defined]
    assert "opens: The wyrm counts its coins" in user
    # Only a section's FIRST chunk can be digested; the rest never travels.
    assert "More coins, more counting" not in user


def test_document_without_sections_still_gets_a_gist(kb: KnowledgeBase) -> None:
    note_id = summarize_document(kb, OTHER, client=metered(kb, response()))

    assert kb.store.get_note(kind=SUMMARY_KIND, doc_id=OTHER) is not None
    assert kb.store.note_sources(note_id) == ["c-tax"]


def test_document_without_chunks_raises(kb: KnowledgeBase) -> None:
    kb.store.add_document(doc_id="doc-empty", filename="empty.pdf", title="Empty")

    with pytest.raises(RuntimeError):
        summarize_document(kb, "doc-empty", client=metered(kb, response()))


# ----------------------------------------------------------------- failures


class ExplodingClient:
    """A provider that fails the way a quota-exhausted or broken one does."""

    model = "exploding"

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def generate(self, **kwargs: object) -> LLMResult:
        self.calls += 1
        raise self.error


def test_all_keys_exhausted_propagates(kb: KnowledgeBase) -> None:
    inner = ExplodingClient(AllKeysExhausted("no quota left"))
    client = MeteredClient(inner, store=kb.store, stage="summarize", role="synth")

    with pytest.raises(AllKeysExhausted):
        summarize_document(kb, DOC, client=client)

    # One attempt only: a retry would be doomed, and the runner stops the batch.
    assert inner.calls == 1
    assert kb.store.get_note(kind=SUMMARY_KIND, doc_id=DOC) is None


def test_provider_failure_leaves_no_half_written_note(kb: KnowledgeBase) -> None:
    inner = ExplodingClient(RuntimeError("provider is down"))
    client = MeteredClient(inner, store=kb.store, stage="summarize", role="synth")

    with pytest.raises(RuntimeError):
        summarize_document(kb, DOC, client=client)

    assert kb.store.get_note(kind=SUMMARY_KIND, doc_id=DOC) is None
    assert kb.store.stale_notes() == []


def test_unparsable_response_is_retried_then_raises(kb: KnowledgeBase) -> None:
    client = metered(kb, LLMResult(text="not json at all"))

    with pytest.raises(RuntimeError):
        summarize_document(kb, DOC, client=client)

    assert len(client.inner.calls) == summaries.MAX_ATTEMPTS  # type: ignore[attr-defined]
    assert kb.store.get_note(kind=SUMMARY_KIND, doc_id=DOC) is None


def test_fenced_json_is_accepted(kb: KnowledgeBase) -> None:
    fenced = LLMResult(text=f"```json\n{response().text}\n```")

    note_id = summarize_document(kb, DOC, client=metered(kb, fenced))

    assert kb.store.get_note(note_id=note_id) is not None


def test_empty_summary_writes_nothing(kb: KnowledgeBase) -> None:
    with pytest.raises(RuntimeError):
        summarize_document(kb, DOC, client=metered(kb, response(summary="   ")))

    assert kb.store.get_note(kind=SUMMARY_KIND, doc_id=DOC) is None


# -------------------------------------------------------------------- stage


def test_stage_goes_through_the_factory_and_meters_every_call(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        summaries, "_client_factory", lambda knowledge_base: metered(kb, response())
    )

    summarize_stage(kb, DOC)

    assert kb.store.get_note(kind=SUMMARY_KIND, doc_id=DOC) is not None
    calls = kb.store.get_llm_calls()
    assert len(calls) == 1
    assert (calls[0]["stage"], calls[0]["role"]) == ("summarize", "synth")


def test_stage_is_a_no_op_when_a_fresh_summary_exists(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    summarize_document(kb, DOC, client=metered(kb, response()))

    def explode(knowledge_base: KnowledgeBase) -> object:
        raise AssertionError("the stage must not call a provider for a fresh summary")

    monkeypatch.setattr(summaries, "_client_factory", explode)
    summarize_stage(kb, DOC)  # no exception: nothing was generated


def test_stage_regenerates_a_stale_summary(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    note_id = summarize_document(kb, DOC, client=metered(kb, response()))
    kb.store.set_note_stale(note_id)
    monkeypatch.setattr(
        summaries,
        "_client_factory",
        lambda knowledge_base: metered(kb, response("A fresh gist after staleness.")),
    )

    summarize_stage(kb, DOC)

    note = kb.store.get_note(note_id=note_id)
    assert note is not None
    assert "fresh gist" in note["text"]
    assert not note["stale"]


def test_stage_signature_fits_the_runner(kb: KnowledgeBase) -> None:
    """Callable[[KnowledgeBase, str], None] — what register_stage expects."""
    from wikirag.ingestion import runner

    runner.register_stage("summarize", summarize_stage)
    try:
        assert runner.STAGES["summarize"] is summarize_stage
        assert "summarize" in runner.stage_sequence("S1")
    finally:
        runner.STAGES.pop("summarize", None)


def test_digest_minimum_is_honored_when_the_budget_is_tiny(kb: KnowledgeBase) -> None:
    """Below one digest's worth of budget, titles travel alone — never a sliver."""
    client = metered(kb, response())
    summarize_document(kb, DOC, client=client, budget_tokens=1)

    user = client.inner.calls[0]["user"]  # type: ignore[attr-defined]
    assert "opens:" not in user or MIN_DIGEST_TOKENS <= 0
