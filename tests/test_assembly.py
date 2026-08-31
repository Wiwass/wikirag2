"""Assembly tests: window expansion, dedup, budget ladder, caps, provenance.

Offline by construction: a temp KB under WIKIRAG_HOME, no network, no models.
The candidate contract is duck-typed on purpose (a local stand-in, so these
tests do not depend on retrieval/search.py being implemented).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pytest

from wikirag.config import KnowledgeBase, create_kb
from wikirag.ingestion.chunking import estimate_tokens
from wikirag.retrieval.assembly import Context, assemble


@dataclass
class FakeCandidate:
    """Duck-typed stand-in for retrieval.search.Candidate."""

    owner_kind: str
    owner_id: str
    text: str
    doc_id: str | None = None
    section_id: str | None = None
    rrf_score: float = 0.0
    score: float | None = None


@pytest.fixture()
def kb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[KnowledgeBase]:
    monkeypatch.setenv("WIKIRAG_HOME", str(tmp_path))
    handle = create_kb("test")
    seed(handle)
    yield handle
    handle.close()


def seed(kb: KnowledgeBase) -> None:
    """doc1: two sections, four chunks. doc2: six one-chunk sections.
    doc3: one wide section of five long chunks (radius ladder, skip cases)."""
    store = kb.store
    store.add_document(doc_id="doc1", filename="doc1.pdf", title="Grimoire")
    store.add_sections(
        "doc1",
        [
            {"id": "doc1:s1", "ordinal": 0, "title": "Magic", "level": 1},
            {"id": "doc1:s2", "ordinal": 1, "title": "Vis", "level": 1},
        ],
    )
    store.add_chunks(
        [
            {"id": "doc1#0", "doc_id": "doc1", "section_id": "doc1:s1", "ordinal": 0,
             "text": "alpha alpha alpha"},
            {"id": "doc1#1", "doc_id": "doc1", "section_id": "doc1:s1", "ordinal": 1,
             "text": "bravo bravo bravo"},
            {"id": "doc1#2", "doc_id": "doc1", "section_id": "doc1:s1", "ordinal": 2,
             "text": "charlie charlie charlie"},
            {"id": "doc1#3", "doc_id": "doc1", "section_id": "doc1:s2", "ordinal": 3,
             "text": "delta delta delta"},
        ]
    )

    store.add_document(doc_id="doc2", filename="doc2.pdf")
    store.add_sections(
        "doc2",
        [
            {"id": f"doc2:s{i}", "ordinal": i, "title": f"Part {i}", "level": 1}
            for i in range(6)
        ],
    )
    store.add_chunks(
        [
            {"id": f"doc2#{i}", "doc_id": "doc2", "section_id": f"doc2:s{i}",
             "ordinal": i, "text": f"echo{i} echo{i} echo{i}"}
            for i in range(6)
        ]
    )

    store.add_document(doc_id="doc3", filename="doc3.pdf", title="Tome")
    store.add_sections("doc3", [{"id": "doc3:s1", "ordinal": 0, "title": "Wide", "level": 1}])
    store.add_chunks(
        [
            {"id": f"doc3#{i}", "doc_id": "doc3", "section_id": "doc3:s1", "ordinal": i,
             "text": " ".join([f"foxtrot{i}"] * 20)}
            for i in range(5)
        ]
    )


def chunk_hit(chunk_id: str, doc_id: str, section_id: str, score: float) -> FakeCandidate:
    return FakeCandidate(
        owner_kind="chunk",
        owner_id=chunk_id,
        text="",
        doc_id=doc_id,
        section_id=section_id,
        rrf_score=score / 100,
        score=score,
    )


BIG = 10_000


def check_budget(context: Context, budget: int) -> None:
    assert context.token_count == estimate_tokens(context.text)
    assert context.token_count <= budget


# ------------------------------------------------------------------ expansion


def test_chunk_expands_to_section_window_in_order(kb: KnowledgeBase) -> None:
    context = assemble(kb, [chunk_hit("doc1#1", "doc1", "doc1:s1", 9.0)], BIG)

    assert len(context.entries) == 1
    entry = context.entries[0]
    assert entry.chunk_ids == ["doc1#0", "doc1#1", "doc1#2"]
    assert entry.text == "alpha alpha alpha\n\nbravo bravo bravo\n\ncharlie charlie charlie"
    assert context.text == "[1] Grimoire — Magic\n" + entry.text
    check_budget(context, BIG)


def test_note_is_passed_through_unexpanded(kb: KnowledgeBase) -> None:
    keyword_id = kb.store.upsert_keyword("Vis")
    note_id = kb.store.upsert_note(
        kind="keyword", text="vis is raw magical energy", keyword_id=keyword_id
    )
    candidate = FakeCandidate(
        owner_kind="note",
        owner_id=str(note_id),
        text="vis is raw magical energy",
        score=8.0,
    )

    context = assemble(kb, [candidate], BIG)

    entry = context.entries[0]
    assert entry.owner_kind == "note"
    assert entry.chunk_ids == []
    assert entry.text == "vis is raw magical energy"
    assert context.text == "[1] Note: Vis\nvis is raw magical energy"


def test_note_without_keyword_falls_back_to_owner_id(kb: KnowledgeBase) -> None:
    note_id = kb.store.upsert_note(kind="overview", text="the whole KB, briefly")
    candidate = FakeCandidate(
        owner_kind="note", owner_id=str(note_id), text="the whole KB, briefly", score=5.0
    )

    context = assemble(kb, [candidate], BIG)

    assert context.text.startswith(f"[1] Note: {note_id}\n")


# ---------------------------------------------------------------------- dedup


def test_two_hits_in_one_section_produce_one_entry(kb: KnowledgeBase) -> None:
    context = assemble(
        kb,
        [
            chunk_hit("doc1#0", "doc1", "doc1:s1", 9.0),
            chunk_hit("doc1#2", "doc1", "doc1:s1", 8.0),
        ],
        BIG,
    )

    assert len(context.entries) == 1
    assert context.entries[0].chunk_ids == ["doc1#0", "doc1#1", "doc1#2"]
    assert context.text.count("charlie charlie charlie") == 1


def test_dedup_leaves_only_unused_chunks_to_a_later_entry(kb: KnowledgeBase) -> None:
    """A budget-trimmed first entry leaves the rest of the section to the next."""
    hits = [
        chunk_hit("doc3#2", "doc3", "doc3:s1", 9.0),
        chunk_hit("doc3#0", "doc3", "doc3:s1", 8.0),
    ]
    budget = assemble(kb, [hits[0]], BIG).token_count - 1  # one chunk short of the window

    context = assemble(kb, hits, budget)

    assert [e.chunk_ids for e in context.entries] == [
        ["doc3#1", "doc3#2", "doc3#3"],
        ["doc3#0"],
    ]
    included = [cid for e in context.entries for cid in e.chunk_ids]
    assert len(included) == len(set(included))
    check_budget(context, budget)


def test_leftovers_never_glue_a_prefix_to_a_suffix(kb: KnowledgeBase) -> None:
    """A used middle splits a window; an entry keeps one contiguous run."""
    hits = [
        chunk_hit("doc3#2", "doc3", "doc3:s1", 9.0),
        chunk_hit("doc3#4", "doc3", "doc3:s1", 8.0),
    ]
    budget = assemble(kb, [hits[0]], BIG).token_count - 1

    context = assemble(kb, hits, budget)

    # doc3#0 survives too, but it is on the far side of the used middle.
    assert [e.chunk_ids for e in context.entries] == [
        ["doc3#1", "doc3#2", "doc3#3"],
        ["doc3#4"],
    ]


def test_already_included_hit_still_contributes_its_neighbours(
    kb: KnowledgeBase,
) -> None:
    """The hit is gone, so the nearest surviving run stands in for it."""
    hits = [
        chunk_hit("doc3#2", "doc3", "doc3:s1", 9.0),
        chunk_hit("doc3#3", "doc3", "doc3:s1", 8.0),
    ]
    budget = assemble(kb, [hits[0]], BIG).token_count - 1

    context = assemble(kb, hits, budget)

    assert [e.chunk_ids for e in context.entries] == [
        ["doc3#1", "doc3#2", "doc3#3"],
        ["doc3#4"],
    ]


def test_candidate_whose_window_is_fully_included_is_skipped(
    kb: KnowledgeBase,
) -> None:
    hits = [
        chunk_hit("doc1#0", "doc1", "doc1:s1", 9.0),
        chunk_hit("doc1#1", "doc1", "doc1:s1", 8.0),
        chunk_hit("doc1#2", "doc1", "doc1:s1", 7.0),
    ]

    context = assemble(kb, hits, BIG)

    assert len(context.entries) == 1


# --------------------------------------------------------------------- budget


@pytest.mark.parametrize("budget", [1, 5, 10, 12, 20, 40, 80, 200])
def test_budget_is_never_exceeded(kb: KnowledgeBase, budget: int) -> None:
    context = assemble(
        kb,
        [
            chunk_hit("doc1#1", "doc1", "doc1:s1", 9.0),
            chunk_hit("doc1#3", "doc1", "doc1:s2", 8.0),
            chunk_hit("doc2#0", "doc2", "doc2:s0", 7.0),
        ],
        budget,
    )
    check_budget(context, budget)


def test_tiny_budget_falls_back_to_the_hit_chunk(kb: KnowledgeBase) -> None:
    hit = chunk_hit("doc1#1", "doc1", "doc1:s1", 9.0)
    budget = estimate_tokens("[1] Grimoire — Magic\nbravo bravo bravo")

    context = assemble(kb, [hit], budget)

    assert context.entries[0].chunk_ids == ["doc1#1"]
    assert context.entries[0].text == "bravo bravo bravo"
    check_budget(context, budget)


def test_partial_window_is_centred_on_the_hit(kb: KnowledgeBase) -> None:
    """Between "hit only" and "whole section": the widest radius that fits."""
    hit = chunk_hit("doc3#2", "doc3", "doc3:s1", 9.0)
    full = assemble(kb, [hit], BIG)
    assert full.entries[0].chunk_ids == [f"doc3#{i}" for i in range(5)]

    context = assemble(kb, [hit], full.token_count - 1)

    assert context.entries[0].chunk_ids == ["doc3#1", "doc3#2", "doc3#3"]
    check_budget(context, full.token_count - 1)

    narrow = assemble(kb, [hit], context.token_count - 1)
    assert narrow.entries[0].chunk_ids == ["doc3#2"]


def test_entry_that_does_not_fit_is_skipped_but_later_ones_still_fit(
    kb: KnowledgeBase,
) -> None:
    long_hit = chunk_hit("doc3#2", "doc3", "doc3:s1", 9.0)
    small_hit = chunk_hit("doc2#0", "doc2", "doc2:s0", 8.0)
    small_alone = assemble(kb, [small_hit], BIG)

    # A budget that fits the small entry but not even one chunk of the big one.
    budget = small_alone.token_count
    context = assemble(kb, [long_hit, small_hit], budget)

    assert [e.owner_id for e in context.entries] == ["doc2#0"]
    assert context.entries[0].ref == 1
    check_budget(context, budget)


def test_zero_budget_gives_empty_context(kb: KnowledgeBase) -> None:
    context = assemble(kb, [chunk_hit("doc1#0", "doc1", "doc1:s1", 9.0)], 0)
    assert context == Context(text="", entries=[], token_count=0)


def test_empty_candidates_give_empty_context(kb: KnowledgeBase) -> None:
    context = assemble(kb, [], BIG)
    assert context.text == ""
    assert context.entries == []
    assert context.token_count == 0


# ------------------------------------------------------------------ diversity


def test_per_document_cap(kb: KnowledgeBase) -> None:
    candidates = [
        chunk_hit(f"doc2#{i}", "doc2", f"doc2:s{i}", 9.0 - i) for i in range(6)
    ]

    default = assemble(kb, candidates, BIG)
    assert len(default.entries) == 4  # DEFAULT_MAX_ENTRIES_PER_DOC
    assert [e.owner_id for e in default.entries] == [f"doc2#{i}" for i in range(4)]

    capped = assemble(kb, candidates, BIG, max_entries_per_doc=2)
    assert [e.owner_id for e in capped.entries] == ["doc2#0", "doc2#1"]


def test_cap_is_per_document_not_global(kb: KnowledgeBase) -> None:
    candidates = [
        chunk_hit("doc2#0", "doc2", "doc2:s0", 9.0),
        chunk_hit("doc2#1", "doc2", "doc2:s1", 8.0),
        chunk_hit("doc1#3", "doc1", "doc1:s2", 7.0),
    ]

    context = assemble(kb, candidates, BIG, max_entries_per_doc=2)

    assert [e.doc_id for e in context.entries] == ["doc2", "doc2", "doc1"]


# ----------------------------------------------------------------- provenance


def test_refs_are_sequential_from_one(kb: KnowledgeBase) -> None:
    candidates = [
        chunk_hit("doc1#1", "doc1", "doc1:s1", 9.0),
        chunk_hit("doc1#3", "doc1", "doc1:s2", 8.0),
        chunk_hit("doc2#0", "doc2", "doc2:s0", 7.0),
    ]

    context = assemble(kb, candidates, BIG)

    assert [e.ref for e in context.entries] == [1, 2, 3]
    for entry in context.entries:
        assert f"[{entry.ref}] " in context.text


def test_provenance_fields(kb: KnowledgeBase) -> None:
    context = assemble(
        kb,
        [
            chunk_hit("doc1#3", "doc1", "doc1:s2", 9.0),
            chunk_hit("doc2#0", "doc2", "doc2:s0", 8.0),
        ],
        BIG,
    )

    first, second = context.entries
    assert (first.owner_kind, first.owner_id) == ("chunk", "doc1#3")
    assert (first.doc_id, first.section_id) == ("doc1", "doc1:s2")
    assert (first.doc_title, first.section_title) == ("Grimoire", "Vis")
    assert first.chunk_ids == ["doc1#3"]
    assert first.score == 9.0
    # doc2 has no title: the filename stands in.
    assert second.doc_title == "doc2.pdf"
    assert second.section_title == "Part 0"


def test_score_falls_back_to_rrf_when_unscored(kb: KnowledgeBase) -> None:
    candidate = FakeCandidate(
        owner_kind="chunk",
        owner_id="doc1#3",
        text="",
        doc_id="doc1",
        section_id="doc1:s2",
        rrf_score=0.031,
        score=None,
    )

    context = assemble(kb, [candidate], BIG)

    assert context.entries[0].score == pytest.approx(0.031)


def test_entries_are_separated_by_a_blank_line(kb: KnowledgeBase) -> None:
    context = assemble(
        kb,
        [
            chunk_hit("doc1#3", "doc1", "doc1:s2", 9.0),
            chunk_hit("doc2#0", "doc2", "doc2:s0", 8.0),
        ],
        BIG,
    )

    assert context.text == (
        "[1] Grimoire — Vis\ndelta delta delta"
        "\n\n"
        "[2] doc2.pdf — Part 0\necho0 echo0 echo0"
    )


def test_output_is_deterministic(kb: KnowledgeBase) -> None:
    candidates = [
        chunk_hit("doc1#1", "doc1", "doc1:s1", 9.0),
        chunk_hit("doc2#0", "doc2", "doc2:s0", 8.0),
        chunk_hit("doc1#0", "doc1", "doc1:s1", 7.0),
    ]

    runs = [assemble(kb, candidates, 60) for _ in range(3)]

    assert all(run == runs[0] for run in runs)
