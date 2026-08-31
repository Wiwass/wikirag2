"""Xref extraction tests: pattern coverage, resolution, precision, idempotency.

Offline by construction: a temp KB under WIKIRAG_HOME, fixture text written
straight into the store, no network and no models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from wikirag.config import WIKIRAG_HOME_ENV, KnowledgeBase, create_kb
from wikirag.ingestion.xrefs import (
    XREF_ABBREVIATIONS_KEY,
    apply_xrefs,
    extract_xrefs,
    load_abbreviations,
)

CORE = "core"
GUIDE = "guide"

# Fixture chunks: one pattern form per chunk, plus a near-miss chunk.
CHUNK_TEXTS: dict[str, str] = {
    # 1. cued locator, English, twice with different cues.
    "c0": "The rules are described in Chapter 5, and see Chapter 5 for the details.",
    # 2. named locator via a configured abbreviation, single page and range.
    "c1": "Vis extraction is covered in ArM5 p.83 and again in ArM5 pp. 84-85.",
    # 3. cued document reference by title, and by abbreviation.
    "c2": "As described in The Mysteries, initiation is slow; see Myst too.",
    # 4. quoted heading behind a cue.
    "c3": 'See the chapter "Raw Vis" before running a covenant.',
    # 5. cued locator, Italian: one resolvable chapter, one unresolvable section.
    "c4": "Vedi capitolo 1 e cfr. sezione 7 per il resto.",
    # 6. near misses: no cue, no locator, no known title.
    "c5": (
        "He reached the sea chapter of his life, page after page, at 83 years "
        'old, and could see 5 people whispering "the wind is cold" nearby. '
        "Chapter 5 was hard to read."
    ),
}


@pytest.fixture()
def kb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KnowledgeBase:
    monkeypatch.setenv(WIKIRAG_HOME_ENV, str(tmp_path))
    handle = create_kb("testkb")
    handle.set_config(
        XREF_ABBREVIATIONS_KEY,
        json.dumps({"ArM5": "Ars Magica Fifth Edition", "Myst": GUIDE}),
    )

    handle.store.add_document(
        doc_id=GUIDE, filename="mysteries.pdf", title="The Mysteries"
    )
    handle.store.add_sections(
        GUIDE, [{"id": "guide:s0", "ordinal": 0, "title": "Chapter 1: Initiation"}]
    )

    handle.store.add_document(
        doc_id=CORE, filename="core.pdf", title="Ars Magica Fifth Edition"
    )
    handle.store.add_sections(
        CORE,
        [
            {"id": "core:s0", "ordinal": 0, "title": "Chapter 1: Introduction"},
            {"id": "core:s1", "ordinal": 1, "title": "Chapter 5: Magic"},
            {"id": "core:s2", "ordinal": 2, "title": "Raw Vis"},
        ],
    )
    handle.store.add_chunks(
        [
            {
                "id": f"{CORE}:{cid}",
                "doc_id": CORE,
                "section_id": "core:s1",
                "ordinal": ordinal,
                "text": text,
            }
            for ordinal, (cid, text) in enumerate(CHUNK_TEXTS.items())
        ]
    )
    yield handle
    handle.close()


def rows_for(kb: KnowledgeBase, chunk: str) -> list[dict[str, Any]]:
    return [
        r
        for r in extract_xrefs(kb, CORE)
        if r["from_chunk_id"] == f"{CORE}:{chunk}"
    ]


def payload(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["raw_ref"], row["to_doc_id"], row["to_section_id"], row["kind"])


# ------------------------------------------------------------------ patterns


def test_cued_chapter_resolves_to_own_document_section(kb: KnowledgeBase) -> None:
    rows = rows_for(kb, "c0")
    assert [payload(r) for r in rows] == [
        ("described in Chapter 5", CORE, "core:s1", "chapter"),
        ("see Chapter 5", CORE, "core:s1", "chapter"),
    ]


def test_abbreviation_page_references(kb: KnowledgeBase) -> None:
    rows = rows_for(kb, "c1")
    assert [payload(r) for r in rows] == [
        ("ArM5 p.83", CORE, None, "page"),  # abbreviation -> document title
        ("ArM5 pp. 84-85", CORE, None, "page"),
    ]


def test_document_references_by_title_and_by_abbreviation(kb: KnowledgeBase) -> None:
    rows = rows_for(kb, "c2")
    assert [payload(r) for r in rows] == [
        ("As described in The Mysteries", GUIDE, None, "document"),
        ("see Myst", GUIDE, None, "document"),  # abbreviation -> doc_id
    ]


def test_quoted_heading_behind_a_cue_resolves_to_a_section(kb: KnowledgeBase) -> None:
    rows = rows_for(kb, "c3")
    assert [payload(r) for r in rows] == [
        ('See the chapter "Raw Vis"', CORE, "core:s2", "section")
    ]


def test_italian_cues_and_unresolved_reference_is_kept(kb: KnowledgeBase) -> None:
    rows = rows_for(kb, "c4")
    assert [payload(r) for r in rows] == [
        ("Vedi capitolo 1", CORE, "core:s0", "chapter"),
        # No section 7 in this document: the raw ref survives, unresolved.
        ("cfr. sezione 7", None, None, "section"),
    ]


# ----------------------------------------------------------------- precision


def test_near_misses_produce_nothing(kb: KnowledgeBase) -> None:
    assert rows_for(kb, "c5") == []


def test_named_locator_wins_over_the_overlapping_document_match(
    kb: KnowledgeBase,
) -> None:
    kb.store.add_chunks(
        [
            {
                "id": f"{CORE}:c6",
                "doc_id": CORE,
                "section_id": None,
                "ordinal": 99,
                "text": "Vedi ArM5 p.83 per le regole.",
            }
        ]
    )
    assert [payload(r) for r in rows_for(kb, "c6")] == [("ArM5 p.83", CORE, None, "page")]


def test_no_abbreviations_configured_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(WIKIRAG_HOME_ENV, str(tmp_path))
    with create_kb("bare") as bare:
        assert load_abbreviations(bare) == {}
        bare.store.add_document(doc_id=CORE, filename="core.pdf", title=None)
        bare.store.add_chunks(
            [
                {
                    "id": "core:c0",
                    "doc_id": CORE,
                    "section_id": None,
                    "ordinal": 0,
                    "text": "Vis extraction is covered in ArM5 p.83.",
                }
            ]
        )
        # Without the abbreviation, "ArM5 p.83" carries no cue: nothing matches.
        assert extract_xrefs(bare, CORE) == []


def test_malformed_abbreviation_config_is_reported(kb: KnowledgeBase) -> None:
    kb.set_config(XREF_ABBREVIATIONS_KEY, "{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_abbreviations(kb)
    kb.set_config(XREF_ABBREVIATIONS_KEY, json.dumps(["ArM5"]))
    with pytest.raises(ValueError, match="JSON object"):
        load_abbreviations(kb)


# --------------------------------------------------------------- persistence


def test_apply_xrefs_writes_and_is_idempotent(kb: KnowledgeBase) -> None:
    count = apply_xrefs(kb, CORE)
    assert count == len(extract_xrefs(kb, CORE)) > 0

    def stored() -> list[tuple[Any, ...]]:
        return [
            payload(r) + (r["from_chunk_id"],)
            for chunk in CHUNK_TEXTS
            for r in kb.store.xrefs_from(f"{CORE}:{chunk}")
        ]

    first = stored()
    assert apply_xrefs(kb, CORE) == count
    assert stored() == first  # replaced, not appended

    # A document with no references still applies cleanly.
    kb.store.add_document(doc_id="empty", filename="empty.pdf", title="Empty Book")
    assert apply_xrefs(kb, "empty") == 0


def test_apply_xrefs_leaves_other_documents_alone(kb: KnowledgeBase) -> None:
    kb.store.add_chunks(
        [
            {
                "id": "guide:c0",
                "doc_id": GUIDE,
                "section_id": "guide:s0",
                "ordinal": 0,
                "text": "See Chapter 1 of this book.",
            }
        ]
    )
    assert apply_xrefs(kb, GUIDE) == 1
    apply_xrefs(kb, CORE)
    assert [payload(r) for r in kb.store.xrefs_from("guide:c0")] == [
        ("See Chapter 1", GUIDE, "guide:s0", "chapter")
    ]
