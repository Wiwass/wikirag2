"""Tests for wikirag.retrieval.orient — offline: FakeEmbedder, temp KB, no network.

The S0 case comes first, because it is the behavioural requirement the module
exists to satisfy: a KB with only chunks (no notes, no keywords — every KB is
one until Wave 4) must orient to nothing, cheaply, without resolving an
embedder, and boost must be the identity there.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest

from wikirag.config import WIKIRAG_HOME_ENV, KnowledgeBase, create_kb
from wikirag.llm.embeddings import FakeEmbedder
from wikirag.retrieval import orient as orient_module
from wikirag.retrieval.orient import MAX_BOOST, Regions, orient
from wikirag.retrieval.search import Candidate
from wikirag.storage.vectors import OwnerLike, VectorIndex

DRAGONS = "doc-dragons"
LEDGERS = "doc-ledgers"

# One document per topic, so an orientation hit is unambiguous. Section titles
# are the query's own words in the dragon document only.
CHUNKS: tuple[tuple[str, str, str, str], ...] = (
    ("c1", DRAGONS, "s-hoard", "the wyrm counts its coins"),
    ("c2", DRAGONS, "s-flight", "wings beat above the crags"),
    ("c3", LEDGERS, "s-tax", "quarterly filings and receipts"),
    ("c4", LEDGERS, "s-audit", "the auditor reviews the books"),
)

DRAGON_SUMMARY = "A bestiary of dragons, their hoard and their flight."
LEDGER_SUMMARY = "Bookkeeping: quarterly filings, taxation and audits."

SECTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    DRAGONS: (("s-hoard", "Dragon hoard"), ("s-flight", "Flight")),
    LEDGERS: (("s-tax", "Taxation"), ("s-audit", "Audit")),
}


class SpyEmbedder(FakeEmbedder):
    """FakeEmbedder that records whether anything was actually encoded."""

    def __init__(self) -> None:
        super().__init__()
        self.query_calls = 0

    def encode_query(self, text: str) -> np.ndarray:
        self.query_calls += 1
        return super().encode_query(text)


@pytest.fixture()
def kb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[KnowledgeBase]:
    """An S0 KB: documents, sections, chunks. No notes, no keywords."""
    monkeypatch.setenv(WIKIRAG_HOME_ENV, str(tmp_path))
    knowledge_base = create_kb("kb")
    store = knowledge_base.store
    for doc_id, title in ((DRAGONS, "Bestiary"), (LEDGERS, "Accounts")):
        store.add_document(doc_id=doc_id, filename=f"{doc_id}.pdf", title=title)
        store.add_sections(
            doc_id,
            [
                {"id": sid, "ordinal": i, "title": stitle}
                for i, (sid, stitle) in enumerate(SECTIONS[doc_id])
            ],
        )
    store.add_chunks(
        [
            {"id": cid, "doc_id": doc, "section_id": sid, "ordinal": i, "text": text}
            for i, (cid, doc, sid, text) in enumerate(CHUNKS)
        ]
    )
    with knowledge_base:
        yield knowledge_base


@pytest.fixture()
def embedder() -> SpyEmbedder:
    return SpyEmbedder()


def forbid_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the shared-encoder resolution path fail loudly if it is taken."""

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("orient resolved an embedder")

    monkeypatch.setattr(orient_module, "get_embedder", explode)


def seed_summaries(kb: KnowledgeBase) -> dict[str, int]:
    """One L1 summary per document; the dragon one carries the query's words."""
    return {
        DRAGONS: kb.store.upsert_note(
            kind="summary", doc_id=DRAGONS, text=DRAGON_SUMMARY, sources=["c1"]
        ),
        LEDGERS: kb.store.upsert_note(
            kind="summary", doc_id=LEDGERS, text=LEDGER_SUMMARY, sources=["c3"]
        ),
    }


def index_notes(
    kb: KnowledgeBase, embedder: FakeEmbedder, items: Sequence[tuple[OwnerLike, str]]
) -> None:
    index = VectorIndex(kb.vectors_dir, embedder.model_name, embedder.dim)
    for owner, text in items:
        index.upsert(owner, "hash", embedder.encode_passages([text])[0])
    index.save()


def candidate(owner_id: str, doc_id: str, section_id: str, rrf: float) -> Candidate:
    return Candidate(
        owner_kind="chunk",
        owner_id=owner_id,
        text=f"text of {owner_id}",
        doc_id=doc_id,
        section_id=section_id,
        rrf_score=rrf,
    )


# ------------------------------------------------------------------- the S0 case


def test_s0_kb_orients_to_nothing_and_never_resolves_an_embedder(
    kb: KnowledgeBase, embedder: SpyEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    forbid_embedder(monkeypatch)
    # A chunk-only vector index exists, as on any indexed S0 KB: orientation
    # must still not open it, because there is no gist to search in it.
    index_notes(kb, embedder, [(("chunk", cid), text) for cid, _, _, text in CHUNKS])

    regions = orient(kb, "dragon hoard", embedder=embedder)

    assert regions.is_empty()
    assert (regions.doc_ids, regions.section_ids, regions.keyword_ids) == ({}, {}, {})
    assert embedder.query_calls == 0


def test_s0_boost_is_the_identity(kb: KnowledgeBase, embedder: SpyEmbedder) -> None:
    pool = [
        candidate("c1", DRAGONS, "s-hoard", 0.03),
        candidate("c3", LEDGERS, "s-tax", 0.02),
    ]
    regions = orient(kb, "dragon hoard", embedder=embedder)

    boosted = regions.boost(pool)

    assert boosted == pool
    assert [c.rrf_score for c in boosted] == [0.03, 0.02]


# -------------------------------------------------------------- ablation flag


def test_ablation_flag_short_circuits(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_summaries(kb)
    forbid_embedder(monkeypatch)

    def explode(*args: object, **kwargs: object) -> None:  # no store access at all
        raise AssertionError("orient touched the store while disabled")

    monkeypatch.setattr(kb.store, "list_documents", explode)
    monkeypatch.setattr(kb.store, "fts_search", explode)
    monkeypatch.setattr(kb.store, "list_keywords", explode)

    assert orient(kb, "dragon hoard", enabled=False).is_empty()


def test_blank_query_orients_to_nothing(kb: KnowledgeBase) -> None:
    seed_summaries(kb)
    assert orient(kb, "   ").is_empty()


# ------------------------------------------------------------------- regions


def test_summaries_make_the_right_document_the_strongest_region(
    kb: KnowledgeBase, embedder: SpyEmbedder
) -> None:
    notes = seed_summaries(kb)
    index_notes(
        kb,
        embedder,
        [
            (("note", str(notes[DRAGONS])), DRAGON_SUMMARY),
            (("note", str(notes[LEDGERS])), LEDGER_SUMMARY),
        ],
    )

    regions = orient(kb, "dragon hoard", embedder=embedder)

    assert regions.doc_ids[DRAGONS] == 1.0
    assert regions.doc_ids.get(LEDGERS, 0.0) < 1.0
    assert all(0.0 <= score <= 1.0 for score in regions.doc_ids.values())
    assert embedder.query_calls == 1  # the gist exists, so the dense channel runs


def test_lexical_channel_alone_orients_without_an_index(kb: KnowledgeBase) -> None:
    seed_summaries(kb)
    # No index file at all: the dense channel contributes nothing and no
    # embedder is resolved, yet orientation still works.
    regions = orient(kb, "taxation audits")
    assert regions.doc_ids[LEDGERS] == 1.0
    assert DRAGONS not in regions.doc_ids


def test_section_regions_come_from_matching_titles(kb: KnowledgeBase) -> None:
    seed_summaries(kb)
    regions = orient(kb, "dragon hoard")
    assert regions.section_ids["s-hoard"] == pytest.approx(regions.doc_ids[DRAGONS])
    assert "s-flight" not in regions.section_ids
    assert all(0.0 <= score <= 1.0 for score in regions.section_ids.values())


def test_keyword_names_matched_whole_word_only(kb: KnowledgeBase) -> None:
    hoard = kb.store.upsert_keyword("Dragon hoard")
    audit = kb.store.upsert_keyword("audit")
    kb.store.upsert_keyword("art")  # substring of "chart": must not match

    regions = orient(kb, "where is the dragon hoard charted?")

    assert regions.keyword_ids == {hoard: 1.0}
    assert audit not in regions.keyword_ids
    # Keywords alone are a region: no notes are needed for the concept layer.
    assert not regions.is_empty()


def test_overview_note_raises_no_document_region(kb: KnowledgeBase) -> None:
    kb.store.upsert_note(kind="overview", text="An atlas of dragons and ledgers alike.")
    regions = orient(kb, "atlas of dragons")
    assert regions.doc_ids == {}
    assert regions.is_empty()


# --------------------------------------------------------------------- boost


def test_boost_promotes_candidates_in_active_regions_within_the_cap() -> None:
    regions = Regions(doc_ids={DRAGONS: 1.0}, section_ids={"s-hoard": 0.5})
    # c3 leads the first stage; c1 is close enough behind for orientation to
    # overtake it, c4 (outside every region) is not.
    pool = [
        candidate("c3", LEDGERS, "s-tax", 0.030),
        candidate("c1", DRAGONS, "s-hoard", 0.028),
        candidate("c4", LEDGERS, "s-audit", 0.010),
    ]

    boosted = regions.boost(pool)

    assert [c.owner_id for c in boosted] == ["c1", "c3", "c4"]
    assert {c.owner_id for c in boosted} == {"c1", "c3", "c4"}  # nothing filtered
    by_id = {c.owner_id: c for c in boosted}
    # c1 sits in both an active document and an active section: capped bonus.
    assert by_id["c1"].rrf_score == pytest.approx(0.028 * (1 + MAX_BOOST))
    assert by_id["c3"].rrf_score == pytest.approx(0.030)  # outside: untouched
    assert by_id["c4"].rrf_score == pytest.approx(0.010)
    assert [c.rrf_score for c in pool] == [0.030, 0.028, 0.010]  # inputs intact


def test_boost_cannot_overturn_a_wide_gap() -> None:
    regions = Regions(doc_ids={DRAGONS: 1.0})
    pool = [
        candidate("c3", LEDGERS, "s-tax", 0.030),
        candidate("c1", DRAGONS, "s-hoard", 0.020),
    ]
    assert [c.owner_id for c in regions.boost(pool)] == ["c3", "c1"]


def test_boost_leaves_the_reranker_scale_alone() -> None:
    regions = Regions(doc_ids={DRAGONS: 1.0})
    scored = candidate("c1", DRAGONS, "s-hoard", 0.02)
    scored.score = 0.87
    assert regions.boost([scored])[0].score == 0.87


def test_boost_is_deterministic() -> None:
    regions = Regions(doc_ids={DRAGONS: 1.0, LEDGERS: 1.0})
    pool = [
        candidate("c1", DRAGONS, "s-hoard", 0.02),
        candidate("c3", LEDGERS, "s-tax", 0.02),  # exact tie, broken by owner_id
        candidate("c2", DRAGONS, "s-flight", 0.02),
    ]
    first = [(c.owner_id, c.rrf_score) for c in regions.boost(pool)]
    second = [(c.owner_id, c.rrf_score) for c in regions.boost(pool)]
    assert first == second
    assert [owner_id for owner_id, _ in first] == ["c1", "c2", "c3"]


def test_boost_of_empty_pool() -> None:
    assert Regions(doc_ids={DRAGONS: 1.0}).boost([]) == []
    assert Regions().boost([]) == []


def test_orient_result_boosts_end_to_end(kb: KnowledgeBase) -> None:
    seed_summaries(kb)
    regions = orient(kb, "dragon hoard")
    pool = [
        candidate("c3", LEDGERS, "s-tax", 0.030),
        candidate("c1", DRAGONS, "s-hoard", 0.028),
    ]
    assert [c.owner_id for c in regions.boost(pool)] == ["c1", "c3"]
