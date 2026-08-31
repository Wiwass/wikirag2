"""Tests for wikirag.retrieval.search — offline: FakeEmbedder, temp KB, no network.

The fixture KB holds five short chunks and two notes; the vector index is
built by hand from FakeEmbedder vectors so each test controls exactly which
owners the dense channel can see.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from wikirag.config import WIKIRAG_HOME_ENV, KnowledgeBase, create_kb
from wikirag.llm.embeddings import FakeEmbedder
from wikirag.retrieval import search as search_module
from wikirag.retrieval.search import RRF_K, Candidate, hybrid_search
from wikirag.storage.vectors import OwnerLike, VectorIndex

DOC = "doc1"

# Texts are chosen so the lexical and dense channels can be steered apart:
# "dragon hoard" is exactly the query of the RRF-math test (cosine 1.0 under
# FakeEmbedder, and the only chunk carrying both terms), while "ledger"/
# "quarterly" share no token with any query used here.
CHUNKS: tuple[tuple[str, str, str], ...] = (
    ("c1", "s1", "dragon hoard"),
    ("c2", "s1", "the hoard of coins rests in the vault"),
    ("c3", "s2", "a knight polishes his armour before dawn"),
    ("c4", "s2", "quarterly ledger entries and tax filings"),
    ("c5", "s2", "wyverns nest above the northern crags"),
)


@pytest.fixture()
def kb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[KnowledgeBase]:
    monkeypatch.setenv(WIKIRAG_HOME_ENV, str(tmp_path))
    knowledge_base = create_kb("kb")
    knowledge_base.store.add_document(doc_id=DOC, filename="doc1.pdf", title="Bestiary")
    knowledge_base.store.add_sections(
        DOC,
        [
            {"id": "s1", "ordinal": 0, "title": "Treasure"},
            {"id": "s2", "ordinal": 1, "title": "Fauna"},
        ],
    )
    knowledge_base.store.add_chunks(
        [
            {"id": cid, "doc_id": DOC, "section_id": sid, "ordinal": i, "text": text}
            for i, (cid, sid, text) in enumerate(CHUNKS)
        ]
    )
    with knowledge_base:
        yield knowledge_base


@pytest.fixture()
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


def build_index(
    kb: KnowledgeBase,
    embedder: FakeEmbedder,
    items: Sequence[tuple[OwnerLike, str]],
) -> None:
    """Write a vector index holding exactly ``items`` — (owner, text) pairs."""
    index = VectorIndex(kb.vectors_dir, embedder.model_name, embedder.dim)
    for owner, text in items:
        index.upsert(owner, "hash", embedder.encode_passages([text])[0])
    index.save()


def index_all_chunks(kb: KnowledgeBase, embedder: FakeEmbedder) -> None:
    build_index(kb, embedder, [(("chunk", cid), text) for cid, _, text in CHUNKS])


def by_id(candidates: Sequence[Candidate], owner_id: str) -> Candidate:
    return next(c for c in candidates if c.owner_id == owner_id)


# ----------------------------------------------------------------- channels


def test_lexical_hit_surfaces(kb: KnowledgeBase, embedder: FakeEmbedder) -> None:
    index_all_chunks(kb, embedder)
    results = hybrid_search(kb, ["ledger"], k=5, embedder=embedder)
    assert results[0].owner_id == "c4"
    assert results[0].owner_kind == "chunk"
    assert results[0].text == "quarterly ledger entries and tax filings"
    assert (results[0].doc_id, results[0].section_id) == (DOC, "s2")
    assert results[0].score is None  # rerank.py's field, untouched here


def test_dense_hit_surfaces_without_lexical_match(
    kb: KnowledgeBase, embedder: FakeEmbedder
) -> None:
    index_all_chunks(kb, embedder)
    query = "wyverns nest above the northern crags"
    assert kb.store.fts_search("basilisk") == []
    # No shared token with any chunk, so only the dense channel can answer;
    # c5's text is the query itself, so it is the nearest neighbour.
    results = hybrid_search(kb, ["basilisk"], k=5, embedder=embedder)
    assert results
    dense_only = hybrid_search(kb, [query], k=5, embedder=embedder)
    assert dense_only[0].owner_id == "c5"


def test_both_channels_outrank_single_channel(
    kb: KnowledgeBase, embedder: FakeEmbedder
) -> None:
    index_all_chunks(kb, embedder)
    results = hybrid_search(kb, ["dragon hoard"], k=5, embedder=embedder)
    top = results[0]
    # c1 is the query verbatim: BM25 rank 1 (only chunk with both terms) and
    # cosine 1.0, hence dense rank 1 — two firsts, hand-computed.
    assert top.owner_id == "c1"
    assert top.rrf_score == pytest.approx(2.0 / (RRF_K + 1))
    assert top.rrf_score > by_id(results, "c2").rrf_score


def test_fts_only_when_vector_index_missing(
    kb: KnowledgeBase, embedder: FakeEmbedder
) -> None:
    assert not (kb.vectors_dir / "index.npz").exists()
    results = hybrid_search(kb, ["ledger"], k=5, embedder=embedder)
    assert [c.owner_id for c in results] == ["c4"]
    assert results[0].rrf_score == pytest.approx(1.0 / (RRF_K + 1))


def test_empty_vector_index_contributes_nothing(
    kb: KnowledgeBase, embedder: FakeEmbedder
) -> None:
    build_index(kb, embedder, [])
    results = hybrid_search(kb, ["ledger"], k=5, embedder=embedder)
    assert [c.owner_id for c in results] == ["c4"]
    assert results[0].rrf_score == pytest.approx(1.0 / (RRF_K + 1))


def test_no_hits_anywhere(kb: KnowledgeBase) -> None:
    assert hybrid_search(kb, ["ledger"], k=0) == []
    assert hybrid_search(kb, ["", "   "], k=5) == []


# ------------------------------------------------------------- multi-anchor


def test_multi_query_fusion_boosts_shared_hit(
    kb: KnowledgeBase, embedder: FakeEmbedder
) -> None:
    index_all_chunks(kb, embedder)
    single = hybrid_search(kb, ["dragon"], k=5, embedder=embedder)
    fused = hybrid_search(kb, ["dragon", "hoard"], k=5, embedder=embedder)
    # c1 is lexically hit by both anchors, c3 by neither.
    assert fused[0].owner_id == "c1"
    assert by_id(fused, "c1").rrf_score > by_id(single, "c1").rrf_score
    assert by_id(fused, "c1").rrf_score > by_id(fused, "c3").rrf_score


def test_k_cap_and_dedup(kb: KnowledgeBase, embedder: FakeEmbedder) -> None:
    index_all_chunks(kb, embedder)
    results = hybrid_search(kb, ["dragon hoard", "dragon hoard"], k=2, embedder=embedder)
    assert len(results) == 2
    keys = [(c.owner_kind, c.owner_id) for c in results]
    assert len(set(keys)) == 2
    # Repeating an anchor doubles every contribution, it does not duplicate rows.
    once = hybrid_search(kb, ["dragon hoard"], k=2, embedder=embedder)
    assert results[0].rrf_score == pytest.approx(2 * once[0].rrf_score)
    assert [c.rrf_score for c in results] == sorted(
        (c.rrf_score for c in results), reverse=True
    )


# ------------------------------------------------------------------- notes


def test_notes_surface_as_candidates(kb: KnowledgeBase, embedder: FakeEmbedder) -> None:
    doc_note = kb.store.upsert_note(
        kind="summary", doc_id=DOC, text="a bestiary of dragons and their hoards"
    )
    keyword_id = kb.store.upsert_keyword("Alchemy")
    free_note = kb.store.upsert_note(
        kind="concept", keyword_id=keyword_id, text="alchemy transmutes base metals"
    )
    build_index(kb, embedder, [(("note", str(free_note)), "alchemy transmutes base metals")])

    lexical = hybrid_search(kb, ["bestiary"], k=5, embedder=embedder)
    note_hit = by_id(lexical, str(doc_note))
    assert note_hit.owner_kind == "note"
    assert (note_hit.doc_id, note_hit.section_id) == (DOC, None)

    # Dense-only note: hydrated through get_note, no doc_id of its own.
    dense = hybrid_search(kb, ["philosophers stone"], k=5, embedder=embedder)
    hydrated = by_id(dense, str(free_note))
    assert hydrated.owner_kind == "note"
    assert hydrated.text == "alchemy transmutes base metals"
    assert (hydrated.doc_id, hydrated.section_id) == (None, None)


# ------------------------------------------------------------- store drift


def test_stale_vector_entries_are_skipped(
    kb: KnowledgeBase, embedder: FakeEmbedder
) -> None:
    build_index(
        kb,
        embedder,
        [
            (("chunk", "c1"), "dragon hoard"),
            (("chunk", "ghost"), "dragon hoard"),
            (("note", "999"), "dragon hoard"),
        ],
    )
    results = hybrid_search(kb, ["dragon hoard"], k=5, embedder=embedder)
    ids = {(c.owner_kind, c.owner_id) for c in results}
    assert ("chunk", "c1") in ids
    assert ("chunk", "ghost") not in ids
    assert ("note", "999") not in ids


# ------------------------------------------------------------------ laziness


def test_no_model_is_resolved_when_embedder_is_given(
    kb: KnowledgeBase, embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_all_chunks(kb, embedder)
    monkeypatch.setattr(
        search_module,
        "get_embedder",
        lambda *a, **kw: pytest.fail("get_embedder must not be called"),
    )
    assert hybrid_search(kb, ["dragon hoard"], k=5, embedder=embedder)


def test_no_model_is_resolved_without_a_vector_index(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        search_module,
        "get_embedder",
        lambda *a, **kw: pytest.fail("get_embedder must not be called"),
    )
    assert [c.owner_id for c in hybrid_search(kb, ["ledger"], k=5)] == ["c4"]
