"""Store tests: schema, transactionality, FTS sync, cascades, ops tables.

Offline by construction: a temp SQLite file, no network, no models.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from wikirag.storage.store import SCHEMA_VERSION, Store, fts_query


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "kb.sqlite")
    s.migrate()
    yield s
    s.close()


def seed_document(store: Store, doc_id: str = "doc1") -> None:
    store.add_document(doc_id=doc_id, filename=f"{doc_id}.pdf", title="Test Doc")
    store.add_sections(
        doc_id,
        [
            {"id": f"{doc_id}:s1", "ordinal": 0, "title": "Magic", "level": 1},
            {
                "id": f"{doc_id}:s2",
                "parent_id": f"{doc_id}:s1",
                "ordinal": 1,
                "title": "Vis",
                "level": 2,
            },
        ],
    )
    store.add_chunks(
        [
            {
                "id": f"{doc_id}:c0",
                "doc_id": doc_id,
                "section_id": f"{doc_id}:s1",
                "ordinal": 0,
                "text": "Hermetic magic is structured by arts and techniques.",
                "token_count": 9,
            },
            {
                "id": f"{doc_id}:c1",
                "doc_id": doc_id,
                "section_id": f"{doc_id}:s1",
                "ordinal": 1,
                "text": "A magus casts spontaneous spells without a formula.",
                "token_count": 8,
            },
            {
                "id": f"{doc_id}:c2",
                "doc_id": doc_id,
                "section_id": f"{doc_id}:s2",
                "ordinal": 2,
                "text": "Raw vis is condensed magical power gathered at auras.",
                "token_count": 9,
            },
        ]
    )


# ------------------------------------------------------------------ schema


def test_migrate_creates_schema_and_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "kb.sqlite"
    with Store(path) as s:
        assert s.migrate() == SCHEMA_VERSION
        assert s.migrate() == SCHEMA_VERSION  # no-op on an up-to-date DB
        tables = {
            r["name"]
            for r in s._rows(
                s._db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            )
        }
    expected = {
        "documents",
        "sections",
        "chunks",
        "xrefs",
        "keywords",
        "chunk_keywords",
        "notes",
        "note_sources",
        "note_links",
        "chunks_fts",
        "notes_fts",
        "embeddings",
        "jobs",
        "kb_config",
        "llm_calls",
        "query_log",
        "consolidation_queue",
    }
    assert expected <= tables

    # Reopening an existing DB keeps its version and data.
    with Store(path) as s2:
        assert s2.migrate() == SCHEMA_VERSION


def test_pragmas(store: Store) -> None:
    assert store._db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store._db.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_future_schema_version_refuses(tmp_path: Path) -> None:
    with Store(tmp_path / "kb.sqlite") as s:
        s.migrate()
        s._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        with pytest.raises(RuntimeError):
            s.migrate()


# ---------------------------------------------------------- transactionality


def test_failing_batch_leaves_nothing(store: Store) -> None:
    seed_document(store)
    before = len(store.get_document_chunks("doc1"))
    bad_batch = [
        {"id": "doc1:c3", "doc_id": "doc1", "section_id": None, "ordinal": 3, "text": "fine"},
        # duplicate ID: the whole batch must roll back
        {"id": "doc1:c0", "doc_id": "doc1", "section_id": None, "ordinal": 4, "text": "dup"},
    ]
    with pytest.raises(sqlite3.IntegrityError):
        store.add_chunks(bad_batch)
    assert len(store.get_document_chunks("doc1")) == before
    assert store.get_chunks(["doc1:c3"]) == []
    assert not store.fts_search("fine", kinds=("chunk",))


def test_unknown_document_rejected(store: Store) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        store.add_chunks(
            [{"id": "x:c0", "doc_id": "missing", "section_id": None, "ordinal": 0, "text": "t"}]
        )


def test_transaction_groups_stage_output(store: Store) -> None:
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.add_document(doc_id="d2", filename="d2.pdf")
            store.add_chunks(
                [{"id": "d2:c0", "doc_id": "d2", "section_id": None, "ordinal": 0, "text": "t"}]
            )
            raise RuntimeError("stage crashed")
    assert store.get_document("d2") is None
    assert store.get_chunks(["d2:c0"]) == []


# ---------------------------------------------------------------------- FTS


def test_fts_finds_inserted_text(store: Store) -> None:
    seed_document(store)
    hits = store.fts_search("spontaneous spells", kinds=("chunk",))
    assert [h["owner_id"] for h in hits][:1] == ["doc1:c1"]
    assert hits[0]["doc_id"] == "doc1"
    assert hits[0]["score"] > 0  # -bm25: higher is better


def test_fts_query_sanitizes_operators(store: Store) -> None:
    seed_document(store)
    # Raw FTS5 would choke on the quote/parenthesis; the helper must not.
    assert store.fts_search('what is "raw vis" (really)?', kinds=("chunk",))
    assert fts_query("raw vis") == '"raw" OR "vis"'
    assert fts_query("???") == ""
    assert store.fts_search("???") == []


def test_fts_stays_in_sync_on_delete_and_update(store: Store) -> None:
    seed_document(store)
    assert store.fts_search("auras", kinds=("chunk",))
    with store.transaction() as db:
        db.execute("DELETE FROM chunks WHERE id = 'doc1:c2'")
    assert store.fts_search("auras", kinds=("chunk",)) == []

    with store.transaction() as db:
        db.execute("UPDATE chunks SET text = 'certamen duels of will' WHERE id = 'doc1:c0'")
    assert not store.fts_search("hermetic", kinds=("chunk",))
    assert [h["owner_id"] for h in store.fts_search("certamen", kinds=("chunk",))] == ["doc1:c0"]

    # The external-content index agrees with the content table.
    assert store._db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check')")


def test_fts_searches_notes_as_content(store: Store) -> None:
    seed_document(store)
    kid = store.upsert_keyword("Raw Vis")
    store.upsert_note(kind="keyword", keyword_id=kid, text="Vis is harvested at auras.", sources=["doc1:c2"])
    hits = store.fts_search("vis")
    kinds = {h["owner_kind"] for h in hits}
    assert kinds == {"chunk", "note"}


# ------------------------------------------------------------------ reading


def test_get_chunks_preserves_requested_order(store: Store) -> None:
    seed_document(store)
    got = store.get_chunks(["doc1:c2", "doc1:c0", "missing"])
    assert [c["id"] for c in got] == ["doc1:c2", "doc1:c0"]


def test_section_window_and_neighbors(store: Store) -> None:
    seed_document(store)
    window = store.section_window("doc1:c0")
    assert [c["id"] for c in window] == ["doc1:c0", "doc1:c1"]
    assert [c["id"] for c in store.neighbors("doc1:c1", radius=1)] == [
        "doc1:c0",
        "doc1:c1",
        "doc1:c2",
    ]
    assert store.section_window("nope") == []


def test_sectionless_chunk_falls_back_to_document_window(store: Store) -> None:
    store.add_document(doc_id="d3", filename="d3.md")
    store.add_chunks(
        [
            {"id": "d3:c0", "doc_id": "d3", "section_id": None, "ordinal": 0, "text": "a"},
            {"id": "d3:c1", "doc_id": "d3", "section_id": None, "ordinal": 1, "text": "b"},
        ]
    )
    assert [c["id"] for c in store.section_window("d3:c0")] == ["d3:c0", "d3:c1"]


def test_xrefs_roundtrip(store: Store) -> None:
    seed_document(store)
    store.add_xrefs(
        [
            {
                "from_chunk_id": "doc1:c0",
                "raw_ref": "see Chapter 2",
                "to_doc_id": "doc1",
                "to_section_id": "doc1:s2",
                "kind": "section",
            }
        ]
    )
    refs = store.xrefs_from("doc1:c0")
    assert len(refs) == 1 and refs[0]["to_section_id"] == "doc1:s2"


# ------------------------------------------------------------------- forget


def test_forget_document_cascades(store: Store) -> None:
    seed_document(store)
    seed_document(store, "doc2")
    kid = store.upsert_keyword("Vis")
    store.tag_chunks([("doc1:c2", kid), ("doc2:c2", kid)])
    store.add_xrefs([{"from_chunk_id": "doc1:c0", "raw_ref": "cf. p.83"}])
    store.enqueue_jobs("doc1", ["chunk", "embed"])
    note_id = store.upsert_note(
        kind="keyword", keyword_id=kid, text="Vis note", sources=["doc1:c2", "doc2:c2"]
    )
    summary_id = store.upsert_note(kind="summary", doc_id="doc1", text="Doc summary", sources=["doc1:c0"])
    store.upsert_embedding(
        owner_kind="chunk", owner_id="doc1:c0", model="m", dim=3, vector_ref="0", content_hash="h"
    )
    store.upsert_embedding(
        owner_kind="note", owner_id=str(summary_id), model="m", dim=3, vector_ref="1", content_hash="h"
    )
    orphan = store.upsert_note(kind="keyword", keyword_id=store.upsert_keyword("Certamen"),
                               text="only doc1", sources=["doc1:c1"])

    assert store.forget_document("doc1") is True
    assert store.forget_document("doc1") is False  # already gone

    assert store.get_document("doc1") is None
    assert store.get_document_chunks("doc1") == []
    assert store.get_sections("doc1") == []
    assert store.xrefs_from("doc1:c0") == []
    assert store.pending_jobs("doc1") == []
    assert store.chunks_for_keyword(kid) == [] or [
        c["doc_id"] for c in store.chunks_for_keyword(kid)
    ] == ["doc2"]
    assert store.get_embedding("chunk", "doc1:c0", "m") is None
    assert store.get_embedding("note", str(summary_id), "m") is None
    assert store.get_note(summary_id) is None  # doc summary went with the doc
    assert store.get_note(orphan) is None  # nothing grounds it any more
    surviving = store.get_note(note_id)
    assert surviving is not None and surviving["stale"] == 1
    assert store.note_sources(note_id) == ["doc2:c2"]
    assert [h["owner_id"] for h in store.fts_search("hermetic", kinds=("chunk",))] == ["doc2:c0"]
    assert store.get_document("doc2") is not None
    assert len(store.get_document_chunks("doc2")) == 3


# ----------------------------------------------------------------- concepts


def test_keyword_upsert_is_canonical(store: Store) -> None:
    a = store.upsert_keyword("Raw Vis")
    b = store.upsert_keyword("raw vis")
    assert a == b
    assert store.get_keyword("RAW VIS")["name"] == "Raw Vis"
    assert store.get_keyword(keyword_id=a)["id"] == a
    assert [k["id"] for k in store.list_keywords()] == [a]
    with pytest.raises(ValueError):
        store.get_keyword()


def test_tagging_and_note_links(store: Store) -> None:
    seed_document(store)
    vis = store.upsert_keyword("Vis")
    aura = store.upsert_keyword("Aura")
    store.tag_chunks([("doc1:c2", vis), ("doc1:c2", vis)])  # idempotent
    assert [c["id"] for c in store.chunks_for_keyword(vis)] == ["doc1:c2"]
    assert [k["id"] for k in store.keywords_for_chunk("doc1:c2")] == [vis]

    note_id = store.upsert_note(kind="keyword", keyword_id=vis, text="v", sources=["doc1:c2"])
    store.add_note_links(
        note_id, [{"to_keyword_id": aura, "type": "found-at", "grounding_chunk_id": "doc1:c2"}]
    )
    links = store.note_links_from(note_id)
    assert links[0]["to_keyword"] == "Aura" and links[0]["type"] == "found-at"


def test_note_upsert_replaces_sources_and_clears_stale(store: Store) -> None:
    seed_document(store)
    vis = store.upsert_keyword("Vis")
    first = store.upsert_note(kind="keyword", keyword_id=vis, text="v1", sources=["doc1:c0"])
    store.set_note_stale(first)
    assert [n["id"] for n in store.stale_notes()] == [first]

    second = store.upsert_note(kind="keyword", keyword_id=vis, text="v2", sources=["doc1:c2"])
    assert second == first  # same subject, same note
    note = store.get_note(kind="keyword", keyword_id=vis)
    assert note["text"] == "v2" and note["stale"] == 0
    assert store.note_sources(first) == ["doc1:c2"]
    assert store.stale_notes() == []

    overview = store.upsert_note(kind="overview", text="the whole KB")
    assert store.get_note(kind="overview")["id"] == overview
    with pytest.raises(ValueError):
        store.get_note()


# --------------------------------------------------------------- embeddings


def test_embeddings_upsert_and_hashes(store: Store) -> None:
    seed_document(store)
    store.upsert_embedding(
        owner_kind="chunk", owner_id="doc1:c0", model="m1", dim=4, vector_ref="0", content_hash="h1"
    )
    store.upsert_embedding(
        owner_kind="chunk", owner_id="doc1:c0", model="m1", dim=4, vector_ref="7", content_hash="h2"
    )
    row = store.get_embedding("chunk", "doc1:c0", "m1")
    assert row["vector_ref"] == "7" and row["content_hash"] == "h2"
    assert store.embedding_hashes("chunk", "m1") == {"doc1:c0": "h2"}
    store.delete_embedding("chunk", "doc1:c0")
    assert store.embedding_hashes("chunk", "m1") == {}


# --------------------------------------------------------------------- jobs


def test_job_lifecycle(store: Store) -> None:
    seed_document(store)
    store.enqueue_jobs("doc1", ["extract", "chunk"])
    store.enqueue_jobs("doc1", ["chunk", "embed"])  # existing jobs untouched
    pending = store.pending_jobs("doc1")
    assert [j["stage"] for j in pending] == ["extract", "chunk", "embed"]

    job = pending[0]
    store.start_job(job["id"])
    assert store.get_job("doc1", "extract")["status"] == "running"
    store.finish_job(job["id"])
    done = store.get_job("doc1", "extract")
    assert done["status"] == "done" and done["attempts"] == 1
    assert [j["stage"] for j in store.pending_jobs("doc1")] == ["chunk", "embed"]

    chunk_job = store.get_job("doc1", "chunk")
    for _ in range(3):
        store.start_job(chunk_job["id"])
        store.finish_job(chunk_job["id"], status="failed", error="boom")
    failed = store.get_job("doc1", "chunk")
    assert failed["attempts"] == 3 and failed["last_error"] == "boom"
    # out of retries: no longer offered
    assert [j["stage"] for j in store.pending_jobs("doc1", max_attempts=3)] == ["embed"]
    assert [j["stage"] for j in store.pending_jobs()] == ["embed"]


# ------------------------------------------------------------------- config


def test_kb_config_roundtrip(store: Store) -> None:
    assert store.config_get("synthesis_level") is None
    assert store.config_get("synthesis_level", "S0") == "S0"
    store.config_set("synthesis_level", "S2")
    store.config_set("synthesis_level", "S3")  # upsert
    store.config_set("role.enrich", "gemini-flash")
    assert store.config_get("synthesis_level") == "S3"
    assert store.config_all() == {"synthesis_level": "S3", "role.enrich": "gemini-flash"}


# ----------------------------------------------------------------- metering


def test_llm_calls_roundtrip(store: Store) -> None:
    store.record_llm_call(
        stage="enrich",
        role="enrich",
        model="gemini-2.5-flash",
        tokens_in=1200,
        tokens_out=300,
        cost=0.0021,
        wall_ms=850,
    )
    store.record_llm_call(
        stage="answer", role="answer", model="local", tokens_in=10, tokens_out=5, cost=0.0, wall_ms=12
    )
    calls = store.get_llm_calls()
    assert len(calls) == 2
    assert set(calls[0]) == {
        "id",
        "ts",
        "stage",
        "role",
        "model",
        "tokens_in",
        "tokens_out",
        "cost",
        "wall_ms",
    }
    assert calls[0]["stage"] == "enrich" and calls[0]["cost"] == pytest.approx(0.0021)
    assert calls[1]["model"] == "local"
    assert calls[0]["ts"].endswith("+00:00")


# -------------------------------------------------------------------- usage


def test_query_log_and_consolidation_queue(store: Store) -> None:
    qid = store.log_query(
        query="how does vis work?",
        depth="R1",
        anchors=["doc1:c2"],
        hit_keywords=["vis"],
        no_answer=False,
    )
    assert isinstance(qid, int)
    logged = store.get_queries()
    assert logged[0]["anchors"] == ["doc1:c2"]
    assert logged[0]["hit_keywords"] == ["vis"]
    assert logged[0]["no_answer"] is False

    vis = store.upsert_keyword("Vis")
    store.queue_consolidation(vis, "query-miss")
    store.queue_consolidation(vis, "query-miss")  # same reason: one row
    store.queue_consolidation(vis, "hot")
    demand = store.consolidation_demand()
    assert {d["reason"] for d in demand} == {"query-miss", "hot"}
    assert demand[0]["name"] == "Vis"
    store.dequeue_consolidation(vis)
    assert store.consolidation_demand() == []
