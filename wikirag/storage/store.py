"""SQLite store: schema, migrations, and transactional access.

Responsibilities
----------------
- Own the schema in docs/storage.md: documents, sections, chunks, xrefs,
  keywords, chunk_keywords, notes (with staleness), note_sources,
  note_links (typed, grounded), chunks_fts (FTS5, BM25), embeddings meta,
  jobs, kb_config, llm_calls, query_log, consolidation_queue.
- Stable chunk IDs derived from (doc_id, structural path, ordinal) —
  deterministic across re-runs on the same source file. The IDs are
  *derived by the chunker* (Wave 1); the store accepts caller-provided
  string IDs and never invents them.
- Transactions per stage output: a document's chunks + FTS rows commit
  together or not at all. WAL mode; single writer, concurrent readers.
- Cascading deletes make `forget <doc>` mostly schema (docs/synthesis.md).
- Job table drives ingestion resumability (replaces v1's pending-JSON files).

Schema notes (beyond the docs/storage.md draft)
-----------------------------------------------
- `notes.doc_id`: L1 summaries are per document, so a note needs a subject
  pointer besides `keyword_id`. It is also what makes a document's summaries
  disappear with the document via ON DELETE CASCADE.
- `notes_fts`: notes are content (docs/storage.md), so they get the same
  FTS5 external-content treatment as chunks and are searchable through the
  single `fts_search` entry point.
- FTS tables are kept in sync by triggers, which SQLite also fires for rows
  removed by foreign-key cascades — so `forget_document` needs no index
  bookkeeping of its own.

Public API
----------
Lifecycle:
  Store(path), .path, .migrate() -> int, .close(), context manager,
  .transaction() (reentrant; only the outermost block commits)

Documents and structure:
  add_document(*, doc_id, filename, title=None, status=..., meta=None)
  get_document(doc_id), list_documents(status=None)
  set_document_status(doc_id, status)
  add_sections(doc_id, sections), get_sections(doc_id)
  add_chunks(chunks), get_chunks(chunk_ids), get_document_chunks(doc_id)
  section_window(chunk_id), neighbors(chunk_id, radius=1)
  add_xrefs(xrefs), xrefs_from(chunk_id)
  forget_document(doc_id) -> bool

Concept layer:
  upsert_keyword(name, canonical_name=None) -> int
  get_keyword(name=None, keyword_id=None), list_keywords()
  tag_chunks(pairs), keywords_for_chunk(chunk_id), chunks_for_keyword(kid)
  upsert_note(*, kind, text, keyword_id=None, doc_id=None, sources=()) -> int
  get_note(note_id=None, *, kind=None, keyword_id=None, doc_id=None)
  note_sources(note_id), set_note_stale(note_id, stale=True), stale_notes()
  add_note_links(from_note_id, links), note_links_from(note_id)

Embeddings meta:
  upsert_embedding(*, owner_kind, owner_id, model, dim, vector_ref,
                   content_hash)
  get_embedding(owner_kind, owner_id, model)
  embedding_hashes(owner_kind, model), delete_embedding(...)

Retrieval:
  fts_search(query, *, limit=20, kinds=("chunk", "note"), raw=False)

Operations:
  enqueue_jobs(doc_id, stages), get_job(doc_id, stage),
  pending_jobs(doc_id=None, max_attempts=3), start_job(job_id),
  finish_job(job_id, status="done", error=None)
  config_get(key, default=None), config_set(key, value), config_all()
  record_llm_call(...), get_llm_calls()
  log_query(*, query, depth, anchors=(), hit_keywords=(), no_answer=False)
  get_queries(limit=100)
  queue_consolidation(keyword_id, reason), consolidation_demand(limit=50),
  dequeue_consolidation(keyword_id)
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# Order matters: chunks must exist before the FTS triggers reference them.
_SCHEMA = """
CREATE TABLE documents (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    title       TEXT,
    ingested_at TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'new',
    meta_json   TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE sections (
    id        TEXT PRIMARY KEY,
    doc_id    TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_id TEXT REFERENCES sections(id) ON DELETE CASCADE,
    ordinal   INTEGER NOT NULL,
    title     TEXT,
    level     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX sections_doc ON sections(doc_id, ordinal);

CREATE TABLE chunks (
    id          TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    section_id  TEXT REFERENCES sections(id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX chunks_doc ON chunks(doc_id, ordinal);
CREATE INDEX chunks_section ON chunks(section_id, ordinal);

CREATE TABLE xrefs (
    id            INTEGER PRIMARY KEY,
    from_chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    raw_ref       TEXT NOT NULL,
    to_doc_id     TEXT REFERENCES documents(id) ON DELETE SET NULL,
    to_section_id TEXT REFERENCES sections(id) ON DELETE SET NULL,
    kind          TEXT NOT NULL DEFAULT 'unknown'
);
CREATE INDEX xrefs_from ON xrefs(from_chunk_id);

CREATE TABLE keywords (
    id             INTEGER PRIMARY KEY,
    name           TEXT NOT NULL,
    canonical_name TEXT NOT NULL UNIQUE
);

CREATE TABLE chunk_keywords (
    chunk_id   TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    keyword_id INTEGER NOT NULL REFERENCES keywords(id) ON DELETE CASCADE,
    PRIMARY KEY (chunk_id, keyword_id)
);
CREATE INDEX chunk_keywords_kw ON chunk_keywords(keyword_id);

CREATE TABLE notes (
    id         INTEGER PRIMARY KEY,
    keyword_id INTEGER REFERENCES keywords(id) ON DELETE CASCADE,
    doc_id     TEXT REFERENCES documents(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL,
    text       TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    stale      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX notes_subject ON notes(kind, keyword_id, doc_id);

CREATE TABLE note_sources (
    note_id  INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    PRIMARY KEY (note_id, chunk_id)
);
CREATE INDEX note_sources_chunk ON note_sources(chunk_id);

CREATE TABLE note_links (
    id                 INTEGER PRIMARY KEY,
    from_note_id       INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    to_keyword_id      INTEGER NOT NULL REFERENCES keywords(id) ON DELETE CASCADE,
    type               TEXT NOT NULL,
    grounding_chunk_id TEXT REFERENCES chunks(id) ON DELETE SET NULL
);
CREATE INDEX note_links_from ON note_links(from_note_id);
CREATE INDEX note_links_to ON note_links(to_keyword_id);

CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='rowid');
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE VIRTUAL TABLE notes_fts USING fts5(text, content='notes', content_rowid='id');
CREATE TRIGGER notes_ai AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER notes_ad AFTER DELETE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER notes_au AFTER UPDATE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO notes_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE embeddings (
    owner_kind   TEXT NOT NULL,
    owner_id     TEXT NOT NULL,
    model        TEXT NOT NULL,
    dim          INTEGER NOT NULL,
    vector_ref   TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    PRIMARY KEY (owner_kind, owner_id, model)
);

CREATE TABLE jobs (
    id         INTEGER PRIMARY KEY,
    doc_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    stage      TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'pending',
    attempts   INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE (doc_id, stage)
);
CREATE INDEX jobs_status ON jobs(status);

CREATE TABLE kb_config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE llm_calls (
    id         INTEGER PRIMARY KEY,
    ts         TEXT NOT NULL,
    stage      TEXT NOT NULL,
    role       TEXT NOT NULL,
    model      TEXT NOT NULL,
    tokens_in  INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    cost       REAL NOT NULL,
    wall_ms    INTEGER NOT NULL
);

CREATE TABLE query_log (
    id                INTEGER PRIMARY KEY,
    ts                TEXT NOT NULL,
    query             TEXT NOT NULL,
    depth             TEXT NOT NULL,
    anchors_json      TEXT NOT NULL DEFAULT '[]',
    hit_keywords_json TEXT NOT NULL DEFAULT '[]',
    no_answer         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE consolidation_queue (
    keyword_id INTEGER NOT NULL REFERENCES keywords(id) ON DELETE CASCADE,
    reason     TEXT NOT NULL,
    queued_at  TEXT NOT NULL,
    PRIMARY KEY (keyword_id, reason)
);
"""

_FTS_TOKEN = re.compile(r"\w+", re.UNICODE)


def _now() -> str:
    """ISO-8601 UTC timestamp, the single time format used in the store."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 OR-query.

    User queries contain quotes, hyphens and other FTS5 operators; passing
    them through raises OperationalError. Words are extracted and quoted,
    OR-joined for recall (BM25 ranks, the caller cuts).
    """
    tokens = _FTS_TOKEN.findall(text)
    return " OR ".join(f'"{t}"' for t in tokens)


class Store:
    """One SQLite database per knowledge base; the KB's source of truth."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: transactions are explicit (see .transaction()).
        self._db = sqlite3.connect(str(self.path), isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA synchronous = NORMAL")
        self._depth = 0

    # ---------------------------------------------------------------- lifecycle

    def migrate(self) -> int:
        """Create or upgrade the schema; returns the resulting version."""
        version = int(self._db.execute("PRAGMA user_version").fetchone()[0])
        if version == SCHEMA_VERSION:
            return version
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"{self.path} has schema version {version}, "
                f"this build understands {SCHEMA_VERSION}"
            )
        # executescript() commits any open transaction first, so the BEGIN goes
        # inside the script rather than through .transaction().
        script = "BEGIN IMMEDIATE;\n"
        if version == 0:
            script += _SCHEMA
        script += f"\nPRAGMA user_version = {SCHEMA_VERSION};\nCOMMIT;"
        self._db.executescript(script)
        return SCHEMA_VERSION

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Reentrant write transaction; only the outermost block commits."""
        if self._depth:
            self._depth += 1
            try:
                yield self._db
            finally:
                self._depth -= 1
            return
        self._db.execute("BEGIN IMMEDIATE")
        self._depth = 1
        try:
            yield self._db
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        else:
            self._db.execute("COMMIT")
        finally:
            self._depth = 0

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _rows(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
        return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def _row(cursor: sqlite3.Cursor) -> dict[str, Any] | None:
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    # ---------------------------------------------------------------- documents

    def add_document(
        self,
        *,
        doc_id: str,
        filename: str,
        title: str | None = None,
        status: str = "new",
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        with self.transaction() as db:
            db.execute(
                "INSERT INTO documents (id, filename, title, ingested_at, status, meta_json)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (doc_id, filename, title, _now(), status, json.dumps(dict(meta or {}))),
            )

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        return self._row(
            self._db.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        )

    def list_documents(self, status: str | None = None) -> list[dict[str, Any]]:
        if status is None:
            cur = self._db.execute("SELECT * FROM documents ORDER BY ingested_at, id")
        else:
            cur = self._db.execute(
                "SELECT * FROM documents WHERE status = ? ORDER BY ingested_at, id",
                (status,),
            )
        return self._rows(cur)

    def set_document_status(self, doc_id: str, status: str) -> None:
        with self.transaction() as db:
            db.execute("UPDATE documents SET status = ? WHERE id = ?", (status, doc_id))

    # ----------------------------------------------------------------- structure

    def add_sections(
        self, doc_id: str, sections: Sequence[Mapping[str, Any]]
    ) -> None:
        """Insert a document's section tree. Parents must precede children."""
        with self.transaction() as db:
            db.executemany(
                "INSERT INTO sections (id, doc_id, parent_id, ordinal, title, level)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        s["id"],
                        doc_id,
                        s.get("parent_id"),
                        s["ordinal"],
                        s.get("title"),
                        s.get("level", 0),
                    )
                    for s in sections
                ],
            )

    def get_sections(self, doc_id: str) -> list[dict[str, Any]]:
        return self._rows(
            self._db.execute(
                "SELECT * FROM sections WHERE doc_id = ? ORDER BY ordinal", (doc_id,)
            )
        )

    def add_chunks(self, chunks: Sequence[Mapping[str, Any]]) -> None:
        """Insert a stage's chunks; chunk rows and FTS rows commit together."""
        with self.transaction() as db:
            db.executemany(
                "INSERT INTO chunks (id, doc_id, section_id, ordinal, text, token_count)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        c["id"],
                        c["doc_id"],
                        c.get("section_id"),
                        c["ordinal"],
                        c["text"],
                        c.get("token_count", 0),
                    )
                    for c in chunks
                ],
            )

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[dict[str, Any]]:
        """Fetch chunks by ID, in the order requested (missing IDs skipped)."""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        rows = {
            r["id"]: r
            for r in self._rows(
                self._db.execute(
                    f"SELECT * FROM chunks WHERE id IN ({placeholders})", tuple(chunk_ids)
                )
            )
        }
        return [rows[cid] for cid in chunk_ids if cid in rows]

    def get_document_chunks(self, doc_id: str) -> list[dict[str, Any]]:
        return self._rows(
            self._db.execute(
                "SELECT * FROM chunks WHERE doc_id = ? ORDER BY ordinal", (doc_id,)
            )
        )

    def section_window(self, chunk_id: str) -> list[dict[str, Any]]:
        """Read-big expansion: every chunk of the hit's section, in order.

        Chunks without a section fall back to their document neighborhood.
        """
        chunk = self._row(
            self._db.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
        )
        if chunk is None:
            return []
        if chunk["section_id"] is None:
            return self.neighbors(chunk_id, radius=2)
        return self._rows(
            self._db.execute(
                "SELECT * FROM chunks WHERE section_id = ? ORDER BY ordinal",
                (chunk["section_id"],),
            )
        )

    def neighbors(self, chunk_id: str, radius: int = 1) -> list[dict[str, Any]]:
        """Chunks adjacent to this one in document order, inclusive."""
        chunk = self._row(
            self._db.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
        )
        if chunk is None:
            return []
        return self._rows(
            self._db.execute(
                "SELECT * FROM chunks WHERE doc_id = ? AND ordinal BETWEEN ? AND ?"
                " ORDER BY ordinal",
                (chunk["doc_id"], chunk["ordinal"] - radius, chunk["ordinal"] + radius),
            )
        )

    def add_xrefs(self, xrefs: Sequence[Mapping[str, Any]]) -> None:
        with self.transaction() as db:
            db.executemany(
                "INSERT INTO xrefs (from_chunk_id, raw_ref, to_doc_id, to_section_id, kind)"
                " VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        x["from_chunk_id"],
                        x["raw_ref"],
                        x.get("to_doc_id"),
                        x.get("to_section_id"),
                        x.get("kind", "unknown"),
                    )
                    for x in xrefs
                ],
            )

    def xrefs_from(self, chunk_id: str) -> list[dict[str, Any]]:
        return self._rows(
            self._db.execute(
                "SELECT * FROM xrefs WHERE from_chunk_id = ? ORDER BY id", (chunk_id,)
            )
        )

    def forget_document(self, doc_id: str) -> bool:
        """Delete a document and everything derived from it.

        Cascades do the work; the two things schema cannot express are handled
        here: notes grounded in the deleted chunks become stale (and are
        removed when nothing grounds them any more), and embedding rows —
        polymorphic, hence unconstrained — are deleted for the affected owners.
        """
        with self.transaction() as db:
            if db.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone() is None:
                return False
            db.execute(
                "UPDATE notes SET stale = 1 WHERE id IN ("
                "  SELECT note_id FROM note_sources WHERE chunk_id IN ("
                "    SELECT id FROM chunks WHERE doc_id = ?))",
                (doc_id,),
            )
            db.execute(
                "DELETE FROM embeddings WHERE owner_kind = 'chunk' AND owner_id IN ("
                "  SELECT id FROM chunks WHERE doc_id = ?)",
                (doc_id,),
            )
            db.execute(
                "DELETE FROM embeddings WHERE owner_kind = 'note' AND owner_id IN ("
                "  SELECT CAST(id AS TEXT) FROM notes WHERE doc_id = ?)",
                (doc_id,),
            )
            db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            emptied = [
                str(r["id"])
                for r in db.execute(
                    "SELECT id FROM notes WHERE stale = 1 AND NOT EXISTS ("
                    "  SELECT 1 FROM note_sources WHERE note_id = notes.id)"
                ).fetchall()
            ]
            if emptied:
                placeholders = ",".join("?" * len(emptied))
                db.execute(
                    f"DELETE FROM notes WHERE id IN ({placeholders})", tuple(emptied)
                )
                db.execute(
                    f"DELETE FROM embeddings WHERE owner_kind = 'note'"
                    f" AND owner_id IN ({placeholders})",
                    tuple(emptied),
                )
            return True

    # ------------------------------------------------------------------ concepts

    def upsert_keyword(self, name: str, canonical_name: str | None = None) -> int:
        """Get or create a keyword; the canonical name is the identity."""
        canonical = (canonical_name or name).strip().lower()
        with self.transaction() as db:
            db.execute(
                "INSERT INTO keywords (name, canonical_name) VALUES (?, ?)"
                " ON CONFLICT(canonical_name) DO NOTHING",
                (name, canonical),
            )
            return int(
                db.execute(
                    "SELECT id FROM keywords WHERE canonical_name = ?", (canonical,)
                ).fetchone()[0]
            )

    def get_keyword(
        self, name: str | None = None, *, keyword_id: int | None = None
    ) -> dict[str, Any] | None:
        if keyword_id is not None:
            cur = self._db.execute("SELECT * FROM keywords WHERE id = ?", (keyword_id,))
        elif name is not None:
            cur = self._db.execute(
                "SELECT * FROM keywords WHERE canonical_name = ?", (name.strip().lower(),)
            )
        else:
            raise ValueError("get_keyword needs a name or a keyword_id")
        return self._row(cur)

    def list_keywords(self) -> list[dict[str, Any]]:
        return self._rows(
            self._db.execute("SELECT * FROM keywords ORDER BY canonical_name")
        )

    def tag_chunks(self, pairs: Iterable[tuple[str, int]]) -> None:
        """Attach keywords to chunks by reference; no text is copied."""
        with self.transaction() as db:
            db.executemany(
                "INSERT INTO chunk_keywords (chunk_id, keyword_id) VALUES (?, ?)"
                " ON CONFLICT DO NOTHING",
                list(pairs),
            )

    def keywords_for_chunk(self, chunk_id: str) -> list[dict[str, Any]]:
        return self._rows(
            self._db.execute(
                "SELECT k.* FROM keywords k JOIN chunk_keywords ck ON ck.keyword_id = k.id"
                " WHERE ck.chunk_id = ? ORDER BY k.canonical_name",
                (chunk_id,),
            )
        )

    def chunks_for_keyword(self, keyword_id: int) -> list[dict[str, Any]]:
        return self._rows(
            self._db.execute(
                "SELECT c.* FROM chunks c JOIN chunk_keywords ck ON ck.chunk_id = c.id"
                " WHERE ck.keyword_id = ? ORDER BY c.doc_id, c.ordinal",
                (keyword_id,),
            )
        )

    # --------------------------------------------------------------------- notes

    def upsert_note(
        self,
        *,
        kind: str,
        text: str,
        keyword_id: int | None = None,
        doc_id: str | None = None,
        sources: Sequence[str] = (),
    ) -> int:
        """Create or replace the note for a subject; returns its ID.

        A note's subject is (kind, keyword_id, doc_id) — keyword notes, per
        document summaries, and the single overview all fit that shape. Writing
        a note clears its stale flag and replaces its grounding source set.
        """
        with self.transaction() as db:
            existing = db.execute(
                "SELECT id FROM notes WHERE kind = ? AND keyword_id IS ? AND doc_id IS ?",
                (kind, keyword_id, doc_id),
            ).fetchone()
            if existing is None:
                cur = db.execute(
                    "INSERT INTO notes (keyword_id, doc_id, kind, text, updated_at, stale)"
                    " VALUES (?, ?, ?, ?, ?, 0)",
                    (keyword_id, doc_id, kind, text, _now()),
                )
                note_id = int(cur.lastrowid)
            else:
                note_id = int(existing[0])
                db.execute(
                    "UPDATE notes SET text = ?, updated_at = ?, stale = 0 WHERE id = ?",
                    (text, _now(), note_id),
                )
                db.execute("DELETE FROM note_sources WHERE note_id = ?", (note_id,))
            db.executemany(
                "INSERT INTO note_sources (note_id, chunk_id) VALUES (?, ?)"
                " ON CONFLICT DO NOTHING",
                [(note_id, cid) for cid in sources],
            )
            return note_id

    def get_note(
        self,
        note_id: int | None = None,
        *,
        kind: str | None = None,
        keyword_id: int | None = None,
        doc_id: str | None = None,
    ) -> dict[str, Any] | None:
        if note_id is not None:
            cur = self._db.execute("SELECT * FROM notes WHERE id = ?", (note_id,))
        elif kind is not None:
            cur = self._db.execute(
                "SELECT * FROM notes WHERE kind = ? AND keyword_id IS ? AND doc_id IS ?",
                (kind, keyword_id, doc_id),
            )
        else:
            raise ValueError("get_note needs a note_id or a kind")
        return self._row(cur)

    def note_sources(self, note_id: int) -> list[str]:
        return [
            r["chunk_id"]
            for r in self._rows(
                self._db.execute(
                    "SELECT chunk_id FROM note_sources WHERE note_id = ? ORDER BY chunk_id",
                    (note_id,),
                )
            )
        ]

    def set_note_stale(self, note_id: int, stale: bool = True) -> None:
        with self.transaction() as db:
            db.execute(
                "UPDATE notes SET stale = ? WHERE id = ?", (1 if stale else 0, note_id)
            )

    def stale_notes(self) -> list[dict[str, Any]]:
        return self._rows(
            self._db.execute("SELECT * FROM notes WHERE stale = 1 ORDER BY id")
        )

    def add_note_links(
        self, from_note_id: int, links: Sequence[Mapping[str, Any]]
    ) -> None:
        """Typed, grounded wikilinks (D10). Targets must be existing keywords."""
        with self.transaction() as db:
            db.executemany(
                "INSERT INTO note_links (from_note_id, to_keyword_id, type, grounding_chunk_id)"
                " VALUES (?, ?, ?, ?)",
                [
                    (
                        from_note_id,
                        link["to_keyword_id"],
                        link["type"],
                        link.get("grounding_chunk_id"),
                    )
                    for link in links
                ],
            )

    def note_links_from(self, note_id: int) -> list[dict[str, Any]]:
        return self._rows(
            self._db.execute(
                "SELECT nl.*, k.name AS to_keyword FROM note_links nl"
                " JOIN keywords k ON k.id = nl.to_keyword_id"
                " WHERE nl.from_note_id = ? ORDER BY nl.id",
                (note_id,),
            )
        )

    # ---------------------------------------------------------------- embeddings

    def upsert_embedding(
        self,
        *,
        owner_kind: str,
        owner_id: str,
        model: str,
        dim: int,
        vector_ref: str,
        content_hash: str,
    ) -> None:
        with self.transaction() as db:
            db.execute(
                "INSERT INTO embeddings (owner_kind, owner_id, model, dim, vector_ref, content_hash)"
                " VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(owner_kind, owner_id, model) DO UPDATE SET"
                " dim = excluded.dim, vector_ref = excluded.vector_ref,"
                " content_hash = excluded.content_hash",
                (owner_kind, owner_id, model, dim, vector_ref, content_hash),
            )

    def get_embedding(
        self, owner_kind: str, owner_id: str, model: str
    ) -> dict[str, Any] | None:
        return self._row(
            self._db.execute(
                "SELECT * FROM embeddings WHERE owner_kind = ? AND owner_id = ? AND model = ?",
                (owner_kind, owner_id, model),
            )
        )

    def embedding_hashes(self, owner_kind: str, model: str) -> dict[str, str]:
        """owner_id -> content_hash, for incremental re-embedding."""
        return {
            r["owner_id"]: r["content_hash"]
            for r in self._rows(
                self._db.execute(
                    "SELECT owner_id, content_hash FROM embeddings"
                    " WHERE owner_kind = ? AND model = ?",
                    (owner_kind, model),
                )
            )
        }

    def delete_embedding(
        self, owner_kind: str, owner_id: str, model: str | None = None
    ) -> None:
        with self.transaction() as db:
            if model is None:
                db.execute(
                    "DELETE FROM embeddings WHERE owner_kind = ? AND owner_id = ?",
                    (owner_kind, owner_id),
                )
            else:
                db.execute(
                    "DELETE FROM embeddings WHERE owner_kind = ? AND owner_id = ? AND model = ?",
                    (owner_kind, owner_id, model),
                )

    # -------------------------------------------------------------------- search

    def fts_search(
        self,
        query: str,
        *,
        limit: int = 20,
        kinds: Sequence[str] = ("chunk", "note"),
        raw: bool = False,
    ) -> list[dict[str, Any]]:
        """BM25 search over chunks and notes.

        Returns rows of {owner_kind, owner_id, text, score, doc_id, section_id}
        with score = -bm25 (higher is better), best first. `raw=True` passes
        the query to FTS5 verbatim for callers that build operator syntax.
        """
        match = query if raw else fts_query(query)
        if not match:
            return []
        results: list[dict[str, Any]] = []
        if "chunk" in kinds:
            results += [
                {
                    "owner_kind": "chunk",
                    "owner_id": r["id"],
                    "text": r["text"],
                    "score": -r["bm25"],
                    "doc_id": r["doc_id"],
                    "section_id": r["section_id"],
                }
                for r in self._rows(
                    self._db.execute(
                        "SELECT c.id, c.text, c.doc_id, c.section_id, bm25(chunks_fts) AS bm25"
                        " FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid"
                        " WHERE chunks_fts MATCH ? ORDER BY bm25 LIMIT ?",
                        (match, limit),
                    )
                )
            ]
        if "note" in kinds:
            results += [
                {
                    "owner_kind": "note",
                    "owner_id": str(r["id"]),
                    "text": r["text"],
                    "score": -r["bm25"],
                    "doc_id": r["doc_id"],
                    "section_id": None,
                }
                for r in self._rows(
                    self._db.execute(
                        "SELECT n.id, n.text, n.doc_id, bm25(notes_fts) AS bm25"
                        " FROM notes_fts JOIN notes n ON n.id = notes_fts.rowid"
                        " WHERE notes_fts MATCH ? ORDER BY bm25 LIMIT ?",
                        (match, limit),
                    )
                )
            ]
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]

    # ---------------------------------------------------------------------- jobs

    def enqueue_jobs(self, doc_id: str, stages: Sequence[str]) -> None:
        """Register the stages a document still owes; existing jobs are kept."""
        with self.transaction() as db:
            db.executemany(
                "INSERT INTO jobs (doc_id, stage, status, updated_at)"
                " VALUES (?, ?, 'pending', ?) ON CONFLICT(doc_id, stage) DO NOTHING",
                [(doc_id, stage, _now()) for stage in stages],
            )

    def get_job(self, doc_id: str, stage: str) -> dict[str, Any] | None:
        return self._row(
            self._db.execute(
                "SELECT * FROM jobs WHERE doc_id = ? AND stage = ?", (doc_id, stage)
            )
        )

    def pending_jobs(
        self, doc_id: str | None = None, max_attempts: int = 3
    ) -> list[dict[str, Any]]:
        """Jobs still to run: never finished, and not out of retries."""
        sql = (
            "SELECT * FROM jobs WHERE status IN ('pending', 'failed', 'running')"
            " AND attempts < ?"
        )
        params: tuple[Any, ...] = (max_attempts,)
        if doc_id is not None:
            sql += " AND doc_id = ?"
            params += (doc_id,)
        return self._rows(self._db.execute(sql + " ORDER BY id", params))

    def start_job(self, job_id: int) -> None:
        with self.transaction() as db:
            db.execute(
                "UPDATE jobs SET status = 'running', attempts = attempts + 1,"
                " updated_at = ? WHERE id = ?",
                (_now(), job_id),
            )

    def finish_job(
        self, job_id: int, status: str = "done", error: str | None = None
    ) -> None:
        with self.transaction() as db:
            db.execute(
                "UPDATE jobs SET status = ?, last_error = ?, updated_at = ? WHERE id = ?",
                (status, error, _now(), job_id),
            )

    # -------------------------------------------------------------------- config

    def config_get(self, key: str, default: str | None = None) -> str | None:
        row = self._db.execute(
            "SELECT value FROM kb_config WHERE key = ?", (key,)
        ).fetchone()
        return default if row is None else str(row[0])

    def config_set(self, key: str, value: str) -> None:
        with self.transaction() as db:
            db.execute(
                "INSERT INTO kb_config (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )

    def config_all(self) -> dict[str, str]:
        return {
            r["key"]: r["value"]
            for r in self._rows(self._db.execute("SELECT key, value FROM kb_config"))
        }

    # ----------------------------------------------------------------- metering

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
        """Frozen contract with llm/meter.py; the store sets the timestamp."""
        with self.transaction() as db:
            db.execute(
                "INSERT INTO llm_calls (ts, stage, role, model, tokens_in, tokens_out, cost, wall_ms)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (_now(), stage, role, model, tokens_in, tokens_out, cost, wall_ms),
            )

    def get_llm_calls(self) -> list[dict[str, Any]]:
        return self._rows(self._db.execute("SELECT * FROM llm_calls ORDER BY id"))

    # -------------------------------------------------------------------- usage

    def log_query(
        self,
        *,
        query: str,
        depth: str,
        anchors: Sequence[str] = (),
        hit_keywords: Sequence[str] = (),
        no_answer: bool = False,
    ) -> int:
        """Record a query; usage drives lazy consolidation (D11)."""
        with self.transaction() as db:
            cur = db.execute(
                "INSERT INTO query_log (ts, query, depth, anchors_json,"
                " hit_keywords_json, no_answer) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    _now(),
                    query,
                    depth,
                    json.dumps(list(anchors)),
                    json.dumps(list(hit_keywords)),
                    int(no_answer),
                ),
            )
            return int(cur.lastrowid)

    def get_queries(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._rows(
            self._db.execute("SELECT * FROM query_log ORDER BY ts DESC, id DESC LIMIT ?", (limit,))
        )
        for row in rows:
            row["anchors"] = json.loads(row.pop("anchors_json"))
            row["hit_keywords"] = json.loads(row.pop("hit_keywords_json"))
            row["no_answer"] = bool(row["no_answer"])
        return rows

    def queue_consolidation(self, keyword_id: int, reason: str) -> None:
        with self.transaction() as db:
            db.execute(
                "INSERT INTO consolidation_queue (keyword_id, reason, queued_at)"
                " VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
                (keyword_id, reason, _now()),
            )

    def consolidation_demand(self, limit: int = 50) -> list[dict[str, Any]]:
        """Keywords awaiting synthesis, oldest request first."""
        return self._rows(
            self._db.execute(
                "SELECT q.keyword_id, q.reason, q.queued_at, k.name"
                " FROM consolidation_queue q JOIN keywords k ON k.id = q.keyword_id"
                " ORDER BY q.queued_at, q.keyword_id LIMIT ?",
                (limit,),
            )
        )

    def dequeue_consolidation(self, keyword_id: int) -> None:
        with self.transaction() as db:
            db.execute(
                "DELETE FROM consolidation_queue WHERE keyword_id = ?", (keyword_id,)
            )
