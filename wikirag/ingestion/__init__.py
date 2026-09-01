"""Ingestion: idempotent per-document stages, tracked as jobs in the store.

Stage sequence (contracts fixed in docs/ingestion.md, enrich methods open):

    extract -> chunk -> xrefs -> index [S0] -> summarize [S1+] -> enrich [S2+]

Note synthesis is NOT an ingest stage: L2 notes are consolidated lazily
(wikirag.synthesis.consolidation, D11). Ingest pays only for the layers
above plus keyword tagging.

Non-negotiable rules:
1. No verbatim echo: LLM stages receive numbered chunks, return IDs + new
   prose, never copies of source text.
2. Chunking is deterministic and local; same file => same chunks and IDs.
3. Chunk size fits the embedding model (its max_seq_tokens is an input).
4. Every stage meters its cost.

Modules:
    runner    job scheduling, resume, retry policy
    chunking  deterministic structure-aware chunker
    xrefs     native cross-reference extraction (pattern-based, no LLM)
    enrich    OPEN research area: tagging + contextualization methods

This package also holds the default stage implementations (extract_stage,
chunk_stage, xrefs_stage, index_stage), `index_notes` (the whole-KB pass
that makes synthesis output retrievable on the dense channel too) and
`register_default_stages()`,
which wires them into the runner's registry. Wiring is explicit — the
runner's registry stays empty at import time so it remains testable with
fakes; the CLI (and any integration entry point) calls
`register_default_stages()` once at startup.

Stage idempotency: extract and chunk treat existing rows for the document
as proof of completion (both are deterministic, so re-running would write
identical rows); xrefs replaces its rows transactionally; index skips
chunks whose content hash already matches. A job interrupted between the
store commit and the jobs-table update is therefore safe to re-run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from wikirag.config import KnowledgeBase

# kb_config keys the stages honor (with their defaults).
EMBEDDING_MODEL_KEY = "embedding_model"
CHUNK_MAX_TOKENS_KEY = "chunk_max_tokens"

_EMBED_BATCH = 64


def _source_pdf(kb: "KnowledgeBase", doc_id: str) -> Path:
    document = kb.store.get_document(doc_id)
    if document is None:
        raise RuntimeError(f"unknown document {doc_id!r}")
    filename = str(document["filename"])
    for directory in (kb.raw_dir, kb.archive_dir):
        path = directory / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"{filename} not found in raw/ or archive/")


# One-slot cache so the extract and chunk stages of the same document parse
# the PDF once per run instead of twice; keyed on (path, mtime) so a
# replaced file is never served stale.
_extract_cache: dict[tuple[str, float], Any] = {}


def _extracted(kb: "KnowledgeBase", doc_id: str) -> Any:
    from wikirag.extraction.pdf import extract

    path = _source_pdf(kb, doc_id)
    key = (str(path), path.stat().st_mtime)
    if key not in _extract_cache:
        _extract_cache.clear()
        _extract_cache[key] = extract(path)
    return _extract_cache[key]


def extract_stage(kb: "KnowledgeBase", doc_id: str) -> None:
    """PDF -> section tree in the store. Zero sections = extraction failure."""
    doc = _extracted(kb, doc_id)
    if not doc.sections:
        raise RuntimeError("extraction found no text (scan without a text layer?)")
    if kb.store.get_sections(doc_id):
        return
    from wikirag.ingestion.chunking import section_rows

    kb.store.add_sections(doc_id, section_rows(doc_id, doc))


def chunk_stage(kb: "KnowledgeBase", doc_id: str) -> None:
    """Sections -> canonical chunks with stable IDs."""
    if kb.store.get_document_chunks(doc_id):
        return
    from wikirag.ingestion.chunking import DEFAULT_MAX_TOKENS, chunk_document

    max_tokens = int(kb.store.config_get(CHUNK_MAX_TOKENS_KEY, str(DEFAULT_MAX_TOKENS)))
    kb.store.add_chunks(
        chunk_document(doc_id, _extracted(kb, doc_id), max_tokens=max_tokens)
    )


def xrefs_stage(kb: "KnowledgeBase", doc_id: str) -> None:
    """Chunks -> native cross-references (already idempotent)."""
    from wikirag.ingestion.xrefs import apply_xrefs

    apply_xrefs(kb, doc_id)


def _default_embedder(model_name: str) -> Any:
    from wikirag.llm.embeddings import get_embedder

    return get_embedder(model_name)


# Seam for tests and offline smoke runs: swap in a FakeEmbedder factory.
_embedder_factory: Callable[[str], Any] = _default_embedder


def content_hash(text: str) -> str:
    """The content hash embeddings are tracked by (store + vector index)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def index_stage(kb: "KnowledgeBase", doc_id: str) -> None:
    """Chunks -> vectors in the KB index + embeddings meta rows.

    Incremental by content hash: only chunks whose text or model changed are
    re-embedded (docs/storage.md).
    """
    from wikirag.llm.embeddings import DEFAULT_EMBEDDING_MODEL
    from wikirag.storage.vectors import INDEX_FILENAME, VectorIndex

    model = kb.store.config_get(EMBEDDING_MODEL_KEY, DEFAULT_EMBEDDING_MODEL)
    assert model is not None
    chunks = kb.store.get_document_chunks(doc_id)
    known = kb.store.embedding_hashes("chunk", model)
    todo = [
        (str(c["id"]), str(c["text"]), h)
        for c in chunks
        if known.get(str(c["id"])) != (h := content_hash(str(c["text"])))
    ]
    if not todo:
        return
    embedder = _embedder_factory(model)
    index = VectorIndex.open(kb.vectors_dir, model, embedder.dim)
    for start in range(0, len(todo), _EMBED_BATCH):
        batch = todo[start : start + _EMBED_BATCH]
        vectors = embedder.encode_passages([text for _, text, _ in batch])
        for (chunk_id, _, text_hash), vector in zip(batch, vectors):
            index.upsert(("chunk", chunk_id), text_hash, vector)
            kb.store.upsert_embedding(
                owner_kind="chunk",
                owner_id=chunk_id,
                model=model,
                dim=embedder.dim,
                vector_ref=INDEX_FILENAME,
                content_hash=text_hash,
            )
    index.save()


def index_notes(kb: "KnowledgeBase", *, embedder: Any | None = None) -> int:
    """Embed notes that have no current vector, and return how many.

    Notes are content (docs/storage.md): a summary or a keyword note is
    retrievable exactly like a chunk, so it needs both channels. FTS gets
    them from the store's triggers; the dense channel needs this pass.
    Notes are written by synthesis rather than by a per-document stage, so
    this runs over the whole KB instead of hanging off `index_stage`.
    """
    from wikirag.llm.embeddings import DEFAULT_EMBEDDING_MODEL
    from wikirag.storage.vectors import INDEX_FILENAME, VectorIndex

    model = kb.store.config_get(EMBEDDING_MODEL_KEY, DEFAULT_EMBEDDING_MODEL)
    assert model is not None
    known = kb.store.embedding_hashes("note", model)
    todo: list[tuple[str, str, str]] = []
    for note in _all_notes(kb):
        note_id = str(note["id"])
        text = str(note["text"])
        text_hash = content_hash(text)
        if known.get(note_id) != text_hash:
            todo.append((note_id, text, text_hash))
    if not todo:
        return 0

    embedder = embedder or _embedder_factory(model)
    index = VectorIndex.open(kb.vectors_dir, model, embedder.dim)
    for start in range(0, len(todo), _EMBED_BATCH):
        batch = todo[start : start + _EMBED_BATCH]
        vectors = embedder.encode_passages([text for _, text, _ in batch])
        for (note_id, _, text_hash), vector in zip(batch, vectors):
            index.upsert(("note", note_id), text_hash, vector)
            kb.store.upsert_embedding(
                owner_kind="note",
                owner_id=note_id,
                model=model,
                dim=embedder.dim,
                vector_ref=INDEX_FILENAME,
                content_hash=text_hash,
            )
    index.save()
    return len(todo)


def _all_notes(kb: "KnowledgeBase") -> list[dict[str, Any]]:
    """Every note in the store. The store has no list-all API by design;
    notes are reached by subject, so this walks the subjects."""
    seen: dict[int, dict[str, Any]] = {}
    overview = kb.store.get_note(kind="overview")
    if overview is not None:
        seen[int(overview["id"])] = overview
    for document in kb.store.list_documents():
        note = kb.store.get_note(kind="summary", doc_id=str(document["id"]))
        if note is not None:
            seen[int(note["id"])] = note
    for keyword in kb.store.list_keywords():
        note = kb.store.get_note(kind="keyword", keyword_id=int(keyword["id"]))
        if note is not None:
            seen[int(note["id"])] = note
    return list(seen.values())


def register_default_stages() -> None:
    """Wire the real stage implementations into the runner's registry."""
    from wikirag.ingestion import runner

    runner.register_stage("extract", extract_stage)
    runner.register_stage("chunk", chunk_stage)
    runner.register_stage("xrefs", xrefs_stage)
    runner.register_stage("index", index_stage)
    # S1+ / S2+ stages. Imported here rather than at module import so an S0
    # ingest never pulls in the synthesis layer.
    from wikirag.ingestion.enrich import enrich_stage
    from wikirag.synthesis.summaries import summarize_stage

    runner.register_stage("summarize", summarize_stage)
    runner.register_stage("enrich", enrich_stage)
