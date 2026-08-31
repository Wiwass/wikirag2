"""SQLite store: schema, migrations, and transactional access.

Responsibilities
----------------
- Own the schema in docs/storage.md: documents, sections, chunks, xrefs,
  keywords, chunk_keywords, notes (with staleness), note_sources,
  note_links (typed, grounded), chunks_fts (FTS5, BM25), embeddings meta,
  jobs, kb_config, llm_calls, query_log, consolidation_queue.
- Stable chunk IDs derived from (doc_id, structural path, ordinal) —
  deterministic across re-runs on the same source file.
- Transactions per stage output: a document's chunks + FTS rows commit
  together or not at all. WAL mode; single writer, concurrent readers.
- Cascading deletes make `forget <doc>` mostly schema (docs/synthesis.md).
- Job table drives ingestion resumability (replaces v1's pending-JSON files).

Planned API
-----------
- class Store: open/close, migrate()
- write ops grouped by stage: add_document, add_chunks, add_xrefs,
  tag_chunks, upsert_note, record_job, record_llm_call, log_query,
  queue_consolidation, forget_document, ...
- read ops for retrieval: fts_search, get_chunks, section_window (the
  read-big expansion), get_note, neighbors, xrefs_from, consolidation
  demand, config get/set. Read APIs double as the agent's tool backends.
"""
