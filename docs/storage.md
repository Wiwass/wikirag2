# Storage

One SQLite database per knowledge base is the source of truth. The markdown vault is an export. This replaces v1's constellation of JSON files (`keywords_map`, `keyword_chunks`, `pending_keywords`, `raw_windows`, three `.npz` stores + meta) and all the `sync_*` reconciliation code they required.

## Layout on disk

```
knowledge_bases/<name>/
├── kb.sqlite          # everything: content, indexes, links, jobs, config
├── vectors/           # embedding matrices (npz or sqlite-vec — see open questions)
├── raw/               # inbox: PDFs waiting for ingest
├── archive/           # ingested PDFs
└── vault/             # exported markdown view (regenerable, safe to delete)
```

## Schema draft

```sql
-- Sources
documents(id, filename, title, ingested_at, status, meta_json)
sections(id, doc_id, parent_id, ordinal, title, level)        -- extracted structure

-- Canonical content
chunks(id, doc_id, section_id, ordinal, text, token_count)
-- id: stable, derived from (doc_id, section path, ordinal)

-- Concept layer (S2+)
keywords(id, name, canonical_name)                            -- canonical_name for case/plural dedup
chunk_keywords(chunk_id, keyword_id)                          -- LLM tags chunks; no text copied
notes(id, keyword_id NULL, kind, text, updated_at)            -- kind: keyword|summary|overview
note_links(from_note_id, to_keyword_id)                       -- the wikilink graph, first-class

-- Retrieval indexes
chunks_fts(text)                                              -- FTS5 external-content table → BM25
embeddings(owner_kind, owner_id, model, dim, vector_ref, content_hash)
-- owner_kind: chunk|note  → notes are retrievable like any content

-- Operations
jobs(id, doc_id, stage, status, attempts, last_error, updated_at)
kb_config(key, value)                                         -- S level, model names, etc.
```

## Design points

- **Transactionality**: a stage's output (e.g. all chunks of a document + FTS rows) commits atomically. A crash never leaves a half-written index; v1's atomic-rename dance disappears.
- **Incrementality**: appending a document touches only its rows. v1 rewrote multi-MB JSON files on every block.
- **Embeddings track content by hash** (as in v1, the one mechanism worth keeping): re-embedding happens only when text or model changes. Orphan cleanup becomes a foreign-key delete, not a sync pass.
- **Notes are content.** A keyword note is indexed (FTS + embedding) exactly like chunks; retrieval decides by score, not by hardcoded tier priority.
- **The wikilink graph gets a real table.** v1 stored links only inside markdown text and re-parsed with regex at query time; `note_links` makes graph expansion a join, and orphan-link detection a query.

## Vault export

`wikirag export-vault <kb>` renders `vault/` from the store: one file per keyword note, per document summary, plus `overview.md`. Wikilinks render as `[[name]]` for Obsidian. Deleting the vault loses nothing. An optional watch mode can re-export after ingest so the vault stays fresh.

Open question (deferred): whether vault edits made by the user in Obsidian should flow *back* into the store (two-way sync). Default answer for now: no — the vault is read-only output; user knowledge enters through ingestion.

## Open questions

- **Vector index**: plain npz matrices loaded in RAM were fine at v1 scale (~tens of thousands of vectors) and keep dependencies at zero; `sqlite-vec` would keep everything in one file and scale further. Decide at skeleton time with a micro-benchmark.
- **Embedding model**: modern, long-context (≥8k), ideally multilingual (Italian queries over English corpora). Candidates: bge-m3, gte-modernbert-base, EmbeddingGemma. Decide with the retrieval floor benchmark.
