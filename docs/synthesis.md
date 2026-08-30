# Synthesis — the S slider

Everything above a bare index: the layer that makes a KB *readable* and supports global/sensemaking questions. Per-KB setting, stored in `kb_config`, changeable later (raising S on an existing KB runs only the missing stages).

## Levels

**S0 — Index only.** Chunks, FTS, embeddings. The KB answers questions but has no human-facing artifacts. This is also the *control* every higher level must justify itself against in the benchmark.

**S1 — Summaries.** One summary note per document (or per major section for large books). Summaries are indexed content: they serve mid-level "what does book X say about..." queries and become the raw material for cheap overview building (v1's Phase F insight, kept).

**S2 — Wiki.** Keyword notes with wikilinks:

- Keywords come from the `enrich` stage as **tags on chunks** (`chunk_keywords`), not as owners of copied text.
- A keyword note is synthesized *from its referenced chunks*, with links to other keywords chosen from an embedding-shortlisted candidate list (fixes v1's orphan-link generation at the source: the model links only IDs it was given).
- Notes store their links in `note_links` as they are written; no regex re-parsing later.
- Regeneration is per-keyword and queued in `jobs` (v1's pending-queue resumability, kept but generalized).

**S3 — Atlas.** The curation pass: overview map note (map-reduce over S1 summaries, as v1 Phase F), hub pages for high-centrality keywords, a link-quality pass (merge near-duplicate keywords, prune weak links). This is where "vault readability" gets its own budget knob: S3 work is purely for the human product and global queries, and the benchmark is expected to show little R0–R1 gain; that is fine and stated up front.

## Grounding rule

Synthesis prose must cite its sources internally: each note records the chunk IDs it was built from (`note_sources(note_id, chunk_id)` — add to schema at skeleton time). This gives:

- honest regeneration (a note is stale iff its source set changed),
- provenance in chat ("according to [doc], ..."),
- and the option to render footnote-style references in the vault.

## Maintenance operations

- `forget <doc>`: delete document rows; cascading FKs remove chunks, tags, embeddings; notes whose source set shrank are queued for regeneration, notes with empty sources are deleted. What took v1 ~140 lines of careful bookkeeping becomes mostly schema.
- `regenerate-notes`, `rebuild-overview`: re-run synth stages from stored chunks, no API re-ingest.
- Orphan links cannot be *created* under the S2 design; a checker query remains as an invariant test, not a recovery subsystem.
