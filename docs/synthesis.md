# Synthesis — the S slider

**S is consolidation eagerness, not an amount of upfront work.** The system consolidates what gets used (like memory does), instead of paying to synthesize everything in advance (v1's cost sink, unproven benefit — see D11). The memory model has four layers:

| Layer | Artifact | Built |
|---|---|---|
| L0 verbatim | chunks + section tree + native xrefs | at ingest, deterministic, no LLM |
| L1 gist | per-document/section summaries | at ingest, one cheap pass per doc |
| L2 concepts | keyword notes, typed grounded links | **lazily**, on demand + background consolidation |
| L3 atlas | overview, hub pages | periodically, from L1 + usage stats |

## Levels

**S0 — Index only.** L0 + embeddings + FTS. The control every higher level must beat in the benchmark.

**S1 — Gist.** S0 + L1 summaries (indexed content; entry points for orientation and mid-level queries; raw material for the atlas).

**S2 — Wiki (lazy).** Enables the L2 concept layer:

- The enrich stage tags chunks with keywords (`chunk_keywords`) — cheap, by reference.
- **Notes are not generated at ingest.** A note is synthesized the first time a query needs it (R2 consolidation trigger, or an explicit vault request), then cached; it is stale iff its source set changed.
- **Background consolidation**: a job periodically synthesizes notes for *hot* concepts (frequently hit in `query_log`) and *central* ones (incoming links, tag counts). The vault grows around actual use.
- Wikilinks: chosen from an embedding-shortlisted candidate list of existing keywords (orphans impossible by construction), stored in `note_links` with a type and, when possible, a grounding chunk; provenance in `note_sources`.

**S3 — Atlas.** S2 + proactive consolidation of everything central, overview map (map-reduce over L1, cached as jobs), hub pages, link curation (merge near-duplicate keywords, prune weak links). Purely for the human product and global queries; little R0–R1 benchmark gain expected, stated up front.

## Associations (what replaces the v1 graph)

1. **Embedding space** (primary, free at encoding): the analogue of associative memory; the neighborhood *is* the association.
2. **Native structural graph** (exact, free): section adjacency, document hierarchy, and **explicit cross-references in the text** ("see Chapter 5", "cfr. ArM5 p.83") extracted by pattern matching at ingest into `xrefs`. The only graph materialized eagerly.
3. **Semantic wikilinks** (cautious): only from synthesis, only typed and grounded. They serve the vault and agent navigation; as a retrieval channel they are experimental (D10).

## Grounding rule

Every note records the chunk IDs it was built from (`note_sources`). This gives honest staleness detection, provenance in chat, and optional footnote rendering in the vault.

## Latency tradeoff (accepted)

Lazy consolidation means the first R2 query touching an unconsolidated concept may pay synthesis latency inline or defer it to background (config flag, default: defer — the current query proceeds with chunks, the note is ready next time). Decision recorded in D11.

## Maintenance

- `forget <doc>`: cascading FK deletes; notes with shrunken source sets become stale (regenerate on next touch), empty ones are deleted.
- `regenerate-notes` / `rebuild-overview`: re-run synthesis from stored chunks, no API re-ingest.
- Orphan links cannot be created; a checker query remains as an invariant test.
