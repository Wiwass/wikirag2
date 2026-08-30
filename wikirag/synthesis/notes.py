"""S2: keyword notes — the wiki layer, rebuilt on chunk references.

Per keyword: gather its tagged chunks (chunk_keywords), synthesize a
wiki-style note, choose wikilinks from an embedding-shortlisted candidate
list of *existing* keywords. The model links only IDs it was given, so
orphan links cannot be created (v1's whole orphan-recovery subsystem
reduces to an invariant check).

Links are written to note_links as the note is stored; provenance to
note_sources. Notes are indexed like any content. Regeneration is
per-keyword, queued as jobs.

Deferred-synthesis variant (LazyGraphRAG pattern, open): build/refresh a
note the first time a query needs it, then cache. Same API, different
trigger.

Planned API
-----------
- generate_note(kb, keyword_id, client) -> note_id
- stale_notes(kb) -> [keyword_id]   (source set changed since last synth)
"""
