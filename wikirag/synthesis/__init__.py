"""Synthesis: the S slider — consolidation eagerness (docs/synthesis.md, D11).

Memory layers: L0 verbatim (chunks/xrefs, built by ingestion), L1 gist
(summaries), L2 concepts (keyword notes, lazy), L3 atlas (overview/hubs).
Above L1, synthesis follows use: lazy on first need + background
consolidation of hot/central concepts, driven by query_log.

Grounding rule: every note records the chunk IDs it was built from
(note_sources); a note is stale iff its source set changed. Links are
typed and grounded (note_links); orphans impossible by construction.

Modules:
    summaries      L1: per-document/section summary notes (at ingest, S1+)
    notes          L2: synthesize one keyword note from its tagged chunks
    consolidation  the policy: what gets synthesized, when (lazy/background)
    atlas          L3: overview map, hub pages, link curation (S3)
"""
