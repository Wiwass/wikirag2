"""Synthesis: the S slider above S0 (docs/synthesis.md).

Everything human-facing and everything that supports global questions.
Grounding rule: every note records the chunk IDs it was built from
(note_sources); a note is stale iff its source set changed.

Modules:
    summaries  S1: per-document/section summary notes
    notes      S2: keyword notes with wikilinks (on chunk references)
    atlas      S3: overview map, hub pages, link curation
"""
